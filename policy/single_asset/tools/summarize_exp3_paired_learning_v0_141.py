"""Independent fixed-denominator paired evaluation plus raw-camera galleries."""
import argparse,copy,json,math,sys,time,subprocess,textwrap
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from run_exp3_remote_pilot_jobs_v0_125 import emit,sha,ROBOT
from verify_exp3_paired_robot_episode_v0_134 import inspect

def wilson(success,n):
    if not n:return None
    z=1.959963984540054;p=success/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0.,c-h),min(1.,c+h)]

def counts(rows):
    n=len(rows);k=sum(x.get('sim_task_success') is True for x in rows)
    return dict(denominator=n,successes=k,failures=sum(x.get('sim_task_success') is False for x in rows),blocked_or_not_evaluated=sum(x.get('sim_task_success') is None for x in rows),raw_success_rate=k/n if n else None,wilson_95_ci=wilson(k,n),failure_reasons=dict(Counter(reason for row in rows for reason in row['failure_reasons'])))

def difference(before,after):
    a,b=before['raw_success_rate'],after['raw_success_rate']
    return dict(absolute_percentage_points=100*(b-a),relative_improvement=(b-a)/a if a else None,relative_improvement_reason=None if a else 'undefined_zero_untrained_success_rate')

def annotate(raw,destination,title,receipt):
    import imageio_ffmpeg
    from PIL import Image,ImageDraw,ImageFont
    if not raw.exists():return None
    # Header is outside the original camera image. The raw original is never
    # re-encoded, overwritten, cropped or replaced as verification evidence.
    lines=[line for part in title for line in textwrap.wrap(part,width=64)];height=max(64,10*len(lines)+12);height+=height%2
    label=destination.with_suffix('.label.txt')
    with label.open('x') as f:f.write('\n'.join(lines)+'\n')
    header=Image.new('RGB',(320,height),'black');draw=ImageDraw.Draw(header);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',8)
    draw.multiline_text((3,4),'\n'.join(lines),font=font,fill='white',spacing=2);header_path=destination.with_suffix('.header.png');header.save(header_path)
    # The bundled FFmpeg has no drawtext filter. A static text-only PNG is
    # overlaid OUTSIDE the original raw image; object motion is never drawn.
    filt=f'[0:v]pad=iw:ih+{height}:0:{height}:color=black[canvas];[canvas][1:v]overlay=0:0:eof_action=repeat:repeatlast=1[out]'
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-v','error','-threads','1','-i',str(raw),'-i',str(header_path),'-filter_complex_threads','1','-filter_complex',filt,'-map','[out]','-c:v','libx264','-threads','1','-crf','24','-preset','veryfast','-pix_fmt','yuv420p','-an',str(destination)]
    started=time.monotonic();r=subprocess.run(command,text=True,capture_output=True)
    result=dict(command=command,returncode=r.returncode,stderr=r.stderr,runtime_seconds=time.monotonic()-started,source_raw_video=str(raw),source_sha256=sha(raw),output=str(destination) if destination.exists() else None,output_sha256=sha(destination) if destination.exists() else None,label=str(label),label_sha256=sha(label),derivative_display_only=True,raw_camera_original_unchanged=True)
    emit(receipt,result);return result

def one_episode(args):
    asset,held,variant,run,out=args;aid=asset['asset_id'];index=held['index'];job=run/'evaluation'/variant/f"gpu_{asset['gpu_index']}"/'assets'/aid/'worker/episodes'/aid/f'episode_{index:03d}';destination=out/'episodes'/variant/aid/f'episode_{index:03d}';destination.mkdir(parents=True,exist_ok=False)
    row=dict(asset_id=aid,candidate_id=asset['candidate_id'],asset_role=asset['asset_role'],task_id=asset['task_id'],model_variant=variant,seed=held['seed'],heldout_index=index,initial_state_registration=held,job=str(job),model_task_success_not_scripted=True,denominator_member=True,sim_task_success=None,evidence_valid=False,failure_reasons=[],artifact_paths={},**ROBOT)
    if not (job/'execution_receipt.json').exists():
        row.update(status='blocked',failure_reasons=['episode_not_completed_or_infrastructure_blocked'])
    else:
        try:
            proof=inspect(job);row['verification']=proof;row['evidence_valid']=proof['evidence_valid'];row['sim_task_success']=proof['sim_task_success'] if proof['evidence_valid'] else None
            row['failure_reasons']=sorted(set(proof['failure_reasons']+proof['task_failure_reasons']));row['status']='success' if row['sim_task_success'] is True else 'failed' if row['sim_task_success'] is False else 'blocked'
            registered=json.loads((job/'initial_state_registration.json').read_text());spec=json.loads((job/'task_spec.json').read_text());expected=copy.deepcopy(asset['task_spec']);expected['camera_reset_delta']=held['camera_reset_delta']
            if 'initial_joint_position_si' in held:expected['workcell']['articulated_initial_joint_rad']=held['initial_joint_position_si']
            if any(registered.get(k)!=v for k,v in held.items()) or spec!=expected:raise ValueError('paired_initial_condition_or_task_definition_changed')
            if spec['asset_sha256']!=asset['asset_sha256'] or str(spec['candidate_id'])!=str(asset['candidate_id']):raise ValueError('paired_candidate_changed')
            physics=json.loads((job/'episode/case_result.json').read_text());row['sim_ready_status']=physics['sim_ready_status'];row['model_checkpoint_sha256']=physics['policy_identity']['adapter_sha256'];row['runtime_seconds']=json.loads((job/'call_receipt.json').read_text())['runtime_seconds']
            row['failure_stages']=proof['full_task_evaluation'].get('stages',{});row['camera_capture_type']='raw_isaac_rgb';row['fallback_used']=False;row['teleport_used']=False;row['post_play_transform_writeback']=False
            row['artifact_paths']=dict(raw_policy_video=str(job/'episode/rollout_raw.mp4'),raw_evidence_video=str(job/'episode/evidence_raw.mp4'),raw_state_trace=str(job/'episode/complete_task_raw_trace.json'),actions=str(job/'episode/state_trace.json'),setup=str(job/'episode/setup_frame.png'),final=str(job/'episode/final_frame.png'),model_responses=str(job/'episode/policy_responses.json'),verifier=str(destination/'independent_verification.json'))
        except Exception as exc:row.update(status='blocked',sim_task_success=None,evidence_valid=False,failure_reasons=sorted(set(row['failure_reasons']+[type(exc).__name__+':'+str(exc)])))
    emit(destination/'independent_verification.json',row)
    title=[aid,asset['task_id'],'seed='+str(held['seed']),variant,'success='+str(row['sim_task_success'])]
    for name in ['rollout_raw','evidence_raw']:
        path=job/'episode'/(name+'.mp4')
        if path.exists():
            result=annotate(path,destination/(name+'_annotated.mp4'),title,destination/(name+'_annotation_receipt.json'))
            if result and result['returncode']==0:row['artifact_paths'][name+'_annotated']=result['output']
            elif result:row.setdefault('display_artifact_failures',[]).append(result['stderr'])
    row['verification_sha256']=sha(destination/'independent_verification.json');emit(destination/'per_episode.json',row)
    return row

def gallery(rows,assets,out):
    from PIL import Image,ImageDraw,ImageFont
    fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';font=ImageFont.truetype(fontpath,12)
    canvas=Image.new('RGB',(1280,10*296),(238,242,246));draw=ImageDraw.Draw(canvas);text=['# Full robot task paired evaluation: complete gallery','', 'All frames and videos below come from actual Isaac cameras; missing evidence is not replaced by animations.','']
    for i,asset in enumerate(assets):
        aid=asset['asset_id'];draw.text((5,i*296+2),aid+'  (fixed first held-out episode, never best-of)',font=font,fill='black')
        for j,variant in enumerate(['untrained_vla','trained_vla']):
            row=next(x for x in rows if x['asset_id']==aid and x['model_variant']==variant and x['heldout_index']==0)
            for k,key in enumerate(['setup','final']):
                x=(2*j+k)*320;y=i*296+40;path=Path(row['artifact_paths'].get(key,''))
                if path.is_file():
                    with Image.open(path) as image:canvas.paste(image.convert('RGB').resize((320,240)),(x,y))
                else:draw.text((x+8,y+20),'No valid capture: '+row['status'],font=font,fill='red')
                draw.text((x+3,i*296+22),variant+' '+key+'; success='+str(row['sim_task_success']),font=font,fill='black')
    canvas.save(out/'paired_first_episode_contact_sheet.png')
    for row in rows:
        text.extend(['## '+row['asset_id']+' / '+row['model_variant']+' / '+str(row['heldout_index']), '', 'seed='+str(row['seed'])+'; status='+row['status']+'; sim_task_success='+str(row['sim_task_success']), ''])
        for key in ['setup','final']:
            path=row['artifact_paths'].get(key)
            if path:text.extend(['!['+key+']('+path+')',''])
        for key in ['raw_policy_video','raw_evidence_video','rollout_raw_annotated','evidence_raw_annotated']:
            path=row['artifact_paths'].get(key)
            if path:text.extend(['['+key+']('+path+')',''])
        text.extend(['Failure reasons: '+json.dumps(row['failure_reasons']), ''])
    with (out/'all_episode_gallery.md').open('x') as f:f.write('\n'.join(text)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--configuration',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();cfg=json.loads(a.configuration.read_text());tasks=[]
    for variant in cfg['variants']:
        for asset in cfg['assets']:
            for held in asset['heldout_episodes']:tasks.append((asset,held,variant,a.run,a.output))
    if len(tasks)!=400:raise ValueError('paired_denominator_changed')
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(one_episode,tasks))
    for variant in cfg['variants']:
        with (a.output/(variant+'_per_episode.jsonl')).open('x') as f:
            for row in rows:
                if row['model_variant']==variant:f.write(json.dumps(row,sort_keys=True)+'\n')
    by_variant={v:counts([x for x in rows if x['model_variant']==v]) for v in cfg['variants']}
    by_asset={asset['asset_id']:{v:counts([x for x in rows if x['asset_id']==asset['asset_id'] and x['model_variant']==v]) for v in cfg['variants']} for asset in cfg['assets']}
    by_role={role:{v:counts([x for x in rows if x['asset_role']==role and x['model_variant']==v]) for v in cfg['variants']} for role in ['rigid','articulated']}
    by_task={task:{v:counts([x for x in rows if x['task_id']==task and x['model_variant']==v]) for v in cfg['variants']} for task in sorted({x['task_id'] for x in rows})}
    comparisons={aid:difference(v['untrained_vla'],v['trained_vla']) for aid,v in by_asset.items()}
    summary=dict(total_assets=10,rigid_assets=5,articulated_assets=5,denominator_per_variant=200,by_variant=by_variant,by_asset=by_asset,by_role=by_role,by_task=by_task,per_asset_paired_change=comparisons,paired_change=difference(by_variant['untrained_vla'],by_variant['trained_vla']),oracle_gap='not_measured',teacher_upper_bound='formal training-source collection separately; not a matched heldout VLA condition',only_limited_simulation_scope=True,exp2_relation_evaluation='not_in_scope',**ROBOT)
    emit(a.output/'summary.json',summary);gallery(rows,cfg['assets'],a.output)
    report=['# Exp3 complete robot tasks: paired VLA evaluation','', 'Isaac/PhysX simulation only, no real-robot experiment; Exp2 object-level asset construction does not imply a successful scene-relation reconstruction.','', 'The untrained baseline is the same public base checkpoint with the same untrained 9-D robot adapter, not a forced reinterpretation of native LIBERO 7-D actions. The trained model uses the fixed final-step checkpoint; no checkpoint is selected from held-out results.','', '| Asset | Untrained successes / 20 | Trained successes / 20 | Not evaluated (untrained/trained) |','|---|---:|---:|---:|']
    for aid,value in by_asset.items():
        u,t=value['untrained_vla'],value['trained_vla'];report.append(f"| {aid} | {u['successes']} | {t['successes']} | {u['blocked_or_not_evaluated']}/{t['blocked_or_not_evaluated']} |")
    report+=['','```json',json.dumps(summary,ensure_ascii=False,indent=2),'```','','[Raw videos, display videos and frames of every episode](all_episode_gallery.md)','','![Fixed first held-out episode, paired comparison](paired_first_episode_contact_sheet.png)','','Success is recomputed independently from raw state, contact and physics gates; videos are evidence only. Scripted-teacher success, sim-ready status, training loss and produced actions never count as VLA task success.']
    with (a.output/'final_report.md').open('x') as f:f.write('\n'.join(report)+'\n')
    duplicate=len({(x['asset_id'],x['model_variant'],x['seed']) for x in rows})!=400;safe=all(x['robot_task_success'] is None for x in rows)
    verification=dict(report_valid=not duplicate and safe,all_400_conditions_recorded=len(rows)==400,unique_terminal_rows=not duplicate,all_evaluation_evidence_valid=all(x['evidence_valid'] for x in rows),physics_or_task_failure_not_hidden=True,robot_task_success_always_null=safe,frozen_evaluation_configuration_sha256=sha(a.configuration),task_success_count=sum(x['sim_task_success'] is True for x in rows),**ROBOT)
    emit(a.output/'independent_verification.json',verification)
    hashes={str(path.relative_to(a.output)):sha(path) for path in a.output.rglob('*') if path.is_file()};emit(a.output/'freeze_receipt.json',dict(output_files_sha256=hashes,summary_sha256=sha(a.output/'summary.json')))
    emit(a.output/'execution_receipt.json',dict(command=[sys.executable,*sys.argv],runtime_seconds=time.monotonic()-start,code_sha256=sha(__file__),storage_bytes=sum(path.stat().st_size for path in a.run.rglob('*') if path.is_file()),**ROBOT));print(json.dumps(by_variant),flush=True)
    return 0 if verification['report_valid'] else 1

if __name__=='__main__':raise SystemExit(main())
