"""Independent clock/raw-video audit; never repair an old success label."""
import argparse, json, math, sys
from pathlib import Path
from freeze_exp3_franka_training_dataset_v0_2 import emit, read, sha


def check_clock(trace, receipt, fps=10, physics_hz=60):
    errors=[];states=trace.get('states',[]);frames=receipt.get('frames',[])
    if len(states)<2:errors.append('insufficient_state_samples')
    if any('simulation_time_s' not in s for s in states):
        errors.append('actual_simulator_state_timestamps_missing')
    elif states:
        origin=states[0]['simulation_time_s']
        for i,s in enumerate(states):
            if not all(isinstance(s.get(k),(float,int)) and math.isfinite(s[k]) for k in ('timestamp_s','simulation_time_s')):
                errors.append('nonfinite_state_clock');break
            if abs(s['timestamp_s']-(s['simulation_time_s']-origin))>1e-6:
                errors.append('nominal_clock_differs_from_simulator');break
            if i and abs(s['simulation_time_s']-states[i-1]['simulation_time_s']-1/physics_hz)>1e-6:
                errors.append('physics_sample_gap');break
    if len(frames)<2:errors.append('insufficient_video_frames')
    else:
        times=[f['simulation_time_s'] for f in frames]
        if any(b<=a or b-a>1/fps+1e-5 for a,b in zip(times,times[1:])):errors.append('video_capture_gap_or_duplicate')
        if abs((len(frames)-1)/fps-(times[-1]-times[0]))>1/fps+1e-5:errors.append('video_playback_not_simulation_duration')
        if states and 'simulation_time_s' in states[0]:
            if times[0]>states[0]['simulation_time_s']+1e-6 or times[-1]<states[-1]['simulation_time_s']-1e-6:errors.append('video_does_not_cover_episode')
    if receipt.get('camera_capture_type')!='raw_isaac_rgb' or receipt.get('motion_synthesized') is not False:errors.append('raw_isaac_provenance_missing')
    return {'passed':not errors,'errors':errors,'trace_samples':len(states),'frame_count':len(frames)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import imageio.v2 as imageio
    result=check_clock(read(a.episode/'complete_task_raw_trace.json'),read(a.episode/'synchronized_video_receipt.json'))
    videos=[]
    for name in ('rollout_raw.mp4','evidence_raw.mp4'):
        path=a.episode/name;row={'path':str(path),'sha256':sha(path) if path.exists() else None}
        try:
            reader=imageio.get_reader(path);nonblack=0;count=0
            for frame in reader:
                count+=1;nonblack+=int(frame.mean()>1 and frame.max()>8 and frame.std()>1)
            reader.close();row.update(frame_count=count,nonblack_count=nonblack,decoded=True)
            if count!=result['frame_count'] or nonblack!=count or count<2:result['errors'].append(name+':video_content_or_sync_invalid')
        except Exception as exc:
            row.update(decoded=False,error=f'{type(exc).__name__}:{exc}');result['errors'].append(name+':decode_failed')
        videos.append(row)
    result.update(passed=not result['errors'],videos=videos,scope='clock, raw RGB provenance and decode; NOT task or framing admission',historical_outputs_unchanged=True)
    emit(a.output/'independent_clock_video_verification.json',result)
    emit(a.output/'execution_receipt.json',{'command':[sys.executable,*sys.argv],'code_sha256':sha(__file__),'input_hashes':{n:sha(a.episode/n) for n in ('complete_task_raw_trace.json','synchronized_video_receipt.json')},'output_sha256':sha(a.output/'independent_clock_video_verification.json')})
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
