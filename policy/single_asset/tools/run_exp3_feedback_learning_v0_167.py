"""Detached registered feedback training -> verifier -> same paired Isaac."""
import argparse,json,sys,time,traceback
from pathlib import Path
import run_exp3_precision_learning_v0_156 as lifecycle
from run_exp3_remote_pilot_jobs_v0_125 import emit,sha,ROBOT

_popen=lifecycle.subprocess.Popen
def Popen(command,*args,**kwargs):
    command=[str(Path(x).with_name('serve_exp3_feedback_vla_v0_167.py')) if str(x).endswith('serve_exp3_precision_vla_v0_156.py') else x for x in command]
    return _popen(command,*args,**kwargs)
lifecycle.subprocess.Popen=Popen


def main():
    p=argparse.ArgumentParser();p.add_argument('--configuration',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args();cfg=lifecycle.check_definition(a.configuration);root=Path(a.output or cfg['output']).resolve();root.mkdir(parents=True,exist_ok=False);repo=Path(cfg['repo']);start=time.monotonic();emit(root/'configuration.json',dict(**cfg,actual_output=str(root),invocation=[sys.executable,*sys.argv]));failure=None
    try:
        tc=json.loads(Path(cfg['training_configuration']).read_text());emit(root/'feature_cache_reference.json',dict(path=tc['feature_cache_manifest'],sha256=tc['feature_cache_manifest_sha256'],read_only=True,original_raw_training_source=True))
        lifecycle.wait_idle(root);env=lifecycle.environment(root,repo,cfg['isaac_python']);env['CUDA_VISIBLE_DEVICES']='0'
        command=[cfg['vla_python'],str(repo/'tools/train_exp3_oft_feedback_v0_167.py'),'--configuration',cfg['training_configuration'],'--output',str(root/'training')];r=lifecycle.run_owned(command,root,'training',env,repo,True)
        if r['subprocess_returncode'] or not (root/'training/checkpoint_manifest.json').exists():raise RuntimeError('training_failed')
        command=[cfg['vla_python'],str(repo/'tools/verify_exp3_feedback_training_v0_167.py'),'--configuration',cfg['training_configuration'],'--run',str(root),'--output',str(root/'training_independent')];r=lifecycle.run_owned(command,root,'training_verifier',env,repo,False)
        if r['subprocess_returncode'] or json.loads((root/'training_independent/independent_verification.json').read_text()).get('paired_evaluation_allowed') is not True:raise RuntimeError('independent_training_gate_failed')
        for variant in json.loads(Path(cfg['evaluation_configuration']).read_text())['variants']:lifecycle.paired_variant(cfg,root,variant)
        command=[cfg['isaac_python'],str(repo/'tools/summarize_exp3_paired_learning_v0_141.py'),'--configuration',cfg['evaluation_configuration'],'--run',str(root),'--output',str(root/'final_admission')];r=lifecycle.run_owned(command,root,'paired_verifier',lifecycle.environment(root/'cpu_verifier',repo,cfg['isaac_python']),repo,False)
        if r['subprocess_returncode']:raise RuntimeError('paired_verifier_failed')
        status='training_and_paired_evaluation_completed_requires_report_review'
    except Exception as exc:
        status='blocked';failure=type(exc).__name__+':'+str(exc);emit(root/'failure.json',dict(failure_reason=failure,traceback=traceback.format_exc(),**ROBOT))
    emit(root/'execution_receipt.json',dict(status=status,failure_reason=failure,runtime_seconds=time.monotonic()-start,code_sha256=sha(__file__),configuration_sha256=sha(a.configuration),**ROBOT));emit(root/'completion_marker.json',dict(status=status,execution_receipt_sha256=sha(root/'execution_receipt.json')))


if __name__=='__main__':main()
