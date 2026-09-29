"""Local recurrent VLA service, memory isolated and reset per episode socket."""
import argparse,base64,hashlib,io,json,os,socketserver,struct,sys,time,threading,signal
from pathlib import Path
from run_exp3_remote_pilot_jobs_v0_125 import emit,sha,ROBOT
from exp3_local_robot_policy_client_v0_134 import receive,JOINTS
import exp3_oft_recurrent_adapter_v0_166 as adapter


def main():
    p=argparse.ArgumentParser();p.add_argument('--training-configuration',type=Path,required=True);p.add_argument('--checkpoint-manifest',type=Path,required=True);p.add_argument('--variant',choices=['untrained_vla','trained_vla'],required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--socket',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if a.socket.exists():raise ValueError('socket_already_owned')
    a.socket.parent.mkdir(parents=True,exist_ok=True);cfg=json.loads(a.training_configuration.read_text());cm=json.loads(a.checkpoint_manifest.read_text());key='initial_adapter' if a.variant=='untrained_vla' else 'final_adapter';cp=Path(cm[key]);digest=cm[key+'_sha256']
    if sha(cp)!=digest:raise ValueError('checkpoint_changed')
    import numpy as np,torch
    from PIL import Image
    from safetensors.torch import load_file
    torch.cuda.set_device(a.gpu);device=torch.device('cuda',a.gpu);os.environ['HF_MODULES_CACHE']=str(a.output/'hf_local_code_cache');start=time.monotonic();base,processor=adapter.build_features(cfg,device);head=adapter.RecurrentHead(cfg['feature_dim'],cfg['memory_width']).to(device);head.load_state_dict(load_file(str(cp)),strict=True);head.eval();torch.cuda.synchronize();load=time.monotonic()-start;lock=threading.Lock()
    identity=dict(model_variant=a.variant,adapter_sha256=digest,base_checkpoint_inventory_sha256=sha(cfg['checkpoint_inventory']),checkpoint_path=str(cp),gpu_index=a.gpu,gpu_name=torch.cuda.get_device_name(a.gpu),model_load_seconds=load,model_inputs=['raw RGB','robot joint proprioception','instruction'],temporal_memory='learned GRU from prior allowed observations only; zero reset on new episode connection',transport='local Unix socket; no remote API',code_sha256=sha(__file__),training_checkpoint_read=a.variant=='trained_vla',**ROBOT);emit(a.output/'policy_identity.json',identity)
    emit(a.output/'precision_adapter_binding.json',dict(adapter_id=adapter.ADAPTER_ID,horizon_arm_scale_rad=list(adapter.HORIZON_ARM_SCALE),finger_max_m=adapter.FINGER_MAX,no_object_commands=True,no_fallback=True,memory_reset='new_episode_connection',memory_inputs=['prior RGB-language features','prior measured robot proprio'],future_inputs=False))
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            memory=None;query_index=0
            while True:
                try:
                    size=struct.unpack('!I',receive(self.request,4))[0]
                    if size>4<<20:raise ValueError('request_too_large')
                    raw=receive(self.request,size);data=json.loads(raw);received=time.monotonic()
                    if set(data)!=set(['rgb_png_base64','proprio','instruction']):raise ValueError('unregistered_policy_input')
                    q=np.asarray(data['proprio'],np.float32)
                    if q.shape!=(9,) or not np.isfinite(q).all():raise ValueError('invalid_proprio')
                    with Image.open(io.BytesIO(base64.b64decode(data['rgb_png_base64']))) as im:
                        if im.size!=(320,240):raise ValueError('image_contract_changed')
                        inputs=processor(adapter.prompt(data['instruction']),im.convert('RGB'),return_tensors='pt')
                    with lock:
                        tick=time.monotonic();inputs={k:v.to(device) for k,v in inputs.items()}
                        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):ft=adapter.features(base,**inputs)
                        with torch.inference_mode():prediction,memory=head(ft[:,None],torch.tensor(q[None,None],device=device),memory);prediction=prediction[0,0].cpu().numpy()
                        torch.cuda.synchronize();runtime=time.monotonic()-tick
                    targets=np.concatenate([q[None,:7]+prediction[:,:7]*np.asarray(adapter.HORIZON_ARM_SCALE)[:,None],(np.clip(prediction[:,7:],-1,1)+1)*.5*adapter.FINGER_MAX],-1)
                    response=dict(status='ok',joint_target_chunk=targets.tolist(),normalized_prediction=prediction.tolist(),input_sha256=hashlib.sha256(raw).hexdigest(),model_variant=a.variant,adapter_sha256=digest,joint_names=JOINTS,frame='robot_joint_space',arm_units='rad',finger_units='m',control_hz=10,model_runtime_seconds=runtime,queue_and_preprocess_seconds=tick-received,peak_memory_bytes=torch.cuda.max_memory_allocated(a.gpu),fallback_used=False,query_index=query_index,memory_reset_at_episode_start=True,memory_source='prior_allowed_observations_only',object_state_input=False,future_observation_input=False)
                    query_index+=1
                    with lock,(a.output/'inference.jsonl').open('a') as f:f.write(json.dumps(response,sort_keys=True)+'\n')
                    encoded=json.dumps(response).encode();self.request.sendall(struct.pack('!I',len(encoded))+encoded)
                except (ConnectionError,BrokenPipeError):return
                except Exception as exc:
                    if 'disconnected' in str(exc):return
                    encoded=json.dumps(dict(status='failed',failure_reason=type(exc).__name__+':'+str(exc))).encode();self.request.sendall(struct.pack('!I',len(encoded))+encoded);return
    class Server(socketserver.ThreadingUnixStreamServer):daemon_threads=True
    with Server(str(a.socket),Handler) as server:
        stop=threading.Event();signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set());server.timeout=.5;emit(a.output/'ready.json',dict(pid=os.getpid(),socket=str(a.socket),identity_sha256=sha(a.output/'policy_identity.json')))
        while not stop.is_set():server.handle_request()
    emit(a.output/'execution_receipt.json',dict(status='service_stopped_after_evaluation',runtime_seconds=time.monotonic()-start,model_load_seconds=load,model_variant=a.variant,peak_memory_bytes=torch.cuda.max_memory_allocated(a.gpu),command=[sys.executable,*sys.argv],**ROBOT))


if __name__=='__main__':main()
