"""Generic local IPC robot policy boundary. No teacher, planner or object action."""
import base64,hashlib,io,json,socket,struct,time
import numpy as np
from PIL import Image

JOINTS=[f'panda_joint{i}' for i in range(1,8)]+['panda_finger_joint1','panda_finger_joint2']

def receive(sock,n):
    chunks=[]
    while n:
        part=sock.recv(n)
        if not part:raise RuntimeError('local_policy_disconnected_no_fallback')
        chunks.append(part);n-=len(part)
    return b''.join(chunks)

class Client:
    def __init__(self,path,variant,expected_adapter_sha256):
        if variant not in ('untrained_vla','trained_vla'):raise ValueError('invalid_policy_variant')
        self.socket=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);self.socket.settimeout(120);self.socket.connect(path);self.variant=variant;self.expected=expected_adapter_sha256
    def query(self,rgb,proprio,instruction):
        image=Image.fromarray(np.asarray(rgb,dtype=np.uint8),mode='RGB');b=io.BytesIO();image.save(b,format='PNG')
        payload=dict(rgb_png_base64=base64.b64encode(b.getvalue()).decode(),proprio=list(map(float,proprio)),instruction=instruction)
        message=json.dumps(payload,sort_keys=True,separators=(',',':')).encode();digest=hashlib.sha256(message).hexdigest();start=time.monotonic();self.socket.sendall(struct.pack('!I',len(message))+message)
        size=struct.unpack('!I',receive(self.socket,4))[0]
        if size>1<<20:raise RuntimeError('invalid_policy_response_size')
        response=json.loads(receive(self.socket,size))
        if response.get('status')!='ok':raise RuntimeError('policy_inference_failed_no_fallback:'+str(response.get('failure_reason')))
        if response.get('input_sha256')!=digest or response.get('model_variant')!=self.variant or response.get('adapter_sha256')!=self.expected:raise RuntimeError('policy_identity_or_input_hash_mismatch')
        if response.get('joint_names')!=JOINTS or response.get('frame')!='robot_joint_space' or response.get('arm_units')!='rad' or response.get('finger_units')!='m' or response.get('control_hz')!=10:raise RuntimeError('robot_action_spec_mismatch')
        values=np.asarray(response['joint_target_chunk'],dtype=float)
        if values.shape!=(8,9) or not np.isfinite(values).all():raise RuntimeError('invalid_robot_action_chunk')
        response['rpc_wall_seconds']=time.monotonic()-start;response['response_sha256']=hashlib.sha256(json.dumps(response,sort_keys=True).encode()).hexdigest()
        return values,response
    def close(self):self.socket.close()
