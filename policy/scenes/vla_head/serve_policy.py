"""Local VLA policy service for the scene suite: frozen OpenVLA-OFT features (v0.166 extraction) +
learned feedback-aware recurrent head (v0.167 architecture). Same Unix-socket contract as the v0.166
service: request {rgb_png_base64 (320x240 workspace crop), rgb_wrist_png_base64 (320x240, when the head was trained with 2 images),
proprio[9], instruction} -> {joint_target_chunk[8][9]}.
Memory is zeroed per connection (one connection per episode)."""

import argparse, base64, hashlib, io, json, os, socketserver, struct, sys, time, threading, signal
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--checkpoint-manifest", required=True)
p.add_argument("--variant", choices=["untrained_vla", "trained_vla"], required=True)
p.add_argument("--gpu", type=int, required=True)
p.add_argument("--output", required=True)
p.add_argument("--socket", required=True)
a = p.parse_args()
cfg = json.load(open(a.config))
sys.path.insert(0, cfg["tools_dir"])
out = Path(a.output)
out.mkdir(parents=True, exist_ok=False)
if Path(a.socket).exists():
    raise SystemExit("socket_already_owned")
Path(a.socket).parent.mkdir(parents=True, exist_ok=True)
cm = json.load(open(a.checkpoint_manifest))
key = "initial_adapter" if a.variant == "untrained_vla" else "final_adapter"
cp = Path(cm[key])
digest = cm[key + "_sha256"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


if sha(cp) != digest:
    raise SystemExit("checkpoint_changed")
import numpy as np, torch
from PIL import Image
from safetensors.torch import load_file
import exp3_oft_recurrent_adapter_v0_166 as adapter
import exp3_oft_feedback_adapter_v0_167 as feedback

SCALES = np.asarray(cm["horizon_arm_scale_rad"], np.float32)
FINGER_MAX = 0.04
JOINTS = ["panda_joint%d" % i for i in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"]
NIMG = int(cfg.get("num_images", 1))
KEYS = {"rgb_png_base64", "proprio", "instruction"} | ({"rgb_wrist_png_base64"} if NIMG == 2 else set())
if int(cfg["feature_dim"]) != 4096 * NIMG:
    raise SystemExit("feature_dim_mismatch")
torch.cuda.set_device(a.gpu)
device = torch.device("cuda", a.gpu)
os.environ["HF_MODULES_CACHE"] = str(out / "hf_local_code_cache")
start = time.time()
base, processor = adapter.build_features(cfg, device)
head = feedback.RecurrentHead(cfg["feature_dim"], cfg["memory_width"], cm["proprio_mean"], cm["proprio_std"]).to(device)
head.load_state_dict(load_file(str(cp)), strict=True)
head.eval()
torch.cuda.synchronize()
load = time.time() - start
lock = threading.Lock()
(out / "policy_identity.json").write_text(
    json.dumps(
        dict(
            model_variant=a.variant,
            adapter_sha256=digest,
            checkpoint_path=str(cp),
            base_checkpoint=cfg["base_checkpoint"],
            gpu_index=a.gpu,
            gpu_name=torch.cuda.get_device_name(a.gpu),
            model_load_seconds=load,
            model_inputs=(["raw RGB (workspace crop)", "wrist RGB"] if NIMG == 2 else ["raw RGB"])
            + ["robot joint proprioception", "instruction"],
            num_images=NIMG,
            temporal_memory="GRU over prior observations only; zero reset per connection",
            horizon_arm_scale_rad=SCALES.tolist(),
            code_sha256=sha(__file__),
        ),
        indent=1,
    )
)


def receive(sock, n):
    buf = b""
    while len(buf) < n:
        part = sock.recv(n - len(buf))
        if not part:
            raise ConnectionError("disconnected")
        buf += part
    return buf


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        memory = None
        qi = 0
        while True:
            try:
                size = struct.unpack("!I", receive(self.request, 4))[0]
                if size > 4 << 20:
                    raise ValueError("request_too_large")
                raw = receive(self.request, size)
                data = json.loads(raw)
                received = time.monotonic()
                if set(data) != KEYS:
                    raise ValueError("unregistered_policy_input")
                q = np.asarray(data["proprio"], np.float32)
                if q.shape != (9,) or not np.isfinite(q).all():
                    raise ValueError("invalid_proprio")
                ins = []
                for key in ["rgb_png_base64", "rgb_wrist_png_base64"][:NIMG]:
                    with Image.open(io.BytesIO(base64.b64decode(data[key]))) as im:
                        if im.size != (320, 240):
                            raise ValueError("image_contract_changed")
                        ins.append(
                            processor(adapter.prompt(data["instruction"]), im.convert("RGB"), return_tensors="pt")
                        )
                with lock:
                    tick = time.monotonic()
                    fts = []
                    for inputs in ins:
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                            fts.append(adapter.features(base, **inputs))
                    ft = torch.cat(fts, -1)
                    with torch.inference_mode():
                        pred, memory = head(ft[:, None], torch.tensor(q[None, None], device=device), memory)
                        pred = pred[0, 0].cpu().numpy()
                    torch.cuda.synchronize()
                    runtime = time.monotonic() - tick
                targets = np.concatenate(
                    [q[None, :7] + pred[:, :7] * SCALES[:, None], (np.clip(pred[:, 7:], -1, 1) + 1) * 0.5 * FINGER_MAX],
                    -1,
                )
                resp = dict(
                    status="ok",
                    joint_target_chunk=targets.tolist(),
                    normalized_prediction=pred.tolist(),
                    input_sha256=hashlib.sha256(raw).hexdigest(),
                    model_variant=a.variant,
                    adapter_sha256=digest,
                    joint_names=JOINTS,
                    frame="robot_joint_space",
                    control_hz=10,
                    model_runtime_seconds=runtime,
                    queue_seconds=tick - received,
                    query_index=qi,
                    fallback_used=False,
                )
                qi += 1
                with lock, (out / "inference.jsonl").open("a") as f:
                    f.write(json.dumps(resp, sort_keys=True) + "\n")
                enc = json.dumps(resp).encode()
                self.request.sendall(struct.pack("!I", len(enc)) + enc)
            except (ConnectionError, BrokenPipeError):
                return
            except Exception as exc:
                if "disconnected" in str(exc):
                    return
                enc = json.dumps(
                    dict(status="failed", failure_reason=type(exc).__name__ + ":" + str(exc)[:200])
                ).encode()
                try:
                    self.request.sendall(struct.pack("!I", len(enc)) + enc)
                except Exception:
                    pass
                return


class Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


with Server(a.socket, Handler) as server:
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    server.timeout = 0.5
    (out / "ready.json").write_text(json.dumps(dict(pid=os.getpid(), socket=a.socket, variant=a.variant, load_s=load)))
    print("ready", a.socket, flush=True)
    while not stop.is_set():
        server.handle_request()
(out / "execution_receipt.json").write_text(
    json.dumps(
        dict(
            status="service_stopped",
            runtime_s=time.time() - start,
            model_variant=a.variant,
            peak_memory_bytes=torch.cuda.max_memory_allocated(a.gpu),
        )
    )
)
