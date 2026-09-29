"""Isaac Sim scene runner for the AffordCraft multi-asset manipulation suite (v20).

Builds one randomized scene episode per loop from `scene_specs.py`, runs either the scripted
multi-step teacher (`--mode teacher`), a served VLA policy (`--mode policy`), or a DAgger-style mixed
rollout (`--mode dagger`: the teacher's plan runs while the served policy executes random bursts of
ticks; the teacher's clamped target stays the recorded label), records the 10 Hz policy observations
(320x240 workspace crop of the third-person render and the 320x240 wrist view), proprio, executed 9-D
joint targets, object/joint states and contact evidence, and evaluates every episode with the
independent `scene_eval.py` predicate. Read-only on assets; writes only under --output."""

import argparse, json, time, math, os, sys, random, hashlib, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# Optional start-up workaround: when the per-user inotify watch budget of the host is exhausted, Kit's Omniverse-client
# file thread can abort the process during start-up. If an LD_PRELOAD shim libnoinotify.so that makes inotify
# unavailable is placed next to this file, the runner re-executes itself with it (Kit and omniclient then disable
# directory-change notifications; simulation is unaffected). The shim is not part of this release.
_SHIM = HERE / "libnoinotify.so"
if _SHIM.is_file() and not os.environ.get("SR_NO_INOTIFY_SHIM") and str(_SHIM) not in os.environ.get("LD_PRELOAD", ""):
    _env = dict(os.environ)
    _env["LD_PRELOAD"] = str(_SHIM) + ((":" + _env["LD_PRELOAD"]) if _env.get("LD_PRELOAD") else "")
    print(json.dumps(dict(reexec_with_inotify_shim=str(_SHIM), pid=os.getpid())), flush=True)
    os.execve(sys.executable, [sys.executable] + sys.argv, _env)
import scene_specs as S
from scene_eval import evaluate, HOLD_TICKS

p = argparse.ArgumentParser()
p.add_argument("--gpu", type=int, required=True)
p.add_argument("--scene", required=True)
p.add_argument("--episodes", type=int, default=1)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--output", required=True)
p.add_argument("--mode", choices=["teacher", "policy", "dagger"], default="teacher")
p.add_argument("--policy-socket", default="")
p.add_argument("--variant", default="")
p.add_argument("--budget-ticks", type=int, default=800)
p.add_argument(
    "--franka",
    default=os.environ.get("AFFORDCRAFT_FRANKA_USD", "franka_flat.usd"),
    help="flattened local copy of the Isaac Sim Franka USD",
)
p.add_argument("--video", default="true")
p.add_argument("--stop-file", default="")
p.add_argument("--keep-frames", default="true")
p.add_argument("--start-index", type=int, default=0)
p.add_argument("--joint-friction", type=float, default=-1.0)
p.add_argument("--plan-prefix", type=int, default=999)
p.add_argument("--settle-ticks", type=int, default=0)
p.add_argument("--cam-res", default="1280x960")
p.add_argument("--cam-fov", type=float, default=42.0)
p.add_argument("--cam-az", type=float, default=-140.0)
p.add_argument("--cam-el", type=float, default=30.0)
p.add_argument("--cam-dist", type=float, default=2.6)
p.add_argument("--cam-target", default="-0.25,0.0,1.20")
p.add_argument(
    "--camera-probe",
    default="",
    help="semicolon-separated az,el,dist,fov[,tx,ty,tz] specs: render each from the initial and the opened scene state, then exit",
)
p.add_argument(
    "--obs-crop",
    default="40,60,1040,810",
    help="x0,y0,x1,y1 (pixels of a 1280x960 render, scaled to --cam-res) cropped from the third-person render for the 320x240 policy observation; empty = full frame",
)
p.add_argument("--wrist", default="true")
p.add_argument("--wrist-res", default="320x240")
p.add_argument("--wrist-fov", type=float, default=70.0)
p.add_argument(
    "--wrist-mount",
    default="0.055,0.0,0.02,25",
    help="wrist camera in the panda_hand frame: x,y,z offset (m) and pitch (deg) of the optical axis from the hand +z (finger direction) toward -x",
)
p.add_argument(
    "--dagger-beta",
    type=float,
    default=0.35,
    help="dagger mode: expected fraction of ticks executed by the served policy",
)
p.add_argument("--dagger-burst", default="8,40")
p.add_argument("--dagger-prefix-max", type=int, default=200)
p.add_argument("--dagger-prefix-prob", type=float, default=0.5)
a = p.parse_args()
OUT = Path(a.output)
OUT.mkdir(parents=True, exist_ok=True)
if os.environ.get("SR_MINIMAL"):
    for _k in (
        "SR_NO_DRIVES",
        "SR_NO_SOLVER",
        "SR_NO_SUBSCRIBE",
        "SR_NO_RESUME",
        "SR_CAM_INIT_EARLY",
        "SR_LINK_VIEW",
        "SR_NO_PADS",
        "SR_NO_CONTACT",
        "SR_NO_DEFAULT_STATE",
        "SR_NO_HIDE",
    ):
        os.environ[_k] = "1"
CAM_W, CAM_H = [int(v) for v in a.cam_res.lower().split("x")]
OBS_RES = (320, 240)
from isaacsim import SimulationApp

_APP_W, _APP_H = [int(v) for v in os.environ.get("SR_APP_RES", "320x240").lower().split("x")]
app = SimulationApp(
    {
        "headless": True,
        "hide_ui": True,
        "width": _APP_W,
        "height": _APP_H,
        "renderer": "RayTracedLighting",
        "anti_aliasing": 0,
        "multi_gpu": False,
        "max_gpu_count": 1,
        "active_gpu": a.gpu,
        "physics_gpu": a.gpu,
        "fast_shutdown": True,
    }
)
import numpy as np, omni.usd, carb
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.prims import Articulation as CoreArticulation
from isaacsim.core.utils.stage import create_new_stage, is_stage_loading
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.sensors.camera import Camera
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.kinematics_solver import KinematicsSolver
from omni.physx import get_physx_simulation_interface, get_physx_scene_query_interface
from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, UsdShade, Gf, PhysxSchema, PhysicsSchemaTools
import isaacsim.core.utils.numpy.rotations as rot_utils
from PIL import Image

TABLE = S.TABLE_Z
BASE = np.array(S.BASE)
Q0 = np.array([0, -0.9427, 0, -2.6669, 0, 1.664, math.pi / 4, 0.04, 0.04])
FPS = 10
SUB = 6
JOINT_LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0])
JOINT_HI = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04])
TRANSIT = TABLE + 0.34
OPEN = 0.04
CLOSED = 0.0
VMAX_TICK = 0.06  # 0.6 rad/s at 10 Hz, identical to limit_target() used in policy mode
cfg_gripper = dict(type="force", stiffness=4000.0, damping=150.0, max_force=120.0)
# gripper geometry (measured from the flat Franka asset): right_gripper frame is 1.2 cm above the fingertip and 3.4 cm below the palm hull
GRASP_DEPTH = 0.022
PALM_ABOVE_EE = 0.034
HAND_TOP_ABOVE_EE = 0.14
TIP_BELOW_EE = 0.012
# handle-part table of the articulated library entries (PartNet part samples per link), as loaded by the executed runs
HANDLES = json.load(open(os.environ.get("AFFORDCRAFT_SCENE_HANDLES", str(HERE / "data" / "handles_v1.json"))))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def bounds_of(prims):
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], False, True)
    lo = np.full(3, 1e9)
    hi = np.full(3, -1e9)
    for q in prims:
        r = bbox.ComputeWorldBound(q).ComputeAlignedRange()
        lo = np.minimum(lo, np.array(r.GetMin()))
        hi = np.maximum(hi, np.array(r.GetMax()))
    return lo, hi


def collision_meshes(prim):
    return [
        q
        for q in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
        if q.IsA(UsdGeom.Mesh) and q.GetName().startswith("collision")
    ]


def wait_load():
    for _ in range(100):
        app.update()
        if not is_stage_loading():
            break


def pose_R(approach, finger):
    z = np.asarray(approach, float)
    z = z / np.linalg.norm(z)
    y = np.asarray(finger, float)
    y = y - z * np.dot(y, z)
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    return np.stack([x, y, z], 1)


DOWN = lambda finger=(0, 1.0, 0): pose_R([0, 0, -1], finger)


def slerp_R(R0, R1, s):
    Rrel = np.asarray(R0).T @ np.asarray(R1)
    c = min(1.0, max(-1.0, (np.trace(Rrel) - 1) / 2))
    ang = math.acos(c)
    if ang < 1e-6 or ang > math.pi - 1e-3:
        return np.asarray(R1)
    ax = np.array([Rrel[2, 1] - Rrel[1, 2], Rrel[0, 2] - Rrel[2, 0], Rrel[1, 0] - Rrel[0, 1]]) / (2 * math.sin(ang))
    return np.asarray(R0) @ rot_axis(ax, s * ang)


def rot_axis(ax, ang):
    ax = np.asarray(ax, float)
    ax = ax / np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def look_at(pos, tgt):
    f = np.array(tgt) - np.array(pos)
    f /= np.linalg.norm(f)
    up = np.array([0, 0, 1.0])
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return rot_utils.rot_matrices_to_quats(np.stack([f, -r, u], 1))


def limit_target(current, requested, dt_s=0.1, vmax=0.6):
    delta = requested - current
    m = float(np.max(np.abs(delta)))
    f = min(1.0, vmax * dt_s / m) if m else 1.0
    return current + f * delta


def mesh_points(meshes, per_mesh=None, limit=6000):
    pts = []
    for m in meshes:
        P = UsdGeom.Mesh(m).GetPointsAttr().Get()
        if not P:
            continue
        M = omni.usd.get_world_transform_matrix(m)
        arr = np.array([[q[0], q[1], q[2]] for q in P], float)
        if per_mesh and len(arr) > per_mesh:
            arr = arr[np.random.default_rng(0).choice(len(arr), per_mesh, replace=False)]
        R = np.array([[M[0][0], M[0][1], M[0][2]], [M[1][0], M[1][1], M[1][2]], [M[2][0], M[2][1], M[2][2]]])
        t = np.array([M[3][0], M[3][1], M[3][2]])
        pts.append(arr @ R + t)
    P = np.concatenate(pts) if pts else np.zeros((1, 3))
    return P if len(P) <= limit else P[np.random.default_rng(0).choice(len(P), limit, replace=False)]


class PolicyClient:
    def __init__(self, path):
        import socket

        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(180)
        self.sock.connect(path)

    def query(self, rgb, wrist, proprio, instruction):
        import base64, io, struct

        b = io.BytesIO()
        Image.fromarray(rgb).save(b, format="PNG")
        req = dict(
            rgb_png_base64=base64.b64encode(b.getvalue()).decode(),
            proprio=[float(x) for x in proprio],
            instruction=instruction,
        )
        if wrist is not None:
            bw = io.BytesIO()
            Image.fromarray(wrist).save(bw, format="PNG")
            req["rgb_wrist_png_base64"] = base64.b64encode(bw.getvalue()).decode()
        msg = json.dumps(req, sort_keys=True, separators=(",", ":")).encode()
        self.sock.sendall(struct.pack("!I", len(msg)) + msg)
        n = struct.unpack("!I", self._recv(4))[0]
        r = json.loads(self._recv(n))
        if r.get("status") != "ok":
            raise RuntimeError("policy_failed:" + str(r)[:200])
        return np.asarray(r["joint_target_chunk"], float), r

    def _recv(self, n):
        buf = b""
        while len(buf) < n:
            part = self.sock.recv(n - len(buf))
            if not part:
                raise ConnectionError("policy_disconnected")
            buf += part
        return buf

    def close(self):
        self.sock.close()


class Episode:
    def __init__(self, spec, name, index, rng, ep_dir, seed):
        self.spec = spec
        self.name = name
        self.index = index
        self.rng = rng
        self.dir = ep_dir
        self.seed = seed
        self.log = []
        self.trace = []
        self.tick = 0
        self.frames = []
        self.held = None
        self.sub = None
        self.writer = None
        self.first_frame = None
        self.last_frame = None
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "frames").mkdir(exist_ok=True)
        (self.dir / "frames_wrist").mkdir(exist_ok=True)
        self.wcam = None
        self.crop = None
        self.dagger = None
        self.dagger_stats = None
        self.last_obs = None
        self.last_wobs = None

    def note(self, **k):
        k["tick"] = self.tick
        self.log.append(k)
        print(json.dumps(k, default=str), flush=True)

    # ---------------- scene construction ----------------
    def build(self):
        create_new_stage()
        self.stage = stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        sc = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
        sc.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
        sc.CreateGravityMagnitudeAttr().Set(9.81)
        self.world = world = World(physics_dt=1 / 60, rendering_dt=1 / FPS, stage_units_in_meters=1.0)
        world.scene.add(
            FixedCuboid(
                prim_path="/World/Table",
                name="table",
                position=np.array([0, 0, TABLE - 0.025]),
                scale=np.array([3.0, 3.0, 0.05]),
                color=np.array([0.42, 0.36, 0.30]),
            )
        )
        world.scene.add(
            FixedCuboid(
                prim_path="/World/Pedestal",
                name="pedestal",
                position=np.array([BASE[0], BASE[1], (TABLE + BASE[2]) / 2]),
                scale=np.array([0.24, 0.24, BASE[2] - TABLE]),
                color=np.array([0.3, 0.32, 0.34]),
            )
        )
        dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
        dome.CreateIntensityAttr(320.0)
        key = UsdLux.DistantLight.Define(stage, "/World/Key")
        key.CreateIntensityAttr(1500.0)
        UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(35, -25, 25))
        self.assets = {}
        self.objects = {}
        self.placements = {}
        baked = json.load(open(Path(S.BAKED_DIR) / "baked_manifest.json"))
        for art in self.spec["articulated"]:
            adef = S.ARTICULATED[art["asset"]]
            b = baked[art["asset"]]
            usd = b["usd"]
            jit = art.get("jitter", 0.0)
            plat = float(adef.get("platform", 0.0))
            yaw = art.get("yaw", 180.0) + self.rng.uniform(-4, 4)
            fx = art["front_x"] + self.rng.uniform(-jit, jit)
            y = art["y"] + self.rng.uniform(-jit, jit)
            prim = self.place(art["key"], usd, fx, y, yaw, front_x=True, z_extra=plat + 0.003)
            info = dict(
                prim=prim,
                key=art["key"],
                asset=art["asset"],
                cid=adef["cid"],
                scale=adef["scale"],
                facing=adef.get("facing", "handle"),
                initial=art.get("initial"),
                links={},
                platform=plat,
            )
            for q in Usd.PrimRange(prim):
                if q.HasAPI(UsdPhysics.RigidBodyAPI):
                    info["links"][q.GetName()] = q
            info["joints"] = self.joint_frames(prim)
            self.assets[art["key"]] = info
            jn = self.primary_joint(art["key"])
            flip = False
            if jn is not None:
                j = info["joints"][jn]
                c = np.array(self.placements[art["key"]]["center"])
                if info["facing"] == "handle":
                    flip = self.handle_of(art["key"], jn)["center"][0] > c[0]
                elif info["facing"] == "hinge_front":
                    flip = j["pivot"][0] > c[0]
                elif info["facing"] == "hinge_back":
                    flip = j["pivot"][0] < c[0]
            if flip:
                yaw += 180.0
                prim = self.place(art["key"], usd, fx, y, yaw, front_x=True, existing=prim, z_extra=plat + 0.003)
                info["joints"] = self.joint_frames(prim)
                self.note(auto_faced=art["key"], yaw=round(yaw, 1))
            if plat > 0:
                lo, hi = self.placements[art["key"]]["aabb"]
                world.scene.add(
                    FixedCuboid(
                        prim_path="/World/Platform_" + art["key"],
                        name="platform_" + art["key"],
                        position=np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, TABLE + plat / 2]),
                        scale=np.array([hi[0] - lo[0] + 0.04, hi[1] - lo[1] + 0.04, plat]),
                        color=np.array([0.5, 0.5, 0.52]),
                    )
                )
            for q in Usd.PrimRange(prim):
                if q.HasAPI(UsdPhysics.ArticulationRootAPI):
                    PhysxSchema.PhysxArticulationAPI.Apply(q).CreateEnabledSelfCollisionsAttr().Set(False)
            for jname, j in info["joints"].items():
                if adef.get("clamp_deg") is not None:
                    lo_, hi_ = adef["clamp_deg"]
                    j["prim"].GetAttribute("physics:lowerLimit").Set(float(max(j["lower"], lo_)))
                    j["prim"].GetAttribute("physics:upperLimit").Set(float(min(j["upper"], hi_)))
                    j["lower"] = float(max(j["lower"], lo_))
                    j["upper"] = float(min(j["upper"], hi_))
                PhysxSchema.PhysxJointAPI.Apply(j["prim"]).CreateJointFrictionAttr().Set(
                    float(adef.get("friction", 0.3)) if a.joint_friction < 0 else a.joint_friction
                )
                kind = "angular" if j["type"] == "revolute" else "linear"
                d = UsdPhysics.DriveAPI.Apply(j["prim"], kind)
                d.CreateStiffnessAttr().Set(0.0)
                d.CreateDampingAttr().Set(float(adef.get("damping", 0.02 if kind == "angular" else 2.0)))
                d.CreateMaxForceAttr().Set(1e6)
            info["link_aabb_closed"] = {ln: bounds_of(collision_meshes(pr)) for ln, pr in info["links"].items()}
            init = art.get("initial")
            info["initial_q_authored"] = {}
            if isinstance(init, dict):
                for jname, v in init.items():
                    j = info["joints"][jname]
                    kind = "angular" if j["type"] == "revolute" else "linear"
                    st = PhysxSchema.JointStateAPI.Apply(j["prim"], kind)
                    st.CreatePositionAttr().Set(float(v))
                    st.CreateVelocityAttr().Set(0.0)
                    info["initial_q_authored"][jname] = float(v)
            children = {j["body1"] for j in info["joints"].values()}
            body_links = [ln for ln in info["links"] if ln not in children]
            if body_links and adef.get("anchor", True):
                from pxr import Sdf

                bl = info["links"][body_links[0]]
                fj = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(str(prim.GetPath()) + "/Joints/world_anchor"))
                fj.CreateBody1Rel().SetTargets([bl.GetPath()])
                M = omni.usd.get_world_transform_matrix(bl)
                fj.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in M.ExtractTranslation()]))
                qr = M.ExtractRotationQuat()
                fj.CreateLocalRot0Attr().Set(
                    Gf.Quatf(float(qr.GetReal()), Gf.Vec3f(*[float(v) for v in qr.GetImaginary()]))
                )
                fj.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
                fj.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
                info["anchored"] = body_links[0]
            info["view"] = world.scene.add(
                CoreArticulation(
                    prim_paths_expr=str(prim.GetPath()), name="art_" + art["key"], reset_xform_properties=False
                )
            )
        occupied = [self.placements[k]["aabb"] for k in self.assets]
        for ro in self.spec["rigid"]:
            rdef = S.RIGID[ro["asset"]]
            usd = S.asset_usd(rdef["cid"])
            z_abs = None
            if "inside" in ro:
                x, y, z_abs = self.inside_point(*ro["inside"])
            else:
                for _ in range(40):
                    x = self.rng.uniform(*ro["region"]["x"])
                    y = self.rng.uniform(*ro["region"]["y"])
                    if all(
                        not (ab[0][0] - 0.06 < x < ab[1][0] + 0.06 and ab[0][1] - 0.06 < y < ab[1][1] + 0.06)
                        for ab in occupied
                    ):
                        break
            yaw = self.rng.uniform(0, 360) if rdef["label"] in ("pen", "book", "mouse") else self.rng.uniform(-20, 20)
            prim = self.place(ro["key"], usd, x, y, 0.0, z_abs=z_abs)
            if ro.get("align"):
                P = mesh_points(collision_meshes(prim))
                Q = P[:, :2] - P[:, :2].mean(0)
                w, v = np.linalg.eigh(Q.T @ Q / len(Q))
                minor = v[:, 0]
                ang = math.degrees(math.atan2(minor[1], minor[0]))
                yaw = (90.0 - ang if ro["align"] == "y" else -ang) + self.rng.uniform(-8, 8)
            prim = self.place(ro["key"], usd, x, y, yaw, z_abs=z_abs, existing=prim)
            self.objects[ro["key"]] = dict(
                prim=prim, key=ro["key"], asset=ro["asset"], label=rdef["label"], meshes=collision_meshes(prim)
            )
            occupied.append(self.placements[ro["key"]]["aabb"])
        used = {S.RIGID[ro["asset"]]["label"] for ro in self.spec["rigid"]}
        cands = [k for k in S.DISTRACTOR_KEYS if S.RIGID[k]["label"] not in used]
        for i in range(int(self.spec.get("distractors", 0))):
            if not cands:
                break
            k = self.rng.choice(cands)
            cands = [c for c in cands if S.RIGID[c]["label"] != S.RIGID[k]["label"]]
            usd = S.asset_usd(S.RIGID[k]["cid"])
            corridors = [
                (
                    self.placements[k]["aabb"][0][0] - 0.36,
                    self.placements[k]["aabb"][0][0] + 0.02,
                    self.placements[k]["aabb"][0][1] - 0.08,
                    self.placements[k]["aabb"][1][1] + 0.08,
                )
                for k in self.assets
            ]
            for _ in range(80):
                x = self.rng.uniform(-0.50, -0.20)
                y = self.rng.uniform(-0.46, 0.46)
                if all(
                    not (ab[0][0] - 0.08 < x < ab[1][0] + 0.08 and ab[0][1] - 0.08 < y < ab[1][1] + 0.08)
                    for ab in occupied
                ) and all(not (c[0] < x < c[1] and c[2] < y < c[3]) for c in corridors):
                    break
            else:
                continue
            key = "distractor_%d" % i
            prim = self.place(key, usd, x, y, self.rng.uniform(0, 360))
            self.objects[key] = dict(
                prim=prim, key=key, asset=k, label=S.RIGID[k]["label"], meshes=collision_meshes(prim), distractor=True
            )
            occupied.append(self.placements[key]["aabb"])
        for k, info in self.assets.items():
            self.set_material(info["prim"], float(S.ARTICULATED[info["asset"]].get("mu", 1.0)))
        for k, o in self.objects.items():
            self.set_material(o["prim"], 1.5)
        tops = [self.placements[k]["aabb"][1][2] for k in self.assets]
        self.transit_z = float(min(TABLE + 0.64, max(TRANSIT, (max(tops) + 0.10) if tops else TRANSIT)))
        # robot: author the base translation on the prim (the flat asset ignores the wrapper position before reset)
        self.fr = fr = world.scene.add(
            Franka(
                prim_path="/World/Franka",
                name="franka",
                position=BASE,
                orientation=np.array([1.0, 0, 0, 0]),
                usd_path=a.franka,
            )
        )
        wait_load()
        if not os.environ.get("SR_NO_DEFAULT_STATE"):
            fr.set_joints_default_state(positions=Q0)
        xf = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Franka"))
        ops = xf.GetOrderedXformOps()
        if ops and ops[0].GetOpType() == UsdGeom.XformOp.TypeTranslate:
            ops[0].Set(Gf.Vec3d(*[float(v) for v in BASE]))
        # gripper: the imported asset uses acceleration-type finger drives (squeeze force ~0.1 N with 14 g fingers);
        # use force-type drives with the Isaac Lab Franka gains so the pinch can hold handles and objects
        for jp in (
            ()
            if os.environ.get("SR_NO_DRIVES")
            else ("/World/Franka/joints/panda_finger_joint1", "/World/Franka/joints/panda_finger_joint2")
        ):
            jprim = stage.GetPrimAtPath(jp)
            if jprim.IsValid():
                d = UsdPhysics.DriveAPI.Apply(jprim, "linear")
                d.CreateTypeAttr().Set("force")
                d.CreateStiffnessAttr().Set(float(cfg_gripper["stiffness"]))
                d.CreateDampingAttr().Set(float(cfg_gripper["damping"]))
                d.CreateMaxForceAttr().Set(float(cfg_gripper["max_force"]))
        self.gripper_drive = dict(cfg_gripper)
        self.pads = self.add_finger_pads() if not os.environ.get("SR_NO_PADS") else {}
        if os.environ.get("SR_TOUCH_PROXIES"):
            touched = 0
            for q in Usd.PrimRange(stage.GetPrimAtPath("/World/Franka"), Usd.TraverseInstanceProxies()):
                if q.IsA(UsdGeom.Mesh) and "collision" not in q.GetPath().pathString.lower():
                    m = UsdGeom.Mesh(q)
                    P = m.GetPointsAttr().Get()
                    _ = m.GetNormalsAttr().Get()
                    _ = UsdShade.MaterialBindingAPI(q).ComputeBoundMaterial()[0]
                    for c in q.GetChildren():
                        if c.GetTypeName() == "GeomSubset":
                            _ = UsdShade.MaterialBindingAPI(c).ComputeBoundMaterial()[0]
                            _ = UsdGeom.Subset(c).GetIndicesAttr().Get()
                    touched += 1
            self.note(robot_visual_proxies_touched=touched)
        if os.environ.get("SR_DEINSTANCE"):
            n_de = 0
            for q in Usd.PrimRange(stage.GetPrimAtPath("/World/Franka")):
                if q.IsInstanceable():
                    q.SetInstanceable(False)
                    n_de += 1
            self.note(robot_deinstanced=n_de)
        # render only the Franka's visual meshes: the imported convex collision hulls (purpose default) were drawn on top of them
        hidden = 0
        for q in Usd.PrimRange(stage.GetPrimAtPath("/World/Franka")):
            if q.GetName() == "collisions" and not os.environ.get("SR_NO_HIDE"):
                UsdGeom.Imageable(q).MakeInvisible()
                hidden += 1
        self.note(robot_collision_hulls_hidden=hidden)
        _cn = os.environ.get("SR_CAM_NAME") or f"Cam_{self.index:04d}"
        self.cam = cam = world.scene.add(
            Camera(prim_path=f"/World/{_cn}", name=_cn.lower(), resolution=(CAM_W, CAM_H), frequency=FPS)
        )
        if a.wrist == "true":
            WW, WH = [int(v) for v in a.wrist_res.lower().split("x")]
            self.wcam = world.scene.add(
                Camera(
                    prim_path="/World/Franka/panda_hand/wrist_cam",
                    name=f"wrist_{_cn.lower()}",
                    resolution=(WW, WH),
                    frequency=FPS,
                )
            )
            self._wrist_res = (WW, WH)
        if a.obs_crop:
            v = [float(x) for x in a.obs_crop.split(",")]
            sx = CAM_W / 1280.0
            sy = CAM_H / 960.0
            self.crop = [int(round(v[0] * sx)), int(round(v[1] * sy)), int(round(v[2] * sx)), int(round(v[3] * sy))]
        self.placements["observation"] = dict(crop=self.crop, resolution=list(OBS_RES), wrist=a.wrist == "true")
        if not os.environ.get("SR_NO_CONTACT"):
            # contact reports on scene bodies only: the API on the robot's links breaks the rendering of its distal links
            n_cr = 0
            for pr in [
                q
                for q in stage.Traverse()
                if q.HasAPI(UsdPhysics.RigidBodyAPI) and not q.GetPath().pathString.startswith("/World/Franka")
            ]:
                PhysxSchema.PhysxContactReportAPI.Apply(pr).CreateThresholdAttr().Set(0.0)
                n_cr += 1
            self.note(contact_report_bodies=n_cr, robot_links_excluded=True)
        if os.environ.get("SR_CAM_INIT_EARLY"):
            cam.set_clipping_range(0.03, 20.0)
            cam.initialize()
        world.reset()
        fr.set_world_pose(position=BASE, orientation=np.array([1.0, 0, 0, 0]))
        fr.set_joint_positions(Q0)
        fr.set_joint_velocities(np.zeros(9))
        fr.apply_action(ArticulationAction(joint_positions=Q0))
        if os.environ.get("SR_LINK_VIEW"):
            from isaacsim.core.prims import RigidPrim as _RP

            self._linkview = _RP(prim_paths_expr="/World/Franka/panda_link[0-7]", name="franka_links_view")
            self.note(link_view_created=True)
        # camera: fixed workcell view from behind-right of the robot (appliance fronts visible) with small per-episode jitter
        tgt = np.array([float(v) for v in a.cam_target.split(",")])
        az = a.cam_az + self.rng.uniform(-2, 2)
        el = a.cam_el + self.rng.uniform(-1, 1)
        dist = a.cam_dist * (1 + self.rng.uniform(-0.01, 0.01))
        if not (os.environ.get("SR_NO_LATE_CAM_INIT") and os.environ.get("SR_CAM_INIT_EARLY")):
            cam.set_clipping_range(0.03, 20.0)
            cam.initialize()
        if self.wcam is not None:
            mx, my, mz, pitch = [float(v) for v in a.wrist_mount.split(",")]
            pr = math.radians(pitch)
            fwd = np.array([-math.sin(pr), 0.0, math.cos(pr)])
            up = np.array([1.0, 0.0, 0.0])
            up = up - fwd * float(up @ fwd)
            up /= np.linalg.norm(up)
            rgt = np.cross(fwd, up)
            Rc = np.stack([fwd, -rgt, up], 1)
            self.wcam.set_clipping_range(0.01, 10.0)
            self.wcam.initialize()
            self.wcam.set_local_pose(
                translation=np.array([mx, my, mz]), orientation=rot_utils.rot_matrices_to_quats(Rc), camera_axes="world"
            )
            wap = 2.0955
            self.wcam.set_horizontal_aperture(wap)
            self.wcam.set_focal_length(wap / (2 * math.tan(math.radians(a.wrist_fov) / 2)))
            self.placements["wrist_camera"] = dict(
                parent="/World/Franka/panda_hand",
                mount_m=[mx, my, mz],
                pitch_deg=pitch,
                resolution=list(self._wrist_res),
                horizontal_fov_deg=a.wrist_fov,
                axes="camera +X = hand +z tilted toward -x, image up = hand +x (fingers at the bottom of the frame)",
            )
        self._cam_pending = (az, el, dist, a.cam_fov, tgt)
        if not os.environ.get("SR_NO_PRE_PLAY_CAM"):
            self.set_camera(az, el, dist, a.cam_fov, tgt)
        self.contact_log = []
        self.contact_samples = {}
        self.sub = (
            None
            if os.environ.get("SR_NO_SUBSCRIBE")
            else get_physx_simulation_interface().subscribe_contact_report_events(self.on_contact)
        )
        if os.environ.get("SR_NO_SOLVER"):
            self.solver = None
        else:
            self.solver = KinematicsSolver(fr, end_effector_frame_name="right_gripper")
            self.solver.get_kinematics_solver().set_robot_base_pose(BASE, np.array([1.0, 0, 0, 0]))
        world.play()
        if not os.environ.get("SR_NO_RESUME"):
            cam.resume()
            if self.wcam is not None:
                self.wcam.resume()
        for i in range(30):
            world.step(render=False)
        warm = 0
        rgb = None
        for i in range(120):
            world.step(render=True)
            app.update()
            rgb = cam.get_rgb()
            warm += 1
            if rgb is not None and getattr(rgb, "size", 0) and float(np.asarray(rgb)[:, :, :3].mean()) > 2.0:
                break
        if rgb is None or not getattr(rgb, "size", 0) or float(np.asarray(rgb)[:, :, :3].mean()) <= 2.0:
            raise RuntimeError("camera_rgb_activation_failed_after_120_raw_isaac_warmup_frames")
        self.note(
            camera_warmup_renders=warm,
            rgb_mean=round(float(np.asarray(rgb)[:, :, :3].mean()), 1),
            camera_capture_type="raw_isaac_rgb",
        )
        if self.wcam is not None:
            wr = None
            ww = 0
            for i in range(60):
                wr = self.wcam.get_rgb()
                if wr is not None and getattr(wr, "size", 0) and float(np.asarray(wr)[:, :, :3].mean()) > 2.0:
                    break
                world.step(render=True)
                app.update()
                ww += 1
            if wr is None or not getattr(wr, "size", 0):
                raise RuntimeError("wrist_camera_rgb_activation_failed")
            self.note(
                wrist_camera_warmup_renders=ww,
                wrist_rgb_mean=round(float(np.asarray(wr)[:, :, :3].mean()), 1),
                wrist_camera=self.placements.get("wrist_camera"),
            )
        if os.environ.get("SR_LINK_DUMP"):
            self.link_dump("after_warmup")
        for k, info in self.assets.items():
            info["q0"] = self.joint_q(k)
            self.note(
                asset=k,
                cid=info["cid"],
                scale=info["scale"],
                platform=info["platform"],
                q_after_settle=info["q0"],
                initial_q_authored=info["initial_q_authored"],
                joints={
                    jn: dict(
                        type=j["type"],
                        axis=j["axis"].round(3).tolist(),
                        pivot=j["pivot"].round(3).tolist(),
                        limits=[j["lower"], j["upper"]],
                        body1=j["body1"],
                    )
                    for jn, j in info["joints"].items()
                },
                aabb=[v.round(3).tolist() for v in self.placements[k]["aabb"]],
            )
        for k, o in self.objects.items():
            lo, hi = bounds_of(o["meshes"])
            o["aabb0"] = (lo, hi)
            self.note(object=k, asset=o["asset"], aabb=[lo.round(3).tolist(), hi.round(3).tolist()])
        self.note(
            robot=dict(
                link0=self.world_pos("/World/Franka/panda_link0"), hand=self.world_pos("/World/Franka/panda_hand")
            ),
            camera=self.placements["camera"],
            transit_z=round(self.transit_z, 3),
        )

    def place(self, key, usd, x, y, yaw, z_extra=0.003, front_x=False, z_abs=None, existing=None):
        prim = existing
        if prim is None:
            prim = self.stage.DefinePrim("/World/Scene/" + key, "Xform")
            prim.GetReferences().AddReference(usd, "/World/Asset")
            wait_load()
        xf = UsdGeom.XformCommonAPI(prim)
        xf.SetTranslate(Gf.Vec3d(0, 0, 0))
        xf.SetRotate(Gf.Vec3f(0, 0, float(yaw)))
        app.update()
        lo, hi = bounds_of(collision_meshes(prim))
        cx = (x + (hi[0] - lo[0]) / 2) if front_x else x
        z = (TABLE + z_extra) if z_abs is None else z_abs
        xf.SetTranslate(Gf.Vec3d(cx - (lo[0] + hi[0]) / 2, y - (lo[1] + hi[1]) / 2, z - lo[2]))
        app.update()
        lo, hi = bounds_of(collision_meshes(prim))
        self.placements[key] = dict(
            usd=usd,
            usd_sha256=sha(usd),
            x=float(cx),
            y=float(y),
            yaw=float(yaw),
            aabb=(lo, hi),
            center=((lo + hi) / 2).tolist(),
        )
        return prim

    def add_finger_pads(self):
        """flat parallel pad boxes on both fingertips (the imported hull has a tip chamfer that ejects thin bars)."""
        out = {}
        for name in ("panda_leftfinger", "panda_rightfinger"):
            link = self.stage.GetPrimAtPath("/World/Franka/" + name)
            hand = self.stage.GetPrimAtPath("/World/Franka/panda_hand")
            if not link.IsValid():
                continue
            Ml = omni.usd.get_world_transform_matrix(link)
            Mli = Ml.GetInverse()
            pts = []
            for q in Usd.PrimRange(link, Usd.TraverseInstanceProxies()):
                if (
                    q.IsA(UsdGeom.Mesh)
                    and q.HasAPI(UsdPhysics.CollisionAPI)
                    or (q.IsA(UsdGeom.Mesh) and "collisions" in str(q.GetPath()))
                ):
                    P = UsdGeom.Mesh(q).GetPointsAttr().Get()
                    if not P:
                        continue
                    Mm = omni.usd.get_world_transform_matrix(q) * Mli
                    pts += [Mm.Transform(Gf.Vec3d(*pp)) for pp in P]
            if not pts:
                continue
            A = np.array([[v[0], v[1], v[2]] for v in pts])
            lo = A.min(0)
            hi = A.max(0)
            hand_c = np.array(Mli.Transform(omni.usd.get_world_transform_matrix(hand).ExtractTranslation()))
            inward = (
                1.0 if abs(lo[1] - 0.0) < abs(hi[1] - 0.0) else -1.0
            )  # pad face is the y-face at ~0; the body extends away from it
            if abs(lo[1]) > abs(hi[1]):
                inward = -1.0
            face = lo[1] if inward > 0 else hi[1]
            zt = hi[2] if abs(hi[2] - hand_c[2]) > abs(lo[2] - hand_c[2]) else lo[2]
            zs = 1.0 if zt == hi[2] else -1.0
            cube = UsdGeom.Cube.Define(self.stage, link.GetPath().AppendChild("pad"))
            cube.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(cube.GetPrim())
            xf.ClearXformOpOrder()
            cx = float((lo[0] + hi[0]) / 2)
            cy = float(face + inward * 0.002)
            cz = float(zt - zs * 0.014)
            xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, cz))
            xf.AddScaleOp().Set(Gf.Vec3f(0.018, 0.004, 0.026))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            UsdGeom.Imageable(cube.GetPrim()).CreatePurposeAttr().Set("guide")
            out[name] = dict(
                local_aabb=[lo.round(4).tolist(), hi.round(4).tolist()], pad_center=[cx, cy, cz], inward=inward
            )
        self.note(finger_pads=out)
        return out

    def set_material(self, prim, mu):
        n = 0
        for q in Usd.PrimRange(prim):
            if q.HasAPI(UsdPhysics.MaterialAPI):
                m = UsdPhysics.MaterialAPI(q)
                m.CreateStaticFrictionAttr().Set(float(mu))
                m.CreateDynamicFrictionAttr().Set(float(mu))
                m.CreateRestitutionAttr().Set(0.0)
                PhysxSchema.PhysxMaterialAPI.Apply(q).CreateFrictionCombineModeAttr().Set("max")
                n += 1
        return n

    def body_aabb(self, asset_key):
        info = self.assets[asset_key]
        children = {j["body1"] for j in info["joints"].values()}
        ms = [m for ln, pr in info["links"].items() if ln not in children for m in collision_meshes(pr)]
        return bounds_of(ms) if ms else self.placements[asset_key]["aabb"]

    def primary_joint(self, asset_key):
        for step in self.spec["plan"]:
            if len(step) > 2 and step[1] == asset_key and step[0] != "pick":
                return step[2]
        return next(iter(self.assets[asset_key]["joints"]), None)

    def joint_frames(self, root):
        r = {}
        for j in Usd.PrimRange(root):
            if not (j.IsA(UsdPhysics.RevoluteJoint) or j.IsA(UsdPhysics.PrismaticJoint)):
                continue
            J = UsdPhysics.Joint(j)
            b0 = self.stage.GetPrimAtPath(J.GetBody0Rel().GetTargets()[0])
            M0 = omni.usd.get_world_transform_matrix(b0)
            T = Gf.Matrix4d(1.0)
            q = J.GetLocalRot0Attr().Get()
            T.SetRotate(Gf.Quatd(float(q.GetReal()), Gf.Vec3d(q.GetImaginary())))
            T.SetTranslateOnly(Gf.Vec3d(J.GetLocalPos0Attr().Get()))
            F0 = T * M0
            tok = str(j.GetAttribute("physics:axis").Get() or "X")
            v = Gf.Vec3d(1 if tok == "X" else 0, 1 if tok == "Y" else 0, 1 if tok == "Z" else 0)
            ax = np.array(F0.TransformDir(v))
            ax /= max(np.linalg.norm(ax), 1e-9)
            typ = "prismatic" if j.IsA(UsdPhysics.PrismaticJoint) else "revolute"
            lo = float(j.GetAttribute("physics:lowerLimit").Get())
            hi = float(j.GetAttribute("physics:upperLimit").Get())
            r[j.GetName()] = dict(
                pivot=np.array(F0.ExtractTranslation()),
                axis=ax,
                body1=str(J.GetBody1Rel().GetTargets()[0]).split("/")[-1],
                type=typ,
                lower=lo,
                upper=hi,
                prim=j,
            )
        return r

    def inside_point(self, asset_key, joint):
        info = self.assets[asset_key]
        j = info["joints"][joint]
        llo, lhi = info["link_aabb_closed"][j["body1"]]
        if j["type"] == "prismatic":
            d = float(info["initial_q_authored"].get(joint, 0.0))
            ax = j["axis"].copy()
            if ax[0] > 0:
                ax = -ax
            shift = ax * d
            return float(llo[0] + shift[0] + 0.06), float((llo[1] + lhi[1]) / 2 + shift[1]), float(llo[2] + 0.012)
        lo, hi = self.placements[asset_key]["aabb"]
        return float(lo[0] + 0.16), float((llo[1] + lhi[1]) / 2), float(llo[2] + 0.01)

    # ---------------- state ----------------
    def world_pos(self, path):
        pr = self.stage.GetPrimAtPath(path)
        return [round(float(v), 4) for v in omni.usd.get_world_transform_matrix(pr).ExtractTranslation()]

    def joint_q(self, key):
        info = self.assets[key]
        q = np.asarray(info["view"].get_joint_positions()).reshape(-1)
        names = list(info["view"].dof_names)
        return {n: float(q[i]) for i, n in enumerate(names)}

    def on_contact(self, headers, data):
        for h in headers:
            p0 = str(PhysicsSchemaTools.intToSdfPath(h.actor0))
            p1 = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
            self.contact_log.append((self.tick, p0, p1, int(h.num_contact_data)))
            k = "|".join(
                sorted(
                    [
                        p0.split("/")[2] if p0.startswith("/World/") else p0,
                        p1.split("/")[2] if p1.startswith("/World/") else p1,
                    ]
                )
            )
            lst = self.contact_samples.setdefault(k, [])
            if len(lst) < 2 and (p0, p1) not in lst:
                lst.append((p0, p1))

    def contact_summary(self, since):
        rows = [c for c in self.contact_log if c[0] >= since]
        out = {}

        def cls(pth):
            if pth.startswith("/World/Franka"):
                return "robot"
            if pth.startswith("/World/Scene/"):
                return pth.split("/")[3]
            return pth.split("/")[-1]

        for t, p0, p1, n in rows:
            k = "|".join(sorted([cls(p0), cls(p1)]))
            out[k] = out.get(k, 0) + 1
        return out

    def touching(self, rec, asset_key):
        return any(set(k.split("|")) == {asset_key, "robot"} for k in rec["contact_pairs"])

    def proprio(self):
        return np.asarray(self.fr.get_joint_positions(), float)

    def ee(self):
        pos, rot = self.solver.compute_end_effector_pose()
        return np.asarray(pos, float), np.asarray(rot, float)

    def ee_R(self):
        rot = self.ee()[1]
        return rot_utils.quats_to_rot_matrices(rot) if rot.shape == (4,) else rot.reshape(3, 3)

    def snapshot(self):
        objs = {}
        for k, o in self.objects.items():
            lo, hi = bounds_of(o["meshes"])
            objs[k] = dict(center=((lo + hi) / 2).round(4).tolist(), aabb=[lo.round(4).tolist(), hi.round(4).tolist()])
        joints = {k: self.joint_q(k) for k in self.assets}
        links = {}
        for k, info in self.assets.items():
            links[k] = {}
            for jn, j in info["joints"].items():
                link = info["links"].get(j["body1"])
                if link is not None:
                    lo, hi = bounds_of(collision_meshes(link))
                    links[k][jn] = dict(far_edge_z=float(hi[2]), aabb=[lo.round(4).tolist(), hi.round(4).tolist()])
        sup = {}
        for k, o in objs.items():
            c = o["center"]
            b = o["aabb"][0][2]
            try:
                hit = get_physx_scene_query_interface().raycast_closest(
                    carb.Float3(float(c[0]), float(c[1]), float(b) + 0.004), carb.Float3(0, 0, -1), 0.5
                )
                pth = str(hit.get("rigidBody", "")) if hit.get("hit") else ""
                if "/World/Scene/" + k in pth:
                    hit = get_physx_scene_query_interface().raycast_closest(
                        carb.Float3(float(c[0]), float(c[1]), float(b) - 0.004), carb.Float3(0, 0, -1), 0.5
                    )
                    pth = str(hit.get("rigidBody", "")) if hit.get("hit") else ""
                sup[k] = dict(
                    body=pth.split("/")[3] if pth.startswith("/World/Scene/") else pth.split("/")[-1],
                    link=pth.split("/")[-1],
                    z=round(float(hit["position"][2]), 4) if hit.get("hit") else None,
                )
            except Exception as ex:
                sup[k] = dict(body="", link="", z=None, error=str(ex)[:60])
        cs = self.contact_summary(self.tick)
        contacts = {k: sum(v for kk, v in cs.items() if set(kk.split("|")) == {k, "robot"}) for k in self.objects}
        support = {k: sum(v for kk, v in cs.items() if k in kk.split("|") and "robot" not in kk) for k in self.objects}
        partners = {
            k: sorted({x for kk in cs for x in kk.split("|") if k in kk.split("|") and x != k and x != "robot"})
            for k in self.objects
        }
        return dict(
            objects=objs,
            joints=joints,
            links=links,
            contacts=contacts,
            support=support,
            support_partners=partners,
            support_hit=sup,
            contact_pairs=cs,
        )

    # ---------------- control ----------------
    def link_dump(self, tag):
        """USD world position, Fabric (usdrt) world position and PhysX rigid-body pose of every robot link."""
        names = [
            "panda_link0",
            "panda_link1",
            "panda_link2",
            "panda_link3",
            "panda_link4",
            "panda_link5",
            "panda_link6",
            "panda_link7",
            "panda_link8",
            "panda_hand",
            "panda_leftfinger",
            "panda_rightfinger",
        ]
        rows = {}
        rt = None
        try:
            import usdrt

            rt = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
        except Exception as ex:
            rows["_usdrt"] = str(ex)[:80]
        physx = {}
        try:
            from isaacsim.core.prims import RigidPrim

            v = RigidPrim(prim_paths_expr="/World/Franka/panda_link[0-7]", name="diag_links_" + tag)
            pos, _ = v.get_world_poses()
            physx = {f"panda_link{i}": [round(float(x), 3) for x in pos[i]] for i in range(len(pos))}
        except Exception as ex:
            rows["_physx"] = str(ex)[:120]
        for n in names:
            p = self.stage.GetPrimAtPath("/World/Franka/" + n)
            if not p.IsValid():
                continue
            M = omni.usd.get_world_transform_matrix(p)
            usd = [round(float(M[3][j]), 3) for j in range(3)]
            fab = None
            if rt is not None:
                try:
                    rp = rt.GetPrimAtPath("/World/Franka/" + n)
                    for attr in ("_worldPosition", "omni:fabric:worldMatrix"):
                        a = rp.GetAttribute(attr)
                        if a and a.IsValid():
                            val = a.Get()
                            fab = [
                                round(float(x), 3)
                                for x in (val if attr == "_worldPosition" else [val[3][0], val[3][1], val[3][2]])
                            ]
                            break
                    if fab is None:
                        xf = usdrt.Rt.Xformable(rp)
                        if hasattr(xf, "GetWorldPositionAttr"):
                            a = xf.GetWorldPositionAttr()
                            fab = [round(float(x), 3) for x in a.Get()] if a.IsValid() else "no_world_pos_attr"
                        else:
                            fab = "no_api"
                except Exception as ex:
                    fab = "err:" + str(ex)[:60]
            rows[n] = dict(usd=usd, fabric=fab, physx=physx.get(n))
            if os.environ.get("SR_MESH_DUMP"):
                vis = p.GetChild("visuals")
                rows[n]["children_instanceable"] = {c.GetName(): c.IsInstanceable() for c in p.GetChildren()}
                if vis.IsValid():
                    Mv = omni.usd.get_world_transform_matrix(vis)
                    rows[n]["visuals_usd"] = [round(float(Mv[3][j]), 3) for j in range(3)]
                    for q in Usd.PrimRange(vis, Usd.TraverseInstanceProxies()):
                        if q.IsA(UsdGeom.Mesh):
                            Mm = omni.usd.get_world_transform_matrix(q)
                            P = UsdGeom.Mesh(q).GetPointsAttr().Get()
                            c = None
                            if P:
                                A = np.array([[v[0], v[1], v[2]] for v in P[:: max(1, len(P) // 500)]])
                                A = A @ np.array(
                                    [
                                        [Mm[0][0], Mm[0][1], Mm[0][2]],
                                        [Mm[1][0], Mm[1][1], Mm[1][2]],
                                        [Mm[2][0], Mm[2][1], Mm[2][2]],
                                    ]
                                ) + np.array([Mm[3][0], Mm[3][1], Mm[3][2]])
                                c = A.mean(0).round(3).tolist()
                            rows[n]["mesh_usd_center"] = c
                            rows[n]["mesh_proxy"] = q.IsInstanceProxy()
                            break
                    if rt is not None:
                        try:
                            rp = rt.GetPrimAtPath(vis.GetPath().pathString)
                            a = rp.GetAttribute("_worldPosition")
                            rows[n]["visuals_fabric"] = (
                                [round(float(x), 3) for x in a.Get()] if a and a.IsValid() else "no_attr"
                            )
                            a2 = rp.GetAttribute("omni:fabric:worldMatrix")
                            rows[n]["visuals_fabric_matrix"] = (
                                [round(float(a2.Get()[3][j]), 3) for j in range(3)]
                                if a2 and a2.IsValid()
                                else "no_attr"
                            )
                        except Exception as ex:
                            rows[n]["visuals_fabric"] = "err:" + str(ex)[:60]
        self.note(link_dump=tag, links=rows)

    def set_camera(self, az, el, dist, fov_deg, tgt):
        """place the camera on a sphere around tgt (azimuth/elevation in degrees, distance in m) with the requested horizontal FOV."""
        cam = self.cam
        azr = math.radians(az)
        elr = math.radians(el)
        tgt = np.asarray(tgt, float)
        cp = tgt + dist * np.array([math.cos(elr) * math.cos(azr), math.cos(elr) * math.sin(azr), math.sin(elr)])
        cam.set_world_pose(position=cp, orientation=look_at(cp, tgt), camera_axes="world")
        aperture = 2.0955
        cam.set_horizontal_aperture(aperture)
        cam.set_focal_length(aperture / (2 * math.tan(math.radians(fov_deg) / 2)))
        self.placements["camera"] = dict(
            position=cp.round(4).tolist(),
            target=tgt.round(4).tolist(),
            azimuth_deg=round(az, 2),
            elevation_deg=round(el, 2),
            distance_m=round(dist, 4),
            horizontal_fov_deg=round(math.degrees(cam.get_horizontal_fov()), 2),
            resolution=[CAM_W, CAM_H],
            observation_resolution=list(OBS_RES),
            focal_length_stage_units=round(cam.get_focal_length(), 4),
        )
        return cp

    def observe(self, rgb):
        """full-resolution frame (uint8 HxWx3) and the 320x240 policy observation: the workspace crop (--obs-crop) of the frame, resized."""
        im = np.asarray(rgb)[:, :, :3].astype(np.uint8)
        src = im[self.crop[1] : self.crop[3], self.crop[0] : self.crop[2]] if self.crop else im
        obs = (
            src
            if (src.shape[1], src.shape[0]) == OBS_RES
            else np.asarray(Image.fromarray(np.ascontiguousarray(src)).resize(OBS_RES, Image.LANCZOS))
        )
        return im, obs

    def observe_wrist(self, rgb):
        im = np.asarray(rgb)[:, :, :3].astype(np.uint8)
        return (
            im
            if (im.shape[1], im.shape[0]) == OBS_RES
            else np.asarray(Image.fromarray(im).resize(OBS_RES, Image.LANCZOS))
        )

    def grab(self):
        """current third-person render (retries the annotator) and, when mounted, the wrist render."""
        rgb = self.cam.get_rgb()
        for _ in range(12):
            if rgb is not None and getattr(rgb, "size", 0):
                break
            self.world.render()
            app.update()
            rgb = self.cam.get_rgb()
        if rgb is None or not getattr(rgb, "size", 0):
            for _ in range(60):
                self.world.step(render=True)
                app.update()
                rgb = self.cam.get_rgb()
                if rgb is not None and getattr(rgb, "size", 0) and float(np.asarray(rgb)[:, :, :3].mean()) > 2.0:
                    break
        if rgb is None or not getattr(rgb, "size", 0) or float(np.asarray(rgb)[:, :, :3].mean()) <= 2.0:
            raise RuntimeError("missing_rgb_after_raw_isaac_retry")
        wr = None
        if self.wcam is not None:
            wr = self.wcam.get_rgb()
            for _ in range(12):
                if wr is not None and getattr(wr, "size", 0):
                    break
                self.world.render()
                app.update()
                wr = self.wcam.get_rgb()
            if wr is None or not getattr(wr, "size", 0):
                raise RuntimeError("missing_wrist_rgb")
        return rgb, wr

    def dagger_execute(self, t, q, obs, wobs, phase):
        """dagger mode: in policy bursts the served policy's command is executed while the teacher's clamped target `t` stays the label."""
        d = self.dagger
        teacher_only = phase.endswith("_close") or any(
            s in phase for s in ("release", "arc", "lift", "grasp", "press", "pull", "final_hold")
        )
        if d["burst_left"] > 0 and teacher_only:
            d["burst_left"] = 0
        if d["burst_left"] <= 0:
            if d["gap_left"] > 0 or teacher_only:
                if d["gap_left"] > 0:
                    d["gap_left"] -= 1
                d["n_teacher"] += 1
                return t, dict(burst=False)
            d["burst_left"] = self.rng.randint(d["lmin"], d["lmax"])
            d["chunk"] = None
            d["ci"] = 0
            d["bursts"] += 1
            mean_gap = (d["lmin"] + d["lmax"]) / 2 * (1 - d["beta"]) / max(d["beta"], 1e-3)
            d["gap_left"] = int(self.rng.expovariate(1 / mean_gap)) if mean_gap > 0 else 0
        if d["chunk"] is None or d["ci"] >= 4:
            d["chunk"], resp = d["client"].query(obs, wobs, q, self.spec["instruction"])
            d["ci"] = 0
            d["queries"] += 1
        raw = d["chunk"][d["ci"]]
        ex = np.clip(raw, JOINT_LO, JOINT_HI)
        ex[:7] = limit_target(q[:7], ex[:7])
        ex[7:] = t[7:]
        d["ci"] += 1
        d["burst_left"] -= 1
        d["n_policy"] += 1  # arm from the policy, fingers from the teacher (no drops during bursts)
        return ex, dict(burst=True, chunk_index=d["ci"] - 1, raw=np.asarray(raw).round(5).tolist())

    def camera_probe(self, specs):
        """render each camera spec from the initial state and from the opened state (plan targets applied, arm raised over the furniture)."""

        def render(tag):
            for _ in range(3):
                self.world.step(render=True)
                app.update()
            rgb = self.cam.get_rgb()
            im, _ = self.observe(rgb)
            Image.fromarray(im).save(self.dir / f"{tag}.png")
            return im

        parsed = []
        for k, spec in enumerate([x for x in specs.split(";") if x.strip()]):
            v = [float(x) for x in spec.split(",")]
            tgt = np.array(v[4:7]) if len(v) >= 7 else np.array([float(x) for x in a.cam_target.split(",")])
            parsed.append((k, v[0], v[1], v[2], v[3], tgt))
        for k, az, el, dist, fov, tgt in parsed:
            self.set_camera(az, el, dist, fov, tgt)
            render(f"cam{k:02d}_initial")
            self.note(camera_probe=k, state="initial", camera=self.placements["camera"])
        # opened state: plan targets on the articulated joints, arm above the tallest furniture front
        for step in self.spec["plan"]:
            if step[0] in ("open_edge", "lift_lid", "open_prismatic") and step[1] in self.assets:
                info = self.assets[step[1]]
                names = list(info["view"].dof_names)
                q = np.asarray(info["view"].get_joint_positions()).reshape(-1).copy()
                if step[2] in names:
                    q[names.index(step[2])] = step[3] if step[0] == "open_prismatic" else math.radians(step[3])
                    info["view"].set_joint_positions(q)
                    info["view"].set_joint_velocities(np.zeros_like(q))
        for _ in range(15):
            self.world.step(render=False)
        tops = [self.placements[k]["aabb"][1][2] for k in self.assets]
        front = min([self.placements[k]["aabb"][0][0] for k in self.assets] or [-0.2])
        ys = [self.placements[k]["center"][1] for k in self.assets] or [0.0]
        try:
            self.goto(
                [front - 0.05, float(np.mean(ys)), max(tops or [TABLE + 0.3]) + 0.12], DOWN(), OPEN, 25, "probe_arm"
            )
        except Exception as ex:
            self.note(probe_arm_failed=str(ex)[:120])
        if os.environ.get("SR_LINK_DUMP"):
            self.link_dump("after_probe_arm")
        for k, az, el, dist, fov, tgt in parsed:
            self.set_camera(az, el, dist, fov, tgt)
            render(f"cam{k:02d}_opened")
            self.note(camera_probe=k, state="opened", camera=self.placements["camera"])

    def step_tick(self, target9, phase):
        if getattr(self, "_cam_pending", None) is not None and os.environ.get("SR_NO_PRE_PLAY_CAM"):
            self.set_camera(*self._cam_pending)
            self._cam_pending = None
        q = self.proprio()
        rgb, wr = self.grab()
        full, im = self.observe(rgb)
        path = self.dir / "frames" / f"frame_{self.tick:05d}.png"
        wim = None
        wpath = None
        if a.keep_frames == "true":
            Image.fromarray(im).save(path)
        if wr is not None:
            wim = self.observe_wrist(wr)
            wpath = self.dir / "frames_wrist" / f"frame_{self.tick:05d}.png"
            if a.keep_frames == "true":
                Image.fromarray(wim).save(wpath)
            pip = (
                wim
                if (wim.shape[1], wim.shape[0]) == (320, 240)
                else np.asarray(Image.fromarray(wim).resize((320, 240)))
            )
            H, W = full.shape[:2]
            ph, pw = pip.shape[:2]
            if H > ph + 8 and W > pw + 8:
                full = full.copy()
                y0 = H - ph - 6
                x0 = W - pw - 6
                full[y0 - 2 : y0 + ph + 2, x0 - 2 : x0 + pw + 2] = 255
                full[y0 : y0 + ph, x0 : x0 + pw] = pip
        self.last_obs = im
        self.last_wobs = wim
        if self.first_frame is None:
            self.first_frame = full
        self.last_frame = full
        if a.video == "true" and not a.camera_probe:
            if self.writer is None:
                try:
                    import imageio.v2 as imageio

                    self.writer = imageio.get_writer(
                        str(self.dir / "review.mp4"),
                        fps=FPS,
                        codec="libx264",
                        quality=6,
                        macro_block_size=None,
                        ffmpeg_log_level="error",
                    )
                except Exception as ex:
                    print("video_writer_failed", ex, flush=True)
                    self.writer = False
            if self.writer:
                self.writer.append_data(full)
        t = np.clip(np.asarray(target9, float), JOINT_LO, JOINT_HI)
        t[:7] = q[:7] + np.clip(t[:7] - q[:7], -VMAX_TICK, VMAX_TICK)
        ex = t
        dg = None
        if self.dagger is not None:
            ex, dg = self.dagger_execute(t, q, im, wim, phase)
        self.fr.apply_action(ArticulationAction(joint_positions=ex[:7], joint_indices=np.arange(7)))
        self.fr.gripper.apply_action(ArticulationAction(joint_positions=ex[7:]))
        for k in range(SUB):
            self.world.step(render=(k == SUB - 1))
        snap = self.snapshot()
        qa = self.proprio().round(6).tolist()
        rec = dict(
            tick=self.tick,
            timestamp_s=self.tick / FPS,
            phase=phase,
            observation=dict(
                path=str(path), path_wrist=(str(wpath) if wpath is not None else None), proprio=q.round(6).tolist()
            ),
            action=t.round(6).tolist(),
            **({"executed": ex.round(6).tolist(), "dagger": dg} if dg is not None else {}),
            ee=self.ee()[0].round(4).tolist(),
            proprio=qa,
            **snap,
        )
        self.trace.append(rec)
        self.tick += 1
        if self.tick >= a.budget_ticks:
            raise TimeoutError("budget_ticks")
        return rec

    def solve_ik(self, target, R, max_jump=1.2):
        """Lula IK with warm start; solutions whose joints 1-6 jump more than max_jump rad from the current state are
        rejected (branch flips); joint 7 may wrap. Returns (q7, tolerance) or (None, None)."""
        q = self.proprio()[:7]
        best = None
        R = np.asarray(R, float)
        for Rm in (R, R @ np.diag([-1.0, -1.0, 1.0])):
            qq = rot_utils.rot_matrices_to_quats(Rm)
            for pt, ot in ((0.002, 0.04), (0.004, 0.08), (0.008, 0.12)):
                act, ok = self.solver.compute_inverse_kinematics(
                    target_position=np.asarray(target, float),
                    target_orientation=qq,
                    position_tolerance=pt,
                    orientation_tolerance=ot,
                )
                if ok and act is not None:
                    qi = np.asarray(act.joint_positions)[:7]
                    jump = float(np.max(np.abs(qi[:6] - q[:6])))
                    if jump <= max_jump:
                        return qi, ot
                    if best is None:
                        best = (qi, ot, jump)
        if best is not None:
            self.ik_jumps = getattr(self, "ik_jumps", 0) + 1
        return None, None

    def goto(
        self, target, R, finger, ticks, phase, max_delta=0.04, stop_on_contact=None, cartesian=False, converge=0.0
    ):
        """move the right_gripper frame to target with orientation R: joint-space interpolation to the IK solution
        (default) or straight-line Cartesian interpolation with the orientation held. converge>0 keeps servoing
        after the interpolation until the EE error is below converge (max 8 extra ticks)."""
        target = np.asarray(target, float)
        q0 = self.proprio()[:7]
        qs, tol = self.solve_ik(target, R, max_jump=9.0)
        if qs is None:
            self.note(ik_unreachable=phase, target=target.round(3).tolist())
            qs = q0
        ee0, _ = self.ee()
        n = max(
            ticks,
            int(math.ceil(float(np.max(np.abs(qs - q0))) / max_delta)),
            int(math.ceil(float(np.linalg.norm(target - ee0)) / 0.025)) if cartesian else 0,
        )
        stopped = False
        md = min(max_delta, 0.04) if cartesian else min(max_delta, 0.04)
        R = np.asarray(R, float)
        R0 = self.ee_R()
        Rgoal = R
        if cartesian and float(np.trace(R0.T @ (R @ np.diag([-1.0, -1.0, 1.0])))) > float(np.trace(R0.T @ R)) + 1e-6:
            Rgoal = R @ np.diag([-1.0, -1.0, 1.0])
        for i in range(n):
            s = (i + 1) / n
            s = s * s * (3 - 2 * s)
            if cartesian:
                tgt = ee0 + (target - ee0) * s
                qi, _ = self.solve_ik(tgt, slerp_R(R0, Rgoal, s))
                q = self.proprio()[:7]
                arm = q + np.clip(qi - q, -md, md) if qi is not None else q
            else:
                arm = q0 + (qs - q0) * s
            rec = self.step_tick(np.r_[arm, finger, finger], phase)
            if stop_on_contact and any(set(k.split("|")) == set(stop_on_contact) for k in rec["contact_pairs"]):
                stopped = True
                break
        extra = 0
        if cartesian and not stopped and float(np.linalg.norm(self.ee()[0] - target)) > 0.04 and qs is not q0:
            q1 = self.proprio()[:7]
            m = max(6, int(math.ceil(float(np.max(np.abs(qs - q1))) / 0.04)))
            self.note(
                cartesian_fallback=phase,
                err=round(float(np.linalg.norm(self.ee()[0] - target)), 3),
                ticks=m,
                jump=round(float(np.max(np.abs(qs - q1))), 3),
            )
            for i in range(m):
                s2 = (i + 1) / m
                s2 = s2 * s2 * (3 - 2 * s2)
                rec = self.step_tick(np.r_[q1 + (qs - q1) * s2, finger, finger], phase + "_js")
                if stop_on_contact and any(set(k.split("|")) == set(stop_on_contact) for k in rec["contact_pairs"]):
                    stopped = True
                    break
        if stopped:
            converge = 0.0
        while converge > 0 and not stopped and extra < 8:
            ee, _ = self.ee()
            if float(np.linalg.norm(ee - target)) < converge:
                break
            qi, _ = self.solve_ik(target, R)
            q = self.proprio()[:7]
            arm = q + np.clip(qi - q, -0.04, 0.04) if qi is not None else q
            self.step_tick(np.r_[arm, finger, finger], phase + "_converge")
            extra += 1
        ee, _ = self.ee()
        err = float(np.linalg.norm(ee - target))
        self.note(
            phase=phase,
            target=target.round(3).tolist(),
            reached=ee.round(3).tolist(),
            err=round(err, 4),
            ticks=n + extra,
            ik_tol=tol,
            stopped=stopped,
            fingers=self.proprio()[7:].round(4).tolist(),
        )
        return err

    def hold(self, ticks, phase, finger=None):
        for i in range(ticks):
            q = self.proprio()
            f = q[7:] if finger is None else np.array([finger, finger])
            self.step_tick(np.r_[q[:7], f], phase)

    def transit(self, target, R, finger, phase, z=None):
        ee, _ = self.ee()
        z = max(self.transit_z if z is None else z, ee[2])
        self.goto([ee[0], ee[1], z], R, finger, 8, phase + "_up")
        self.goto([target[0], target[1], z], R, finger, 16, phase + "_over", cartesian=True)
        return self.goto(target, R, finger, 12, phase + "_down")

    def front_approach(self, pre, R, finger, phase, front_x):
        """reach a pre-grasp pose in front of a furniture face without passing over the furniture: retract to the
        approach plane x=front_x-0.16 at the current height, move within that plane, then advance to pre."""
        ee, _ = self.ee()
        xa = min(front_x - 0.16, float(pre[0]))
        z_mid = max(ee[2], float(pre[2]), TABLE + 0.16)
        if ee[0] > xa + 0.01:
            self.goto([xa, ee[1], max(ee[2], TABLE + 0.16)], R, finger, 8, phase + "_retract", cartesian=True)
        ee, _ = self.ee()
        self.goto([xa, ee[1], z_mid], R, finger, 6, phase + "_lift")
        self.goto([xa, pre[1], pre[2]], R, finger, 14, phase + "_align")
        return self.goto(pre, R, finger, 8, phase + "_advance", cartesian=True, converge=0.006)

    def arc(
        self,
        ax,
        piv,
        R,
        deg,
        phase,
        q_read,
        expect_sign=1,
        finger=CLOSED,
        asset_key=None,
        rotate_hand=True,
        step_deg=1.5,
        gate=0.02,
    ):
        """rotate the current EE pose about (piv,ax) by deg with closed-loop pacing: the commanded angle advances only
        while the EE tracks its target within `gate`. Flips the axis once if the joint moves the wrong way."""
        g0, _ = self.ee()
        Rk = np.eye(3)
        q_start = q_read()
        flipped = False
        fails = 0
        errs = []
        lost = 0
        th = 0.0
        total = math.radians(deg)
        sgn = 1.0 if deg >= 0 else -1.0
        dth = math.radians(step_deg)
        k = 0
        max_ticks = int(abs(deg) / step_deg * 3) + 10
        q_prev = q_start
        while abs(th) < abs(total) and k < max_ticks:
            k += 1
            Rk = rot_axis(ax, th)
            tgt = piv + Rk @ (g0 - piv)
            ee, _ = self.ee()
            err = float(np.linalg.norm(ee - tgt))
            q_cur = q_read()
            if err < gate or (asset_key is not None and abs(q_cur - q_prev) > 0.004):
                th = sgn * min(abs(th) + dth, abs(total))
                Rk = rot_axis(ax, th)
                tgt = piv + Rk @ (g0 - piv)
            q_prev = q_cur
            qi, _ = self.solve_ik(tgt, (Rk @ R) if rotate_hand else R)
            q = self.proprio()
            arm = q[:7] + np.clip(qi - q[:7], -0.06, 0.06) if qi is not None else q[:7]
            if qi is None:
                fails += 1
            rec = self.step_tick(np.r_[arm, finger, finger], phase)
            if k % 6 == 0:
                errs.append(round(err, 3))
            if finger == CLOSED and asset_key is not None:
                lost = 0 if self.touching(rec, asset_key) else lost + 1
                if lost >= 8:
                    self.note(arc_lost_grip=phase, k=k, th_deg=round(math.degrees(th), 1))
                    break
            if k == 4 and not flipped and expect_sign:
                dq = (q_read() - q_start) * expect_sign * sgn
                if dq < -0.01:
                    ax = -ax
                    flipped = True
                    self.note(arc_axis_flipped=phase, dq=round(float(dq), 4))
        self.note(
            arc=phase,
            ticks=k,
            reached_deg=round(math.degrees(th), 1),
            ik_fail=fails,
            track_err=errs,
            q_start=round(float(q_start), 3),
            q_end=round(float(q_read()), 3),
        )
        return Rk, q_start, q_read()

    # ---------------- teacher primitives ----------------
    def grasp_geometry(self, obj_key, mode, axis_hint=None):
        """top grasp on the object's upper body: centre = median-filtered centre of the top band (handles and spouts
        excluded as radial outliers), finger axis = narrowest body direction that is perpendicular to any handle."""
        o = self.objects[obj_key]
        lo, hi = bounds_of(o["meshes"])
        ext = hi - lo
        P = mesh_points(o["meshes"])
        top = P[P[:, 2] >= hi[2] - max(0.03, 0.35 * ext[2])]
        if len(top) < 8:
            top = P
        c = np.median(top[:, :2], 0)
        r = np.linalg.norm(top[:, :2] - c, axis=1)
        rm = np.median(r)
        body = top[r <= max(1.3 * rm, rm + 0.008)]
        outl = top[r > max(1.3 * rm, rm + 0.008)]
        c2 = body[:, :2].mean(0)
        handle = None
        if len(outl) >= 4 and len(outl) >= 0.03 * len(top):
            hd = outl[:, :2].mean(0) - c2
            n = np.linalg.norm(hd)
            if n > 0.015:
                handle = hd / n
        Q = body[:, :2] - c2
        w, v = np.linalg.eigh(Q.T @ Q / max(len(Q), 1))
        minor = v[:, 0]
        major = v[:, 1]
        wid = lambda ax: float(np.ptp(body[:, :2] @ ax))
        cands = [minor, major] if handle is None else [np.array([-handle[1], handle[0]])]
        if axis_hint == "y":
            cands = [np.array([0, 1.0])] + cands
        elif axis_hint == "x":
            cands = [np.array([1.0, 0])] + cands
        finger2 = None
        for ax in cands:
            if wid(ax) <= 0.074 and (handle is None or abs(float(ax @ handle)) < 0.5):
                finger2 = ax
                break
        if finger2 is None:
            finger2 = min(cands, key=wid)
        width = wid(finger2)
        length = wid(np.array([-finger2[1], finger2[0]]))
        finger = np.array([finger2[0], finger2[1], 0.0])
        finger /= np.linalg.norm(finger)
        f2 = finger[:2] / np.linalg.norm(finger[:2])
        p2 = np.array([-f2[1], f2[0]])
        pf = body[:, :2] @ f2
        pp = body[:, :2] @ p2
        fmax = pf.max()
        fmin = pf.min()
        tang = body[(pf > fmax - 0.004) | (pf < fmin + 0.004)]
        c2 = 0.5 * (fmax + fmin) * f2 + float(np.median(tang[:, :2] @ p2)) * p2
        if o.get("label") == "box":
            c2 = body[:, :2].mean(0)
        hollow = False
        try:
            hit = get_physx_scene_query_interface().raycast_closest(
                carb.Float3(float(c2[0]), float(c2[1]), float(hi[2] + 0.3)), carb.Float3(0, 0, -1), 1.0
            )
            hollow = bool(hit.get("hit")) and float(hit["position"][2]) < hi[2] - 0.015
        except Exception:
            pass
        if mode == "top":
            R = pose_R([0, 0, -1], finger)
            gz = max(hi[2] - GRASP_DEPTH, TABLE + 0.016)
            return dict(
                R=R,
                point=np.array([c2[0], c2[1], gz]),
                lo=lo,
                hi=hi,
                width=width,
                length=length,
                approach=np.array([0, 0, -1.0]),
                hollow=hollow,
                handle=None if handle is None else handle.round(2).tolist(),
            )
        pitch = math.radians(45)
        apr = np.array([math.cos(pitch), 0, -math.sin(pitch)])
        R = pose_R(apr, [0, 1.0, 0])
        gz = max(lo[2] + 0.45 * ext[2], TABLE + 0.02)
        return dict(
            R=R,
            point=np.array([c2[0] - 0.005, c2[1], gz]),
            lo=lo,
            hi=hi,
            width=float(ext[1]),
            length=float(ext[0]),
            approach=apr,
            hollow=hollow,
            handle=None,
        )

    def pick(self, obj_key, mode="top", axis_hint=None):
        g = self.grasp_geometry(obj_key, mode, axis_hint)
        R = g["R"]
        pt = g["point"]
        self.note(
            grasp_geometry=obj_key,
            mode=mode,
            point=pt.round(3).tolist(),
            width=round(g["width"], 3),
            length=round(g["length"], 3),
            finger=R[:, 1].round(2).tolist(),
            hollow=g.get("hollow"),
            handle=g.get("handle"),
        )
        if g["width"] > 0.076:
            raise RuntimeError(f'object_too_wide:{obj_key}:{g["width"]:.3f}')
        OPENG = (
            OPEN
            if self.objects[obj_key].get("label") == "box"
            else float(min(OPEN, max(0.02, g["width"] / 2 + 0.004 + 0.008)))
        )
        if mode == "top":
            self.transit(pt + np.array([0, 0, 0.12]), R, OPENG, f"pick_{obj_key}_pre")
            self.goto(
                pt + np.array([0, 0, 0.045]), R, OPENG, 8, f"pick_{obj_key}_hover", cartesian=True, converge=0.003
            )
            err = self.goto(pt, R, OPENG, 8, f"pick_{obj_key}_descend", cartesian=True, converge=0.004)
            if err > 0.012:
                self.goto(pt + np.array([0, 0, 0.06]), R, OPENG, 6, f"pick_{obj_key}_reup", cartesian=True)
                g = self.grasp_geometry(obj_key, mode, axis_hint)
                R = g["R"]
                pt = g["point"]
                self.note(pick_retry=obj_key, err=round(err, 4), point=pt.round(3).tolist())
                self.goto(
                    pt + np.array([0, 0, 0.06]), R, OPENG, 6, f"pick_{obj_key}_recentre", cartesian=True, converge=0.004
                )
                self.goto(pt, R, OPENG, 8, f"pick_{obj_key}_descend2", cartesian=True, converge=0.005)
        else:
            pre = pt - g["approach"] * 0.12
            self.transit(pre + np.array([0, 0, 0.04]), R, OPEN, f"pick_{obj_key}_pre")
            self.goto(pre, R, OPEN, 6, f"pick_{obj_key}_align", cartesian=True)
            self.goto(pt, R, OPEN, 10, f"pick_{obj_key}_approach", cartesian=True, converge=0.005)
        self.hold(7, f"pick_{obj_key}_close", finger=CLOSED)
        lift = pt + np.array([0, 0, 0.14])
        self.goto(lift, R, CLOSED, 12, f"pick_{obj_key}_lift", cartesian=True)
        lo, hi = bounds_of(self.objects[obj_key]["meshes"])
        lifted = float(lo[2] - g["lo"][2])
        self.note(
            pick=obj_key,
            mode=mode,
            lifted_m=round(lifted, 4),
            width=round(g["width"], 3),
            fingers=self.proprio()[7:].round(4).tolist(),
        )
        self.held = dict(
            key=obj_key,
            R=R,
            ee_minus_bottom=float(self.ee()[0][2] - lo[2]),
            height=float(hi[2] - lo[2]),
            mode=mode,
            half_x=float((hi[0] - lo[0]) / 2),
            half_y=float((hi[1] - lo[1]) / 2),
        )
        if lifted < 0.04:
            raise RuntimeError(f"grasp_failed:{obj_key}")

    def cavity_floor(self, x, y, z_probe):
        hit = get_physx_scene_query_interface().raycast_closest(
            carb.Float3(float(x), float(y), float(z_probe)), carb.Float3(0, 0, -1), 1.5
        )
        return float(hit["position"][2]) if hit.get("hit") else TABLE

    def ray_walk(self, x, y, z, dz, limit=1.5, step=0.012, tries=15):
        """first surface along +/-z from (x,y,z), walking through any solid the origin starts in."""
        q = get_physx_scene_query_interface()
        for _ in range(tries):
            hit = q.raycast_closest(carb.Float3(float(x), float(y), float(z)), carb.Float3(0, 0, float(dz)), limit)
            if not hit.get("hit"):
                return None
            hz = float(hit["position"][2])
            if abs(hz - z) > step:
                return hz
            z += dz * step
        return None

    def cavity_probe(self, x, y, z_top):
        """floor = first surface below the door-top height at (x,y); ceiling = first surface above floor+0.02."""
        q = get_physx_scene_query_interface()
        z = float(z_top - 0.02)
        floor = TABLE
        for _ in range(15):
            hit = q.raycast_closest(carb.Float3(float(x), float(y), z), carb.Float3(0, 0, -1), 1.5)
            if not hit.get("hit"):
                floor = TABLE
                break
            floor = float(hit["position"][2])
            if z - floor > 0.012:
                break
            z -= 0.012
        up = q.raycast_closest(carb.Float3(float(x), float(y), float(floor + 0.02)), carb.Float3(0, 0, 1), 1.0)
        ceil = float(up["position"][2]) if up.get("hit") else float(z_top + 0.10)
        return floor, ceil

    def place_inside(self, asset_key, joint, offset=(0, 0)):
        info = self.assets[asset_key]
        j = info["joints"][joint]
        lo, hi = self.placements[asset_key]["aabb"]
        held = self.held
        if j["type"] == "prismatic":
            link = info["links"][j["body1"]]
            llo, lhi = bounds_of(collision_meshes(link))
            x_in = float(llo[0] + 0.055 + offset[0])
            y_in = float((llo[1] + lhi[1]) / 2 + offset[1])
            floor = self.cavity_floor(x_in, y_in, lhi[2] - 0.004)
            R = held["R"]
            above = np.array([x_in, y_in, floor + 0.12 + held["ee_minus_bottom"]])
            self.transit(above, R, CLOSED, f"place_{asset_key}_pre")
            self.goto(
                np.array([x_in, y_in, floor + 0.008 + held["ee_minus_bottom"]]),
                R,
                CLOSED,
                10,
                f"place_{asset_key}_lower",
                cartesian=True,
            )
            self.hold(4, f"place_{asset_key}_release", finger=OPEN)
            self.goto(above, R, OPEN, 8, f"place_{asset_key}_retreat")
            self.held = None
            return
        dlo, dhi = info["link_aabb_closed"][j["body1"]]
        blo, bhi = self.body_aabb(asset_key)
        y_in = float((dlo[1] + dhi[1]) / 2 + offset[1])
        x_front = float(blo[0])
        emb = held["ee_minus_bottom"]
        h = held["height"]
        x_obj = float(x_front + max(0.10, held.get("half_x", 0.03) + 0.05) + offset[0])
        need = emb + 0.30
        comps = []
        z = float(dhi[2])  # vertical hand: hand (0.14) + wrist (0.07) above the EE + forearm clearance at the front lip
        for _ in range(3):
            fl, ce = self.cavity_probe(x_obj, y_in, z)
            comps.append((fl, ce))
            if fl <= TABLE + 0.005 or ce - fl < 0.02:
                break
            z = fl - 0.01
        fits = [c for c in comps if c[1] - c[0] >= need]
        hmax = max(c[1] - c[0] for c in comps)
        floor, ceil = fits[0] if fits else next(c for c in comps if c[1] - c[0] >= 0.9 * hmax)
        cavity_h = float(ceil - floor)
        self.note(
            cavity_scan=asset_key,
            compartments=[[round(c[0], 3), round(c[1], 3)] for c in comps],
            chosen=[round(floor, 3), round(ceil, 3)],
        )
        lip = floor
        lips = []
        entry_ceil = ceil
        for xp in (x_front + 0.012, x_front + 0.03, x_front + 0.05, (x_front + x_obj) / 2):
            cz = self.ray_walk(xp, y_in, floor + 0.02, +1.0)
            cz = ceil if cz is None else min(cz, ceil)
            fz = self.ray_walk(xp, y_in, cz - 0.01, -1.0)
            fz = TABLE if fz is None else fz
            lips.append([round(fz, 3), round(cz, 3)])
            if floor + 0.004 < fz <= floor + 0.08:
                lip = max(lip, fz)
            entry_ceil = min(entry_ceil, cz)
        cavity_h = float(min(ceil, entry_ceil) - floor)
        zb = lip + 0.03  # object bottom height while entering
        if cavity_h >= need:
            R = held["R"]
            x_ee = x_obj
            z_ee = zb + emb
            mode = "vertical"
        else:
            t = math.radians(35)
            apr = np.array([math.sin(t), 0, -math.cos(t)])
            R = pose_R(apr, held["R"][:, 1])
            x_ee = x_obj - math.sin(t) * emb
            z_ee = zb + math.cos(t) * emb + math.sin(t) * held.get("half_x", 0.03)
            mode = "tilted35"
        self.note(
            entry_lips=asset_key,
            lips=lips,
            lip=round(lip, 3),
            floor=round(floor, 3),
            entry_ceil=round(entry_ceil, 3),
            cavity_h=round(cavity_h, 3),
        )
        self.note(
            place_inside=asset_key,
            x_front=round(x_front, 3),
            x_obj=round(x_obj, 3),
            x_ee=round(x_ee, 3),
            y_in=round(y_in, 3),
            floor=round(floor, 3),
            ceiling=round(ceil, 3),
            cavity_h=round(cavity_h, 3),
            need_vertical=round(need, 3),
            mode=mode,
            finger=R[:, 1].round(2).tolist(),
            z_ee=round(z_ee, 3),
            half_x=round(held.get("half_x", 0), 3),
        )
        xa = float(x_front - 0.12)
        front = np.array([xa, y_in, z_ee + 0.03])
        ee, _ = self.ee()
        zt = max(self.transit_z, ee[2])
        if ee[0] > xa + 0.01:
            self.goto(
                [xa, ee[1], max(ee[2], TABLE + 0.16)],
                held["R"],
                CLOSED,
                8,
                f"place_{asset_key}_front_retract",
                cartesian=True,
            )
        ee, _ = self.ee()
        self.goto([xa, ee[1], zt], held["R"], CLOSED, 8, f"place_{asset_key}_front_up")
        self.goto([xa, front[1], zt], held["R"], CLOSED, 14, f"place_{asset_key}_front_over", cartesian=True)
        self.goto(front, R, CLOSED, 12, f"place_{asset_key}_front_down", cartesian=True, converge=0.006)
        self.goto(
            np.array([x_ee, y_in, z_ee + 0.01]),
            R,
            CLOSED,
            12,
            f"place_{asset_key}_insert",
            max_delta=0.04,
            cartesian=True,
            converge=0.006,
            stop_on_contact=(held["key"], asset_key),
        )
        ee, _ = self.ee()
        if ee[0] < x_ee - 0.02 and self.held is not None:
            self.note(insert_bumped=asset_key, ee=ee.round(3).tolist())
            self.goto(ee + np.array([-0.01, 0, 0.02]), R, CLOSED, 4, f"place_{asset_key}_insert_lift", cartesian=True)
            self.goto(
                np.array([x_ee, y_in, z_ee + 0.03]),
                R,
                CLOSED,
                10,
                f"place_{asset_key}_insert2",
                max_delta=0.04,
                cartesian=True,
                converge=0.006,
            )
        self.goto(
            np.array([x_ee, y_in, z_ee - (zb - floor) - 0.012]),
            R,
            CLOSED,
            8,
            f"place_{asset_key}_lower",
            stop_on_contact=(held["key"], asset_key),
            cartesian=True,
        )
        self.hold(4, f"place_{asset_key}_release", finger=OPEN)
        ee, _ = self.ee()
        self.goto(
            ee + np.array([0, 0, min(0.04, max(0.01, min(ceil, entry_ceil) - 0.02 - HAND_TOP_ABOVE_EE - ee[2]))]),
            R,
            OPEN,
            5,
            f"place_{asset_key}_unhook",
            cartesian=True,
        )
        ee, _ = self.ee()
        self.goto(
            np.array([xa, y_in, ee[2] + 0.02]),
            R,
            OPEN,
            12,
            f"place_{asset_key}_retreat",
            max_delta=0.04,
            cartesian=True,
        )
        self.held = None

    def place_on_top(self, asset_key, joint):
        info = self.assets[asset_key]
        link = info["links"][info["joints"][joint]["body1"]]
        llo, lhi = bounds_of(collision_meshes(link))
        held = self.held
        x = float((llo[0] + lhi[0]) / 2)
        y = float((llo[1] + lhi[1]) / 2)
        top = float(lhi[2])
        R = held["R"]
        above = np.array([x, y, top + 0.12 + held["ee_minus_bottom"]])
        self.transit(above, R, CLOSED, f"place_{asset_key}_pre")
        self.goto(
            np.array([x, y, top - 0.01 + held["ee_minus_bottom"]]),
            R,
            CLOSED,
            10,
            f"place_{asset_key}_lower",
            cartesian=True,
            stop_on_contact=(held["key"], asset_key),
        )
        self.hold(4, f"place_{asset_key}_release", finger=OPEN)
        self.goto(above, R, OPEN, 8, f"place_{asset_key}_retreat")
        self.held = None

    def drop_inside(self, asset_key, joint):
        info = self.assets[asset_key]
        body = [ln for ln in info["links"] if ln != info["joints"][joint]["body1"]]
        lo, hi = bounds_of([m for ln in body for m in collision_meshes(info["links"][ln])])
        held = self.held
        x = float((lo[0] + hi[0]) / 2 - 0.02)
        y = float((lo[1] + hi[1]) / 2)
        R = held["R"]
        above = np.array([x, y, hi[2] + 0.10 + held["ee_minus_bottom"]])
        self.transit(above, R, CLOSED, f"drop_{asset_key}_pre")
        self.goto(np.array([x, y, hi[2] + 0.03 + held["ee_minus_bottom"]]), R, CLOSED, 6, f"drop_{asset_key}_lower")
        self.hold(5, f"drop_{asset_key}_release", finger=OPEN)
        self.goto(above, R, OPEN, 8, f"drop_{asset_key}_retreat")
        self.held = None

    def handle_of(self, asset_key, joint, outward=None):
        info = self.assets[asset_key]
        j = info["joints"][joint]
        link = j["body1"]
        H = HANDLES.get(info["cid"], {"links": {}})
        s = info["scale"]
        body = info["links"][link]
        M = omni.usd.get_world_transform_matrix(body)
        allp = H["links"].get(link, [])
        parts = [p for p in allp if p["part"].split("/")[-1] == "handle"]
        blo, bhi = bounds_of(collision_meshes(body))
        bc = (blo + bhi) / 2
        thin = int(np.argmin(bhi - blo))
        lid = j["type"] == "revolute" and thin == 2 and abs(j["axis"][2]) < 0.5
        cover = [p for p in allp if p["part"].split("/")[-1] == "cover_handle"]
        if lid and cover:
            part = cover[0]
            pts = np.array([list(M.Transform(Gf.Vec3d(*[cc * s for cc in q]))) for q in part["sample"]])
            hc = pts.mean(0)
            hext = pts.max(0) - pts.min(0)
            bar = int(np.argmax(hext[:2]))
            hc[2] = pts[:, 2].max() - 0.004
            bar_dir = np.zeros(3)
            bar_dir[bar] = 1.0
            fing = np.cross(np.array([0, 0, 1.0]), bar_dir)
            fing /= np.linalg.norm(fing)
            return dict(
                center=hc,
                ext=hext,
                bar=bar,
                outward=np.array([0, 0, 1.0]),
                finger=fing,
                door_aabb=(blo, bhi),
                has_handle=True,
                lid=True,
            )
        if lid:
            piv = j["pivot"]
            ax = j["axis"]
            P = mesh_points(collision_meshes(body))
            d = (P - piv) - np.outer((P - piv) @ ax, ax)
            r = np.linalg.norm(d[:, :2], axis=1)
            far = P[r >= r.max() - 0.02]
            hc = np.array([far[:, 0].mean(), far[:, 1].mean(), bhi[2] - 0.005])
            radial = hc - piv
            radial[2] = 0
            radial /= max(np.linalg.norm(radial), 1e-9)
            return dict(
                center=hc,
                ext=np.array([0.02, 0.02, bhi[2] - blo[2]]),
                bar=1,
                outward=np.array([0, 0, 1.0]),
                finger=radial,
                door_aabb=(blo, bhi),
                has_handle=False,
                lid=True,
            )
        if parts:
            part = parts[0]
            pts = np.array([list(M.Transform(Gf.Vec3d(*[cc * s for cc in q]))) for q in part["sample"]])
            hc = pts.mean(0)
            hext = pts.max(0) - pts.min(0)
            bar = int(np.argmax(hext))
        else:
            hc = np.array([blo[0], bc[1], bhi[2] - 0.02])
            hext = np.array([0.01, bhi[1] - blo[1], 0.01])
            bar = 1
        if outward is None:
            outward = np.array([-1.0, 0, 0])
        hc = hc.copy()
        hc[0] = min(hc[0], blo[0] + 0.01)
        bar_dir = np.zeros(3)
        bar_dir[bar] = 1.0
        fing = np.cross(outward, bar_dir)
        fing /= max(np.linalg.norm(fing), 1e-9)
        return dict(
            center=hc,
            ext=hext,
            bar=bar,
            outward=np.asarray(outward, float),
            finger=fing,
            door_aabb=(blo, bhi),
            has_handle=bool(parts),
            lid=False,
        )

    def drawer_edge(self, asset_key, joint):
        """top-edge pinch of the drawer front panel (form closure when pulling): returns None if the edge is covered."""
        info = self.assets[asset_key]
        j = info["joints"][joint]
        link = info["links"][j["body1"]]
        P = mesh_points(collision_meshes(link), per_mesh=400, limit=8000)
        xmin = P[:, 0].min()
        slab = P[P[:, 0] <= xmin + 0.05]
        if len(slab) < 20:
            return None
        top = slab[:, 2].max()
        rim = slab[slab[:, 2] >= top - 0.015]
        thick = float(np.ptp(rim[:, 0]))
        xc = float(rim[:, 0].mean())
        yc = float(rim[:, 1].mean())
        if thick > 0.07:
            return None
        hit = get_physx_scene_query_interface().raycast_closest(
            carb.Float3(xc, yc, top + 0.5), carb.Float3(0, 0, -1), 1.0
        )
        if not hit.get("hit") or abs(float(hit["position"][2]) - top) > 0.02:
            return None
        return dict(point=np.array([xc, yc, top - GRASP_DEPTH + 0.004]), thickness=thick, top=float(top))

    def handle_arc(self, asset_key, joint, deg, label):
        info = self.assets[asset_key]
        j = info["joints"][joint]
        h = self.handle_of(asset_key, joint)
        hc = h["center"]
        outward = h["outward"]
        ax = j["axis"].copy()
        piv = j["pivot"]
        test = rot_axis(ax, math.radians(5)) @ (hc - piv) - (hc - piv)
        if np.dot(test, outward) < 0:
            ax = -ax
        R = pose_R(-outward, h["finger"])
        self.note(
            handle=asset_key,
            joint=joint,
            center=hc.round(3).tolist(),
            ext=h["ext"].round(3).tolist(),
            outward=outward.tolist(),
            finger=h["finger"].round(2).tolist(),
            axis=ax.round(3).tolist(),
            pivot=piv.round(3).tolist(),
            has_handle=h["has_handle"],
            lid=h["lid"],
            deg=deg,
        )
        pre = hc + outward * 0.12
        front_x = float(self.placements[asset_key]["aabb"][0][0])
        hg = S.HANDLE_GEOM.get(info["cid"])
        depth = 0.002
        if hg and not h["lid"]:
            k = info["scale"] / hg["at_scale"]
            g = hg["g"] * k
            thick = hg["thick"] * k
            depth = max(-(g - TIP_BELOW_EE - 0.005), thick / 2 - PALM_ABOVE_EE + 0.004, -0.03)
            self.note(handle_depth=asset_key, g=round(g, 4), thick=round(thick, 4), depth=round(depth, 4))
        if h["lid"]:
            self.transit(pre, R, OPEN, f"{label}_{asset_key}_pre")
        else:
            self.front_approach(pre, R, OPEN, f"{label}_{asset_key}_pre", front_x)
        self.goto(hc + outward * depth, R, OPEN, 8, f"{label}_{asset_key}_reach", cartesian=True, converge=0.004)
        self.hold(6, f"{label}_{asset_key}_grasp", finger=CLOSED)
        Rk, q_start, qn = self.arc(
            ax,
            piv,
            R,
            deg,
            f"{label}_{asset_key}_arc",
            lambda: self.joint_q(asset_key)[joint],
            asset_key=asset_key,
            rotate_hand=not h["lid"],
            gate=0.012,
            expect_sign=0,
        )
        self.note(arc_done=asset_key, joint=joint, q_start=round(q_start, 3), q_now=round(qn, 3), deg=deg)
        ee, _ = self.ee()
        self.hold(3, f"{label}_{asset_key}_release", finger=OPEN)
        back = ee + (Rk @ outward) * 0.08 if not h["lid"] else ee + np.array([-0.06, 0, 0.10])
        back[2] = max(back[2], ee[2] + 0.03)
        if abs(j["axis"][2]) < 0.5 and not h["lid"]:
            back = ee + np.array([-0.08, 0, 0.10])
        self.goto(back, (Rk @ R) if not h["lid"] else R, OPEN, 8, f"{label}_{asset_key}_retreat", cartesian=True)
        return q_start, qn

    def door_edge(self, asset_key, joint):
        """pinch pose on the door's free edge with the hand parallel to the hinge axis: vertical hinge -> pinch the top
        corner of the free edge from above; horizontal (y) hinge -> pinch the -y end of the free (top) edge from the side.
        """
        info = self.assets[asset_key]
        j = info["joints"][joint]
        link = info["links"][j["body1"]]
        P = mesh_points(collision_meshes(link), per_mesh=400, limit=8000)
        piv = j["pivot"]
        ax = j["axis"] / np.linalg.norm(j["axis"])
        d = P - piv
        along = d @ ax
        radial = d - np.outer(along, ax)
        r = np.linalg.norm(radial, axis=1)
        far = P[r >= r.max() - 0.025]
        e = radial[np.argmax(r)] / r.max()
        n = np.cross(ax, e)
        n /= np.linalg.norm(n)
        if n[0] > 0:
            n = -n
        if abs(ax[2]) > 0.9:
            corner = far[far[:, 2] >= far[:, 2].max() - 0.03]
            gp = np.array([corner[:, 0].mean(), corner[:, 1].mean(), far[:, 2].max() - GRASP_DEPTH])
            approach = np.array([0, 0, -1.0])
        else:
            ymin = far[:, 1].min()
            end = far[np.abs(far[:, 1] - ymin) < 0.04]
            gp = np.array([end[:, 0].mean(), ymin + GRASP_DEPTH, end[:, 2].mean()])
            approach = np.array([0, 1.0, 0])
        thick = float(np.ptp(far @ n))
        return dict(gp=gp, approach=approach, n=n, ax=ax, piv=piv, thickness=thick, edge=far.mean(0))

    def open_edge(self, asset_key, joint, deg, attempts=4):
        """open a hinged door to an absolute angle `deg` by pinching its free edge; regrasps when the pinch is lost."""
        info = self.assets[asset_key]
        j = info["joints"][joint]
        target = math.radians(deg)
        for attempt in range(attempts):
            q_now = self.joint_q(asset_key)[joint]
            if q_now >= target * 0.85:
                break
            g = self.door_edge(asset_key, joint)
            gp = g["gp"]
            apr = g["approach"]
            n = g["n"]
            ax = g["ax"].copy()
            piv = g["piv"]
            if g["thickness"] > 0.075:
                raise RuntimeError(f'door_too_thick:{asset_key}:{g["thickness"]:.3f}')
            R = pose_R(apr, n)
            pre = gp - apr * 0.10
            test = rot_axis(ax, math.radians(5)) @ (g["edge"] - piv) - (g["edge"] - piv)
            if (test[2] < 0) if abs(ax[2]) < 0.5 else (np.dot(test, np.array([-1.0, 0, 0])) < 0):
                ax = -ax
            remaining = math.degrees(target - q_now) + 8.0
            self.note(
                door_edge=asset_key,
                joint=joint,
                attempt=attempt,
                gp=gp.round(3).tolist(),
                approach=apr.tolist(),
                n=n.round(2).tolist(),
                thickness=round(g["thickness"], 3),
                q_now=round(q_now, 3),
                remaining_deg=round(remaining, 1),
            )
            if attempt == 0 or apr[2] < -0.5:
                self.transit(pre, R, OPEN, f"open_{asset_key}_pre")
            else:
                self.goto(pre, R, OPEN, 10, f"open_{asset_key}_pre", cartesian=True)
            self.goto(gp, R, OPEN, 8, f"open_{asset_key}_reach", cartesian=True, converge=0.004)
            self.hold(6, f"open_{asset_key}_grasp", finger=CLOSED)
            Rk, q_start, qn = self.arc(
                ax,
                piv,
                R,
                remaining,
                f"open_{asset_key}_arc",
                lambda: self.joint_q(asset_key)[joint],
                asset_key=asset_key,
                rotate_hand=True,
                gate=0.02,
                expect_sign=0,
            )
            self.note(
                arc_done=asset_key,
                joint=joint,
                attempt=attempt,
                q_start=round(q_start, 3),
                q_now=round(qn, 3),
                target=round(target, 3),
            )
            ee, _ = self.ee()
            self.hold(3, f"open_{asset_key}_release", finger=OPEN)
            back = ee - (Rk @ apr) * 0.08
            self.goto(back, Rk @ R, OPEN, 8, f"open_{asset_key}_retreat", cartesian=True)
        qn = self.joint_q(asset_key)[joint]
        if qn < target * 0.75:
            raise RuntimeError(
                f"open_failed:{asset_key}:{joint}:{qn:.3f}<{target*0.75:.3f}"
            )  # v18: 0.75*80 deg = the 60 deg evaluator goal

    def lift_lid(self, asset_key, joint, deg):
        """cap lids without a free edge (trash cans): hook the closed fingertips under the lid's front overhang with a
        horizontal hand and push the lid up along its hinge arc; the hand keeps its orientation."""
        info = self.assets[asset_key]
        j = info["joints"][joint]
        lid = info["links"][j["body1"]]
        body = [info["links"][ln] for ln in info["links"] if ln != j["body1"]]
        llo, lhi = bounds_of(collision_meshes(lid))
        blo, bhi = bounds_of([m for pr in body for m in collision_meshes(pr)])
        piv = j["pivot"]
        ax = j["axis"] / np.linalg.norm(j["axis"])
        P = mesh_points(collision_meshes(lid))
        lip = P[P[:, 0] <= llo[0] + 0.025]
        lip_z = float(lip[:, 2].min()) if len(lip) else float(llo[2])
        overhang = float(blo[0] - llo[0])
        depth = min(max(overhang * 0.6, 0.006), 0.02)
        y_c = float((llo[1] + lhi[1]) / 2)
        tip_x = float(llo[0] + depth)
        ee_x = tip_x - TIP_BELOW_EE
        z_ee = float(lip_z - 0.003 - 0.0105)
        R = pose_R([1.0, 0, 0], [0, 1.0, 0])
        edge = np.array([tip_x, y_c, llo[2]])
        test = rot_axis(ax, math.radians(5)) @ (edge - piv) - (edge - piv)
        if test[2] < 0:
            ax = -ax
        self.note(
            lift_lid=asset_key,
            joint=joint,
            overhang=round(overhang, 4),
            depth=round(depth, 4),
            ee=[round(ee_x, 3), round(y_c, 3), round(z_ee, 3)],
            pivot=piv.round(3).tolist(),
            axis=ax.round(3).tolist(),
            lid_z=[round(float(llo[2]), 3), round(float(lhi[2]), 3)],
            lip_z=round(lip_z, 4),
            body_front=round(float(blo[0]), 3),
        )
        q_start = self.joint_q(asset_key)[joint]
        pre = np.array([ee_x - 0.07, y_c, z_ee])
        self.transit(pre, R, CLOSED, f"lift_{asset_key}_pre")
        self.goto(np.array([ee_x, y_c, z_ee]), R, CLOSED, 10, f"lift_{asset_key}_hook", cartesian=True, converge=0.004)
        Rk, _, qn = self.arc(
            ax,
            piv,
            R,
            deg + 3.0,
            f"lift_{asset_key}_arc",
            lambda: self.joint_q(asset_key)[joint],
            expect_sign=0,
            rotate_hand=False,
            gate=0.025,
            asset_key=None,
            finger=CLOSED,
        )
        self.note(lifted=asset_key, joint=joint, q_start=round(q_start, 3), q_now=round(qn, 3), target_deg=deg)
        ee, _ = self.ee()
        self.goto(ee + np.array([-0.10, 0, 0.04]), R, CLOSED, 8, f"lift_{asset_key}_retreat", cartesian=True)
        if qn < math.radians(deg) * 0.8:
            raise RuntimeError(f"lift_failed:{asset_key}:{joint}:{qn:.3f}")

    def open_revolute(self, asset_key, joint, deg):
        q_start, qn = self.handle_arc(asset_key, joint, deg, "open")
        if (qn - q_start) < math.radians(deg) * 0.5:
            raise RuntimeError(f"open_failed:{asset_key}:{joint}:{qn-q_start:.3f}")

    def open_prismatic(self, asset_key, joint, dist):
        info = self.assets[asset_key]
        j = info["joints"][joint]
        outward = np.array([-1.0, 0, 0])
        ax = j["axis"].copy()
        if np.dot(ax, outward) < 0:
            ax = -ax
        edge = self.drawer_edge(asset_key, joint)
        if edge is not None:
            R = DOWN([1.0, 0, 0])
            pt = edge["point"]
            self.note(
                drawer_edge=asset_key, joint=joint, point=pt.round(3).tolist(), thickness=round(edge["thickness"], 3)
            )
            self.transit(pt + np.array([-0.02, 0, 0.12]), R, OPEN, f"open_{asset_key}_pre")
            self.goto(pt, R, OPEN, 10, f"open_{asset_key}_reach", cartesian=True, converge=0.004)
            self.hold(6, f"open_{asset_key}_grasp", finger=CLOSED)
        else:
            h = self.handle_of(asset_key, joint, outward=outward)
            hc = h["center"]
            R = pose_R(-outward, h["finger"])
            self.note(
                handle=asset_key,
                joint=joint,
                center=hc.round(3).tolist(),
                ext=h["ext"].round(3).tolist(),
                finger=h["finger"].round(2).tolist(),
                axis=ax.round(3).tolist(),
            )
            pre = hc + outward * 0.12
            self.front_approach(pre, R, OPEN, f"open_{asset_key}_pre", float(self.placements[asset_key]["aabb"][0][0]))
            self.goto(hc + outward * 0.002, R, OPEN, 8, f"open_{asset_key}_reach", cartesian=True, converge=0.004)
            self.hold(6, f"open_{asset_key}_grasp", finger=CLOSED)
        g0, _ = self.ee()
        q_start = self.joint_q(asset_key)[joint]
        n = int(max(10, dist / 0.01))
        for i in range(n):
            tgt = g0 + ax * dist * (i + 1) / n
            qi, _ = self.solve_ik(tgt, R)
            q = self.proprio()
            arm = q[:7] + np.clip(qi - q[:7], -0.05, 0.05) if qi is not None else q[:7]
            rec = self.step_tick(np.r_[arm, CLOSED, CLOSED], f"open_{asset_key}_pull")
        qn = self.joint_q(asset_key)[joint]
        self.note(
            opened=asset_key,
            joint=joint,
            q_start=round(q_start, 3),
            q_now=round(qn, 3),
            target_m=dist,
            fingers=self.proprio()[7:].round(4).tolist(),
        )
        ee, _ = self.ee()
        self.hold(3, f"open_{asset_key}_release", finger=OPEN)
        self.goto(ee + np.array([-0.04, 0, 0.10]), R, OPEN, 8, f"open_{asset_key}_retreat", cartesian=True)
        if qn - q_start < dist * 0.5:
            raise RuntimeError(f"open_failed:{asset_key}:{joint}:{qn-q_start:.3f}")

    def push_close_prismatic(self, asset_key, joint):
        info = self.assets[asset_key]
        j = info["joints"][joint]
        link = info["links"][j["body1"]]
        R = DOWN([0, 1.0, 0])
        for attempt in range(2):
            llo, lhi = bounds_of(collision_meshes(link))
            q_start = self.joint_q(asset_key)[joint]
            if q_start < 0.02:
                break
            px = float(llo[0])
            py = float((llo[1] + lhi[1]) / 2)
            pz = float(max(lhi[2] - 0.06, llo[2] + 0.03))
            pre = np.array([px - 0.06, py, pz + 0.10])
            self.transit(pre, R, CLOSED, f"close_{asset_key}_pre")
            self.goto(
                np.array([px - 0.025, py, pz]), R, CLOSED, 8, f"close_{asset_key}_touch", cartesian=True, converge=0.005
            )
            self.goto(
                np.array([px - 0.025 + q_start + 0.03, py, pz]),
                R,
                CLOSED,
                int(max(10, q_start / 0.008)),
                f"close_{asset_key}_push",
                cartesian=True,
                converge=0.006,
            )
            qn = self.joint_q(asset_key)[joint]
            self.note(closed=asset_key, joint=joint, attempt=attempt, q_start=round(q_start, 3), q_now=round(qn, 3))
            ee, _ = self.ee()
            self.goto(ee + np.array([-0.06, 0, 0.10]), R, OPEN, 8, f"close_{asset_key}_retreat", cartesian=True)

    def close_revolute(self, asset_key, joint):
        info = self.assets[asset_key]
        j = info["joints"][joint]
        q_start = self.joint_q(asset_key)[joint]
        if abs(j["axis"][2]) < 0.9:
            self.handle_arc(asset_key, joint, -math.degrees(q_start) * 0.97, "close")
            return
        link = info["links"][j["body1"]]
        llo, lhi = bounds_of(collision_meshes(link))
        ax = j["axis"].copy()
        piv = j["pivot"]
        c = (llo + lhi) / 2
        pt = np.array([c[0], c[1], c[2]])
        test = rot_axis(ax, -math.radians(5)) @ (pt - piv) - (pt - piv)
        closing = test / np.linalg.norm(test)
        R = pose_R(closing, [0, 0, 1.0])
        self.note(
            close_push=asset_key,
            point=pt.round(3).tolist(),
            closing=closing.round(2).tolist(),
            q_start=round(q_start, 3),
        )
        pre = pt - closing * 0.12
        self.front_approach(
            pre, R, CLOSED, f"close_{asset_key}_pre", float(min(pre[0], self.placements[asset_key]["aabb"][0][0]))
        )
        self.goto(pt - closing * 0.03, R, CLOSED, 6, f"close_{asset_key}_touch", cartesian=True)
        Rk, _, qn = self.arc(
            ax, piv, R, -math.degrees(q_start), f"close_{asset_key}_arc", lambda: self.joint_q(asset_key)[joint]
        )
        self.note(closed=asset_key, joint=joint, q_start=round(q_start, 3), q_now=round(qn, 3))
        ee, _ = self.ee()
        self.goto(ee - (Rk @ closing) * 0.10 + np.array([0, 0, 0.05]), Rk @ R, OPEN, 8, f"close_{asset_key}_retreat")

    def push_rotate_closed(self, asset_key, joint):
        info = self.assets[asset_key]
        j = info["joints"][joint]
        link = info["links"][j["body1"]]
        llo, lhi = bounds_of(collision_meshes(link))
        ax = j["axis"].copy()
        piv = j["pivot"]
        q_start = self.joint_q(asset_key)[joint]
        blo, bhi = self.body_aabb(asset_key)
        closing = np.array([(blo[0] + bhi[0]) / 2 - piv[0], 0, 0])
        closing /= max(np.linalg.norm(closing), 1e-9)
        P = mesh_points(collision_meshes(link))
        top = P[P[:, 2] >= lhi[2] - 0.02]
        edge = np.array([top[:, 0].mean(), top[:, 1].mean(), lhi[2]])
        band = P[(P[:, 2] >= lhi[2] - 0.05) & (P[:, 2] <= lhi[2] - 0.03)]
        if len(band) == 0:
            band = top
        face_x = float(band[:, 0].min()) if closing[0] > 0 else float(band[:, 0].max())
        pt = np.array([face_x - 0.016 * closing[0], edge[1], lhi[2] - 0.045 + TIP_BELOW_EE])
        R = DOWN()
        far = pt - closing * 0.07
        self.note(
            push_rotate=asset_key,
            edge=edge.round(3).tolist(),
            face_x=round(face_x, 3),
            closing=closing.round(2).tolist(),
            q_start=round(q_start, 3),
            limits=[j["lower"], j["upper"]],
            pivot=piv.round(3).tolist(),
            axis=ax.round(3).tolist(),
        )
        self.transit(far + np.array([0, 0, 0.12]), R, CLOSED, f"close_{asset_key}_pre")
        self.goto(far, R, CLOSED, 8, f"close_{asset_key}_down", cartesian=True)
        self.goto(pt, R, CLOSED, 8, f"close_{asset_key}_touch", cartesian=True, converge=0.006)
        r = pt - piv
        sign = 1.0 if np.dot(np.cross(ax, r), closing) > 0 else -1.0
        Rk, _, qn = self.arc(
            sign * ax,
            piv,
            R,
            110.0,
            f"close_{asset_key}_arc",
            lambda: self.joint_q(asset_key)[joint],
            expect_sign=0,
            rotate_hand=False,
            gate=0.03,
            asset_key=None,
        )
        llo2, lhi2 = bounds_of(collision_meshes(link))
        pc = np.array([(llo2[0] + lhi2[0]) / 2, (llo2[1] + lhi2[1]) / 2, lhi2[2] + TIP_BELOW_EE + 0.03])
        if S.ARTICULATED[info["asset"]].get("press", True):
            self.goto(pc, R, CLOSED, 8, f"close_{asset_key}_press_pre", cartesian=True)
            self.goto(
                pc - np.array([0, 0, 0.045]),
                R,
                CLOSED,
                6,
                f"close_{asset_key}_press",
                cartesian=True,
                stop_on_contact=(asset_key, "robot"),
            )
            ee, _ = self.ee()
            self.goto(ee - np.array([0, 0, 0.035]), R, CLOSED, 8, f"close_{asset_key}_press2", cartesian=True)
        qn = self.joint_q(asset_key)[joint]
        self.note(closed=asset_key, joint=joint, q_start=round(q_start, 3), q_now=round(qn, 3))
        ee, _ = self.ee()
        self.goto(ee + np.array([-0.06, 0, 0.12]), R, OPEN, 10, f"close_{asset_key}_retreat", cartesian=True)

    def retreat(self):
        ee, _ = self.ee()
        self.goto([ee[0], ee[1], max(self.transit_z, ee[2])], DOWN(), OPEN, 8, "retreat_up")
        self.goto([-0.40, 0.0, self.transit_z], DOWN(), OPEN, 12, "retreat_home")

    def run_teacher(self):
        if a.settle_ticks:
            self.hold(a.settle_ticks, "settle")
            self.note(settle_q={k: self.joint_q(k) for k in self.assets})
        for step in self.spec["plan"][: a.plan_prefix]:
            op = step[0]
            if op == "open_revolute":
                self.open_revolute(step[1], step[2], step[3])
            elif op == "open_edge":
                self.open_edge(step[1], step[2], step[3])
            elif op == "lift_lid":
                self.lift_lid(step[1], step[2], step[3])
            elif op == "open_prismatic":
                self.open_prismatic(step[1], step[2], step[3])
            elif op == "pick":
                self.pick(step[1], step[2], step[3] if len(step) > 3 else None)
            elif op == "place_inside":
                self.place_inside(step[1], step[2], step[3] if len(step) > 3 else (0, 0))
            elif op == "place_on_top":
                self.place_on_top(step[1], step[2])
            elif op == "drop_inside":
                self.drop_inside(step[1], step[2])
            elif op == "push_close_prismatic":
                self.push_close_prismatic(step[1], step[2])
            elif op in ("push_close_revolute", "close_revolute"):
                self.close_revolute(step[1], step[2])
            elif op == "push_rotate_closed":
                self.push_rotate_closed(step[1], step[2])
            elif op == "retreat":
                self.retreat()
            else:
                raise ValueError("unknown_plan_op:" + op)
        self.hold(HOLD_TICKS + 2, "final_hold")

    def run_policy(self, client):
        chunk = None
        ci = 0
        instr = self.spec["instruction"]
        resp = None
        while True:
            q = self.proprio()
            rgb, wr = self.grab()
            if chunk is None or ci >= 4:
                chunk, resp = client.query(
                    self.observe(rgb)[1], self.observe_wrist(wr) if wr is not None else None, q, instr
                )
                ci = 0
            raw = chunk[ci]
            t = np.clip(raw, JOINT_LO, JOINT_HI)
            t[:7] = limit_target(q[:7], t[:7])
            ci += 1
            rec = self.step_tick(t, "policy")
            rec["policy"] = dict(
                chunk_index=ci - 1,
                raw=raw.round(5).tolist(),
                response_sha256=hashlib.sha256(json.dumps(resp, sort_keys=True).encode()).hexdigest(),
            )
            if self.tick >= HOLD_TICKS + 2 and self.tick % 5 == 0 and self.evaluate_now()["status"] == "passed":
                self.note(policy_terminal="success")
                break

    def run_dagger(self, client):
        lmin, lmax = [int(v) for v in a.dagger_burst.split(",")]
        self.dagger = dict(
            client=client,
            beta=a.dagger_beta,
            lmin=lmin,
            lmax=lmax,
            burst_left=0,
            gap_left=0,
            chunk=None,
            ci=0,
            n_policy=0,
            n_teacher=0,
            bursts=0,
            queries=0,
        )
        if self.rng.random() < a.dagger_prefix_prob:
            self.dagger["burst_left"] = self.rng.randint(0, a.dagger_prefix_max)
            self.dagger["bursts"] += int(self.dagger["burst_left"] > 0)
        self.note(dagger_start=dict(beta=a.dagger_beta, burst=[lmin, lmax], prefix=self.dagger["burst_left"]))
        try:
            self.run_teacher()
        finally:
            d = self.dagger
            self.dagger_stats = dict(
                policy_ticks=d["n_policy"],
                teacher_ticks=d["n_teacher"],
                bursts=d["bursts"],
                queries=d["queries"],
                beta_target=a.dagger_beta,
                burst=[lmin, lmax],
            )
            self.dagger = None
            self.note(dagger_stats=self.dagger_stats)

    def evaluate_now(self):
        final = {}
        meta = {}
        for k, info in self.assets.items():
            children = {j["body1"] for j in info["joints"].values()}
            body_links = [ln for ln in info["links"] if ln not in children]
            final[k] = {"body": [v.tolist() for v in self.placements[k]["aabb"]]}
            if body_links:
                ms = [m for ln in body_links for m in collision_meshes(info["links"][ln])]
                if ms:
                    lo, hi = bounds_of(ms)
                    final[k]["body"] = [lo.tolist(), hi.tolist()]
            for jn, j in info["joints"].items():
                lo, hi = bounds_of(collision_meshes(info["links"][j["body1"]]))
                final[k][jn] = [lo.tolist(), hi.tolist()]
            meta[k] = {jn: dict(type=j["type"]) for jn, j in info["joints"].items()}
        return evaluate(self.trace, self.spec["goals"], final, meta, TABLE)

    def finish(self, status, error=None):
        try:
            self.note(contact_samples={k: v for k, v in getattr(self, "contact_samples", {}).items()})
        except Exception:
            pass
        result = (
            self.evaluate_now()
            if len(self.trace) >= HOLD_TICKS + 2
            else dict(status="failed", failure_reason="trace_too_short", goals=[])
        )
        if error and result["status"] != "passed":
            result["teacher_error"] = error
        with (self.dir / "trace.jsonl").open("w") as f:
            for r in self.trace:
                f.write(json.dumps(r) + "\n")
        (self.dir / "episode.json").write_text(
            json.dumps(
                dict(
                    scene=self.name,
                    index=self.index,
                    seed=self.seed,
                    mode=a.mode,
                    variant=a.variant,
                    dagger=self.dagger_stats,
                    instruction=self.spec["instruction"],
                    transit_z=getattr(self, "transit_z", None),
                    placements={
                        k: {kk: (vv if not isinstance(vv, tuple) else [x.tolist() for x in vv]) for kk, vv in v.items()}
                        for k, v in self.placements.items()
                    },
                    assets={
                        k: dict(
                            cid=i["cid"],
                            scale=i["scale"],
                            platform=i["platform"],
                            q0=i.get("q0"),
                            initial_q_authored=i.get("initial_q_authored"),
                        )
                        for k, i in self.assets.items()
                    },
                    objects={
                        k: dict(asset=o["asset"], label=o["label"], distractor=o.get("distractor", False))
                        for k, o in self.objects.items()
                    },
                    ticks=self.tick,
                    result=result,
                    teacher_status=status,
                    log=self.log,
                    robot_base=BASE.tolist(),
                    control_hz=FPS,
                    physics_hz=60,
                    franka_usd=a.franka,
                    gripper_drive=cfg_gripper,
                ),
                indent=1,
                default=str,
            )
        )
        if self.writer:
            try:
                self.writer.close()
            except Exception as ex:
                print("video_close_failed", ex)
            self.writer = None
        if self.first_frame is not None:
            Image.fromarray(self.first_frame).save(self.dir / "first.png")
            Image.fromarray(self.last_frame).save(self.dir / "last.png")
        print(
            json.dumps(
                dict(
                    episode=self.index,
                    scene=self.name,
                    status=result["status"],
                    reason=result.get("failure_reason"),
                    ticks=self.tick,
                    teacher_status=status,
                )
            ),
            flush=True,
        )
        return result


def main():
    spec = S.SCENES[a.scene]
    summary = []
    client = None
    for i in range(a.start_index, a.start_index + a.episodes):
        if a.stop_file and Path(a.stop_file).exists():
            break
        seed = a.seed * 100000 + i
        rng = random.Random(seed)
        ep = Episode(spec, a.scene, i, rng, OUT / f"episode_{i:04d}", seed)
        t0 = time.time()
        status = "completed"
        err = None
        try:
            ep.build()
            if a.camera_probe:
                ep.camera_probe(a.camera_probe)
                (ep.dir / "camera_probe.json").write_text(json.dumps(ep.log, indent=1, default=str))
                print(json.dumps(dict(camera_probe_done=str(ep.dir))), flush=True)
                try:
                    ep.sub = None
                    ep.world.stop()
                    World.clear_instance()
                except Exception:
                    pass
                continue
            if a.mode == "teacher":
                ep.run_teacher()
            elif a.mode == "dagger":
                client = PolicyClient(a.policy_socket)
                ep.run_dagger(client)
                client.close()
                client = None
            else:
                client = PolicyClient(a.policy_socket)
                ep.run_policy(client)
                client.close()
                client = None
        except TimeoutError as ex:
            status = "budget_exhausted"
            err = str(ex)
        except Exception as ex:
            status = "teacher_exception"
            err = str(ex)[:300]
            print(traceback.format_exc()[-1500:], flush=True)
        try:
            res = ep.finish(status, err)
        except Exception:
            print("finish_failed", traceback.format_exc()[-800:], flush=True)
            res = dict(status="failed", failure_reason="finish_failed")
        summary.append(
            dict(
                episode=i,
                seed=seed,
                status=res["status"],
                reason=res.get("failure_reason"),
                teacher_status=status,
                error=err,
                ticks=ep.tick,
                wall_s=round(time.time() - t0, 1),
            )
        )
        with (OUT / "summary.jsonl").open("a") as f:
            f.write(json.dumps(summary[-1]) + "\n")
        try:
            ep.sub = None
            ep.world.stop()
            World.clear_instance()
        except Exception:
            pass
    print(
        json.dumps(dict(scene=a.scene, episodes=len(summary), passed=sum(s["status"] == "passed" for s in summary))),
        flush=True,
    )


main()
app.close()
