#!/usr/bin/env python3
"""Lane runner for the official PAct inference over the inputs of a run (registered 200-input subset or the remaining
1,800 inputs).

Sub-commands
  prepare  : build lane-<k>/ (input_manifest.jsonl, workspace with the adapter's
             *_processed.png + *_mask.exr per pending case, failure cases for
             inputs the mask adapter could not localize/segment).
  infer    : run the OFFICIAL infer_imgs.py unchanged (runpy, same env/shim trick as
             tools/pact_probe_local_dino_revision006.py) over the pending cases of the
             lane, with non-invasive instrumentation: per-case wall time, sub-stage
             timers, peak GPU memory, per-case exception capture (a failing case is
             recorded as failed; the lane continues unless the CUDA context is poisoned).
  finalize : per case result.json / timing.json / native_manifest.json (contract
             affordcraft.external_native_manifest.v1), official json_to_urdf, and the
             lane execution_receipt.json.

Nothing here changes PAct weights, sampler settings or the official code; wrappers
only observe.  Append-only: every attempt gets its own --outdir and case records
are written with open(..., 'x').
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import types
from pathlib import Path

METHOD_ID = "pact-v0.1"
SCHEMA = "affordcraft.external_native_manifest.v1"
RUN_ID = None  # run id written into the records: --run-id, default the name of the run root (set in main)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jload(p: Path):
    return json.loads(Path(p).read_text())


def jdump_x(p: Path, obj) -> None:
    with open(p, "x") as f:
        json.dump(obj, f, indent=2)


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_exr_rgb(path):
    import numpy as np
    import OpenEXR
    import Imath

    f = OpenEXR.InputFile(str(path))
    hdr = f.header()
    dw = hdr["dataWindow"]
    w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    ch = {
        c: np.frombuffer(f.channel(c, Imath.PixelType(Imath.PixelType.FLOAT)), np.float32).reshape(h, w) for c in "RGB"
    }
    return np.stack([ch["R"], ch["G"], ch["B"]], -1)


# --------------------------------------------------------------------------- prepare
def lane_records(cfg, lane: int):
    subset = jload(Path(cfg["subset_path"]))
    man = {}
    with open(cfg["manifest_path"]) as f:
        for line in f:
            r = json.loads(line)
            man[r["source_id"]] = r
    out = []
    for idx, x in enumerate(subset["inputs"]):
        if idx % cfg["n_lanes"] != lane:
            continue
        r = man[x["input_id"]]
        out.append(
            {
                "subset_index": idx,
                "lane": lane,
                "source_id": r["source_id"],
                "requested_category": r["requested_category"],
                "image": r["image"],
                "instruction": x.get("instruction"),
            }
        )
    return out


def lane_path(a) -> Path:
    return a.root / (a.lane_name or f"lane-{a.lane}")


def cmd_prepare(a):
    cfg = jload(a.root / "configuration.json")
    lane_dir = lane_path(a)
    lane_dir.mkdir(exist_ok=True)
    (lane_dir / "cases").mkdir(exist_ok=True)
    (lane_dir / "interrupted").mkdir(exist_ok=True)
    (lane_dir / "logs").mkdir(exist_ok=True)
    recs = lane_records(cfg, a.lane)
    if a.ids:
        want = set(a.ids.split(","))
        recs = [r for r in recs if r["source_id"] in want]
    mp = lane_dir / "input_manifest.jsonl"
    if not mp.exists():
        with open(mp, "x") as f:
            for r in recs:
                ad = a.root / "masks" / r["source_id"] / "adapter.json"
                st = jload(ad) if ad.exists() else None
                r2 = dict(r)
                r2["adapter_status"] = st["status"] if st else "missing"
                r2["adapter_failure_reason"] = st["failure_reason"] if st else "adapter_not_run"
                r2["adapter_n_parts"] = (st.get("part_labeling") or {}).get("n_parts") if st else None
                f.write(json.dumps(r2) + "\n")
    # adapter failures become failed cases now (never re-tried with annotations)
    for r in recs:
        sid = r["source_id"]
        cdir = lane_dir / "cases" / sid
        ad = a.root / "masks" / sid / "adapter.json"
        st = jload(ad) if ad.exists() else None
        if st is None or st["status"] == "ok":
            continue
        if (cdir / "result.json").exists():
            continue
        cdir.mkdir(exist_ok=True)
        stage = {
            "localization": "localization",
            "alpha": "segmentation",
            "part_mask": "part_mask",
            "input": "input",
        }.get(st["stage_reached"], st["stage_reached"])
        write_failed_case(cdir, r, st, stage, st["failure_reason"], cfg)
    # a case whose official attempt was hard-killed twice (no finished record, moved to interrupted/) stays failed
    for r in recs:
        sid = r["source_id"]
        cdir = lane_dir / "cases" / sid
        if (cdir / "result.json").exists():
            continue
        n_int = len([p for p in (lane_dir / "interrupted").glob(f"{sid}-attempt-*") if p.is_dir()])
        if n_int >= 2:
            cdir.mkdir(exist_ok=True)
            ad = a.root / "masks" / sid / "adapter.json"
            st = jload(ad) if ad.exists() else None
            write_failed_case(cdir, r, st, "inference", f"official_pipeline_hard_crash_x{n_int}", cfg)
    # workspace with pending cases only
    ws = lane_dir / "workspace"
    assets = ws / "assets" / "real_world_examples"
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir(parents=True)
    src_script = Path(cfg["official"]["source_dir"]) / "infer_imgs.py"
    dst_script = ws / "infer_imgs.py"
    if not dst_script.exists():
        shutil.copy2(src_script, dst_script)
    assert sha256_file(dst_script) == cfg["official"]["file_hashes"]["infer_imgs.py"]
    pending = []
    for r in recs:
        sid = r["source_id"]
        cdir = lane_dir / "cases" / sid
        if (cdir / "result.json").exists():
            continue
        if cdir.exists() and list(cdir.glob("official_attempt-*_finished.json")):
            continue  # inference already happened (ok or failed); only finalize is missing -> never re-run
        ad = a.root / "masks" / sid / "adapter.json"
        if not ad.exists() or jload(ad)["status"] != "ok":
            continue
        d = assets / sid
        d.mkdir()
        for suf in ("_processed.png", "_mask.exr"):
            shutil.copy2(a.root / "masks" / sid / f"{sid}{suf}", d / f"{sid}{suf}")
        pending.append(sid)
    with open(lane_dir / "pending.json", "w") as f:
        json.dump({"written_utc": utc(), "pending": pending}, f, indent=2)
    print(f"[prepare] lane {a.lane}: {len(recs)} cases, pending {len(pending)}")
    return 0


def write_failed_case(cdir: Path, r, adapter, stage, reason, cfg, official_rec=None, seconds=None, peak=None):
    sid = r["source_id"]
    t = {
        "adapter": (adapter or {}).get("timing_seconds", {}),
        "official": (official_rec or {}).get("stage_seconds", {}),
        "official_wall_seconds": (official_rec or {}).get("wall_seconds"),
    }
    total = (
        seconds
        if seconds is not None
        else float((adapter or {}).get("timing_seconds", {}).get("total", 0.0))
        + float((official_rec or {}).get("wall_seconds") or 0.0)
    )
    res = {
        "source_id": sid,
        "method_id": METHOD_ID,
        "asset_emitted": False,
        "part_count": None,
        "failure_reason": reason,
        "runtime_seconds": total,
        "peak_gpu_mem": peak,
        "peak_gpu_mem_unit": "MiB",
        "stage_reached": stage,
        "adapter_n_parts": ((adapter or {}).get("part_labeling") or {}).get("n_parts"),
        "run_id": RUN_ID,
        "written_utc": utc(),
    }
    jdump_x(cdir / "result.json", res)
    jdump_x(cdir / "timing.json", t)
    nm = {
        "schema": SCHEMA,
        "source_id": sid,
        "method_id": METHOD_ID,
        "run_id": RUN_ID,
        "asset_emitted": False,
        "stage_reached": stage,
        "failure_reason": reason,
        "units": "m",
        "metric_size_source": "none",
        "native_formats": [],
        "mjcf": None,
        "urdf": None,
        "root_link": None,
        "links": [],
        "joints": [],
        "physics_contract": False,
        "notes": f"PAct (official infer_imgs.py) could not produce an asset for this input: {reason}. "
        f"Stage reached: {stage}. Automatic part-mask adapter (Grounding-DINO tiny + SAM ViT-B, category-name prompt only, no annotations) "
        f"status: {(adapter or {}).get('status')}. The case stays failed; it is never dropped from the denominator.",
    }
    jdump_x(cdir / "native_manifest.json", nm)


# --------------------------------------------------------------------------- infer
class GpuSampler(threading.Thread):
    """nvidia-smi view of this process's GPU memory (MiB), sampled every `period` s."""

    def __init__(self, pid: int, period: float = 5.0):
        super().__init__(daemon=True)
        self.pid, self.period, self.peak, self.stop_flag = pid, period, 0, False
        self.lock = threading.Lock()

    def reset(self):
        with self.lock:
            self.peak = 0

    def run(self):
        while not self.stop_flag:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=20,
                ).stdout
                for line in out.splitlines():
                    p, m = [x.strip() for x in line.split(",")[:2]]
                    if p.isdigit() and int(p) == self.pid and m.isdigit():
                        with self.lock:
                            self.peak = max(self.peak, int(m))
            except Exception:
                pass
            time.sleep(self.period)


def cmd_infer(a):
    cfg = jload(a.root / "configuration.json")
    lane_dir = lane_path(a)
    ws = lane_dir / "workspace"
    pending = jload(lane_dir / "pending.json")["pending"]
    if not pending:
        print("[infer] nothing pending")
        return 0
    attempts = (
        sorted(p.name for p in (lane_dir / "official_out").glob("attempt-*"))
        if (lane_dir / "official_out").exists()
        else []
    )
    attempt = f"attempt-{len(attempts) + 1:03d}"
    outdir = lane_dir / "official_out" / attempt
    outdir.mkdir(parents=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["ATTN_BACKEND"] = "sdpa"
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    sys.path.insert(0, cfg["compat"]["flash_attn_shim_dir"])
    sys.path.insert(0, cfg["compat"]["dinov2_repo"])
    sys.path.insert(0, str(Path(cfg["official"]["source_dir"]) / "modules"))
    sys.path.insert(0, cfg["official"]["source_dir"])
    sys.modules.setdefault("ipdb", types.ModuleType("ipdb"))
    import numpy as np
    import torch
    import imageio.v3 as iio

    # --- compat 1: DINOv2 from the local checkpoint instead of torch.hub network lookup (identical to the earlier PAct probe run)
    original_hub_load = torch.hub.load
    dino_cache = {}

    def hub_load(repo_or_dir, model, *args, **kw):
        if str(repo_or_dir) == "facebookresearch/dinov2":
            if model != "dinov2_vitl14_reg":
                raise RuntimeError(f"Unexpected DINO model requested by PAct: {model}")
            if model not in dino_cache:
                from dinov2.hub.backbones import dinov2_vitl14_reg

                m = dinov2_vitl14_reg(pretrained=False)
                state = torch.load(cfg["compat"]["dinov2_checkpoint"], map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                m.load_state_dict(state, strict=True)
                dino_cache[model] = m
            return dino_cache[model]
        return original_hub_load(repo_or_dir, model, *args, **kw)

    torch.hub.load = hub_load

    # --- compat 2: imageio EXR read.  The official loader calls imageio.v3.imread(<mask>.exr) and expects the
    # float32 (H,W,3) label array that imageio's EXR-FI (FreeImage) plugin returns.  FreeImage is not installable
    # here (no network); without it imageio silently falls back to pyav and returns 8-bit video frames, which would
    # corrupt the labels.  We return the same float32 RGB array through OpenEXR for '.exr' paths only.
    if cfg["compat"].get("exr_read_shim", True):
        _orig_imread = iio.imread

        def _imread(uri, *args, **kw):
            if not args and not kw and str(uri).lower().endswith(".exr"):
                return read_exr_rgb(uri)
            return _orig_imread(uri, *args, **kw)

        iio.imread = _imread

    # --- instrumentation (observe only)
    from modules.pact.pipelines import pact_i23d_gen_pipe as pipe_mod
    from modules.pact.utils import postprocessing_utils
    from modules.pact.pipelines.pact_i23d_gen_pipe import PActPipeline, BatchRunResult

    state = {"sid": None, "stage_seconds": {}, "export_error": None, "case_started": None}
    sampler = GpuSampler(os.getpid())
    sampler.start()

    def timed(name, fn):
        def w(*args, **kw):
            t = time.perf_counter()
            try:
                return fn(*args, **kw)
            finally:
                state["stage_seconds"][name] = state["stage_seconds"].get(name, 0.0) + (time.perf_counter() - t)

        return w

    for name in ("get_part_coords", "get_slat_arti", "animate_with_articulation", "export_arti_objects"):
        setattr(PActPipeline, name, timed(name, getattr(PActPipeline, name)))
    postprocessing_utils.to_glb = timed("to_glb_texture_bake", postprocessing_utils.to_glb)
    _orig_export = pipe_mod.export_arti_obj_to_singapo_style

    def _export(*args, **kw):
        try:
            return _orig_export(*args, **kw)
        except Exception as e:  # official code swallows this; we record it and re-raise for it to print
            state["export_error"] = f"{type(e).__name__}: {str(e)[:500]}"
            state["export_traceback"] = traceback.format_exc()[-3000:]
            raise

    pipe_mod.export_arti_obj_to_singapo_style = _export
    _orig_run = PActPipeline.run_inference_batch

    def run_batch(self, batch, **kw):
        ids = batch["id"]
        sid = Path(ids[0]).parent.name
        assert len(set(ids)) == 1, ids
        cdir = lane_dir / "cases" / sid
        cdir.mkdir(exist_ok=True)
        state.update({"sid": sid, "stage_seconds": {}, "export_error": None, "export_traceback": None})
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        sampler.reset()
        rec = {
            "source_id": sid,
            "attempt": attempt,
            "outdir": str(outdir),
            "started_utc": utc(),
            "pid": os.getpid(),
            "num_parts_input": int(batch["num_parts"][0]),
            "status": "running",
        }
        jdump_x(cdir / f"official_{attempt}_started.json", rec)
        t0 = time.perf_counter()
        err = None
        poisoned = False
        try:
            res = _orig_run(self, batch, **kw)
        except torch.cuda.OutOfMemoryError as e:
            err = f"cuda_out_of_memory: {str(e)[:300]}"
            res = BatchRunResult([], [], [])
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:500]}"
            rec["traceback"] = traceback.format_exc()[-4000:]
            poisoned = "CUDA" in str(e) or "cuda" in type(e).__name__.lower()
            res = BatchRunResult([], [], [])
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        # the official script appends seed<..>_slatcfg<..>/ to --outdir; export dirs are <class_id>@<obj_id>@<img_id>
        exp = sorted(p for p in outdir.glob(f"*/exported_arti_objects/{sid}@*") if p.is_dir())
        obj_json = [p / "object.json" for p in exp if (p / "object.json").exists()]
        rec.update(
            {
                "finished_utc": utc(),
                "wall_seconds": wall,
                "stage_seconds": dict(state["stage_seconds"]),
                "torch_peak_allocated_mib": float(torch.cuda.max_memory_allocated() / 2**20),
                "torch_peak_reserved_mib": float(torch.cuda.max_memory_reserved() / 2**20),
                "nvidia_smi_peak_mib": int(sampler.peak),
                "error": err,
                "export_error": state["export_error"],
                "export_traceback": state.get("export_traceback"),
                "export_dirs": [str(p) for p in exp],
                "object_json": [str(p) for p in obj_json],
                "status": "ok" if (err is None and obj_json) else "failed",
                "cuda_context_poisoned": poisoned,
            }
        )
        jdump_x(cdir / f"official_{attempt}_finished.json", rec)
        print(
            f"[infer] {sid} -> {rec['status']} {wall:.1f}s err={err} export_error={state['export_error']} objs={len(obj_json)}",
            flush=True,
        )
        if err is not None and "out_of_memory" in err:
            torch.cuda.empty_cache()
        if poisoned:
            raise RuntimeError(f"CUDA context poisoned on {sid}: {err}")
        return res

    PActPipeline.run_inference_batch = run_batch

    ep = ws / "infer_imgs.py"
    os.chdir(str(ws))
    sys.argv = [
        str(ep),
        "--model",
        cfg["checkpoint"]["path"],
        "--data_dir",
        "assets/real_world_examples",
        "--outdir",
        str(outdir),
        "--batch_size",
        "1",
        "--export_arti_objects",
        "--save_glb",
    ]
    jdump_x(
        outdir / "harness_launch.json",
        {
            "argv": sys.argv,
            "cwd": str(ws),
            "python": sys.executable,
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "started_utc": utc(),
            "pending": pending,
            "env": {k: os.environ.get(k) for k in ("ATTN_BACKEND", "HF_HUB_OFFLINE", "PYTHONPATH")},
        },
    )
    import runpy

    t0 = time.perf_counter()
    rc = 0
    try:
        runpy.run_path(str(ep), run_name="__main__")
    except SystemExit as e:
        rc = int(e.code or 0)
    except Exception:  # noqa: BLE001
        rc = 1
        (outdir / "harness_crash.txt").write_text(traceback.format_exc())
        print(traceback.format_exc(), flush=True)
    sampler.stop_flag = True
    jdump_x(
        outdir / "harness_finished.json",
        {"finished_utc": utc(), "wall_seconds": time.perf_counter() - t0, "returncode": rc},
    )
    return rc


# --------------------------------------------------------------------------- finalize
def unit(v):
    n = math.sqrt(sum(float(x) ** 2 for x in v))
    return [float(x) / n for x in v] if n > 1e-8 else None


def build_native_manifest(sid, cdir: Path, export_dir: Path, urdf_path, r, adapter, official_rec):
    import trimesh

    obj = jload(export_dir / "object.json")
    nodes = obj["diffuse_tree"]
    by_id = {n["id"]: n for n in nodes}
    roots = [n for n in nodes if n.get("parent", -1) == -1]
    assert len(roots) == 1, f"{sid}: {len(roots)} roots"
    root = roots[0]
    objdir = cdir / "obj"
    objdir.mkdir(exist_ok=True)

    def pivot(n):
        if n["id"] == root["id"]:
            return [0.0, 0.0, 0.0]
        if n["joint"]["type"] == "fixed":
            return [float(x) for x in (n.get("aabb") or {}).get("center", n["joint"]["axis"]["origin"])]
        return [float(x) for x in n["joint"]["axis"]["origin"]]

    links, joints, notes_extra, masses = [], [], [], {}
    urdf_mass = {}
    if urdf_path and Path(urdf_path).exists():
        import xml.etree.ElementTree as ET

        tree = ET.parse(urdf_path)
        for link in tree.getroot().findall("link"):
            m = link.find("inertial/mass")
            if m is not None:
                urdf_mass[link.get("name")] = float(m.get("value"))
    for n in nodes:
        pid = n["id"]
        ply_rel = n["plys"][0]
        ply = export_dir / ply_rel
        mesh = trimesh.load(ply, force="mesh", process=False)
        objp = objdir / f"part_{pid}.obj"
        if not objp.exists():
            mesh.export(objp)
        piv = pivot(n)
        uname = f"{n.get('name', 'part')}_{pid}".replace("-", "_")
        links.append(
            {
                "name": f"part_{pid}",
                "visual_obj": str(objp.resolve()),
                "mesh_scale": [1.0, 1.0, 1.0],
                "density_kg_m3": None,
                "mass_kg": urdf_mass.get(uname),
                "origin_xyz": [-p for p in piv],
                "origin_rpy": [0.0, 0.0, 0.0],
            }
        )
        notes_extra.append(
            f"part_{pid}: semantic='{n.get('name')}', joint='{n['joint']['type']}', pact_range={n['joint'].get('range')}, "
            f"pact_axis_dir={n['joint']['axis'].get('direction')}, pact_axis_origin={n['joint']['axis'].get('origin')}, ply={ply_rel}, glb={(n.get('glb') or [None])[0]}, "
            f"verts={len(mesh.vertices)}, faces={len(mesh.faces)}, aabb_center={n.get('aabb', {}).get('center')}, aabb_size={n.get('aabb', {}).get('size')}"
        )
        if pid == root["id"]:
            continue
        jt = n["joint"]["type"]
        ctype = {
            "fixed": "fixed",
            "revolute": "revolute",
            "prismatic": "prismatic",
            "continuous": "continuous",
            "screw": "prismatic",
        }[jt]
        ax = unit(n["joint"]["axis"]["direction"]) if ctype != "fixed" else [0.0, 0.0, 0.0]
        if ctype != "fixed" and ax is None:
            ax = [0.0, 0.0, 1.0]
            notes_extra.append(
                f"part_{pid}: zero-length predicted axis replaced by [0,0,1] (same fallback as official json_to_urdf.normalize)"
            )
        rng = n["joint"].get("range") or [0.0, 0.0]
        lo, hi = float(rng[0]), float(rng[1])
        if ctype == "revolute":
            lo, hi = math.radians(lo), math.radians(hi)
        elif ctype == "continuous":
            lo, hi = 0.0, 2 * math.pi
        if lo > hi:
            lo, hi = hi, lo
        parent = by_id[n["parent"]]
        ppiv = pivot(parent)
        joints.append(
            {
                "name": f"joint_{pid}",
                "type": ctype,
                "parent": f"part_{parent['id']}",
                "child": f"part_{pid}",
                "axis": ax,
                "origin_xyz": [piv[i] - ppiv[i] for i in range(3)],
                "lower": lo,
                "upper": hi,
            }
        )
    nm = {
        "schema": SCHEMA,
        "source_id": sid,
        "method_id": METHOD_ID,
        "run_id": RUN_ID,
        "asset_emitted": True,
        "stage_reached": "export",
        "failure_reason": None,
        "units": "m",
        "metric_size_source": "none",
        "native_formats": ["PLY", "GLB", "JSON(diffuse_tree)"] + (["URDF(official json_to_urdf)"] if urdf_path else []),
        "mjcf": None,
        "urdf": (str(Path(urdf_path).resolve()) if urdf_path else None),
        "root_link": f"part_{root['id']}",
        "links": links,
        "joints": joints,
        "physics_contract": bool(
            urdf_path
        ),  # official URDF declares mass (helper min-mass fallback) + GLB collision meshes; see notes
        "notes": (
            f"PAct official export (infer_imgs.py --export_arti_objects --save_glb, seed 42, official sampler defaults) at {export_dir}. "
            f"Input to PAct: automatic part-mask adapter (Grounding-DINO tiny box from the category name '{r['requested_category']}' -> SAM ViT-B alpha -> "
            f"SAM automatic masks merged by PAct's official label_parts code; {((adapter or {}).get('part_labeling') or {}).get('n_parts')} input parts; no annotations read). "
            f"Frames/units: all part meshes share one object frame = PAct's exported SINGAPO-style frame (TRELLIS internal frame rotated by the exporter's "
            f"rot_matrix (x,y,z)->(x,z,-y), i.e. y-up), normalized extent (no metric scale in the export: metric_size_source=none, mesh_scale=[1,1,1]; the 'm' unit is nominal). "
            f"Link frames: root link frame = object frame; every non-root link frame is placed at its joint pivot (axis.origin for movable joints, aabb.center for fixed "
            f"joints, exactly as the official scripts/json_to_urdf.py determine_link_origins), so link origin_xyz = -pivot expresses the mesh (stored in the object frame) "
            f"in the joint child frame; joint origin_xyz = pivot_child - pivot_parent (parent root pivot = 0) in the parent link frame; axis = normalized predicted "
            f"axis.direction in the object frame (= parent frame, identity rotations). Limits: PAct revolute range [0, deg] -> radians; prismatic range [0, d] in the "
            f"normalized object units; continuous -> [0, 2pi]; 'screw' (if any) mapped to prismatic; when PAct's range is descending (e.g. [0, -106 deg]) lower/upper "
            f"are swapped so lower <= upper, exactly as the official json_to_urdf does. Joint tree as exported by PAct (root = 'base' part or largest part, "
            f"all other parts children of the root). OBJ files are lossless conversions (trimesh) of the exported ply/part_<id>.ply (kept next to object.json); "
            f"GLBs are PAct's textured meshes. mass_kg values (if any) come from the official scripts/json_to_urdf.py heuristic (max(1 kg, aabb volume)) written into "
            f"the official URDF, not from a predicted physical parameter; density is not declared; the URDF uses the same GLB meshes as collision geometry, so "
            f"physics_contract=true follows the contract's mechanical rule (mass declared + collision geometry) while the mass values are the helper's constant fallback, "
            f"not a PAct prediction. " + " | ".join(notes_extra)
        ),
    }
    jdump_x(cdir / "native_manifest.json", nm)
    return nm


def cmd_mark_interrupted(a):
    """Move case dirs whose latest official attempt started but never finished to interrupted/."""
    lane_dir = lane_path(a)
    moved = []
    for cdir in sorted((lane_dir / "cases").iterdir()):
        if not cdir.is_dir() or (cdir / "result.json").exists():
            continue
        started = sorted(cdir.glob("official_attempt-*_started.json"))
        if not started:
            continue
        att = started[-1].name.replace("official_", "").replace("_started.json", "")
        if (cdir / f"official_{att}_finished.json").exists():
            continue
        dst = lane_dir / "interrupted" / f"{cdir.name}-{att}"
        shutil.move(str(cdir), str(dst))
        moved.append({"source_id": cdir.name, "attempt": att, "moved_to": str(dst)})
    with open(lane_dir / "interrupted" / f"moved_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json", "x") as f:
        json.dump(moved, f, indent=2)
    print(f"[mark-interrupted] moved {len(moved)}: {[m['source_id'] for m in moved]}")
    return 0


def cmd_finalize(a):
    cfg = jload(a.root / "configuration.json")
    lane_dir = lane_path(a)
    recs = [json.loads(l) for l in open(lane_dir / "input_manifest.jsonl")]
    src = Path(cfg["official"]["source_dir"])
    py = sys.executable
    counts = {"cases": len(recs), "localized": 0, "masked": 0, "inferred": 0, "exported": 0, "failed": 0}
    stage_times = {}
    for r in recs:
        sid = r["source_id"]
        cdir = lane_dir / "cases" / sid
        ad_p = a.root / "masks" / sid / "adapter.json"
        adapter = jload(ad_p) if ad_p.exists() else None
        if adapter and adapter.get("localization", {}).get("chosen"):
            counts["localized"] += 1
        if adapter and adapter["status"] == "ok":
            counts["masked"] += 1
        if (cdir / "result.json").exists():
            res = jload(cdir / "result.json")
            counts["exported" if res["asset_emitted"] else "failed"] += 1
            if res["asset_emitted"]:
                counts["inferred"] += 1
            continue
        fin = sorted(cdir.glob("official_attempt-*_finished.json")) if cdir.exists() else []
        if not fin:
            continue  # still pending
        off = jload(fin[-1])
        if off["status"] == "ok":
            counts["inferred"] += 1
        else:
            counts["failed"] += 1
            reason = off.get("error") or off.get("export_error") or "official_export_missing"
            stage = "inference" if off.get("error") else "export"
            write_failed_case(
                cdir,
                r,
                adapter,
                stage,
                f"official_pipeline_failed: {reason}",
                cfg,
                official_rec=off,
                peak=max(int(off.get("nvidia_smi_peak_mib") or 0), int(off.get("torch_peak_reserved_mib") or 0)),
            )
            continue
        export_dir = Path(off["object_json"][0]).parent
        # official URDF helper (unchanged script, run from the official source root)
        urdf_path = export_dir / "object_fromJson2urdf.urdf"
        urdf_log = cdir / "official_json_to_urdf.log"
        if not urdf_path.exists():
            p = subprocess.run(
                [
                    py,
                    str(src / "scripts" / "json_to_urdf.py"),
                    "--input",
                    str(export_dir / "object.json"),
                    "--output",
                    str(urdf_path),
                    "--glb",
                ],
                cwd=str(src),
                capture_output=True,
                text=True,
            )
            urdf_log.write_text(f"returncode={p.returncode}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}\n")
        urdf_ok = urdf_path.exists()
        t = time.perf_counter()
        try:
            nm = build_native_manifest(sid, cdir, export_dir, str(urdf_path) if urdf_ok else None, r, adapter, off)
        except Exception as e:  # noqa: BLE001
            counts["failed"] += 1
            (cdir / "finalize_error.txt").write_text(traceback.format_exc())
            write_failed_case(
                cdir,
                r,
                adapter,
                "export",
                f"native_manifest_build_failed: {type(e).__name__}: {str(e)[:200]}",
                cfg,
                official_rec=off,
            )
            continue
        conv = time.perf_counter() - t
        counts["exported"] += 1
        peak = max(
            int(off.get("nvidia_smi_peak_mib") or 0),
            int(off.get("torch_peak_reserved_mib") or 0),
            int((adapter or {}).get("torch_peak_alloc_mib") or 0),
        )
        total = float(adapter["timing_seconds"]["total"]) + float(off["wall_seconds"]) + conv
        res = {
            "source_id": sid,
            "method_id": METHOD_ID,
            "asset_emitted": True,
            "part_count": len(nm["links"]),
            "failure_reason": None,
            "runtime_seconds": total,
            "peak_gpu_mem": peak,
            "peak_gpu_mem_unit": "MiB",
            "stage_reached": "export",
            "adapter_n_parts": adapter["part_labeling"]["n_parts"],
            "joint_count": len(nm["joints"]),
            "joint_types": sorted({j["type"] for j in nm["joints"]}),
            "export_dir": str(export_dir),
            "official_urdf": urdf_ok,
            "run_id": RUN_ID,
            "written_utc": utc(),
        }
        jdump_x(cdir / "result.json", res)
        jdump_x(
            cdir / "timing.json",
            {
                "adapter": adapter["timing_seconds"],
                "official": off["stage_seconds"],
                "official_wall_seconds": off["wall_seconds"],
                "finalize_conversion_seconds": conv,
                "total_seconds": total,
            },
        )
        for k, v in adapter["timing_seconds"].items():
            stage_times.setdefault("adapter." + k, []).append(v)
        for k, v in off["stage_seconds"].items():
            stage_times.setdefault("official." + k, []).append(v)
        stage_times.setdefault("official.wall", []).append(off["wall_seconds"])
    pending = [r["source_id"] for r in recs if not (lane_dir / "cases" / r["source_id"] / "result.json").exists()]
    with open(lane_dir / "pending.json", "w") as f:
        json.dump({"written_utc": utc(), "pending": pending}, f, indent=2)
    summary = {
        "lane": a.lane,
        "counts": counts,
        "pending": len(pending),
        "written_utc": utc(),
        "median_stage_seconds": {k: float(sorted(v)[len(v) // 2]) for k, v in stage_times.items()},
    }
    print(json.dumps(summary, indent=1))
    if pending and not a.receipt:
        return 0
    if a.receipt:
        results = {}
        for r in recs:
            cdir = lane_dir / "cases" / r["source_id"]
            for fn in ("result.json", "native_manifest.json", "timing.json"):
                p = cdir / fn
                if p.exists():
                    results[f"{r['source_id']}/{fn}"] = sha256_file(p)
        peaks = [
            jload(lane_dir / "cases" / r["source_id"] / "result.json").get("peak_gpu_mem") or 0
            for r in recs
            if (lane_dir / "cases" / r["source_id"] / "result.json").exists()
        ]
        gpu_idx = a.gpu
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        ).stdout
        rec = {
            "run_id": RUN_ID,
            "lane": a.lane,
            "host": socket.gethostname(),
            "gpu_index": gpu_idx,
            "gpu": [l for l in smi.splitlines() if l.startswith(f"{gpu_idx},")],
            "counts": counts,
            "pending": pending,
            "peak_gpu_mem_mib_max": max(peaks) if peaks else None,
            "median_stage_seconds": summary["median_stage_seconds"],
            "attempts": (
                sorted(p.name for p in (lane_dir / "official_out").glob("attempt-*"))
                if (lane_dir / "official_out").exists()
                else []
            ),
            "interrupted": (
                sorted(p.name for p in (lane_dir / "interrupted").iterdir())
                if (lane_dir / "interrupted").exists()
                else []
            ),
            "result_sha256": results,
            "input_manifest_sha256": sha256_file(lane_dir / "input_manifest.jsonl"),
            "finished_utc": utc(),
            "no_result_selection": True,
        }
        rp = lane_dir / "execution_receipt.json"
        if rp.exists():
            rp = lane_dir / f"execution_receipt_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
        jdump_x(rp, rec)
        print(f"[finalize] receipt written {rp.name}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "infer", "finalize", "mark-interrupted"])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--lane", type=int, required=True)
    ap.add_argument("--lane-name", type=str, default="", help="directory name override (smoke runs); default lane-<k>")
    ap.add_argument("--ids", type=str, default="", help="prepare: restrict to these source ids (smoke runs)")
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--receipt", action="store_true")
    ap.add_argument("--run-id", type=str, default="", help="run id written into the records (default: name of --root)")
    a = ap.parse_args()
    global RUN_ID
    RUN_ID = a.run_id or a.root.name
    return {
        "prepare": cmd_prepare,
        "infer": cmd_infer,
        "finalize": cmd_finalize,
        "mark-interrupted": cmd_mark_interrupted,
    }[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
