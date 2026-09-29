#!/usr/bin/env python
"""PhysX-Anything stage B over one lane: decoder (the body of the official 2_decoder.py, pipeline loaded once per lane from the
official weights), then the official 3_split.py and 4_simready_gen.py run UNMODIFIED as subprocesses from a per-case working
directory (test_demo/<source_id> -> the case dir, mjcf_source -> the official mjcf_source), then native_manifest.json from the
official MJCF plus a MuJoCo cross-check. Runs under the sm_120 env (geom5090). Compatibility changes recorded in
configuration.json: ATTN_BACKEND=xformers with FA3 dispatch disabled and the cutlass forward op forced (the fork defaults to
flash_attn, which is not installed; xformers' default dispatch picks a Hopper kernel that fails on sm_120), SPCONV_ALGO=native
(as in the official 2_decoder.py), torch.hub/dinov2 and rembg/u2net served from pre-populated local caches.
Skips cases whose result.json exists; requires vlm_result.json; an interrupted stage-B case has its stage-B artefacts moved to
interrupted/ (VLM outputs stay untouched) before recomputation."""
import os, sys, json, time, traceback, argparse, subprocess, shutil, gc, functools
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from mjcf_native_manifest import build_native_manifest
from test_mjcf_parser import check as mjcf_check

STAGE_B_FILES = [
    "sample.glb",
    "objs",
    "basic_info.json",
    "basic.urdf",
    "basic.xml",
    "desert.png",
    "geom_started.json",
    "split.log",
    "export.log",
    "split_stdout.log",
    "export_stdout.log",
    "decoder_error.txt",
    "native_manifest.json",
    "mjcf_check.json",
    "timing.json",
    "timing_geom.json",
]
EXPORT_ARGS = [
    "--voxel_define",
    "32",
    "--basepath",
    "./test_demo",
    "--process",
    "0",
    "--fixed_base",
    "0",
    "--deformable",
    "0",
]


def run_official(script, args, cwd, stdout_path, timeout):
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with open(stdout_path, "w", encoding="utf-8") as out:
        t = time.perf_counter()
        r = subprocess.run(
            [sys.executable, str(SRC / script)] + args,
            cwd=str(cwd),
            stdout=out,
            stderr=subprocess.STDOUT,
            env=env,
            timeout=timeout,
        )
    return r.returncode, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane-dir", required=True)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    lane = Path(a.lane_dir)
    cases = lane / "cases"
    cases.mkdir(parents=True, exist_ok=True)
    fh = open(lane / "geom_stage.log", "a", encoding="utf-8")
    rows = rows_of(lane / "input_manifest.jsonl")
    if a.only:
        rows = [r for r in rows if r["source_id"] == a.only]
    log(
        fh,
        "stage B start host=%s cuda_visible=%s rows=%d"
        % (os.uname().nodename, os.environ.get("CUDA_VISIBLE_DEVICES"), len(rows)),
    )
    os.environ.update(
        SPCONV_ALGO="native",
        ATTN_BACKEND="xformers",
        SPARSE_ATTN_BACKEND="xformers",
        TORCH_HOME=str(TORCH_HOME),
        U2NET_HOME=str(U2NET_HOME),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONDONTWRITEBYTECODE="1",
    )
    assert (
        sha256(DINO_CKPT) == "36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51"
    ), "dinov2 checkpoint hash mismatch"
    assert sha256(U2NET) == "8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491", "u2net hash mismatch"
    import numpy as np, torch
    from PIL import Image
    from xformers.ops.fmha.dispatch import _set_use_fa3

    _set_use_fa3(False)  # Hopper-only FA3 is invalid on sm_120
    import xformers.ops as xops
    from xformers.ops.fmha import cutlass

    xops.memory_efficient_attention = functools.partial(xops.memory_efficient_attention, op=(cutlass.FwOp, None))
    sys.path.insert(0, str(SRC))
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils

    t = time.perf_counter()
    pipeline = TrellisImageTo3DPipeline.from_pretrained(str(DECODER))
    pipeline.cuda()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - t
    env_path = lane / "geom_environment.json"
    if not env_path.exists():
        write_json_once(
            env_path,
            dict(
                python=sys.version,
                torch=torch.__version__,
                cuda=torch.version.cuda,
                model_load_seconds=load_seconds,
                models=sorted(pipeline.models.keys()),
                sparse_structure_sampler_params=pipeline.sparse_structure_sampler_params,
                slat_sampler_params=pipeline.slat_sampler_params,
                env={
                    k: os.environ.get(k)
                    for k in (
                        "SPCONV_ALGO",
                        "ATTN_BACKEND",
                        "SPARSE_ATTN_BACKEND",
                        "TORCH_HOME",
                        "U2NET_HOME",
                        "CUDA_VISIBLE_DEVICES",
                    )
                },
                xformers_fa3_enabled=False,
                xformers_op="cutlass.FwOp",
                gpu=capture(
                    ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total", "--format=csv,noheader"]
                ),
                pip_freeze=capture([sys.executable, "-m", "pip", "freeze"]),
                argv=sys.argv,
                utc=utc(),
            ),
        )
    log(fh, "decoder pipeline loaded in %.1fs" % load_seconds)
    for row in rows:
        sid = row["source_id"]
        case = cases / sid
        if (case / "result.json").exists():
            log(fh, "skip %s (result exists)" % sid)
            continue
        if not (case / "vlm_result.json").exists():
            log(fh, "skip %s (no vlm_result.json: VLM stage pending or crashed)" % sid)
            continue
        if (case / "geom_started.json").exists():
            dest = lane / "interrupted" / ("%s-geom-%s" % (sid, stamp()))
            dest.mkdir(parents=True)
            moved = []
            for n in STAGE_B_FILES:
                if (case / n).exists() or (case / n).is_symlink():
                    os.replace(str(case / n), str(dest / n))
                    moved.append(n)
            log(fh, "moved interrupted stage-B artefacts of %s to %s: %s" % (sid, dest, moved))
        write_json_once(
            case / "geom_started.json",
            dict(
                source_id=sid, utc=utc(), host=os.uname().nodename, cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES")
            ),
        )
        vlm = read_json(case / "vlm_result.json")
        tv = read_json(case / "timing_vlm.json")
        timing = dict(
            vlm_seconds=tv["vlm_seconds"],
            vlm_peak_gpu_allocated_bytes=tv["vlm_peak_gpu_allocated_bytes"],
            vlm_peak_gpu_reserved_bytes=tv["vlm_peak_gpu_reserved_bytes"],
            decoder_seconds=None,
            decoder_peak_gpu_allocated_bytes=None,
            decoder_peak_gpu_reserved_bytes=None,
            split_seconds=None,
            export_seconds=None,
            manifest_seconds=None,
        )
        error = None
        stage = "decoder"
        stage_reached = "vlm"
        t_case = time.perf_counter()
        work = lane / "work" / sid
        try:
            # ---------------- 2_decoder.py body (verbatim semantics) ----------------
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            image = Image.open(
                str(P / row["image"]["path"])
            )  # the original photograph, as 2_decoder.py opens the demo file
            newcoords = np.load(str(case / "allind.npy"))
            size = 32
            resolution = 64
            newcoords = newcoords + 32 - (size) // 2
            ss = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long)
            ss[:, newcoords[:, 0], newcoords[:, 1], newcoords[:, 2]] = 1
            ss = ss.cuda().float().unsqueeze(0)
            with torch.no_grad():  # the fork's run() is decorated @torch.no_grad(); run_control() is not, and the retained graph (27 GB) OOMs 32 GB cards
                outputs = pipeline.run_control(
                    ss,
                    image,
                    seed=1,
                )
            glb = postprocessing_utils.to_glb(
                outputs["gaussian"][0],
                outputs["mesh"][0],
                simplify=0.5,
                texture_size=1024,
            )
            glb.export(str(case / "sample.glb"))
            del outputs, ss
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            timing["decoder_seconds"] = time.perf_counter() - t0
            timing["decoder_peak_gpu_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            timing["decoder_peak_gpu_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
            stage_reached = "decoder"
            # ---------------- official 3_split.py (unmodified, subprocess) ----------------
            stage = "split"
            if work.exists():
                shutil.rmtree(work)
            (work / "test_demo").mkdir(parents=True)
            os.symlink(str(case), str(work / "test_demo" / sid))
            os.symlink(str(SRC / "mjcf_source"), str(work / "mjcf_source"))
            rc, secs = run_official("3_split.py", ["--index", "0"], work, case / "split_stdout.log", timeout=7200)
            if (work / "exp_split0.log").exists():
                shutil.copy2(str(work / "exp_split0.log"), str(case / "split.log"))
            timing["split_seconds"] = secs
            if rc != 0:
                raise RuntimeError("3_split.py exit %d" % rc)
            objs = sorted((case / "objs").glob("*/*.obj")) if (case / "objs").is_dir() else []
            if not objs:
                raise RuntimeError("3_split.py produced no part OBJ")
            stage_reached = "split"
            # ---------------- official 4_simready_gen.py (unmodified, subprocess) ----------------
            stage = "export"
            rc, secs = run_official("4_simready_gen.py", EXPORT_ARGS, work, case / "export_stdout.log", timeout=3600)
            if (work / "exp_urdf.log").exists():
                shutil.copy2(str(work / "exp_urdf.log"), str(case / "export.log"))
            timing["export_seconds"] = secs
            if rc != 0:
                raise RuntimeError("4_simready_gen.py exit %d" % rc)
            if not ((case / "basic.xml").is_file() and (case / "basic.urdf").is_file()):
                raise RuntimeError("4_simready_gen.py wrote no basic.xml/basic.urdf")
            stage_reached = "export"
        except Exception as e:
            error = "%s: %s" % (type(e).__name__, str(e)[:500])
            with open(case / "decoder_error.txt", "a", encoding="utf-8") as f:
                f.write("stage=%s\n" % stage + traceback.format_exc())
            log(fh, "FAILED %s at %s: %s" % (sid, stage, error))
            gc.collect()
            torch.cuda.empty_cache()
        asset_emitted = error is None and stage_reached == "export"
        # ---------------- native manifest + MuJoCo cross-check (never fails the case) ----------------
        t0 = time.perf_counter()
        manifest_error = None
        check = None
        try:
            objs_n = len(list((case / "objs").glob("*/*.obj"))) if (case / "objs").is_dir() else 0
            man = build_native_manifest(
                str(case),
                sid,
                METHOD_ID,
                asset_emitted,
                failure_reason=None if asset_emitted else "stage_%s_failed: %s" % (stage, error),
                extra=dict(
                    stage_reached=stage_reached,
                    vlm_part_count=vlm["part_count"],
                    split_part_obj_count=objs_n,
                    glb=str(case / "sample.glb") if (case / "sample.glb").is_file() else None,
                ),
            )
            if asset_emitted:
                check = mjcf_check(str(case / "basic.xml"))
                write_json_once(case / "mjcf_check.json", check)
                man["mjcf_mujoco_load"] = dict(
                    ok=check["mujoco_load_ok"],
                    error=check["mujoco_error"],
                    parser_vs_mujoco_ok=check["ok"],
                    mismatches=check["mismatches"],
                    mujoco_body_mass_kg=check.get("mujoco_body_mass_kg"),
                )
            write_json_once(case / "native_manifest.json", man)
        except Exception as e:
            manifest_error = "%s: %s" % (type(e).__name__, str(e)[:500])
            with open(case / "decoder_error.txt", "a", encoding="utf-8") as f:
                f.write("stage=native_manifest\n" + traceback.format_exc())
            log(fh, "MANIFEST ERROR %s: %s" % (sid, manifest_error))
        timing["manifest_seconds"] = time.perf_counter() - t0
        runtime = tv["vlm_seconds"] + (time.perf_counter() - t_case)
        timing["total_runtime_seconds"] = runtime
        timing["utc"] = utc()
        write_json_once(case / "timing.json", timing)
        write_json_once(
            case / "result.json",
            dict(
                source_id=sid,
                method_id=METHOD_ID,
                stage_reached=stage_reached,
                asset_emitted=asset_emitted,
                part_count=vlm["part_count"],
                failure_reason=None if asset_emitted else "stage_%s_failed: %s" % (stage, error),
                runtime_seconds=runtime,
                native_manifest_ok=manifest_error is None,
                native_manifest_error=manifest_error,
                mjcf_mujoco_load_ok=(check["mujoco_load_ok"] if check else None),
                mjcf_parser_check_ok=(check["ok"] if check else None),
                utc=utc(),
            ),
        )
        log(
            fh,
            "%s %s stage_reached=%s parts=%d decoder=%.1fs split=%s export=%s mujoco=%s"
            % (
                "OK" if asset_emitted else "FAIL",
                sid,
                stage_reached,
                vlm["part_count"],
                timing["decoder_seconds"] or -1,
                timing["split_seconds"],
                timing["export_seconds"],
                check["mujoco_load_ok"] if check else None,
            ),
        )
        if work.exists():
            shutil.rmtree(work, ignore_errors=True)
    log(fh, "stage B done")


if __name__ == "__main__":
    main()
