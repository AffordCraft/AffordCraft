#!/usr/bin/env python
"""PartCrafter lane runner for the AffordCraft external baseline (registered 200-input subset and remaining 1,800 inputs).

Official code path (PartCrafter source commit is recorded in configuration.json). The per-case flow reproduces
scripts/inference_partcrafter.py invoked as
    --image_path <photo> --part_suggest --part_provider openai_compatible --rmbg --seed 0 --num_tokens 1024
    --num_inference_steps 50 --guidance_scale 7.0 --max_num_expanded_coords 1e9      (no --use_flash_decoder, no --style_transfer)
i.e.  1. set_seed(seed)
      2. num_parts = src.utils.vlm_utils.suggest_num_parts(image_path, MAX_NUM_PARTS=16, mode="object", provider, model_name)
      3. run_triposg(): prepare_image(rmbg_net, white bg) -> PartCrafterPipeline(image=[img]*num_parts, attention_kwargs=
         {"num_parts": num_parts}, num_tokens, generator seeded, num_inference_steps, guidance_scale, max_num_expanded_coords,
         use_flash_decoder=False).meshes ; None meshes (decoding error) replaced by a dummy 1-vertex mesh
      4. part_XX.glb per part, object.glb composite via get_colored_mesh_composition, manifest.json
Differences (all listed in configuration.json["compatibility_changes"]): weights come from the local revision-pinned
directories instead of snapshot_download(); the VLM provider is the module api_part_count_provider (same prompt
text and parser as the official gemini provider, OpenAI-compatible transport) registered at runtime under the provider
name "openai_compatible"; the body of run_triposg is inlined so the rmbg and generation stages can be timed separately;
models are loaded once per lane process; per-part OBJ exports, processed_input.png, vlm_reply.json, timing.json,
result.json, native_manifest.json and the reduced visualization renders are additions for the downstream contract.
Append-only: a case whose result.json exists is skipped; a case directory without result.json (interrupted) is moved
to lane-k/interrupted/ before the lane resumes.
"""
import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import socket
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np

METHOD_ID = "partcrafter-v0.1"
SCHEMA = "affordcraft.external_native_manifest.v1"


def now_iso():
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_x(path, obj):
    """Write once; never overwrite (contract: open(..., 'x'))."""
    with open(path, "x") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision-root", required=True)
    ap.add_argument("--lane", type=int, required=True)
    ap.add_argument("--tag", required=True, help="alphanumeric host-gpu tag for the receipt")
    ap.add_argument("--limit", type=int, default=None, help="process at most N pending cases (smoke test)")
    ap.add_argument("--render", action="store_true", help="write reduced visualization renders (8 views)")
    ap.add_argument(
        "--vlm-workers",
        type=int,
        default=2,
        help="VLM calls prefetched in worker threads (the GPU loop consumes them in subset order)",
    )
    args = ap.parse_args()

    root = os.path.abspath(args.revision_root)
    cfg = json.load(open(os.path.join(root, "configuration.json")))
    lane_dir = os.path.join(root, f"lane-{args.lane}")
    cases_dir = os.path.join(lane_dir, "cases")
    os.makedirs(cases_dir, exist_ok=True)
    rows = load_rows(os.path.join(lane_dir, "input_manifest.jsonl"))
    gen = cfg["generation_parameters"]
    src_root = cfg["source"]["path"]
    image_root = cfg["inputs"]["image_root"]

    # append-only bookkeeping: move interrupted case dirs aside
    for r in rows:
        d = os.path.join(cases_dir, r["source_id"])
        if os.path.isdir(d) and not os.path.exists(os.path.join(d, "result.json")):
            dst = os.path.join(
                lane_dir, "interrupted", f'{r["source_id"]}-{_dt.datetime.now().strftime("%Y%m%dT%H%M%S")}'
            )
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(d, dst)
            print(f"[lane] moved interrupted case dir to {dst}", flush=True)

    pending = [r for r in rows if not os.path.exists(os.path.join(cases_dir, r["source_id"], "result.json"))]
    if args.limit is not None:
        pending = pending[: args.limit]
    print(
        f"[lane] {now_iso()} lane={args.lane} tag={args.tag} rows={len(rows)} pending={len(pending)} host={socket.gethostname()}",
        flush=True,
    )
    if not pending:
        return 0

    # ---- official code + API provider -------------------------------------------------------------------------
    sys.path.insert(0, src_root)
    sys.path.insert(0, os.path.join(root, "code"))
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import torch
    import trimesh
    from PIL import Image
    from accelerate.utils import set_seed
    import api_part_count_provider as api_provider
    from src.utils import providers as _providers

    _providers.PROVIDERS["openai_compatible"] = "api_part_count_provider"
    from src.utils.vlm_utils import suggest_num_parts
    from src.utils.image_utils import prepare_image
    from src.utils.data_utils import get_colored_mesh_composition
    from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
    from src.models.briarmbg import BriaRMBG
    from scripts.inference_partcrafter import MAX_NUM_PARTS  # official constant (16); importing does not run main()

    device = "cuda"
    dtype = torch.float16
    t_load0 = time.time()
    rmbg_net = BriaRMBG.from_pretrained(cfg["checkpoints"]["rmbg_1_4"]["path"]).to(device)
    rmbg_net.eval()
    pipe = PartCrafterPipeline.from_pretrained(cfg["checkpoints"]["partcrafter"]["path"]).to(device, dtype)
    load_s = time.time() - t_load0
    print(f"[lane] models loaded in {load_s:.1f}s; gpu={torch.cuda.get_device_name()}", flush=True)

    vlm_model = cfg["part_suggest"]["model"]
    seed = int(gen["seed"])
    n_done = n_ok = n_fail = 0
    lane_t0 = time.time()
    fatal_cuda = False

    def vlm_task(row):
        """Official suggest_num_parts (API provider) for one case, run in a worker thread; returns the raw exchange too."""
        api_provider.reset_log()
        t0 = time.time()
        out = {
            "num_parts": None,
            "error": None,
            "traceback": None,
            "seconds": None,
            "exchange": None,
            "started": now_iso(),
        }
        try:
            out["num_parts"] = suggest_num_parts(
                os.path.join(image_root, row["image"]["path"]),
                MAX_NUM_PARTS,
                mode="object",
                provider="openai_compatible",
                model_name=vlm_model,
            )
        except Exception as e:  # reported inside the case (failure_reason), never crashes the lane
            out["error"] = e
            out["traceback"] = traceback.format_exc()
        out["seconds"] = round(time.time() - t0, 3)
        out["exchange"] = api_provider.get_log()
        return out

    pool = ThreadPoolExecutor(max_workers=max(1, args.vlm_workers))
    futures = {row["source_id"]: pool.submit(vlm_task, row) for row in pending}

    for row in pending:
        sid = row["source_id"]
        case_dir = os.path.join(cases_dir, sid)
        image_path = os.path.join(image_root, row["image"]["path"])
        os.makedirs(case_dir, exist_ok=False)
        clog = open(os.path.join(case_dir, "case.log"), "a")

        def log(msg):
            line = f"{now_iso()} {msg}"
            clog.write(line + "\n")
            clog.flush()
            print(f"[{sid}] {msg}", flush=True)

        timing = {
            "source_id": sid,
            "started": now_iso(),
            "gpu": torch.cuda.get_device_name(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "model_load_seconds_lane": round(load_s, 3),
        }
        result = {
            "source_id": sid,
            "method_id": METHOD_ID,
            "asset_emitted": False,
            "part_count": None,
            "failure_reason": None,
            "runtime_seconds": None,
            "stage_reached": None,
            "requested_category": row.get("requested_category"),
            "vlm_provider": "openai_compatible",
            "vlm_model": vlm_model,
            "vlm_suggested_parts": None,
            "parts_decoded": None,
            "dummy_parts": [],
            "render": None,
            "lane": args.lane,
            "tag": args.tag,
            "host": socket.gethostname(),
        }
        links = []
        stage = "input"
        t_case0 = time.time()
        try:
            log(f"start image={image_path} category={row.get('requested_category')}")
            h = sha256_file(image_path)
            if h != row["image"]["sha256"]:
                raise RuntimeError(f"input image sha256 mismatch: {h} != {row['image']['sha256']}")
            result["image_sha256_verified"] = True
            stage = "vlm"
            set_seed(seed)  # official: set_seed(args.seed) before the VLM call and generation
            torch.cuda.reset_peak_memory_stats()
            # -- stage vlm (official suggest_num_parts through the API provider; prefetched in a worker thread) --
            t0 = time.time()
            v = futures[sid].result()
            timing["vlm_wait_seconds"] = round(time.time() - t0, 3)
            timing["vlm_seconds"] = v["seconds"]
            timing["vlm_started"] = v["started"]
            write_json_x(
                os.path.join(case_dir, "vlm_reply.json"),
                {
                    "provider": "openai_compatible",
                    "model": vlm_model,
                    "max_num_parts": MAX_NUM_PARTS,
                    "mode": "object",
                    "exchange": v["exchange"],
                    "seconds": v["seconds"],
                    "error": None if v["error"] is None else f"{type(v['error']).__name__}: {str(v['error'])[:500]}",
                },
            )
            if v["error"] is not None:
                clog.write((v["traceback"] or "") + "\n")
                clog.flush()
                raise v["error"]
            num_parts = v["num_parts"]
            result["vlm_suggested_parts"] = num_parts
            result["part_count"] = num_parts
            log(f"vlm suggested num_parts={num_parts} (call {v['seconds']}s, waited {timing['vlm_wait_seconds']}s)")
            # -- stage rmbg (run_triposg: rmbg=True branch) --
            stage = "rmbg"
            t0 = time.time()
            img_pil = prepare_image(image_path, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)
            timing["rmbg_seconds"] = round(time.time() - t0, 3)
            img_pil.save(os.path.join(case_dir, "processed_input.png"))
            log(f"rmbg+crop done in {timing['rmbg_seconds']}s size={img_pil.size}")
            # -- stage generation (run_triposg: pipe(...) call, identical arguments) --
            stage = "generation"
            t0 = time.time()
            with torch.no_grad():
                outputs = pipe(
                    image=[img_pil] * num_parts,
                    attention_kwargs={"num_parts": num_parts},
                    num_tokens=int(gen["num_tokens"]),
                    generator=torch.Generator(device=pipe.device).manual_seed(seed),
                    num_inference_steps=int(gen["num_inference_steps"]),
                    guidance_scale=float(gen["guidance_scale"]),
                    max_num_expanded_coords=gen["max_num_expanded_coords"],
                    use_flash_decoder=bool(gen["use_flash_decoder"]),
                ).meshes
            torch.cuda.synchronize()
            timing["generation_seconds"] = round(time.time() - t0, 3)
            dummy = []
            for i in range(len(outputs)):
                if outputs[i] is None:
                    # official: decoding error -> dummy mesh
                    outputs[i] = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
                    dummy.append(i)
            result["dummy_parts"] = dummy
            result["parts_decoded"] = len(outputs) - len(dummy)
            log(f"generation done in {timing['generation_seconds']}s parts={len(outputs)} dummy={dummy}")
            # -- stage export (official GLBs + composite + manifest; OBJ per decoded part for the contract) --
            stage = "export"
            t0 = time.time()
            part_stats = []
            for i, mesh in enumerate(outputs):
                if i not in dummy:
                    obj_path = os.path.join(case_dir, f"part_{i:02}.obj")
                    mesh.export(obj_path)  # before the composite assigns vertex colors
                    ext = mesh.bounds.tolist() if len(mesh.faces) else None
                    part_stats.append(
                        {
                            "index": i,
                            "vertices": int(len(mesh.vertices)),
                            "faces": int(len(mesh.faces)),
                            "bounds": ext,
                            "watertight": bool(mesh.is_watertight),
                        }
                    )
                    links.append(
                        {
                            "name": f"part_{i:02}",
                            "visual_obj": obj_path,
                            "mesh_scale": [1.0, 1.0, 1.0],
                            "density_kg_m3": None,
                            "mass_kg": None,
                            "origin_xyz": [0.0, 0.0, 0.0],
                            "origin_rpy": [0.0, 0.0, 0.0],
                        }
                    )
                mesh.export(os.path.join(case_dir, f"part_{i:02}.glb"))
            merged_mesh = get_colored_mesh_composition(outputs)
            merged_mesh.export(os.path.join(case_dir, "object.glb"))
            write_json_x(
                os.path.join(case_dir, "manifest.json"),
                {
                    "image_path": image_path,
                    "num_parts": num_parts,
                    "style_transferred": False,
                    "vlm_suggested": True,
                    "parts": [{"index": i, "file": f"part_{i:02}.glb"} for i in range(num_parts)],
                    "composite_file": "object.glb",
                },
            )
            timing["export_seconds"] = round(time.time() - t0, 3)
            result["part_stats"] = part_stats
            result["asset_emitted"] = len(links) > 0
            result["stage_reached"] = "export"
            log(f"export done in {timing['export_seconds']}s links={len(links)}")
            # -- optional reduced renders (visualization only, never affects the result) --
            if args.render:
                t0 = time.time()
                try:
                    from src.utils.render_utils import render_views_around_mesh, render_normal_views_around_mesh

                    views = render_views_around_mesh(merged_mesh, num_views=8, radius=4)
                    normals = render_normal_views_around_mesh(merged_mesh, num_views=8, radius=4)
                    w, h = views[0].size
                    strip = Image.new("RGB", (w * 8, h * 2), (255, 255, 255))
                    for k in range(8):
                        strip.paste(views[k].convert("RGB"), (k * w, 0))
                        strip.paste(normals[k].convert("RGB"), (k * w, h))
                    strip.save(os.path.join(case_dir, "rendering_strip.png"))
                    views[0].save(os.path.join(case_dir, "rendering.png"))
                    normals[0].save(os.path.join(case_dir, "rendering_normal.png"))
                    result["render"] = "ok"
                except Exception as e:  # rendering is not part of the method output
                    result["render"] = f"failed: {type(e).__name__}: {str(e)[:200]}"
                    log(f"render failed: {type(e).__name__}: {e}")
                timing["render_seconds"] = round(time.time() - t0, 3)
            n_ok += 1
        except Exception as e:
            tb = traceback.format_exc()
            result["failure_reason"] = f"{stage}: {type(e).__name__}: {str(e)[:500]}"
            result["stage_reached"] = stage
            clog.write(tb + "\n")
            clog.flush()
            log(f"FAILED at stage {stage}: {type(e).__name__}: {str(e)[:300]}")
            n_fail += 1
            msg = str(e).lower()
            if "out of memory" in msg:
                torch.cuda.empty_cache()
            elif "illegal" in msg or "device-side assert" in msg or "cuda error" in msg or "cublas" in msg:
                fatal_cuda = True
        finally:
            timing["peak_gpu_mem_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            timing["peak_gpu_mem_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
            timing["total_seconds"] = round(time.time() - t_case0, 3)
            timing["finished"] = now_iso()
            result["runtime_seconds"] = timing["total_seconds"]
            write_json_x(os.path.join(case_dir, "timing.json"), timing)
            write_json_x(
                os.path.join(case_dir, "native_manifest.json"),
                {
                    "schema": SCHEMA,
                    "source_id": sid,
                    "method_id": METHOD_ID,
                    "run_id": os.path.basename(root),
                    "asset_emitted": result["asset_emitted"],
                    "stage_reached": result["stage_reached"],
                    "failure_reason": result["failure_reason"],
                    "units": "m",
                    "metric_size_source": "none",
                    "native_formats": ["GLB", "OBJ"] if result["asset_emitted"] else [],
                    "mjcf": None,
                    "urdf": None,
                    "root_link": links[0]["name"] if links else None,
                    "links": links,
                    "joints": [],
                    "physics_contract": False,
                    "notes": (
                        "PartCrafter (official checkpoint 69a0ffc1) part-level image-to-3D: num_parts suggested by the VLM through the "
                        f"official --part_suggest path (API provider, model {vlm_model}), background removed by the official --rmbg "
                        "(BriaRMBG-1.4) preprocessing. Links = the decoded parts in the method's own normalized frame (pipeline bounds "
                        "-1.005..1.005, no metric scale; mesh_scale 1); no joints, no mass/density, no collision geometry -> fixed assembly "
                        "is authored downstream. Native outputs in the case dir: part_XX.glb per part, object.glb composite, manifest.json "
                        f"(official format), part_XX.obj = OBJ export of each decoded part. dummy (undecodable) parts: {result['dummy_parts']}."
                    ),
                },
            )
            write_json_x(os.path.join(case_dir, "result.json"), result)
            clog.close()
            n_done += 1
            print(
                f"[lane] progress done={n_done} ok={n_ok} fail={n_fail} elapsed={time.time() - lane_t0:.0f}s",
                flush=True,
            )
            if fatal_cuda:
                print("[lane] fatal CUDA state; exiting 3 so the wrapper restarts the process", flush=True)
                pool.shutdown(wait=False, cancel_futures=True)
                sys.exit(3)
    pool.shutdown(wait=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
