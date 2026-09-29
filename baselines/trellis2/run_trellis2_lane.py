#!/usr/bin/env python
"""TRELLIS.2-4B lane runner for the AffordCraft external baseline (registered 200-input subset and remaining 1,800 inputs).

Per case this reproduces the official example.py flow (source commit recorded in configuration.json):
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(<local TRELLIS.2-4B>)   # pipeline.json defaults
    mesh = pipeline.run(image)[0]                # seed 42, pipeline_type '1024_cascade', sampler params from pipeline.json,
                                                 # preprocess_image=True (official BiRefNet/RMBG-2.0 background removal + crop)
    mesh.simplify(16777216)
    glb = o_voxel.postprocess.to_glb(..., decimation_target=1000000, texture_size=4096, remesh=True, remesh_band=1, remesh_project=0)
    glb.export("<case>/mesh.glb", extension_webp=True)
Differences (all listed in configuration.json["compatibility_changes"]): weights are loaded from the local revision-pinned
directory and an offline huggingface_hub cache (DINOv3 / RMBG-2.0 from ModelScope, sha256 == HF LFS oids); the pipeline is
loaded once per lane process; the 120-frame 1024px PBR video of example.py is replaced by a reduced 8-view 512px render
strip; additions: processed_input.png, mesh.obj (geometry of the exported GLB), timing.json, result.json, native_manifest.json.
Append-only: a case whose result.json exists is skipped; a case directory without result.json is moved to interrupted/.
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

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # as in example.py

METHOD_ID = "trellis2-4b-v0.1"
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
    with open(path, "x") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_envmap_tensor(path):
    """example.py reads the HDRI with cv2 (OPENCV_IO_ENABLE_OPENEXR=1). opencv-python-headless 5.0 in this env has no EXR
    codec, so fall back to imageio, then to a uniform white environment. Visualization only; never touches the asset."""
    import numpy as np
    import torch

    try:
        import cv2

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dtype=torch.float32, device="cuda"), "cv2"
    except Exception:
        pass
    try:
        import imageio.v3 as iio

        img = np.asarray(iio.imread(path))
        while img.ndim > 3:
            img = img[0]
        img = img[..., :3].astype(np.float32)
        src = "imageio"
        if img.max() > 1.5:  # 8-bit decode (ffmpeg plugin): LDR environment, visualization only
            img = img / 255.0
            src = "imageio_ldr_uint8"
        return torch.tensor(np.ascontiguousarray(img), dtype=torch.float32, device="cuda"), src
    except Exception:
        pass
    return torch.ones((256, 512, 3), dtype=torch.float32, device="cuda"), "uniform_white_fallback"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision-root", required=True)
    ap.add_argument("--lane", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(args.revision_root)
    cfg = json.load(open(os.path.join(root, "configuration.json")))
    lane_dir = os.path.join(root, f"lane-{args.lane}")
    cases_dir = os.path.join(lane_dir, "cases")
    os.makedirs(cases_dir, exist_ok=True)
    rows = [json.loads(l) for l in open(os.path.join(lane_dir, "input_manifest.jsonl")) if l.strip()]
    gen = cfg["generation_parameters"]
    src_root = cfg["source"]["path"]
    image_root = cfg["inputs"]["image_root"]
    ckpt = cfg["checkpoints"]["trellis2_4b"]["path"]

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

    sys.path.insert(0, src_root)
    os.chdir(src_root)  # example.py is run from the source root (assets/hdri relative path)
    import cv2
    import numpy as np
    import torch
    import trimesh
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.utils import render_utils
    from trellis2.renderers import EnvMap
    import o_voxel

    t_load0 = time.time()
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(ckpt)
    pipeline.cuda()
    envmap = None
    envmap_source = None
    if args.render:
        env_t, envmap_source = load_envmap_tensor(os.path.join(src_root, "assets/hdri/forest.exr"))
        envmap = EnvMap(env_t)
    load_s = time.time() - t_load0
    print(
        f"[lane] pipeline loaded in {load_s:.1f}s; low_vram={pipeline.low_vram} default_pipeline_type={pipeline.default_pipeline_type} envmap={envmap_source} gpu={torch.cuda.get_device_name()}",
        flush=True,
    )
    sampler_defaults = {
        "sparse_structure": pipeline.sparse_structure_sampler_params,
        "shape_slat": pipeline.shape_slat_sampler_params,
        "tex_slat": pipeline.tex_slat_sampler_params,
    }

    seed = int(gen["seed"])
    n_done = n_ok = n_fail = 0
    lane_t0 = time.time()
    fatal_cuda = False
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
            "pipeline_type": gen["pipeline_type"],
            "resolution_used": None,
            "mesh_stats": None,
            "render": None,
            "envmap_source": envmap_source,
            "lane": args.lane,
            "tag": args.tag,
            "host": socket.gethostname(),
            "sampler_params": sampler_defaults,
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
            stage = "preprocess"
            torch.cuda.reset_peak_memory_stats()
            image = Image.open(image_path)
            # -- stage preprocess (official pipeline.preprocess_image: RMBG-2.0 BiRefNet + crop), timed separately --
            t0 = time.time()
            proc = pipeline.preprocess_image(image)
            timing["preprocess_seconds"] = round(time.time() - t0, 3)
            proc.save(os.path.join(case_dir, "processed_input.png"))
            log(f"preprocess done in {timing['preprocess_seconds']}s size={proc.size}")
            # -- stage generation (official pipeline.run with preprocess_image=False since it was done above) --
            stage = "generation"
            t0 = time.time()
            meshes, latent = pipeline.run(
                proc,
                num_samples=1,
                seed=seed,
                preprocess_image=False,
                return_latent=True,
                pipeline_type=gen["pipeline_type"],
                max_num_tokens=int(gen["max_num_tokens"]),
            )
            mesh = meshes[0]
            res_used = int(latent[2])
            torch.cuda.synchronize()
            timing["generation_seconds"] = round(time.time() - t0, 3)
            result["resolution_used"] = res_used
            n_v_raw, n_f_raw = int(mesh.vertices.shape[0]), int(mesh.faces.shape[0])
            log(
                f"generation done in {timing['generation_seconds']}s resolution={res_used} raw_vertices={n_v_raw} raw_faces={n_f_raw}"
            )
            # -- stage export (official: simplify + o_voxel.postprocess.to_glb + glb.export) --
            stage = "export"
            t0 = time.time()
            torch.cuda.empty_cache()  # amendment 001: release torch's cached blocks so CuMesh's own cudaMalloc gets them (shared GPUs)
            mesh.simplify(int(gen["simplify_target"]))
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=int(gen["decimation_target"]),
                texture_size=int(gen["texture_size"]),
                remesh=bool(gen["remesh"]),
                remesh_band=gen["remesh_band"],
                remesh_project=gen["remesh_project"],
                verbose=False,
            )
            glb_path = os.path.join(case_dir, "mesh.glb")
            glb.export(glb_path, extension_webp=True)
            obj_path = os.path.join(case_dir, "mesh.obj")
            trimesh.Trimesh(vertices=np.asarray(glb.vertices), faces=np.asarray(glb.faces), process=False).export(
                obj_path
            )
            timing["export_seconds"] = round(time.time() - t0, 3)
            result["mesh_stats"] = {
                "raw_vertices": n_v_raw,
                "raw_faces": n_f_raw,
                "export_vertices": int(len(glb.vertices)),
                "export_faces": int(len(glb.faces)),
                "bounds": np.asarray(glb.bounds).tolist(),
                "watertight": bool(glb.is_watertight),
                "glb_bytes": os.path.getsize(glb_path),
            }
            links.append(
                {
                    "name": "mesh",
                    "visual_obj": obj_path,
                    "mesh_scale": [1.0, 1.0, 1.0],
                    "density_kg_m3": None,
                    "mass_kg": None,
                    "origin_xyz": [0.0, 0.0, 0.0],
                    "origin_rpy": [0.0, 0.0, 0.0],
                }
            )
            result["asset_emitted"] = True
            result["part_count"] = 1
            result["stage_reached"] = "export"
            log(
                f"export done in {timing['export_seconds']}s faces={len(glb.faces)} glb={os.path.getsize(glb_path)//1024} KB"
            )
            if args.render:
                t0 = time.time()
                try:
                    yaws = [i * 2 * np.pi / 8 + np.pi / 2 for i in range(8)]
                    pitch = [0.25] * 8
                    extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, 2, 40)
                    frames = render_utils.render_frames(
                        mesh, extr, intr, {"resolution": 512, "bg_color": (0, 0, 0)}, verbose=False, envmap=envmap
                    )
                    shaded, normal = frames["shaded"], frames["normal"]
                    strip = Image.new("RGB", (512 * 8, 1024))
                    for k in range(8):
                        strip.paste(Image.fromarray(shaded[k]), (k * 512, 0))
                        strip.paste(Image.fromarray(normal[k]), (k * 512, 512))
                    strip.save(os.path.join(case_dir, "rendering_strip.png"))
                    Image.fromarray(shaded[0]).save(os.path.join(case_dir, "rendering.png"))
                    result["render"] = "ok"
                except Exception as e:
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
                # amendment 003: a CUDA OOM (torch or CuMesh) leaves non-torch allocations behind and the next case fails too;
                # end this process after the case is recorded so the wrapper restarts it with a fresh CUDA context.
                torch.cuda.empty_cache()
                fatal_cuda = True
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
                    "root_link": "mesh" if links else None,
                    "links": links,
                    "joints": [],
                    "physics_contract": False,
                    "notes": (
                        "TRELLIS.2-4B (official checkpoint af44b45f, pipeline_type "
                        + str(gen["pipeline_type"])
                        + ") single textured mesh from the official "
                        "image-to-3D pipeline with the official RMBG-2.0 background removal; exported by the official o_voxel.postprocess.to_glb "
                        "(decimation 1M, 4096 texture, remesh). One link = the whole object in the method's normalized frame (aabb -0.5..0.5, no metric "
                        "scale; mesh_scale 1); no parts, no joints, no mass/density, no collision geometry. mesh.obj holds the geometry of mesh.glb."
                    ),
                },
            )
            write_json_x(os.path.join(case_dir, "result.json"), result)
            clog.close()
            n_done += 1
            torch.cuda.empty_cache()
            print(
                f"[lane] progress done={n_done} ok={n_ok} fail={n_fail} elapsed={time.time() - lane_t0:.0f}s",
                flush=True,
            )
            if fatal_cuda:
                print("[lane] fatal CUDA state; exiting 3 so the wrapper restarts the process", flush=True)
                sys.exit(3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
