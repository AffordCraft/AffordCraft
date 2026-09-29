#!/usr/bin/env python3
"""Official PhysX-Omni TRELLIS decoder and native URDF/MJCF exporter.

Per-case paths, provenance and offline checkpoint resolution are explicit
infrastructure adapters. No substitute mesh, missing-link deletion or physical
success is allowed. The native exporter may emit inadequate physics parameters;
those remain native outputs and require a separate gate/standard adapter.
"""
from __future__ import annotations
import argparse
import gc
from contextlib import contextmanager
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from execution_utils import once, lines_once, read, rows, sha, lock, ROBOT, now, capture

DINO_SHA = "36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51"
U2NET_SHA = "8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491"
DINO_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"


def filetree(p):
    return [
        dict(relative_path=str(f.relative_to(p)), **lock(f))
        for f in sorted(p.rglob("*"))
        if f.is_file() and "__pycache__" not in f.parts and ".cache" not in f.parts
    ]


def offline_dino(source, checkpoint):
    import torch

    original = torch.hub.load

    def load(repo, model, *args, **kwargs):
        if repo != "facebookresearch/dinov2" or model != "dinov2_vitl14_reg" or kwargs.get("pretrained") is not True:
            raise RuntimeError("unregistered_hub_request")
        result = original(str(source), model, source="local", pretrained=False)
        result.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        return result

    torch.hub.load = load


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--representation-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--query-manifest", type=Path)
    p.add_argument("--resource-lock-tag", default="gpu1")
    p.add_argument("--register", action="store_true")
    a = p.parse_args()
    w = a.workspace
    out = a.output
    official = w / "physx-omni"
    dino = w / "dinov2_vitl14_reg4_pretrain.pth.part"
    u2net = w / "u2net.onnx"
    trellis = w / "trellis-image-large-25e0d31f"
    if sha(dino) != DINO_SHA or sha(u2net) != U2NET_SHA:
        raise RuntimeError("auxiliary_checkpoint_hash_mismatch")
    manifest = a.query_manifest or a.representation_root / "native_representation_results.jsonl"
    if not manifest.is_file():
        raise RuntimeError("registered_input_manifest_missing")
    config = dict(
        method_id="physx-omni-v0.1",
        native_or_adapted="native",
        model="official_PhysX_Omni_TRELLIS_image_large",
        input_manifest=lock(manifest),
        representation_configuration=lock(a.representation_root / "configuration.json"),
        stream_in_frozen_input_order=bool(a.query_manifest),
        runner=lock(Path(__file__)),
        upstream_scripts=[lock(official / n) for n in ("decoder_each.py", "2infer_geo.py", "3jsongen_update.py")],
        trellis_files=filetree(trellis),
        dino_checkpoint=lock(dino),
        u2net_checkpoint=lock(u2net),
        dino_source_commit=DINO_COMMIT,
        attention_backend="xformers_cutlass_explicit",
        sparse_attention_backend="xformers_cutlass_explicit",
        xformers_fa3_enabled=False,
        spconv_algorithm="native",
        seed=1,
        formats=["mesh", "gaussian", "radiance_field"],
        simplify=0.5,
        texture_size=1024,
        official_obj_rotation_degrees=[90, 0, 0],
        export=dict(voxel_define=64, process=0, fixed_base=0, deformable=0),
        native_format_priority=["MJCF", "URDF"],
        input_image="retained_full_source_cond_img",
        timing_contract="measured_phase_events;GPU_synchronized;no_phase_imputation;shared_model_load_separate",
        resource_lock_tag=a.resource_lock_tag,
        path_resolution="exact_local_checkpoint_replaces_hub_identifier_only",
        scale_policy="unchanged_official_exporter_model_predicted_cm_to_m",
        mass_policy="unchanged_official_exporter_native_values_no_repair",
        **ROBOT,
    )
    out.mkdir(parents=True, exist_ok=True)
    if a.register:
        once(out / "configuration.json", config)
        print("registered official geometry")
        return 0
    if read(out / "configuration.json") != config:
        raise RuntimeError("geometry_configuration_drift")
    # Coordinate only our own geometry/Isaac lanes. Other users' GPU processes
    # are never stopped or reconfigured. Held for this model's GPU residency.
    if not a.resource_lock_tag.replace("_", "").isalnum():
        raise ValueError("invalid_resource_lock_tag")
    resource_lock = (w / ("resource-" + a.resource_lock_tag + ".lock")).open("a")
    fcntl.flock(resource_lock, fcntl.LOCK_EX)
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        SPCONV_ALGO="native",
        ATTN_BACKEND="xformers",
        SPARSE_ATTN_BACKEND="xformers",
        U2NET_HOME=str(w),
        TOKENIZERS_PARALLELISM="false",
    )
    import numpy as np
    import torch
    from xformers.ops.fmha.dispatch import _set_use_fa3

    _set_use_fa3(False)  # Hopper-only FA3 is invalid on this registered sm120 GPU.
    import functools
    import xformers.ops as xops
    from xformers.ops.fmha import cutlass

    xops.memory_efficient_attention = functools.partial(xops.memory_efficient_attention, op=(cutlass.FwOp, None))
    import trimesh
    from PIL import Image

    sys.path.insert(0, str(official))
    offline_dino(w / f"dinov2-{DINO_COMMIT}", dino)
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils

    t = time.perf_counter()
    pipeline = TrellisImageTo3DPipeline.from_pretrained(str(trellis))
    pipeline.cuda()
    torch.cuda.synchronize()
    once(
        out / "environment.json",
        dict(
            python=sys.version,
            torch=torch.__version__,
            cuda=torch.version.cuda,
            trimesh=trimesh.__version__,
            model_load_seconds=time.perf_counter() - t,
            gpu=capture(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader",
                ]
            ),
            pip_freeze=capture([sys.executable, "-m", "pip", "freeze"]),
            exact_run_command=sys.argv,
        ),
    )
    results = []
    events = []

    @contextmanager
    def measured(phase, sync=False, link=None):
        if sync:
            torch.cuda.synchronize()
        st = time.perf_counter()
        error = None
        try:
            yield
        except Exception as exc:
            error = type(exc).__name__
            raise
        finally:
            if sync:
                torch.cuda.synchronize()
            events.append(
                dict(
                    phase=phase,
                    link=link,
                    runtime_seconds=time.perf_counter() - st,
                    timing_status="measured_runtime",
                    error=error,
                )
            )

    for entry in rows(manifest):
        sid = entry["source_id"]
        dest = out / "cases" / sid
        result_path = dest / "native_asset_result.json"
        if result_path.exists():
            results.append(read(result_path))
            continue
        parent = a.representation_root / "cases" / sid / "native_representation_result.json"
        wait_started = time.perf_counter()
        while not parent.is_file():
            if (a.representation_root / "upstream_blocked.json").is_file():
                raise RuntimeError("upstream_representation_infrastructure_blocked")
            if (a.representation_root / "representation_completion.json").is_file():
                raise RuntimeError("missing_source_in_completed_representation")
            time.sleep(30)
        r = read(parent)
        wait_seconds = time.perf_counter() - wait_started
        dest.mkdir(parents=True, exist_ok=True)
        if (dest / "started.json").exists():
            raise RuntimeError("interrupted_asset_requires_new_attempt")
        once(
            dest / "started.json",
            dict(
                source_id=sid,
                utc=now(),
                parent=lock(a.representation_root / "cases" / sid / "native_representation_result.json"),
            ),
        )
        start = time.perf_counter()
        events.clear()
        err = None
        emitted = False
        parts = 0
        geometry_seconds = None
        export_seconds = None
        phase = "input_validation"
        torch.cuda.reset_peak_memory_stats()
        try:
            if not r["native_representation_emitted"]:
                raise RuntimeError("upstream_model_inference_failed")
            src = a.representation_root / "cases" / sid
            # A write-isolated copy is used because the official exporter writes
            # basic.json, URDF, MJCF and texture references beside the geometry.
            case = dest / "native" / sid
            case.mkdir(parents=True, exist_ok=False)
            for pth in sorted(src.iterdir()):
                if pth.name in ("cond_img.png", "basic_info.txt", "allind.npy") or (
                    pth.name.startswith("ind_") and pth.suffix == ".npy"
                ):
                    shutil.copy2(pth, case / pth.name)
                    if sha(pth) != sha(case / pth.name):
                        raise RuntimeError("copy_hash_mismatch")
            arrays = sorted(case.glob("ind_*.npy"), key=lambda x: int(x.stem.split("_")[1]))
            parts = len(arrays)
            coords = np.load(case / "allind.npy", allow_pickle=False)
            offset = 0
            slices = []
            for array in arrays:
                one = np.load(array, allow_pickle=False)
                if one.ndim != 2 or one.shape[1] != 3 or len(one) == 0:
                    raise ValueError("empty_or_invalid_native_link_voxels")
                slices.append([offset, offset + len(one)])
                offset += len(one)
            if offset != len(coords) or parts != r["part_count"]:
                raise ValueError("native_part_count_mismatch")
            gpu_coords = torch.tensor(np.concatenate([np.zeros((len(coords), 1)), coords], 1), device="cuda").int()
            events.append(
                dict(
                    phase="input_copy_hash_and_voxel_validation",
                    runtime_seconds=time.perf_counter() - start,
                    timing_status="measured_runtime",
                )
            )
            phase = "native_geometry_generation"
            t = time.perf_counter()
            with measured("native_decoder_inference", True):
                outputs = pipeline.run_decoder(gpu_coords, Image.open(case / "cond_img.png"), seed=1, eachcoords=slices)
            if len(outputs) != parts:
                raise ValueError("decoder_part_count_mismatch")
            geometry = []
            for i, output in enumerate(outputs):
                if output is None or len(output["mesh"][0].vertices) == 0:
                    raise ValueError(f"native_link_{i}_mesh_missing")
                folder = case / "objs" / str(i)
                folder.mkdir(parents=True)
                mesh = output["mesh"][0]
                v = mesh.vertices.detach().cpu().numpy()
                f = mesh.faces.detach().cpu().numpy()
                if not np.isfinite(v).all() or len(f) == 0:
                    raise ValueError("invalid_native_geometry")
                with measured("native_mesh_simplification_and_texture_bake", True, i):
                    glb = postprocessing_utils.to_glb(output["gaussian"][0], mesh, simplify=0.5, texture_size=1024)
                with measured("native_visual_GLBOBJ_export", False, i):
                    glb.export(folder / f"{i}.glb")
                    glb.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(90), [1, 0, 0]))
                    glb.export(folder / f"{i}.obj")
                geometry.append(
                    dict(
                        link=i,
                        raw_vertex_count=len(v),
                        raw_face_count=len(f),
                        raw_bounds=[v.min(axis=0).tolist(), v.max(axis=0).tolist()],
                        obj=lock(folder / f"{i}.obj"),
                        glb=lock(folder / f"{i}.glb"),
                    )
                )
            torch.cuda.synchronize()
            geometry_seconds = time.perf_counter() - t
            once(
                dest / "geometry_report.json",
                dict(
                    links=geometry, link_count=parts, topology_repair_used=False, upstream_simplification_fraction=0.5
                ),
            )
            phase = "native_structured_export"
            export_dir = dest / "export_process"
            export_dir.mkdir()
            shutil.copytree(official / "mjcf_source", export_dir / "mjcf_source")
            command = [
                sys.executable,
                str(official / "3jsongen_update.py"),
                "--basepath",
                str(dest / "native"),
                "--voxel_define",
                "64",
                "--process",
                "0",
                "--fixed_base",
                "0",
                "--deformable",
                "0",
            ]
            once(
                export_dir / "command.json",
                dict(command=command, cwd=str(export_dir), upstream=lock(official / "3jsongen_update.py")),
            )
            t = time.perf_counter()
            with (export_dir / "stdout.log").open("xb") as log:
                process = subprocess.run(command, cwd=export_dir, stdout=log, stderr=subprocess.STDOUT)
            export_seconds = time.perf_counter() - t
            events.append(
                dict(phase="native_URDF_MJCF_export", runtime_seconds=export_seconds, timing_status="measured_runtime")
            )
            once(export_dir / "receipt.json", dict(returncode=process.returncode, runtime_seconds=export_seconds))
            if process.returncode or not (case / "basic.xml").is_file() or not (case / "basic.urdf").is_file():
                raise RuntimeError("official_structured_export_missing_or_failed")
            emitted = True
            del outputs, gpu_coords
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:
            err = f"{type(exc).__name__}:{exc}"
            with (dest / "failure.log").open("x") as log:
                log.write(traceback.format_exc())
        lines_once(dest / "phase_timing_events.jsonl", events)
        once(
            dest / "timing.json",
            dict(
                source_id=sid,
                timing_status="measured_runtime",
                phase_events=lock(dest / "phase_timing_events.jsonl"),
                total_runtime_seconds=time.perf_counter() - start,
                upstream_wait_runtime_seconds=wait_seconds,
                geometry_runtime_seconds=geometry_seconds,
                export_runtime_seconds=export_seconds,
                adapter_runtime_seconds=None,
                collision_runtime_seconds=None,
                format_load_runtime_seconds=None,
                physics_runtime_seconds=None,
                missing_phases_status="not_run_native_geometry_and_export_only",
                nested_timings_must_not_be_summed=True,
                peak_memory_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_memory_bytes=torch.cuda.max_memory_reserved(),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                gpu=capture(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
                        "--format=csv,noheader",
                    ]
                ),
                **ROBOT,
            ),
        )
        result = dict(
            source_id=sid,
            method_id="physx-omni-v0.1",
            native_or_adapted="native",
            asset_emitted=emitted,
            part_count=parts,
            format_load=None,
            collision_valid=None,
            physical_sanity=None,
            articulation_valid=None,
            sim_ready_status=(
                "retrieval_only"
                if emitted
                else ("blocked" if phase == "native_structured_export" else "model_inference_failed")
            ),
            failure_reason=["native_PhysX_gate_pending"] if emitted else [err],
            failed_phase=None if emitted else phase,
            task_match=None,
            scripted_success=None,
            input_sha256=r["input_sha256"],
            representation_sha256=sha(a.representation_root / "cases" / sid / "native_representation_result.json"),
            geometry_runtime_seconds=geometry_seconds,
            export_runtime_seconds=export_seconds,
            total_runtime_seconds=time.perf_counter() - start,
            upstream_wait_seconds=wait_seconds,
            peak_memory=torch.cuda.max_memory_allocated(),
            timing_status="measured_runtime",
            timing_scope="native_geometry_and_export_only",
            phase_timing=lock(dest / "timing.json"),
            config_sha256=sha(out / "configuration.json"),
            exact_run_command=sys.argv,
            artifact_paths=filetree(dest),
            fallback_used=False,
            teleport_used=False,
            post_play_transform_writeback=False,
            **ROBOT,
        )
        once(result_path, result)
        results.append(result)
        print(json.dumps(dict(source_id=sid, asset_emitted=emitted, parts=parts, error=err)), flush=True)
    lines_once(out / "native_asset_results.jsonl", results)
    once(
        out / "geometry_completion.json",
        dict(
            status="native_asset_generation_phase_complete_not_PhysX_complete",
            cases=len(results),
            asset_emitted=sum(x["asset_emitted"] for x in results),
            next_phase="native_gate_then_standard_adapter_then_adapted_gate",
            **ROBOT,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
