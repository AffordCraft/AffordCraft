#!/usr/bin/env python3
"""Same official PhysX-Omni generation, with explicit measured phase timing.

Per-case parallelism is a compute-only repeat on fixed round-robin shards.
No tokens, image contract, checkpoint, retrieval, or output is substituted.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import resource
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from execution_utils import once, lines_once, rows, read, sha, lock, ROBOT, now, capture


def files(path):
    return [
        dict(relative_path=str(p.relative_to(path)), **lock(p))
        for p in sorted(path.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and ".cache" not in p.parts
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--input-manifest", type=Path, required=True)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--registration", action="store_true")
    a = p.parse_args()
    w = a.workspace
    out = a.output
    official = w / "physx-omni"
    checkpoint = w / "physx-omni-765cd275"
    processor_path = w / "qwen2.5-vl-7b-processor"
    out.mkdir(parents=True, exist_ok=True)
    definition = dict(
        method_id="physx-omni-v0.1",
        run_id="timed_fixed_shard_repeat_v0_3",
        runner_code=lock(__file__),
        source_code=lock(official / "1vlm_demo.py"),
        prompt=lock(official / "dataset/example_64_finetune_rle.txt"),
        checkpoint_files=files(checkpoint),
        processor_files=files(processor_path),
        input_manifest=lock(a.input_manifest),
        seed=1,
        attn_implementation="sdpa",
        dtype="bfloat16",
        do_sample=False,
        temperature=0,
        max_length=32768,
        image_resize=[512, 512],
        resampling="PIL.LANCZOS",
        min_pixels=65536,
        max_pixels=262144,
        same_generation_parameters_as_parent=True,
        parent_runner_name="paper_completion_omni_native_v0_2.py",
        timing_changes="synchronize_cuda_before_and_after_model_generate;measure_API_boundaries;no_algorithm_change",
        model_input_fields=["full_source_rgb", "official_prompt_only"],
        native_output="physical_description_and_per_link_voxel_representation",
        fixed_denominator_context="whole registered dataset;shard is execution partition only",
        model_warmup="no_separate_warmup;first_query_cold_flagged",
        **ROBOT,
    )
    if a.registration:
        once(out / "configuration.json", definition)
        print("registered", len(rows(a.input_manifest)))
        return 0
    if read(out / "configuration.json") != definition:
        raise RuntimeError("frozen_configuration_drift")
    import torch
    import numpy as np
    from PIL import Image
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    torch.manual_seed(1)
    np.random.seed(1)
    spec = importlib.util.spec_from_file_location("official_physx_omni", official / "1vlm_demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    load_start = time.perf_counter()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(checkpoint),
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": "cuda:0"},
        local_files_only=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        str(processor_path), min_pixels=65536, max_pixels=262144, local_files_only=True
    )
    processor.image_processor.min_pixels = 65536
    processor.image_processor.max_pixels = 262144
    processor.image_processor.size["shortest_edge"] = 65536
    processor.image_processor.size["longest_edge"] = 262144
    torch.cuda.synchronize()
    model_load = time.perf_counter() - load_start
    events = []
    call_context = {"call_id": None}

    def timed(phase, function, synchronize=False):
        def invoke(*args, **kwargs):
            if synchronize:
                torch.cuda.synchronize()
            t = time.perf_counter()
            error = None
            try:
                return function(*args, **kwargs)
            except Exception as exc:
                error = type(exc).__name__
                raise
            finally:
                if synchronize:
                    torch.cuda.synchronize()
                events.append(
                    dict(
                        phase=phase,
                        call_id=call_context["call_id"],
                        runtime_seconds=time.perf_counter() - t,
                        timing_status="measured_runtime",
                        error=error,
                    )
                )

        return invoke

    class ProcessorTimingProxy:
        def __call__(self, *args, **kwargs):
            return timed("processor_encode", processor)(*args, **kwargs)

        def __getattr__(self, name):
            value = getattr(processor, name)
            return timed(name, value) if name in ["apply_chat_template", "batch_decode"] else value

    mod.processor = ProcessorTimingProxy()
    mod.process_vision_info = timed("vision_input_preparation", mod.process_vision_info)
    model.generate = timed("model_generate", model.generate, synchronize=True)
    gpu_query = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,driver_version",
        "--format=csv",
    ]
    env = dict(
        python=sys.version,
        platform=platform.platform(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        transformers=__import__("transformers").__version__,
        gpu=capture(gpu_query),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu_uuid=os.environ.get("CUDA_VISIBLE_DEVICES"),
        processes=capture(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv"]
        ),
        model_load_runtime_seconds=model_load,
        model_warmup_runtime_seconds=None,
        model_warmup_status="not_run_no_hidden_warmup",
        exact_run_command=[sys.executable, *sys.argv],
        pip_freeze=capture([sys.executable, "-m", "pip", "freeze"]),
        **ROBOT,
    )
    once(out / "environment.json", env)
    prompt = (official / "dataset/example_64_finetune_rle.txt").read_text()
    started = time.perf_counter()
    all_rows = rows(a.input_manifest)
    for idx, r in enumerate(all_rows):
        case = out / "cases" / r["source_id"]
        result_path = case / "native_representation_result.json"
        if result_path.exists():
            continue
        case.mkdir(parents=True, exist_ok=True)
        if (case / "started.json").exists():
            raise RuntimeError("interrupted_case_requires_new_attempt_not_overwrite")
        once(
            case / "started.json",
            dict(
                source_id=r["source_id"],
                utc=now(),
                input=r["image"],
                configuration_sha256=sha(out / "configuration.json"),
                **ROBOT,
            ),
        )
        t = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        events.clear()
        error = None
        parts = 0
        allcoord = []
        image_seconds = None
        generation_seconds = None
        parse_seconds = 0.0
        persist_seconds = 0.0
        gpu_before = capture(gpu_query)
        cpu_load_before = os.getloadavg()
        basic_seconds = None
        try:
            image = a.input_root / r["image"]["path"]
            check = time.perf_counter()
            if sha(image) != r["image"]["sha256"]:
                raise RuntimeError("source_image_hash_mismatch")
            events.append(
                dict(
                    phase="source_hash_validation",
                    runtime_seconds=time.perf_counter() - check,
                    timing_status="measured_runtime",
                )
            )
            image_start = time.perf_counter()
            with Image.open(image) as img:
                im = img.convert("RGB")
            im.save(case / "cond_img.png")
            im = im.resize((512, 512), Image.Resampling.LANCZOS)
            image_seconds = time.perf_counter() - image_start
            messages = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": prompt}]}]
            call_context["call_id"] = "basic_info"
            gen_start = time.perf_counter()
            basic = mod.generate_save(model, messages, str(case), "basic_info")
            basic_seconds = time.perf_counter() - gen_start
            while "l_" + str(parts) in basic:
                parts += 1
            for part in range(parts):
                question = f"Based on the structured description of l_{part}, generate its 3D voxel (grid=64) in the 3D RLE (linear scan) format. Output one run per line as: start_index length"
                msg = mod.addmessage(messages, basic, question)
                call_context["call_id"] = "coord_" + str(part)
                output = mod.generate_save(model, msg, str(case), "coord_" + str(part), save=True)
                tp = time.perf_counter()
                v = mod.decode_voxel_2drle_by_z(
                    mod.string_to_runs_by_z_lossless_robust(output, D=64), shape=(64, 64, 64)
                )
                elapsed = time.perf_counter() - tp
                parse_seconds += elapsed
                events.append(
                    dict(
                        phase="voxel_parse",
                        call_id=call_context["call_id"],
                        runtime_seconds=elapsed,
                        timing_status="measured_runtime",
                    )
                )
                ts = time.perf_counter()
                np.save(case / f"ind_{part}.npy", v)
                allcoord.append(v)
                persist_seconds += time.perf_counter() - ts
            if not allcoord:
                raise ValueError("official_generation_no_link_voxels")
            ts = time.perf_counter()
            np.save(case / "allind.npy", np.concatenate(allcoord))
            persist_seconds += time.perf_counter() - ts
            torch.cuda.synchronize()
            generation_seconds = time.perf_counter() - gen_start
        except Exception as exc:
            error = f"{type(exc).__name__}:{exc}"
            with (case / "failure.log").open("x") as f:
                f.write(traceback.format_exc())
        # Write timing even on a failed attempt; never substitute a successful sample.
        sums = defaultdict(float)
        for event in events:
            sums[event["phase"]] += event["runtime_seconds"]
        lines_once(case / "phase_timing_events.jsonl", events)
        timing = dict(
            source_id=r["source_id"],
            timing_status="measured_runtime",
            source_loading_runtime_seconds=image_seconds,
            generation_runtime_seconds=generation_seconds,
            model_inference_runtime_seconds=sums.get("model_generate"),
            parsing_runtime_seconds=parse_seconds,
            representation_storage_runtime_seconds=persist_seconds,
            preprocessing_runtime_seconds=sum(
                sums.get(k, 0.0) for k in ["apply_chat_template", "vision_input_preparation", "processor_encode"]
            ),
            output_decode_runtime_seconds=sums.get("batch_decode"),
            basic_description_runtime_seconds=basic_seconds,
            adapter_runtime_seconds=None,
            collision_runtime_seconds=None,
            export_runtime_seconds=None,
            format_load_runtime_seconds=None,
            physics_runtime_seconds=None,
            absent_phases_status="not_run_representation_stage_only",
            model_load_shared_runtime_seconds=model_load,
            model_load_is_per_case=False,
            cold_first_query=idx == 0,
            phase_timings_are_nested=True,
            nested_timings_must_not_be_summed_into_total=True,
            gpu_before=gpu_before,
            gpu_after=capture(gpu_query),
            cpu_load_before=cpu_load_before,
            cpu_load_after=os.getloadavg(),
            peak_memory_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_memory_bytes=torch.cuda.max_memory_reserved(),
            process_max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            wall_clock_scope="current_case_including_hashes_and_resource_probes_excluding_model_load",
            timing_comparability="check_resource_trace_for_shared_gpu",
            **ROBOT,
        )
        artifacts = [lock(x) for x in sorted(case.iterdir()) if x.is_file() and x.name != "started.json"]
        timing["total_runtime_seconds"] = time.perf_counter() - t
        once(case / "timing.json", timing)
        result = dict(
            source_id=r["source_id"],
            method_id="physx-omni-v0.1",
            run_id="timed_fixed_shard_repeat_v0_3",
            dataset=r.get("dataset"),
            native_or_adapted="native",
            input_sha256=r["image"]["sha256"],
            config_sha256=sha(out / "configuration.json"),
            part_count=parts,
            asset_emitted=False,
            native_representation_emitted=error is None,
            requested_category=r.get("requested_category"),
            candidate_category=None,
            output_sha256=hashlib.sha256(
                json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            format_load=None,
            collision_valid=None,
            physical_sanity=None,
            articulation_valid=None,
            sim_ready_status="model_inference_failed" if error else "retrieval_only",
            failure_reason=[error] if error else ["official_geometry_decode_and_structured_export_pending"],
            scripted_success=None,
            task_match=None,
            runtime_seconds=timing["total_runtime_seconds"],
            peak_memory=timing["peak_memory_bytes"],
            timing_status="measured_runtime",
            timing_scope="native_representation_only_not_end_to_end",
            timing=lock(case / "timing.json"),
            exact_run_command=[sys.executable, *sys.argv],
            artifact_paths=artifacts,
            fallback_used=False,
            teleport_used=False,
            post_play_transform_writeback=False,
            **ROBOT,
        )
        once(result_path, result)
        print(
            json.dumps(
                dict(
                    source_id=r["source_id"],
                    representation_emitted=error is None,
                    parts=parts,
                    runtime_seconds=result["runtime_seconds"],
                    error=error,
                )
            ),
            flush=True,
        )
    result = [read(out / "cases" / r["source_id"] / "native_representation_result.json") for r in all_rows]
    lines_once(out / "native_representation_results.jsonl", result)
    once(
        out / "representation_completion.json",
        dict(
            status="native_representation_phase_complete_not_sim_ready",
            count=len(result),
            native_representation_success=sum(r["native_representation_emitted"] for r in result),
            runtime_seconds=time.perf_counter() - started,
            model_load_runtime_seconds=model_load,
            structured_asset_count=0,
            sim_ready_success=0,
            **ROBOT,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
