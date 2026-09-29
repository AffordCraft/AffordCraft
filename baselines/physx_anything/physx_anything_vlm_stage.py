#!/usr/bin/env python
"""PhysX-Anything stage A (VLM) over one lane, with the official 1_vlm_demo.py functions imported unmodified.

Per case (subset order of the lane manifest) this reproduces the body of the official `__main__` loop of 1_vlm_demo.py:
512x512 LANCZOS resize of the whole RGB photograph, the official overall_prompt.txt, remove_bg=False, the official
generate_save (greedy: do_sample=False, temperature=0, max_length=32768; the checkpoint's generation_config supplies the rest),
one follow-up question per part (official wording), dash_str_to_ints/voxel_decode, ind_<k>.npy/.ply and allind.npy.
Compatibility changes (recorded in configuration.json): attn_implementation="sdpa" instead of "flash_attention_2"
(flash-attn is not installed for sm_120); the Qwen2.5-VL processor is loaded from the local copy of
Qwen/Qwen2.5-VL-7B-Instruct's processor files with the same min/max pixels; the model is loaded ONCE for the whole lane.
Skips cases whose result.json or vlm_result.json exists; an interrupted case directory is moved to interrupted/ first.
"""
import os, sys, json, time, traceback, importlib.util, argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane-dir", required=True)
    ap.add_argument("--only", default=None, help="optional single source_id (smoke test)")
    a = ap.parse_args()
    lane = Path(a.lane_dir)
    cases = lane / "cases"
    cases.mkdir(parents=True, exist_ok=True)
    fh = open(lane / "vlm_stage.log", "a", encoding="utf-8")
    rows = rows_of(lane / "input_manifest.jsonl")
    if a.only:
        rows = [r for r in rows if r["source_id"] == a.only]
    log(
        fh,
        "stage A start host=%s cuda_visible=%s rows=%d"
        % (os.uname().nodename, os.environ.get("CUDA_VISIBLE_DEVICES"), len(rows)),
    )
    import torch, numpy as np, trimesh
    from PIL import Image
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    spec = importlib.util.spec_from_file_location("physx_anything_vlm_demo", str(SRC / "1_vlm_demo.py"))
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)  # defines generate_save/addmessage/dash_str_to_ints/voxel_decode only
    t = time.perf_counter()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(VLM_CKPT), torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto"
    )
    min_pixels = 65536
    max_pixels = 262144
    processor = AutoProcessor.from_pretrained(str(PROCESSOR), min_pixels=min_pixels, max_pixels=max_pixels)
    processor.image_processor.min_pixels = min_pixels
    processor.image_processor.max_pixels = max_pixels
    processor.image_processor.size["shortest_edge"] = min_pixels
    processor.image_processor.size["longest_edge"] = max_pixels
    demo.processor = (
        processor  # generate_save() reads the module-level name `processor`, exactly as in the official script
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - t
    with open(SRC / "dataset" / "overall_prompt.txt", "r", encoding="utf-8") as f:
        basicqu = f.read()
    env_path = lane / "vlm_environment.json"
    if not env_path.exists():
        write_json_once(
            env_path,
            dict(
                python=sys.version,
                torch=torch.__version__,
                cuda=torch.version.cuda,
                transformers=__import__("transformers").__version__,
                attn_implementation="sdpa",
                model_load_seconds=load_seconds,
                generation_config=json.loads(model.generation_config.to_json_string()),
                device=str(model.device),
                gpu=capture(
                    ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total", "--format=csv,noheader"]
                ),
                pip_freeze=capture([sys.executable, "-m", "pip", "freeze"]),
                argv=sys.argv,
                utc=utc(),
            ),
        )
    log(fh, "model loaded in %.1fs device=%s" % (load_seconds, model.device))
    for row in rows:
        sid = row["source_id"]
        case = cases / sid
        if (case / "result.json").exists() or (case / "vlm_result.json").exists():
            log(fh, "skip %s (already has result)" % sid)
            continue
        if case.exists():
            dest = move_to_interrupted(case, lane / "interrupted", "vlm")
            log(fh, "moved interrupted case dir %s -> %s" % (sid, dest))
        case.mkdir()
        write_json_once(
            case / "vlm_started.json",
            dict(
                source_id=sid, utc=utc(), host=os.uname().nodename, cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES")
            ),
        )
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t_case = time.perf_counter()
        error = None
        stage = "input_validation"
        parts = []
        index = 0
        try:
            image_path = P / row["image"]["path"]
            h = sha256(image_path)
            if h != row["image"]["sha256"]:
                raise RuntimeError("input_image_sha256_mismatch %s" % h)
            input_image = Image.open(image_path)
            im_resized = input_image.resize((512, 512), Image.LANCZOS)
            # remove_bg is False for this run (official default): the model sees the whole photograph.
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": im_resized.convert("RGB")},
                        {"type": "text", "text": basicqu},
                    ],
                }
            ]
            stage = "vlm_basic_info"
            t0 = time.perf_counter()
            torch.manual_seed(1)
            basicoutput = demo.generate_save(model, messages, str(case), "basic_info")
            torch.cuda.synchronize()
            parts.append(
                dict(
                    name="basic_info",
                    seconds=time.perf_counter() - t0,
                    chars=len(basicoutput),
                    tokens=len(processor.tokenizer(basicoutput).input_ids),
                )
            )
            index = 0
            while "l_" + str(index) in basicoutput:
                index += 1
            allcoord = []
            for part in range(index):
                stage = "vlm_coord_%d" % part
                question = (
                    "Based on the structured description of l_"
                    + str(part)
                    + ", generate its 3D voxel grid in the following format (voxel grid=32, use numbers from 0 to 32767, merge maximal consecutive runs: 199...216 -> 199-216): 184 198 199-216 230-237..."
                )
                messages1 = demo.addmessage(messages, basicoutput, question)
                t0 = time.perf_counter()
                torch.manual_seed(1)
                output1 = demo.generate_save(model, messages1, str(case), "coord_" + str(part), save=True)
                torch.cuda.synchronize()
                parts.append(
                    dict(
                        name="coord_%d" % part,
                        seconds=time.perf_counter() - t0,
                        chars=len(output1),
                        tokens=len(processor.tokenizer(output1).input_ids),
                    )
                )
                idx_back = demo.dash_str_to_ints(output1)
                voxels_back = demo.voxel_decode(idx_back)
                allcoord.append(voxels_back)
                np.save(str(case / ("ind_" + str(part) + ".npy")), voxels_back)
                partply = trimesh.points.PointCloud(voxels_back)  # save_part_ply=True (official default)
                partply.export(str(case / ("ind_" + str(part) + ".ply")))
                log(fh, "%s part %d/%d voxels=%d %.1fs" % (sid, part, index, len(voxels_back), parts[-1]["seconds"]))
            stage = "vlm_allind"
            np.save(str(case / "allind.npy"), np.concatenate(allcoord))
        except Exception as e:  # the case fails; the lane continues
            error = "%s: %s" % (type(e).__name__, str(e)[:500])
            with open(case / "vlm_error.txt", "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
            log(fh, "FAILED %s at %s: %s" % (sid, stage, error))
            if "CUDA" in error or "out of memory" in error.lower():
                torch.cuda.empty_cache()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t_case
        timing = dict(
            vlm_seconds=seconds,
            vlm_peak_gpu_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            vlm_peak_gpu_reserved_bytes=int(torch.cuda.max_memory_reserved()),
            vlm_calls=parts,
            vlm_part_count=index,
            vlm_stage_at_end=stage,
            vlm_error=error,
            utc=utc(),
        )
        write_json_once(case / "timing_vlm.json", timing)
        if error is None:
            write_json_once(
                case / "vlm_result.json",
                dict(source_id=sid, method_id=METHOD_ID, vlm_ok=True, part_count=index, vlm_seconds=seconds, utc=utc()),
            )
            log(
                fh,
                "OK %s parts=%d %.1fs peak_alloc=%.1fGB"
                % (sid, index, seconds, torch.cuda.max_memory_allocated() / 1e9),
            )
        else:
            write_json_once(
                case / "result.json",
                dict(
                    source_id=sid,
                    method_id=METHOD_ID,
                    stage_reached="vlm",
                    asset_emitted=False,
                    part_count=index,
                    failure_reason="vlm_stage_failed_at_%s: %s" % (stage, error),
                    runtime_seconds=seconds,
                    utc=utc(),
                ),
            )
    log(fh, "stage A done")


if __name__ == "__main__":
    main()
