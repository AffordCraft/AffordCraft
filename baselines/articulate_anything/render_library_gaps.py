#!/usr/bin/env python
"""render_library_gaps.py <shard> <shards> -- complete the official Articulate-Anything image preprocessing for the
PartNet-Mobility library objects that have no (or an empty) robot_frontview.png: the same call the official
preprocess_partnet.py uses for modality=image, render_partnet_obj(obj_id, gpu_id, cfg, "stationary") (rotate_urdf ->
render -> combine_meshes), with the SAPIEN-3 render port. Zero-byte PNGs are removed before rendering (logged). Objects
whose render still fails stay without a PNG and are therefore excluded from retrieval, exactly as in the official
get_candidate_objs(). Run from the patched source tree (cwd) with AA_OPENAI_COMPAT=1 and AA_RENDER_SCRIPT set."""
import json
import os
import sys
import time
import traceback

AA_WORK = os.path.abspath(os.environ.get("AA_WORK", "aa_run"))
LIB = os.path.join(os.path.abspath(os.environ.get("ARTICULATE_ANYTHING_CKPT", "partnet-mobility-v0")), "dataset")
SRC = os.environ.get("AA_SRC", os.path.join(AA_WORK, "src"))
GAPS = os.path.join(AA_WORK, "logs/library_render_gaps.json")
LOG = os.path.join(AA_WORK, "logs/library_render_gaps_shard%d.jsonl")


def main():
    shard, shards = int(sys.argv[1]), int(sys.argv[2])
    os.chdir(SRC)
    sys.path.insert(0, SRC)
    from articulate_anything.utils.utils import load_config
    from articulate_anything.preprocess.preprocess_partnet import render_partnet_obj

    cfg = load_config()
    cfg.dataset_dir = LIB
    gaps = json.load(open(GAPS))
    ids = sorted(set(gaps["bad"] + gaps["missing"]), key=int)
    ids = [o for i, o in enumerate(ids) if i % shards == shard]
    os.environ["AA_HYDRA_RUN_DIR"] = os.path.join(AA_WORK, "logs/hydra_library_render")
    with open(LOG % shard, "a") as fh:
        for obj in ids:
            png = os.path.join(LIB, obj, "robot_frontview.png")
            rec = dict(obj=obj, category=gaps["categories"].get(obj), t0=time.time(), removed_empty_png=False)
            if os.path.exists(png) and os.path.getsize(png) == 0:
                os.remove(png)
                rec["removed_empty_png"] = True
            try:
                render_partnet_obj(obj, os.environ.get("AA_GPU", "0"), cfg, "stationary")
                rec["ok"] = os.path.exists(png) and os.path.getsize(png) > 0
            except Exception:
                rec["ok"] = False
                rec["error"] = traceback.format_exc()[-1500:]
            rec["seconds"] = round(time.time() - rec["t0"], 1)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(obj, rec["category"], "ok" if rec["ok"] else "FAILED", rec["seconds"], flush=True)
    print("SHARD_DONE", shard)


if __name__ == "__main__":
    main()
