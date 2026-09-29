#!/usr/bin/env python
"""check_library_renders.py -- Articulate-Anything library check before the second attempt of the run.
The first 26 cases of the first attempt include 5 failures 'PIL.UnidentifiedImageError: cannot identify image file
datasets/partnet-mobility-v0/dataset/<obj>/robot_frontview.png' (objects 45443, 101399): the library completion
(render_library_gaps.py) removed only zero-byte renders, so a truncated non-empty render stayed in the retrieval library.
This script opens every robot_frontview.png of the library with PIL (open + load); an unreadable one is moved to a
quarantine directory (kept) and rendered again with the same official call render_library_gaps.py uses
(render_partnet_obj(obj, <AA_GPU>, cfg, 'stationary'), SAPIEN-3 render port); an object whose new render is still unreadable
stays without a PNG and is therefore excluded from retrieval, as in the official get_candidate_objs().
Log: $AA_WORK/logs/library_png_fix_<ts>.jsonl. Run from the patched source tree (cwd = AA_SRC) with
AA_OPENAI_COMPAT=1 and AA_RENDER_SCRIPT set, like render_library_gaps.py. usage: check_library_renders.py [--check-only]
"""
import json, os, shutil, sys, time, traceback

AA_WORK = os.path.abspath(os.environ.get("AA_WORK", "aa_run"))
LIB = os.path.join(os.path.abspath(os.environ.get("ARTICULATE_ANYTHING_CKPT", "partnet-mobility-v0")), "dataset")
SRC = os.environ.get("AA_SRC", os.path.join(AA_WORK, "src"))
TS = time.strftime("%Y%m%d-%H%M%S")
LOG = os.path.join(AA_WORK, "logs/library_png_fix_%s.jsonl" % TS)
QUAR = os.path.join(AA_WORK, "logs/library_png_quarantine_%s" % TS)


def readable(png):
    from PIL import Image

    try:
        with Image.open(png) as im:
            im.load()
        return True, None
    except Exception as e:
        return False, repr(e)[:200]


def main():
    check_only = "--check-only" in sys.argv
    objs = sorted(
        (o for o in os.listdir(LIB) if os.path.isdir(os.path.join(LIB, o))), key=lambda x: int(x) if x.isdigit() else 0
    )
    bad = []
    for o in objs:
        png = os.path.join(LIB, o, "robot_frontview.png")
        if os.path.exists(png):
            ok, err = readable(png)
            if not ok:
                bad.append((o, os.path.getsize(png), err))
    print(
        json.dumps(
            {
                "objects": len(objs),
                "with_png": sum(os.path.exists(os.path.join(LIB, o, "robot_frontview.png")) for o in objs),
                "unreadable": [b[0] for b in bad],
            }
        ),
        flush=True,
    )
    if check_only or not bad:
        return
    os.chdir(SRC)
    sys.path.insert(0, SRC)
    from articulate_anything.utils.utils import load_config
    from articulate_anything.preprocess.preprocess_partnet import render_partnet_obj

    cfg = load_config()
    cfg.dataset_dir = LIB
    os.environ["AA_HYDRA_RUN_DIR"] = os.path.join(AA_WORK, "logs/hydra_library_render")
    os.makedirs(QUAR, exist_ok=True)
    with open(LOG, "a") as fh:
        for o, size, err in bad:
            png = os.path.join(LIB, o, "robot_frontview.png")
            rec = dict(
                obj=o,
                old_size=size,
                old_error=err,
                quarantined=os.path.join(QUAR, o + "_robot_frontview.png"),
                t0=time.time(),
            )
            shutil.move(png, rec["quarantined"])
            try:
                render_partnet_obj(o, os.environ.get("AA_GPU", "0"), cfg, "stationary")
                rec["ok"] = os.path.exists(png) and readable(png)[0]
            except Exception:
                rec["ok"] = False
                rec["error"] = traceback.format_exc()[-1500:]
            if not rec["ok"] and os.path.exists(png):
                shutil.move(
                    png, os.path.join(QUAR, o + "_robot_frontview_rerender.png")
                )  # unreadable again: excluded from retrieval
            rec["seconds"] = round(time.time() - rec["t0"], 1)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(o, "ok" if rec["ok"] else "EXCLUDED", rec["seconds"], flush=True)
    print("FIX_DONE", LOG)


if __name__ == "__main__":
    main()
