"""Small pure-python steps of the scene training pipeline (called by post_train.sh / eval.sh):
config (derive the round configuration), audit (video/trace coverage of a collection tag), freeze (dataset receipt and
coverage/scale gates), complete (completion.json of a trained round), eval_summary (task and first-sub-goal counts)."""

import hashlib, json, sys, time
from pathlib import Path

cmd = sys.argv[1]
if cmd == "config":
    src, dst, framework, steps = sys.argv[2:6]
    d = json.load(open(src))
    ver = Path(framework).name.split("_v")[-1]
    nimg = int(sys.argv[6]) if len(sys.argv) > 6 else 2
    d.update(
        {
            "method_id": f"affordcraft_scenes_vla_v{ver}_wrist_crop_dagger",
            "parent_method_id": d.get("method_id"),
            "scene_framework": framework,
            "collection_tag": f"collect{ver}",
            "num_images": nimg,
            "feature_dim": 4096 * nimg,
            "observation": {
                "third_person": "1280x960 render, workspace crop x 40-1040 / y 60-810, resized to 320x240",
                "wrist": "camera on panda_hand (mount 0.055,0,0.02 m, pitch 25 deg, 70 deg FOV), 320x240",
            },
            "feature_extraction": "frozen OpenVLA-OFT (libero_spatial) v0.166 action-token mean per image, concatenated over images",
            "dagger": "round >=1 datasets add mixed rollouts: teacher plan with served-policy bursts (beta 0.35, 8-40 ticks, prefix <=200); labels = teacher clamped targets",
            "camera": {
                "render": "1280x960",
                "horizontal_fov_deg": 42.0,
                "azimuth_deg": -140.0,
                "elevation_deg": 30.0,
                "distance_m": 2.6,
                "target": [-0.25, 0.0, 1.20],
                "policy_observation": "same view downsampled to 320x240",
            },
            "contact_report_api": "scene rigid bodies only (robot links excluded: v19g, avoids the RTX distal-link rendering drop)",
            "teacher_command_clamp_rad_per_tick": 0.06,
            "video_required": True,
            "raw_isaac_rgb_required": True,
            "success_predicate": "independent_scene_eval_status_passed",
            "max_steps": int(steps),
            "feature_interpreter": "${OPENVLA_OFT_PYTHON} (timm 0.9.10, transformers 4.40.1, torch 2.7.1+cu128)",
            "created_time": time.time(),
            "robot_real_world": "not_evaluated",
            "robot_control_status": "simulated_robot_only",
            "robot_task_success": None,
        }
    )
    json.dump(d, open(dst, "w"), indent=1)
    print("config written", dst)
elif cmd == "audit":
    root = Path(sys.argv[2])
    out = Path(sys.argv[3])
    rows = []
    counts = {}
    for ep in sorted(root.glob("*_w*/episode_*/episode.json")):
        d = json.loads(ep.read_text())
        result = d.get("result", {})
        status = result.get("status")
        row = {
            "episode": str(ep),
            "scene": d.get("scene"),
            "seed": d.get("seed"),
            "status": status,
            "failure_reason": result.get("failure_reason"),
            "trace": (ep.parent / "trace.jsonl").is_file(),
            "video": (ep.parent / "review.mp4").is_file(),
            "frames": len(list((ep.parent / "frames").glob("*.png"))),
            "frames_wrist": len(list((ep.parent / "frames_wrist").glob("*.png"))),
            "teacher_status": d.get("teacher_status"),
            "ticks": d.get("ticks"),
            "mode": d.get("mode"),
            "dagger": d.get("dagger"),
        }
        row["eligible_success"] = (
            status == "passed"
            and d.get("teacher_status") == "completed"
            and row["trace"]
            and row["video"]
            and row["frames"] > 1
        )
        rows.append(row)
        c = counts.setdefault(row["scene"], {"success": 0, "failed": 0, "eligible": 0})
        c["success" if status == "passed" else "failed"] += 1
        c["eligible"] += int(row["eligible_success"])
    json.dump(
        {
            "created": time.time(),
            "root": str(root),
            "episodes": len(rows),
            "per_scene": counts,
            "rows": rows,
            "all_success_episodes_have_video_and_trace": all(
                x["eligible_success"] for x in rows if x["status"] == "passed"
            ),
        },
        open(out, "w"),
        indent=1,
    )
    print(json.dumps({"episodes": len(rows), "per_scene": counts}))
    if len(counts) != 10 or any(v["eligible"] == 0 for v in counts.values()):
        raise SystemExit("video_or_success_coverage_blocked")
elif cmd == "freeze":
    p = sys.argv[2]
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    m = json.load(open(p))
    json.dump(
        {
            "created": time.time(),
            "manifest": p,
            "manifest_sha256": h,
            "episode_count": m["episode_count"],
            "frame_count": m["frame_count"],
            "per_scene": m["per_scene"],
            "horizon_arm_scale_rad": m["horizon_arm_scale_rad"],
            "observed_max_abs_delta_per_horizon": m["observed_max_abs_delta_per_horizon"],
            "success_only": m["success_only"],
            "max_horizon_delta_rad": m.get("max_horizon_delta_rad"),
            "excluded_dynamics_count": len(m.get("excluded_dynamics", [])),
            "rejected": m.get("rejected"),
            "video_audit": "video_audit.json",
        },
        open(sys.argv[3], "w"),
        indent=1,
    )
    print("frozen", m["episode_count"], "episodes", m["frame_count"], "frames; scales", m["horizon_arm_scale_rad"])
    if len(m["per_scene"]) != 10 or any(v < 1 for v in m["per_scene"].values()):
        raise SystemExit("dataset_scene_coverage_blocked")
    if max(m["horizon_arm_scale_rad"]) > 1.0:
        raise SystemExit("horizon_scale_inflated:%s" % m["horizon_arm_scale_rad"])
elif cmd == "complete":
    out = Path(sys.argv[2])
    cm = json.load(open(out / "training/checkpoint_manifest.json"))
    json.dump(
        {
            "completed": True,
            "created": time.time(),
            "dataset_manifest": json.load(open(out / "dataset/manifest.json")),
            "checkpoint_manifest": cm,
            "video_audit_summary": {k: v for k, v in json.load(open(out / "video_audit.json")).items() if k != "rows"},
            "robot_real_world": "not_evaluated",
            "robot_control_status": "simulated_robot_only",
            "robot_task_success": None,
        },
        open(out / "completion.json", "w"),
        indent=1,
    )
    print("completion.json written")
elif cmd == "eval_summary":
    out = Path(sys.argv[2])
    sub = sys.argv[3] if len(sys.argv) > 3 else "eval"
    report = {
        "created": time.time(),
        "eval_dir": str(out / sub),
        "variants": {},
        "robot_real_world": "not_evaluated",
        "robot_control_status": "simulated_robot_only",
        "robot_task_success": None,
    }
    for var in ("untrained_vla", "trained_vla"):
        rows = []
        per = {}
        for p in sorted((out / sub / var).glob("*/episode_*/episode.json")):
            d = json.loads(p.read_text())
            r = {
                "scene": d["scene"],
                "episode": d["index"],
                "seed": d["seed"],
                "status": d["result"]["status"],
                "failure_reason": d["result"].get("failure_reason"),
                "ticks": d.get("ticks"),
                "video": (p.parent / "review.mp4").is_file(),
                "trace": (p.parent / "trace.jsonl").is_file(),
                "teacher_status": d.get("teacher_status"),
            }
            goals = d["result"].get("goals", [])
            r["goals_ok"] = [bool(g.get("ok")) for g in goals]
            r["stage1"] = bool(goals and goals[0].get("ok"))
            rows.append(r)
            c = per.setdefault(d["scene"], [0, 0, 0])
            c[1] += 1
            c[0] += r["status"] == "passed"
            c[2] += int(r["stage1"])
        report["variants"][var] = {
            "episodes": len(rows),
            "passed": sum(r["status"] == "passed" for r in rows),
            "stage1_reached": sum(r["stage1"] for r in rows),
            "per_scene": {k: f"{v[0]}/{v[1]}" for k, v in per.items()},
            "per_scene_stage1": {k: f"{v[2]}/{v[1]}" for k, v in per.items()},
            "all_have_video": all(r["video"] for r in rows),
            "rows": rows,
        }
    (out / ("eval_summary.json" if sub == "eval" else f"eval_summary_{sub}.json")).write_text(
        json.dumps(report, indent=1)
    )
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in report["variants"].items()}))
else:
    raise SystemExit("unknown step " + cmd)
