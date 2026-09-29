"""Freeze a success-only training dataset from scene_runner teacher episodes (no Isaac).

Row convention (matches the served policy): image[t] and proprio[t] are captured BEFORE action[t] is
applied; the label chunk is action[t..t+7], the executed 9-D joint targets. Episodes whose independent
evaluator result is not `passed` (or whose teacher raised) are excluded. Nothing is relabelled."""

import argparse, json, hashlib, math, random
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--runs", nargs="+", required=True)
p.add_argument("--output", required=True)
p.add_argument("--max-per-scene", type=int, default=10**9)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--default-scales", default="0.06,0.17,0.30,0.40,0.48,0.55,0.62,0.68")
p.add_argument("--chunk", type=int, default=8)
p.add_argument(
    "--max-horizon-delta",
    type=float,
    default=0.0,
    help="exclude passed episodes whose executed chunk-horizon joint delta exceeds this many rad (0 = off)",
)
p.add_argument(
    "--modes",
    default="teacher",
    help="comma-separated episode modes accepted (teacher, dagger); labels are always the teacher targets",
)
p.add_argument(
    "--include-failed-modes",
    default="",
    help="comma-separated modes whose FAILED episodes are also included (DAgger: policy-visited states with teacher labels); traces truncated by --failed-truncate-ticks, need >= --failed-min-ticks; object-off-table episodes still rejected",
)
p.add_argument("--failed-truncate-ticks", type=int, default=45)
p.add_argument("--failed-min-ticks", type=int, default=120)
a = p.parse_args()
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
DEF = [float(x) for x in a.default_scales.split(",")]
C = a.chunk


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


episodes = {}
rejected = {}
for root in a.runs:
    for ep in sorted(Path(root).glob("**/episode_*/episode.json")):
        e = json.loads(ep.read_text())
        scene = e["scene"]
        key = f"{scene}|{e['result'].get('status')}|{e.get('teacher_status')}"
        ok = (
            e.get("mode") in a.modes.split(",")
            and e["result"].get("status") == "passed"
            and e.get("teacher_status") == "completed"
        )
        failed_ok = False
        if (
            not ok
            and e.get("mode") in [m for m in a.include_failed_modes.split(",") if m]
            and e.get("teacher_status") in ("completed", "teacher_exception", "budget_exhausted")
        ):
            fell = any(g.get("kind") == "object_on_table" and not g.get("ok") for g in e["result"].get("goals", []))
            failed_ok = (not fell) and int(e.get("ticks", 0)) >= a.failed_min_ticks + a.failed_truncate_ticks
        if not (ok or failed_ok):
            rejected[key] = rejected.get(key, 0) + 1
            continue
        e["_failed_included"] = bool(failed_ok and not ok)
        episodes.setdefault(scene, []).append((ep.parent, e))
rng = random.Random(a.seed)
rows = []
per_scene = {}
needed = np.zeros(C)
exceed = np.zeros(C)
total = 0
proprio_all = []
excluded = []
wrist_rows = [0, 0]
per_mode = {}
policy_ticks = 0
failed_included = 0
with (out / "frame_index.jsonl").open("w") as fi, (out / "episodes.jsonl").open("w") as fe:
    for scene in sorted(episodes):
        eps = episodes[scene]
        rng.shuffle(eps)
        eps = eps[: a.max_per_scene]
        per_scene[scene] = 0
        for d, e in sorted(eps, key=lambda x: str(x[0])):
            trace = [json.loads(l) for l in (d / "trace.jsonl").open()]
            if e.get("_failed_included"):
                trace = trace[: len(trace) - a.failed_truncate_ticks]
                failed_included += 1
            n = len(trace)
            acts = np.array([t["action"] for t in trace], np.float32)
            props = np.array([t["observation"]["proprio"] for t in trace], np.float32)
            ep_need = np.zeros(C)
            ep_exceed = np.zeros(C)
            worst = (0.0, 0, 0)
            for i in range(n):
                for k in range(C):
                    j = min(i + k, n - 1)
                    m = float(np.max(np.abs(acts[j, :7] - props[i, :7])))
                    ep_need[k] = max(ep_need[k], m)
                    if m > worst[0]:
                        worst = (m, i, k)
                    if m > DEF[k]:
                        ep_exceed[k] += 1
            if a.max_horizon_delta > 0 and float(ep_need.max()) > a.max_horizon_delta:
                key = f"{scene}|passed|horizon_delta_gt_{a.max_horizon_delta}"
                rejected[key] = rejected.get(key, 0) + 1
                excluded.append(
                    dict(
                        episode_job=str(d.resolve()),
                        scene=scene,
                        seed=e["seed"],
                        max_delta_rad=round(worst[0], 4),
                        tick=worst[1],
                        horizon=worst[2] + 1,
                        phase=trace[worst[1]]["phase"],
                        per_horizon=ep_need.round(4).tolist(),
                    )
                )
                continue
            needed = np.maximum(needed, ep_need)
            exceed += ep_exceed
            per_scene[scene] += 1
            total += n
            proprio_all.append(props)
            job = str(d.resolve())
            fe.write(
                json.dumps(
                    dict(
                        episode_job=job,
                        scene=scene,
                        seed=e["seed"],
                        ticks=n,
                        instruction=e["instruction"],
                        result=e["result"]["status"],
                        mode=e.get("mode"),
                        failed_included=bool(e.get("_failed_included")),
                        assets=e["assets"],
                        objects=e["objects"],
                    )
                )
                + "\n"
            )
            for i, t in enumerate(trace):
                path = Path(t["observation"]["path"])
                if not path.exists():
                    raise FileNotFoundError(str(path))
                wp = t["observation"].get("path_wrist")
                row = dict(
                    episode_job=job,
                    scene=scene,
                    index=i,
                    path=str(path.resolve()),
                    sha256=sha(path),
                    proprio=t["observation"]["proprio"],
                    action=t["action"],
                    instruction=e["instruction"],
                    phase=t["phase"],
                    mode=e.get("mode"),
                    dagger_burst=bool((t.get("dagger") or {}).get("burst", False)),
                    failed_episode=bool(e.get("_failed_included")),
                )
                if wp:
                    wp = Path(wp)
                    if not wp.exists():
                        raise FileNotFoundError(str(wp))
                    row["path_wrist"] = str(wp.resolve())
                    row["sha256_wrist"] = sha(wp)
                    wrist_rows[0] += 1
                else:
                    wrist_rows[1] += 1
                fi.write(json.dumps(row) + "\n")
            per_mode[e.get("mode")] = per_mode.get(e.get("mode"), 0) + 1
            if e.get("dagger"):
                policy_ticks += int(e["dagger"].get("policy_ticks", 0))
if wrist_rows[0] and wrist_rows[1]:
    raise SystemExit("mixed_wrist_and_no_wrist_frames:%d/%d" % (wrist_rows[0], wrist_rows[1]))
scales = [max(DEF[k], float(math.ceil(needed[k] * 1.05 * 1000) / 1000)) for k in range(C)]
P = np.concatenate(proprio_all) if proprio_all else np.zeros((1, 9))
mean = P.mean(0)
std = np.maximum(P.std(0), np.array([0.1] * 7 + [0.01, 0.01]))
manifest = dict(
    episode_count=sum(per_scene.values()),
    per_scene=per_scene,
    frame_count=total,
    rejected=rejected,
    horizon_arm_scale_rad=scales,
    default_scales=DEF,
    observed_max_abs_delta_per_horizon=needed.round(4).tolist(),
    frames_exceeding_default_scale_per_horizon=exceed.astype(int).tolist(),
    proprio_mean=mean.round(6).tolist(),
    proprio_std=std.round(6).tolist(),
    chunk=C,
    execute_prefix=4,
    control_hz=10,
    label_convention="observation before action; action[t..t+7] executed 9-D joint targets",
    success_only=(not a.include_failed_modes),
    failed_episodes_included=failed_included,
    include_failed_modes=[m for m in a.include_failed_modes.split(",") if m],
    failed_truncate_ticks=a.failed_truncate_ticks,
    failed_min_ticks=a.failed_min_ticks,
    max_horizon_delta_rad=a.max_horizon_delta,
    excluded_dynamics=excluded,
    num_images=(2 if wrist_rows[0] and not wrist_rows[1] else 1),
    wrist_frames=wrist_rows[0],
    frames_without_wrist=wrist_rows[1],
    modes=a.modes.split(","),
    episodes_per_mode=per_mode,
    policy_executed_ticks=policy_ticks,
    dataset_files_sha256={n: sha(out / n) for n in ("frame_index.jsonl", "episodes.jsonl")},
    runs=[str(Path(r).resolve()) for r in a.runs],
    robot_real_world="not_evaluated",
    robot_control_status="simulated_robot_only",
)
(out / "manifest.json").write_text(json.dumps(manifest, indent=1))
print(json.dumps({k: v for k, v in manifest.items() if k != "dataset_files_sha256"}))
