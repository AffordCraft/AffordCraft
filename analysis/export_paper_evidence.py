"""Export paper evidence from the main campaign and the one-factor study. Read-only over campaign directories; writes one
JSON file. Partial campaigns are exported with complete=False and are never paper results.

usage: export_paper_evidence.py --main <main campaign dir> [--study <one-factor study dir>]
       --inputs <single_inputs.json> --scene-inputs <detected_inputs.json> [--index-dir <visual index>] --output evidence_v2.json
"""

from pathlib import Path
import argparse, json, math, statistics, time
from collections import Counter, defaultdict

OUTCOMES = ("physical_pass", "physics_validation_failed", "no_structured_export", "infrastructure_blocked", "not_run")
PHASE_GROUP = {
    "grounding": "perception",
    "retrieval": "perception",
    "multimodal_ranking": "perception",
    "asset_construction": "adaptation",
    "repair": "adaptation",
    "construction_physics": "validation",
    "final_physics": "validation",
}


def wilson(k, n):
    if not n:
        return None
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"success": k, "total": n, "rate": p, "lower": max(0.0, mid - half), "upper": min(1.0, mid + half)}


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return rows


def quantile(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    i = (len(xs) - 1) * q
    lo = math.floor(i)
    hi = math.ceil(i)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def attempts_of(campaign, kind=None):
    state = read(Path(campaign) / "queue_state.json", {"jobs": []})
    for job in state["jobs"]:
        if kind and job["kind"] != kind:
            continue
        for a in job["attempts"]:
            yield job, a


def case_rows(campaign, kind):
    """Terminal per-case rows: the verified final_results when present, else live shard outputs (partial)."""
    root = Path(campaign)
    final = root / "final_results"
    name = {"single": "exp1_per_case.jsonl", "scene": "exp2_per_prediction.jsonl"}[kind]
    if (final / name).exists():
        return jsonl(final / name), True
    rows = {}
    for job, a in attempts_of(campaign, kind):
        out = Path(a["output"])
        for ref in read(out / "inherited_terminal_references.json", []) or []:
            rows[ref["record"]["input_id"]] = ref["record"]
        for r in jsonl(out / "per_input.jsonl"):
            rows[r["input_id"]] = r
    return list(rows.values()), False


def outcome(r):
    if r.get("physical_pass") is True:
        return "physical_pass"
    if r.get("status") in ("infrastructure_blocked", "not_run"):
        return r["status"]
    if r.get("selected_asset") or r.get("status") == "final_validation_failed":
        return "physics_validation_failed"
    return "no_structured_export"


def summarize(rows, noun, denominator):
    by_cat = defaultdict(Counter)
    roles = Counter()
    reasons = Counter()
    statuses = Counter()
    attempt_status = Counter()
    timings = defaultdict(list)
    e2e = []
    per_group = defaultdict(list)
    considered = []
    repairs = 0
    repair_rescues = 0
    for r in rows:
        o = outcome(r)
        c = noun.get(r["input_id"], "?")
        by_cat[c][o] += 1
        statuses[str(r.get("status"))] += 1
        if o == "physical_pass":
            roles[(r.get("selected_asset") or {}).get("support_role", "?")] += 1
            selected = [a for a in r.get("attempts", []) if a.get("status") == "selected"]
            if selected and selected[-1].get("repairs"):
                repair_rescues += 1
        elif o == "physics_validation_failed":
            for x in (r.get("final_validation") or {}).get("failure_reasons", []):
                reasons["final:" + x] += 1
        else:
            reasons[str(r.get("failure_reason"))] += 1
        for a in r.get("attempts", []):
            attempt_status[str(a.get("status"))] += 1
            repairs += int(bool(a.get("repairs")))
        groups = Counter()
        for p in r.get("phase_timings", []):
            timings[p["phase"]].append(p["runtime_seconds"])
            groups[PHASE_GROUP.get(p["phase"], "other")] += p["runtime_seconds"]
        for g, v in groups.items():
            per_group[g].append(v)
        if r.get("total_wall_seconds") is not None:
            e2e.append(r["total_wall_seconds"])
        if r.get("considered_candidates") is not None:
            considered.append(r["considered_candidates"])
    noun_counts = Counter(noun.values())
    k = sum(v["physical_pass"] for v in by_cat.values())
    n_done = len(rows)
    cats = {
        c: {
            "n": noun_counts.get(c, 0),
            "done": sum(by_cat[c].values()),
            **{o: by_cat[c][o] for o in OUTCOMES},
            "wilson95": wilson(by_cat[c]["physical_pass"], noun_counts.get(c, 0)),
        }
        for c in sorted(noun_counts)
    }
    return {
        "denominator": denominator,
        "completed_cases": n_done,
        "physical_pass": k,
        "wilson95_over_denominator": wilson(k, denominator),
        "wilson95_over_completed": wilson(k, n_done),
        "outcome_counts": dict(Counter(outcome(r) for r in rows)),
        "status_counts": dict(statuses),
        "attempt_status_counts": dict(attempt_status),
        "repairs_attempted": repairs,
        "passes_whose_selected_asset_was_repaired": repair_rescues,
        "selected_support_roles": dict(roles),
        "failure_reasons_nonexclusive": dict(reasons.most_common()),
        "by_category": cats,
        "timing_seconds": {
            "per_phase_median": {p: statistics.median(v) for p, v in timings.items()},
            "per_phase_mean": {p: statistics.mean(v) for p, v in timings.items()},
            "per_case_group_median": {g: statistics.median(v) for g, v in per_group.items()},
            "per_case_group_mean": {g: statistics.mean(v) for g, v in per_group.items()},
            "e2e_median": statistics.median(e2e) if e2e else None,
            "e2e_mean": statistics.mean(e2e) if e2e else None,
            "e2e_p95": quantile(e2e, 0.95),
            "considered_candidates_mean": statistics.mean(considered) if considered else None,
            "note": "throughput run with concurrent workers per GPU and a pre-warmed geometry cache; not an exclusive-device latency measurement",
        },
    }


def resources(campaign):
    """Worker wall time, model-load time and peak GPU memory of each worker's own process tree."""
    total = 0.0
    loads = []
    peaks = []
    for job, a in attempts_of(campaign):
        out = Path(a["output"])
        pr = read(out / "progress.json", {}) or {}
        total += pr.get("elapsed_seconds", 0)
        ml = read(out / "model_load_timing.json", {}) or {}
        if ml.get("measured_runtime_seconds"):
            loads.append(ml["measured_runtime_seconds"])
        pids = set()
        main = read(Path(a["receipt_dir"]) / "process.json", {}) or {}
        if main.get("pid"):
            pids.add(int(main["pid"]))
        for phase in ("construction", "final"):
            proc = read(out / "services" / phase / "process.json", {}) or {}
            if proc.get("pid"):
                pids.add(int(proc["pid"]))
        peak = 0
        for row in jsonl(Path(a["receipt_dir"]) / "resources.jsonl"):
            used = 0
            for line in (row.get("processes") or {}).get("rows", []):
                parts = [s.strip() for s in line.split(",")]
                if len(parts) == 4:
                    try:
                        if int(parts[1]) in pids:
                            used += float(parts[3])
                    except ValueError:
                        pass
            peak = max(peak, used)
        if pids and peak:
            peaks.append(peak)
    return {
        "worker_wall_hours": total / 3600,
        "model_load_seconds_median": statistics.median(loads) if loads else None,
        "peak_gpu_memory_mib_per_worker_tree_max": max(peaks) if peaks else None,
        "peak_gpu_memory_mib_per_worker_tree_median": statistics.median(peaks) if peaks else None,
        "gpu_note": "each worker holds one A100 allocation for its whole run (Qwen3-VL-8B + two PhysX processes); memory sums the worker tree across all visible devices",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--study")
    ap.add_argument("--output", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--scene-inputs", required=True)
    ap.add_argument("--index-dir")
    a = ap.parse_args()
    single_inputs = json.loads(Path(a.inputs).read_text())["inputs"]
    noun = {r["input_id"]: r["target_noun"] for r in single_inputs}
    scene_inputs = json.loads(Path(a.scene_inputs).read_text())["inputs"]
    scene_noun = {r["input_id"]: r["target_noun"] for r in scene_inputs}
    cfg = read(Path(a.main) / "configuration.json", {})
    summary = read(Path(a.main) / "final_results/summary.json")
    rows, verified = case_rows(a.main, "single")
    exp1 = summarize(rows, noun, len(single_inputs))
    exp1["verified_final_results"] = verified
    srows, sverified = case_rows(a.main, "scene")
    exp2 = summarize(srows, scene_noun, len(scene_inputs))
    exp2["verified_final_results"] = sverified
    ev = read(Path(a.main) / "final_results/exp2_evaluation.json")
    if ev:
        exp2["reference_evaluation"] = {k: v for k, v in ev.items() if k != "scenes"}
    res = resources(a.main)
    cases = exp1["completed_cases"] + exp2["completed_cases"]
    res["worker_wall_seconds_per_case"] = res["worker_wall_hours"] * 3600 / cases if cases else None
    passes = exp1["physical_pass"] + exp2["physical_pass"]
    res["worker_gpu_hours_per_pass"] = res["worker_wall_hours"] / passes if passes else None
    if a.index_dir:
        idx = read(Path(a.index_dir) / "index.json", {})
        res["offline_index_wall_seconds"] = idx.get("offline_wall_seconds")
        res["offline_index_entries"] = idx.get("eligible_entries")
    out = {
        "schema": "affordcraft.paper.evidence.v2",
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "main_campaign": {
            "path": str(a.main),
            "method_version": cfg.get("method_version"),
            "complete": bool(summary and summary.get("complete")),
            "verifier_summary": summary,
            "build_threads": cfg.get("build_threads"),
            "construction": cfg.get("construction"),
            "resources": res,
            "exp1": exp1,
            "exp2": exp2,
        },
        "study": None,
        "robot_real_world": "not_evaluated",
        "robot_control_status": "not_evaluated",
        "robot_task_success": None,
        "task_match": "not_evaluated",
    }
    if a.study:
        ssum = read(Path(a.study) / "study_results/summary.json")
        scfg = read(Path(a.study) / "configuration.json", {})
        live = {}
        if not ssum:
            for job, at in attempts_of(a.study):
                pr = read(Path(at["output"]) / "progress.json", {}) or {}
                d = live.setdefault(
                    str(job.get("track")) + "/" + str(job.get("variant")), {"completed": 0, "physical_passes": 0}
                )
                d["completed"] += pr.get("completed", 0)
                d["physical_passes"] += pr.get("physical_passes", 0)
        out["study"] = {
            "path": str(a.study),
            "method_version": scfg.get("method_version"),
            "complete": bool(ssum and ssum.get("complete")),
            "verifier_summary": ssum,
            "live_partial_counts_not_results": live,
            "tracks": scfg.get("tracks"),
        }
    Path(a.output).write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": a.output,
                "exp1_done": exp1["completed_cases"],
                "exp1_pass": exp1["physical_pass"],
                "exp2_done": exp2["completed_cases"],
                "exp2_pass": exp2["physical_pass"],
                "main_complete": out["main_campaign"]["complete"],
                "study_complete": (out["study"] or {}).get("complete"),
                "worker_hours": round(res["worker_wall_hours"], 2),
            }
        )
    )


if __name__ == "__main__":
    main()
