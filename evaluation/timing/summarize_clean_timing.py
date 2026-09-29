#!/usr/bin/env python3
"""Clean-GPU timing: summary over the 32-input timing sample. Runs after every machine's JSON/log/monitor files were
collected into the run root. Writes summary/clean_timing_summary_<stamp>.json and summary/clean_timing_summary_latest.json.

Ours (per_input.jsonl of the cold and warm regimes, written by scripts/run_pipeline.py): per case the phase runtimes
grouped as in the campaign metrics (perception = grounding + retrieval + multimodal_ranking; adaptation =
asset_construction + repair; validation = construction_physics + final_physics; group medians over the cases that ran
the group), E2E = total_wall_seconds, GPU minutes per pass = sum(E2E) / passes. Checked against the main campaign's
per-case file: this reproduces its 31.6 / 67.3 / 6.5 / 85.0.
External methods: the campaign aggregator (evaluation/aggregate/aggregate_external.py) is imported and pointed at this
run root, so stage medians, E2E (native / adapted path), peak memory and verdict precedence follow the same definitions;
gate verdicts come from gates/<machine>/<method>/cases. Lane walls (lane.wall.json, model loading included) give the
worker seconds per input of one exclusive GPU. Campaign comparison: the same 32 inputs in the campaign records.
GPU memory and energy: 1-s nvidia-smi samples monitor/<host>_gpu.csv of the machine named in each wall record.
Throughput for the paper: passes per worker-hour = campaign pass rate x 3600 / clean mean seconds per input.
usage: summarize_clean_timing.py --root <run root> --campaign-per-case <exp1_per_case.jsonl>
       --campaign-aggregate <aggregate of the external runs> [--gate-machines m1 m2 ...] [--finalization FIN.json]"""
import argparse, json, glob, os, sys, math, statistics as st, time, csv
from pathlib import Path

RT = Path(os.environ.get("AFFORDCRAFT_RUNS", "runs")) / "clean_timing"
CAMPAIGN_PER_CASE = None  # exp1 per-case records of the main campaign (final_results/exp1_per_case.jsonl)
CAMPAIGN_AGGREGATE = None  # aggregate of the external runs over the registered inputs (aggregate_external.py output)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aggregate"))
import aggregate_external as AX  # noqa: E402

SAMPLE = []
GROUPS = {
    "perception": ("grounding", "retrieval", "multimodal_ranking"),
    "adaptation": ("asset_construction", "repair"),
    "validation": ("construction_physics", "final_physics"),
}
EXT = {
    "physx-anything-v0.1": "physx-anything",
    "pact-v0.1": "pact",
    "partcrafter-v0.1": "partcrafter",
    "trellis2-4b-v0.1": "trellis2",
}
# gate machine directories gates/<machine>/ in verdict precedence order (--gate-machines); default: sorted names
GATE_MACHINES = ()


def med(v):
    return round(st.median(v), 1) if v else None


def mean(v):
    return round(st.mean(v), 1) if v else None


def p95(v):
    return round(sorted(v)[max(0, int(math.ceil(0.95 * len(v))) - 1)], 1) if v else None


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def walls(pattern):
    out = []
    for f in sorted(glob.glob(str(RT / pattern))):
        d = load(f)
        if d:
            d["_file"] = os.path.relpath(f, RT)
            out.append(d)
    return out


def ours(reg):
    rows = []
    for f in sorted(glob.glob(str(RT / f"ours-{reg}/shard*/per_input.jsonl"))):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                r["_shard"] = f.split("/")[-2]
                rows.append(r)
    seen = {}
    for r in rows:
        seen.setdefault(r["input_id"], r)
    rows = [seen[s] for s in SAMPLE if s in seen]
    if not rows:
        return None
    g = {}
    for name, ph in GROUPS.items():
        v = [
            sum(p["runtime_seconds"] for p in r.get("phase_timings") or [] if p["phase"] in ph)
            for r in rows
            if any(p["phase"] in ph for p in r.get("phase_timings") or [])
        ]
        g[name] = {"median": med(v), "mean": mean(v), "n": len(v)}
    w = [r["total_wall_seconds"] for r in rows if r.get("total_wall_seconds") is not None]
    passes = sum(1 for r in rows if r.get("physical_pass"))
    sw = walls(f"ours-{reg}/shard*.wall.json")
    shard_wall = sum(d["end_unix"] - d["start_unix"] for d in sw)
    prep = [load(p) for p in sorted(glob.glob(str(RT / f"ours-{reg}/prep-shard*.json")))]
    return {
        "n": len(rows),
        "of": len(SAMPLE),
        "passes": passes,
        "status_counts": {s: sum(1 for r in rows if r["status"] == s) for s in sorted({r["status"] for r in rows})},
        "group": g,
        "e2e_median": med(w),
        "e2e_mean": mean(w),
        "e2e_p95": p95(w),
        "e2e_max": round(max(w), 1) if w else None,
        "gpu_min_per_pass_sample": round(sum(w) / 60 / passes, 1) if passes else None,
        "shards": [
            dict(
                {k: d.get(k) for k in ("shard", "gpu", "host", "start_unix", "end_unix", "exit_code", "_file")},
                monitor=window(d.get("host"), d["gpu"], d["start_unix"], d["end_unix"]),
            )
            for d in sw
        ],
        "shard_wall_seconds_per_input": round(shard_wall / len(rows), 1) if sw else None,
        "cache": [
            {
                k: (p or {}).get(k)
                for k in (
                    "shard",
                    "linked_levels",
                    "levels_newer_than_cutoff_left_out",
                    "cutoff_unix",
                    "cpu_quota",
                    "memory_limit_bytes",
                    "loadavg",
                    "gpu_query",
                )
            }
            for p in prep
        ],
        "rows": [
            {
                "input_id": r["input_id"],
                "status": r["status"],
                "physical_pass": r.get("physical_pass"),
                "total_wall_seconds": r.get("total_wall_seconds"),
                "groups": {
                    n: round(sum(p["runtime_seconds"] for p in r.get("phase_timings") or [] if p["phase"] in ph), 2)
                    for n, ph in GROUPS.items()
                },
                "shard": r["_shard"],
            }
            for r in rows
        ],
    }


def collect_omni72():
    rep = {}
    for f in glob.glob(str(RT / "physx-omni/lane-*/native_representation/cases/*/native_representation_result.json")):
        d = AX.load_json(f)
        if not d:
            continue
        t = AX.load_json(os.path.join(os.path.dirname(f), "timing.json")) or {}
        rep[d["source_id"]] = {
            "rev": "072",
            "emitted": bool(d.get("native_representation_emitted")),
            "parts": d.get("part_count"),
            "seconds": d.get("runtime_seconds"),
            "peak_gib": (t.get("peak_memory_bytes") or 0) / 2**30,
            "error": None if d.get("native_representation_emitted") else str(d.get("failure_reason"))[:120],
            "stages": (
                {"representation_seconds": d.get("runtime_seconds")}
                if isinstance(d.get("runtime_seconds"), (int, float))
                else {}
            ),
            "dups": [],
        }
    geo = {}
    for f in sorted(glob.glob(str(RT / "physx-omni/lane-*/native_geometry*/cases/*/native_asset_result.json"))):
        d = AX.load_json(f)
        if not d:
            continue
        sid = d["source_id"]
        fr = " ".join(str(x) for x in (d.get("failure_reason") or []))
        man = AX.load_json(os.path.join(os.path.dirname(f), "native_manifest.json"))
        oom = "OutOfMemoryError" in fr and not d.get("asset_emitted")
        geo.setdefault(
            sid,
            {
                "rev": "072",
                "emitted": bool(d.get("asset_emitted")),
                "parts": d.get("part_count"),
                "seconds": d.get("total_runtime_seconds"),
                "peak_gib": (d.get("peak_memory") or 0) / 2**30,
                "infra": oom,
                "capacity_exhausted": False,
                "error": None if d.get("asset_emitted") else fr[:120],
                "path": f,
                "movable_joints": AX.movable_joints(man) if man else None,
                "stages": {k: float(v) for k, v in d.items() if k.endswith("_seconds") and isinstance(v, (int, float))},
            },
        )
    gates = AX.collect_gates(
        [f"gates/{n}/physx-omni" for n in GATE_MACHINES if (RT / f"gates/{n}/physx-omni").is_dir()]
    )
    return rep, geo, gates


def clean_worker(rows):
    """mean seconds one exclusive worker spends per input, failed inputs included: native path = method stages + native
    gate (0 where the native condition does not apply or nothing was exported); adapted path = native path + our
    materialization + the adapted gate where an asset was adapted. Only rows with a gate verdict count."""
    nat, ada = [], []
    for r in rows:
        e = r.get("e2e_native_seconds")
        if e is None:
            continue
        nat.append(e)
        if r.get("adapted_status") == "evaluated" and r.get("adapted_materialize_seconds") is not None:
            ada.append(e + r["adapted_materialize_seconds"] + (r.get("adapted_gate_seconds") or 0))
        elif r.get("adapted_status") in ("not_run", "not_applicable") or r.get("export") is False:
            ada.append(e)
    return {"native_mean_s": mean(nat), "native_n": len(nat), "adapted_mean_s": mean(ada), "adapted_n": len(ada)}


def lane_walls(meth):
    ws = walls(f"{meth}/lane-*/lane.wall.json")
    out = []
    for d in ws:
        lane_dir = RT / os.path.dirname(d["_file"])
        n = len(glob.glob(str(lane_dir / "cases/*/result.json")))
        s = d.get("start_unix", d.get("stage_a_start_unix"))
        e = d.get("end_unix")
        out.append(
            {
                "file": d["_file"],
                "host": d.get("host"),
                "gpu": d.get("gpu"),
                "gpu_name": d.get("gpu_name"),
                "cases": n,
                "wall_seconds": round(e - s, 1) if s and e else None,
                "monitor": window(d.get("host"), d.get("gpu"), s, e) if s and e else None,
            }
        )
    tot = sum(x["wall_seconds"] for x in out if x["wall_seconds"])
    n = sum(x["cases"] for x in out if x["wall_seconds"])
    pk = [x["monitor"]["peak_gib"] for x in out if x.get("monitor") and x["monitor"]["peak_gib"]]
    en = sum(x["monitor"]["energy_wh"] or 0 for x in out if x.get("monitor"))
    return {
        "lanes": out,
        "worker_seconds_per_input": round(tot / n, 1) if n else None,
        "monitor_peak_gib": max(pk) if pk else None,
        "energy_wh_per_input": round(en / n, 2) if n and en else None,
    }


def campaign_same_ids(mid, track_pref=("full2000", "subset200")):
    agg = load(CAMPAIGN_AGGREGATE) if CAMPAIGN_AGGREGATE else None
    m = ((agg or {}).get("methods") or {}).get(mid) or {}
    for tr in track_pref:
        s = m.get(tr)
        if s and s.get("rows"):
            rows = {r["source_id"]: r for r in s["rows"]}
            sel = [rows[i] for i in SAMPLE if i in rows]
            e2n = [r["e2e_native_seconds"] for r in sel if r.get("e2e_native_seconds")]
            e2a = [r["e2e_adapted_seconds"] for r in sel if r.get("e2e_adapted_seconds")]
            mat = [r["adapted_materialize_seconds"] for r in sel if r.get("adapted_materialize_seconds")]
            s1 = [r["stage1_seconds"] for r in sel if r.get("stage1_seconds")]
            geo = [r["geometry_seconds"] for r in sel if r.get("geometry_seconds")]
            c = s["counts"]
            n = s["denominator"]
            return {
                "track": tr,
                "n_same_ids": len(sel),
                "stage1_median": med(s1),
                "geometry_median": med(geo),
                "e2e_native_median": med(e2n),
                "e2e_native_mean": mean(e2n),
                "e2e_adapted_median": med(e2a),
                "e2e_adapted_mean": mean(e2a),
                "materialize_median": med(mat),
                "track_pass_rates": {
                    "denominator": n,
                    "native_pass": c["native_pass"],
                    "adapted_pass": c["adapted_pass"],
                    "export": c["export"],
                    "complete": s.get("complete"),
                    "pending": c.get("pending"),
                    "infra_blocked": c.get("infra_blocked"),
                },
                "track_timing": s.get("timing"),
            }
    return None


MON = {}


def monitor(node):
    """1-s nvidia-smi samples of one machine: monitor/<host>_gpu.csv, rows 'timestamp, index, memory.used, utilization.gpu,
    power.draw' (nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv,noheader -l 1)
    """
    if node not in MON:
        rows = []
        f = RT / f"monitor/{node}_gpu.csv"
        if f.exists():
            for row in csv.reader(open(f)):
                try:
                    ts = time.mktime(time.strptime(row[0].strip().split(".")[0], "%Y/%m/%d %H:%M:%S"))
                    rows.append(
                        (
                            ts,
                            int(row[1]),
                            float(row[2].split()[0]),
                            float(row[4].split()[0]) if len(row) > 4 and row[4].strip()[0].isdigit() else None,
                        )
                    )
                except Exception:
                    continue
        MON[node] = rows
    return MON[node]


def window(node, gpu, t0, t1):
    """nvidia-smi samples (1 s) of one GPU inside [t0, t1]: peak memory in use (GiB) and energy (Wh, sum of 1 s power samples)."""
    v = [(m, w) for ts, i, m, w in monitor(node) if i == gpu and t0 <= ts <= t1]
    if not v:
        return {"peak_gib": None, "energy_wh": None, "samples": 0}
    return {
        "peak_gib": round(max(m for m, _ in v) / 1024, 1),
        "energy_wh": round(sum(w or 0 for _, w in v) / 3600, 1),
        "samples": len(v),
    }


def monitor_peaks():
    out = {}
    for f in sorted(glob.glob(str(RT / "monitor/*_gpu.csv"))):
        node = os.path.basename(f).split("_gpu")[0]
        pk = {}
        for row in csv.reader(open(f)):
            try:
                i = int(row[1])
                mem = float(row[2].split()[0])
            except Exception:
                continue
            pk[i] = max(pk.get(i, 0), mem)
        out[node] = {str(k): round(v / 1024, 1) for k, v in sorted(pk.items())}
    return out


def main():
    global RT, SAMPLE, GATE_MACHINES, CAMPAIGN_PER_CASE, CAMPAIGN_AGGREGATE
    a = argparse.ArgumentParser()
    a.add_argument(
        "--root",
        default=str(RT),
        help="run root of the clean timing run (inputs/, ours-*/, <method>/, gates/, monitor/)",
    )
    a.add_argument("--campaign-per-case", required=True, help="exp1 per-case records of the main campaign (JSONL)")
    a.add_argument(
        "--campaign-aggregate", required=True, help="aggregate of the external runs over the registered inputs"
    )
    a.add_argument(
        "--gate-machines", nargs="*", default=None, help="gates/<machine> directories in verdict precedence order"
    )
    a.add_argument(
        "--finalization", default=None, help="finalization record passed to the aggregator (see evaluation/aggregate)"
    )
    args = a.parse_args()
    RT = Path(args.root)
    CAMPAIGN_PER_CASE = Path(args.campaign_per_case)
    CAMPAIGN_AGGREGATE = Path(args.campaign_aggregate)
    SAMPLE = [c["input_id"] for c in json.load(open(RT / "inputs/sample32.json"))["inputs"]]
    GATE_MACHINES = (
        tuple(args.gate_machines)
        if args.gate_machines
        else tuple(
            sorted(
                p.name
                for p in (RT / "gates").iterdir()
                if p.is_dir() and p.name not in ("inbox", "dispatch") and not p.name.startswith("cache-")
            )
        )
    )
    AX.FINALIZATION = args.finalization
    AX.R = RT
    out = {
        "schema": "affordcraft.clean_timing.v1",
        "written": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "sample": {"n": len(SAMPLE), "path": str(RT / "inputs/sample32.json")},
        "protocol": "one worker per exclusive GPU (drivers refuse a GPU with >= 1 GiB in use); RTX 5090 32 GB for every stage except PhysX-Omni geometry+export (RTX PRO 6000 96 GB: 91 GiB peak); gates on 580-driver machines with GATE_MAT_SLOTS = gate lanes of the machine and an empty geometry cache",
        "ours": {"cold": ours("cold"), "warm": ours("warm")},
        "external": {},
        "monitor_peak_gib": monitor_peaks(),
    }
    note = "e2e native = method stages + native gate seconds (0 when the native condition does not apply); adapted = method stages + materialize + adapted gate"
    for mid, meth in EXT.items():
        AX.METHODS = {
            mid: (
                meth,
                [f"gates/{n}/{meth}" for n in GATE_MACHINES if (RT / f"gates/{n}/{meth}").is_dir()],
                "clean timing, exclusive RTX 5090",
            )
        }
        AX.EXTRA_RUNS = {}
        AX.EXTRA_GATES = {}
        res, gates, dups, dev = AX.collect_generic(mid)
        gates.pop("_duplicates", None)
        s = AX.summarize(SAMPLE, res, None, gates, "clean_timing_sample32", dev, note)
        s["lane_walls"] = lane_walls(meth)
        s["campaign_same_ids"] = campaign_same_ids(mid)
        s["gated"] = sum(1 for r in s["rows"] if r.get("native_status") or r.get("adapted_status"))
        s["clean_worker"] = clean_worker(s["rows"])
        out["external"][mid] = s
    rep, geo, gates = collect_omni72()
    gates.pop("_duplicates", None)
    s = AX.summarize(
        SAMPLE, rep, geo, gates, "clean_timing_sample32", "representation RTX 5090, geometry RTX PRO 6000", note
    )
    s["lane_walls"] = {
        "representation": [
            dict(
                d,
                _wall=round(d["end_unix"] - d["registration_start_unix"], 1),
                monitor=window(d.get("host"), d["gpu"], d["registration_start_unix"], d["end_unix"]),
            )
            for d in walls("physx-omni/lane-*/rep.wall.json")
        ],
        "geometry": [
            dict(
                d,
                _wall=round(d["end_unix"] - d["registration_start_unix"], 1),
                monitor=window(d.get("host"), d["gpu"], d["registration_start_unix"], d["end_unix"]),
            )
            for d in walls("physx-omni/lane-*/geo.wall.json")
        ],
    }
    s["campaign_same_ids"] = campaign_same_ids("physx-omni-v0.1")
    s["gated"] = sum(1 for r in s["rows"] if r.get("native_status") or r.get("adapted_status"))
    s["clean_worker"] = clean_worker(s["rows"])
    out["external"]["physx-omni-v0.1"] = s
    ca = [json.loads(l) for l in open(CAMPAIGN_PER_CASE)]
    ca = {r["input_id"]: r for r in ca}
    sel = [ca[i] for i in SAMPLE if i in ca]
    w = [r["total_wall_seconds"] for r in sel]
    out["ours"]["campaign_same_ids"] = {
        "n": len(sel),
        "passes": sum(1 for r in sel if r.get("physical_pass")),
        "e2e_median": med(w),
        "e2e_mean": mean(w),
        "e2e_p95": p95(w),
        "group_median": {
            n: med(
                [
                    sum(p["runtime_seconds"] for p in r.get("phase_timings") or [] if p["phase"] in ph)
                    for r in sel
                    if any(p["phase"] in ph for p in r.get("phase_timings") or [])
                ]
            )
            for n, ph in GROUPS.items()
        },
        "campaign_pass_rate_2000": 1703 / 2000,
    }
    os.makedirs(RT / "summary", exist_ok=True)
    p = RT / "summary" / f"clean_timing_summary_{time.strftime('%Y%m%d-%H%M')}.json"
    p.write_text(json.dumps(out, indent=1))
    (RT / "summary/clean_timing_summary_latest.json").write_text(json.dumps(out, indent=1))
    brief = {
        "ours_cold": {
            k: (out["ours"]["cold"] or {}).get(k)
            for k in ("n", "passes", "e2e_median", "e2e_mean", "shard_wall_seconds_per_input")
        },
        "ours_warm": {k: (out["ours"]["warm"] or {}).get(k) for k in ("n", "passes", "e2e_median", "e2e_mean")},
    }
    for mid, s in out["external"].items():
        t = s["timing"]
        brief[mid] = {
            "gen": s["counts"]["stage1_emitted"],
            "export": s["counts"]["export"],
            "gated": s["gated"],
            "stage1_med": t["stage1_median_s"],
            "geo_med": t["geometry_median_s"],
            "mat_med": t["adapted_materialize_median_s"],
            "e2e_nat_med": t["e2e_native_median_s"],
            "e2e_ada_med": t["e2e_adapted_median_s"],
            "peak": t["peak_gib_max"],
            "lane_s_per_input": (
                (s["lane_walls"] or {}).get("worker_seconds_per_input") if isinstance(s["lane_walls"], dict) else None
            ),
        }
    print(json.dumps(brief, indent=0)[:3000])
    print("written", p)


if __name__ == "__main__":
    main()
