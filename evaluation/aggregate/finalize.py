#!/usr/bin/env python3
"""Finalization rules of the external runs: two terminal rules for adapted builds that the container memory guard kept
killing (materialize_process_-9) or that the build time limit ended. Both only mark verdicts; nothing is recomputed or
deleted, and every record is written once (open 'x').

cutoff  (aggregator rule v2.12, used for the 2,000-input runs of the five local methods)
    1. runs aggregate_external.py on the verdicts present at the cutoff and lists every source id whose chosen verdict
       is still infrastructure-blocked;
    2. writes the finalization record: cutoff time, decision, rule, the ids with method, gate stage, statuses, reasons
       and the number of earlier attempts kept; refuses if any blocked id is blocked in generation (not in the gate) or
       blocked for a reason other than the memory guard;
    3. runs the aggregator again with --finalization <record>: a listed id whose chosen verdict is still blocked is
       counted as capacity exhausted (physical pass false), exactly the terminal rule v2.6 applies to a big-memory retry
       that failed on every attempt. A later unblocked verdict still takes precedence (collect_gates rule), and verdicts
       of gate runs listed under post_finalization_gates (started after the cutoff) are never finalized.
retry-exhausted  (aggregator rule v2.6, applied to one gate run; used for the PartCrafter 2,000-input gates)
    an adapted build that the memory guard killed (or the build time limit ended) at the origin lane, again in the
    big-memory re-gate (retry-bigmem, two builders per 360/450 GiB machine) and, for the cases tried, a third time alone
    on a 450 GiB machine (a clone stage, --third-attempt), is capacity exhausted (physical pass false). For each id whose
    retry-bigmem verdict is still blocked, writes retry_exhausted.json beside that verdict (the aggregator turns it into
    capacity_exhausted) unless the third attempt returned an unblocked verdict, and records everything in
    <gate run>/finalization_retry_exhausted.json.
usage: finalize.py cutoff --out FIN.json --final-aggregate AGG.json [--methods ...] [--config ...] [--runs-root ...]
       finalize.py retry-exhausted --gate-run DIR [--third-attempt clone-bigmem1s]
"""
import argparse, glob, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGG = HERE / "aggregate_external.py"
LOCAL_METHODS = ("pact-v0.1", "physx-omni-v0.1", "physx-anything-v0.1", "trellis2-4b-v0.1", "partcrafter-v0.1")
CUTOFF_RULE = (
    "a gate condition that is still infrastructure_blocked by the container memory guard (adapted build process killed, "
    "materialize_process_-9) at this cutoff, after its origin re-attempts and any big-memory retries so far, is counted as "
    "capacity exhausted (physical pass false), the terminal rule v2.6 applies to a big-memory retry that failed on every "
    "attempt; the native condition of these inputs was evaluated normally"
)
RETRY_RULE = (
    "an adapted build that the container memory guard killed (or the build time limit ended) at the origin lane, again in the "
    "big-memory re-gate (retry-bigmem) and, for the cases tried, a third time alone on a big-memory machine (the third-attempt "
    "clone stage) is capacity exhausted (physical pass false)"
)


def run_agg(args, out, finalization=None):
    cmd = [sys.executable, str(AGG), "--method", "all", "--out", str(out), "--config", args.config]
    for flag, v in (
        ("--runs-root", args.runs_root),
        ("--subset", args.subset),
        ("--manifest", args.manifest),
        ("--finalization", finalization),
    ):
        if v:
            cmd += [flag, str(v)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    return json.load(open(out))


def gate_dirs(cfg):
    """every gate run of the configuration (method gates, extension gates, PhysX-Omni gates)"""
    out = [g for m in cfg["methods"].values() for g in m["gates"]] + [
        g for v in cfg.get("extra_gates", {}).values() for g in v
    ]
    return out + list(cfg["physx_omni"]["gates"])


def cutoff(args):
    sys.path.insert(0, str(HERE))
    import aggregate_external as AX

    cfg = AX.load_config(args.config)
    R = Path(AX._expand(args.runs_root or cfg.get("runs_root") or str(AX.R)))
    out = Path(args.out)
    pre = run_agg(args, out.with_name(out.stem + "_prefinal_aggregate.json"))
    ids = {}
    for mid, m in pre["methods"].items():
        if mid not in args.methods:
            continue  # e.g. the API routes, whose runs were not part of the cutoff
        for track in ("subset200", "full2000"):
            s = m.get(track)
            if not s:
                continue
            rows = {r["source_id"]: r for r in s["rows"]}
            for sid in s["blocked_ids"]:
                r = rows[sid]
                reasons = [
                    str(x) for x in (r.get("adapted_failure_reasons") or []) + (r.get("native_failure_reasons") or [])
                ]
                if r.get("stage1_infra") or r.get("geometry_infra_failure"):
                    sys.exit(f"refusing: {mid} {sid} is blocked in generation, not in the gate")
                if not any("materialize_process_-9" in x for x in reasons):
                    sys.exit(f"refusing: {mid} {sid} is blocked for another reason: {reasons}")
                ids.setdefault(
                    sid,
                    {
                        "method_id": mid,
                        "tracks": [],
                        "gate_stage": r.get("gate_stage"),
                        "native_status": r.get("native_status"),
                        "adapted_status": r.get("adapted_status"),
                        "reasons": reasons[:3],
                    },
                )
                ids[sid]["tracks"].append(track)
    for sid, v in ids.items():
        v["earlier_attempts_kept"] = len(
            [
                p
                for g in gate_dirs(cfg)
                for p in glob.glob(str(R / g / "interrupted" / (sid + "-*")))
                + glob.glob(str(R / g / "clone-*" / "interrupted" / (sid + "-*")))
            ]
        )
    counts = {}
    for v in ids.values():
        counts[v["method_id"]] = counts.get(v["method_id"], 0) + 1
    rec = {
        "schema": "affordcraft.external_finalization.v1",
        "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": args.decision,
        "rule": CUTOFF_RULE,
        "counts": counts,
        "ids": ids,
    }
    with open(out, "x") as f:
        json.dump(rec, f, indent=1)
    print("finalization record:", counts)
    final = Path(args.final_aggregate)
    fin_agg = run_agg(args, final.with_name(final.name + ".tmp"), finalization=out)
    os.replace(final.with_name(final.name + ".tmp"), final)
    for mid, m in fin_agg["methods"].items():
        for track in ("subset200", "full2000"):
            s = m.get(track)
            if s:
                c = s["counts"]
                print(
                    mid,
                    track,
                    "complete" if s["complete"] else "INCOMPLETE",
                    "pending",
                    c["pending"],
                    "blocked",
                    c["infra_blocked"],
                    "capacity",
                    c["capacity_exhausted"],
                    "export",
                    c["export"],
                    "native",
                    c["native_pass"],
                    "adapted",
                    c["adapted_pass"],
                )


def retry_exhausted(args):
    R = Path(args.gate_run)
    third = args.third_attempt
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    rec = {"written": ts, "rule": RETRY_RULE, "third_attempt_stage": third, "ids": {}}
    for d in sorted(glob.glob(str(R / "retry-bigmem/cases/*/case_result.json"))):
        v = json.load(open(d))
        a = v.get("adapted") or {}
        if a.get("status") != "infrastructure_blocked":
            continue
        sid = v["source_id"]
        origin = R / "cases" / sid / "case_result.json"
        o = (json.load(open(origin)).get("adapted") or {}) if origin.exists() else {}
        c = R / third / "cases" / sid / "case_result.json"
        cl = (json.load(open(c)).get("adapted") or {}) if c.exists() else None
        entry = {
            "origin": o.get("reason"),
            "retry_bigmem": a.get("reason"),
            "third_attempt": (cl or {}).get("reason") if cl else "not reached (second round stopped)",
        }
        if cl and cl.get("status") != "infrastructure_blocked":
            entry["note"] = "third-attempt verdict unblocked; not marked"
            rec["ids"][sid] = entry
            continue
        m = Path(d).with_name("retry_exhausted.json")
        if not m.exists():
            with open(m, "x") as f:
                json.dump(
                    {
                        "source_id": sid,
                        "written": ts,
                        "attempts": entry,
                        "rule": "capacity exhausted: killed by the memory guard or the build time limit on every big-memory attempt",
                    },
                    f,
                    indent=1,
                )
        entry["marked"] = str(m)
        rec["ids"][sid] = entry
    with open(R / "finalization_retry_exhausted.json", "x") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({k: v for k, v in rec["ids"].items()}, indent=1)[:3000])
    print("marked", sum(1 for v in rec["ids"].values() if "marked" in v))


def main():
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = a.add_subparsers(dest="rule", required=True)
    c = sub.add_parser("cutoff", help="v2.12 finalization record + final aggregate")
    c.add_argument("--out", required=True, help="finalization record to write (must not exist)")
    c.add_argument("--final-aggregate", required=True, help="aggregate written with the finalization applied")
    c.add_argument(
        "--methods", nargs="+", default=list(LOCAL_METHODS), help="methods whose blocked verdicts are finalized"
    )
    c.add_argument(
        "--decision",
        default="finalize the 2,000-input runs at this cutoff instead of waiting for further big-memory retries",
    )
    c.add_argument("--config", default=str(HERE / "methods.json"))
    c.add_argument("--runs-root")
    c.add_argument("--subset")
    c.add_argument("--manifest")
    r = sub.add_parser("retry-exhausted", help="v2.6 terminal rule for one gate run")
    r.add_argument("--gate-run", required=True, help="gate run directory (holds cases/, retry-bigmem/cases/)")
    r.add_argument(
        "--third-attempt",
        default="clone-bigmem1s",
        help="clone stage of the third attempt (one builder alone on a big-memory machine)",
    )
    args = a.parse_args()
    cutoff(args) if args.rule == "cutoff" else retry_exhausted(args)


if __name__ == "__main__":
    main()
