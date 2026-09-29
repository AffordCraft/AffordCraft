#!/usr/bin/env python3
"""Aggregate the external-method runs into paper evidence (read-only over the run directories).

Per input of the target set (registered 200 subset; the frozen 2,000-input manifest for the methods extended to it) the
chain is:
  method run (stage 1 [+ stage 2 geometry/export for PhysX-Omni]) -> native_manifest.json -> gates (native, adapted).
Counts use the FULL denominator (inputs without a result, without an export, or blocked by infrastructure are never
dropped; infrastructure blocks are counted separately and make the summary incomplete). Timing: per-case wall seconds
of the method's own stages as measured inside the stage processes + the native gate seconds; peak GPU memory as
recorded by the stage. Gate verdict precedence is fixed (see collect_gates), never outcome-based.
The run layout (which directories hold the method runs and the gate runs of every method, in precedence order) is read
from a JSON configuration (methods.json beside this file); directory names are relative to --runs-root.
usage: aggregate_external.py [--config methods.json] [--runs-root DIR] [--finalization FIN.json] --method <id|all> --out <json>
"""
import argparse, glob, hashlib, json, math, os, statistics as st
from pathlib import Path

R = Path(os.environ.get("AFFORDCRAFT_RUNS", "runs"))
SUBSET = R / "inputs/subset_200_inputs.json"
FULL = R / "inputs/input_manifest_2000.jsonl"
MOVABLE = ("revolute", "continuous", "prismatic", "spherical")

# method id -> (method run directory, gate run directories in precedence order, device note); from the configuration
METHODS = {}
# v2.7: the 2000-input extensions of subset methods (same code, weights and wrappers; separate runs): method runs appended
# after the subset run (first record per source_id in this order stands) and their gate runs appended after the subset
# gates (collect_gates precedence). They yield a 'full2000' track when the run dirs exist.
EXTRA_RUNS = {}
EXTRA_GATES = {}
# v2.13: gate runs started after the v2.12 finalization cutoff; the finalization never applies to their verdicts
POST_FINALIZATION_GATES = ()
# PhysX-Omni (two generation stages): representation runs, geometry runs ([label, glob relative to the runs root] in
# precedence order), gate runs, and the lane-directory marker of the recompute lanes on the largest available devices
OMNI = {"representation": [], "geometry": [], "gates": [], "capacity_lane_marker": None}
# v2.12 finalization record (written once at the cutoff by finalize.py); None: no finalization
FINALIZATION = None
INFRA_MARKERS = (
    "OutOfMemoryError",
    "api_error",
    "http_403",
    "http_429",
    "http_5",
    "infrastructure",
    "ECONNRESET",
    "timed out",
    "Connection",
    "CUDA error",
)


def _expand(p):
    """${AFFORDCRAFT_RUNS} (default 'runs') and other environment variables in a configured path"""
    return os.path.expandvars(str(p).replace("${AFFORDCRAFT_RUNS}", os.environ.get("AFFORDCRAFT_RUNS", "runs")))


def load_config(path):
    """Fill METHODS / EXTRA_RUNS / EXTRA_GATES / POST_FINALIZATION_GATES / OMNI from the JSON run-layout configuration."""
    global METHODS, EXTRA_RUNS, EXTRA_GATES, POST_FINALIZATION_GATES, OMNI
    cfg = json.load(open(path))
    METHODS = {m: (v["run"], list(v["gates"]), v["device"]) for m, v in cfg["methods"].items()}
    EXTRA_RUNS = {m: list(v) for m, v in cfg.get("extra_runs", {}).items()}
    EXTRA_GATES = {m: list(v) for m, v in cfg.get("extra_gates", {}).items()}
    POST_FINALIZATION_GATES = tuple(cfg.get("post_finalization_gates", ()))
    OMNI = dict(cfg["physx_omni"])
    return cfg


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    e = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - e), min(1.0, c + e)]


def med(v):
    return round(st.median(v), 1) if v else None


def p95(v):
    return round(sorted(v)[max(0, int(math.ceil(0.95 * len(v))) - 1)], 1) if v else None


_SHA = {}


def _sha(p):
    st = os.stat(p)
    k = (p, st.st_mtime_ns, st.st_size)
    if k not in _SHA:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        _SHA[k] = h.hexdigest()
    return _SHA[k]


UNPARSEABLE = []


def load_json(p):
    try:
        return json.load(open(p))
    except FileNotFoundError:
        return None
    except Exception:
        UNPARSEABLE.append(str(p))
        return None


def blocked(d):
    return any((d.get(c) or {}).get("status") == "infrastructure_blocked" for c in ("native", "adapted"))


def collect_gates(revs):
    """Gate verdicts of one method: the origin lanes' cases/ of every listed gate run plus the big-memory retry stage
    (retry-bigmem/cases/) for cases the 180 GiB machines could not build, and (v2.11) the clone stages
    clone-<name>/cases/ (clone lanes working the same hashed lanes from the other end of the list). Precedence is fixed,
    not outcome-based: a verdict without an infrastructure block replaces a blocked one; among unblocked duplicates the
    first seen (origin lane, then retry, then clones in name order; runs in the listed order) stands and the other is
    listed under _duplicates."""
    gates = {}
    gate_dups = []
    allc = {}
    for rev in revs:
        clones = tuple(os.path.relpath(p, R / rev) for p in sorted(glob.glob(str(R / rev / "clone-*" / "cases"))))
        for sub in ("cases", "retry-bigmem/cases") + clones:
            for f in sorted(glob.glob(str(R / rev / sub / "*/case_result.json"))):
                d = load_json(f)
                if not d:
                    continue
                d["_gate_run"] = rev
                d["_stage"] = sub
                sid = d["source_id"]
                allc.setdefault(sid, []).append(d)
                # v2.6: a big-memory retry verdict whose adapted build was killed on every attempt (retry_exhausted.json
                # beside it, written by retry_select.py) is terminal: the blocked condition = capacity exhausted (fail)
                if (
                    sub.startswith("retry")
                    and blocked(d)
                    and os.path.exists(os.path.join(os.path.dirname(f), "retry_exhausted.json"))
                ):
                    for cnd in ("native", "adapted"):
                        if (d.get(cnd) or {}).get("status") == "infrastructure_blocked":
                            d[cnd] = dict(d[cnd], status="capacity_exhausted", physical_pass=False)
                    d["_capacity_exhausted"] = True
                cur = gates.get(sid)
                if cur is None:
                    gates[sid] = d
                elif blocked(cur) and not blocked(d):
                    gate_dups.append({"source_id": sid, "superseded_blocked_verdict": cur["_stage"], "run": rev})
                    gates[sid] = d
                else:
                    gate_dups.append({"source_id": sid, "ignored": f})
    # v2.12: finalization of the 2,000-input runs. A listed id (the finalization record, written once at the cutoff by
    # finalize.py) whose chosen verdict is still blocked by the memory guard is counted as capacity exhausted (physical
    # pass false), the terminal rule v2.6 applies after failed big-memory retries; an unblocked verdict, when one exists,
    # still stands.
    fin = load_json(str(FINALIZATION)) if FINALIZATION else None
    if fin:
        for sid, d in gates.items():
            if sid in fin["ids"] and d.get("_gate_run") not in POST_FINALIZATION_GATES and blocked(d):
                for cnd in ("native", "adapted"):
                    if (d.get(cnd) or {}).get("status") == "infrastructure_blocked":
                        d[cnd] = dict(d[cnd], status="capacity_exhausted", physical_pass=False)
                d["_capacity_exhausted"] = True
                d["_finalized"] = fin["written"]
    gates["_duplicates"] = gate_dups
    gates["_all"] = allc
    return gates


def movable_joints(man):
    return sum(1 for j in (man or {}).get("joints") or [] if j.get("type") in MOVABLE)


def collect_omni():
    """PhysX-Omni: representation from the configured representation runs (subset run first), geometry from the
    configured geometry runs (incl. the OOM recompute lanes on the largest devices), gates from the configured gate runs
    (+ retry-bigmem)."""
    rep = {}
    for rev, pat in OMNI["representation"]:
        for f in glob.glob(str(R / pat)):
            d = load_json(f)
            if not d:
                continue
            t = load_json(os.path.join(os.path.dirname(f), "timing.json")) or {}
            rec = {
                "rev": rev,
                "emitted": bool(d.get("native_representation_emitted")),
                "parts": d.get("part_count"),
                "seconds": d.get("runtime_seconds"),
                "peak_gib": (t.get("peak_memory_bytes") or 0) / 2**30,
                "error": (d.get("failure_reason") or [None])[0] if not d.get("native_representation_emitted") else None,
                "stages": (
                    {"representation_seconds": d.get("runtime_seconds")}
                    if isinstance(d.get("runtime_seconds"), (int, float))
                    else {}
                ),
            }
            sid = d["source_id"]
            order = tuple(label for label, _ in OMNI["representation"])
            if sid not in rep:
                rep[sid] = {**rec, "dups": []}
            elif order.index(rev) < order.index(rep[sid]["rev"]):
                rep[sid] = {**rec, "dups": rep[sid]["dups"] + [{k: v for k, v in rep[sid].items() if k != "dups"}]}
            else:
                rep[sid]["dups"].append(rec)
    geo = {}
    for rev, pat in OMNI["geometry"]:
        for f in glob.glob(str(R / pat)):
            if os.path.islink(os.path.dirname(f)):
                continue  # inherited symlink of a later attempt: counted once via its original
            d = load_json(f)
            if not d:
                continue
            sid = d["source_id"]
            fr = " ".join(str(x) for x in (d.get("failure_reason") or []))
            man = load_json(os.path.join(os.path.dirname(f), "native_manifest.json"))
            oom = "OutOfMemoryError" in fr and not d.get("asset_emitted")
            # v2.4: an OOM recorded by a recompute lane on the largest available devices (A100 80 GB or RTX PRO 6000
            # 96 GB; lane directories marked by OMNI['capacity_lane_marker']) is terminal capacity exhaustion (export
            # False), not an open infrastructure block
            capacity = oom and OMNI["capacity_lane_marker"] in f
            rec = {
                "rev": rev,
                "emitted": bool(d.get("asset_emitted")),
                "parts": d.get("part_count"),
                "seconds": d.get("total_runtime_seconds"),
                "peak_gib": (d.get("peak_memory") or 0) / 2**30,
                "infra": oom and not capacity,
                "capacity_exhausted": capacity,
                "error": None if d.get("asset_emitted") else fr[:120],
                "path": f,
                "movable_joints": movable_joints(man) if man else None,
                "stages": {k: float(v) for k, v in d.items() if k.endswith("_seconds") and isinstance(v, (int, float))},
            }
            cur = geo.get(sid)
            if (
                cur is None
                or (cur["infra"] and not rec["infra"])
                or (not cur["emitted"] and rec["emitted"] and not rec["infra"])
                or (cur.get("capacity_exhausted") and rec["emitted"])
            ):
                geo[sid] = rec
    gates = collect_gates(tuple(OMNI["gates"]))
    return rep, geo, gates


def collect_generic(method):
    """Single-stage methods: lane-*/cases/<sid>/{result.json, timing.json, native_manifest.json}; the first record per
    source_id in sorted lane order stands and duplicates are listed. API/OOM failures are infrastructure, never method
    failures."""
    run, gate_revs, device = METHODS[method]
    res = {}
    dups = []
    runs = [run] + [x for x in EXTRA_RUNS.get(method, []) if (R / x).is_dir()]
    gate_revs = list(gate_revs) + [x for x in EXTRA_GATES.get(method, []) if (R / x).is_dir()]
    # v2.4: cases re-executed once on an exclusive GPU after a co-tenancy OOM (amendment 002 of the run); an OOM that
    # recurs there is terminal capacity exhaustion for the method on the reference device (export False)
    rerun_ids = set()
    for x in runs:
        am2 = load_json(str(R / x / "amendment_002_oom_rerun.json")) or {}
        rerun_ids |= {m["source_id"] for m in am2.get("moved") or []}
    for f in [f for x in runs for f in sorted(glob.glob(str(R / x / "lane-*/cases/*/result.json")))]:
        d = load_json(f)
        if not d:
            continue
        sid = d["source_id"]
        if sid in res:
            dups.append(f)
            continue
        t = load_json(os.path.join(os.path.dirname(f), "timing.json")) or {}
        man = load_json(os.path.join(os.path.dirname(f), "native_manifest.json"))
        secs = (
            d.get("runtime_seconds")
            or t.get("total_seconds")
            or t.get("total_runtime_seconds")
            or t.get("wall_seconds")
            or d.get("total_runtime_seconds")
        )
        peak = max(
            [
                float(v) / 2**30
                for k, v in t.items()
                if k.endswith("_bytes") and "peak" in k and isinstance(v, (int, float))
            ]
            or [0]
        )
        if not peak and isinstance(t.get("peak_gpu_mem"), (int, float)):
            peak = float(t["peak_gpu_mem"]) / 1024  # MiB
        if not peak and isinstance(d.get("peak_gpu_mem"), (int, float)):
            peak = float(d["peak_gpu_mem"]) / 1024
        fr = str(d.get("failure_reason") or "")
        stages = {k: float(v) for k, v in t.items() if k.endswith("_seconds") and isinstance(v, (int, float))}
        if isinstance(t.get("adapter"), dict):
            stages.update({"adapter_" + k: float(v) for k, v in t["adapter"].items() if isinstance(v, (int, float))})
        if isinstance(t.get("official"), dict):
            stages.update({"official_" + k: float(v) for k, v in t["official"].items() if isinstance(v, (int, float))})
        infra = (not d.get("asset_emitted")) and any(m in fr for m in INFRA_MARKERS)
        capacity = infra and sid in rerun_ids and ("out of memory" in fr.lower() or "OutOfMemoryError" in fr)
        res[sid] = {
            "emitted": bool(d.get("asset_emitted")),
            "parts": d.get("part_count"),
            "seconds": float(secs) if secs is not None else None,
            "peak_gib": peak or None,
            "infra": infra and not capacity,
            "capacity_exhausted": capacity,
            "error": None if d.get("asset_emitted") else fr[:120],
            "movable_joints": movable_joints(man) if man else None,
            "manifest": man is not None,
            "stages": stages,
        }
    gates = collect_gates(gate_revs)
    return res, gates, dups, device


def summarize(ids, stage1, stage2, gates, label, device, e2e_note):
    n = len(ids)
    rows = []
    stale = 0
    allc = gates.get("_all", {})
    for sid in ids:
        r = stage1.get(sid)
        g = stage2.get(sid) if stage2 is not None else None
        c = gates.get(sid)
        chosen = (g if stage2 is not None else r) or {}

        # v2.5: a verdict computed from an earlier manifest of the same case (recorded sha differs from
        # the manifest now at that path, e.g. after an OOM recompute or an exclusive-GPU rerun superseded the export) is
        # stale; likewise a verdict that saw no asset while the record chosen by precedence carries an emitted export.
        # v2.9: when the first-precedence verdict is stale, the first non-stale verdict of the other gate run
        # (or stage) stands, unblocked before blocked; only when none exists is the case pending (the gate recomputes).
        def _stale(v):
            # v2.9b: the sha rule is dropped. Every gate machine generates its own native_manifest.json for a case, and
            # the collected copy is whichever synced last, so a sha comparison says nothing about the
            # verdict (it flagged 194 of 200 subset verdicts as stale). What matters: a verdict that saw no asset while
            # the chosen geometry record (precedence in collect_omni) carries an emitted export.
            if not v:
                return False
            return bool(v.get("asset_emitted") is False and chosen.get("emitted"))

        if c is not None and _stale(c):
            alts = [v for v in allc.get(sid, []) if v is not c and not _stale(v)]
            unb = [v for v in alts if not blocked(v)]
            c = (unb or alts or [None])[0]
            if c is None:
                stale += 1
        nat = (c or {}).get("native") or {}
        ada = (c or {}).get("adapted") or {}
        last = g if stage2 is not None else r
        row = {
            "source_id": sid,
            "stage1": r["emitted"] if r else None,
            "stage1_seconds": r["seconds"] if r else None,
            "stage1_infra": r.get("infra") if r else None,
            "capacity_exhausted": bool(
                (r or {}).get("capacity_exhausted")
                or (g or {}).get("capacity_exhausted")
                or (c or {}).get("_capacity_exhausted")
            ),
            "export": last["emitted"] if last else None,
            "geometry_seconds": g["seconds"] if g else None,
            "geometry_infra_failure": g["infra"] if g else None,
            "native_status": nat.get("status"),
            "native_pass": nat.get("physical_pass"),
            "native_failure_reasons": nat.get("failure_reasons")
            or ([nat.get("reason")] if nat.get("reason") else None),
            "native_gate_seconds": nat.get("gate_seconds"),
            "adapted_status": ada.get("status"),
            "adapted_pass": ada.get("physical_pass"),
            "adapted_failure_reasons": ada.get("failure_reasons")
            or ([ada.get("reason")] if ada.get("reason") else None),
            "adapted_materialize_seconds": (c or {}).get("adapted_materialize_seconds"),
            "adapted_gate_seconds": ada.get("gate_seconds"),
            "movable_joints": (last or {}).get("movable_joints"),
            "peak_gib": max([(r or {}).get("peak_gib") or 0, (g or {}).get("peak_gib") or 0]) or None,
            "gate_stage": (c or {}).get("_stage"),
            "stages": {
                **((r or {}).get("stages") or {}),
                **({"geometry_" + k: v for k, v in ((g or {}).get("stages") or {}).items()} if g else {}),
            },
        }
        e2e = (
            [row["stage1_seconds"]]
            + ([row["geometry_seconds"]] if stage2 is not None else [])
            + [
                (
                    row["native_gate_seconds"]
                    if nat.get("status") == "evaluated"
                    else (
                        0.0
                        if nat.get("status") in ("not_applicable", "not_run", "blocked", "capacity_exhausted")
                        else None
                    )
                )
            ]
        )
        row["e2e_native_seconds"] = sum(e2e) if all(x is not None for x in e2e) else None
        # adapted path: method stages + our adaptation (materialize) + the adapted gate; the E2E of methods whose native
        # condition does not apply (single meshes without physical parameters)
        base = [row["stage1_seconds"]] + ([row["geometry_seconds"]] if stage2 is not None else [])
        ada_extra = [
            row["adapted_materialize_seconds"],
            row["adapted_gate_seconds"] if ada.get("status") == "evaluated" else None,
        ]
        row["e2e_adapted_seconds"] = sum(base + ada_extra) if all(x is not None for x in base + ada_extra) else None
        rows.append(row)
    # v2.3: a non-infrastructure 'blocked' verdict (native export not loadable by the frozen parser /
    # MuJoCo, or authoring failed) is a terminal method-side failure: physical_pass False in both conditions, counted
    # under native_not_loadable; it never made a summary incomplete before, it made it pending forever.
    terminal = [
        x
        for x in rows
        if x["native_status"] in ("evaluated", "not_run", "not_applicable", "blocked", "capacity_exhausted")
        or (x["export"] is False and not (x["stage1_infra"] or x["geometry_infra_failure"]))
    ]
    infra = [
        x
        for x in rows
        if x["native_status"] == "infrastructure_blocked"
        or x["adapted_status"] == "infrastructure_blocked"
        or x["geometry_infra_failure"]
        or x["stage1_infra"]
    ]
    k = {
        "stage1_emitted": sum(1 for x in rows if x["stage1"]),
        "export": sum(1 for x in rows if x["export"]),
        "native_pass": sum(1 for x in rows if x["native_pass"] is True),
        "adapted_pass": sum(1 for x in rows if x["adapted_pass"] is True),
        "native_not_applicable": sum(1 for x in rows if x["native_status"] == "not_applicable"),
        "native_pass_articulated": sum(1 for x in rows if x["native_pass"] is True and (x["movable_joints"] or 0) > 0),
        "adapted_pass_articulated": sum(
            1 for x in rows if x["adapted_pass"] is True and (x["movable_joints"] or 0) > 0
        ),
        "export_articulated": sum(1 for x in rows if x["export"] and (x["movable_joints"] or 0) > 0),
        "gate_evaluated": sum(
            1 for x in rows if x["native_status"] == "evaluated" or x["adapted_status"] == "evaluated"
        ),
        "infra_blocked": len(infra),
        "chain_terminal": len(terminal),
        "no_result_yet": sum(1 for x in rows if x["stage1"] is None),
        "native_not_loadable": sum(1 for x in rows if x["native_status"] == "blocked"),
        "capacity_exhausted": sum(1 for x in rows if x["capacity_exhausted"]),
        "gate_verdict_stale": stale,
    }
    e2e = [x["e2e_native_seconds"] for x in rows if x["e2e_native_seconds"]]
    e2a = [x["e2e_adapted_seconds"] for x in rows if x["e2e_adapted_seconds"]]
    pending = [
        x["source_id"]
        for x in rows
        if x["stage1"] is None
        or (
            x["export"]
            and x["native_status"]
            not in ("evaluated", "not_run", "not_applicable", "infrastructure_blocked", "blocked", "capacity_exhausted")
        )
        or (x["export"] and x["adapted_status"] in (None, "authored"))
    ]
    k["pending"] = len(pending)
    stage_keys = sorted({kk for x in rows for kk in x["stages"]})
    stage_medians = {kk: med([x["stages"][kk] for x in rows if kk in x["stages"]]) for kk in stage_keys}
    summ = {
        "label": label,
        "denominator": n,
        "counts": k,
        "pending_ids": pending[:50],
        "blocked_ids": [x["source_id"] for x in infra][:200],
        "capacity_exhausted_ids": [x["source_id"] for x in rows if x["capacity_exhausted"]],
        "stage_medians": stage_medians,
        "rates": {
            m: {"k": k[m], "n": n, "rate": k[m] / n, "wilson95": wilson(k[m], n)}
            for m in ("export", "native_pass", "adapted_pass")
        },
        "timing": {
            "stage1_median_s": med([x["stage1_seconds"] for x in rows if x["stage1_seconds"]]),
            "geometry_median_s": med([x["geometry_seconds"] for x in rows if x["geometry_seconds"]]),
            "native_gate_median_s": med([x["native_gate_seconds"] for x in rows if x["native_gate_seconds"]]),
            "adapted_materialize_median_s": med(
                [x["adapted_materialize_seconds"] for x in rows if x["adapted_materialize_seconds"]]
            ),
            "e2e_native_median_s": med(e2e),
            "e2e_native_p95_s": p95(e2e),
            "e2e_native_mean_s": round(st.mean(e2e), 1) if e2e else None,
            "e2e_adapted_median_s": med(e2a),
            "e2e_adapted_p95_s": p95(e2a),
            "e2e_adapted_mean_s": round(st.mean(e2a), 1) if e2a else None,
            "e2e_adapted_n": len(e2a),
            "native_applies": any(x["native_status"] == "evaluated" for x in rows),
            "worker_minutes_per_native_pass": (
                round(sum(e2e) / 60 / k["native_pass"], 1) if k["native_pass"] and e2e else None
            ),
            "worker_minutes_per_adapted_pass": (
                round(
                    (
                        sum(e2e)
                        + sum((x["adapted_materialize_seconds"] or 0) + (x["adapted_gate_seconds"] or 0) for x in rows)
                    )
                    / 60
                    / k["adapted_pass"],
                    1,
                )
                if k["adapted_pass"] and e2e
                else None
            ),
            "peak_gib_max": round(max([x["peak_gib"] for x in rows if x["peak_gib"]] or [0]), 1),
            "peak_gib_median": med([x["peak_gib"] for x in rows if x["peak_gib"]]),
            "device": device,
            "note": e2e_note,
        },
        "complete": k["chain_terminal"] == n and not infra,
        "rows": rows,
    }
    return summ


def run_one(method, ids200, ids2000):
    if method == "physx-omni-v0.1":
        rep, geo, gates = collect_omni()
        dups = gates.pop("_duplicates", [])
        note = (
            "e2e = representation + geometry/export + native gate seconds per input; adapted condition timed separately"
        )
        dev = "RTX 5090 32 GB, one lane per GPU per stage (representation, geometry, gate on different machines); CUDA-OOM geometry cases recomputed on A100 80 GB"
        return {
            "gate_duplicates_ignored": dups,
            "subset200": summarize(ids200, rep, geo, gates, "registered_subset_200", dev, note),
            "full2000": summarize(ids2000, rep, geo, gates, "frozen_manifest_2000", dev, note),
        }
    res, gates, dups, dev = collect_generic(method)
    gdups = gates.pop("_duplicates", [])
    note = "e2e = method run seconds + native gate seconds per input (0 when the native condition is not applicable); adapted condition timed separately"
    out = {
        "result_duplicates_ignored": dups,
        "gate_duplicates_ignored": gdups,
        "subset200": summarize(ids200, res, None, gates, "registered_subset_200", dev, note),
    }
    if any((R / x).is_dir() for x in EXTRA_RUNS.get(method, [])):
        out["full2000"] = summarize(
            ids2000,
            res,
            None,
            gates,
            "frozen_manifest_2000",
            dev + "; remaining 1800 in " + ", ".join(EXTRA_RUNS[method]),
            note,
        )
    return out


def main():
    global R, FINALIZATION
    a = argparse.ArgumentParser()
    a.add_argument("--method", default="all")
    a.add_argument("--out", required=True)
    a.add_argument(
        "--config", default=str(Path(__file__).resolve().with_name("methods.json")), help="run layout (methods.json)"
    )
    a.add_argument("--runs-root", default=None, help="directory the run names of the configuration are relative to")
    a.add_argument("--subset", default=None, help='registered 200-input subset (JSON: {"inputs": [{"input_id": ...}]})')
    a.add_argument("--manifest", default=None, help="frozen 2,000-input manifest (JSONL with source_id)")
    a.add_argument("--finalization", default=None, help="finalization record written by finalize.py (v2.12 rule)")
    args = a.parse_args()
    cfg = load_config(args.config)
    R = Path(_expand(args.runs_root or cfg.get("runs_root") or str(R)))
    SUBSET = Path(_expand(args.subset or cfg.get("subset_inputs") or str(R / "inputs/subset_200_inputs.json")))
    FULL = Path(_expand(args.manifest or cfg.get("full_manifest") or str(R / "inputs/input_manifest_2000.jsonl")))
    FINALIZATION = args.finalization
    sub = json.load(open(SUBSET))
    ids200 = [r["input_id"] for r in (sub["inputs"] if isinstance(sub, dict) else sub)]
    ids2000 = [json.loads(l)["source_id"] for l in open(FULL) if l.strip()]
    methods = ["physx-omni-v0.1"] + list(METHODS) if args.method == "all" else [args.method]
    out = {"schema": "affordcraft.external_method_evidence.v2.13", "methods": {}}
    for m in methods:
        out["methods"][m] = run_one(m, ids200, ids2000)
        for k in ("subset200", "full2000"):
            s = out["methods"][m].get(k)
            if not s:
                continue
            print(
                m,
                k,
                "N",
                s["denominator"],
                "terminal",
                s["counts"]["chain_terminal"],
                "export",
                s["counts"]["export"],
                "native",
                s["counts"]["native_pass"],
                "n/a",
                s["counts"]["native_not_applicable"],
                "adapted",
                s["counts"]["adapted_pass"],
                "artic(native/adapted/export)",
                s["counts"]["native_pass_articulated"],
                s["counts"]["adapted_pass_articulated"],
                s["counts"]["export_articulated"],
                "infra",
                s["counts"]["infra_blocked"],
                "no_result",
                s["counts"]["no_result_yet"],
                "complete",
                s["complete"],
                "timing",
                {
                    kk: v
                    for kk, v in s["timing"].items()
                    if kk.endswith("_s") or kk.startswith("worker") or kk.startswith("peak")
                },
            )
    out["unparseable_mirror_files"] = sorted(set(UNPARSEABLE))
    Path(args.out).write_text(json.dumps(out, indent=1))
    Path(args.out).with_name("aggregate_unparseable.txt").write_text(
        chr(10).join(sorted(set(UNPARSEABLE))) + (chr(10) if UNPARSEABLE else "")
    )
    if UNPARSEABLE:
        print("UNPARSEABLE mirror files:", len(set(UNPARSEABLE)))
    print("written", args.out)


if __name__ == "__main__":
    main()
