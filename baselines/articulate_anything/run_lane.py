#!/usr/bin/env python
"""run_lane.py -- process the cases of one lane (<run root>/lane-<k>) sequentially, append-only.

For every row of lane-<k>/input_manifest.jsonl: skip when cases/<source_id>/result.json exists; a case directory without
result.json (interrupted earlier run) is moved to lane-<k>/interrupted/ (kept) before recomputation; run_case.py runs in
its own process group with a hard 3600 s limit; a timeout or crash without result.json is finalized as a failed case
(result/timing/native_manifest with the failure reason) so that no case is ever dropped from the denominator.
The lane's execution_receipt.json is written at the end (lane_receipt.py)."""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CASE_TIMEOUT_SECONDS, CODE, P, PY, REV, log, move_to_interrupted, rows_of, utc


def run_case(lane_dir, row, lane_fh, env):
    sid = row["source_id"]
    case_dir = lane_dir / "cases" / sid
    image = str(P / row["image"]["path"])
    if (case_dir / "result.json").exists():
        log(lane_fh, "skip finished %s" % sid)
        return "skipped"
    if case_dir.exists():
        dest = move_to_interrupted(case_dir, lane_dir / "interrupted", "restart")
        log(lane_fh, "moved interrupted case dir %s -> %s" % (sid, dest))
    case_dir.mkdir(parents=True)
    cmd = [
        PY,
        str(CODE / "run_case.py"),
        "--source-id",
        sid,
        "--image",
        image,
        "--image-sha256",
        row["image"]["sha256"],
        "--category",
        row["requested_category"],
        "--case-dir",
        str(case_dir),
    ]
    log(lane_fh, "start %s category=%s" % (sid, row["requested_category"]))
    t0 = time.time()
    forced = None
    with open(case_dir / "case.log", "a", encoding="utf-8") as clog:
        clog.write("%s %s\n" % (utc(), " ".join(cmd)))
        clog.flush()
        proc = subprocess.Popen(
            cmd, stdout=clog, stderr=subprocess.STDOUT, env=env, start_new_session=True, cwd=env["AA_SRC"]
        )
        try:
            rc = proc.wait(timeout=CASE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            forced = "timeout_%ds" % CASE_TIMEOUT_SECONDS
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            rc = proc.wait()
            clog.write("%s KILLED after %d s (rc=%s)\n" % (utc(), CASE_TIMEOUT_SECONDS, rc))
    if not (case_dir / "result.json").exists():
        forced = forced or ("runner_crash_exit_%s" % rc)
        fin = cmd + ["--finalize-only", "--forced-failure", forced]
        with open(case_dir / "case.log", "a", encoding="utf-8") as clog:
            clog.write("%s finalize-only: %s\n" % (utc(), forced))
            clog.flush()
            subprocess.run(fin, stdout=clog, stderr=subprocess.STDOUT, env=env, cwd=env["AA_SRC"])
    status = "failed(%s)" % forced if forced else ("exported" if rc == 0 else "failed(rc=%s)" % rc)
    log(lane_fh, "end %s %s %.0fs" % (sid, status, time.time() - t0))
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", type=int, required=True)
    ap.add_argument("--lanes", type=int, required=True)
    ap.add_argument("--only", default=None, help="comma-separated source_ids (smoke test)")
    args = ap.parse_args()
    lane_dir = REV / ("lane-%d" % args.lane)
    manifest = lane_dir / "input_manifest.jsonl"
    if not manifest.exists():
        raise SystemExit("lane manifest missing: %s (run make_lane_manifests.py first)" % manifest)
    if not (REV / "configuration.json").exists():
        raise SystemExit("configuration.json missing in %s (register_configuration.py must run before any lane)" % REV)
    rows = rows_of(manifest)
    if args.only:
        keep = set(args.only.split(","))
        rows = [r for r in rows if r["source_id"] in keep]
    env = dict(os.environ)
    for k in ("EVAL_API_BASE", "EVAL_API_KEY", "AA_SRC", "AA_OPENAI_COMPAT", "AA_RENDER_SCRIPT"):
        if k not in env:
            raise SystemExit("environment variable %s missing" % k)
    (lane_dir / "cases").mkdir(parents=True, exist_ok=True)
    with open(lane_dir / "lane_runner.log", "a", encoding="utf-8") as lane_fh:
        log(
            lane_fh,
            "lane %d/%d start host=%s pid=%d rows=%d"
            % (args.lane, args.lanes, os.uname().nodename, os.getpid(), len(rows)),
        )
        counts = {}
        for row in rows:
            try:
                st = run_case(lane_dir, row, lane_fh, env)
            except Exception as e:  # noqa - keep the lane alive
                st = "runner_exception"
                log(lane_fh, "runner exception on %s: %r" % (row["source_id"], e))
            counts[st] = counts.get(st, 0) + 1
        log(lane_fh, "lane %d done counts=%s" % (args.lane, counts))
    if not args.only:
        subprocess.run(
            [PY, str(CODE / "lane_receipt.py"), str(lane_dir), str(args.lane), str(args.lanes)],
            env=env,
            cwd=env["AA_SRC"],
        )


if __name__ == "__main__":
    main()
