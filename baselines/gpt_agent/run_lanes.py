"""Run controller: registers the run root (configuration.json first), partitions the 200-case subset into lanes and
processes them with N concurrent case threads sharing one catalog/encoder. Restart-safe: finished cases (result.json)
are skipped, half-done case directories are moved to interrupted/ and redone from scratch.

usage: python run_lanes.py --root <run_root> [--lanes 16] [--concurrency 16] [--smoke <source_id>] [--limit N]
"""

from __future__ import annotations
import argparse, json, os, shutil, sys, threading, time, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    PROJECT,
    INPUT_MANIFEST,
    SUBSET_JSON,
    CONTRACT_MD,
    PROTOCOL,
    MODEL_ID,
    METHOD_ID,
    PREVIEW_MAX_SIDE,
    RENDER_SIZE,
    IMAGE_CONTEXT_KEEP_ROUNDS,
    now_iso,
    sha256_file,
    sha256_bytes,
    canonical_json,
    write_json,
    read_json,
)
from geometry import Catalog
from agent_loop import ApiClient, CaseRunner, SYSTEM_PROMPT, TOOLS

HARNESS_FILES = [
    "common.py",
    "geometry.py",
    "render_worker.py",
    "agent_loop.py",
    "run_lanes.py",
    "offline_test.py",
    "status.py",
]


def load_cases():
    subset = read_json(SUBSET_JSON)
    rows = [json.loads(l) for l in INPUT_MANIFEST.read_text().splitlines() if l.strip()]
    by_sid = {r["source_id"]: r for r in rows}
    cases = []
    for i, item in enumerate(subset["inputs"]):
        sid = item["input_id"]
        r = by_sid[sid]
        path = PROJECT / r["image"]["path"]
        cases.append(
            {
                "index": i,
                "source_id": sid,
                "requested_category": r["requested_category"],
                "image_path": str(path),
                "image_sha256": r["image"]["sha256"],
            }
        )
    return cases, subset


def verify_image(case):
    digest = sha256_file(case["image_path"])
    if digest != case["image_sha256"]:
        raise RuntimeError(f"image hash mismatch for {case['source_id']}")
    return digest


def register(root: Path, cases, subset, lanes, concurrency, catalog: Catalog, tool_choice):
    root.mkdir(parents=True, exist_ok=True)
    cfg_path = root / "configuration.json"
    harness_dir = Path(__file__).resolve().parent
    harness_hashes = {f: sha256_file(harness_dir / f) for f in HARNESS_FILES if (harness_dir / f).exists()}
    if not cfg_path.exists():
        cfg = {
            "revision": root.name,
            "method_id": METHOD_ID,
            "created": now_iso(),
            "method": "general-agent baseline: gpt-6-astra tool-calling agent assembling an interactive asset from one photograph with modelling tools and the AffordCraft catalog",
            "model": {
                "id": MODEL_ID,
                "api": "OpenAI-compatible /chat/completions via an OpenAI-compatible endpoint (EVAL_API_BASE / EVAL_API_KEY)",
                "reasoning_effort_requested": PROTOCOL["reasoning_effort"],
                "reasoning_effort_used": PROTOCOL["reasoning_effort"],
                "reasoning_effort_probe": "the API endpoint accepts 'none','minimal','low','medium','high','xhigh','max' (invalid values are rejected with HTTP 400); 'max' accepted",
                "temperature": PROTOCOL["temperature"],
                "max_completion_tokens_per_call": PROTOCOL["per_call_max_tokens"],
                "max_tokens_note": "the API endpoint accepted both max_tokens and max_completion_tokens but did not truncate (a 60-token cap returned 2.2K completion tokens); the per-call cap is therefore requested but the cumulative budget is enforced by the harness after every call",
                "tool_choice": tool_choice,
                "per_call_timeout_seconds": PROTOCOL["per_call_timeout_seconds"],
                "per_call_total_timeout_seconds": PROTOCOL["per_call_total_timeout_seconds"],
                "transport": "server-sent-event streaming (stream=true, stream_options.include_usage) because the gateway of the API endpoint returns HTTP 504 after 300 s for non-streamed calls; chunks are re-assembled into one message; the 600 s timeout is an inactivity timeout, the total cap 1800 s",
                "max_retries": PROTOCOL["max_retries"],
                "retry_backoff": "exponential 5 s .. 160 s on HTTP 429/5xx/network errors/stream errors; HTTP 4xx other than 429 ends the case with api_error (context-length 4xx: up to 3 image-pruning retries first)",
            },
            "budgets": {
                "max_tool_rounds": PROTOCOL["max_rounds"],
                "cumulative_completion_tokens": PROTOCOL["max_completion_tokens_total"],
                "wall_clock_seconds_per_case": PROTOCOL["wall_clock_seconds"],
                "enforcement": "checked before every model call and after every response; exceeding any budget ends the case with asset_emitted:false and failure_reason budget_exhausted:<rounds|completion_tokens|wall_clock>; completion tokens include reasoning tokens as reported by usage.completion_tokens",
            },
            "protocol": {
                "input": "whole photograph (original JPEG bytes, hash-verified) + the task instruction with the requested category; no box, mask, annotation, pipeline output or evaluation answer",
                "instruction_template": "Construct an interactive simulation asset for the <Category> visible in this image. Preserve its mechanism and natural support conditions.",
                "isolation": "one fresh conversation per case; no shared state between cases except read-only catalog caches",
                "catalog_access": "search_catalog runs the frozen CatalogIndex.retrieve (DINOv2-large CLS index visual-large-001) on the whole photograph or an agent-chosen crop; inspect_entry/catalog imports read the same catalog sources through the frozen source_parser",
                "adapted_from": "single-image agent baseline of Guo (2026, GPT6-real2sim); multi-view episode replay not reproduced",
                "no_result_selection": True,
                "append_only": True,
                "failed_cases_stay_failed": True,
                "tool_image_delivery": "the API endpoint drops image parts inside tool messages (verified), so images returned by tools are appended as a user message right after the tool results, labelled with the tool call id",
                "image_context_policy": f"tool images older than {IMAGE_CONTEXT_KEEP_ROUNDS} rounds are replaced by text stubs in later requests (the input photograph is always kept); previews are downscaled to max side {PREVIEW_MAX_SIDE} px JPEG; assembly renders are {RENDER_SIZE[0]}x{RENDER_SIZE[1]}",
                "non_tool_reply_handling": "a reply without tool calls receives a user-role message 'No tool was called. Continue by calling a tool; only submit_asset ends the task.' and counts as a round",
                "native_export": "harness materialises the submitted spec: baked OBJ per link (uniform scale, part pose, joint rotations applied; link frames world-aligned at rest with origin at the joint pivot), URDF (visual+collision mesh, hull inertials), MJCF (mesh geoms with density or mass, hinge/slide joints, freejoint root when free_standing); native_manifest.json per the external contract",
            },
            "inputs": {
                "manifest": str(INPUT_MANIFEST),
                "manifest_sha256": sha256_file(INPUT_MANIFEST),
                "subset": str(SUBSET_JSON),
                "subset_sha256": sha256_file(SUBSET_JSON),
                "image_root": str(PROJECT),
                "n_cases": len(cases),
                "order": "registered_subset_order",
                "image_hashes_verified": True,
                "per_case_hash_check": "the case runner re-verifies each image sha256 before use",
            },
            "lanes": {
                "count": lanes,
                "partition": f"case index i (subset order) -> lane i mod {lanes}; each lane processes its cases in order",
                "concurrency": concurrency,
                "concurrency_policy": 'cases in flight limited by a semaphore; automatically reduced to 8 when >= 8 HTTP 429 responses occur within 5 minutes; control.json {"concurrency": n} overrides at runtime; changes are logged in events.jsonl',
            },
            "tools": [
                {"name": t["function"]["name"], "schema_sha256": sha256_bytes(canonical_json(t).encode()), "schema": t}
                for t in TOOLS
            ],
            "tools_sha256": sha256_bytes(canonical_json(TOOLS).encode()),
            "system_prompt": SYSTEM_PROMPT,
            "system_prompt_sha256": sha256_bytes(SYSTEM_PROMPT.encode()),
            "harness": {"files": harness_hashes, "python": sys.executable, "copy": str(root / "harness")},
            "catalog": catalog.identity(),
            "contract": {
                "path": str(CONTRACT_MD),
                "sha256": sha256_file(CONTRACT_MD) if CONTRACT_MD.exists() else None,
            },
            "host": os.uname().nodename,
        }
        write_json(cfg_path, cfg, exclusive=True)
        (root / "harness").mkdir(exist_ok=True)
        for f in HARNESS_FILES:
            if (harness_dir / f).exists():
                shutil.copy2(harness_dir / f, root / "harness" / f)
        with (root / "lane_assignment.jsonl").open("x") as fh:
            for c in cases:
                fh.write(
                    json.dumps(
                        {
                            "index": c["index"],
                            "source_id": c["source_id"],
                            "lane": c["index"] % lanes,
                            "requested_category": c["requested_category"],
                        }
                    )
                    + "\n"
                )
    else:
        cfg = read_json(cfg_path)
        drift = {f: h for f, h in harness_hashes.items() if cfg["harness"]["files"].get(f) != h}
        if drift:
            # amended harness: keep the new copy next to the original one (append-only) and record it
            existing = sorted(p.name for p in (root / "harness").glob("amendment-*"))
            latest = read_json(root / "harness" / existing[-1] / "hashes.json") if existing else cfg["harness"]["files"]
            if any(latest.get(f) != h for f, h in harness_hashes.items()):
                dest = root / "harness" / f"amendment-{len(existing) + 1:03d}-{time.strftime('%Y%m%d-%H%M%S')}"
                dest.mkdir(parents=True)
                for f in HARNESS_FILES:
                    if (harness_dir / f).exists():
                        shutil.copy2(harness_dir / f, dest / f)
                write_json(dest / "hashes.json", harness_hashes)
                with (root / "events.jsonl").open("a") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "t": now_iso(),
                                "event": "harness_amendment_on_restart",
                                "files_changed_vs_configuration": drift,
                                "copy": str(dest),
                            }
                        )
                        + "\n"
                    )
    return cfg


class Controller:
    def __init__(self, root: Path, cases, lanes, concurrency, catalog, client, tool_choice):
        self.root = root
        self.cases = cases
        self.lanes = lanes
        self.catalog = catalog
        self.client = client
        self.tool_choice = tool_choice
        self.concurrency = concurrency
        self.sem = threading.Semaphore(concurrency)
        self.sem_lock = threading.Lock()
        self.events = (root / "events.jsonl").open("a", encoding="utf-8")
        self.elock = threading.Lock()
        self.results = {}
        self.inflight = {}
        self.stop = False
        self.drain = False

    def event(self, e):
        with self.elock:
            self.events.write(json.dumps({"t": now_iso(), **e}, ensure_ascii=False) + "\n")
            self.events.flush()

    def set_concurrency(self, n, reason):
        with self.sem_lock:
            if n == self.concurrency:
                return
            if n > self.concurrency:
                for _ in range(n - self.concurrency):
                    self.sem.release()
            else:
                # acquire the difference lazily: mark pending reductions
                self._pending_reduce = getattr(self, "_pending_reduce", 0) + (self.concurrency - n)
            self.event({"event": "concurrency_change", "from": self.concurrency, "to": n, "reason": reason})
            self.concurrency = n

    def acquire(self):
        while True:
            self.sem.acquire()
            with self.sem_lock:
                pend = getattr(self, "_pending_reduce", 0)
                if pend > 0:
                    self._pending_reduce = pend - 1
                    continue  # swallow this permit to realise a reduction
            return

    def release(self):
        self.sem.release()

    def case_dir(self, c):
        return self.root / f"lane-{c['index'] % self.lanes}" / "cases" / c["source_id"]

    def lane_worker(self, lane):
        for c in self.cases:
            if c["index"] % self.lanes != lane or self.stop:
                continue
            d = self.case_dir(c)
            if (d / "result.json").exists():
                self.results[c["source_id"]] = read_json(d / "result.json")
                continue
            if d.exists():
                dest = self.root / "interrupted" / f"{c['source_id']}-{time.strftime('%Y%m%d-%H%M%S')}"
                dest.parent.mkdir(exist_ok=True)
                shutil.move(str(d), str(dest))
                self.event({"event": "interrupted_case_moved", "source_id": c["source_id"], "to": str(dest)})
            if self.drain:
                break
            self.acquire()
            try:
                if self.stop or self.drain:
                    break
                verify_image(c)
                self.inflight[c["source_id"]] = time.time()
                self.event({"event": "case_start", "source_id": c["source_id"], "lane": lane, "index": c["index"]})
                runner = CaseRunner(
                    self.catalog, self.client, {**c, "lane": lane}, d, self.root.name, tool_choice=self.tool_choice
                )
                res = runner.run()
                self.results[c["source_id"]] = res
                self.event(
                    {
                        "event": "case_end",
                        "source_id": c["source_id"],
                        "lane": lane,
                        "asset_emitted": res["asset_emitted"],
                        "failure_reason": res["failure_reason"],
                        "rounds": res["rounds"],
                        "completion_tokens": res["completion_tokens"],
                        "prompt_tokens": res["prompt_tokens"],
                        "runtime_seconds": res["runtime_seconds"],
                    }
                )
            except Exception as exc:
                self.event(
                    {
                        "event": "case_crash",
                        "source_id": c["source_id"],
                        "lane": lane,
                        "error": repr(exc)[:400],
                        "traceback": traceback.format_exc()[-3000:],
                    }
                )
            finally:
                self.inflight.pop(c["source_id"], None)
                self.release()

    def monitor(self):
        last_429_check = 0
        while not self.stop:
            time.sleep(30)
            try:
                ctl = self.root / "control.json"
                if ctl.exists():
                    try:
                        c = read_json(ctl)
                        want = int(c.get("concurrency", self.concurrency))
                        if 1 <= want <= 64 and want != self.concurrency:
                            self.set_concurrency(want, "control.json")
                        if c.get("drain") and not self.drain:
                            self.drain = True
                            self.event(
                                {
                                    "event": "drain_requested",
                                    "note": "no new cases are started; the controller exits when the in-flight cases finish",
                                }
                            )
                    except Exception:
                        pass
                hits = len([x for x in self.client.rate_limit_hits if time.time() - x < 300])
                if hits >= 8 and self.concurrency > 8:
                    self.set_concurrency(8, f"{hits} HTTP 429 responses within 5 minutes")
                done = [r for r in self.results.values()]
                prog = {
                    "t": now_iso(),
                    "finished": len(done),
                    "total": len(self.cases),
                    "emitted": sum(1 for r in done if r["asset_emitted"]),
                    "budget_exhausted": sum(
                        1 for r in done if (r.get("failure_reason") or "").startswith("budget_exhausted")
                    ),
                    "errors": sum(
                        1
                        for r in done
                        if r.get("failure_reason") and not r["failure_reason"].startswith("budget_exhausted")
                    ),
                    "in_flight": {k: round(time.time() - v) for k, v in self.inflight.items()},
                    "concurrency": self.concurrency,
                    "api_stats": dict(self.client.stats),
                    "completion_tokens_total": sum(r["completion_tokens"] for r in done),
                    "prompt_tokens_total": sum(r["prompt_tokens"] for r in done),
                }
                write_json(self.root / "progress.json", prog)
            except Exception as exc:
                self.event({"event": "monitor_error", "error": repr(exc)[:300]})

    def run(self):
        threads = [
            threading.Thread(target=self.lane_worker, args=(k,), name=f"lane-{k}", daemon=True)
            for k in range(self.lanes)
        ]
        mon = threading.Thread(target=self.monitor, name="monitor", daemon=True)
        for t in threads:
            t.start()
        mon.start()
        for t in threads:
            t.join()
        self.stop = True
        mon.join(timeout=5)
        done = list(self.results.values())
        summary = {
            "t": now_iso(),
            "finished": len(done),
            "total": len(self.cases),
            "emitted": sum(1 for r in done if r["asset_emitted"]),
            "failure_reasons": {},
            "api_stats": dict(self.client.stats),
        }
        for r in done:
            fr = r.get("failure_reason") or "none"
            summary["failure_reasons"][fr] = summary["failure_reasons"].get(fr, 0) + 1
        write_json(
            self.root / "progress.json",
            {
                **summary,
                "in_flight": {},
                "concurrency": self.concurrency,
                "completion_tokens_total": sum(r["completion_tokens"] for r in done),
                "prompt_tokens_total": sum(r["prompt_tokens"] for r in done),
            },
        )
        self.event({"event": "controller_end", **summary})
        return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--lanes", type=int, default=16)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--smoke", default=None, help="run only this source_id (lane partition is still index mod lanes)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tool-choice", default="required")
    args = ap.parse_args()
    root = Path(args.root)
    cases, subset = load_cases()
    if args.smoke:
        cases = [c for c in cases if c["source_id"] == args.smoke]
        if not cases:
            raise SystemExit("unknown smoke case")
    if args.limit:
        cases = cases[: args.limit]
    for c in cases:
        verify_image(c)
    catalog = Catalog(device="cuda:0")
    cfg = register(root, cases, subset, args.lanes, args.concurrency, catalog, args.tool_choice)
    client = ApiClient()
    ctl = Controller(root, cases, args.lanes, args.concurrency, catalog, client, args.tool_choice)
    ctl.event(
        {
            "event": "controller_start",
            "pid": os.getpid(),
            "cases": len(cases),
            "lanes": args.lanes,
            "concurrency": args.concurrency,
            "host": os.uname().nodename,
        }
    )
    summary = ctl.run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
