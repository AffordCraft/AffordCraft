#!/usr/bin/env python3
"""Clean-GPU timing: setup of the external-method runs over the 32-input timing sample (the scientific part of the
executed setup; copying the wrapper code of every registered run into <run root>/<method>/code is described in
README.md). Files are written once ('x' / exist_ok=False); a lane manifest that exists must match exactly.
  lanes <method dir> <n_lanes> [--no-write]
      <method dir>/lane-<k>/input_manifest.jsonl = the frozen 2,000-input manifest rows (unchanged) of the sample inputs
      whose sample index i has i % n_lanes == k, in sample order; <method dir>/lane_assignment.json. Lanes used in the
      paper's run: PhysX-Omni 4, PhysX-Anything 4, PAct 4 (--no-write: its lane runner writes its own lane manifests from
      the same rule), TRELLIS.2 2, PartCrafter 2.
  partcrafter-replay <registered subset run> <method dir>
      writes <method dir>/replay_table.json (per sample input: the part count and the call duration the registered run
      recorded, or its recorded VLM failure) and <method dir>/code/run_partcrafter_lane_replay.py = the registered lane
      runner code/run_partcrafter_lane.py with vlm_task replaced by the replay below, nothing else changed. The GPU
      stages (background removal, generation, export, render) are measured on the clean GPU, the VLM stage is the
      recorded API latency; no network call is made.
Run root layout: <run root>/inputs/{sample32.json, input_manifest.jsonl (frozen 2,000-input manifest)}.
usage: setup_timing_run.py [--root <run root>] lanes <method dir> <n> | partcrafter-replay <registered run> <method dir>
"""
import argparse, sys, os, json, hashlib, datetime, glob, socket

P = os.environ.get("AFFORDCRAFT_PROJECT_ROOT", "workspace")  # image paths of the manifest are relative to it
RT = os.path.join(os.environ.get("AFFORDCRAFT_RUNS", "runs"), "clean_timing")
FULL_SHA = (
    "139981fed148e2875f0e43a3d6e1cd30736d1db3088e4303b80d7a2f5d85a0df"  # frozen 2,000-input manifest of the paper
)
SAMPLE_SHA = "c4a2d8bfac3c166ad62a440fd3a09206f912401fc8591ca62ddcafc4da6d33c4"  # sample32.json of the paper's run
INPUT_SET = (
    "clean_timing_sample32: the first 32 inputs of the registered 200-input subset in sha256(input_id) "
    "order, kept in subset order (inputs/sample32.json)"
)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def lock(p):
    return {"path": p, "sha256": sha(p), "bytes": os.path.getsize(p)}


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def dump_x(p, obj):
    with open(p, "x") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def lanes(mdir, n, write=True):
    FULL = RT + "/inputs/input_manifest.jsonl"
    SAMPLE = RT + "/inputs/sample32.json"
    assert sha(FULL) == FULL_SHA, "frozen manifest drift"
    assert sha(SAMPLE) == SAMPLE_SHA, "sample drift"
    man = {}
    for l in open(FULL):
        if l.strip():
            r = json.loads(l)
            man[r["source_id"]] = r
    sample = json.load(open(SAMPLE))["inputs"]
    assert len(sample) == 32
    part = {k: [] for k in range(n)}
    for i, c in enumerate(sample):
        r = man[c["input_id"]]
        assert r["image"]["sha256"] == c["image"]["sha256"], c["input_id"]
        assert sha(os.path.join(P, r["image"]["path"])) == r["image"]["sha256"], c["input_id"]
        part[i % n].append((i, r))
    os.makedirs(mdir, exist_ok=True)
    if write:
        for k in range(n):
            d = f"{mdir}/lane-{k}"
            os.makedirs(d, exist_ok=True)
            p = d + "/input_manifest.jsonl"
            text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for _, r in part[k])
            if os.path.exists(p):
                assert open(p).read() == text, ("lane manifest drift", p)
            else:
                with open(p, "x") as f:
                    f.write(text)
    p = mdir + "/lane_assignment.json"
    if not os.path.exists(p):
        dump_x(
            p,
            {
                "schema": "affordcraft.clean_timing.lanes.v1",
                "method_dir": mdir,
                "n_lanes": n,
                "rule": "sample index i -> lane i % n_lanes, sample order; one lane per exclusive GPU",
                "lane_manifests_written": write,
                "sample": lock(SAMPLE),
                "manifest": lock(FULL),
                "input_set": INPUT_SET,
                "lanes": {
                    str(k): [{"sample_index": i, "source_id": r["source_id"]} for i, r in part[k]] for k in range(n)
                },
                "written": now(),
                "host": socket.gethostname(),
            },
        )
    return {k: [r["source_id"] for _, r in v] for k, v in part.items()}


REPLAY_TASK = '''    REPLAY = json.load(open(os.path.join(root, "replay_table.json")))["cases"]

    def vlm_task(row):
        """clean-timing replay, no network call: the part count the registered subset run received for this input
        (official suggest_num_parts, same model) and that call's recorded duration; a recorded VLM failure is replayed
        as the same failure. The GPU stages are measured, the VLM stage is the recorded API latency."""
        t0 = time.time()
        rec = REPLAY[row["source_id"]]
        out = {"num_parts": None, "error": None, "traceback": None, "seconds": rec["seconds"], "exchange": None, "started": now_iso()}
        if rec["error"] is not None:
            out["error"] = RuntimeError("replayed VLM failure of the registered run: " + str(rec["error"]))
        else:
            out["num_parts"] = rec["num_parts"]
        out["exchange"] = {"replayed_from": rec["source"], "replay_lookup_seconds": round(time.time() - t0, 6)}
        return out

'''


def partcrafter_replay(src, m):
    SAMPLE = RT + "/inputs/sample32.json"
    s = open(m + "/code/run_partcrafter_lane.py").read()
    a, b = s.index("    def vlm_task(row):"), s.index("    pool = ThreadPoolExecutor(")
    with open(m + "/code/run_partcrafter_lane_replay.py", "x") as f:
        f.write(s[:a] + REPLAY_TASK + s[b:])
    cases = {}
    for c in json.load(open(SAMPLE))["inputs"]:
        sid = c["input_id"]
        ds = [d for d in glob.glob(f"{src}/lane-*/cases/{sid}") if os.path.exists(d + "/result.json")]
        assert len(ds) == 1, (sid, ds)
        d = ds[0]
        res = json.load(open(d + "/result.json"))
        vr = json.load(open(d + "/vlm_reply.json"))
        tm = json.load(open(d + "/timing.json")) if os.path.exists(d + "/timing.json") else None
        assert (vr.get("error") is None) == (res.get("vlm_suggested_parts") is not None), sid
        cases[sid] = {
            "num_parts": res.get("vlm_suggested_parts"),
            "seconds": vr.get("seconds"),
            "error": vr.get("error"),
            "source": {
                "case_dir": d,
                "result_sha256": sha(d + "/result.json"),
                "vlm_reply_sha256": sha(d + "/vlm_reply.json"),
            },
            "campaign": {
                "asset_emitted": res.get("asset_emitted"),
                "failure_reason": res.get("failure_reason"),
                "timing": tm,
            },
        }
    dump_x(
        m + "/replay_table.json",
        {
            "schema": "affordcraft.clean_timing.partcrafter_replay.v1",
            "source_revision": src,
            "written": now(),
            "host": socket.gethostname(),
            "cases": cases,
        },
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=RT, help="run root of the clean timing run")
    ap.add_argument("--manifest-sha256", default=FULL_SHA, help="expected sha256 of inputs/input_manifest.jsonl")
    ap.add_argument(
        "--sample-sha256",
        default=SAMPLE_SHA,
        help="expected sha256 of inputs/sample32.json (the paper value holds for its own paths)",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    la = sub.add_parser("lanes")
    la.add_argument("method_dir")
    la.add_argument("n_lanes", type=int)
    la.add_argument("--no-write", action="store_true")
    pr = sub.add_parser("partcrafter-replay")
    pr.add_argument("registered_run")
    pr.add_argument("method_dir")
    args = ap.parse_args()
    RT, FULL_SHA, SAMPLE_SHA = args.root, args.manifest_sha256, args.sample_sha256
    if args.cmd == "lanes":
        lanes(args.method_dir, args.n_lanes, write=not args.no_write)
    else:
        partcrafter_replay(args.registered_run, args.method_dir)
    print("ok", args.cmd)
