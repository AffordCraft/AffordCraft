#!/usr/bin/env python3
"""Clean-GPU timing: preparation of one AffordCraft shard on the machine that runs it.
usage: prepare_affordcraft_shard.py <run root> <regime cold|warm> <shard> <gpu> [<campaign geometry cache>]
       (the cache defaults to $AFFORDCRAFT_GEOMETRY_CACHE, else $AFFORDCRAFT_RUNS/main_campaign/geometry_cache)
1. every code/model lock of the main-campaign definition (<run root>/inputs/locks_exp1.json: {"locks": [{"path",
   "sha256"}]}, 65 paths in the paper's run) and every sample image is hashed; any mismatch -> exit 3 and nothing runs
   (run_pipeline --calibration skips the freeze check, this is the same check done here)
2. the shard's own geometry cache <run root>/ours-<regime>/cache-shard<k>: hard links of the campaign cache's level
   dirs (<key>/level_<n>/, published atomically and never modified afterwards).
   cold: the level dirs whose result.json is older than the start of the main campaign (CUT, the start time of its
        first worker), i.e. the cache that campaign started from; nothing built by it or later.
   warm: every level dir present now (the cache after the 2000-input campaign and the external gates).
   The run publishes new builds into its own copy only.
3. writes <run root>/ours-<regime>/prep-shard<k>.json: checks, snapshot counts and a digest of the linked level list,
   the machine's GPU/driver/CPU quota/memory limit/load, compute processes already on the GPU."""
import sys, os, json, hashlib, subprocess, time, socket

rt, reg, k, gpu = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
assert reg in ("cold", "warm")
# geometry cache of the main campaign (the builder's --cache directory): argument 5, else AFFORDCRAFT_GEOMETRY_CACHE
SRC = (
    sys.argv[5]
    if len(sys.argv) > 5
    else (
        os.environ.get("AFFORDCRAFT_GEOMETRY_CACHE")
        or os.path.join(os.environ.get("AFFORDCRAFT_RUNS", "runs"), "main_campaign/geometry_cache")
    )
)
CUT = 1789667914  # unix time at which the main campaign's first worker started


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sh(c):
    try:
        return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:
        return repr(e)


t0 = time.time()
locks = json.load(open(rt + "/inputs/locks_exp1.json"))["locks"]
bad = [x["path"] for x in locks if not os.path.exists(x["path"]) or sha(x["path"]) != x["sha256"]]
sample = json.load(open(rt + "/inputs/sample32.json"))
badimg = [
    c["input_id"]
    for c in sample["inputs"]
    if not os.path.exists(c["image"]["path"]) or sha(c["image"]["path"]) != c["image"]["sha256"]
]
rec = {
    "schema": "affordcraft.revision072.prep.v1",
    "regime": reg,
    "shard": k,
    "gpu": gpu,
    "host": socket.gethostname(),
    "locks_checked": len(locks),
    "locks_mismatch": bad,
    "images_checked": len(sample["inputs"]),
    "images_mismatch": badimg,
    "check_seconds": round(time.time() - t0, 1),
}
out = f"{rt}/ours-{reg}/prep-shard{k}.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
if bad or badimg:
    json.dump(rec, open(out, "w"), indent=1)
    print("prep: lock/image mismatch", bad[:3], badimg[:3], flush=True)
    sys.exit(3)
snap = f"{rt}/ours-{reg}/cache-shard{k}"
os.makedirs(snap, exist_ok=False)
t1 = time.time()
levels = files = 0
skipped = 0
h = hashlib.sha256()
for key in sorted(os.listdir(SRC)):
    kd = os.path.join(SRC, key)
    if not os.path.isdir(kd):
        continue
    for lv in sorted(os.listdir(kd)):
        ld = os.path.join(kd, lv)
        rj = os.path.join(ld, "result.json")
        if not lv.startswith("level_") or not os.path.isfile(rj):
            continue
        if reg == "cold" and os.stat(rj).st_mtime >= CUT:
            skipped += 1
            continue
        dd = os.path.join(snap, key, lv)
        os.makedirs(dd)
        for fn in os.listdir(ld):
            fp = os.path.join(ld, fn)
            if os.path.isfile(fp):
                os.link(fp, os.path.join(dd, fn))
                files += 1
        levels += 1
        h.update(f"{key}/{lv}\n".encode())
lim = sh("cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes")
cpu = sh(
    "cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo $(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us) $(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)"
)
rec.update(
    {
        "geometry_cache_source": SRC,
        "snapshot": snap,
        "cutoff_unix": CUT if reg == "cold" else None,
        "linked_levels": levels,
        "linked_files": files,
        "levels_newer_than_cutoff_left_out": skipped,
        "linked_level_list_sha256": h.hexdigest(),
        "snapshot_seconds": round(time.time() - t1, 1),
        "gpu_query": sh(
            "nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version --format=csv,noheader"
        ),
        "compute_apps_before": sh("nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader"),
        "cpu_quota": cpu,
        "memory_limit_bytes": lim,
        "nproc": sh("nproc"),
        "loadavg": sh("cat /proc/loadavg"),
        "cpu_model": sh("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"),
        "prepared_unix": time.time(),
    }
)
json.dump(rec, open(out, "w"), indent=1)
print(
    json.dumps(
        {
            k2: rec[k2]
            for k2 in (
                "regime",
                "shard",
                "linked_levels",
                "levels_newer_than_cutoff_left_out",
                "check_seconds",
                "snapshot_seconds",
            )
        }
    ),
    flush=True,
)
