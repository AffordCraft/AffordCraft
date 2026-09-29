"""Supervise only this job and preserve process and scientific outcomes separately."""

from pathlib import Path
import argparse, json, os, subprocess, sys, time

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from affordcraft import paths


def save(p, v):
    p.write_text(json.dumps(v, indent=2), encoding="utf-8")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--jobs", required=True)
    a.add_argument("--output", required=True)
    a.add_argument("--gpu", type=int, default=0)
    args = a.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=True,
    )
    row = next(x for x in gpu.stdout.splitlines() if int(x.split(",")[0]) == args.gpu)
    if int(row.split(",")[-1]) < 12000:
        raise RuntimeError("Insufficient GPU headroom; no other processes will be stopped")
    cmd = [
        paths.ISAAC_PYTHON,
        str(R / "scripts/physics_worker.py"),
        "--jobs",
        args.jobs,
        "--output",
        str(out / "runtime"),
        "--gpu",
        str(args.gpu),
    ]
    env = {
        **os.environ,
        "PYTHONPATH": str(R) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
    }
    start = time.time()
    with (out / "stdout.log").open("w") as log:
        child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env)
        save(
            out / "process.json",
            {
                "pid": child.pid,
                "supervisor_pid": os.getpid(),
                "argv": cmd,
                "gpu_preflight": row,
                "started": start,
                "license_acceptance_performed": False,
                "other_processes_terminated": False,
            },
        )
        print(json.dumps({"pid": child.pid, "output": str(out), "gpu": args.gpu}), flush=True)
        while child.poll() is None:
            time.sleep(3)
            done = out / "runtime/in_process_completion.json"
            save(
                out / "status.json",
                {"running": True, "elapsed_seconds": time.time() - start, "in_process_completed": done.exists()},
            )
        rc = child.returncode
    done = out / "runtime/in_process_completion.json"
    science = json.loads(done.read_text()) if done.exists() else None
    receipt = {
        "exit_code": rc,
        "in_process": science,
        "wall_seconds": time.time() - start,
        "complete": rc == 0 and science is not None,
        "process_science_disagreement": rc != 0 and science is not None,
    }
    save(out / "execution_receipt.json", receipt)
    save(out / "status.json", {"running": False, "exit_code": rc, "in_process_completed": science is not None})
    print(
        json.dumps(
            {"exit_code": rc, "in_process_completed": science is not None, "wall_seconds": receipt["wall_seconds"]}
        ),
        flush=True,
    )
    raise SystemExit(0 if receipt["complete"] else 1)


if __name__ == "__main__":
    main()
