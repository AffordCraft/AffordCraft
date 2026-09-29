"""Evidence-preserving process boundaries; no geometry or physics gate changes."""

from pathlib import Path
import hashlib
import json
import os
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def atomic_status(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".pending")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def known_coacd_abort(returncode, log):
    # Narrow, frozen signature established by an isolated CPU reproduction.
    # OOM/SIGKILL, other SIGABRT/SIGSEGV and dependency errors are NOT accepted.
    return (
        returncode == -6
        and "std::logic_error" in log
        and "unexpected code path was hit" in log
        and "coacd/__init__.py" in log
        and "run_coacd" in log
    )


def completion(records, expected, outcomes):
    return (
        len(records) == expected
        and len({r["input_id"] for r in records}) == expected
        and all(type(r.get("physical_pass")) is bool for r in records)
        and set(outcomes) == {"construction", "final"}
        and all(r.get("exit_code") == 0 and r.get("in_process_completed") is True for r in outcomes.values())
    )


def supervised_completion(exit_code, summary):
    return exit_code == 0 and isinstance(summary, dict) and summary.get("complete") is True


def health_probe(code_root, python, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    cmd = [python, str(Path(code_root) / "scripts/coacd_health.py"), "--output", str(directory / "result.json")]
    started = time.perf_counter()
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONFAULTHANDLER": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    with (directory / "stdout.log").open("x") as log:
        try:
            proc = subprocess.run(
                cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=90
            )
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = None
    result = directory / "result.json"
    data = json.loads(result.read_text()) if result.exists() else {}
    evidence = {
        "command": cmd,
        "exit_code": rc,
        "measured_runtime_seconds": time.perf_counter() - started,
        "passed": rc == 0 and data.get("passed") is True,
        "result": data,
        "log_sha256": sha(directory / "stdout.log"),
    }
    save_new(directory / "execution_receipt.json", evidence)
    return evidence
