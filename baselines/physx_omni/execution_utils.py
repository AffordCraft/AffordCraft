#!/usr/bin/env python3
"""Append-only provenance helpers used by omni_geometry_runner.py.

Excerpt of the executed helper module: only the definitions the geometry runner imports (ROBOT, now, canonical,
sha, read, rows, once, lines_once, lock, capture), copied verbatim. The rest of that module (campaign registration,
logged command execution, benchmark download) is not used by the runner and is not published.
"""
from __future__ import annotations
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROBOT = dict(robot_real_world="not_evaluated", robot_control_status="not_evaluated", robot_task_success=None)


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(x):
    return (json.dumps(x, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def rows(p):
    with Path(p).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def once(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(value)
    if p.exists():
        if p.read_bytes() != data:
            raise RuntimeError(f"append-only conflict: {p}")
        return
    with p.open("xb") as f:
        f.write(data)


def lines_once(p, seq):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("xb") as f:
        for x in seq:
            f.write(canonical(x))


def lock(p):
    p = Path(p).resolve()
    return dict(path=str(p), sha256=sha(p), size_bytes=p.stat().st_size)


def capture(command, cwd=None):
    p = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return dict(command=command, returncode=p.returncode, output=p.stdout)
