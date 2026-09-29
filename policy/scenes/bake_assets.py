"""Bake the articulated scene assets at their scene scales (pure USD, no Isaac). Idempotent: skips
existing files whose source sha matches.

RESCALE is a uniform USD rescaling helper (called as `RESCALE SRC SCALE DST`) that lived in the scene run root
and is not part of this release; set AFFORDCRAFT_RESCALE_USDA to its location."""

import json, os, subprocess, sys, hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import scene_specs as S

PY = os.environ.get("AFFORDCRAFT_RUNTIME_PYTHON", sys.executable)
RESCALE = os.environ.get("AFFORDCRAFT_RESCALE_USDA", os.path.join(os.path.dirname(S.BAKED_DIR), "rescale_usda.py"))
inv = json.load(open(S.INVENTORY))
out = Path(S.BAKED_DIR)
out.mkdir(parents=True, exist_ok=True)
manifest = {}
for key, spec in S.ARTICULATED.items():
    src = os.path.expandvars(inv[spec["cid"]]["usd"])
    dst = out / f"{spec['cid']}_s{int(round(spec['scale']*100)):03d}.usda"
    if not dst.exists():
        print(
            subprocess.run(
                [PY, RESCALE, src, str(spec["scale"]), str(dst)], capture_output=True, text=True
            ).stdout.strip()
        )
    manifest[key] = dict(
        cid=spec["cid"],
        scale=spec["scale"],
        usd=str(dst),
        source=src,
        source_sha256=hashlib.sha256(Path(src).read_bytes()).hexdigest(),
        baked_sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),
    )
(out / "baked_manifest.json").write_text(json.dumps(manifest, indent=1))
print("baked", len(manifest))
