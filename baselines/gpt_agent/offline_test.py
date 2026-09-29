"""Offline unit test of the tools (no API): search -> inspect -> primitive -> render -> submit on one subset case,
using a scripted fake model. Writes into a scratch run root and checks the native manifest against the contract keys.
usage: python offline_test.py <scratch_root> [source_id]
"""

from __future__ import annotations
import json, shutil, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_json
from run_lanes import load_cases, verify_image
from geometry import Catalog
from agent_loop import CaseRunner, TOOLS

CONTRACT_KEYS = [
    "schema",
    "source_id",
    "method_id",
    "run_id",
    "asset_emitted",
    "stage_reached",
    "failure_reason",
    "units",
    "metric_size_source",
    "native_formats",
    "mjcf",
    "urdf",
    "root_link",
    "links",
    "joints",
    "physics_contract",
    "notes",
]
LINK_KEYS = ["name", "visual_obj", "mesh_scale", "density_kg_m3", "mass_kg", "origin_xyz", "origin_rpy"]
JOINT_KEYS = ["name", "type", "parent", "child", "axis", "origin_xyz", "lower", "upper"]


class FakeClient:
    """Replays a fixed script of tool calls; the last one is submit_asset."""

    def __init__(self, script):
        self.script = list(script)
        self.rate_limit_hits = []
        self.stats = {"calls": 0, "retries": 0, "http_429": 0, "http_5xx": 0, "network_errors": 0}
        self.last_tool_results = []

    def chat(self, payload, log=None, **kw):
        self.stats["calls"] += 1
        # record the last tool results to let the script react
        self.last_tool_results = [m for m in payload["messages"] if m["role"] == "tool"]
        step = self.script.pop(0)
        if callable(step):
            step = step(self)
        name, args = step
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f'call_{self.stats["calls"]}',
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            ],
        }
        return {
            "id": "fake",
            "model": "fake",
            "choices": [{"message": msg, "finish_reason": "tool_calls"}],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 50,
                "completion_tokens_details": {"reasoning_tokens": 20},
            },
        }, [{"attempt": 0, "status": 200, "seconds": 0.0}]


def main():
    scratch = Path(sys.argv[1])
    sid = sys.argv[2] if len(sys.argv) > 2 else None
    cases, subset = load_cases()
    case = next(c for c in cases if sid is None or c["source_id"] == sid)
    verify_image(case)
    print("case", case["source_id"], case["requested_category"])
    t0 = time.time()
    catalog = Catalog(device="cuda:0")
    print("catalog+encoder loaded", round(time.time() - t0, 1), "s")
    state = {}

    def after_search(client):
        res = json.loads(client.last_tool_results[-1]["content"])
        assert res["returned"] >= 1, res
        eid = res["entries"][0]["entry_id"]
        # prefer an articulated entry if any
        for e in res["entries"]:
            if e.get("source_type") == "partnet_mobility_urdf":
                eid = e["entry_id"]
                break
        state["eid"] = eid
        return ("inspect_entry", {"entry_id": eid})

    def after_inspect(client):
        res = json.loads(client.last_tool_results[-1]["content"])
        assert "links" in res and res["link_count"] >= 1, res
        state["inspect"] = res
        return ("make_primitive", {"kind": "box", "size_m": [0.6, 0.4, 0.02]})

    def build_spec(client):
        ins = state["inspect"]
        ext = ins["extent_source_units"]
        scale = 0.8 / max(ext)
        zmin = ins["bounds_min"][2] * scale
        spec = {
            "parts": [
                {
                    "id": "base",
                    "source": {"type": "primitive", "part_id": "prim_0"},
                    "density_kg_m3": 700,
                    "pose": {"xyz": [0, 0, 0.01], "rpy": [0, 0, 0]},
                },
                {
                    "id": "obj",
                    "source": {"type": "catalog_entry", "entry_id": state["eid"]},
                    "scale": scale,
                    "density_kg_m3": 600,
                },
                {
                    "id": "knob",
                    "source": {"type": "catalog_link", "entry_id": state["eid"], "link": ins["root_link"]},
                    "scale": scale * 0.1,
                    "mass_kg": 0.05,
                },
            ],
            "joints": [
                {
                    "name": "mount",
                    "type": "fixed",
                    "parent": "base",
                    "child": "obj",
                    "origin": {"xyz": [0, 0, 0.02 - zmin], "rpy": [0, 0, 0]},
                },
                {
                    "name": "knob_hinge",
                    "type": "revolute",
                    "parent": "obj",
                    "child": "knob",
                    "axis": [0, 0, 1],
                    "origin": {"xyz": [0.0, 0.0, 0.9], "rpy": [0, 0, 0]},
                    "limits": {"lower": 0, "upper": 1.2},
                },
            ],
            "root": "base",
            "support": "free_standing",
            "notes": "offline test assembly",
        }
        state["spec"] = spec
        return ("render_assembly", {"spec": spec})

    def after_render(client):
        res = json.loads(client.last_tool_results[-1]["content"])
        assert "links" in res and "extent_m" in res, res
        print(
            "render result keys",
            list(res.keys()),
            "extent",
            res["extent_m"],
            "lowest z",
            res["lowest_point_z_m"],
            "backend",
            res.get("render_backend"),
        )
        # an invalid spec first (missing density) to exercise validation
        bad = json.loads(json.dumps(state["spec"]))
        del bad["parts"][0]["density_kg_m3"]
        return ("submit_asset", {"spec": bad})

    def after_bad_submit(client):
        res = json.loads(client.last_tool_results[-1]["content"])
        assert "error" in res, res
        print("validation error (expected):", res["error"][:120])
        return ("submit_asset", {"spec": state["spec"]})

    script = [
        ("search_catalog", {"category": case["requested_category"], "query": "test", "k": 4}),
        after_search,
        after_inspect,
        build_spec,
        after_render,
        after_bad_submit,
    ]
    client = FakeClient(script)
    run_root = scratch / "offline-run"
    if run_root.exists():
        shutil.rmtree(run_root)
    d = run_root / "lane-0" / "cases" / case["source_id"]
    runner = CaseRunner(catalog, client, {**case, "lane": 0}, d, "offline-test", tool_choice="required")
    res = runner.run()
    print("result", json.dumps(res, indent=1)[:1500])
    man = read_json(d / "native_manifest.json")
    assert list(man.keys()) == CONTRACT_KEYS, list(man.keys())
    for L in man["links"]:
        assert list(L.keys()) == LINK_KEYS, list(L.keys())
        assert Path(L["visual_obj"]).is_file()
    for J in man["joints"]:
        assert list(J.keys()) == JOINT_KEYS, list(J.keys())
    assert man["asset_emitted"] is True and man["physics_contract"] is True, man
    assert Path(man["urdf"]).is_file() and Path(man["mjcf"]).is_file()
    # URDF parses; MJCF parses
    import xml.etree.ElementTree as ET

    ET.parse(man["urdf"])
    ET.parse(man["mjcf"])
    print("links", len(man["links"]), "joints", len(man["joints"]), "files", sorted(p.name for p in d.iterdir()))
    print("transcript lines", sum(1 for _ in (d / "transcript.jsonl").open()))
    print("OFFLINE TEST OK", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
