"""Per-case agent loop: OpenAI-compatible tool calling against gpt-6-astra with the AffordCraft modelling tools."""

from __future__ import annotations
import json, math, os, shutil, ssl, threading, time, traceback
from pathlib import Path
import requests
from PIL import Image
from common import (
    PROTOCOL,
    MODEL_ID,
    METHOD_ID,
    PREVIEW_MAX_SIDE,
    IMAGE_CONTEXT_KEEP_ROUNDS,
    CACHE_ROOT,
    now_iso,
    sha256_file,
    sha256_bytes,
    image_to_data_uri,
    file_to_data_uri,
    write_json,
    read_json,
    finite,
)
from geometry import Assembly, SpecError, make_primitive_mesh, write_obj, export_native, render_entry

SYSTEM_PROMPT = """You are an autonomous simulation-asset engineer. You receive ONE photograph and the name of the object category to reconstruct. Build an interactive simulation asset of that object as a rigid-body assembly with joints, using ONLY the tools provided, and finish by calling submit_asset.

Requirements of a good asset
1. Correct mechanism: the parts that move in the real object move in the asset with the right joint type (revolute hinge or prismatic slide), a plausible axis, pivot and motion range; parts that do not move are rigidly attached (fixed joints or a single link).
2. Metric size: all coordinates are metres; the assembly must have plausible real-world dimensions for the object in the photograph.
3. Free-standing support: world +Z is up and the ground plane is z = 0; place the asset so that it rests on the ground in a stable pose (lowest point at z = 0). For objects that are normally mounted (windows, doors, wall faucets, ...), model the supporting structure the asset needs (e.g. a frame or wall segment as the root part) so that the whole asset stands on the ground.
4. Physical parameters: every part declares density_kg_m3 or mass_kg; collision geometry is derived from the meshes automatically.

Tools. Each model turn is one round. You have at most 32 rounds in total and a cumulative output budget of 128,000 tokens; the case ends the moment submit_asset accepts a spec. If the budgets run out before that, NO asset is recorded, so submit a reasonable asset early enough and do not spend rounds on unnecessary exploration.
- search_catalog: visual retrieval of catalog assets that resemble the photograph (the same DINOv2 index and ranking as the reference pipeline: up to 12 slots for entries of the requested catalog category, the rest category-blind visual neighbours), returning per entry: id, category, source type, extent, links, joint summary and a preview image. Catalog: 2,335 PartNet-Mobility articulated objects (URDF links + joints, approximately metric) and 9,037 rigid meshes (Objaverse: not metric; GSO and YCB: metric).
- inspect_entry: full link/joint listing of one entry with per-link extents, centres and joint pivots/axes, plus renders at the rest and the opened configuration (links colour-coded).
- make_primitive: creates a box / cylinder / sphere mesh you can use as a part (centred at its own origin).
- render_assembly: renders your current assembly spec from two views on the ground grid (grid step 0.1 m, 0.25 m or 1 m as stated in the result; axis triad at the world origin: x red, y green, z blue) and reports world bounding boxes, pivots and warnings. Use it to check size, placement and mechanism before submitting.
- submit_asset: validates and materialises the spec (URDF + MJCF + meshes). Validation errors are returned to you (this costs a round); only an accepted submission ends the task.

Assembly spec (JSON) used by render_assembly and submit_asset
{
 "parts": [ {"id": "<name>", "source": <source>, "scale": 1.0, "density_kg_m3": 600, "pose": {"xyz": [x, y, z], "rpy": [r, p, y]}}, ... ],
 "joints": [ {"name": "<name>", "type": "revolute" | "prismatic" | "fixed", "parent": "<link>", "child": "<link>", "axis": [x, y, z], "origin": {"xyz": [..], "rpy": [..]}, "limits": {"lower": a, "upper": b}}, ... ],
 "root": "<part id>",
 "support": "free_standing" | "fixed_root",
 "notes": "<free text>"
}
"mass_kg" may be given instead of "density_kg_m3"; "scale", "pose", "support" and "notes" are optional. <source> is one of
 {"type": "catalog_entry", "entry_id": "<id>"}: imports the whole entry with all its links and joints; its root link takes the part id and its other links are named "<part id>.<link name>" (usable as parent/child of your own joints);
 {"type": "catalog_link", "entry_id": "<id>", "link": "<link name>"}: one link mesh of an entry in that link's own frame (origin = the link's joint pivot);
 {"type": "primitive", "part_id": "prim_N"}: a primitive created with make_primitive.
Conventions (URDF semantics): rpy are fixed-axis roll/pitch/yaw in radians; "scale" uniformly scales the part's mesh (and an imported entry's joint offsets and prismatic limits); a joint "origin" is the pose of the child link frame in the parent link frame; a joint "axis" is expressed in the child (joint) frame; revolute limits are radians, prismatic limits metres; the rest configuration is 0 and should lie inside the limits. A part "pose" is the transform of its mesh relative to its link frame; for the root part the link frame is the world, so the root pose places the whole asset. Every non-root part must be the child of exactly one joint and the joint graph must be a tree with exactly one root.

Suggested procedure: identify the object and its mechanism from the photograph; search the catalog with the object's category (try related categories if it is missing); prefer an articulated entry whose mechanism matches and scale it to the real size, adjusting or adding joints/parts where the mechanism differs, or assemble parts and primitives yourself; check with render_assembly; then submit. You cannot ask questions and receive no other input; there is no human in the loop."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_catalog",
            "description": "Visual retrieval over the asset catalog using the input photograph (optionally a crop of it). Returns up to k entries with id, category, source type, extent, link count, joint summary and one preview image each.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": 'Catalog category to prefer (e.g. "StorageFurniture", "Door", "Window", "Chair", "Table", "Bottle", "Laptop"). Unknown names fall back to category-blind visual neighbours and the reply lists similar catalog category names.',
                    },
                    "query": {
                        "type": "string",
                        "description": "Free-text note of what you are looking for. It is recorded only; the ranking is visual.",
                    },
                    "k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Number of entries to return (1-10).",
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 10,
                        "description": "Skip this many top-ranked entries (0-10) to page through the 20-entry retrieval list.",
                    },
                    "box_normalized": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "Optional [x0, y0, x1, y1] in 0-1 image coordinates: crop the photograph to this box before retrieval.",
                    },
                },
                "required": ["category", "query", "k"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_entry",
            "description": "Full description of one catalog entry: links (extent, centre, frame origin), joints (type, parent, child, pivot, axis, limits), size and units note, plus its catalog preview and renders at the rest and the opened configuration.",
            "parameters": {"type": "object", "properties": {"entry_id": {"type": "string"}}, "required": ["entry_id"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_primitive",
            "description": 'Create a primitive mesh part centred at its own origin. box: size_m=[x, y, z]; cylinder: size_m=[radius, height] (axis along z); sphere: size_m=[radius]. Returns the part_id to reference as {"type": "primitive", "part_id": ...}.',
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["box", "cylinder", "sphere"]},
                    "size_m": {"type": "array", "items": {"type": "number"}, "minItems": 1, "maxItems": 3},
                },
                "required": ["kind", "size_m"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "render_assembly",
            "description": "Validate an assembly spec and render it (two views at the rest configuration and one view with every joint at its far limit) on the ground grid; returns the images, the world bounding boxes of all links, joint pivots/axes in world coordinates, the lowest point and warnings.",
            "parameters": {
                "type": "object",
                "properties": {"spec": {"type": "object", "description": "Assembly spec (see the system message)."}},
                "required": ["spec"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_asset",
            "description": "Final answer: validate and materialise the assembly spec as the native asset (URDF + MJCF + meshes). An accepted submission ends the task; a rejected one returns the validation error.",
            "parameters": {
                "type": "object",
                "properties": {"spec": {"type": "object", "description": "Assembly spec (see the system message)."}},
                "required": ["spec"],
            },
        },
    },
]


class ApiError(RuntimeError):
    pass


class ApiClient:
    """Minimal OpenAI-compatible chat client with retries (429/5xx/network) and a 429 storm counter."""

    def __init__(self):
        self.base = os.environ["EVAL_API_BASE"].strip().rstrip("/")
        self._key = os.environ["EVAL_API_KEY"].strip()
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=64)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.lock = threading.Lock()
        self.rate_limit_hits = []
        self.stats = {"calls": 0, "retries": 0, "http_429": 0, "http_5xx": 0, "network_errors": 0}

    def note_429(self):
        with self.lock:
            t = time.time()
            self.rate_limit_hits.append(t)
            self.rate_limit_hits = [x for x in self.rate_limit_hits if t - x < 300]
            return len(self.rate_limit_hits)

    def _stream_once(self, payload, inactivity_timeout, total_timeout):
        """One streamed chat completion. Returns (status, data_or_None, error_text, info)."""
        body = dict(payload, stream=True, stream_options={"include_usage": True})
        t0 = time.time()
        r = self.session.post(
            self.base + "/chat/completions",
            json=body,
            timeout=(30, inactivity_timeout),
            stream=True,
            headers={
                "Authorization": "Bearer " + self._key,
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
        )
        if r.status_code != 200:
            text = r.text[:800]
            r.close()
            return r.status_code, None, text, {"seconds": round(time.time() - t0, 2)}
        message = {"role": "assistant", "content": None}
        tool_calls = {}
        usage = None
        finish = None
        resp_id = None
        model = None
        chunks = 0
        first = None
        reasoning = []
        stream_error = None
        try:
            for line in r.iter_lines():
                if time.time() - t0 > total_timeout:
                    stream_error = f"stream_total_timeout_{total_timeout}s"
                    break
                if not line:
                    continue
                if first is None:
                    first = round(time.time() - t0, 2)
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                try:
                    j = json.loads(data)
                except Exception:
                    continue
                chunks += 1
                if isinstance(j, dict) and j.get("error"):
                    stream_error = "stream_error: " + json.dumps(j["error"])[:400]
                    break
                resp_id = j.get("id") or resp_id
                model = j.get("model") or model
                if j.get("usage"):
                    usage = j["usage"]
                for c in j.get("choices") or []:
                    d = c.get("delta") or {}
                    if d.get("content"):
                        message["content"] = (message["content"] or "") + d["content"]
                    if d.get("reasoning_content"):
                        reasoning.append(d["reasoning_content"])
                    for tc in d.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        slot = tool_calls.setdefault(
                            idx, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}}
                        )
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        f = tc.get("function") or {}
                        if f.get("name"):
                            slot["function"]["name"] += f["name"]
                        if f.get("arguments"):
                            slot["function"]["arguments"] += f["arguments"]
                    if c.get("finish_reason"):
                        finish = c["finish_reason"]
        finally:
            r.close()
        info = {
            "seconds": round(time.time() - t0, 2),
            "first_chunk_s": first,
            "chunks": chunks,
            "reasoning_chars": sum(len(x) for x in reasoning),
        }
        if stream_error:
            return 200, None, stream_error, info
        if finish is None and not tool_calls and message["content"] is None:
            return 200, None, "stream_ended_without_content", info
        if tool_calls:
            message["tool_calls"] = [tool_calls[k] for k in sorted(tool_calls)]
            for tc in message["tool_calls"]:
                if not tc["id"]:
                    tc["id"] = "call_%d_%s" % (int(time.time() * 1000) % 100000000, tc["function"]["name"])
        if usage is None:
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "usage_missing": True}
        data = {
            "id": resp_id,
            "model": model,
            "choices": [
                {"index": 0, "message": message, "finish_reason": finish or ("tool_calls" if tool_calls else "stop")}
            ],
            "usage": usage,
        }
        if reasoning:
            data["reasoning_content"] = "".join(reasoning)[:20000]
        return 200, data, None, info

    def chat(
        self, payload, timeout=PROTOCOL["per_call_timeout_seconds"], max_retries=PROTOCOL["max_retries"], log=None
    ):
        """Streamed request with retries. `timeout` is the inactivity timeout (no bytes for that long); the whole
        stream is capped at PROTOCOL['per_call_total_timeout_seconds']."""
        attempts = []
        delay = 5.0
        total_timeout = PROTOCOL.get("per_call_total_timeout_seconds", 1800)
        for attempt in range(max_retries + 1):
            status = None
            data = None
            err = None
            info = {}
            try:
                status, data, err, info = self._stream_once(payload, timeout, total_timeout)
            except requests.RequestException as exc:
                err = type(exc).__name__ + ": " + str(exc)[:200]
            rec = {"attempt": attempt, "status": status, "error": err, **info}
            with self.lock:
                self.stats["calls"] += 1
            attempts.append(rec)
            if status == 200 and data is not None:
                return data, attempts
            if status == 429:
                rec["rate_limit_hits_5min"] = self.note_429()
                with self.lock:
                    self.stats["http_429"] += 1
            elif status is not None and status >= 500:
                with self.lock:
                    self.stats["http_5xx"] += 1
            elif status is not None and status != 200:
                raise ApiError("http_%s: " % status + (err or "")[:600])
            else:
                with self.lock:
                    self.stats["network_errors"] += 1
            if log:
                log({"event": "api_retry", "attempt": rec})
            if attempt < max_retries:
                with self.lock:
                    self.stats["retries"] += 1
                time.sleep(delay + (0.5 * delay * (hash(threading.get_ident()) % 100) / 100.0))
                delay = min(delay * 2, 160.0)
        raise ApiError("retries_exhausted: " + json.dumps(attempts[-1])[:600])


def _compact(value, limit=20000):
    text = json.dumps(value, ensure_ascii=False)
    if len(text) > limit:
        text = text[:limit] + f"... [truncated {len(text) - limit} chars]"
    return text


class CaseRunner:
    def __init__(self, catalog, client, case, case_dir, run_id, tool_choice="required", event_log=None):
        self.catalog = catalog
        self.client = client
        self.case = case
        self.dir = Path(case_dir)
        self.run_id = run_id
        self.tool_choice = tool_choice
        self.event_log = event_log or (lambda e: None)
        self.sid = case["source_id"]
        self.dir.mkdir(parents=True, exist_ok=False)
        (self.dir / "parts").mkdir()
        (self.dir / "renders").mkdir()
        self.transcript = (self.dir / "transcript.jsonl").open("a", encoding="utf-8")
        self.tlock = threading.Lock()
        self.messages = []
        self.image_round = {}  # index of a user image message -> round it was produced in
        self.prim_counter = 0
        self.round = 0
        self.completion_total = 0
        self.prompt_total = 0
        self.reasoning_total = 0
        self.calls = []
        self.started = time.time()
        self.submitted = None
        self.validation_failures = 0
        self.tool_counts = {}
        self.keep_rounds = IMAGE_CONTEXT_KEEP_ROUNDS
        self.context_prunes = 0

    # ---- logging -----------------------------------------------------------------------------------------------------
    def log(self, event):
        event = {"t": now_iso(), "elapsed_s": round(time.time() - self.started, 2), **event}
        with self.tlock:
            self.transcript.write(json.dumps(event, ensure_ascii=False) + "\n")
            self.transcript.flush()

    @staticmethod
    def strip_images(message):
        m = dict(message)
        c = m.get("content")
        if isinstance(c, list):
            out = []
            for part in c:
                if part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    out.append(
                        {
                            "type": "image_url",
                            "image_sha256": sha256_bytes(url.encode()),
                            "data_uri_chars": len(url),
                            "label": part.get("_label"),
                        }
                    )
                else:
                    out.append(part)
            m["content"] = out
        return m

    # ---- messages ----------------------------------------------------------------------------------------------------
    def initial_messages(self):
        uri, nbytes, sha = file_to_data_uri(self.case["image_path"])
        instruction = f"Construct an interactive simulation asset for the {self.case['requested_category']} visible in this image. Preserve its mechanism and natural support conditions."
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": uri, "detail": "high"}, "_label": "input_photograph"},
                ],
            },
        ]
        self.log(
            {
                "event": "case_start",
                "source_id": self.sid,
                "requested_category": self.case["requested_category"],
                "image_path": self.case["image_path"],
                "image_sha256": sha,
                "image_bytes": nbytes,
                "instruction": instruction,
                "system_prompt_sha256": sha256_bytes(SYSTEM_PROMPT.encode()),
                "model": MODEL_ID,
                "reasoning_effort": PROTOCOL["reasoning_effort"],
                "tool_choice": self.tool_choice,
                "lane": self.case.get("lane"),
            }
        )

    def payload_messages(self):
        """Copy of the conversation for the API: image parts of user messages older than IMAGE_CONTEXT_KEEP_ROUNDS rounds
        (except the input photograph) are replaced by text stubs; private keys are dropped."""
        out = []
        for i, m in enumerate(self.messages):
            c = m.get("content")
            if isinstance(c, list):
                parts = []
                stale = i in self.image_round and (self.round - self.image_round[i]) > self.keep_rounds
                for p in c:
                    if p.get("type") == "image_url":
                        if stale:
                            parts.append(
                                {
                                    "type": "text",
                                    "text": f"[image '{p.get('_label')}' no longer in context; call the tool again to see it]",
                                }
                            )
                        else:
                            parts.append({"type": "image_url", "image_url": p["image_url"]})
                    else:
                        parts.append({"type": p.get("type", "text"), "text": p.get("text", "")})
                mm = {"role": m["role"], "content": parts}
            else:
                mm = {k: v for k, v in m.items() if not k.startswith("_")}
            out.append(mm)
        return out

    # ---- budgets -------------------------------------------------------------------------------------------------------
    def budget_state(self):
        if self.round >= PROTOCOL["max_rounds"]:
            return "budget_exhausted:rounds"
        if self.completion_total >= PROTOCOL["max_completion_tokens_total"]:
            return "budget_exhausted:completion_tokens"
        if time.time() - self.started >= PROTOCOL["wall_clock_seconds"]:
            return "budget_exhausted:wall_clock"
        return None

    # ---- main loop -----------------------------------------------------------------------------------------------------
    def run(self):
        failure = None
        try:
            self.initial_messages()
            while True:
                failure = self.budget_state()
                if failure or self.submitted:
                    break
                remaining = PROTOCOL["max_completion_tokens_total"] - self.completion_total
                payload = {
                    "model": MODEL_ID,
                    "messages": self.payload_messages(),
                    "tools": TOOLS,
                    "tool_choice": self.tool_choice,
                    "reasoning_effort": PROTOCOL["reasoning_effort"],
                    "max_completion_tokens": int(min(PROTOCOL["per_call_max_tokens"], max(remaining, 1))),
                }
                self.log(
                    {
                        "event": "request",
                        "round": self.round + 1,
                        "messages_total": len(self.messages),
                        "new_messages": [self.strip_images(m) for m in self.messages[self._last_logged :]],
                        "max_completion_tokens": payload["max_completion_tokens"],
                        "tool_choice": self.tool_choice,
                    }
                )
                self._last_logged = len(self.messages)
                t0 = time.time()
                try:
                    data, attempts = self.client.chat(payload, log=self.log)
                except ApiError as exc:
                    text = str(exc)
                    if self.tool_choice == "required" and "tool_choice" in text and self.round == 0:
                        self.tool_choice = "auto"
                        self.log({"event": "tool_choice_fallback", "error": text[:300]})
                        continue
                    low = text.lower()
                    if (
                        text.startswith("http_4")
                        and any(k in low for k in ("context", "too long", "maximum", "token", "too large", "exceed"))
                        and self.context_prunes < 3
                    ):
                        # context overflow: keep fewer rounds of tool images (1, then 0, then none at all) and retry without consuming a round
                        self.context_prunes += 1
                        self.keep_rounds = {1: 1, 2: 0, 3: -1}[self.context_prunes]
                        self.log({"event": "context_prune", "keep_rounds": self.keep_rounds, "error": text[:300]})
                        continue
                    failure = "api_error:" + text[:200]
                    self.log({"event": "api_error", "error": text[:1200]})
                    break
                dt = time.time() - t0
                self.round += 1
                usage = data.get("usage") or {}
                ct = int(usage.get("completion_tokens") or 0)
                pt = int(usage.get("prompt_tokens") or 0)
                rt = int(((usage.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0)
                self.completion_total += ct
                self.prompt_total += pt
                self.reasoning_total += rt
                choice = data["choices"][0]
                msg = choice["message"]
                self.calls.append(
                    {
                        "round": self.round,
                        "seconds": round(dt, 2),
                        "attempts": len(attempts),
                        "prompt_tokens": pt,
                        "completion_tokens": ct,
                        "reasoning_tokens": rt,
                        "cumulative_completion_tokens": self.completion_total,
                        "cumulative_prompt_tokens": self.prompt_total,
                        "finish_reason": choice.get("finish_reason"),
                        "tool_calls": [tc.get("function", {}).get("name") for tc in (msg.get("tool_calls") or [])],
                        "response_id": data.get("id"),
                        "model": data.get("model"),
                    }
                )
                self.log(
                    {
                        "event": "response",
                        "round": self.round,
                        "seconds": round(dt, 2),
                        "attempts": attempts,
                        "usage": usage,
                        "finish_reason": choice.get("finish_reason"),
                        "message": msg,
                        "response_id": data.get("id"),
                        "model": data.get("model"),
                        "cumulative": {"completion": self.completion_total, "prompt": self.prompt_total},
                        "reasoning_content": data.get("reasoning_content"),
                    }
                )
                assistant = {"role": "assistant"}
                if msg.get("content"):
                    assistant["content"] = msg["content"]
                tool_calls = msg.get("tool_calls") or []
                if tool_calls:
                    assistant["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]},
                        }
                        for tc in tool_calls
                    ]
                if "content" not in assistant and not tool_calls:
                    assistant["content"] = ""
                self.messages.append(assistant)
                if self.completion_total > PROTOCOL["max_completion_tokens_total"]:
                    failure = "budget_exhausted:completion_tokens"
                    self.log(
                        {
                            "event": "budget_exhausted",
                            "which": "completion_tokens",
                            "cumulative": self.completion_total,
                            "note": "the last response is not acted upon",
                        }
                    )
                    break
                if not tool_calls:
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "No tool was called. Continue by calling a tool; only submit_asset ends the task.",
                        }
                    )
                    continue
                images = []
                for tc in tool_calls:
                    name = tc["function"]["name"]
                    args_text = tc["function"].get("arguments") or "{}"
                    if self.submitted:
                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": json.dumps(
                                    {"error": "the asset was already submitted; this call is ignored"}
                                ),
                            }
                        )
                        continue
                    t1 = time.time()
                    result, imgs = self.execute_tool(name, args_text)
                    self.tool_counts[name] = self.tool_counts.get(name, 0) + 1
                    self.log(
                        {
                            "event": "tool",
                            "round": self.round,
                            "tool_call_id": tc["id"],
                            "name": name,
                            "arguments": args_text[:20000],
                            "seconds": round(time.time() - t1, 2),
                            "result": result,
                            "images": [{"label": lab, "sha256": sha, "bytes": nb} for lab, _, nb, sha in imgs],
                        }
                    )
                    self.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": _compact(result)})
                    for lab, uri, nb, sha in imgs:
                        images.append((tc["id"], name, lab, uri))
                if images and not self.submitted:
                    parts = [
                        {
                            "type": "text",
                            "text": "Images returned by the tool call(s) of the previous turn, in order: "
                            + ", ".join(f"{lab} (from {name}, call {cid})" for cid, name, lab, _ in images),
                        }
                    ]
                    for cid, name, lab, uri in images:
                        parts.append({"type": "text", "text": f"[{lab}]"})
                        parts.append({"type": "image_url", "image_url": {"url": uri}, "_label": lab})
                    self.messages.append({"role": "user", "content": parts})
                    self.image_round[len(self.messages) - 1] = self.round
        except Exception as exc:
            failure = "harness_error:" + repr(exc)[:200]
            self.log({"event": "harness_error", "traceback": traceback.format_exc()[-4000:]})
        return self.finish(failure)

    _last_logged = 0

    # ---- tools ---------------------------------------------------------------------------------------------------------
    def execute_tool(self, name, args_text):
        try:
            args = json.loads(args_text)
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except Exception as exc:
            return {"error": f"could not parse tool arguments as JSON: {exc}"}, []
        try:
            if name == "search_catalog":
                return self.tool_search(args)
            if name == "inspect_entry":
                return self.tool_inspect(args)
            if name == "make_primitive":
                return self.tool_primitive(args)
            if name == "render_assembly":
                return self.tool_render(args)
            if name == "submit_asset":
                return self.tool_submit(args)
            return {
                "error": f"unknown tool {name!r}; available: search_catalog, inspect_entry, make_primitive, render_assembly, submit_asset"
            }, []
        except SpecError as exc:
            return {"error": str(exc)[:4000]}, []
        except Exception as exc:
            self.log({"event": "tool_exception", "name": name, "traceback": traceback.format_exc()[-4000:]})
            return {"error": "tool failed: " + repr(exc)[:600]}, []

    def _image(self, path, label):
        uri, nb, sha = image_to_data_uri(Image.open(path), max_side=PREVIEW_MAX_SIDE)
        return (label, uri, nb, sha)

    def tool_search(self, args):
        category = str(args.get("category", "")).strip()
        query = str(args.get("query", ""))[:500]
        k = args.get("k", 10)
        if not isinstance(k, int) or isinstance(k, bool) or not (1 <= k <= 10):
            return {"error": "k must be an integer in 1..10"}, []
        offset = args.get("offset", 0)
        if not isinstance(offset, int) or isinstance(offset, bool) or not (0 <= offset <= 10):
            return {"error": "offset must be an integer in 0..10"}, []
        im = Image.open(self.case["image_path"]).convert("RGB")
        crop_note = None
        box = args.get("box_normalized")
        if box is not None:
            if not (isinstance(box, list) and len(box) == 4 and all(finite(v) for v in box)):
                return {"error": "box_normalized must be [x0, y0, x1, y1] with numbers in 0..1"}, []
            x0, y0, x1, y1 = [min(max(float(v), 0.0), 1.0) for v in box]
            w, h = im.size
            px = (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
            if px[2] - px[0] < 8 or px[3] - px[1] < 8:
                return {"error": "box_normalized is empty or too small (needs x1 > x0 and y1 > y0)"}, []
            im = im.crop(px)
            crop_note = {"box_normalized": [x0, y0, x1, y1], "crop_pixels": list(px)}
        resolved = self.catalog.resolve_category(category)
        ranked = self.catalog.retrieve(im, resolved or category, limit=20)
        chosen = ranked[offset : offset + k]
        entries = []
        images = []
        for rank, e in enumerate(chosen, start=offset + 1):
            eid = str(e["candidate_id"])
            try:
                s = self.catalog.summary(eid)
            except Exception as exc:
                s = {"entry_id": eid, "category": e.get("category"), "summary_error": repr(exc)[:200]}
            s = dict(s)
            s["rank"] = rank
            s["retrieval_score"] = round(float(e.get("retrieval_score", 0.0)), 4)
            s["preview_image"] = f"entry {eid} preview"
            entries.append(s)
            try:
                images.append(
                    self._image(self.catalog.summary(eid, detailed=True)["preview_path"], f"entry {eid} preview")
                )
            except Exception as exc:
                s["preview_image"] = "unavailable: " + repr(exc)[:100]
        result = {
            "category_requested": category,
            "category_matched": resolved,
            "query_recorded": query,
            "crop": crop_note,
            "offset": offset,
            "returned": len(entries),
            "note": "ranking = frozen visual retrieval (12 category-preferred slots, then category-blind); extents are in source units (see units_note); previews follow as images",
            "entries": entries,
        }
        if resolved is None:
            result["similar_catalog_categories"] = self.catalog.category_suggestions(category, query)
            result["note"] += "; the category is not a catalog category, so all 20 slots are category-blind"
        return result, images

    def tool_inspect(self, args):
        eid = str(args.get("entry_id", "")).strip()
        if eid not in self.catalog.entries:
            return {"error": f"unknown entry_id {eid!r}"}, []
        s = self.catalog.summary(eid, detailed=True)
        out_dir = CACHE_ROOT / "entry_render"
        rest = out_dir / f"{eid}_rest_view0.png"
        opened = out_dir / f"{eid}_open_view0.png"
        with self.catalog._entry_lock(eid + ":render"):
            if not rest.exists():
                render_entry(self.catalog, eid, out_dir)
        legend = None
        g = self.catalog.load_geometry(eid)
        names = [n for n in g["links"] if g["links"][n]["mesh"] is not None]
        from geometry import PALETTE_NAMES

        legend = {n: PALETTE_NAMES[i % len(PALETTE_NAMES)] for i, n in enumerate(names)}
        images = [
            self._image(s["preview_path"], f"entry {eid} catalog preview"),
            self._image(rest, f"entry {eid} render at rest configuration (3/4 view, +Z up)"),
        ]
        if opened.exists():
            images.append(self._image(opened, f"entry {eid} render with every joint at its far limit"))
        result = {k: v for k, v in s.items() if k != "preview_path"}
        result["render_link_colours"] = legend
        if s.get("whole_entry_importable", True):
            result["import_hint"] = {
                "whole_entry": {"type": "catalog_entry", "entry_id": eid},
                "single_link": {
                    "type": "catalog_link",
                    "entry_id": eid,
                    "link": (s["root_link"] if s["root_link"] in names else names[0]),
                },
                "link_names_after_whole_entry_import": [
                    "<part id>" + ("" if n == s["root_link"] else "." + n) for n in names
                ],
            }
        else:
            result["import_hint"] = {
                "whole_entry": None,
                "reason": s.get("import_note"),
                "single_link": {"type": "catalog_link", "entry_id": eid, "link": names[0]},
                "links_with_geometry": names,
            }
        return result, images

    def tool_primitive(self, args):
        mesh, meta = make_primitive_mesh(args.get("kind"), args.get("size_m"))
        pid = f"prim_{self.prim_counter}"
        self.prim_counter += 1
        write_obj(self.dir / "parts" / f"{pid}.obj", mesh)
        write_json(self.dir / "parts" / f"{pid}.json", meta)
        lo, hi = mesh.bounds
        return {
            "part_id": pid,
            **meta,
            "bounds_min": [round(float(x), 4) for x in lo],
            "bounds_max": [round(float(x), 4) for x in hi],
            "usage": {"type": "primitive", "part_id": pid},
        }, []

    def tool_render(self, args):
        asm = Assembly(self.catalog, args.get("spec"), self.dir / "parts")
        self.render_counter = getattr(self, "render_counter", 0) + 1
        tag = f"round{self.round:02d}_r{self.render_counter}"
        outs, backend = asm.render(self.dir / "renders", tag + "_rest")
        images = [
            self._image(outs[0], f"assembly rest configuration, view A (azimuth 40 deg)"),
            self._image(outs[1], f"assembly rest configuration, view B (azimuth 130 deg)"),
        ]
        q = asm.open_configuration()
        if q:
            o2, _ = asm.render(self.dir / "renders", tag + "_open", views=[{"azim": 40, "elev": 25}], q=q)
            images.append(self._image(o2[0], "assembly with every joint at its far limit, view A"))
        d = asm.describe()
        ext = max(d["extent_m"])
        d["grid_step_m"] = 0.1 if ext < 1.2 else (0.25 if ext < 3 else 1.0)
        d["link_colours"] = asm.colour_legend()
        d["render_backend"] = backend
        if d["lowest_point_z_m"] > 0.01 or d["lowest_point_z_m"] < -0.01:
            d["warnings"].append(
                f"lowest point is at z = {d['lowest_point_z_m']} m; a free-standing asset should rest on z = 0"
            )
        return d, images

    def tool_submit(self, args):
        spec = args.get("spec")
        asm = Assembly(self.catalog, spec, self.dir / "parts")
        native = self.dir / "native"
        if native.exists():
            n = 1
            while (self.dir / f"native_failed_{n}").exists():
                n += 1
            shutil.move(str(native), str(self.dir / f"native_failed_{n}"))
        try:
            export = export_native(asm, native)
        except SpecError:
            raise
        except Exception as exc:
            self.validation_failures += 1
            raise SpecError("materialisation failed: " + repr(exc)[:400])
        write_json(self.dir / "spec.json", spec, exclusive=True)
        write_json(
            native / "assembly.json",
            {
                "description": asm.describe(),
                "export": {k: v for k, v in export.items() if k not in ("links", "joints")},
                "link_records": export["links"],
                "joint_records": export["joints"],
            },
        )
        self.submitted = {"spec": spec, "export": export, "description": asm.describe(), "round": self.round}
        return {
            "accepted": True,
            "links": len(export["links"]),
            "joints": len(export["joints"]),
            "urdf": export["urdf"],
            "mjcf": export["mjcf"],
            "message": "asset materialised; the task is complete",
        }, []

    # ---- finish ------------------------------------------------------------------------------------------------------
    def finish(self, failure):
        runtime = time.time() - self.started
        emitted = self.submitted is not None and failure is None
        export = self.submitted["export"] if emitted else None
        timing = {
            "source_id": self.sid,
            "wall_seconds": round(runtime, 2),
            "rounds": self.round,
            "calls": self.calls,
            "cumulative": {
                "completion_tokens": self.completion_total,
                "prompt_tokens": self.prompt_total,
                "reasoning_tokens": self.reasoning_total,
            },
            "tool_counts": self.tool_counts,
            "started": now_iso(),
        }
        write_json(self.dir / "timing.json", timing)
        result = {
            "source_id": self.sid,
            "method_id": METHOD_ID,
            "asset_emitted": emitted,
            "part_count": (len(export["links"]) if emitted else 0),
            "joint_count": (len(export["joints"]) if emitted else 0),
            "rounds": self.round,
            "completion_tokens": self.completion_total,
            "prompt_tokens": self.prompt_total,
            "reasoning_tokens": self.reasoning_total,
            "failure_reason": failure,
            "runtime_seconds": round(runtime, 2),
            "requested_category": self.case["requested_category"],
            "lane": self.case.get("lane"),
            "submitted_round": (self.submitted["round"] if self.submitted else None),
            "physics_contract": (export["physics_contract"] if emitted else False),
            "support": (export["support"] if emitted else None),
            "world_bbox_min_m": (export["world_bbox_min_m"] if emitted else None),
            "world_bbox_max_m": (export["world_bbox_max_m"] if emitted else None),
            "validation_failures": self.validation_failures,
            "context_prunes": self.context_prunes,
            "tool_counts": self.tool_counts,
            "model": MODEL_ID,
            "reasoning_effort": PROTOCOL["reasoning_effort"],
            "tool_choice": self.tool_choice,
            "finished": now_iso(),
            "host": os.uname().nodename,
        }
        write_json(self.dir / "result.json", result)
        notes = (
            "GPT-6 Astra tool-calling agent (single photograph, no box/mask). The agent declared parts (catalog links/entries or primitives), poses, uniform scales, "
            "densities/masses and joints; the harness materialised them: every link frame has the world orientation at the rest configuration with its origin at the "
            "joint pivot (root: the declared root pose), so OBJ files are baked (scale, pose and joint rotations applied) and joint axes are given in the parent frame; "
            "collision geometry = the visual meshes (URDF collision elements, MJCF mesh geoms); URDF inertials from the convex hull; "
            f"support declared by the agent: {export['support'] if emitted else None}"
            + (" (MJCF root has a freejoint)" if emitted and export["support"] == "free_standing" else "")
            + "."
        )
        manifest = {
            "schema": "affordcraft.external_native_manifest.v1",
            "source_id": self.sid,
            "method_id": METHOD_ID,
            "run_id": self.run_id,
            "asset_emitted": emitted,
            "stage_reached": "export" if emitted else "agent_loop",
            "failure_reason": failure,
            "units": "m",
            "metric_size_source": "method",
            "native_formats": ["MJCF", "URDF"] if emitted else [],
            "mjcf": (export["mjcf"] if emitted else None),
            "urdf": (export["urdf"] if emitted else None),
            "root_link": (export["root_link"] if emitted else None),
            "links": (export["links"] if emitted else []),
            "joints": (export["joints"] if emitted else []),
            "physics_contract": (bool(export["physics_contract"]) if emitted else False),
            "notes": notes,
        }
        write_json(self.dir / "native_manifest.json", manifest, exclusive=True)
        self.log(
            {
                "event": "case_end",
                "asset_emitted": emitted,
                "failure_reason": failure,
                "rounds": self.round,
                "completion_tokens": self.completion_total,
                "prompt_tokens": self.prompt_total,
                "runtime_seconds": round(runtime, 2),
            }
        )
        self.transcript.close()
        return result
