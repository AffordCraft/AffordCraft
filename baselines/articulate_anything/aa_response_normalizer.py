#!/usr/bin/env python
"""aa_response_normalizer.py -- format-tolerant normalisation of VLM replies for the official Articulate-Anything parsers.

The official agents parse JSON replies with `response.text.strip().strip('```json').strip()` + `json.loads(..., strict=False)`
and read the object-selector index with `int(x['selected_image'].split()[-1])`. gemini-2.5-flash (API endpoint) does not always
follow the requested format: in the aborted first attempt 8/102 object-selector replies were prose ending in
"The final answer is $\\boxed{N}$" (no JSON at all), 3 had a trailing comma before `}`, 1 returned "selected_image": "3".
This module rewrites such replies into the canonical form the official parsers expect and leaves everything else
untouched (code replies of LinkPlacementActor/JointPredictionActor are never modified). No pipeline logic changes;
prompts are unchanged. When a reply is modified, <out_dir>/response_normalized.json records raw and normalised text.
"""
import json
import os
import re

JSON_AGENTS = {
    "ObjectSelector",
    "ObjectDetector",
    "LinkCritic",
    "JointCritic",
    "JointCriticMultiModalExamples",
    "TargettedAffordanceExtractor",
    "TextTaskSpecifier",
    "TextLayoutPlanner",
}
BOXED_RE = re.compile(r"\\boxed\{\s*(?:\\text\{)?\s*(?:Image\s*)?(\d+)\s*\}?\s*\}", re.I)
FENCE_RE = re.compile(r"```(?:json|JSON)?\s*\n?(\{[\s\S]*?\})\s*```", re.S)


def _first_balanced_object(text):
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def _loads_tolerant(s):
    try:
        return json.loads(s, strict=False), "as_is"
    except Exception:
        pass
    fixed = re.sub(r",(\s*[}\]])", r"\1", s)  # trailing commas
    try:
        return json.loads(fixed, strict=False), "trailing_comma_removed"
    except Exception:
        pass
    fixed2 = re.sub(r"(?<!\\)\n", "\\\\n", fixed)  # raw newlines inside strings
    try:
        return json.loads(fixed2, strict=False), "trailing_comma_and_newlines"
    except Exception:
        return None, None


def _normalize_selected_image(obj, raw_text):
    v = obj.get("selected_image")
    idx = None
    if isinstance(v, bool):
        v = None
    if isinstance(v, (int, float)):
        idx = int(v)
    elif isinstance(v, str):
        m = re.search(r"(\d+)", v)
        if m:
            idx = int(m.group(1))
    if idx is None:
        m = BOXED_RE.search(raw_text)
        if m:
            idx = int(m.group(1))
    if idx is not None:
        obj["selected_image"] = "Image %d" % idx
    return obj


def normalize(text, agent_name, out_dir=None):
    """Return (possibly rewritten) text. Only JSON-reply agents are touched."""
    if agent_name not in JSON_AGENTS or not isinstance(text, str):
        return text
    raw = text
    stripped = text.strip().strip("```json").strip()
    obj, how = _loads_tolerant(stripped)
    kind = None
    if obj is not None and how == "as_is" and isinstance(obj, dict):
        # canonical already; only the selected_image field may need normalising
        if agent_name == "ObjectSelector":
            before = obj.get("selected_image")
            obj = _normalize_selected_image(obj, raw)
            if obj.get("selected_image") == before:
                return text
            kind = "selected_image_normalized"
    else:
        candidate = None
        m = FENCE_RE.search(text)
        if m:
            candidate = m.group(1)
        else:
            candidate = _first_balanced_object(text)
        if candidate is not None:
            obj, how = _loads_tolerant(candidate)
            kind = ("json_extracted:" + how) if obj is not None else None
        if obj is None and agent_name == "ObjectSelector" and BOXED_RE.search(text):
            obj = {"image_description": "", "selected_image": None, "reasoning": text.strip()}
            kind = "boxed_answer_wrapped"
        if obj is None or not isinstance(obj, dict):
            return text  # leave it to the official parser (which will fail -> official failure path)
        if agent_name == "ObjectSelector":
            obj = _normalize_selected_image(obj, raw)
            if obj.get("selected_image") is None:
                return text
    new_text = "```json\n" + json.dumps(obj, indent=4, ensure_ascii=False) + "\n```"
    if out_dir:
        try:
            os.makedirs(out_dir, exist_ok=True)
            k = 0
            while os.path.exists(os.path.join(out_dir, "response_normalized_%d.json" % k)):
                k += 1
            with open(os.path.join(out_dir, "response_normalized_%d.json" % k), "w", encoding="utf-8") as f:
                json.dump(
                    {"agent": agent_name, "kind": kind, "raw_text": raw, "normalized_text": new_text},
                    f,
                    indent=1,
                    ensure_ascii=False,
                )
        except OSError:
            pass
    return new_text
