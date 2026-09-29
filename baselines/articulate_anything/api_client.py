#!/usr/bin/env python
"""api_client.py -- OpenAI-compatible chat-completions VLM client for the Articulate-Anything baseline run.

Replaces google.generativeai (the official default, Gemini 1.5 Flash) with an OpenAI-compatible endpoint (EVAL_API_BASE,
OpenAI-compatible /chat/completions with vision via image_url data URIs). The message layout follows the official
GPTWrapper in articulate_anything/utils/prompt_utils.py: one system message (the agent's system instruction) and one
user message whose content interleaves the text parts and the JPEG-encoded PIL images (detail "low") in the order of
`prompt_parts`. Credentials are read from the environment only (EVAL_API_KEY); they are never logged or written.

Per call it appends one JSON line to $AA_VLM_TRANSCRIPT (agent, model, temperature, text parts verbatim, image
sha256/size, response text, usage incl. reasoning tokens, latency, attempts) and writes vlm_response_<n>.json into the
agent's out_dir. Transient failures (HTTP 408/409/425/429/5xx, timeouts, connection resets, undecodable bodies) are
retried with exponential backoff (max AA_VLM_MAX_ATTEMPTS = 5 attempts). An empty reply (thinking budget exhausted at
the API endpoint) is retried with reasoning_effort="low" and max_tokens=32768. Non-transient HTTP errors (400/401/403/404/413/
422) fail immediately.
"""
import base64
import hashlib
import io
import json
import os
import random
import time

import requests
from PIL import Image

TRANSIENT_STATUS = {408, 409, 425, 429}
_CALL_COUNTER = {"n": 0}


class ApiResponse:
    """Duck-types the google.generativeai response used by Agent.parse_response (only `.text`)."""

    def __init__(self, text, raw, usage):
        self.text = text
        self.raw = raw
        self.usage = usage
        self.usage_metadata = usage


class ApiWrapper:
    def __init__(self, model_name, system_instruction=None, agent_name=None, out_dir=None, **_ignored):
        self.model_name = model_name
        self.system_instruction = system_instruction if system_instruction is not None else ""
        self.agent_name = agent_name
        self.out_dir = out_dir
        self.base_url = os.environ["EVAL_API_BASE"].rstrip("/")
        if "EVAL_API_KEY" not in os.environ:
            raise RuntimeError("EVAL_API_KEY is not set in the environment")
        self.transcript = os.environ.get("AA_VLM_TRANSCRIPT")
        self.timeout = float(os.environ.get("AA_VLM_TIMEOUT", "300"))
        self.max_attempts = int(os.environ.get("AA_VLM_MAX_ATTEMPTS", "5"))
        self.image_format = os.environ.get(
            "AA_VLM_IMAGE_FORMAT", "JPEG"
        )  # official GPTWrapper and Anthropic wrapper: JPEG
        self.image_detail = os.environ.get("AA_VLM_IMAGE_DETAIL", "low")  # official GPTWrapper: "low"

    # ------------------------------------------------------------------ formatting
    def _encode_image(self, img):
        buf = io.BytesIO()
        if self.image_format == "JPEG" and img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(buf, format=self.image_format)
        data = buf.getvalue()
        return base64.b64encode(data).decode("utf-8"), hashlib.sha256(data).hexdigest(), len(data), list(img.size)

    def _format_content(self, prompt_parts):
        if isinstance(prompt_parts, (str, Image.Image, dict)):
            prompt_parts = [prompt_parts]
        content, logged = [], []
        for part in prompt_parts:
            if isinstance(part, str):
                content.append({"type": "text", "text": part})
                logged.append({"type": "text", "text": part})
            elif isinstance(part, Image.Image):
                b64, sha, nbytes, size = self._encode_image(part)
                mime = "image/jpeg" if self.image_format == "JPEG" else "image/png"
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:%s;base64,%s" % (mime, b64), "detail": self.image_detail},
                    }
                )
                logged.append(
                    {
                        "type": "image",
                        "sha256": sha,
                        "bytes": nbytes,
                        "size": size,
                        "format": self.image_format,
                        "detail": self.image_detail,
                    }
                )
            elif isinstance(part, dict):
                content.append(part)
                logged.append({"type": "dict", "keys": sorted(part.keys())})
            else:
                raise TypeError("unsupported prompt part type: %r" % type(part))
        return content, logged

    # ------------------------------------------------------------------ logging
    def _log(self, record):
        _CALL_COUNTER["n"] += 1
        record["call_index_in_process"] = _CALL_COUNTER["n"]
        record["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if self.transcript:
            os.makedirs(os.path.dirname(self.transcript) or ".", exist_ok=True)
            with open(self.transcript, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.out_dir:
            try:
                os.makedirs(self.out_dir, exist_ok=True)
                k = 0
                while True:
                    p = os.path.join(self.out_dir, "vlm_response_%d.json" % k)
                    if not os.path.exists(p):
                        break
                    k += 1
                slim = {kk: vv for kk, vv in record.items() if kk != "prompt_parts"}
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(slim, f, indent=1, ensure_ascii=False)
            except OSError:
                pass

    # ------------------------------------------------------------------ call
    def generate_content(self, prompt_parts, generation_config=None):
        gen = dict(generation_config or {})
        temperature = gen.get("temperature", 0.5)
        content, logged_parts = self._format_content(prompt_parts)
        body = {
            "model": self.model_name,
            "messages": [{"role": "system", "content": self.system_instruction}, {"role": "user", "content": content}],
            "temperature": temperature,
        }
        if "max_tokens" in gen:
            body["max_tokens"] = int(gen["max_tokens"])
        headers = {"Authorization": "Bearer " + os.environ["EVAL_API_KEY"], "Content-Type": "application/json"}
        attempts, text, usage, data, finish = [], None, None, None, None
        t_start = time.time()
        for k in range(self.max_attempts):
            t0 = time.time()
            fallback = "reasoning_effort" in body
            try:
                r = requests.post(self.base_url + "/chat/completions", headers=headers, json=body, timeout=self.timeout)
                status = r.status_code
                if status in TRANSIENT_STATUS or status >= 500:
                    attempts.append(
                        {
                            "attempt": k + 1,
                            "status": status,
                            "seconds": round(time.time() - t0, 2),
                            "error": r.text[:300],
                            "fallback": fallback,
                        }
                    )
                elif status != 200:
                    attempts.append(
                        {
                            "attempt": k + 1,
                            "status": status,
                            "seconds": round(time.time() - t0, 2),
                            "error": r.text[:300],
                            "fallback": fallback,
                        }
                    )
                    self._log(
                        self._record(logged_parts, temperature, body, None, None, None, attempts, t_start, "http_error")
                    )
                    raise RuntimeError("API VLM non-transient HTTP %d: %s" % (status, r.text[:500]))
                else:
                    data = r.json()
                    choice = data["choices"][0]
                    msg = choice.get("message") or {}
                    text = msg.get("content")
                    if isinstance(text, list):  # some endpoints return content parts
                        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
                    finish = choice.get("finish_reason")
                    usage = data.get("usage")
                    attempts.append(
                        {
                            "attempt": k + 1,
                            "status": 200,
                            "seconds": round(time.time() - t0, 2),
                            "finish_reason": finish,
                            "content_chars": len(text or ""),
                            "empty": not (text and text.strip()),
                            "fallback": fallback,
                        }
                    )
                    if text and text.strip():
                        break
                    # empty content: the thinking budget consumed the reply -> low reasoning effort + explicit large budget
                    body["reasoning_effort"] = "low"
                    body["max_tokens"] = max(int(body.get("max_tokens", 0) or 0), 32768)
            except (requests.ConnectionError, requests.Timeout, ValueError, KeyError, IndexError, TypeError) as e:
                attempts.append(
                    {
                        "attempt": k + 1,
                        "status": None,
                        "seconds": round(time.time() - t0, 2),
                        "error": repr(e)[:300],
                        "fallback": fallback,
                    }
                )
            if k + 1 < self.max_attempts:
                time.sleep(min(90.0, (2**k) * (1.0 + random.random())))
        if not (text and text.strip()):
            self._log(self._record(logged_parts, temperature, body, None, usage, finish, attempts, t_start, "failed"))
            raise RuntimeError(
                "API VLM: no content after %d attempts: %s" % (len(attempts), json.dumps(attempts)[:800])
            )
        self._log(self._record(logged_parts, temperature, body, text, usage, finish, attempts, t_start, "ok"))
        return ApiResponse(text, data, usage)

    def _record(self, logged_parts, temperature, body, text, usage, finish, attempts, t_start, status):
        return {
            "status": status,
            "agent": self.agent_name,
            "out_dir": self.out_dir,
            "model": self.model_name,
            "temperature": temperature,
            "reasoning_effort": body.get("reasoning_effort"),
            "max_tokens": body.get("max_tokens"),
            "system_instruction_chars": len(self.system_instruction),
            "system_instruction_sha256": hashlib.sha256(self.system_instruction.encode("utf-8")).hexdigest(),
            "n_images": sum(1 for p in logged_parts if p["type"] == "image"),
            "prompt_parts": logged_parts,
            "response_text": text,
            "finish_reason": finish,
            "usage": usage,
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "completion_tokens": (usage or {}).get("completion_tokens"),
            "reasoning_tokens": ((usage or {}).get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "total_tokens": (usage or {}).get("total_tokens"),
            "attempts": attempts,
            "latency_seconds": round(time.time() - t_start, 2),
        }
