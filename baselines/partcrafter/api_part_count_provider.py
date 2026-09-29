"""Compatibility VLM provider for PartCrafter's official ``--part_suggest`` path (AffordCraft external baseline).

The official tree only ships ``src/utils/providers/gemini_provider.py`` (google-genai SDK, GEMINI_API_KEY). The evaluation
endpoint is OpenAI-compatible (``$EVAL_API_BASE`` + ``/chat/completions``, bearer ``$EVAL_API_KEY``), so this module,
which lives OUTSIDE the official tree, re-implements ``suggest_num_parts`` with the SAME prompt text (imported from the
official gemini_provider), the same ``[image, prompt]`` content order, the same integer parser and the same
retry-once-on-unparsable-reply logic, but sends the request to that endpoint. The official ``src.utils.vlm_utils``
entry point is used unchanged; the provider is registered at runtime by inserting
``PROVIDERS["openai_compatible"] = "api_part_count_provider"`` into the official provider registry (no file edited).

Every exchange (request metadata, full response JSON, parsed value) is appended to a thread-local log (``reset_log`` /
``get_log``) so the lane runner can persist the raw reply per case even when it prefetches calls in worker threads. The API key is read from the environment only and never logged.
"""

import base64
import io
import json
import logging
import os
import ssl
import threading
import time
import urllib.error
import urllib.request

from PIL import Image

from src.utils.providers.gemini_provider import _build_num_parts_prompt, _parse_int_response  # official prompt + parser

logger = logging.getLogger(__name__)

DEFAULT_VLM_MODEL = os.environ.get("PARTCRAFTER_API_MODEL", "gemini-3.8-flash")
TRANSPORT_RETRIES = int(os.environ.get("PARTCRAFTER_API_TRANSPORT_RETRIES", "6"))
TIMEOUT_S = float(os.environ.get("PARTCRAFTER_API_TIMEOUT_S", "240"))
_TLS = threading.local()  # per-thread exchange log (the lane runner prefetches VLM calls in worker threads)


def reset_log():
    _TLS.log = []


def get_log():
    return getattr(_TLS, "log", [])


def _log(rec):
    if not hasattr(_TLS, "log"):
        _TLS.log = []
    _TLS.log.append(rec)


def _endpoint():
    base = os.environ.get("EVAL_API_BASE")
    key = os.environ.get("EVAL_API_KEY")
    if not base or not key:
        raise RuntimeError("EVAL_API_BASE / EVAL_API_KEY must be set in the environment for the API provider.")
    return base.rstrip("/") + "/chat/completions", key


def _ssl_context():
    cafile = os.environ.get("SSL_CERT_FILE") or "/etc/ssl/certs/ca-certificates.crt"
    return ssl.create_default_context(cafile=cafile if os.path.exists(cafile) else None)


def _image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _call(image: Image.Image, prompt: str, model_name: str) -> str:
    """One VLM call: content order [image, prompt] exactly like the official contents=[image, prompt]."""
    url, key = _endpoint()
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(image)}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    record = {"model": model_name, "url": url, "image_size": list(image.size), "attempts": []}
    _log(record)
    ctx = _ssl_context()
    last_err = None
    for attempt in range(1, TRANSPORT_RETRIES + 1):
        req = urllib.request.Request(
            url, data=data, headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S, context=ctx) as resp:
                body = resp.read().decode("utf-8")
                status = resp.status
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:2000]
            record["attempts"].append(
                {
                    "attempt": attempt,
                    "started": t0,
                    "seconds": round(time.time() - t0, 3),
                    "http_status": e.code,
                    "error_body": body,
                }
            )
            last_err = e
            if e.code in (400, 401, 403, 404, 413, 422):
                raise RuntimeError(f"API HTTP {e.code}: {body[:300]}")
            time.sleep(min(60, 5 * attempt))
            continue
        except Exception as e:  # timeouts, connection resets
            record["attempts"].append(
                {
                    "attempt": attempt,
                    "started": t0,
                    "seconds": round(time.time() - t0, 3),
                    "transport_error": f"{type(e).__name__}: {e}",
                }
            )
            last_err = e
            time.sleep(min(60, 5 * attempt))
            continue
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            record["attempts"].append(
                {
                    "attempt": attempt,
                    "started": t0,
                    "seconds": round(time.time() - t0, 3),
                    "http_status": status,
                    "non_json_body": body[:2000],
                }
            )
            last_err = RuntimeError("non-JSON API reply")
            time.sleep(min(60, 5 * attempt))
            continue
        text = ""
        try:
            text = parsed["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            text = ""
        record["attempts"].append(
            {
                "attempt": attempt,
                "started": t0,
                "seconds": round(time.time() - t0, 3),
                "http_status": status,
                "response": parsed,
                "text": text,
            }
        )
        return text
    raise RuntimeError(f"API call failed after {TRANSPORT_RETRIES} transport attempts: {last_err}")


def suggest_num_parts(
    image: Image.Image, max_num_parts: int, mode: str = "object", model_name: str = DEFAULT_VLM_MODEL
) -> int:
    """Same control flow as the official gemini_provider.suggest_num_parts: call, parse int, retry once if unparsable, clamp."""
    prompt = _build_num_parts_prompt(max_num_parts, mode=mode)
    _log({"prompt": prompt, "mode": mode, "max_num_parts": max_num_parts})
    text = _call(image, prompt, model_name)
    try:
        n = _parse_int_response(text)
    except ValueError:
        logger.warning("First VLM response was not a valid integer, retrying...")
        text = _call(image, prompt, model_name)
        n = _parse_int_response(text)
    _log({"parsed_int": n, "clamped": max(1, min(n, max_num_parts))})
    return max(1, min(n, max_num_parts))
