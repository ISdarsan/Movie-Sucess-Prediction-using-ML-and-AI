"""Minimal client for the local Ollama server's native /api/chat (httpx only).

Chosen over the OpenAI-compatible endpoint because it accepts a per-request context size and a JSON-schema `format`.
Every failure becomes an OllamaError with a short `kind`, so the orchestrator can fall back instead of crashing.
"""
from __future__ import annotations

import json
import time
from typing import Any

import httpx

from . import config


class OllamaError(RuntimeError):
    def __init__(self, kind: str, detail: str = "") -> None:
        super().__init__(f"{kind}: {detail}" if detail else kind)
        self.kind = kind


def status(timeout: float = 3.0) -> dict[str, Any]:
    """Is the server reachable and is the configured model installed? Never raises."""
    try:
        r = httpx.get(f"{config.OLLAMA_URL}/api/tags", timeout=timeout)
        r.raise_for_status()
        names = {m.get("name") for m in r.json().get("models", [])}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "model": config.OLLAMA_MODEL, "reason": "ollama_unreachable", "detail": type(exc).__name__}
    if config.OLLAMA_MODEL not in names:
        return {"available": False, "model": config.OLLAMA_MODEL, "reason": "model_not_installed"}
    return {"available": True, "model": config.OLLAMA_MODEL}


def chat_json(system: str, user: str, schema: dict, *, num_predict: int = 500, timeout: float | None = None) -> tuple[dict, dict]:
    """One schema-constrained completion. Returns (parsed_json, meta). temperature 0 for repeatable output."""
    body = {
        "model": config.OLLAMA_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False, "format": schema, "keep_alive": config.OLLAMA_KEEP_ALIVE,
        "options": {"num_ctx": config.OLLAMA_NUM_CTX, "num_predict": num_predict, "temperature": 0},
    }
    t0 = time.perf_counter()
    for attempt in (1, 2):
        try:
            r = httpx.post(f"{config.OLLAMA_URL}/api/chat", json=body, timeout=timeout or config.LLM_TIMEOUT_S)
            r.raise_for_status()
            j = r.json()
            break
        except httpx.TimeoutException as exc:
            raise OllamaError("timeout", type(exc).__name__) from exc
        except httpx.ConnectError as exc:
            raise OllamaError("unreachable", "connection refused") from exc
        except httpx.HTTPStatusError as exc:
            # A 5xx is Ollama's runner failing to load the model (observed: an intermittent CUDA start-up crash that worked on the next try).
            # Retry that ONCE; timeouts and everything else are not retried, so a slow call can never double the wait.
            if exc.response.status_code >= 500 and attempt == 1:
                continue
            raise OllamaError("http_error", str(exc.response.status_code)) from exc
        except Exception as exc:  # noqa: BLE001
            raise OllamaError("error", type(exc).__name__) from exc
    if j.get("done_reason") == "length":
        raise OllamaError("truncated", "output hit the token limit")
    try:
        parsed = json.loads(j["message"]["content"])
    except Exception as exc:  # noqa: BLE001
        raise OllamaError("invalid_json", type(exc).__name__) from exc
    if not isinstance(parsed, dict):
        raise OllamaError("invalid_json", "not an object")
    meta = {"model": config.OLLAMA_MODEL, "seconds": round(time.perf_counter() - t0, 1), "eval_tokens": j.get("eval_count"),
            "load_seconds": round(j.get("load_duration", 0) / 1e9, 1)}
    return parsed, meta
