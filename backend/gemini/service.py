"""Calls Google's Gemini API through the official `google-genai` SDK, one bounded attempt per request.

Every failure becomes a GeminiError with a stable `kind` and a SAFE user-facing message. The raw provider text is only inspected to
classify the failure; it is never returned, and never logged (it can echo request details). One attempt only: the SDK is not configured
to retry, so a failing provider cannot make this loop."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache

import httpx
from google import genai
from google.genai import errors, types

from .config import Settings
from .security import redact

log = logging.getLogger("frame.gemini")

# kind -> (HTTP status, machine-readable error, message safe to show a user)
CATALOG: dict[str, tuple[int, str, str]] = {
    "not_configured": (503, "assistant_not_configured", "The AI assistant is not configured on this server."),
    "invalid_api_key": (503, "assistant_misconfigured", "The AI service rejected this server's credentials. The administrator needs to check the configuration."),
    "rate_limited": (429, "assistant_rate_limited", "The AI service is busy or its quota is used up. Please try again later."),
    "timeout": (504, "assistant_timeout", "The AI service took too long to answer. Please try again."),
    "network": (502, "assistant_unreachable", "The AI service could not be reached. Please try again."),
    "model_unavailable": (503, "assistant_model_unavailable", "The configured AI model is not available."),
    "upstream_error": (502, "assistant_upstream_error", "The AI service had a problem. Please try again."),
    "bad_request": (502, "assistant_rejected_request", "The AI service could not process that request."),
    "declined": (422, "assistant_declined", "The AI service declined to answer that. Try rephrasing your question."),
    "no_reply": (502, "assistant_no_reply", "The AI service returned no reply. Please try again."),
}
MAX_REPLY_CHARS = 8000
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f‪-‮⁦-⁩]")


class GeminiError(Exception):
    def __init__(self, kind: str) -> None:
        super().__init__(kind)
        self.kind = kind
        self.status, self.error, self.message = CATALOG[kind]


@dataclass(frozen=True)
class Reply:
    text: str
    truncated: bool
    model: str


def make_client(api_key: str, timeout_ms: int) -> genai.Client:
    """The only place a client is built (tests replace it). No retry options are set: exactly one attempt per request."""
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_ms))


@lru_cache(maxsize=2)
def _client(api_key: str, timeout_ms: int):
    return make_client(api_key, timeout_ms)


def _chain(exc: BaseException):
    seen = 0
    while exc is not None and seen < 5:
        yield exc
        exc = exc.__cause__ or exc.__context__
        seen += 1


def classify(exc: BaseException) -> str:
    """Map any exception from the SDK or transport to a CATALOG kind. The message is read for matching only."""
    if isinstance(exc, errors.APIError):
        msg = str(exc).lower()
        code = getattr(exc, "code", None)
        status = str(getattr(exc, "status", "") or "").upper()
        if ("api key" in msg and any(w in msg for w in ("not valid", "invalid", "expired", "not found"))) or "api_key_invalid" in msg \
                or code in (401, 403) or status in ("UNAUTHENTICATED", "PERMISSION_DENIED"):
            return "invalid_api_key"
        if code == 429 or status == "RESOURCE_EXHAUSTED":
            return "rate_limited"
        if code == 404 or status == "NOT_FOUND" or ("model" in msg and any(w in msg for w in ("not found", "not supported", "is not available", "no longer available"))):
            return "model_unavailable"
        if code in (408, 504) or status == "DEADLINE_EXCEEDED":
            return "timeout"
        if isinstance(code, int) and code >= 500:
            return "upstream_error"
        return "bad_request"
    for e in _chain(exc):
        if isinstance(e, (httpx.TimeoutException, TimeoutError)) or "timeout" in type(e).__name__.lower() or "timedout" in type(e).__name__.lower():
            return "timeout"
    for e in _chain(exc):
        if isinstance(e, (httpx.TransportError, ConnectionError, OSError)) or type(e).__name__ in ("ConnectionError", "ConnectError", "ProxyError", "SSLError"):
            return "network"
    return "upstream_error"


def _text_of(resp) -> tuple[str, bool]:
    try:
        text = resp.text
    except Exception:  # noqa: BLE001 - the SDK raises when parts are absent
        text = None
    cands = getattr(resp, "candidates", None) or []
    finish = str(getattr(cands[0], "finish_reason", "") or "").upper() if cands else ""
    if not isinstance(text, str) or not text.strip():
        block = getattr(getattr(resp, "prompt_feedback", None), "block_reason", None)
        if block or any(w in finish for w in ("SAFETY", "BLOCKLIST", "PROHIBITED", "RECITATION", "SPII")):
            raise GeminiError("declined")
        raise GeminiError("no_reply")
    clean = _CTRL.sub("", text).strip()[:MAX_REPLY_CHARS]
    if not clean:
        raise GeminiError("no_reply")
    return clean, "MAX_TOKENS" in finish


def generate(message: str, system: str, settings: Settings, max_output_tokens: int | None = None) -> Reply:
    """One bounded request to Gemini. Raises GeminiError only."""
    if not settings.configured:
        raise GeminiError("not_configured")
    key = settings.api_key.reveal()
    try:
        client = _client(key, int(settings.timeout_s * 1000))
        resp = client.models.generate_content(
            model=settings.model, contents=message,
            config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=max_output_tokens or settings.max_output_tokens, temperature=0.4),
        )
        text, truncated = _text_of(resp)
        return Reply(text=text, truncated=truncated, model=settings.model)
    except GeminiError:
        raise
    except Exception as exc:  # noqa: BLE001
        kind = classify(exc)
        # operator diagnostics: the failure class and HTTP code only. The provider's own text is never logged or returned.
        log.warning("gemini request failed: kind=%s exception=%s code=%s", kind, redact(type(exc).__name__, key), getattr(exc, "code", None))
        raise GeminiError(kind) from None
