"""Gemini settings: environment first, then a local git-ignored .env (GEMINI_* keys only, read without touching os.environ).

GEMINI_API_KEY             required. Server-side only.
GEMINI_MODEL               required. There is deliberately NO default: a model that is unsupported or retired must never be chosen silently.
GEMINI_TIMEOUT_SECONDS     default 20  (1-120)    per request to Google
GEMINI_MAX_OUTPUT_TOKENS   default 512 (16-4096)  cost and length cap on every reply
GEMINI_MOVIE_MAX_OUTPUT_TOKENS  default 1024 (16-4096)  cap for movie-aware answers, which explain more
GEMINI_MAX_INPUT_CHARS     default 2000 (50-8000) cap on the user's message
GEMINI_RATE_LIMIT_PER_MINUTE  default 10 (1-120)  per client address
Invalid numbers fall back to the default and are reported by NAME in `problems`, never by value."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

from ..app.config import ROOT
from .security import Secret

ENV_FILE_VAR = "FRAME_ENV_FILE"
_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,79}$")
_KEY_RE = re.compile(r"^[A-Za-z0-9_\-\.]{16,256}$")      # no spaces or control characters; deliberately not tied to one vendor prefix


@dataclass(frozen=True)
class Settings:
    api_key: Secret | None = field(default=None, repr=False)
    model: str | None = None
    timeout_s: float = 20.0
    max_output_tokens: int = 512
    movie_max_output_tokens: int = 1024
    max_input_chars: int = 2000
    rate_per_minute: int = 10
    problems: tuple[str, ...] = ()

    @property
    def configured(self) -> bool:
        return self.api_key is not None and self.model is not None


def _env_file() -> Path:
    return Path(os.getenv(ENV_FILE_VAR) or ROOT / ".env")


def _raw() -> dict[str, str]:
    vals: dict[str, str] = {}
    try:
        p = _env_file()
        if p.is_file():
            vals.update({k: v for k, v in dotenv_values(p).items() if k.startswith("GEMINI_") and v is not None})
    except Exception:  # noqa: BLE001 - an unreadable .env is a configuration problem, not a crash
        vals["__env_unreadable__"] = "1"
    vals.update({k: v for k, v in os.environ.items() if k.startswith("GEMINI_") and v != ""})   # the real environment wins
    return vals


def _num(raw: dict, name: str, default, lo, hi, cast, problems: list[str]):
    v = raw.get(name)
    if v is None:
        return default
    try:
        n = cast(v)
    except (TypeError, ValueError):
        problems.append(name)
        return default
    if not (lo <= n <= hi):
        problems.append(name)
        return default
    return n


def load_settings() -> Settings:
    raw = _raw()
    problems: list[str] = []
    if "__env_unreadable__" in raw:
        problems.append("env_file")
    key = (raw.get("GEMINI_API_KEY") or "").strip().strip('"').strip("'")
    secret = None
    if key:
        if _KEY_RE.match(key):
            secret = Secret(key)
        else:
            problems.append("GEMINI_API_KEY")
    model = (raw.get("GEMINI_MODEL") or "").strip()
    if model.startswith("models/"):
        model = model[len("models/"):]
    if model and not _MODEL_RE.match(model):
        problems.append("GEMINI_MODEL")
        model = ""
    return Settings(
        api_key=secret, model=model or None,
        timeout_s=_num(raw, "GEMINI_TIMEOUT_SECONDS", 20.0, 1, 120, float, problems),
        max_output_tokens=_num(raw, "GEMINI_MAX_OUTPUT_TOKENS", 512, 16, 4096, int, problems),
        movie_max_output_tokens=_num(raw, "GEMINI_MOVIE_MAX_OUTPUT_TOKENS", 1024, 16, 4096, int, problems),
        max_input_chars=_num(raw, "GEMINI_MAX_INPUT_CHARS", 2000, 50, 8000, int, problems),
        rate_per_minute=_num(raw, "GEMINI_RATE_LIMIT_PER_MINUTE", 10, 1, 120, int, problems),
        problems=tuple(problems),
    )


def missing(s: Settings) -> list[str]:
    """Names (never values) of what is not configured."""
    out = []
    if s.api_key is None:
        out.append("GEMINI_API_KEY")
    if s.model is None:
        out.append("GEMINI_MODEL")
    return out
