"""Secret handling helpers: a value that cannot be printed by accident, and redaction of anything that looks like a credential."""
from __future__ import annotations

import re


class Secret:
    """Holds a credential. str(), repr(), format() and logging all show *** ; the value only leaves through reveal()."""
    __slots__ = ("_v",)

    def __init__(self, value: str) -> None:
        self._v = value

    def reveal(self) -> str:
        return self._v

    def __repr__(self) -> str:
        return "Secret(***)"

    __str__ = __repr__

    def __format__(self, spec: str) -> str:
        return "***"

    def __bool__(self) -> bool:
        return bool(self._v)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Secret) and other._v == self._v

    def __hash__(self) -> int:
        return hash(self._v)


# Shapes of Google credentials plus generic "key=..." / bearer forms. Applied to anything that could reach a log line.
_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"\bAQ\.[0-9A-Za-z_\-]{20,}"),
    re.compile(r"\bya29\.[0-9A-Za-z_\-\.]{20,}"),
    re.compile(r"(?i)(api[_-]?key|x-goog-api-key|authorization)\s*[=:]\s*[\"']?(?:Bearer\s+)?[0-9A-Za-z_\-\.]{12,}"),
    re.compile(r"(?i)\bBearer\s+[0-9A-Za-z_\-\.]{12,}"),
]


def redact(text: str, *secrets: str | None) -> str:
    """Remove the configured key (if given) and anything shaped like a credential."""
    out = str(text)
    for s in secrets:
        if s and len(s) >= 8:
            out = out.replace(s, "[redacted]")
    for p in _PATTERNS:
        out = p.sub("[redacted]", out)
    return out
