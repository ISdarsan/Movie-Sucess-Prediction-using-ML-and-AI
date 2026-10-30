"""Helper: list the Gemini models the configured key can use for generateContent. Prints model NAMES only, never the key.

    .venv\\Scripts\\python.exe -m backend.gemini.list_models

Use it to choose GEMINI_MODEL. The backend never picks a model for you."""
from __future__ import annotations

import sys

from google import genai
from google.genai import types

from .config import load_settings, missing
from .security import redact


def main() -> int:
    s = load_settings()
    if s.api_key is None:
        print("GEMINI_API_KEY is not set (environment or .env).")
        return 2
    key = s.api_key.reveal()
    try:
        client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(s.timeout_s * 1000)))
        rows = []
        for m in client.models.list():
            actions = list(getattr(m, "supported_actions", None) or [])
            if "generateContent" in actions:
                rows.append((m.name.removeprefix("models/"), getattr(m, "display_name", "") or ""))
    except Exception as exc:  # noqa: BLE001
        print(f"Could not list models: {type(exc).__name__} (code {getattr(exc, 'code', 'n/a')}). {redact(str(exc)[:160], key)}")
        return 1
    print(f"{len(rows)} models support generateContent for this key:")
    for name, display in sorted(rows):
        print(f"  {name:<44} {display}")
    print("\nSet GEMINI_MODEL in .env to one of the names above. Currently:", s.model or "(not set)", "| missing:", missing(s) or "nothing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
