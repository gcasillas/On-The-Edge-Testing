from __future__ import annotations

import os
from pathlib import Path

from google import genai


def load_local_env() -> None:
    """Load environment variables from the secure user config file if present."""
    env_path = Path.home() / ".config" / "on-the-edge" / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        os.environ.setdefault(key, value)


def get_gemini_client() -> genai.Client:
    """Build a Gemini client using the secure environment file or process environment."""
    load_local_env()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Add GEMINI_API_KEY or GOOGLE_API_KEY to ~/.config/on-the-edge/.env"
        )
    return genai.Client(api_key=api_key)


if __name__ == "__main__":
    client = get_gemini_client()
    print(type(client).__name__)
