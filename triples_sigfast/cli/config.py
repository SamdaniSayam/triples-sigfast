"""
triples_sigfast.cli.config
---------------------------
API key configuration for the triples-sigfast chat copilot.

The API key is resolved in priority order:

1. ``SIGFAST_API_KEY`` environment variable (highest priority).
2. ``~/.sigfast/config.json`` file (persisted by ``set_api_key``).

The config file is stored with restrictive permissions (``0o600``) so that
only the owning user can read it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_CONFIG_DIR = Path.home() / ".sigfast"
_CONFIG_FILE = _CONFIG_DIR / "config.json"


def get_config_path() -> Path:
    """Return the path to the configuration file (``~/.sigfast/config.json``)."""
    return _CONFIG_FILE


def get_api_key() -> str | None:
    """Return the Gemini API key, or *None* if not configured.

    Resolution order:

    1. ``SIGFAST_API_KEY`` environment variable.
    2. ``api_key`` field inside ``~/.sigfast/config.json``.
    """
    # 1. Environment variable takes precedence.
    env_key = os.environ.get("SIGFAST_API_KEY")
    if env_key:
        return env_key

    # 2. Fall back to the on-disk config file.
    if _CONFIG_FILE.is_file():
        try:
            data = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
            return data.get("api_key") or None
        except (json.JSONDecodeError, OSError):
            return None

    return None


def set_api_key(key: str) -> None:
    """Persist *key* to ``~/.sigfast/config.json``.

    Creates the ``~/.sigfast/`` directory if it does not already exist and
    sets the file permissions to ``0o600`` (owner read/write only).
    """
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    _CONFIG_FILE.write_text(
        json.dumps({"api_key": key}, indent=2) + "\n",
        encoding="utf-8",
    )

    # Restrict permissions so only the current user can read the key.
    _CONFIG_FILE.chmod(0o600)
