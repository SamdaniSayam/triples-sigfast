"""
Tests for the triples_sigfast.cli.config module.

Verifies API key resolution logic: environment variable override,
config file fallback, and key persistence via ``set_api_key``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from triples_sigfast.cli.config import get_api_key, set_api_key

# ---------------------------------------------------------------------------
# get_api_key – environment variable
# ---------------------------------------------------------------------------


class TestGetApiKeyFromEnv:
    """``SIGFAST_API_KEY`` env var is the highest-priority source."""

    @patch.dict(os.environ, {"SIGFAST_API_KEY": "env-key-abc123"})
    @patch("triples_sigfast.cli.config._CONFIG_FILE")
    def test_get_api_key_from_env(self, mock_config_file: MagicMock) -> None:
        """When the env var is set, ``get_api_key()`` returns its value."""
        # Ensure the file path is never consulted.
        mock_config_file.is_file.return_value = False
        assert get_api_key() == "env-key-abc123"


# ---------------------------------------------------------------------------
# get_api_key – config file
# ---------------------------------------------------------------------------


class TestGetApiKeyFromFile:
    """Fallback to ``~/.sigfast/config.json`` when no env var is set."""

    @patch.dict(os.environ, {}, clear=True)
    @patch("triples_sigfast.cli.config._CONFIG_FILE")
    def test_get_api_key_from_file(self, mock_config_file: MagicMock) -> None:
        """When no env var is present, the key is read from the config file."""
        mock_config_file.is_file.return_value = True
        mock_config_file.read_text.return_value = json.dumps(
            {"api_key": "file-key-xyz"}
        )

        # Need to also ensure env var is not set
        with patch.dict(os.environ, {}, clear=True):
            # Remove SIGFAST_API_KEY if present
            os.environ.pop("SIGFAST_API_KEY", None)
            assert get_api_key() == "file-key-xyz"


# ---------------------------------------------------------------------------
# get_api_key – env overrides file
# ---------------------------------------------------------------------------


class TestEnvOverridesFile:
    """When both sources provide a key, env wins."""

    @patch("triples_sigfast.cli.config._CONFIG_FILE")
    def test_env_overrides_file(self, mock_config_file: MagicMock) -> None:
        """Environment variable takes precedence over the config file."""
        mock_config_file.is_file.return_value = True
        mock_config_file.read_text.return_value = json.dumps(
            {"api_key": "file-key-xyz"}
        )

        with patch.dict(os.environ, {"SIGFAST_API_KEY": "env-wins"}):
            assert get_api_key() == "env-wins"


# ---------------------------------------------------------------------------
# set_api_key – writes to config.json
# ---------------------------------------------------------------------------


class TestSetApiKey:
    """``set_api_key`` persists the key to ``~/.sigfast/config.json``."""

    @patch("triples_sigfast.cli.config._CONFIG_FILE")
    @patch("triples_sigfast.cli.config._CONFIG_DIR")
    def test_set_api_key(
        self,
        mock_config_dir: MagicMock,
        mock_config_file: MagicMock,
    ) -> None:
        """The key is written as JSON and file permissions are set to 0o600."""
        set_api_key("new-key-42")

        # Directory should be created if missing.
        mock_config_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # File should be written with the correct JSON payload.
        written_text = mock_config_file.write_text.call_args[0][0]
        payload = json.loads(written_text)
        assert payload == {"api_key": "new-key-42"}

        # Permissions should be locked down.
        mock_config_file.chmod.assert_called_once_with(0o600)


# ---------------------------------------------------------------------------
# get_api_key – neither source configured
# ---------------------------------------------------------------------------


class TestGetApiKeyNone:
    """When no env var and no config file exist, returns ``None``."""

    @patch("triples_sigfast.cli.config._CONFIG_FILE")
    def test_get_api_key_none(self, mock_config_file: MagicMock) -> None:
        """Returns None when neither env var nor file provides a key."""
        mock_config_file.is_file.return_value = False

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("SIGFAST_API_KEY", None)
            assert get_api_key() is None
