"""Shared test fixtures for the screener automation test suite."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def tmp_cache_dir(tmp_path):
    """Provide a temporary cache directory."""
    cache_dir = tmp_path / ".cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def mock_yf_ticker():
    """Factory fixture that creates a mock yf.Ticker with custom .info dict."""

    def _make(info_dict: dict):
        mock_ticker = MagicMock()
        mock_ticker.info = info_dict
        return mock_ticker

    return _make


@pytest.fixture
def tmp_checkpoint_dir(tmp_path):
    """Provide a temporary checkpoint directory with helper functions."""
    ckpt = tmp_path / "pipeline" / "2026-01-01"
    ckpt.mkdir(parents=True)

    def _write_meta(meta: dict):
        with open(ckpt / "checkpoint.json", "w") as f:
            json.dump(meta, f)

    def _read_meta() -> dict:
        meta_path = ckpt / "checkpoint.json"
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f)
        return {}

    ckpt.write_meta = _write_meta
    ckpt.read_meta = _read_meta
    return ckpt
