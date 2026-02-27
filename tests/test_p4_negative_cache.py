"""P4: Negative cache for permanently-failed OHLCV tickers."""

import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pandas as pd
import pytest

from src.data import (
    _load_failed_tickers_cache,
    _save_failed_tickers_cache,
    download_ohlcv_batch,
)


class TestFailedTickersCache:
    """Unit tests for the negative cache helpers."""

    def test_save_and_load_failed_tickers(self, tmp_path):
        """Roundtrip: save 2 failed tickers → load returns 2."""
        with patch("src.data._get_ohlcv_cache_dir", return_value=tmp_path):
            failed = {"DELIST1": time.time(), "DELIST2": time.time()}
            _save_failed_tickers_cache(failed, "1y", "1d")

            loaded = _load_failed_tickers_cache("1y", "1d", max_age_days=7)
            assert set(loaded.keys()) == {"DELIST1", "DELIST2"}

    def test_expired_entries_excluded(self, tmp_path):
        """An 8-day-old entry should be excluded with max_age_days=7."""
        with patch("src.data._get_ohlcv_cache_dir", return_value=tmp_path):
            now = time.time()
            failed = {
                "OLD": now - (8 * 86400),   # 8 days old
                "FRESH": now - (1 * 86400),  # 1 day old
            }
            _save_failed_tickers_cache(failed, "1y", "1d")

            loaded = _load_failed_tickers_cache("1y", "1d", max_age_days=7)
            assert "OLD" not in loaded
            assert "FRESH" in loaded

    def test_empty_cache_returns_empty_dict(self, tmp_path):
        """No cache file → {} without error."""
        with patch("src.data._get_ohlcv_cache_dir", return_value=tmp_path):
            loaded = _load_failed_tickers_cache("1y", "1d")
            assert loaded == {}


class TestNegativeCacheIntegration:
    """Integration tests: negative cache filters tickers from download."""

    @patch("src.data.time.sleep")
    @patch("src.data.yf.download")
    @patch("src.data._load_ohlcv_cache", return_value={})
    @patch("src.data._save_ohlcv_cache")
    def test_known_delisted_not_downloaded(
        self, mock_save, mock_load, mock_yf_dl, mock_sleep, tmp_path
    ):
        """A ticker in the negative cache should not appear in the yf.download call."""
        now = time.time()
        neg_cache = {"DELIST1": now}

        with patch("src.data._load_failed_tickers_cache", return_value=neg_cache), \
             patch("src.data._save_failed_tickers_cache"):
            # yf.download returns data for GOOD ticker
            mock_df = pd.DataFrame(
                {"Close": [100.0]},
                index=pd.DatetimeIndex(["2026-01-01"]),
            )
            mock_yf_dl.return_value = mock_df

            result = download_ohlcv_batch(
                ["GOOD", "DELIST1"],
                period="1y",
                interval="1d",
                show_progress=False,
                cache_max_age_hours=0,
            )

            # yf.download should have been called with only "GOOD"
            call_args = mock_yf_dl.call_args
            assert "DELIST1" not in call_args[1].get("tickers", call_args[0][0] if call_args[0] else "")

    @patch("src.data.time.sleep")
    @patch("src.data.yf.download")
    @patch("src.data._load_ohlcv_cache", return_value={})
    @patch("src.data._save_ohlcv_cache")
    def test_all_delisted_batch_skips_retry(
        self, mock_save, mock_load, mock_yf_dl, mock_sleep, tmp_path
    ):
        """A batch of 2 delisted tickers → no 30/60/120s retry sleeps."""
        now = time.time()
        neg_cache = {"DELIST1": now, "DELIST2": now}

        with patch("src.data._load_failed_tickers_cache", return_value=neg_cache), \
             patch("src.data._save_failed_tickers_cache"):
            result = download_ohlcv_batch(
                ["DELIST1", "DELIST2"],
                period="1y",
                interval="1d",
                show_progress=False,
                cache_max_age_hours=0,
            )

            # No yf.download call should have been made
            mock_yf_dl.assert_not_called()

            # No large retry sleeps (30s+)
            for c in mock_sleep.call_args_list:
                if c.args:
                    assert c.args[0] < 30, f"Unexpected large sleep: {c.args[0]}"
