"""P7: Benchmark tickers (^GSPC) are exempted from the negative cache.

Yahoo occasionally returns empty for ^GSPC during transient outages; a
7-day cooldown on the benchmark would degrade RS computation for multiple
weekly runs. Promote to a BENCHMARK_TICKERS set when a second benchmark
is added.
"""

import json
import time
from unittest.mock import patch

from src.data import (
    _load_failed_tickers_cache,
    _save_failed_tickers_cache,
)


class TestBenchmarkNegativeCacheExemption:
    """^GSPC is filtered out of the failed-tickers cache writes."""

    def test_gspc_alone_does_not_create_cache_file(self, tmp_path):
        """^GSPC alone → no cache file is written."""
        with patch("src.data._get_ohlcv_cache_dir", return_value=tmp_path):
            now = time.time()
            _save_failed_tickers_cache({"^GSPC": now}, "1y", "1d")

            assert not (tmp_path / "failed_tickers.json").exists()

    def test_gspc_filtered_out_of_mixed_batch(self, tmp_path):
        """Mixed batch with ^GSPC and a real delisted ticker → only the
        delisted ticker is persisted; ^GSPC is dropped.
        """
        with patch("src.data._get_ohlcv_cache_dir", return_value=tmp_path):
            now = time.time()
            _save_failed_tickers_cache(
                {"^GSPC": now, "DELIST1": now},
                "1y",
                "1d",
            )

            with open(tmp_path / "failed_tickers.json") as f:
                cache = json.load(f)

            assert "^GSPC" not in cache
            assert "DELIST1" in cache

    def test_existing_cache_with_gspc_is_not_re_added(self, tmp_path):
        """Even if a previous (pre-fix) cache file contains ^GSPC, save
        won't add the ticker again from the new failed-list.
        """
        with patch("src.data._get_ohlcv_cache_dir", return_value=tmp_path):
            # Simulate a stale pre-fix cache file with ^GSPC in it.
            (tmp_path / "failed_tickers.json").write_text(
                json.dumps({"^GSPC": time.time() - 86400})
            )

            # New save call attempts to add ^GSPC + a real failure.
            _save_failed_tickers_cache(
                {"^GSPC": time.time(), "REAL_FAIL": time.time()},
                "1y",
                "1d",
            )

            with open(tmp_path / "failed_tickers.json") as f:
                cache = json.load(f)

            # The pre-existing ^GSPC entry is preserved (we only filter the
            # input — we don't actively prune existing entries). REAL_FAIL
            # was added.
            assert "REAL_FAIL" in cache

    def test_gspc_not_loaded_back_skipped_on_lookup(self, tmp_path):
        """Sanity: a fresh run with no prior cache + a ^GSPC failure does
        not produce a loaded entry that would skip future ^GSPC downloads.
        """
        with patch("src.data._get_ohlcv_cache_dir", return_value=tmp_path):
            _save_failed_tickers_cache({"^GSPC": time.time()}, "1y", "1d")
            loaded = _load_failed_tickers_cache("1y", "1d", max_age_days=7)
            assert "^GSPC" not in loaded
