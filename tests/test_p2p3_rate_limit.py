"""P2+P3: Fundamentals rate limiting + crumb error handling."""

import inspect
from unittest.mock import patch, MagicMock, call

import pytest

from src.data import _fetch_single_fundamental, download_fundamentals


class TestExponentialBackoff:
    """Verify retry delays use exponential backoff."""

    @patch("src.data.time.sleep")
    @patch("src.data.yf.Ticker")
    def test_retry_delays_are_exponential(self, mock_ticker_cls, mock_sleep):
        """Sleep calls should be [2, 4, 8, 16] for base=2.0, 4 retries."""
        mock_ticker_cls.return_value.info = property(
            lambda self: (_ for _ in ()).throw(Exception("test error"))
        )
        # Make every call raise
        mock_ticker_cls.return_value = MagicMock()
        type(mock_ticker_cls.return_value).info = property(
            lambda self: (_ for _ in ()).throw(Exception("test error"))
        )

        result = _fetch_single_fundamental("TEST", max_retries=4, retry_delay=2.0)

        assert result is None
        sleep_values = [c.args[0] for c in mock_sleep.call_args_list]
        assert sleep_values == [2.0, 4.0, 8.0, 16.0]

    @patch("src.data.time.sleep")
    @patch("src.data.yf.Ticker")
    def test_crumb_error_triggers_retry(self, mock_ticker_cls, mock_sleep):
        """A 401 'Invalid Crumb' error should trigger retry and eventually succeed."""
        good_info = {
            "marketCap": 1_000_000,
            "trailingPE": 25.0,
            "forwardPE": 20.0,
            "pegRatio": 1.5,
            "priceToBook": 3.0,
            "dividendYield": 0.02,
            "profitMargins": 0.25,
            "revenueGrowth": 0.10,
            "earningsGrowth": 0.05,
            "debtToEquity": 50.0,
            "currentRatio": 1.5,
            "sector": "Technology",
            "industry": "Software",
        }

        call_count = 0

        def info_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("401 Invalid Crumb")
            return good_info

        mock_instance = MagicMock()
        type(mock_instance).info = property(lambda self: info_side_effect())
        mock_ticker_cls.return_value = mock_instance

        result = _fetch_single_fundamental("TEST", max_retries=4, retry_delay=2.0)
        assert result is not None
        assert result["pe_ratio"] == 25.0


class TestDownloadFundamentalsDefaults:
    """Verify the conservative defaults for rate limiting."""

    def test_default_max_workers_is_2(self):
        """download_fundamentals should default to max_workers=2."""
        sig = inspect.signature(download_fundamentals)
        assert sig.parameters["max_workers"].default == 2

    @patch("src.data._save_fundamentals_cache")
    @patch("src.data._fetch_single_fundamental")
    @patch("src.data.time.sleep")
    @patch("src.data._load_fundamentals_cache", return_value={})
    def test_submission_delay_is_1_second(
        self, mock_cache, mock_sleep, mock_fetch, mock_save
    ):
        """Delay between thread submissions should be >= 1.0s."""
        mock_fetch.return_value = {
            "ticker": "TEST",
            "market_cap": 1_000_000,
            "pe_ratio": 25.0,
            "forward_pe": 20.0,
            "peg_ratio": None,
            "price_to_book": 3.0,
            "dividend_yield": 0.02,
            "profit_margin": 0.25,
            "revenue_growth": 0.10,
            "earnings_growth": 0.05,
            "debt_to_equity": 50.0,
            "current_ratio": 1.5,
            "sector": "Technology",
            "industry": "Software",
        }

        download_fundamentals(
            ["TEST1", "TEST2", "TEST3"],
            show_progress=False,
            cache_max_age_days=0,
        )

        # Check that sleep was called with >= 1.0 for submission delays
        sleep_calls = [c.args[0] for c in mock_sleep.call_args_list if c.args]
        submission_delays = [s for s in sleep_calls if s >= 1.0]
        # Should have at least 2 submission delays (between 3 ticker submissions)
        assert len(submission_delays) >= 2
