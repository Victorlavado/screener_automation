"""P1: Sanitize 'Infinity' strings from fundamentals data."""

import math
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from src.data import _sanitize_fundamentals_value, _fetch_single_fundamental, download_fundamentals


# --- Unit tests for _sanitize_fundamentals_value ---

def test_string_infinity_becomes_none():
    assert _sanitize_fundamentals_value("Infinity") is None


def test_string_neg_infinity_becomes_none():
    assert _sanitize_fundamentals_value("-Infinity") is None


def test_float_inf_becomes_none():
    assert _sanitize_fundamentals_value(float("inf")) is None


def test_float_neg_inf_becomes_none():
    assert _sanitize_fundamentals_value(float("-inf")) is None


def test_float_nan_becomes_none():
    assert _sanitize_fundamentals_value(float("nan")) is None


def test_normal_float_passes_through():
    assert _sanitize_fundamentals_value(25.3) == 25.3


def test_string_sector_passes_through():
    assert _sanitize_fundamentals_value("Technology") == "Technology"


def test_zero_passes_through():
    assert _sanitize_fundamentals_value(0) == 0


def test_none_passes_through():
    assert _sanitize_fundamentals_value(None) is None


# --- Integration tests ---

@patch("src.data.yf.Ticker")
def test_infinity_pe_ratio_in_fetch(mock_ticker_cls):
    """Mock yf.Ticker with trailingPE='Infinity' → result pe_ratio is None."""
    mock_instance = MagicMock()
    mock_instance.info = {
        "marketCap": 1_000_000,
        "trailingPE": "Infinity",
        "forwardPE": 15.0,
        "pegRatio": None,
        "priceToBook": 3.2,
        "dividendYield": 0.02,
        "profitMargins": 0.25,
        "revenueGrowth": 0.10,
        "earningsGrowth": 0.05,
        "debtToEquity": 50.0,
        "currentRatio": 1.5,
        "sector": "Technology",
        "industry": "Software",
    }
    mock_ticker_cls.return_value = mock_instance

    result = _fetch_single_fundamental("TEST")
    assert result is not None
    assert result["pe_ratio"] is None
    assert result["market_cap"] == 1_000_000
    assert result["forward_pe"] == 15.0
    assert result["sector"] == "Technology"


@patch("src.data.yf.Ticker")
def test_dataframe_to_parquet_succeeds(mock_ticker_cls, tmp_path):
    """DataFrame from download_fundamentals with Infinity values can be saved to parquet."""
    mock_instance = MagicMock()
    mock_instance.info = {
        "marketCap": 1_000_000,
        "trailingPE": "Infinity",
        "forwardPE": float("inf"),
        "pegRatio": "-Infinity",
        "priceToBook": float("nan"),
        "dividendYield": 0.02,
        "profitMargins": 0.25,
        "revenueGrowth": 0.10,
        "earningsGrowth": 0.05,
        "debtToEquity": 50.0,
        "currentRatio": 1.5,
        "sector": "Technology",
        "industry": "Software",
    }
    mock_ticker_cls.return_value = mock_instance

    df = download_fundamentals(
        ["TEST"],
        show_progress=False,
        cache_max_age_days=0,
    )

    # Should not raise
    output_path = tmp_path / "test.parquet"
    df.to_parquet(output_path)

    # Verify the Infinity values were sanitized
    reloaded = pd.read_parquet(output_path)
    assert reloaded.loc["TEST", "pe_ratio"] is None or pd.isna(reloaded.loc["TEST", "pe_ratio"])
    assert reloaded.loc["TEST", "forward_pe"] is None or pd.isna(reloaded.loc["TEST", "forward_pe"])
