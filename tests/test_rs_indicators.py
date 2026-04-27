"""Tests for compute_rs_indicators (rs_ir, rs_weighted_return, rs_rating).

Covers:
  - vectorized IR with constant alpha (positive case)
  - zero alpha + std floor (negative case where excess vol collapses)
  - short history → NaN
  - EU calendar misalignment (US holiday inside the 63-day window)
  - duplicate timestamps in close (pre-pct_change defense)
  - rs_rating ranking on three tickers
  - rs_rating with NaN history (NaN propagates, doesn't poison rank)
  - missing/empty SP500 → all three columns NaN, no exception
"""

import logging

import numpy as np
import pandas as pd
import pytest

from src.indicators import (
    RS_IR_MIN_OBS,
    RS_IR_STD_FLOOR,
    RS_IR_WINDOW,
    compute_rs_indicators,
)


def _make_indicators_df(tickers, weighted_inputs=None):
    """Build a minimal indicators_df for RS computation.

    weighted_inputs: optional dict of ticker -> (R1M, R3M, R6M, R12M).
    Defaults all to zero so rs_weighted_return is zero everywhere.
    """
    n = len(tickers)
    weighted_inputs = weighted_inputs or {}
    rows = []
    for t in tickers:
        r1m, r3m, r6m, r12m = weighted_inputs.get(t, (0.0, 0.0, 0.0, 0.0))
        rows.append({
            'ticker': t,
            'close': 100.0,
            'pct_change_1m': r1m,
            'pct_change_3m': r3m,
            'pct_change_6m': r6m,
            'pct_change_1y': r12m,
        })
    return pd.DataFrame(rows).set_index('ticker')


def _make_close_df(returns: pd.Series, start_price: float = 100.0) -> pd.DataFrame:
    """Reconstruct a Close-only OHLCV DataFrame from a returns series."""
    prices = start_price * (1 + returns).cumprod()
    return pd.DataFrame({'Close': prices})


class TestInformationRatio:
    """Vectorized IR computation."""

    def test_constant_positive_alpha(self):
        """Stock = SP500 + 0.001/day → IR ≈ 0.001 / 0.005 * sqrt(252) ≈ 3.17.

        Excess returns are constant 0.001 with sample std = 0 → floor binds at
        RS_IR_STD_FLOOR (0.005), giving IR = 0.001/0.005 * sqrt(252) ≈ 3.17.
        """
        rng = np.random.default_rng(0)
        dates = pd.date_range('2025-01-01', periods=100, freq='B')
        sp500_returns = pd.Series(rng.normal(0.0005, 0.01, 100), index=dates)

        stock_returns = sp500_returns + 0.001
        ohlcv = {'AAA': _make_close_df(stock_returns)}
        sp500_close = 100.0 * (1 + sp500_returns).cumprod()

        df = _make_indicators_df(['AAA'])
        result = compute_rs_indicators(df, sp500_close, ohlcv)

        ir = result.loc['AAA', 'rs_ir']
        expected = 0.001 / RS_IR_STD_FLOOR * np.sqrt(252)
        assert abs(ir - expected) < 0.01, f"Expected ~{expected:.3f}, got {ir:.3f}"

    def test_zero_alpha_floor_binds(self):
        """Stock perfectly tracks SP500 → IR ≈ 0 (numerator is 0)."""
        rng = np.random.default_rng(1)
        dates = pd.date_range('2025-01-01', periods=100, freq='B')
        sp500_returns = pd.Series(rng.normal(0.0005, 0.01, 100), index=dates)

        stock_returns = sp500_returns.copy()
        ohlcv = {'AAA': _make_close_df(stock_returns)}
        sp500_close = 100.0 * (1 + sp500_returns).cumprod()

        df = _make_indicators_df(['AAA'])
        result = compute_rs_indicators(df, sp500_close, ohlcv)

        ir = result.loc['AAA', 'rs_ir']
        assert abs(ir) < 0.001, f"Expected IR ≈ 0, got {ir}"

    def test_short_history_returns_nan(self):
        """< RS_IR_MIN_OBS aligned days → rs_ir is NaN."""
        rng = np.random.default_rng(2)
        # Only 30 days — below the 50-day minimum.
        dates = pd.date_range('2025-01-01', periods=30, freq='B')
        sp500_returns = pd.Series(rng.normal(0.0005, 0.01, 30), index=dates)

        stock_returns = sp500_returns + 0.001
        ohlcv = {'AAA': _make_close_df(stock_returns)}
        sp500_close = 100.0 * (1 + sp500_returns).cumprod()

        df = _make_indicators_df(['AAA'])
        result = compute_rs_indicators(df, sp500_close, ohlcv)

        assert pd.isna(result.loc['AAA', 'rs_ir'])

    def test_align_then_tail_eu_holiday(self):
        """Stock has a date present in its index but absent in SP500
        (e.g., July 4 — US closed, EU open). That date is dropped in the
        alignment step before the trailing window is taken.
        """
        rng = np.random.default_rng(3)
        # Simulate 80 business days; remove July 4 from SP500.
        all_dates = pd.date_range('2025-05-01', periods=80, freq='B')
        july_4 = pd.Timestamp('2025-07-04')

        sp500_returns = pd.Series(rng.normal(0.0005, 0.01, 80), index=all_dates)
        # July 4 may or may not land in the business-day range; remove if present.
        if july_4 in sp500_returns.index:
            sp500_returns = sp500_returns.drop(july_4)

        stock_returns = pd.Series(rng.normal(0.001, 0.012, 80), index=all_dates)
        ohlcv = {'EU': _make_close_df(stock_returns)}
        sp500_close = 100.0 * (1 + sp500_returns).cumprod()

        df = _make_indicators_df(['EU'])
        result = compute_rs_indicators(df, sp500_close, ohlcv)

        # Should compute a finite IR (≥ 50 aligned days available).
        ir = result.loc['EU', 'rs_ir']
        assert pd.notna(ir), "Expected finite IR, got NaN"

    def test_duplicate_date_in_close(self):
        """Stock with a duplicated timestamp → deduplicated, no exception."""
        rng = np.random.default_rng(4)
        dates = pd.date_range('2025-01-01', periods=100, freq='B')
        sp500_returns = pd.Series(rng.normal(0.0005, 0.01, 100), index=dates)
        stock_returns = sp500_returns + 0.001

        # Build a close series and inject a duplicate index.
        prices = 100.0 * (1 + stock_returns).cumprod()
        dup_close = pd.concat([prices, prices.iloc[[10]]])
        dup_close = dup_close.sort_index()
        ohlcv = {'AAA': pd.DataFrame({'Close': dup_close})}

        sp500_close = 100.0 * (1 + sp500_returns).cumprod()

        df = _make_indicators_df(['AAA'])
        result = compute_rs_indicators(df, sp500_close, ohlcv)

        # Should not raise, IR should be finite.
        assert pd.notna(result.loc['AAA', 'rs_ir'])


class TestRSRating:
    """IBD-style weighted return + percentile rank.

    Per plan: when SP500 is missing, ALL three columns (rs_ir,
    rs_weighted_return, rs_rating) are NaN — the function bails early.
    These tests therefore provide a valid SP500 series and OHLCV so the
    ranking path runs.
    """

    @staticmethod
    def _valid_sp500_and_ohlcv(tickers):
        """Build a 100-day SP500 series and matching trivial OHLCV per ticker."""
        rng = np.random.default_rng(42)
        dates = pd.date_range('2025-01-01', periods=100, freq='B')
        sp500_returns = pd.Series(rng.normal(0.0005, 0.01, 100), index=dates)
        sp500_close = 100.0 * (1 + sp500_returns).cumprod()

        ohlcv = {}
        for t in tickers:
            stock_returns = sp500_returns + 0.0005
            ohlcv[t] = _make_close_df(stock_returns)
        return sp500_close, ohlcv

    def test_three_tickers_ranking(self):
        """Three tickers with weighted returns 100 / 50 / 0 → top one gets the
        highest percentile, bottom gets the lowest, middle gets the middle.
        """
        df = _make_indicators_df(
            ['HIGH', 'MID', 'LOW'],
            weighted_inputs={
                'HIGH': (100, 100, 100, 100),
                'MID':  (50, 50, 50, 50),
                'LOW':  (0, 0, 0, 0),
            },
        )
        sp500_close, ohlcv = self._valid_sp500_and_ohlcv(df.index)

        result = compute_rs_indicators(df, sp500_close, ohlcv)

        # weighted_return is (0.4+0.3+0.2+0.1) * value = 1.0 * value
        assert abs(result.loc['HIGH', 'rs_weighted_return'] - 100) < 1e-6
        assert abs(result.loc['MID',  'rs_weighted_return'] - 50)  < 1e-6
        assert abs(result.loc['LOW',  'rs_weighted_return'] - 0)   < 1e-6

        # rs_rating uses pct=True average rank → 100, ~66.67, ~33.33
        assert result.loc['HIGH', 'rs_rating'] > result.loc['MID', 'rs_rating']
        assert result.loc['MID',  'rs_rating'] > result.loc['LOW', 'rs_rating']
        assert abs(result.loc['HIGH', 'rs_rating'] - 100) < 0.01

    def test_rs_rating_with_nan_history(self):
        """A ticker with NaN R12M → rs_weighted_return is NaN and rs_rating
        is NaN. The other tickers' rs_rating values are unaffected.
        """
        df = _make_indicators_df(
            ['HIGH', 'MID', 'LOW', 'NEW'],
            weighted_inputs={
                'HIGH': (100, 100, 100, 100),
                'MID':  (50, 50, 50, 50),
                'LOW':  (0, 0, 0, 0),
                'NEW':  (10, 10, 10, np.nan),
            },
        )
        sp500_close, ohlcv = self._valid_sp500_and_ohlcv(df.index)

        result = compute_rs_indicators(df, sp500_close, ohlcv)

        assert pd.isna(result.loc['NEW', 'rs_weighted_return'])
        assert pd.isna(result.loc['NEW', 'rs_rating'])

        # The three other tickers still have finite ratings.
        for t in ('HIGH', 'MID', 'LOW'):
            assert pd.notna(result.loc[t, 'rs_rating'])


class TestEmptySP500:
    """Defensive paths: missing/empty SP500."""

    def test_no_sp500_all_nan(self, caplog):
        """Passing None for sp500_close → all three columns NaN, warning logged."""
        df = _make_indicators_df(['AAA', 'BBB'])

        with caplog.at_level(logging.WARNING):
            result = compute_rs_indicators(df, sp500_close=None, ohlcv={})

        assert result['rs_ir'].isna().all()
        assert result['rs_weighted_return'].isna().all()
        assert result['rs_rating'].isna().all()
        assert any('SP500' in r.message for r in caplog.records)

    def test_empty_sp500_series_all_nan(self):
        """Empty SP500 Series → same as None."""
        df = _make_indicators_df(['AAA'])
        empty = pd.Series([], dtype=float)
        result = compute_rs_indicators(df, sp500_close=empty, ohlcv={})

        assert pd.isna(result.loc['AAA', 'rs_ir'])
        assert pd.isna(result.loc['AAA', 'rs_rating'])
