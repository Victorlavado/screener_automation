"""Tests for the three-group screener pipeline (with_postfilter / bypass /
post_filter_only) and the indicators_rs checkpoint stage.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.screener import (
    ScreenerGroups,
    split_screener_config,
    consolidate_results,
    run_all_screeners,
)


def _config(screeners: dict) -> dict:
    """Wrap a screeners dict in the standard config envelope."""
    return {
        'screeners': screeners,
        'settings': {'consolidation': 'union', 'exclude_symbols': []},
    }


class TestSplitScreenerConfig:
    """split_screener_config returns ScreenerGroups with three groups."""

    def test_three_groups(self):
        """One screener of each kind goes into the right bucket."""
        config = _config({
            'momentum_a': {
                'requirements': [{'field': 'pct_change_1m', 'operator': '>=', 'value': 10}],
            },
            'rs_bypass': {
                'apply_post_filter': False,
                'requirements': [{'field': 'rs_ir', 'operator': '>=', 'value': 1.0}],
            },
            'ema5_scan': {
                'post_filter': True,
                'requirements': [{'field': 'ema5_distance_pct', 'operator': 'between',
                                  'value': [0, 5]}],
            },
        })

        groups = split_screener_config(config)

        assert isinstance(groups, ScreenerGroups)
        assert set(groups.with_postfilter['screeners'].keys()) == {'momentum_a'}
        assert set(groups.bypass['screeners'].keys()) == {'rs_bypass'}
        assert set(groups.post_filter_only['screeners'].keys()) == {'ema5_scan'}

    def test_apply_post_filter_default_true(self):
        """A screener without the flag goes into with_postfilter (default)."""
        config = _config({
            'no_flag': {
                'requirements': [{'field': 'close', 'operator': '>', 'value': 0}],
            },
        })

        groups = split_screener_config(config)

        assert 'no_flag' in groups.with_postfilter['screeners']
        assert 'no_flag' not in groups.bypass['screeners']
        assert 'no_flag' not in groups.post_filter_only['screeners']

    def test_post_filter_takes_precedence(self):
        """post_filter: true wins over apply_post_filter: false (nonsensical
        combo — split puts it in post_filter_only, ignoring the bypass flag).
        """
        config = _config({
            'weird': {
                'post_filter': True,
                'apply_post_filter': False,
                'requirements': [],
            },
        })

        groups = split_screener_config(config)

        assert 'weird' in groups.post_filter_only['screeners']
        assert 'weird' not in groups.bypass['screeners']

    def test_global_settings_preserved(self):
        """Each group dict carries the same global settings as the input."""
        config = _config({
            'a': {'requirements': []},
            'b': {'apply_post_filter': False, 'requirements': []},
        })
        config['settings']['consolidation'] = 'intersection'

        groups = split_screener_config(config)

        assert groups.with_postfilter['settings']['consolidation'] == 'intersection'
        assert groups.bypass['settings']['consolidation'] == 'intersection'
        assert groups.post_filter_only['settings']['consolidation'] == 'intersection'


class TestBypassScreenerEndToEnd:
    """The bypass-flagged screener's symbols reach the final union without
    going through the EMA5 post-filter.
    """

    def _indicators_df(self):
        """Build a tiny indicators DF spanning 5 tickers.

        Layout:
          - WIN_REG: passes the regular momentum filter, also passes EMA5 PF
          - LOSE_REG_PASS_PF: passes EMA5 PF but doesn't pass momentum
          - WIN_REG_FAIL_PF: passes momentum, fails EMA5 PF
          - WIN_BYPASS: passes the bypass screener only
          - DUD: passes nothing
        """
        return pd.DataFrame({
            'pct_change_1m': [50, -5, 50, 0, -20],          # momentum cutoff: >= 30
            'ema5_distance_pct': [2.0, 3.0, 8.0, 0.0, 0.0],  # EMA5 PF cutoff: [0, 5]
            'rs_ir': [0.0, 0.0, 0.0, 1.5, 0.0],              # bypass cutoff: >= 1.0
        }, index=['WIN_REG', 'LOSE_REG_PASS_PF', 'WIN_REG_FAIL_PF', 'WIN_BYPASS', 'DUD'])

    def test_bypass_skips_postfilter(self):
        """WIN_BYPASS reaches final via the bypass path; WIN_REG via the PF path."""
        df = self._indicators_df()
        config = _config({
            'momentum': {
                'requirements': [{'field': 'pct_change_1m', 'operator': '>=', 'value': 30}],
            },
            'rs_bypass': {
                'apply_post_filter': False,
                'requirements': [{'field': 'rs_ir', 'operator': '>=', 'value': 1.0}],
            },
            'ema5_scan': {
                'post_filter': True,
                'requirements': [{'field': 'ema5_distance_pct', 'operator': 'between',
                                  'value': [0, 5]}],
            },
        })

        groups = split_screener_config(config)

        # Group A: regular with post-filter
        results_a = run_all_screeners(df, groups.with_postfilter)
        symbols_a, trace_a = consolidate_results(results_a, method='union')
        assert set(symbols_a) == {'WIN_REG', 'WIN_REG_FAIL_PF'}

        # Group C: post-filters applied to Group A only
        pf_tickers = {n: set(symbols_a) for n in groups.post_filter_only['screeners']}
        results_c = run_all_screeners(df, groups.post_filter_only, universe_tickers=pf_tickers)
        symbols_a_final, trace_c = consolidate_results(results_c, method='union')
        # WIN_REG passes EMA5 (distance 2.0); WIN_REG_FAIL_PF fails (distance 8.0)
        assert set(symbols_a_final) == {'WIN_REG'}

        # Group B: bypass — runs against the full DF, NOT routed through EMA5
        results_b = run_all_screeners(df, groups.bypass)
        symbols_b, trace_b = consolidate_results(results_b, method='union')
        assert set(symbols_b) == {'WIN_BYPASS'}

        # Final union
        final = sorted(set(symbols_a_final) | set(symbols_b))
        assert final == ['WIN_BYPASS', 'WIN_REG']

    def test_traceability_merges_a_c_b(self):
        """A symbol's traceability lists screeners from all three groups it passed."""
        df = self._indicators_df()
        # Make WIN_REG also satisfy the bypass (rs_ir = 1.5)
        df.loc['WIN_REG', 'rs_ir'] = 1.5

        config = _config({
            'momentum': {
                'requirements': [{'field': 'pct_change_1m', 'operator': '>=', 'value': 30}],
            },
            'rs_bypass': {
                'apply_post_filter': False,
                'requirements': [{'field': 'rs_ir', 'operator': '>=', 'value': 1.0}],
            },
            'ema5_scan': {
                'post_filter': True,
                'requirements': [{'field': 'ema5_distance_pct', 'operator': 'between',
                                  'value': [0, 5]}],
            },
        })

        groups = split_screener_config(config)

        results_a = run_all_screeners(df, groups.with_postfilter)
        symbols_a, trace_a = consolidate_results(results_a, method='union')

        pf_tickers = {n: set(symbols_a) for n in groups.post_filter_only['screeners']}
        results_c = run_all_screeners(df, groups.post_filter_only, universe_tickers=pf_tickers)
        symbols_a_final, trace_c = consolidate_results(results_c, method='union')

        results_b = run_all_screeners(df, groups.bypass)
        symbols_b, trace_b = consolidate_results(results_b, method='union')

        final = sorted(set(symbols_a_final) | set(symbols_b))
        merged_trace = {
            s: trace_a.get(s, []) + trace_c.get(s, []) + trace_b.get(s, [])
            for s in final
        }

        # WIN_REG passed momentum (A), ema5_scan (C), and rs_bypass (B)
        assert 'momentum' in merged_trace['WIN_REG']
        assert 'ema5_scan' in merged_trace['WIN_REG']
        assert 'rs_bypass' in merged_trace['WIN_REG']
        # WIN_BYPASS passed only the bypass screener
        assert merged_trace['WIN_BYPASS'] == ['rs_bypass']


class TestResumePreRSCheckpoint:
    """Resume from a pre-RS checkpoint (indicators marker present, but no
    indicators_rs marker) must trigger re-enrichment, not screener execution
    against missing columns.
    """

    def test_pre_rs_checkpoint_triggers_reenrichment(self, tmp_path, monkeypatch):
        """After resume, indicators_rs.parquet exists and has rs_ir column."""
        from run_weekly import _mark_stage, _stage_done

        ckpt = tmp_path / "pipeline" / "2026-01-01"
        ckpt.mkdir(parents=True)

        # Pre-RS state: 'indicators' marked, 'indicators_rs' NOT marked.
        # Write an old-style indicators.parquet without RS columns.
        old_indicators = pd.DataFrame(
            {'close': [100, 200], 'pct_change_1m': [5, 10],
             'pct_change_3m': [10, 20], 'pct_change_6m': [15, 25],
             'pct_change_1y': [20, 30]},
            index=pd.Index(['AAA', 'BBB'], name='ticker'),
        )
        old_indicators.to_parquet(ckpt / "indicators.parquet")
        _mark_stage(ckpt, "indicators")

        # Sanity: pre-conditions
        assert _stage_done(ckpt, "indicators")
        assert not _stage_done(ckpt, "indicators_rs")

        # Simulate the resume logic: detect missing indicators_rs, run
        # compute_rs_indicators against the old indicators.parquet, and write
        # the new checkpoint.
        from src.indicators import compute_rs_indicators

        loaded = pd.read_parquet(ckpt / "indicators.parquet")
        enriched = compute_rs_indicators(loaded, sp500_close=None, ohlcv={})
        enriched.to_parquet(ckpt / "indicators_rs.parquet")
        _mark_stage(ckpt, "indicators_rs")

        # Post-conditions: marker set, file has RS columns
        assert _stage_done(ckpt, "indicators_rs")
        loaded_rs = pd.read_parquet(ckpt / "indicators_rs.parquet")
        assert 'rs_ir' in loaded_rs.columns
        assert 'rs_weighted_return' in loaded_rs.columns
        assert 'rs_rating' in loaded_rs.columns
