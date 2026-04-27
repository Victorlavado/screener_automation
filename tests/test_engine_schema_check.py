"""Tests for the engine-startup schema check (_validate_required_columns).

These guard against the silent zero-symbol-result class of bug — e.g. when
YAML references a column that hasn't been computed (stale checkpoint, schema
drift) the engine should fail fast with a clear error.
"""

import pandas as pd
import pytest

from src.screener import (
    _validate_required_columns,
    run_all_screeners,
)


class TestValidateRequiredColumns:
    """Direct unit tests for the helper."""

    def test_passes_when_all_columns_present(self):
        df = pd.DataFrame({'close': [100], 'sma_50': [95]}, index=['AAA'])
        config = {
            'screeners': {
                'a': {'requirements': [
                    {'field': 'close', 'operator': '>', 'reference': 'sma_50'},
                ]},
            }
        }
        # Should not raise
        _validate_required_columns(df, config)

    def test_raises_on_missing_field(self):
        df = pd.DataFrame({'close': [100]}, index=['AAA'])
        config = {
            'screeners': {
                'a': {'requirements': [
                    {'field': 'rs_ir', 'operator': '>=', 'value': 1.0},
                ]},
            }
        }
        with pytest.raises(ValueError, match='rs_ir'):
            _validate_required_columns(df, config)

    def test_raises_on_missing_reference(self):
        df = pd.DataFrame({'close': [100]}, index=['AAA'])
        config = {
            'screeners': {
                'a': {'requirements': [
                    {'field': 'close', 'operator': '>', 'reference': 'sma_999'},
                ]},
            }
        }
        with pytest.raises(ValueError, match='sma_999'):
            _validate_required_columns(df, config)

    def test_raises_on_missing_sort_by(self):
        df = pd.DataFrame({'close': [100]}, index=['AAA'])
        config = {
            'screeners': {
                'a': {
                    'requirements': [{'field': 'close', 'operator': '>', 'value': 0}],
                    'postprocess': {'sort_by': 'unknown_metric', 'sort_order': 'desc'},
                },
            }
        }
        with pytest.raises(ValueError, match='unknown_metric'):
            _validate_required_columns(df, config)

    def test_lists_all_missing_columns(self):
        df = pd.DataFrame({'close': [100]}, index=['AAA'])
        config = {
            'screeners': {
                'a': {'requirements': [
                    {'field': 'rs_ir', 'operator': '>=', 'value': 1.0},
                    {'field': 'rs_rating', 'operator': '>=', 'value': 80},
                ]},
            }
        }
        with pytest.raises(ValueError) as exc_info:
            _validate_required_columns(df, config)

        msg = str(exc_info.value)
        assert 'rs_ir' in msg and 'rs_rating' in msg

    def test_empty_config_does_not_raise(self):
        df = pd.DataFrame({'close': [100]}, index=['AAA'])
        _validate_required_columns(df, {'screeners': {}})


class TestRunAllScreenersFailsFast:
    """Schema check fires at the start of run_all_screeners."""

    def test_raises_before_any_screener_runs(self):
        df = pd.DataFrame({'close': [100], 'volume': [1000]}, index=['AAA'])
        config = {
            'screeners': {
                'rs': {
                    'requirements': [{'field': 'rs_ir', 'operator': '>=', 'value': 1.0}],
                },
            }
        }
        with pytest.raises(ValueError, match='rs_ir'):
            run_all_screeners(df, config)
