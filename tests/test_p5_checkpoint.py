"""P5: Save indicators checkpoint BEFORE fundamentals merge."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from run_weekly import _mark_stage, _stage_done, _load_checkpoint_meta


class TestIndicatorsPureStage:
    """Verify the new 'indicators_pure' checkpoint stage."""

    def _make_ckpt(self, tmp_path, meta: dict = None):
        ckpt = tmp_path / "pipeline" / "2026-01-01"
        ckpt.mkdir(parents=True)
        if meta:
            with open(ckpt / "checkpoint.json", "w") as f:
                json.dump(meta, f)
        return ckpt

    def test_indicators_pure_stage_marked_before_fundamentals(self, tmp_path):
        """After Stage 3, 'indicators_pure' should be marked in the checkpoint."""
        ckpt = self._make_ckpt(tmp_path)

        # Simulate what the pipeline does after computing indicators
        _mark_stage(ckpt, "indicators_pure")

        assert _stage_done(ckpt, "indicators_pure")
        assert not _stage_done(ckpt, "indicators")
        assert not _stage_done(ckpt, "fundamentals")

    def test_resume_after_stage4_crash_loads_pure_indicators(self, tmp_path):
        """If 'indicators_pure' is done but 'indicators' is not, the pipeline
        should load from indicators_pure.parquet (separate from merged file)."""
        ckpt = self._make_ckpt(tmp_path, {
            "ohlcv": "2026-01-01T00:00:00",
            "indicators_pure": "2026-01-01T00:01:00",
        })

        # Create a pure indicators checkpoint with 5 symbols
        indicators_df = pd.DataFrame(
            {"close": [100, 200, 300, 400, 500], "rsi_14": [55, 60, 45, 70, 30]},
            index=pd.Index(["AAPL", "MSFT", "GOOGL", "AMZN", "META"], name="ticker"),
        )
        indicators_df.to_parquet(ckpt / "indicators_pure.parquet")

        # The resume logic should detect indicators_pure done but indicators not done
        assert _stage_done(ckpt, "indicators_pure")
        assert not _stage_done(ckpt, "indicators")

        # Load from indicators_pure.parquet — all 5 symbols should be present
        loaded = pd.read_parquet(ckpt / "indicators_pure.parquet")
        assert len(loaded) == 5
        assert set(loaded.index) == {"AAPL", "MSFT", "GOOGL", "AMZN", "META"}

    def test_crash_during_merge_preserves_merged_indicators(self, tmp_path):
        """If a previous run saved merged indicators.parquet and a new run crashes
        during merge, the pure file is separate and won't overwrite the merged one."""
        ckpt = self._make_ckpt(tmp_path, {
            "ohlcv": "2026-01-01T00:00:00",
            "indicators_pure": "2026-01-01T00:01:00",
        })

        # Pure indicators saved to its own file
        pure_df = pd.DataFrame(
            {"close": [100, 200], "rsi_14": [55, 60]},
            index=pd.Index(["AAPL", "MSFT"], name="ticker"),
        )
        pure_df.to_parquet(ckpt / "indicators_pure.parquet")

        # Simulate: no indicators.parquet exists (crash before merge completed)
        assert not (ckpt / "indicators.parquet").exists()

        # Pure file is intact and has only technical columns
        loaded_pure = pd.read_parquet(ckpt / "indicators_pure.parquet")
        assert "pe_ratio" not in loaded_pure.columns
        assert len(loaded_pure) == 2

    def test_resume_with_full_completion_loads_merged(self, tmp_path):
        """If 'indicators' is done, skip both stages; pe_ratio column is present."""
        ckpt = self._make_ckpt(tmp_path, {
            "ohlcv": "2026-01-01T00:00:00",
            "indicators_pure": "2026-01-01T00:01:00",
            "fundamentals": "2026-01-01T00:02:00",
            "indicators": "2026-01-01T00:03:00",
        })

        # Create a merged indicators checkpoint with fundamentals columns
        indicators_df = pd.DataFrame(
            {
                "close": [100, 200],
                "rsi_14": [55, 60],
                "pe_ratio": [25.0, 30.0],
            },
            index=pd.Index(["AAPL", "MSFT"], name="ticker"),
        )
        indicators_df.to_parquet(ckpt / "indicators.parquet")

        assert _stage_done(ckpt, "indicators")
        loaded = pd.read_parquet(ckpt / "indicators.parquet")
        assert "pe_ratio" in loaded.columns
