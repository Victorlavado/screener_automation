#!/usr/bin/env python3
"""
Weekly Screener Runner
======================
Main script that orchestrates the complete screening pipeline.

Usage:
    python run_weekly.py [options]

Options:
    --dry-run       Run without saving output files
    --verbose       Show detailed progress
    --screeners     Comma-separated list of screeners to run (default: all)
    --universe      Override universe to use
    --resume        Resume from last checkpoint (same day)
    --no-cache      Ignore all caches and download fresh
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.universe import (
    load_universes_config,
    resolve_universe,
    get_default_universe,
    extract_ticker,
    symbols_to_tickers,
)
from src.data import download_ohlcv_batch, download_fundamentals
from src.indicators import compute_all_indicators
from src.screener import (
    load_screeners_config,
    run_all_screeners,
    consolidate_results,
    build_report_dataframe,
    needs_fundamentals,
    get_prescreen_candidates,
)
from src.export import export_all


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = Path(__file__).parent / ".cache" / "pipeline"


def _get_checkpoint_dir(date_str: str = None) -> Path:
    """Get checkpoint directory for today (or given date)."""
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    d = CHECKPOINT_DIR / date_str
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_checkpoint_meta(ckpt_dir: Path) -> dict:
    meta_path = ckpt_dir / "checkpoint.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            return json.load(f)
    return {}


def _save_checkpoint_meta(ckpt_dir: Path, meta: dict) -> None:
    with open(ckpt_dir / "checkpoint.json", "w") as f:
        json.dump(meta, f, indent=2)


def _mark_stage(ckpt_dir: Path, stage: str) -> None:
    """Mark a stage as completed in checkpoint metadata."""
    meta = _load_checkpoint_meta(ckpt_dir)
    meta[stage] = datetime.now().isoformat()
    _save_checkpoint_meta(ckpt_dir, meta)


def _stage_done(ckpt_dir: Path, stage: str) -> bool:
    return stage in _load_checkpoint_meta(ckpt_dir)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_exchange_mapping(symbols: List[str]) -> Dict[str, str]:
    """Build ticker -> exchange mapping from EXCHANGE:SYMBOL format."""
    mapping = {}
    for symbol in symbols:
        if ':' in symbol:
            exchange, ticker = symbol.split(':', 1)
            mapping[ticker] = exchange
        else:
            mapping[symbol] = 'NYSE'
    return mapping


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    dry_run: bool = False,
    verbose: bool = False,
    screener_filter: List[str] = None,
    universe_override: str = None,
    resume: bool = False,
    no_cache: bool = False,
):
    """
    Execute the complete screening pipeline with checkpoint/resume support.

    Stages and their checkpoint artefacts:
        1. symbols        -> checkpoint.json (symbol list stored inside meta)
        2. ohlcv          -> per-ticker parquet cache (handled by data module)
        3. fundamentals   -> fundamentals.parquet
        4. indicators     -> indicators.parquet
        5. screeners+export -> final output files
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    start_time = datetime.now()
    ckpt_dir = _get_checkpoint_dir()
    cache_age = 0.0 if no_cache else 16.0

    logger.info("=" * 60)
    logger.info("STARTING WEEKLY SCREENER RUN")
    logger.info(f"Date: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    if resume:
        logger.info("Mode: RESUME from checkpoint")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Stage 1: Resolve universes (per-screener + global)
    # ------------------------------------------------------------------
    logger.info("Loading configurations...")
    universes_config = load_universes_config()
    screeners_config = load_screeners_config()

    default_universe = universe_override or get_default_universe(universes_config)
    available_universes = set(universes_config.get('universes', {}).keys())

    # Resolve each screener's universe and collect the union
    screener_defs = screeners_config.get('screeners', {})
    if screener_filter:
        screener_defs = {k: v for k, v in screener_defs.items() if k in screener_filter}

    all_symbols: set = set()
    screener_symbols: dict = {}   # screener_name -> set(EXCHANGE:TICKER)
    screener_tickers: dict = {}   # screener_name -> set(ticker)
    resolved_cache: dict = {}     # universe_name -> set(symbols)  (avoid re-resolving)

    for sname, sdef in screener_defs.items():
        uname = sdef.get('universe', default_universe)
        # Fall back to default if universe doesn't exist
        if uname not in available_universes:
            logger.warning(
                f"Screener '{sname}' references unknown universe '{uname}', "
                f"falling back to '{default_universe}'"
            )
            uname = default_universe

        if uname not in resolved_cache:
            logger.info(f"Resolving universe '{uname}'...")
            resolved_cache[uname] = resolve_universe(uname, universes_config)
            logger.info(f"  '{uname}' -> {len(resolved_cache[uname])} symbols")

        syms = resolved_cache[uname]
        screener_symbols[sname] = syms
        screener_tickers[sname] = set(symbols_to_tickers(syms))
        all_symbols.update(syms)

    symbols = all_symbols
    logger.info(f"Combined universe: {len(symbols)} unique symbols across {len(resolved_cache)} universe(s)")

    if not symbols:
        logger.error("No symbols in universe. Exiting.")
        return

    exchange_mapping = build_exchange_mapping(symbols)
    tickers = symbols_to_tickers(symbols)

    # ------------------------------------------------------------------
    # Stage 2: Download OHLCV (cache handled inside download_ohlcv_batch)
    # ------------------------------------------------------------------
    logger.info("[Stage 2/5] Downloading OHLCV data...")
    ohlcv_data = download_ohlcv_batch(
        tickers,
        period="1y",
        show_progress=verbose,
        cache_max_age_hours=cache_age,
    )
    logger.info(f"OHLCV data available for {len(ohlcv_data)} symbols")

    if not ohlcv_data:
        logger.error("No OHLCV data. Exiting.")
        return

    _mark_stage(ckpt_dir, "ohlcv")

    # ------------------------------------------------------------------
    # Stage 3: Compute technical indicators (no fundamentals yet)
    # ------------------------------------------------------------------
    indicators_path = ckpt_dir / "indicators.parquet"

    if resume and _stage_done(ckpt_dir, "indicators") and indicators_path.exists():
        logger.info("[Stage 3/5] Loading indicators from checkpoint...")
        indicators_df = pd.read_parquet(indicators_path)
        logger.info(f"Loaded indicators for {len(indicators_df)} symbols")
    else:
        logger.info("[Stage 3/5] Computing technical indicators...")
        indicators_df = compute_all_indicators(ohlcv_data)
        logger.info(f"Computed indicators for {len(indicators_df)} symbols")

    # ------------------------------------------------------------------
    # Stage 4: Download fundamentals (only for pre-screened candidates)
    # ------------------------------------------------------------------
    fundamentals_path = ckpt_dir / "fundamentals.parquet"

    if resume and _stage_done(ckpt_dir, "fundamentals") and fundamentals_path.exists():
        logger.info("[Stage 4/5] Loading fundamentals from checkpoint...")
        fundamentals_data = pd.read_parquet(fundamentals_path)
        logger.info(f"Loaded fundamentals for {len(fundamentals_data)} symbols")
    else:
        any_needs_fund = needs_fundamentals(screeners_config)

        if any_needs_fund:
            # Pre-screen: apply technical-only filters to find candidates
            candidates = get_prescreen_candidates(
                indicators_df, screeners_config, screener_tickers
            )
            logger.info(
                f"[Stage 4/5] Pre-screen: {len(candidates)} candidates need fundamentals "
                f"(vs {len(indicators_df)} total)"
            )

            fundamentals_data = download_fundamentals(
                list(candidates),
                show_progress=verbose,
            )
            logger.info(f"Downloaded fundamentals for {len(fundamentals_data)} symbols")
        else:
            logger.info("[Stage 4/5] No screener uses fundamental fields — skipping download")
            fundamentals_data = pd.DataFrame()

        if not fundamentals_data.empty:
            fundamentals_data.to_parquet(fundamentals_path)
        _mark_stage(ckpt_dir, "fundamentals")

    # Merge fundamentals into indicators
    if not fundamentals_data.empty:
        indicators_df = indicators_df.join(fundamentals_data, how='left')

    if not indicators_df.empty:
        indicators_df.to_parquet(indicators_path)
    _mark_stage(ckpt_dir, "indicators")

    # ------------------------------------------------------------------
    # Stage 5: Screen + export
    # ------------------------------------------------------------------
    logger.info("[Stage 5/5] Running screeners...")

    # screener_filter already applied when building screener_defs in Stage 1
    if screener_filter:
        screeners_config['screeners'] = {
            k: v for k, v in screeners_config.get('screeners', {}).items()
            if k in screener_filter
        }
        logger.info(f"Running selected screeners: {list(screeners_config['screeners'].keys())}")

    screener_results = run_all_screeners(
        indicators_df, screeners_config, universe_tickers=screener_tickers
    )

    settings = screeners_config.get('settings', {})
    consolidation_method = settings.get('consolidation', 'union')
    exclude_symbols = settings.get('exclude_symbols', [])

    logger.info(f"Consolidating results (method: {consolidation_method})...")
    final_symbols, traceability = consolidate_results(
        screener_results,
        method=consolidation_method,
        exclude_symbols=exclude_symbols,
    )
    logger.info(f"Final consolidated list: {len(final_symbols)} symbols")

    report_df = build_report_dataframe(
        indicators_df, final_symbols, traceability, exchange_mapping
    )

    if dry_run:
        logger.info("DRY RUN - Skipping file export")
        logger.info(f"Would export {len(final_symbols)} symbols")
        logger.info(f"Top 10: {final_symbols[:10]}")
    else:
        logger.info("Exporting results...")
        output_paths = export_all(
            final_symbols, exchange_mapping, report_df, screener_results
        )
        logger.info("Export complete:")
        for name, path in output_paths.items():
            logger.info(f"  {name}: {path}")

    _mark_stage(ckpt_dir, "export")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    end_time = datetime.now()
    duration = end_time - start_time

    logger.info("")
    logger.info("=" * 60)
    logger.info("SCREENING COMPLETE")
    logger.info(f"Duration: {duration}")
    logger.info(f"Symbols screened: {len(indicators_df)}")
    logger.info(f"Symbols passed: {len(final_symbols)}")
    logger.info(f"Checkpoint: {ckpt_dir}")
    logger.info("=" * 60)

    logger.info("\nPer-screener results:")
    for name, result in screener_results.items():
        logger.info(f"  {name}: {result['count']} symbols")

    return final_symbols, report_df


def main():
    parser = argparse.ArgumentParser(
        description='Weekly stock screener automation'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without saving output files',
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed progress',
    )
    parser.add_argument(
        '--screeners',
        type=str,
        help='Comma-separated list of screeners to run',
    )
    parser.add_argument(
        '--universe',
        type=str,
        help='Override universe to use',
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last checkpoint (same day)',
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Ignore all caches and download everything fresh',
    )

    args = parser.parse_args()

    screener_filter = None
    if args.screeners:
        screener_filter = [s.strip() for s in args.screeners.split(',')]

    try:
        run_pipeline(
            dry_run=args.dry_run,
            verbose=args.verbose,
            screener_filter=screener_filter,
            universe_override=args.universe,
            resume=args.resume,
            no_cache=args.no_cache,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user. Resume with: python run_weekly.py --resume")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        logging.info("Resume with: python run_weekly.py --resume")
        if args.verbose:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
