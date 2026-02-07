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
"""

import argparse
import logging
import sys
from datetime import datetime
from typing import Dict, List

from src.universe import (
    load_universes_config,
    resolve_universe,
    get_default_universe,
    extract_ticker
)
from src.data import download_all_data
from src.indicators import compute_all_indicators
from src.screener import (
    load_screeners_config,
    run_all_screeners,
    consolidate_results,
    build_report_dataframe
)
from src.export import export_all


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def build_exchange_mapping(symbols: List[str]) -> Dict[str, str]:
    """
    Build ticker -> exchange mapping from EXCHANGE:SYMBOL format.
    """
    mapping = {}
    for symbol in symbols:
        if ':' in symbol:
            exchange, ticker = symbol.split(':', 1)
            mapping[ticker] = exchange
        else:
            mapping[symbol] = 'NYSE'  # Default
    return mapping


def run_pipeline(
    dry_run: bool = False,
    verbose: bool = False,
    screener_filter: List[str] = None,
    universe_override: str = None
):
    """
    Execute the complete screening pipeline.

    Args:
        dry_run: If True, don't save output files
        verbose: If True, show detailed progress
        screener_filter: List of screener names to run (None = all)
        universe_override: Override universe name
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("STARTING WEEKLY SCREENER RUN")
    logger.info(f"Date: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Load configurations
    logger.info("Loading configurations...")
    universes_config = load_universes_config()
    screeners_config = load_screeners_config()

    # Determine universe
    universe_name = universe_override or get_default_universe(universes_config)
    logger.info(f"Using universe: {universe_name}")

    # Resolve universe to symbols
    logger.info("Resolving universe symbols...")
    symbols = resolve_universe(universe_name, universes_config)
    logger.info(f"Universe contains {len(symbols)} symbols")

    if not symbols:
        logger.error("No symbols in universe. Exiting.")
        return

    # Build exchange mapping
    exchange_mapping = build_exchange_mapping(symbols)
    tickers = list(exchange_mapping.keys())

    # Download data
    logger.info("Downloading market data...")
    ohlcv_data, fundamentals_data = download_all_data(
        symbols,
        period="1y",
        include_fundamentals=True,
        show_progress=verbose
    )
    logger.info(f"Downloaded data for {len(ohlcv_data)} symbols")

    if not ohlcv_data:
        logger.error("No data downloaded. Exiting.")
        return

    # Compute indicators
    logger.info("Computing technical indicators...")
    indicators_df = compute_all_indicators(ohlcv_data, fundamentals_data)
    logger.info(f"Computed indicators for {len(indicators_df)} symbols")

    # Filter screeners if specified
    if screener_filter:
        filtered_screeners = {
            k: v for k, v in screeners_config.get('screeners', {}).items()
            if k in screener_filter
        }
        if not filtered_screeners:
            logger.error(f"No matching screeners found for: {screener_filter}")
            return
        screeners_config['screeners'] = filtered_screeners
        logger.info(f"Running selected screeners: {list(filtered_screeners.keys())}")

    # Run screeners
    logger.info("Running screeners...")
    screener_results = run_all_screeners(indicators_df, screeners_config)

    # Consolidate results
    settings = screeners_config.get('settings', {})
    consolidation_method = settings.get('consolidation', 'union')
    exclude_symbols = settings.get('exclude_symbols', [])

    logger.info(f"Consolidating results (method: {consolidation_method})...")
    final_symbols, traceability = consolidate_results(
        screener_results,
        method=consolidation_method,
        exclude_symbols=exclude_symbols
    )
    logger.info(f"Final consolidated list: {len(final_symbols)} symbols")

    # Build report DataFrame
    report_df = build_report_dataframe(
        indicators_df,
        final_symbols,
        traceability,
        exchange_mapping
    )

    # Export results
    if dry_run:
        logger.info("DRY RUN - Skipping file export")
        logger.info(f"Would export {len(final_symbols)} symbols")
        logger.info(f"Top 10: {final_symbols[:10]}")
    else:
        logger.info("Exporting results...")
        output_paths = export_all(
            final_symbols,
            exchange_mapping,
            report_df,
            screener_results
        )
        logger.info("Export complete:")
        for name, path in output_paths.items():
            logger.info(f"  {name}: {path}")

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time

    logger.info("")
    logger.info("=" * 60)
    logger.info("SCREENING COMPLETE")
    logger.info(f"Duration: {duration}")
    logger.info(f"Symbols screened: {len(indicators_df)}")
    logger.info(f"Symbols passed: {len(final_symbols)}")
    logger.info("=" * 60)

    # Print summary per screener
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
        help='Run without saving output files'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed progress'
    )
    parser.add_argument(
        '--screeners',
        type=str,
        help='Comma-separated list of screeners to run'
    )
    parser.add_argument(
        '--universe',
        type=str,
        help='Override universe to use'
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
            universe_override=args.universe
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        if args.verbose:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
