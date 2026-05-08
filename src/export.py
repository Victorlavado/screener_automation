"""
Export module.
Generates output files for TradingView import and auditing.
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd


OUTPUT_DIR = Path(__file__).parent.parent / "output"

# TradingView refuses to import watchlists larger than this many symbols.
TV_WATCHLIST_LIMIT = 1000

# Yahoo Finance exchange suffixes that must be stripped for TradingView.
# TradingView identifies the exchange via the prefix (e.g. EURONEXT:DG),
# so the Yahoo suffix (e.g. .PA) is redundant and causes import failures.
YAHOO_SUFFIXES = frozenset({
    '.PA',  # Euronext Paris
    '.AS',  # Euronext Amsterdam
    '.BR',  # Euronext Brussels
    '.OL',  # Oslo Børs
    '.DE',  # XETRA (Germany)
    '.L',   # London Stock Exchange
    '.MI',  # Borsa Italiana (Milan)
    '.MC',  # Bolsa de Madrid
    '.ST',  # OMX Stockholm
    '.CO',  # OMX Copenhagen
    '.AX',  # ASX (Australia)
    '.T',   # TSE (Tokyo)
    '.HK',  # HKEX (Hong Kong)
    '.KS',  # KRX (Korea)
    '.SS',  # SSE (Shanghai)
    '.SZ',  # SZSE (Shenzhen)
})

# Internal exchange prefix -> TradingView exchange prefix.
# Only entries that differ are listed; unlisted codes pass through as-is.
_TV_EXCHANGE_MAP = {
    'EPA': 'EURONEXT',   # Euronext Paris
    'AMS': 'EURONEXT',   # Euronext Amsterdam
    'EBR': 'EURONEXT',   # Euronext Brussels
    'STO': 'OMXSTO',     # OMX Stockholm
    'CPH': 'OMXCOP',     # OMX Copenhagen
    'ARCA': 'AMEX',      # NYSE Arca: TradingView lists Arca-traded ETFs under AMEX
    'BATS': 'AMEX',      # Cboe BZX (ex-BATS): legacy symbols resolve under AMEX in TradingView
}

# US exchanges where dual-class shares use a dot (BRK.B). Outside of these
# (e.g. Copenhagen NOVO-B), the hyphen is the canonical TradingView form.
_US_EXCHANGES = frozenset({'NYSE', 'NASDAQ', 'AMEX', 'ARCA', 'BATS'})
_DUAL_CLASS_RE = re.compile(r'^([A-Z]+)-([A-Z])$')


def _strip_yahoo_suffix(ticker: str) -> str:
    """Strip Yahoo Finance exchange suffix from a ticker symbol.

    Examples:
        DG.PA   -> DG
        EQNR.OL -> EQNR
        AAPL    -> AAPL  (no suffix, unchanged)
    """
    if not ticker:
        return ticker
    dot_pos = ticker.rfind('.')
    if dot_pos > 0:
        suffix = ticker[dot_pos:]
        if suffix in YAHOO_SUFFIXES:
            return ticker[:dot_pos]
    return ticker


def _map_tv_exchange(exchange: str) -> str:
    """Map an internal exchange code to its TradingView equivalent."""
    return _TV_EXCHANGE_MAP.get(exchange, exchange)


def _restore_us_dual_class(ticker: str, tv_exchange: str) -> str:
    """Restore dot notation for US dual-class shares (BRK-B → BRK.B).

    Yahoo Finance uses hyphens (BRK-B) for dual-class shares; TradingView
    uses dots (BRK.B). Only applies to US exchanges — European tickers like
    OMXCOP:NOVO-B keep the hyphen.
    """
    if tv_exchange in _US_EXCHANGES:
        m = _DUAL_CLASS_RE.match(ticker)
        if m:
            return f"{m.group(1)}.{m.group(2)}"
    return ticker


def _to_tradingview_symbol(ticker: str, exchange: str) -> str:
    """Convert an internal ticker + exchange to TradingView format.

    Strips Yahoo suffix, maps the exchange prefix, and restores dot
    notation for US dual-class shares.

    Examples:
        ("DG.PA", "EPA")     -> "EURONEXT:DG"
        ("EQNR.OL", "OSL")   -> "OSL:EQNR"
        ("AAPL", "NASDAQ")   -> "NASDAQ:AAPL"
        ("BRK-B", "NYSE")    -> "NYSE:BRK.B"
        ("NOVO-B.CO", "CPH") -> "OMXCOP:NOVO-B"
    """
    tv_ticker = _strip_yahoo_suffix(ticker)
    tv_exchange = _map_tv_exchange(exchange)
    tv_ticker = _restore_us_dual_class(tv_ticker, tv_exchange)
    return f"{tv_exchange}:{tv_ticker}"


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(exist_ok=True)


def get_date_suffix() -> str:
    """Get current date as string suffix."""
    return datetime.now().strftime("%Y-%m-%d")


def _write_tv_watchlist_file(
    filepath: Path,
    symbols: List[str],
    symbol_to_exchange: Dict[str, str],
) -> None:
    """Write a TradingView-formatted watchlist file (comma-separated)."""
    tv_symbols = [
        _to_tradingview_symbol(s, symbol_to_exchange.get(s, 'NYSE'))
        for s in symbols
    ]
    with open(filepath, 'w') as f:
        f.write(','.join(tv_symbols))


def export_tradingview_watchlist(
    symbols: List[str],
    symbol_to_exchange: Dict[str, str],
    filename: str = None
) -> str:
    """
    Export symbols as TradingView-compatible watchlist.

    TradingView format: EXCHANGE:SYMBOL,EXCHANGE:SYMBOL,...

    Args:
        symbols: List of ticker symbols
        symbol_to_exchange: Mapping of ticker -> exchange
        filename: Output filename (default: candidates_YYYY-MM-DD.txt)

    Returns:
        Path to created file
    """
    ensure_output_dir()

    if filename is None:
        filename = f"candidates_{get_date_suffix()}.txt"

    filepath = OUTPUT_DIR / filename
    _write_tv_watchlist_file(filepath, symbols, symbol_to_exchange)

    print(f"Exported {len(symbols)} symbols to {filepath}")
    return str(filepath)


def export_tradingview_watchlists_per_screener(
    screener_results: Dict[str, Dict],
    symbol_to_exchange: Dict[str, str],
    date_suffix: str = None,
) -> List[str]:
    """
    Export one TradingView watchlist per screener.

    Used when the consolidated list exceeds TV_WATCHLIST_LIMIT and a single
    file can no longer be imported into TradingView. Screeners whose hit
    list itself exceeds the limit are chunked into _part1, _part2, ...

    Args:
        screener_results: { screener_name: { 'symbols': [...], ... }, ... }
        symbol_to_exchange: Mapping of ticker -> exchange
        date_suffix: Date string for filenames (default: today)

    Returns:
        List of paths to the files created.
    """
    ensure_output_dir()

    if date_suffix is None:
        date_suffix = get_date_suffix()

    paths: List[str] = []

    for screener_name, result in screener_results.items():
        symbols = result.get('symbols', []) or []
        if not symbols:
            continue

        if len(symbols) <= TV_WATCHLIST_LIMIT:
            filepath = OUTPUT_DIR / f"candidates_{date_suffix}_{screener_name}.txt"
            _write_tv_watchlist_file(filepath, symbols, symbol_to_exchange)
            print(f"Exported {len(symbols)} symbols ({screener_name}) to {filepath}")
            paths.append(str(filepath))
            continue

        for part_idx, start in enumerate(
            range(0, len(symbols), TV_WATCHLIST_LIMIT), start=1
        ):
            chunk = symbols[start:start + TV_WATCHLIST_LIMIT]
            filepath = OUTPUT_DIR / f"candidates_{date_suffix}_{screener_name}_part{part_idx}.txt"
            _write_tv_watchlist_file(filepath, chunk, symbol_to_exchange)
            print(f"Exported {len(chunk)} symbols ({screener_name} part {part_idx}) to {filepath}")
            paths.append(str(filepath))

    return paths


def export_report_csv(
    report_df: pd.DataFrame,
    filename: str = None
) -> str:
    """
    Export detailed report as CSV.

    Args:
        report_df: DataFrame with screening results and indicators
        filename: Output filename (default: report_YYYY-MM-DD.csv)

    Returns:
        Path to created file
    """
    ensure_output_dir()

    if filename is None:
        filename = f"report_{get_date_suffix()}.csv"

    filepath = OUTPUT_DIR / filename

    report_df.to_csv(filepath, index=False)

    print(f"Exported report with {len(report_df)} rows to {filepath}")
    return str(filepath)


def export_summary(
    screener_results: Dict[str, Dict],
    final_symbols: List[str],
    filename: str = None
) -> str:
    """
    Export a human-readable summary of screening results.

    Args:
        screener_results: Results from each screener
        final_symbols: Final consolidated symbol list
        filename: Output filename (default: summary_YYYY-MM-DD.txt)

    Returns:
        Path to created file
    """
    ensure_output_dir()

    if filename is None:
        filename = f"summary_{get_date_suffix()}.txt"

    filepath = OUTPUT_DIR / filename

    lines = [
        "=" * 60,
        f"SCREENER RESULTS - {get_date_suffix()}",
        "=" * 60,
        "",
    ]

    # Per-screener summary
    for name, result in screener_results.items():
        lines.append(f"Screener: {name}")
        lines.append(f"  Description: {result.get('description', 'N/A')}")
        lines.append(f"  Symbols found: {result['count']}")
        if result['count'] <= 20:
            lines.append(f"  List: {', '.join(result['symbols'])}")
        lines.append("")

    # Final summary
    lines.append("-" * 60)
    lines.append(f"FINAL CONSOLIDATED LIST: {len(final_symbols)} symbols")
    lines.append("-" * 60)

    # Show symbols in groups of 10
    for i in range(0, len(final_symbols), 10):
        chunk = final_symbols[i:i+10]
        lines.append(', '.join(chunk))

    lines.append("")
    lines.append("=" * 60)

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Exported summary to {filepath}")
    return str(filepath)


def export_all(
    final_symbols: List[str],
    symbol_to_exchange: Dict[str, str],
    report_df: pd.DataFrame,
    screener_results: Dict[str, Dict]
) -> Dict[str, str]:
    """
    Export all output files.

    Args:
        final_symbols: Final list of symbols
        symbol_to_exchange: Ticker -> exchange mapping
        report_df: Detailed report DataFrame
        screener_results: Results from each screener

    Returns:
        Dict with paths to all created files
    """
    date_suffix = get_date_suffix()

    paths = {
        'watchlist': export_tradingview_watchlist(
            final_symbols,
            symbol_to_exchange,
            f"candidates_{date_suffix}.txt"
        ),
        'report': export_report_csv(
            report_df,
            f"report_{date_suffix}.csv"
        ),
        'summary': export_summary(
            screener_results,
            final_symbols,
            f"summary_{date_suffix}.txt"
        )
    }

    if len(final_symbols) > TV_WATCHLIST_LIMIT:
        paths['watchlists_per_screener'] = export_tradingview_watchlists_per_screener(
            screener_results,
            symbol_to_exchange,
            date_suffix=date_suffix,
        )

    return paths


if __name__ == "__main__":
    # Test export
    print("Testing export module...")

    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    test_exchange_map = {
        'AAPL': 'NASDAQ',
        'MSFT': 'NASDAQ',
        'GOOGL': 'NASDAQ',
        'AMZN': 'NASDAQ',
        'META': 'NASDAQ',
    }

    test_report = pd.DataFrame({
        'ticker': test_symbols,
        'exchange': ['NASDAQ'] * 5,
        'close': [150, 350, 140, 180, 500],
        'screeners_passed': ['momentum,value', 'momentum', 'value', 'momentum,value', 'momentum']
    })

    test_results = {
        'momentum': {'description': 'Test momentum', 'count': 4, 'symbols': ['AAPL', 'MSFT', 'AMZN', 'META']},
        'value': {'description': 'Test value', 'count': 3, 'symbols': ['AAPL', 'GOOGL', 'AMZN']}
    }

    paths = export_all(test_symbols, test_exchange_map, test_report, test_results)
    print(f"\nCreated files: {paths}")
