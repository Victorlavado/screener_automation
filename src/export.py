"""
Export module.
Generates output files for TradingView import and auditing.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd


OUTPUT_DIR = Path(__file__).parent.parent / "output"

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
}


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


def _to_tradingview_symbol(ticker: str, exchange: str) -> str:
    """Convert an internal ticker + exchange to TradingView format.

    Strips Yahoo suffix and maps the exchange prefix.

    Examples:
        ("DG.PA", "EPA")    -> "EURONEXT:DG"
        ("EQNR.OL", "OSL") -> "OSL:EQNR"
        ("AAPL", "NASDAQ")  -> "NASDAQ:AAPL"
    """
    tv_ticker = _strip_yahoo_suffix(ticker)
    tv_exchange = _map_tv_exchange(exchange)
    return f"{tv_exchange}:{tv_ticker}"


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(exist_ok=True)


def get_date_suffix() -> str:
    """Get current date as string suffix."""
    return datetime.now().strftime("%Y-%m-%d")


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

    # Build TradingView format strings
    tv_symbols = []
    for symbol in symbols:
        exchange = symbol_to_exchange.get(symbol, 'NYSE')
        tv_symbols.append(_to_tradingview_symbol(symbol, exchange))

    # Write comma-separated
    content = ','.join(tv_symbols)

    with open(filepath, 'w') as f:
        f.write(content)

    print(f"Exported {len(symbols)} symbols to {filepath}")
    return str(filepath)


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
