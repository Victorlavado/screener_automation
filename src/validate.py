"""
Validation module.
Compare script results against TradingView export to identify divergences.
"""

import argparse
from pathlib import Path
from typing import Set, Tuple, Dict, List
import pandas as pd


def load_tradingview_export(csv_path: str) -> Set[str]:
    """
    Load symbols from TradingView screener CSV export.

    TradingView exports include a 'Ticker' or 'Symbol' column.
    """
    df = pd.read_csv(csv_path)

    # Try common column names
    symbol_col = None
    for col in ['Ticker', 'Symbol', 'ticker', 'symbol']:
        if col in df.columns:
            symbol_col = col
            break

    if symbol_col is None:
        raise ValueError(f"Could not find symbol column in {csv_path}. Columns: {df.columns.tolist()}")

    symbols = set(df[symbol_col].str.strip().tolist())
    return symbols


def load_script_output(txt_path: str) -> Set[str]:
    """
    Load symbols from script's TradingView-format output.

    Format: EXCHANGE:SYMBOL,EXCHANGE:SYMBOL,...
    """
    with open(txt_path, 'r') as f:
        content = f.read().strip()

    if not content:
        return set()

    # Parse EXCHANGE:SYMBOL format, extract just ticker
    symbols = set()
    for item in content.split(','):
        item = item.strip()
        if ':' in item:
            ticker = item.split(':')[1]
        else:
            ticker = item
        symbols.add(ticker)

    return symbols


def compare_results(
    tv_symbols: Set[str],
    script_symbols: Set[str]
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Compare TradingView results vs script results.

    Returns:
        Tuple of (in_both, only_in_tv, only_in_script)
    """
    in_both = tv_symbols & script_symbols
    only_in_tv = tv_symbols - script_symbols
    only_in_script = script_symbols - tv_symbols

    return in_both, only_in_tv, only_in_script


def generate_divergence_report(
    tv_symbols: Set[str],
    script_symbols: Set[str],
    output_path: str = None
) -> str:
    """
    Generate a human-readable divergence report.

    Args:
        tv_symbols: Symbols from TradingView export
        script_symbols: Symbols from script output
        output_path: Optional path to save report

    Returns:
        Report as string
    """
    in_both, only_tv, only_script = compare_results(tv_symbols, script_symbols)

    total_tv = len(tv_symbols)
    total_script = len(script_symbols)
    match_rate = len(in_both) / max(total_tv, 1) * 100

    lines = [
        "=" * 60,
        "VALIDATION REPORT: Script vs TradingView",
        "=" * 60,
        "",
        f"TradingView symbols:  {total_tv}",
        f"Script symbols:       {total_script}",
        f"Matching symbols:     {len(in_both)}",
        f"Match rate:           {match_rate:.1f}%",
        "",
        "-" * 60,
    ]

    if only_tv:
        lines.append(f"\nIN TRADINGVIEW BUT NOT IN SCRIPT ({len(only_tv)}):")
        lines.append("-" * 40)
        for sym in sorted(only_tv):
            lines.append(f"  - {sym}")
        lines.append("")

    if only_script:
        lines.append(f"\nIN SCRIPT BUT NOT IN TRADINGVIEW ({len(only_script)}):")
        lines.append("-" * 40)
        for sym in sorted(only_script):
            lines.append(f"  + {sym}")
        lines.append("")

    # Possible reasons for divergence
    lines.append("-" * 60)
    lines.append("COMMON REASONS FOR DIVERGENCE:")
    lines.append("-" * 60)
    lines.append("1. Data timing: TV uses real-time, script uses EOD data")
    lines.append("2. Indicator calculation: Slight differences in MA/RSI smoothing")
    lines.append("3. Market session: TV may use extended hours data")
    lines.append("4. Symbol coverage: Different universes or delisted symbols")
    lines.append("5. Fundamental data: Different data providers")
    lines.append("")
    lines.append("=" * 60)

    report = '\n'.join(lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Report saved to: {output_path}")

    return report


def validate(
    tradingview_csv: str,
    script_output: str,
    save_report: bool = True
) -> Dict:
    """
    Main validation function.

    Args:
        tradingview_csv: Path to TradingView screener export CSV
        script_output: Path to script's candidates.txt output
        save_report: Whether to save the report

    Returns:
        Dict with validation results
    """
    print(f"Loading TradingView export: {tradingview_csv}")
    tv_symbols = load_tradingview_export(tradingview_csv)
    print(f"  Found {len(tv_symbols)} symbols")

    print(f"Loading script output: {script_output}")
    script_symbols = load_script_output(script_output)
    print(f"  Found {len(script_symbols)} symbols")

    in_both, only_tv, only_script = compare_results(tv_symbols, script_symbols)

    # Generate report
    report_path = None
    if save_report:
        report_path = Path(script_output).parent / "validation_report.txt"

    report = generate_divergence_report(tv_symbols, script_symbols, report_path)
    print(report)

    return {
        'tv_count': len(tv_symbols),
        'script_count': len(script_symbols),
        'matching': len(in_both),
        'only_tv': list(only_tv),
        'only_script': list(only_script),
        'match_rate': len(in_both) / max(len(tv_symbols), 1) * 100
    }


def main():
    parser = argparse.ArgumentParser(
        description='Validate script results against TradingView export'
    )
    parser.add_argument(
        'tradingview_csv',
        help='Path to TradingView screener export CSV'
    )
    parser.add_argument(
        'script_output',
        help='Path to script candidates.txt output'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save validation report'
    )

    args = parser.parse_args()

    validate(
        args.tradingview_csv,
        args.script_output,
        save_report=not args.no_save
    )


if __name__ == "__main__":
    main()
