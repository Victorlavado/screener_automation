"""
Universe resolution module.
Handles fetching and consolidating symbol lists from various sources.
"""

import os
import json
import time
from pathlib import Path
from typing import List, Set, Dict, Any
from io import StringIO
import pandas as pd
import requests
import yaml


# Cache directory for symbol lists
CACHE_DIR = Path(__file__).parent.parent / ".cache"

# User agent for web requests
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def load_universes_config(config_path: str = "config/universes.yaml") -> Dict[str, Any]:
    """Load universes configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_cache_path(source: str) -> Path:
    """Get cache file path for a given source."""
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{source}_symbols.json"


def is_cache_valid(cache_path: Path, max_age_hours: int = 24) -> bool:
    """Check if cache file exists and is not too old."""
    if not cache_path.exists():
        return False
    age_seconds = time.time() - cache_path.stat().st_mtime
    return age_seconds < (max_age_hours * 3600)


def fetch_sp500_symbols() -> List[str]:
    """Fetch S&P 500 symbols from Wikipedia."""
    cache_path = get_cache_path("sp500")

    if is_cache_valid(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)

    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        tables = pd.read_html(StringIO(response.text))
        df = tables[0]

        symbols = []
        for _, row in df.iterrows():
            symbol = row['Symbol'].replace('.', '-')  # BRK.B -> BRK-B for yfinance
            exchange = "NYSE"
            symbols.append(f"{exchange}:{symbol}")

        with open(cache_path, 'w') as f:
            json.dump(symbols, f)

        return symbols
    except Exception as e:
        print(f"Error fetching S&P 500: {e}")
        return []


def fetch_nasdaq100_symbols() -> List[str]:
    """Fetch NASDAQ 100 symbols from Wikipedia."""
    cache_path = get_cache_path("nasdaq100")

    if is_cache_valid(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)

    try:
        url = "https://en.wikipedia.org/wiki/NASDAQ-100"
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        tables = pd.read_html(StringIO(response.text))
        df = None
        for table in tables:
            if 'Ticker' in table.columns or 'Symbol' in table.columns:
                df = table
                break

        if df is None:
            print("Could not find NASDAQ-100 symbols table")
            return []

        symbol_col = 'Ticker' if 'Ticker' in df.columns else 'Symbol'
        symbols = [f"NASDAQ:{sym}" for sym in df[symbol_col].tolist()]

        with open(cache_path, 'w') as f:
            json.dump(symbols, f)

        return symbols
    except Exception as e:
        print(f"Error fetching NASDAQ-100: {e}")
        return []


def fetch_nasdaq_listed() -> List[str]:
    """Fetch all NASDAQ listed symbols from NASDAQ FTP."""
    cache_path = get_cache_path("nasdaq_listed")

    if is_cache_valid(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)

    try:
        url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        lines = response.text.strip().split('\n')
        symbols = []

        for line in lines[1:]:  # Skip header
            if '|' in line:
                parts = line.split('|')
                symbol = parts[0].strip()
                # Skip test symbols and file creation timestamp
                if symbol and not symbol.startswith('File') and len(symbol) <= 5:
                    # Filter out symbols with special characters
                    if symbol.isalpha() or (symbol.replace('-', '').isalpha()):
                        symbols.append(f"NASDAQ:{symbol}")

        with open(cache_path, 'w') as f:
            json.dump(symbols, f)

        print(f"Fetched {len(symbols)} NASDAQ listed symbols")
        return symbols
    except Exception as e:
        print(f"Error fetching NASDAQ listed: {e}")
        return []


def fetch_nyse_listed() -> List[str]:
    """Fetch NYSE/AMEX listed symbols from NASDAQ FTP (otherlisted.txt)."""
    cache_path = get_cache_path("nyse_listed")

    if is_cache_valid(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)

    try:
        url = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        lines = response.text.strip().split('\n')
        symbols = []

        for line in lines[1:]:  # Skip header
            if '|' in line:
                parts = line.split('|')
                symbol = parts[0].strip()
                exchange_code = parts[2].strip() if len(parts) > 2 else 'N'

                # Map exchange codes
                exchange_map = {'N': 'NYSE', 'A': 'AMEX', 'P': 'ARCA', 'Z': 'BATS'}
                exchange = exchange_map.get(exchange_code, 'NYSE')

                if symbol and not symbol.startswith('File') and len(symbol) <= 5:
                    if symbol.isalpha() or (symbol.replace('-', '').replace('.', '').isalpha()):
                        symbol = symbol.replace('.', '-')  # For yfinance compatibility
                        symbols.append(f"{exchange}:{symbol}")

        with open(cache_path, 'w') as f:
            json.dump(symbols, f)

        print(f"Fetched {len(symbols)} NYSE/AMEX listed symbols")
        return symbols
    except Exception as e:
        print(f"Error fetching NYSE listed: {e}")
        return []


def fetch_russell3000_symbols() -> List[str]:
    """Fetch Russell 3000 symbols (approximation via iShares ETF holdings)."""
    cache_path = get_cache_path("russell3000")

    if is_cache_valid(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)

    try:
        # Use iShares Russell 3000 ETF (IWV) holdings as proxy
        url = "https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund"
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=60)

        if response.status_code != 200:
            print(f"Could not fetch Russell 3000 holdings (status {response.status_code})")
            # Fallback: use combined NASDAQ + NYSE as approximation
            return []

        # Parse CSV (skip metadata rows)
        lines = response.text.split('\n')
        data_start = 0
        for i, line in enumerate(lines):
            if 'Ticker' in line or 'ticker' in line.lower():
                data_start = i
                break

        if data_start == 0:
            return []

        df = pd.read_csv(StringIO('\n'.join(lines[data_start:])))

        ticker_col = None
        for col in df.columns:
            if 'ticker' in col.lower():
                ticker_col = col
                break

        if ticker_col is None:
            return []

        symbols = []
        for ticker in df[ticker_col].dropna():
            ticker = str(ticker).strip()
            if ticker and ticker != '-' and len(ticker) <= 5:
                symbols.append(f"NYSE:{ticker}")  # Default exchange

        with open(cache_path, 'w') as f:
            json.dump(symbols, f)

        print(f"Fetched {len(symbols)} Russell 3000 symbols")
        return symbols
    except Exception as e:
        print(f"Error fetching Russell 3000: {e}")
        return []


def fetch_ftse100_symbols() -> List[str]:
    """Fetch FTSE 100 (UK) symbols from Wikipedia."""
    cache_path = get_cache_path("ftse100")

    if is_cache_valid(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)

    try:
        url = "https://en.wikipedia.org/wiki/FTSE_100_Index"
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        tables = pd.read_html(StringIO(response.text))
        df = None
        for table in tables:
            cols_lower = [str(c).lower() for c in table.columns]
            if any('ticker' in c or 'epic' in c for c in cols_lower):
                df = table
                break

        if df is None:
            return []

        # Find ticker column
        ticker_col = None
        for col in df.columns:
            if 'ticker' in str(col).lower() or 'epic' in str(col).lower():
                ticker_col = col
                break

        if ticker_col is None:
            return []

        symbols = []
        for sym in df[ticker_col].tolist():
            if pd.notna(sym):
                sym = str(sym).strip()
                if '.' in sym:
                    symbols.append(f"LSE:{sym}")
                else:
                    symbols.append(f"LSE:{sym}.L")

        with open(cache_path, 'w') as f:
            json.dump(symbols, f)

        print(f"Fetched {len(symbols)} FTSE 100 symbols")
        return symbols
    except Exception as e:
        print(f"Error fetching FTSE 100: {e}")
        return []


def fetch_dax_symbols() -> List[str]:
    """Fetch DAX (Germany) symbols from Wikipedia."""
    cache_path = get_cache_path("dax")

    if is_cache_valid(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)

    try:
        url = "https://en.wikipedia.org/wiki/DAX"
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        tables = pd.read_html(StringIO(response.text))
        df = None
        for table in tables:
            cols_lower = [str(c).lower() for c in table.columns]
            if any('ticker' in c or 'symbol' in c for c in cols_lower):
                df = table
                break

        if df is None:
            return []

        ticker_col = None
        for col in df.columns:
            if 'ticker' in str(col).lower() or 'symbol' in str(col).lower():
                ticker_col = col
                break

        if ticker_col is None:
            return []

        symbols = []
        for sym in df[ticker_col].tolist():
            if pd.notna(sym):
                sym = str(sym).strip()
                # Wikipedia tickers already have exchange suffix (e.g. ADS.DE, AIR.PA)
                if '.' in sym:
                    symbols.append(f"XETR:{sym}")
                else:
                    symbols.append(f"XETR:{sym}.DE")

        with open(cache_path, 'w') as f:
            json.dump(symbols, f)

        print(f"Fetched {len(symbols)} DAX symbols")
        return symbols
    except Exception as e:
        print(f"Error fetching DAX: {e}")
        return []


def fetch_cac40_symbols() -> List[str]:
    """Fetch CAC 40 (France) symbols from Wikipedia."""
    cache_path = get_cache_path("cac40")

    if is_cache_valid(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)

    try:
        url = "https://en.wikipedia.org/wiki/CAC_40"
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        tables = pd.read_html(StringIO(response.text))
        df = None
        for table in tables:
            cols_lower = [str(c).lower() for c in table.columns]
            if any('ticker' in c or 'symbol' in c for c in cols_lower):
                df = table
                break

        if df is None:
            return []

        ticker_col = None
        for col in df.columns:
            if 'ticker' in str(col).lower() or 'symbol' in str(col).lower():
                ticker_col = col
                break

        if ticker_col is None:
            return []

        symbols = []
        for sym in df[ticker_col].tolist():
            if pd.notna(sym):
                sym = str(sym).strip()
                if '.' in sym:
                    symbols.append(f"EPA:{sym}")
                else:
                    symbols.append(f"EPA:{sym}.PA")

        with open(cache_path, 'w') as f:
            json.dump(symbols, f)

        print(f"Fetched {len(symbols)} CAC 40 symbols")
        return symbols
    except Exception as e:
        print(f"Error fetching CAC 40: {e}")
        return []


def fetch_ibex35_symbols() -> List[str]:
    """Fetch IBEX 35 (Spain) symbols from Wikipedia."""
    cache_path = get_cache_path("ibex35")

    if is_cache_valid(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)

    try:
        url = "https://en.wikipedia.org/wiki/IBEX_35"
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        tables = pd.read_html(StringIO(response.text))
        df = None
        for table in tables:
            cols_lower = [str(c).lower() for c in table.columns]
            if any('ticker' in c or 'symbol' in c for c in cols_lower):
                df = table
                break

        if df is None:
            return []

        ticker_col = None
        for col in df.columns:
            if 'ticker' in str(col).lower() or 'symbol' in str(col).lower():
                ticker_col = col
                break

        if ticker_col is None:
            return []

        symbols = []
        for sym in df[ticker_col].tolist():
            if pd.notna(sym):
                sym = str(sym).strip()
                if '.' in sym:
                    symbols.append(f"BME:{sym}")
                else:
                    symbols.append(f"BME:{sym}.MC")

        with open(cache_path, 'w') as f:
            json.dump(symbols, f)

        print(f"Fetched {len(symbols)} IBEX 35 symbols")
        return symbols
    except Exception as e:
        print(f"Error fetching IBEX 35: {e}")
        return []


def fetch_nikkei225_symbols() -> List[str]:
    """Fetch Nikkei 225 (Japan) symbols from topforeignstocks.com."""
    cache_path = get_cache_path("nikkei225")

    if is_cache_valid(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)

    try:
        url = "https://topforeignstocks.com/indices/the-components-of-the-nikkei-225-index/"
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        tables = pd.read_html(StringIO(response.text))
        df = None
        for table in tables:
            if 'Code' in table.columns:
                df = table
                break

        if df is None:
            return []

        symbols = []
        for code in df['Code'].dropna():
            code = str(code).strip()
            # Codes already have .T suffix (e.g. 6857.T)
            if '.T' in code:
                symbols.append(f"TSE:{code}")

        with open(cache_path, 'w') as f:
            json.dump(symbols, f)

        print(f"Fetched {len(symbols)} Nikkei 225 symbols")
        return symbols
    except Exception as e:
        print(f"Error fetching Nikkei 225: {e}")
        return []


def fetch_hangseng_symbols() -> List[str]:
    """Fetch Hang Seng Index (Hong Kong) symbols from Wikipedia."""
    cache_path = get_cache_path("hangseng")

    if is_cache_valid(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)

    try:
        url = "https://en.wikipedia.org/wiki/Hang_Seng_Index"
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        tables = pd.read_html(StringIO(response.text))
        df = None
        for table in tables:
            if 'Ticker' in table.columns:
                df = table
                break

        if df is None:
            return []

        symbols = []
        for ticker in df['Ticker'].dropna():
            # Format: "SEHK: 5" or "SEHK: 388"
            raw = str(ticker).replace('\xa0', ' ')
            if 'SEHK' in raw:
                code = raw.split(':')[-1].strip()
                if code.isdigit():
                    # yfinance uses 4-digit zero-padded codes with .HK
                    symbols.append(f"HKEX:{code.zfill(4)}.HK")

        with open(cache_path, 'w') as f:
            json.dump(symbols, f)

        print(f"Fetched {len(symbols)} Hang Seng symbols")
        return symbols
    except Exception as e:
        print(f"Error fetching Hang Seng: {e}")
        return []


def fetch_kospi200_symbols() -> List[str]:
    """Fetch KOSPI 200 (South Korea) symbols from Wikipedia."""
    cache_path = get_cache_path("kospi200")

    if is_cache_valid(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)

    try:
        url = "https://en.wikipedia.org/wiki/KOSPI_200"
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        tables = pd.read_html(StringIO(response.text))
        df = None
        for table in tables:
            if 'Symbol' in table.columns:
                df = table
                break

        if df is None:
            return []

        symbols = []
        for sym in df['Symbol'].dropna():
            code = str(sym).strip()
            if code.isdigit():
                symbols.append(f"KRX:{code}.KS")

        with open(cache_path, 'w') as f:
            json.dump(symbols, f)

        print(f"Fetched {len(symbols)} KOSPI 200 symbols")
        return symbols
    except Exception as e:
        print(f"Error fetching KOSPI 200: {e}")
        return []


def fetch_china_a50_symbols() -> List[str]:
    """Fetch FTSE China A50 symbols from Wikipedia (name-to-ticker mapping)."""
    cache_path = get_cache_path("china_a50")

    if is_cache_valid(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)

    # Wikipedia only has company names, no tickers.
    # Hardcoded mapping of the 50 largest A-shares (Shanghai .SS / Shenzhen .SZ)
    # Source: FTSE China A50 as of 2024-2025
    a50_tickers = [
        "600519.SS",  # Kweichow Moutai
        "601318.SS",  # Ping An Insurance
        "600036.SS",  # China Merchants Bank
        "601166.SS",  # Industrial Bank
        "600276.SS",  # Jiangsu Hengrui
        "601888.SS",  # China Tourism Group
        "600900.SS",  # China Yangtze Power
        "600030.SS",  # CITIC Securities
        "601398.SS",  # ICBC
        "601288.SS",  # Agricultural Bank of China
        "601328.SS",  # Bank of Communications
        "600000.SS",  # Shanghai Pudong Dev Bank
        "601988.SS",  # Bank of China
        "601939.SS",  # CCB
        "600887.SS",  # Inner Mongolia Yili
        "601012.SS",  # LONGi Green Energy
        "600809.SS",  # Shanxi Xinghuacun Fen Wine
        "601668.SS",  # China State Construction
        "600690.SS",  # Haier Smart Home
        "601857.SS",  # PetroChina
        "600028.SS",  # Sinopec
        "600048.SS",  # Poly Developments
        "600585.SS",  # Anhui Conch Cement
        "601601.SS",  # China Pacific Insurance
        "600104.SS",  # SAIC Motor
        "601816.SS",  # CGS
        "600309.SS",  # Wanhua Chemical
        "600031.SS",  # Sany Heavy
        "601225.SS",  # Shaanxi Coal
        "600050.SS",  # China Unicom
        "000858.SZ",  # Wuliangye Yibin
        "000333.SZ",  # Midea Group
        "000651.SZ",  # Gree Electric
        "000002.SZ",  # China Vanke
        "000725.SZ",  # BOE Technology
        "000568.SZ",  # Luzhou Laojiao
        "002714.SZ",  # Muyuan Foods
        "002352.SZ",  # S.F. Holding
        "002415.SZ",  # Hikvision
        "002304.SZ",  # Yanghe Brewery
        "300750.SZ",  # CATL
        "300015.SZ",  # Aier Eye Hospital
        "002594.SZ",  # BYD
        "002475.SZ",  # Luxshare Precision
        "002142.SZ",  # Bank of Ningbo
        "002230.SZ",  # iFlytek
        "300059.SZ",  # East Money Info
        "002027.SZ",  # Focus Media
        "002460.SZ",  # Ganfeng Lithium
        "001289.SZ",  # China Resources Land
    ]

    symbols = [f"SSE:{t}" if '.SS' in t else f"SZSE:{t}" for t in a50_tickers]

    with open(cache_path, 'w') as f:
        json.dump(symbols, f)

    print(f"Fetched {len(symbols)} FTSE China A50 symbols")
    return symbols


def fetch_bel20_symbols() -> List[str]:
    """Fetch BEL 20 (Belgium) symbols from Wikipedia."""
    cache_path = get_cache_path("bel20")

    if is_cache_valid(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)

    try:
        url = "https://en.wikipedia.org/wiki/BEL_20"
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        tables = pd.read_html(StringIO(response.text))
        df = None
        for table in tables:
            if 'Ticker symbol' in table.columns:
                df = table
                break

        if df is None:
            return []

        symbols = []
        for ticker in df['Ticker symbol'].dropna():
            # Format: "Euronext Brussels: ABI"
            raw = str(ticker).replace('\xa0', ' ')
            if ':' in raw:
                sym = raw.split(':')[-1].strip()
            else:
                sym = raw.strip()
            if sym:
                symbols.append(f"EBR:{sym}.BR")

        with open(cache_path, 'w') as f:
            json.dump(symbols, f)

        print(f"Fetched {len(symbols)} BEL 20 symbols")
        return symbols
    except Exception as e:
        print(f"Error fetching BEL 20: {e}")
        return []


def fetch_aex25_symbols() -> List[str]:
    """Fetch AEX 25 (Netherlands) symbols from Wikipedia."""
    cache_path = get_cache_path("aex25")

    if is_cache_valid(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)

    try:
        url = "https://en.wikipedia.org/wiki/AEX_index"
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        tables = pd.read_html(StringIO(response.text))
        df = None
        for table in tables:
            if 'Ticker' in table.columns:
                df = table
                break

        if df is None:
            return []

        symbols = []
        for ticker in df['Ticker'].dropna():
            sym = str(ticker).strip()
            # Already has .AS suffix (e.g. ABN.AS)
            if '.AS' in sym:
                symbols.append(f"AMS:{sym}")
            else:
                symbols.append(f"AMS:{sym}.AS")

        with open(cache_path, 'w') as f:
            json.dump(symbols, f)

        print(f"Fetched {len(symbols)} AEX 25 symbols")
        return symbols
    except Exception as e:
        print(f"Error fetching AEX 25: {e}")
        return []


def fetch_omxc25_symbols() -> List[str]:
    """Fetch OMXC25 (Denmark) symbols from Wikipedia."""
    cache_path = get_cache_path("omxc25")

    if is_cache_valid(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)

    try:
        url = "https://en.wikipedia.org/wiki/OMX_Copenhagen_25"
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        tables = pd.read_html(StringIO(response.text))
        df = None
        for table in tables:
            if 'Ticker symbol' in table.columns:
                df = table
                break

        if df is None:
            return []

        symbols = []
        for ticker in df['Ticker symbol'].dropna():
            sym = str(ticker).strip()
            # Replace spaces with hyphens for yfinance (e.g. MAERSK B -> MAERSK-B)
            sym = sym.replace(' ', '-')
            symbols.append(f"CPH:{sym}.CO")

        with open(cache_path, 'w') as f:
            json.dump(symbols, f)

        print(f"Fetched {len(symbols)} OMXC25 symbols")
        return symbols
    except Exception as e:
        print(f"Error fetching OMXC25: {e}")
        return []


def fetch_ftsemib_symbols() -> List[str]:
    """Fetch FTSE MIB (Italy) symbols from Wikipedia."""
    cache_path = get_cache_path("ftsemib")

    if is_cache_valid(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)

    try:
        url = "https://en.wikipedia.org/wiki/FTSE_MIB"
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        tables = pd.read_html(StringIO(response.text))
        df = None
        for table in tables:
            if 'Ticker' in table.columns:
                df = table
                break

        if df is None:
            return []

        symbols = []
        for ticker in df['Ticker'].dropna():
            sym = str(ticker).strip()
            # Already has .MI suffix (e.g. A2A.MI)
            if '.MI' in sym:
                symbols.append(f"MIL:{sym}")
            else:
                symbols.append(f"MIL:{sym}.MI")

        with open(cache_path, 'w') as f:
            json.dump(symbols, f)

        print(f"Fetched {len(symbols)} FTSE MIB symbols")
        return symbols
    except Exception as e:
        print(f"Error fetching FTSE MIB: {e}")
        return []


def fetch_omxs30_symbols() -> List[str]:
    """Fetch OMXS30 (Sweden) symbols from Wikipedia."""
    cache_path = get_cache_path("omxs30")

    if is_cache_valid(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)

    try:
        url = "https://en.wikipedia.org/wiki/OMX_Stockholm_30"
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        tables = pd.read_html(StringIO(response.text))
        df = None
        for table in tables:
            if 'Ticker' in table.columns:
                df = table
                break

        if df is None:
            return []

        symbols = []
        for ticker in df['Ticker'].dropna():
            sym = str(ticker).strip()
            # Already has .ST suffix (e.g. ABB.ST)
            if '.ST' in sym:
                symbols.append(f"STO:{sym}")
            else:
                symbols.append(f"STO:{sym}.ST")

        with open(cache_path, 'w') as f:
            json.dump(symbols, f)

        print(f"Fetched {len(symbols)} OMXS30 symbols")
        return symbols
    except Exception as e:
        print(f"Error fetching OMXS30: {e}")
        return []


def fetch_obx_symbols() -> List[str]:
    """Fetch OBX (Norway) symbols from Wikipedia."""
    cache_path = get_cache_path("obx")

    if is_cache_valid(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)

    try:
        url = "https://en.wikipedia.org/wiki/OBX_Index"
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        tables = pd.read_html(StringIO(response.text))
        df = None
        for table in tables:
            if 'Ticker symbol' in table.columns:
                df = table
                break

        if df is None:
            return []

        symbols = []
        for ticker in df['Ticker symbol'].dropna():
            # Format: "OSE: AKRBP"
            raw = str(ticker).replace('\xa0', ' ')
            if ':' in raw:
                sym = raw.split(':')[-1].strip()
            else:
                sym = raw.strip()
            if sym:
                symbols.append(f"OSL:{sym}.OL")

        with open(cache_path, 'w') as f:
            json.dump(symbols, f)

        print(f"Fetched {len(symbols)} OBX symbols")
        return symbols
    except Exception as e:
        print(f"Error fetching OBX: {e}")
        return []


def fetch_asx200_symbols() -> List[str]:
    """Fetch ASX 200 (Australia) symbols from Wikipedia."""
    cache_path = get_cache_path("asx200")

    if is_cache_valid(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)

    try:
        url = "https://en.wikipedia.org/wiki/S%26P/ASX_200"
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        tables = pd.read_html(StringIO(response.text))
        df = None
        for table in tables:
            cols_lower = [str(c).lower() for c in table.columns]
            if any('code' in c or 'ticker' in c for c in cols_lower):
                df = table
                break

        if df is None:
            return []

        ticker_col = None
        for col in df.columns:
            if 'code' in str(col).lower() or 'ticker' in str(col).lower():
                ticker_col = col
                break

        if ticker_col is None:
            return []

        symbols = [f"ASX:{sym}.AX" for sym in df[ticker_col].tolist() if pd.notna(sym)]

        with open(cache_path, 'w') as f:
            json.dump(symbols, f)

        print(f"Fetched {len(symbols)} ASX 200 symbols")
        return symbols
    except Exception as e:
        print(f"Error fetching ASX 200: {e}")
        return []


def load_watchlist_file(filepath: str) -> List[str]:
    """Load symbols from a watchlist file (TradingView format)."""
    path = Path(filepath)
    if not path.exists():
        print(f"Watchlist file not found: {filepath}")
        return []

    with open(path, 'r') as f:
        content = f.read().strip()

    if not content:
        return []

    # TradingView format: EXCHANGE:SYMBOL,EXCHANGE:SYMBOL,...
    symbols = [s.strip() for s in content.split(',') if s.strip()]
    return symbols


# Map of source names to fetch functions
SOURCE_FETCHERS = {
    # US
    'sp500': fetch_sp500_symbols,
    'nasdaq100': fetch_nasdaq100_symbols,
    'nasdaq_listed': fetch_nasdaq_listed,
    'nyse_listed': fetch_nyse_listed,
    'russell3000': fetch_russell3000_symbols,
    # Europe
    'ftse100': fetch_ftse100_symbols,
    'dax': fetch_dax_symbols,
    'cac40': fetch_cac40_symbols,
    'ibex35': fetch_ibex35_symbols,
    'ftsemib': fetch_ftsemib_symbols,
    'bel20': fetch_bel20_symbols,
    'aex25': fetch_aex25_symbols,
    'omxc25': fetch_omxc25_symbols,
    'omxs30': fetch_omxs30_symbols,
    'obx': fetch_obx_symbols,
    # Asia-Pacific
    'nikkei225': fetch_nikkei225_symbols,
    'hangseng': fetch_hangseng_symbols,
    'kospi200': fetch_kospi200_symbols,
    'china_a50': fetch_china_a50_symbols,
    'asx200': fetch_asx200_symbols,
}


def resolve_universe(universe_name: str, config: Dict[str, Any] = None) -> List[str]:
    """
    Resolve a universe name to a list of symbols.

    Args:
        universe_name: Name of the universe from config
        config: Universes config dict (loaded if not provided)

    Returns:
        List of symbols in EXCHANGE:SYMBOL format
    """
    if config is None:
        config = load_universes_config()

    universes = config.get('universes', {})

    if universe_name not in universes:
        print(f"Universe '{universe_name}' not found in config")
        return []

    universe_def = universes[universe_name]
    universe_type = universe_def.get('type', 'manual')

    symbols: Set[str] = set()

    if universe_type == 'market':
        # Fetch from market sources
        sources = universe_def.get('sources', [])
        for source in sources:
            if source in SOURCE_FETCHERS:
                fetched = SOURCE_FETCHERS[source]()
                symbols.update(fetched)
            else:
                print(f"Unknown market source: {source}")

    elif universe_type == 'watchlist':
        # Load from local watchlist files
        files = universe_def.get('files', [])
        for filepath in files:
            symbols.update(load_watchlist_file(filepath))

    elif universe_type == 'manual':
        # Use manually specified symbols
        symbols.update(universe_def.get('symbols', []))

    return sorted(list(symbols))


def get_default_universe(config: Dict[str, Any] = None) -> str:
    """Get the default universe name from config."""
    if config is None:
        config = load_universes_config()
    return config.get('default_universe', 'us_market')


def extract_ticker(symbol: str) -> str:
    """Extract ticker from EXCHANGE:SYMBOL format."""
    if ':' in symbol:
        return symbol.split(':')[1]
    return symbol


def symbols_to_tickers(symbols: List[str]) -> List[str]:
    """Convert list of EXCHANGE:SYMBOL to just tickers for yfinance."""
    return [extract_ticker(s) for s in symbols]


if __name__ == "__main__":
    # Test universe resolution
    config = load_universes_config()
    print(f"Default universe: {get_default_universe(config)}")

    symbols = resolve_universe('global_all', config)
    print(f"\nGlobal universe: {len(symbols)} symbols")
    print(f"First 10: {symbols[:10]}")
