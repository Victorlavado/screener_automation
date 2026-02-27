"""P6: Fix TradingView symbol formatting for European exchanges.

TradingView uses different exchange prefixes and ticker formats than Yahoo Finance.
The export layer must convert from internal Yahoo-style symbols to TradingView format.

Two issues:
  1. Yahoo Finance suffixes (.PA, .OL, .AS, .DE, .L, .MI, .MC, .BR, .CO, .ST)
     must be stripped — TradingView identifies the exchange via the prefix.
  2. Some exchange prefixes differ:
       EPA  -> EURONEXT  (Paris)
       AMS  -> EURONEXT  (Amsterdam)
       EBR  -> EURONEXT  (Brussels)
       STO  -> OMXSTO    (Stockholm)
       CPH  -> OMXCOP    (Copenhagen)
"""

import pytest

from src.export import (
    _strip_yahoo_suffix,
    _map_tv_exchange,
    _to_tradingview_symbol,
    export_tradingview_watchlist,
)


# ── Unit tests: _strip_yahoo_suffix ──────────────────────────────────────

class TestStripYahooSuffix:
    """Stripping Yahoo Finance exchange suffixes from tickers."""

    def test_strip_pa_suffix(self):
        assert _strip_yahoo_suffix("DG.PA") == "DG"

    def test_strip_ol_suffix(self):
        assert _strip_yahoo_suffix("EQNR.OL") == "EQNR"

    def test_strip_as_suffix(self):
        assert _strip_yahoo_suffix("KPN.AS") == "KPN"

    def test_strip_de_suffix(self):
        assert _strip_yahoo_suffix("CON.DE") == "CON"

    def test_strip_l_suffix(self):
        assert _strip_yahoo_suffix("SDR.L") == "SDR"

    def test_strip_mi_suffix(self):
        assert _strip_yahoo_suffix("MB.MI") == "MB"

    def test_strip_mc_suffix(self):
        assert _strip_yahoo_suffix("LOG.MC") == "LOG"

    def test_strip_br_suffix(self):
        assert _strip_yahoo_suffix("WDP.BR") == "WDP"

    def test_strip_co_suffix(self):
        assert _strip_yahoo_suffix("NOVO-B.CO") == "NOVO-B"

    def test_strip_st_suffix(self):
        assert _strip_yahoo_suffix("ABB.ST") == "ABB"

    def test_us_ticker_unchanged(self):
        assert _strip_yahoo_suffix("AAPL") == "AAPL"

    def test_us_ticker_with_hyphen_unchanged(self):
        assert _strip_yahoo_suffix("BRK-B") == "BRK-B"

    def test_multi_char_ticker_with_suffix(self):
        assert _strip_yahoo_suffix("STMMI.MI") == "STMMI"

    def test_numeric_ticker_with_suffix(self):
        """Tickers like HEN3.DE should strip .DE correctly."""
        assert _strip_yahoo_suffix("HEN3.DE") == "HEN3"

    def test_unknown_suffix_preserved(self):
        """A dot-suffix that is NOT a Yahoo exchange suffix should be preserved."""
        assert _strip_yahoo_suffix("FOO.XY") == "FOO.XY"

    def test_empty_string(self):
        assert _strip_yahoo_suffix("") == ""


# ── Unit tests: _map_tv_exchange ─────────────────────────────────────────

class TestMapTvExchange:
    """Mapping internal exchange codes to TradingView exchange prefixes."""

    def test_epa_to_euronext(self):
        assert _map_tv_exchange("EPA") == "EURONEXT"

    def test_ams_to_euronext(self):
        assert _map_tv_exchange("AMS") == "EURONEXT"

    def test_ebr_to_euronext(self):
        assert _map_tv_exchange("EBR") == "EURONEXT"

    def test_sto_to_omxsto(self):
        assert _map_tv_exchange("STO") == "OMXSTO"

    def test_cph_to_omxcop(self):
        assert _map_tv_exchange("CPH") == "OMXCOP"

    def test_osl_unchanged(self):
        assert _map_tv_exchange("OSL") == "OSL"

    def test_xetr_unchanged(self):
        assert _map_tv_exchange("XETR") == "XETR"

    def test_mil_unchanged(self):
        assert _map_tv_exchange("MIL") == "MIL"

    def test_bme_unchanged(self):
        assert _map_tv_exchange("BME") == "BME"

    def test_lse_unchanged(self):
        assert _map_tv_exchange("LSE") == "LSE"

    def test_nyse_unchanged(self):
        assert _map_tv_exchange("NYSE") == "NYSE"

    def test_nasdaq_unchanged(self):
        assert _map_tv_exchange("NASDAQ") == "NASDAQ"

    def test_amex_unchanged(self):
        assert _map_tv_exchange("AMEX") == "AMEX"

    def test_arca_unchanged(self):
        assert _map_tv_exchange("ARCA") == "ARCA"


# ── Unit tests: _to_tradingview_symbol ───────────────────────────────────

class TestToTradingviewSymbol:
    """Full conversion from internal format to TradingView EXCHANGE:TICKER."""

    def test_french_stock(self):
        """DG.PA on EPA -> EURONEXT:DG"""
        assert _to_tradingview_symbol("DG.PA", "EPA") == "EURONEXT:DG"

    def test_norwegian_stock(self):
        """EQNR.OL on OSL -> OSL:EQNR"""
        assert _to_tradingview_symbol("EQNR.OL", "OSL") == "OSL:EQNR"

    def test_dutch_stock(self):
        """KPN.AS on AMS -> EURONEXT:KPN"""
        assert _to_tradingview_symbol("KPN.AS", "AMS") == "EURONEXT:KPN"

    def test_german_stock(self):
        """CON.DE on XETR -> XETR:CON"""
        assert _to_tradingview_symbol("CON.DE", "XETR") == "XETR:CON"

    def test_uk_stock(self):
        """SDR.L on LSE -> LSE:SDR"""
        assert _to_tradingview_symbol("SDR.L", "LSE") == "LSE:SDR"

    def test_italian_stock(self):
        """MB.MI on MIL -> MIL:MB"""
        assert _to_tradingview_symbol("MB.MI", "MIL") == "MIL:MB"

    def test_spanish_stock(self):
        """LOG.MC on BME -> BME:LOG"""
        assert _to_tradingview_symbol("LOG.MC", "BME") == "BME:LOG"

    def test_belgian_stock(self):
        """WDP.BR on EBR -> EURONEXT:WDP"""
        assert _to_tradingview_symbol("WDP.BR", "EBR") == "EURONEXT:WDP"

    def test_swedish_stock(self):
        """ABB.ST on STO -> OMXSTO:ABB"""
        assert _to_tradingview_symbol("ABB.ST", "STO") == "OMXSTO:ABB"

    def test_danish_stock(self):
        """NOVO-B.CO on CPH -> OMXCOP:NOVO-B"""
        assert _to_tradingview_symbol("NOVO-B.CO", "CPH") == "OMXCOP:NOVO-B"

    def test_us_stock_nyse(self):
        """AAPL on NYSE -> NYSE:AAPL (no change)"""
        assert _to_tradingview_symbol("AAPL", "NYSE") == "NYSE:AAPL"

    def test_us_stock_nasdaq(self):
        """NVDA on NASDAQ -> NASDAQ:NVDA (no change)"""
        assert _to_tradingview_symbol("NVDA", "NASDAQ") == "NASDAQ:NVDA"

    def test_us_stock_amex(self):
        """DNN on AMEX -> AMEX:DNN (no change)"""
        assert _to_tradingview_symbol("DNN", "AMEX") == "AMEX:DNN"


# ── Integration: export_tradingview_watchlist ─────────────────────────────

class TestExportWatchlist:
    """End-to-end test of the TradingView watchlist file output."""

    def test_mixed_us_and_eu_symbols(self, tmp_path):
        """Verify the .txt file contains correct TradingView-format symbols."""
        symbols = ["AAPL", "DG.PA", "EQNR.OL", "KPN.AS", "CON.DE"]
        exchange_map = {
            "AAPL": "NASDAQ",
            "DG.PA": "EPA",
            "EQNR.OL": "OSL",
            "KPN.AS": "AMS",
            "CON.DE": "XETR",
        }

        # Patch OUTPUT_DIR to use tmp_path
        import src.export as export_mod
        original_dir = export_mod.OUTPUT_DIR
        export_mod.OUTPUT_DIR = tmp_path

        try:
            path = export_tradingview_watchlist(
                symbols, exchange_map, filename="test.txt"
            )

            with open(path, "r") as f:
                content = f.read()

            entries = content.split(",")
            assert "NASDAQ:AAPL" in entries
            assert "EURONEXT:DG" in entries
            assert "OSL:EQNR" in entries
            assert "EURONEXT:KPN" in entries
            assert "XETR:CON" in entries

            # Verify NO Yahoo suffixes leaked through
            assert ".PA" not in content
            assert ".OL" not in content
            assert ".AS" not in content
            assert ".DE" not in content
        finally:
            export_mod.OUTPUT_DIR = original_dir

    def test_all_user_reported_symbols(self, tmp_path):
        """Exact symbols the user reported as broken in TradingView."""
        symbols = ["DG.PA", "EQNR.OL", "KOG.OL", "KPN.AS"]
        exchange_map = {
            "DG.PA": "EPA",
            "EQNR.OL": "OSL",
            "KOG.OL": "OSL",
            "KPN.AS": "AMS",
        }

        import src.export as export_mod
        original_dir = export_mod.OUTPUT_DIR
        export_mod.OUTPUT_DIR = tmp_path

        try:
            path = export_tradingview_watchlist(
                symbols, exchange_map, filename="test.txt"
            )

            with open(path, "r") as f:
                content = f.read()

            entries = content.split(",")
            assert entries == [
                "EURONEXT:DG",
                "OSL:EQNR",
                "OSL:KOG",
                "EURONEXT:KPN",
            ]
        finally:
            export_mod.OUTPUT_DIR = original_dir
