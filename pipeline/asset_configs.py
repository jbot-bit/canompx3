#!/usr/bin/env python3
"""
Per-asset configuration for the multi-instrument ingestion pipeline.

Each asset defines:
- dbn_path: Path to the DBN file (None if not yet available)
- symbol: Logical symbol stored in bars_1m.symbol
- outright_pattern: Regex to match outright contracts (exclude spreads)
- prefix_len: Number of chars in symbol prefix before month code
  (MGC=3 -> MGCG5, NQ=2 -> NQH5, MNQ=3 -> MNQH5)
- minimum_start_date: Earliest usable date (None if unknown/no DBN)
- schema_required: Expected DBN schema (always ohlcv-1m)

FAIL-CLOSED: get_asset_config() aborts if:
- Unknown instrument
- dbn_path is None (data not available)
- dbn_path points to non-existent file
"""

import re
import sys
from datetime import date
from pathlib import Path

# Project root (same convention as paths.py)
PROJECT_ROOT = Path(__file__).parent.parent

# Month codes: universal across all CME futures
MONTH_CODES = 'FGHJKMNQUVXZ'

# =============================================================================
# ASSET CONFIGURATIONS
# =============================================================================

ASSET_CONFIGS = {
    "MGC": {
        "dbn_path": PROJECT_ROOT / "DB" / "GOLD_DB_FULLSIZE",
        "symbol": "MGC",
        "outright_pattern": re.compile(r'^MGC[FGHJKMNQUVXZ]\d{1,2}$'),
        "prefix_len": 3,
        "minimum_start_date": date(2019, 1, 1),
        "schema_required": "ohlcv-1m",
        "enabled_sessions": [
            "0900", "1000", "1100", "1800", "2300",
            "LONDON_OPEN", "US_POST_EQUITY",
            # CME_OPEN removed (Feb 2026 session correlation research):
            # r=0.75 with 0900. Never trade both — 0900 has 103 validated
            # strategies vs CME_OPEN's 14. Keeping 0900, dropping CME_OPEN.
        ],
    },
    "MNQ": {
        "dbn_path": PROJECT_ROOT / "DB" / "MNQ_DB" / "glbx-mdp3-20240204-20260203.ohlcv-1m.dbn.zst",
        "symbol": "MNQ",
        "outright_pattern": re.compile(r'^MNQ[FGHJKMNQUVXZ]\d{1,2}$'),
        "prefix_len": 3,
        "minimum_start_date": date(2024, 2, 4),
        "schema_required": "ohlcv-1m",
        "enabled_sessions": [
            "0900", "1000", "1100", "0030",
            "CME_OPEN", "LONDON_OPEN", "US_EQUITY_OPEN", "US_POST_EQUITY", "CME_CLOSE",
            # 1800 removed — use LONDON_OPEN (DST-aware) instead
        ],
    },
    "MCL": {
        "dbn_path": PROJECT_ROOT / "DB" / "MCL_DB",
        "symbol": "MCL",
        "outright_pattern": re.compile(r'^MCL[FGHJKMNQUVXZ]\d{1,2}$'),
        "prefix_len": 3,
        "minimum_start_date": date(2021, 7, 11),
        "schema_required": "ohlcv-1m",
        "enabled_sessions": [
            "0900", "1000", "1100", "1800", "2300",
            "CME_OPEN", "LONDON_OPEN",
        ],
    },
    "MES": {
        "dbn_path": Path(r"C:\db\MES_DB"),
        "symbol": "MES",
        "outright_pattern": re.compile(r'^MES[FGHJKMNQUVXZ]\d{1,2}$'),
        "prefix_len": 3,
        "minimum_start_date": date(2019, 2, 12),
        "schema_required": "ohlcv-1m",
        "enabled_sessions": [
            "0900", "1000", "1100", "0030",
            "CME_OPEN", "LONDON_OPEN", "US_EQUITY_OPEN", "US_DATA_OPEN", "US_POST_EQUITY", "CME_CLOSE",
            # 1800 removed — use LONDON_OPEN (DST-aware) instead
        ],
    },
    "M2K": {
        # Source data is RTY (E-mini Russell 2000, $50/pt) — same price, stored as symbol='M2K'.
        # Identical pattern to GC→MGC, 6E→M6E. Cost model uses M2K micro specs ($5/pt).
        # Quarterly cycle only: H/M/U/Z.
        "dbn_path": PROJECT_ROOT / "DB" / "M2K_DB",
        "symbol": "M2K",
        "outright_pattern": re.compile(r'^RTY[HMUZ]\d{1,2}$'),
        "prefix_len": 3,
        "minimum_start_date": date(2021, 2, 21),
        "schema_required": "ohlcv-1m",
        "enabled_sessions": [
            "0900", "1000", "1100", "0030",
            "CME_OPEN", "LONDON_OPEN", "US_EQUITY_OPEN", "US_DATA_OPEN", "US_POST_EQUITY", "CME_CLOSE",
            # 1800 removed — use LONDON_OPEN (DST-aware) instead
        ],
    },
    "SIL": {
        "dbn_path": PROJECT_ROOT / "DB" / "SL_DB",
        "symbol": "SIL",
        "outright_pattern": re.compile(r'^SI[FGHJKMNQUVXZ]\d{1,2}$'),
        "prefix_len": 2,
        "minimum_start_date": date(2024, 2, 18),
        "schema_required": "ohlcv-1m",
        "enabled_sessions": [
            "1100", "2300", "0030",
            "US_EQUITY_OPEN", "US_DATA_OPEN", "US_POST_EQUITY",
        ],
    },
    "M6E": {
        # Micro EUR/USD futures. Source data is 6E (full-size, 125,000 EUR) — same
        # price, stored as symbol='M6E', source_symbol='6EH25' etc. Identical pattern
        # to GC→MGC (see CLAUDE.md). Cost model uses M6E micro contract specs.
        # Quarterly cycle only: H/M/U/Z (Mar/Jun/Sep/Dec).
        # Sessions prioritise FX events: London open + US data release are primary.
        # Asia sessions (1000/1100) are WATCH-ONLY until data confirms breakout edge.
        "dbn_path": PROJECT_ROOT / "DB" / "M6E_DB",
        "symbol": "M6E",
        "outright_pattern": re.compile(r'^6E[HMUZ]\d{1,2}$'),
        "prefix_len": 2,
        "minimum_start_date": date(2021, 2, 21),
        "schema_required": "ohlcv-1m",
        "enabled_sessions": [
            "1000", "1100",         # Asia — watch-only; AU/NZ/EU data sometimes moves FX
            "1800",                 # Fixed London open (for DST split analysis)
            "LONDON_OPEN",          # Dynamic London open (best FX session)
            "US_DATA_OPEN",         # Dynamic 8:30 ET data release (major FX mover)
            "US_EQUITY_OPEN",       # Dynamic 9:30 ET
            "0030",                 # Fixed US equity open (for DST split analysis)
            "US_POST_EQUITY",       # Dynamic 10:00 ET
        ],
    },
    "NQ": {
        "dbn_path": None,
        "symbol": "NQ",
        "outright_pattern": re.compile(r'^NQ[FGHJKMNQUVXZ]\d{1,2}$'),
        "prefix_len": 2,
        "minimum_start_date": None,
        "schema_required": "ohlcv-1m",
        "enabled_sessions": [],
    },
}


def get_asset_config(instrument: str) -> dict:
    """
    Return validated config for the given instrument.

    FAIL-CLOSED:
    - Prints error and calls sys.exit(1) if instrument unknown
    - Prints error and calls sys.exit(1) if dbn_path is None
    - Prints error and calls sys.exit(1) if dbn_path file does not exist
    """
    instrument = instrument.upper()

    if instrument not in ASSET_CONFIGS:
        print(f"FATAL: Unknown instrument '{instrument}'")
        print(f"       Supported instruments: {sorted(ASSET_CONFIGS.keys())}")
        sys.exit(1)

    config = ASSET_CONFIGS[instrument]

    if config["dbn_path"] is None:
        print(f"FATAL: No DBN file configured for instrument '{instrument}'")
        print(f"       To add {instrument} support, provide a DBN file path in pipeline/asset_configs.py")
        sys.exit(1)

    if not config["dbn_path"].exists():
        print(f"FATAL: DBN file not found for instrument '{instrument}'")
        print(f"       Expected: {config['dbn_path']}")
        sys.exit(1)

    if config["minimum_start_date"] is None:
        print(f"FATAL: No minimum_start_date configured for instrument '{instrument}'")
        print("       Set minimum_start_date in pipeline/asset_configs.py after validating data coverage")
        sys.exit(1)

    return config


def list_instruments() -> list[str]:
    """Return sorted list of all configured instrument names."""
    return sorted(ASSET_CONFIGS.keys())


def get_enabled_sessions(instrument: str) -> list[str]:
    """Return enabled session labels for an instrument. Fail-closed."""
    config = ASSET_CONFIGS.get(instrument.upper())
    if config is None:
        return []
    return config.get("enabled_sessions", [])


def list_available_instruments() -> list[str]:
    """Return sorted list of instruments that have DBN files on disk."""
    available = []
    for name, cfg in ASSET_CONFIGS.items():
        if cfg["dbn_path"] is not None and cfg["dbn_path"].exists():
            available.append(name)
    return sorted(available)
