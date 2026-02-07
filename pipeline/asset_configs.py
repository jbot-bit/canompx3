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
    },
    "MNQ": {
        "dbn_path": None,
        "symbol": "MNQ",
        "outright_pattern": re.compile(r'^MNQ[FGHJKMNQUVXZ]\d{1,2}$'),
        "prefix_len": 3,
        "minimum_start_date": None,
        "schema_required": "ohlcv-1m",
    },
    "NQ": {
        "dbn_path": None,
        "symbol": "NQ",
        "outright_pattern": re.compile(r'^NQ[FGHJKMNQUVXZ]\d{1,2}$'),
        "prefix_len": 2,
        "minimum_start_date": None,
        "schema_required": "ohlcv-1m",
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
        print(f"       Set minimum_start_date in pipeline/asset_configs.py after validating data coverage")
        sys.exit(1)

    return config


def list_instruments() -> list[str]:
    """Return sorted list of all configured instrument names."""
    return sorted(ASSET_CONFIGS.keys())


def list_available_instruments() -> list[str]:
    """Return sorted list of instruments that have DBN files on disk."""
    available = []
    for name, cfg in ASSET_CONFIGS.items():
        if cfg["dbn_path"] is not None and cfg["dbn_path"].exists():
            available.append(name)
    return sorted(available)
