#!/usr/bin/env python3
"""
Per-asset configuration for the multi-instrument ingestion pipeline.

Each asset defines:
- dbn_path: Path to the DBN file (None if not yet available)
- symbol: Logical symbol stored in bars_1m.symbol
- outright_pattern: Regex to match outright contracts (exclude spreads)
- prefix_len: Number of chars in symbol prefix before month code
  (GC=2 -> GCG5, NQ=2 -> NQH5, MNQ=3 -> MNQH5)
- minimum_start_date: Earliest usable date (None if unknown/no DBN)
- schema_required: Expected DBN schema (always ohlcv-1m)
- orb_active: Whether the instrument belongs in the active ORB/live universe

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
MONTH_CODES = "FGHJKMNQUVXZ"

# =============================================================================
# ASSET CONFIGURATIONS
# =============================================================================

ASSET_CONFIGS = {
    "MBT": {
        # Source data is BTC (full-size Bitcoin, 5 BTC/contract) — same price, stored as symbol='MBT'.
        # Identical pattern to GC→MGC, RTY→M2K. Cost model uses MBT micro specs ($0.10/pt).
        # BTC trades ~30k contracts/day vs MBT ~5k — far better 1m bar coverage.
        "dbn_path": PROJECT_ROOT / "DB" / "BTC_DB",
        "symbol": "MBT",
        "outright_pattern": re.compile(r"^BTC[FGHJKMNQUVXZ]\d{1,2}$"),
        "prefix_len": 3,
        "minimum_start_date": date(2021, 2, 1),
        "schema_required": "ohlcv-1m",
        "enabled_sessions": [
            "CME_REOPEN",
            "TOKYO_OPEN",
            "SINGAPORE_OPEN",
            "LONDON_METALS",
            "US_DATA_830",
            "NYSE_OPEN",
            "US_DATA_1000",
            "COMEX_SETTLE",
            "CME_PRECLOSE",
            "NYSE_CLOSE",
        ],
        "orb_active": False,  # 0 validated, NO ORB edge (Mar 2026)
    },
    "MGC": {
        # Real Micro Gold (10oz, $10/pt). Source: MGC.FUT contract series from
        # 2023-09-11 (MGC contract launch). Pre-launch parent GC data is preserved
        # under symbol='GC' (see GC config below) — relabeled 2026-04-08 as part of
        # Phase 2 of canonical-data-redownload (docs/plans/2026-04-07-canonical-data-redownload.md).
        # Cost model uses MGC micro specs ($10/pt, NOT GC's $100/pt).
        "dbn_path": PROJECT_ROOT / "data" / "raw" / "databento" / "ohlcv-1m" / "MGC",
        "symbol": "MGC",
        "orb_active": True,
        "outright_pattern": re.compile(r"^MGC[FGHJKMNQUVXZ]\d{1,2}$"),
        "prefix_len": 3,
        "minimum_start_date": date(2023, 9, 11),  # MGC contract launch date (real micro)
        "schema_required": "ohlcv-1m",
        "enabled_sessions": [
            "CME_REOPEN",
            "TOKYO_OPEN",
            "SINGAPORE_OPEN",
            "LONDON_METALS",
            "EUROPE_FLOW",
            "US_DATA_830",
            "NYSE_OPEN",
            "US_DATA_1000",
            "COMEX_SETTLE",
        ],
    },
    "MNQ": {
        # Real Micro Nasdaq-100 ($2/pt). Source: MNQ.FUT contract series from
        # 2019-05-06 (MNQ contract launch). Pre-launch parent NQ data is preserved
        # under symbol='NQ' (see NQ config below) — relabeled 2026-04-08 as part of
        # Phase 2 of canonical-data-redownload (docs/plans/2026-04-07-canonical-data-redownload.md).
        "dbn_path": PROJECT_ROOT / "data" / "raw" / "databento" / "ohlcv-1m" / "MNQ",
        "symbol": "MNQ",
        "orb_active": True,
        "outright_pattern": re.compile(r"^MNQ[FGHJKMNQUVXZ]\d{1,2}$"),
        "prefix_len": 3,
        "minimum_start_date": date(2019, 5, 6),  # MNQ contract launch date (real micro)
        "schema_required": "ohlcv-1m",
        "enabled_sessions": [
            "CME_REOPEN",
            "TOKYO_OPEN",
            "SINGAPORE_OPEN",
            "LONDON_METALS",
            "EUROPE_FLOW",
            "US_DATA_830",
            "NYSE_OPEN",
            "US_DATA_1000",
            "COMEX_SETTLE",
            "CME_PRECLOSE",
            "NYSE_CLOSE",
            "BRISBANE_1025",
        ],
    },
    "MCL": {
        # Source data is CL (full-size Crude Oil, $1000/pt) — same price, stored as symbol='MCL'.
        # Identical pattern to GC→MGC, RTY→M2K. Cost model uses MCL micro specs ($100/pt).
        # CL trades ~500k contracts/day vs MCL ~50k — far better 1m bar coverage.
        "dbn_path": PROJECT_ROOT / "DB" / "CL_DB",
        "symbol": "MCL",
        "orb_active": False,
        "outright_pattern": re.compile(r"^CL[FGHJKMNQUVXZ]\d{1,2}$"),
        "prefix_len": 2,
        "minimum_start_date": date(2021, 2, 1),
        "schema_required": "ohlcv-1m",
        "enabled_sessions": [
            "CME_REOPEN",
            "TOKYO_OPEN",
            "SINGAPORE_OPEN",
            "LONDON_METALS",
            "US_DATA_830",
            "NYSE_OPEN",
            "US_DATA_1000",
            "COMEX_SETTLE",
        ],
    },
    "MES": {
        # Real Micro S&P 500 ($5/pt). Source: MES.FUT contract series from
        # 2019-05-06 (MES contract launch). Pre-launch parent ES data is preserved
        # under symbol='ES' (see ES config below) — relabeled 2026-04-08 as part of
        # Phase 2 of canonical-data-redownload (docs/plans/2026-04-07-canonical-data-redownload.md).
        "dbn_path": PROJECT_ROOT / "data" / "raw" / "databento" / "ohlcv-1m" / "MES",
        "symbol": "MES",
        "orb_active": True,
        "outright_pattern": re.compile(r"^MES[FGHJKMNQUVXZ]\d{1,2}$"),
        "prefix_len": 3,
        "minimum_start_date": date(2019, 5, 6),  # MES contract launch date (real micro)
        "schema_required": "ohlcv-1m",
        "enabled_sessions": [
            "CME_REOPEN",
            "TOKYO_OPEN",
            "SINGAPORE_OPEN",
            "LONDON_METALS",
            "EUROPE_FLOW",
            "US_DATA_830",
            "NYSE_OPEN",
            "US_DATA_1000",
            "COMEX_SETTLE",
            "CME_PRECLOSE",
            "NYSE_CLOSE",
        ],
    },
    "ES": {
        # Parent E-mini S&P 500 ($50/pt). Source: ES.FUT contract series (E-mini S&P).
        # Pre-2019-05-06 data is the only historical source for the S&P; post-launch
        # is preserved here for parent-vs-micro comparisons but discovery uses MES (real micro).
        # Stored as symbol='ES' (relabeled 2026-04-08 from former 'MES' as part of Phase 2
        # of canonical-data-redownload). Cost model: PARENT specs ($50/pt), NOT MES specs.
        "dbn_path": PROJECT_ROOT / "DB" / "MES_DB_2019-2024",
        "symbol": "ES",
        "orb_active": False,
        "outright_pattern": re.compile(r"^ES[FGHJKMNQUVXZ]\d{1,2}$"),
        "prefix_len": 2,
        "minimum_start_date": date(2010, 6, 6),  # Extended via ES.FUT backfill
        "schema_required": "ohlcv-1m",
        "enabled_sessions": [
            "CME_REOPEN",
            "TOKYO_OPEN",
            "SINGAPORE_OPEN",
            "LONDON_METALS",
            "EUROPE_FLOW",
            "US_DATA_830",
            "NYSE_OPEN",
            "US_DATA_1000",
            "COMEX_SETTLE",
            "CME_PRECLOSE",
            "NYSE_CLOSE",
        ],
    },
    "M2K": {
        # Source data is RTY (E-mini Russell 2000, $50/pt) — same price, stored as symbol='M2K'.
        # Identical pattern to GC→MGC, 6E→M6E. Cost model uses M2K micro specs ($5/pt).
        # Quarterly cycle only: H/M/U/Z.
        "dbn_path": PROJECT_ROOT / "DB" / "M2K_DB",
        "symbol": "M2K",
        "orb_active": True,
        "outright_pattern": re.compile(r"^RTY[HMUZ]\d{1,2}$"),
        "prefix_len": 3,
        "minimum_start_date": date(2021, 2, 21),
        "schema_required": "ohlcv-1m",
        "enabled_sessions": [
            "CME_REOPEN",
            "TOKYO_OPEN",
            "SINGAPORE_OPEN",
            "LONDON_METALS",
            "EUROPE_FLOW",
            "US_DATA_830",
            "NYSE_OPEN",
            "US_DATA_1000",
            "COMEX_SETTLE",
            "CME_PRECLOSE",
            "NYSE_CLOSE",
            "BRISBANE_1025",
        ],
    },
    "SIL": {
        # Source data is SI (full-size Silver, $5000/pt) — same price, stored as symbol='SIL'.
        # Identical pattern to GC→MGC, 6E→M6E, RTY→M2K. Cost model uses SIL micro specs ($1000/pt).
        "dbn_path": PROJECT_ROOT / "DB" / "SL_DB",
        "symbol": "SIL",
        "orb_active": False,
        "outright_pattern": re.compile(r"^SI[FGHJKMNQUVXZ]\d{1,2}$"),
        "prefix_len": 2,
        "minimum_start_date": date(2024, 2, 18),
        "schema_required": "ohlcv-1m",
        "enabled_sessions": [
            "SINGAPORE_OPEN",
            "US_DATA_830",
            "NYSE_OPEN",
            "US_DATA_1000",
        ],
    },
    "M6E": {
        # Micro EUR/USD futures. Source data is 6E (full-size, 125,000 EUR) — same
        # price, stored as symbol='M6E', source_symbol='6EH25' etc. Identical pattern
        # to GC→MGC (see CLAUDE.md). Cost model uses M6E micro contract specs.
        # Quarterly cycle only: H/M/U/Z (Mar/Jun/Sep/Dec).
        # Sessions prioritise FX events: London open + US data release are primary.
        # Asia sessions (TOKYO_OPEN/SINGAPORE_OPEN) are WATCH-ONLY until data confirms breakout edge.
        "dbn_path": PROJECT_ROOT / "DB" / "M6E_DB",
        "symbol": "M6E",
        "orb_active": False,
        "outright_pattern": re.compile(r"^6E[HMUZ]\d{1,2}$"),
        "prefix_len": 2,
        "minimum_start_date": date(2021, 2, 21),
        "schema_required": "ohlcv-1m",
        "enabled_sessions": [
            "TOKYO_OPEN",
            "SINGAPORE_OPEN",  # Asia — watch-only
            "LONDON_METALS",  # London metals (best FX session)
            "US_DATA_830",  # 8:30 ET data release (major FX mover)
            "NYSE_OPEN",  # 9:30 ET
            "US_DATA_1000",  # 10:00 ET
        ],
    },
    "NQ": {
        # Parent E-mini Nasdaq-100 ($20/pt). Source: NQ.FUT contract series (E-mini Nasdaq).
        # Pre-2019-05-06 data is the only historical source for the Nasdaq; post-launch
        # is preserved here for parent-vs-micro comparisons but discovery uses MNQ (real micro).
        # Stored as symbol='NQ' (relabeled 2026-04-08 from former 'MNQ' as part of Phase 2
        # of canonical-data-redownload). Cost model: PARENT specs ($20/pt), NOT MNQ specs.
        "dbn_path": PROJECT_ROOT / "DB" / "NQ_DB_2021-2024",
        "symbol": "NQ",
        "orb_active": False,
        "outright_pattern": re.compile(r"^NQ[FGHJKMNQUVXZ]\d{1,2}$"),
        "prefix_len": 2,
        "minimum_start_date": date(2010, 6, 6),  # Extended via NQ.FUT backfill
        "schema_required": "ohlcv-1m",
        "enabled_sessions": [
            "CME_REOPEN",
            "TOKYO_OPEN",
            "SINGAPORE_OPEN",
            "LONDON_METALS",
            "EUROPE_FLOW",
            "US_DATA_830",
            "NYSE_OPEN",
            "US_DATA_1000",
            "COMEX_SETTLE",
            "CME_PRECLOSE",
            "NYSE_CLOSE",
            "BRISBANE_1025",
        ],
    },
    "2YY": {
        # Research-only rates candidate. Kept out of ACTIVE_ORB_INSTRUMENTS on purpose.
        # Source data is native CME 2-Year Yield futures (parent symbol 2YY).
        # The initial research path is event-window macro work, not ORB/live trading.
        "dbn_path": PROJECT_ROOT / "DB" / "2YY_DB",
        "symbol": "2YY",
        "orb_active": False,
        "outright_pattern": re.compile(r"^2YY[FGHJKMNQUVXZ]\d{1,2}$"),
        "prefix_len": 3,
        "minimum_start_date": date(2021, 2, 1),
        "schema_required": "ohlcv-1m",
        "enabled_sessions": [],
    },
    "ZT": {
        # Research-only rates benchmark. Kept out of ACTIVE_ORB_INSTRUMENTS on purpose.
        # Source data is native CME 2-Year Treasury Note futures (parent symbol ZT).
        # The initial research path is event-window macro work, not ORB/live trading.
        "dbn_path": PROJECT_ROOT / "DB" / "ZT_DB",
        "symbol": "ZT",
        "orb_active": False,
        "outright_pattern": re.compile(r"^ZT[FGHJKMNQUVXZ]\d{1,2}$"),
        "prefix_len": 2,
        "minimum_start_date": date(2021, 2, 1),
        "schema_required": "ohlcv-1m",
        "enabled_sessions": [],
    },
}

# Dead instruments — tested and confirmed NO ORB breakout edge.
# This is the canonical source for dead instruments.
# M2K added 2026-03-18: null test (8 seeds) — 0/18 families survive noise screening.
#   Max family head ExpR = 0.223, below noise P95 (E2=0.20). Dead at any threshold.
DEAD_ORB_INSTRUMENTS = frozenset({"MCL", "SIL", "M6E", "MBT", "M2K"})

# Primary instruments actively traded for ORB breakout.
# Excludes source aliases (ES, NQ) and dead-for-ORB instruments.
# This is the canonical source — import this instead of hardcoding instrument lists.
ACTIVE_ORB_INSTRUMENTS = sorted(
    [
        k
        for k, v in ASSET_CONFIGS.items()
        if v["symbol"] == k and v.get("orb_active", True) and k not in DEAD_ORB_INSTRUMENTS
    ]
)


def get_active_instruments() -> list[str]:
    """Return the list of actively traded ORB instruments (sorted copy).

    Use this instead of hardcoding instrument lists. Dead: MCL, SIL, M6E, MBT, M2K.
    """
    return list(ACTIVE_ORB_INSTRUMENTS)


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
