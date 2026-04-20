#!/usr/bin/env python3
"""
Per-asset configuration for the multi-instrument ingestion pipeline.

Each asset defines:
- dbn_path: Path to the DBN file (None if not yet available)
- symbol: Logical symbol stored in bars_1m.symbol
- outright_pattern: Regex to match outright contracts (exclude spreads)
- prefix_len: Number of chars in symbol prefix before month code
  (GC=2 -> GCG5, NQ=2 -> NQH5, MNQ=3 -> MNQH5)
- minimum_start_date: Earliest usable date (None if unknown/no DBN). For
  micros this is the contract launch date used by `data_era.micro_launch_day`.
- schema_required: Expected DBN schema (always ohlcv-1m)
- orb_active: Whether the instrument belongs in the active ORB/live universe
  (data pipeline, discovery, validator all run on it).
- deployable_expected: Whether this instrument is expected to have deployable
  validated strategies in the current data horizon. Defaults to True. Set to
  False for active-but-research-only instruments where the real-micro data
  horizon is insufficient for T7 era-discipline to let candidates survive.
  Consumed by pulse/staleness alerting only — does NOT affect pipeline,
  discovery, or validator behavior. A research-only instrument still runs
  through the pipeline end to end; pulse just stops alerting on the
  by-design empty deployable state.
- parent_symbol: Canonical parent contract symbol for micros that proxy to
  a full-size instrument, or None for native parents and research-only
  instruments. Consumed by `pipeline.data_era` for PARENT/MICRO
  classification (Phase 3a foundation, Apr 2026).

API split (2026-04-19, PR-A refactor):
- get_asset_config(instrument) -> dict
    Metadata-only. Raises ValueError for unknown instrument. Never touches the
    filesystem. Safe for tests and callers that only need config metadata.
- require_dbn_available(instrument) -> Path
    Fail-closed DBN access. Raises ValueError for unknown instrument / missing
    dbn_path entry / missing minimum_start_date. Raises FileNotFoundError if
    the DBN file/dir does not exist on disk. Call this from any code path that
    will actually open the DBN store.

Prior to 2026-04-19 get_asset_config() performed both roles via sys.exit(1).
That coupled "I need metadata" and "I need the DBN on disk" and made the
module unimportable in CI / test contexts. Callers that need DBN access now
opt into it explicitly. See docs/runtime/stages/pr-a-asset-configs-lazy-dbn.md.
"""

import re
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
        "parent_symbol": "BTC",  # dead micro — uses full-size BTC parent data
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
        # 2022-06-13 (CME Micro Gold launch). Prior minimum_start_date was 2023-09-11
        # (our first download date, not the actual launch). Corrected 2026-04-09 when
        # backfill data (2022-06-13 to 2023-09-10) was downloaded from Databento.
        # Pre-launch parent GC data is preserved under symbol='GC' (see GC config below).
        # Cost model uses MGC micro specs ($10/pt, NOT GC's $100/pt).
        #
        # deployable_expected=False: MGC real-micro horizon is ~3.8yr as of 2026-04,
        # below T7 era-discipline threshold (needs 5+yr for cross-era stability).
        # Discovery still runs on MGC proxy data (GC, 16yr) for research; validator
        # T7 correctly kills the resulting candidates; deployable shelf is expected
        # empty until real-micro horizon reaches 5+yr (~2027-06). See
        # memory/gc_mgc_cross_validation_results.md and Amendment 3.1.
        "dbn_path": PROJECT_ROOT / "data" / "raw" / "databento" / "ohlcv-1m" / "MGC",
        "symbol": "MGC",
        "parent_symbol": "GC",  # active micro — GC parent preserved under symbol='GC'
        "orb_active": True,
        "deployable_expected": False,
        "outright_pattern": re.compile(r"^MGC[FGHJKMNQUVXZ]\d{1,2}$"),
        "prefix_len": 3,
        "minimum_start_date": date(2022, 6, 13),  # CME Micro Gold launch (10oz, first traded 2022-06-13)
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
        "parent_symbol": "NQ",  # active micro — NQ parent preserved under symbol='NQ'
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
        "parent_symbol": "CL",  # dead micro — uses full-size CL parent data
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
        "parent_symbol": "ES",  # active micro — ES parent preserved under symbol='ES'
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
        "parent_symbol": None,  # ES IS the parent
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
        "parent_symbol": "RTY",  # dead micro — uses E-mini RTY parent data
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
        "parent_symbol": "SI",  # dead micro — uses full-size SI parent data
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
        "parent_symbol": "6E",  # dead micro — uses full-size 6E parent data
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
        "parent_symbol": None,  # NQ IS the parent
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
    "GC": {
        # Parent full-size Gold ($100/pt, 100oz). Source: GC.FUT contract series.
        # Pre-2023-09-11 data is the only historical source for gold; post-launch is
        # preserved here for parent-vs-micro comparisons but discovery uses MGC (real micro).
        # Stored as symbol='GC' (relabeled 2026-04-08 from former 'MGC' as part of Phase 2
        # of canonical-data-redownload). Cost model: PARENT specs ($100/pt), NOT MGC specs.
        # Created 2026-04-08 to close the docstring promise in the MGC config above
        # (Bloomey review HIGH finding on commit 82e8b60).
        "dbn_path": PROJECT_ROOT / "DB" / "GOLD_DB_FULLSIZE",
        "symbol": "GC",
        "parent_symbol": None,  # GC IS the parent
        "orb_active": False,
        "outright_pattern": re.compile(r"^GC[FGHJKMNQUVXZ]\d{1,2}$"),
        "prefix_len": 2,
        "minimum_start_date": date(2010, 6, 6),  # Extended via GC.FUT backfill
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
    "2YY": {
        # Research-only rates candidate. Kept out of ACTIVE_ORB_INSTRUMENTS on purpose.
        # Source data is native CME 2-Year Yield futures (parent symbol 2YY).
        # The initial research path is event-window macro work, not ORB/live trading.
        "dbn_path": PROJECT_ROOT / "DB" / "2YY_DB",
        "symbol": "2YY",
        "parent_symbol": None,  # research-only native (no micro counterpart)
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
        "parent_symbol": None,  # research-only native (no micro counterpart)
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

# Active instruments expected to have deployable validated strategies in the
# current data horizon. Strict subset of ACTIVE_ORB_INSTRUMENTS — every
# deployable instrument is also active, but not every active instrument is
# deployable (e.g. MGC is active for research + pipeline + discovery, but its
# real-micro horizon is insufficient for validator T7 survival).
#
# Consumed by pulse and staleness alerting only. Pipeline, discovery, and
# validator consumers MUST continue to use ACTIVE_ORB_INSTRUMENTS — switching
# them would silently skip research work on MGC that we actually want running.
DEPLOYABLE_ORB_INSTRUMENTS = sorted(
    [k for k in ACTIVE_ORB_INSTRUMENTS if ASSET_CONFIGS[k].get("deployable_expected", True)]
)


def get_active_instruments() -> list[str]:
    """Return the list of actively traded ORB instruments (sorted copy).

    Use this instead of hardcoding instrument lists. Dead: MCL, SIL, M6E, MBT, M2K.
    """
    return list(ACTIVE_ORB_INSTRUMENTS)


def get_deployable_instruments() -> list[str]:
    """Return the list of instruments expected to have deployable strategies.

    Strict subset of active instruments. Use this in pulse/alerting contexts
    where empty deployable state should be surfaced as a problem. Use
    `get_active_instruments()` for pipeline/discovery/validator contexts where
    research-only instruments (e.g. MGC) still belong in the run set.
    """
    return list(DEPLOYABLE_ORB_INSTRUMENTS)


def get_asset_config(instrument: str) -> dict:
    """Return metadata config for the given instrument.

    Metadata-only. Does NOT touch the filesystem. Safe to call from tests or
    any context that only needs symbol / outright_pattern / minimum_start_date
    / enabled_sessions / parent_symbol / etc.

    Callers that will actually open the DBN store must additionally call
    `require_dbn_available(instrument)` — this function does NOT validate
    dbn_path existence.

    Raises:
        ValueError: if `instrument` is not in ASSET_CONFIGS.
    """
    instrument = instrument.upper()

    if instrument not in ASSET_CONFIGS:
        raise ValueError(f"Unknown instrument '{instrument}'. Supported instruments: {sorted(ASSET_CONFIGS.keys())}")

    return ASSET_CONFIGS[instrument]


def require_dbn_available(instrument: str) -> Path:
    """Fail-closed DBN path validator. Returns dbn_path after checking disk.

    Call this from every code path that will open the raw DBN store. Kept
    separate from `get_asset_config` so tests and metadata-only callers never
    trigger filesystem validation.

    Raises:
        ValueError: if `instrument` is unknown, dbn_path entry is None, or
            minimum_start_date is None (config incomplete).
        FileNotFoundError: if dbn_path is configured but does not exist on
            disk, or if a DBN directory exists but contains no files for the
            instrument's canonical outright root.
    """
    config = get_asset_config(instrument)

    dbn_path = config["dbn_path"]
    if dbn_path is None:
        raise ValueError(
            f"No DBN file configured for instrument '{instrument}'. "
            f"To add {instrument} support, provide a DBN file path in "
            "pipeline/asset_configs.py"
        )

    if not dbn_path.exists():
        raise FileNotFoundError(f"DBN file not found for instrument '{instrument}'. Expected: {dbn_path}")

    if not _dbn_store_has_matching_files(instrument, dbn_path):
        root = get_outright_root(instrument)
        raise FileNotFoundError(
            f"DBN store for instrument '{instrument}' exists but contains no matching raw DBN files "
            f"for root '{root}'. Expected under: {dbn_path}"
        )

    if config["minimum_start_date"] is None:
        raise ValueError(
            f"No minimum_start_date configured for instrument '{instrument}'. "
            "Set minimum_start_date in pipeline/asset_configs.py after "
            "validating data coverage"
        )

    return dbn_path


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
    """Return sorted list of instruments with matching raw DBN files on disk."""
    available = []
    for name, cfg in ASSET_CONFIGS.items():
        if cfg["dbn_path"] is not None and _dbn_store_has_matching_files(name, cfg["dbn_path"]):
            available.append(name)
    return sorted(available)


# Single source of truth for the outright contract root prefix.
# Used by every script that needs to map instrument → vendor parent symbol
# (Databento `MGC.FUT`, etc.). Replaces parallel hardcoded dicts that drifted
# from canonical post-Phase-2 (commit 82e8b60, Apr 8 2026).
_OUTRIGHT_ROOT_RE = re.compile(r"^\^(\w+?)\[")


def get_outright_root(instrument: str) -> str:
    """Return the contract root prefix for an instrument (canonical derivation).

    The root is the prefix shared by every outright contract symbol for the
    instrument — e.g., `MGC` for `MGCM4`/`MGCZ4`, `RTY` for `RTYH4`/`RTYZ4`
    (M2K data source). Derived from the existing `outright_pattern` regex
    on `ASSET_CONFIGS[instrument]`. Single source of truth — no parallel
    dicts allowed (institutional rigor rule 4 — `.claude/rules/integrity-guardian.md`).

    Examples:
        get_outright_root("MGC") -> "MGC"   # post-Phase-2 real micro
        get_outright_root("M2K") -> "RTY"   # micro Russell uses RTY parent data
        get_outright_root("NQ")  -> "NQ"    # parent contract

    Fail-closed:
        - Unknown instrument           → ValueError
        - Non-canonical outright_pattern → ValueError (catches future patterns
          that don't match the `^<ROOT>[<MONTH_CODES>]\\d+$` shape)

    Consumers (Apr 2026):
        - scripts/tools/refresh_data.py — Databento backfill download
        - scripts/databento_daily.py    — daily research-archive refresh
    """
    key = instrument.upper()
    cfg = ASSET_CONFIGS.get(key)
    if cfg is None:
        raise ValueError(f"Unknown instrument: {instrument!r}. Supported: {sorted(ASSET_CONFIGS.keys())}")
    pattern = cfg["outright_pattern"].pattern
    match = _OUTRIGHT_ROOT_RE.match(pattern)
    if match is None:
        raise ValueError(
            f"Non-canonical outright_pattern for {key}: {pattern!r}. Expected format: ^<ROOT>[<MONTH_CODES>]\\d+$"
        )
    return match.group(1)


def _dbn_store_has_matching_files(instrument: str, dbn_path: Path) -> bool:
    """Return True if the DBN store contains at least one matching raw file.

    Match on the canonical outright root rather than the instrument name because
    some micro instruments intentionally use parent-data roots (for example
    `M2K -> RTY`, `MBT -> BTC`, `M6E -> 6E`).
    """
    if not dbn_path.exists():
        return False

    root = get_outright_root(instrument)
    pattern = re.compile(rf"(^|[^A-Z0-9]){re.escape(root)}([^A-Z0-9]|$)")

    if dbn_path.is_file():
        return bool(pattern.search(dbn_path.name.upper()))

    return any(pattern.search(path.name.upper()) for path in dbn_path.rglob("*.dbn.zst"))
