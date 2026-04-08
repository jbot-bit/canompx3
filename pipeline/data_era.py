"""Canonical PARENT vs MICRO classification for bars_1m / orb_outcomes rows.

Phase 3a foundation module (docs/plans/2026-04-07-canonical-data-redownload.md).

Distinguishes real-micro data from parent-proxy data so that Stage 3b/3c/3d
can enforce era discipline on downstream layers. Reads the canonical
`parent_symbol` field and `outright_pattern` regex from
`pipeline.asset_configs.ASSET_CONFIGS` — no parallel state, no re-encoding.

Two classification paths:

1. **Row-level (bars_1m):** `era_for_source_symbol(instrument, source_symbol)`
   matches `source_symbol` against the instrument's canonical
   `outright_pattern` first. If the instrument is a micro and the source
   doesn't match, it falls back to the parent's `outright_pattern` — that
   branch catches the exact pre-Phase-2 corruption pattern (rows labeled
   `symbol='MNQ'` with `source_symbol='NQH4'`) and classifies them as
   PARENT so drift checks can detect them.

2. **Date-level (orb_outcomes / daily_features):** `era_for_trading_day(
   instrument, trading_day)` compares `trading_day` to
   `micro_launch_day(instrument)` (which derives from the existing
   `minimum_start_date` field on micro configs). Only valid for micros —
   non-micros raise ValueError because the PARENT/MICRO distinction is
   meaningless for instruments that have no micro counterpart.

Fail-closed on every failure mode. Pure functions, no side effects.

Canonical authority (`.claude/rules/integrity-guardian.md` rule 2):
    - parent_symbol field → ASSET_CONFIGS[instrument]['parent_symbol']
    - outright_pattern    → ASSET_CONFIGS[instrument]['outright_pattern']
    - minimum_start_date  → ASSET_CONFIGS[instrument]['minimum_start_date']

Consumers (deferred to later stages):
    - Stage 3b: `trading_app.config.StrategyFilter.requires_micro_data` gate
    - Stage 3c: `orb_outcomes` / `daily_features` rebuild range filter
    - Stage 3d: `check_drift.check_era_discipline` new drift check
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pipeline.asset_configs import ASSET_CONFIGS, get_outright_root

DataEra = Literal["PARENT", "MICRO"]


def _lookup(instrument: str) -> dict:
    """Uppercase-and-lookup with consistent error message. Internal helper."""
    key = instrument.upper()
    cfg = ASSET_CONFIGS.get(key)
    if cfg is None:
        raise ValueError(
            f"Unknown instrument: {instrument!r}. "
            f"Supported: {sorted(ASSET_CONFIGS.keys())}"
        )
    return cfg


def is_micro(instrument: str) -> bool:
    """True iff `instrument` has REAL micro contract data in bars_1m.

    Distinguishes instruments with post-Phase-2 real-micro data from dead
    micros that still use parent-proxy data:

    - MGC, MNQ, MES (active):  True — outright_pattern matches native micro
      contracts (MGCH4, MNQZ5, etc), data IS real micro
    - M2K, MBT, MCL, M6E, SIL (dead):  False — outright_pattern matches
      parent contracts (RTYH4, BTCZ5, etc), data IS parent proxy
    - NQ, ES, GC (native parents):  False — instrument IS the parent
    - 2YY, ZT (research-only native):  False — no micro relationship

    The distinction between "contract is a micro at the exchange" and "data
    is real-micro in our DB" matters because downstream stages (3b/3c/3d)
    care about the DATA era, not the contract nomenclature. Volume-based
    filters that require real micro data should use `is_micro()` directly.

    Implementation: `parent_symbol` must be set AND the outright_pattern
    root must equal the symbol (not the parent). The latter condition
    distinguishes post-Phase-2 active micros (pattern=`^MGC...`) from
    dead micros (pattern=`^RTY...`).

    Raises:
        ValueError: unknown instrument
    """
    cfg = _lookup(instrument)
    parent = cfg.get("parent_symbol")
    if parent is None:
        return False
    pattern_root = get_outright_root(instrument)
    symbol = cfg["symbol"].upper()
    return pattern_root == symbol


def parent_for(instrument: str) -> str | None:
    """Return the parent contract symbol for a micro, or None otherwise.

    Examples:
        parent_for("MNQ") -> "NQ"
        parent_for("MGC") -> "GC"
        parent_for("M2K") -> "RTY"   # dead micro still uses RTY parent data
        parent_for("NQ")  -> None    # NQ IS the parent
        parent_for("2YY") -> None    # research-only native

    Raises:
        ValueError: unknown instrument
    """
    cfg = _lookup(instrument)
    return cfg.get("parent_symbol")


def micro_launch_day(instrument: str) -> date:
    """Return the date the real-micro contract series launched.

    Reads `minimum_start_date` from the canonical config — this field was
    set to the actual launch date for each active micro in Phase 2 of
    canonical-data-redownload (commit 82e8b60). Do NOT re-encode these
    dates in any downstream module.

    Only valid for instruments where `is_micro()` is True — i.e., active
    micros with REAL micro contract data (MGC/MNQ/MES post-Phase-2). Dead
    micros (M2K/MBT/MCL/M6E/SIL) use parent-proxy data and have no real
    micro launch day in bars_1m, so this function raises for them.

    Examples:
        micro_launch_day("MGC") -> date(2022, 6, 13)
        micro_launch_day("MNQ") -> date(2019, 5, 6)
        micro_launch_day("MES") -> date(2019, 5, 6)

    Raises:
        ValueError: unknown instrument OR instrument is not a real micro
            (is_micro() == False)
    """
    if not is_micro(instrument):  # raises ValueError on unknown
        raise ValueError(
            f"{instrument.upper()} is not a real micro (is_micro=False). "
            f"micro_launch_day only defined for instruments with real-micro "
            f"data in bars_1m — see pipeline.data_era.is_micro() semantics."
        )
    cfg = _lookup(instrument)
    launch = cfg.get("minimum_start_date")
    if launch is None:
        raise ValueError(
            f"{instrument.upper()} has no minimum_start_date — config gap"
        )
    return launch


def era_for_source_symbol(instrument: str, source_symbol: str) -> DataEra:
    """Classify a bars_1m row by its source_symbol (the original contract).

    Algorithm:
    1. If source_symbol matches the instrument's own outright_pattern:
       - Derive the pattern root via `get_outright_root(instrument)`
       - If `parent_symbol is None` → PARENT (native/parent instrument)
       - Else if pattern root == `symbol` (MGC/MNQ/MES post-Phase-2) → MICRO
         (pattern represents real-micro contracts, data is real micro)
       - Else if pattern root == `parent_symbol` (M2K→RTY, MBT→BTC etc) →
         PARENT (pattern represents parent contracts, data is parent proxy)
       - Else → ValueError (config inconsistency)
    2. Else if instrument is an active micro (pattern matches native micro
       contracts) AND source matches parent's outright_pattern:
       → PARENT (pre-Phase-2 corruption pattern — caller should treat as
       a contamination signal)
    3. Else → ValueError (source_symbol belongs to neither the instrument
       nor its parent)

    This function is both a classifier (normal case) and a corruption
    detector (case 2 — the exact pattern Phase 2 fixed). Stage 3d's drift
    check will use case 2 to scan `bars_1m` for residual contamination.

    Raises:
        ValueError: null/empty source_symbol, unknown instrument, or
            source_symbol matches neither canonical pattern
    """
    if source_symbol is None or source_symbol == "":
        raise ValueError(
            f"source_symbol is required for era classification (got {source_symbol!r})"
        )

    cfg = _lookup(instrument)
    own_pattern = cfg["outright_pattern"]

    if own_pattern.match(source_symbol):
        parent = cfg.get("parent_symbol")
        if parent is None:
            # Native/parent instrument (NQ/ES/GC/2YY/ZT) — all its data is PARENT
            return "PARENT"
        # Instrument has a parent. Is the outright_pattern pointing to the
        # instrument's own (micro) contracts or the parent's contracts?
        pattern_root = get_outright_root(instrument)
        symbol = cfg["symbol"].upper()
        if pattern_root == symbol:
            # Pattern matches native micro contracts (MGC/MNQ/MES post-Phase-2) →
            # data is real micro
            return "MICRO"
        if pattern_root == parent.upper():
            # Pattern matches parent contracts (M2K→RTY, MBT→BTC, etc) →
            # data is parent proxy
            return "PARENT"
        raise ValueError(
            f"Config inconsistency for {instrument.upper()}: outright_pattern root "
            f"{pattern_root!r} is neither symbol {symbol!r} nor parent {parent!r}"
        )

    # Didn't match own pattern — try parent's pattern for corruption detection.
    # Only makes sense for active micros whose own pattern matches MICRO contracts;
    # for dead micros whose own pattern IS the parent pattern, this branch is a
    # no-op (we already returned PARENT above).
    parent = cfg.get("parent_symbol")
    if parent is not None:
        parent_cfg = ASSET_CONFIGS.get(parent)
        if parent_cfg is not None and parent_cfg["outright_pattern"].match(source_symbol):
            return "PARENT"

    raise ValueError(
        f"source_symbol {source_symbol!r} does not match canonical pattern "
        f"for instrument {instrument.upper()!r} or its parent"
    )


def era_for_trading_day(instrument: str, trading_day: date) -> DataEra:
    """Classify an orb_outcomes / daily_features row by trading_day vs launch.

    Only valid for micros — the PARENT/MICRO distinction is meaningless for
    instruments without a parent relationship (callers must `is_micro()`
    first). For micros:
    - `trading_day >= micro_launch_day(instrument)` → MICRO
    - `trading_day <  micro_launch_day(instrument)` → PARENT

    Note: equal date is MICRO (launch day is inclusive) since any bar on
    the launch day is from the actual micro contract.

    Raises:
        ValueError: unknown instrument OR instrument is not a micro
    """
    launch = micro_launch_day(instrument)  # raises if not micro or unknown
    return "MICRO" if trading_day >= launch else "PARENT"
