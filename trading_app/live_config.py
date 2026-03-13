"""
Declarative live portfolio configuration.

Defines exactly what to trade based on rolling evaluation results:
- CORE tier: always-on strategies (STABLE families from rolling eval)
- REGIME tier: conditionally-gated strategies (fitness-checked before trading)

Usage:
    python -m trading_app.live_config --db-path C:/db/gold.db
    python -m trading_app.live_config --db-path C:/db/gold.db --output live_portfolio.json
"""

import logging
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

import duckdb

from pipeline.asset_configs import get_active_instruments
from pipeline.paths import GOLD_DB_PATH
from trading_app.portfolio import Portfolio, PortfolioStrategy
from trading_app.rolling_portfolio import (
    DEFAULT_LOOKBACK_WINDOWS,
    STABLE_THRESHOLD,
    load_rolling_validated_strategies,
)
from trading_app.strategy_fitness import compute_fitness

# =========================================================================
# Live portfolio specification
# =========================================================================


@dataclass(frozen=True)
class LiveStrategySpec:
    """Declarative specification for a live strategy family."""

    family_id: str  # e.g. "TOKYO_OPEN_E1_ORB_G4"
    tier: str  # "core", "regime", or "hot"
    orb_label: str
    entry_model: str
    filter_type: str
    regime_gate: str | None  # None = always-on, "high_vol" = fitness must be FIT
    # "rolling" = must be STABLE in recent rolling eval
    rr_target: float | None = None  # Resolved at build time from family_rr_locks.
    # None in spec = look up locked RR per instrument.
    exclude_instruments: frozenset[str] | None = None  # Per-instrument BH FDR exclusion.
    # Instruments in this set are skipped during build_live_portfolio.
    # Added Mar 2026: Bloomey audit identified spec-instrument combos
    # that fail BH FDR while other instruments on the same spec survive.
    # Seasonal gate: only trade in these calendar months (1=Jan..12=Dec).
    # None = all months. Outside active months -> weight=0, variant not loaded.
    active_months: frozenset[int] | None = None
    # Manual weight demotion (e.g. 0.5 for decay, 0.0 to disable).
    # None = use default (1.0 for core, fitness-dependent for regime).
    weight_override: float | None = None
    # Auto-recovery: if rolling eval ExpR exceeds this threshold,
    # ignore weight_override and promote back to weight=1.0.
    # None = no auto-recovery. Only fires when source="rolling".
    recovery_expr_threshold: float | None = None

    def __post_init__(self):
        if self.active_months is not None:
            if not self.active_months:
                raise ValueError("active_months must be non-empty if set")
            if not all(1 <= m <= 12 for m in self.active_months):
                raise ValueError("active_months must contain values 1-12")
        if self.weight_override is not None:
            if not 0.0 <= self.weight_override <= 1.0:
                raise ValueError("weight_override must be in [0.0, 1.0]")
        if self.recovery_expr_threshold is not None:
            if self.recovery_expr_threshold <= 0.0:
                raise ValueError("recovery_expr_threshold must be > 0.0")
            if self.weight_override is None:
                raise ValueError("recovery_expr_threshold requires weight_override to be set")


# Lookback for HOT tier rolling stability check (recent months only).
# @research-source: Pardo Ch.7 walk-forward methodology — 10 windows = ~10 months,
# sufficient to distinguish regime from noise in WF rolling evaluation.
# @revalidated-for: E2 event-based (dormant — HOT tier not yet wired to real-time)
HOT_LOOKBACK_WINDOWS = 10

# Minimum rolling stability score for HOT tier to be active.
# @research-source: Pardo Ch.7 — 60% of WF windows passing = robust across regimes.
# Below 0.6 indicates strategy works in <6/10 recent windows: not stable enough for live.
HOT_MIN_STABILITY = 0.6

# Minimum expectancy per trade to include in live portfolio.
# Filters out statistically-validated but practically-thin strategies.
# At 0.10 R, after execution uncertainty (spread variance, slippage noise),
# there is still meaningful edge remaining per trade.
# @research-source live_config calibration — 0.10 R practical edge floor after execution noise
# @revalidated-for E1/E2 event-based sessions (2026-03-12)
LIVE_MIN_EXPECTANCY_R = 0.10

# Minimum expected dollar profit as a multiple of round-trip transaction cost.
# Strategies must earn at least this multiple of their RT cost per trade so that
# execution variance (spread widening, extra slippage) cannot eliminate the edge.
#
# 1.3x rationale: kills strategies where net edge is barely above RT cost
# (MNQ TOKYO $2.93 = 1.07x, MNQ BRISBANE $2.83 = 1.03x — too thin to survive
# live execution uncertainty). Keeps real edge at 1.4x+ (MES strategies at $5-6,
# MNQ strategies at $4.82+). Tighter instruments (MNQ) have a proportionally
# tougher screen than larger ones (MGC) because slippage uncertainty is bigger
# relative to point value.
#
# Adjust if cost structure changes (new broker, exchange fee changes, tighter spreads).
# @research-source live_config calibration — 1.3x RT cost floor derived from MNQ/MES dollar analysis
# @revalidated-for E1/E2 event-based sessions (2026-03-12)
LIVE_MIN_EXPECTANCY_DOLLARS_MULT = 1.3

# The live portfolio: what we actually trade.
#
# TIER 1 (CORE): Always on. Full-period validated (FDR+WF). Target N>=100;
#   REGIME-N variants may be loaded when no CORE-N variant exists (warning logged).
#   Edge family classification informational — PURGED/SINGLETON families may be present.
#   Specs are instrument-agnostic — pass --instrument to select.
#   If a spec has no match for an instrument, it emits WARN and skips.
#
# TIER 2 (REGIME): Fitness-gated. Validated but N<100 (REGIME classification).
#   Only fires when strategy_fitness = FIT. regime_gate="high_vol".
#
# HOT tier removed — rolling_portfolio evaluation not yet wired to real-time
# gate logic. Re-add when rolling eval pipeline is production-ready.
#
# Updated 2026-03-12: Added SINGAPORE_OPEN VOL_RV12_N20 (genuinely independent filter type).
#   Reverted 3 redundant ORB_G variants (NYSE_OPEN G6/G5, US_DATA_1000 G8) — 99%+ day overlap with existing specs.
#   30 unique specs (instrument-agnostic — filter_type is the base name, without _O15/_O30 aperture suffix).
#   E0 fully purged. E2 dominant. E3 soft-retired.
#   Filter_type note: strategy IDs encode aperture as _O15/_O30 suffix (e.g. VOL_RV12_N20_O15),
#   but validated_setups.filter_type stores the base name (e.g. VOL_RV12_N20). Always use base name here.
#   Dollar gate: LIVE_MIN_EXPECTANCY_DOLLARS_MULT (1.3x RT cost) applied on top of ExpR >= 0.10.
#   Strategies that pass R gate but fail dollar gate are SKIP'd with a note (adjust mult if needed).
#
# Active strategies per instrument may be fewer than spec count if dollar gate fires.
# Run `python -m trading_app.live_config --instrument X` to see current actives + SKIP notes.
LIVE_PORTFOLIO = [
    # =========================================================================
    # CORE: always-on, full-period validated, N>=100
    # =========================================================================
    # CME_REOPEN: MGC-dominant session. Only MGC has ROBUST families (ORB_G4_FAST10, ORB_G5_FAST10).
    #   ORB_G5 and VOL_RV12_N20 specs load from validated_setups but ALL families are PURGED/SINGLETON.
    #   ORB_G4_FAST10: MGC 5-member ROBUST, PBO=0.0, head ExpR=0.244
    LiveStrategySpec("CME_REOPEN_E2_ORB_G5", "core", "CME_REOPEN", "E2", "ORB_G5", None),
    LiveStrategySpec("CME_REOPEN_E2_VOL_RV12_N20", "core", "CME_REOPEN", "E2", "VOL_RV12_N20", None),
    # Adversarial audit 2026-03-13: SUSPICIOUS. Seasonal Q4+Q1 clustering (68% wins Nov-Feb),
    # cost fragility at G4 minimum. No recovery — seasonal gate is a hard skip.
    # No academic support for month-based gating (purely empirical — held to higher bar).
    # BH FDR exclusion: MNQ p_adj > 0.05; MGC survives. @research-source bloomey_audit 2026-03-12
    LiveStrategySpec(
        "CME_REOPEN_E2_ORB_G4_FAST10",
        "core",
        "CME_REOPEN",
        "E2",
        "ORB_G4_FAST10",
        None,
        exclude_instruments=frozenset({"MNQ"}),
        active_months=frozenset({11, 12, 1, 2}),
    ),
    # CME_PRECLOSE: MES wins with ORB_G6; MNQ wins with VOL_RV12_N20
    #   ORB_G5: MES 15m aperture, independent day-add vs G6
    LiveStrategySpec("CME_PRECLOSE_E2_ORB_G6", "core", "CME_PRECLOSE", "E2", "ORB_G6", None),
    # Adversarial audit 2026-03-13: LEGIT BUT DECAYING. MNQ head avg_r/yr: 0.35→0.32→0.20→0.16 (2022-2025).
    # Half weight (Chan half-Kelly under parameter uncertainty) until rolling avg ExpR > 0.25.
    # Man AHL: recovery requires same gates, not lower bar.
    LiveStrategySpec(
        "CME_PRECLOSE_E2_VOL_RV12_N20",
        "core",
        "CME_PRECLOSE",
        "E2",
        "VOL_RV12_N20",
        None,
        weight_override=0.5,
        recovery_expr_threshold=0.25,
    ),
    LiveStrategySpec("CME_PRECLOSE_E2_ORB_G5", "core", "CME_PRECLOSE", "E2", "ORB_G5", None),
    # COMEX_SETTLE: MES wins with ORB_G6; MNQ wins with VOL_RV12_N20
    # BH FDR exclusion: MGC and M2K p_adj > 0.05; MES and MNQ survive.
    # @research-source bloomey_audit 2026-03-12
    LiveStrategySpec(
        "COMEX_SETTLE_E2_ORB_G6",
        "core",
        "COMEX_SETTLE",
        "E2",
        "ORB_G6",
        None,
        exclude_instruments=frozenset({"MGC", "M2K"}),
    ),
    LiveStrategySpec("COMEX_SETTLE_E2_VOL_RV12_N20", "core", "COMEX_SETTLE", "E2", "VOL_RV12_N20", None),
    # NYSE_CLOSE: MES ORB_G4; MNQ VOL_RV12_N20. WARNING: ALL families PURGED/SINGLETON.
    #   Strategies load from validated_setups via RR-locks, but no family passes PBO scrutiny.
    #   MES ORB_G4: N=388, ExpR=0.118. MNQ VOL_RV12_N20: N=337, ExpR=0.212.
    LiveStrategySpec("NYSE_CLOSE_E2_ORB_G4", "core", "NYSE_CLOSE", "E2", "ORB_G4", None),
    LiveStrategySpec("NYSE_CLOSE_E2_VOL_RV12_N20", "core", "NYSE_CLOSE", "E2", "VOL_RV12_N20", None),
    # NYSE_OPEN: MES wins VOL_RV12_N20; MNQ wins ORB_G4; M2K VOL_RV12_N20
    #   ORB_G8: MES 8-member ROBUST family, head ExpR=0.080, independent day-add
    LiveStrategySpec("NYSE_OPEN_E2_VOL_RV12_N20", "core", "NYSE_OPEN", "E2", "VOL_RV12_N20", None),
    LiveStrategySpec("NYSE_OPEN_E2_ORB_G4", "core", "NYSE_OPEN", "E2", "ORB_G4", None),
    LiveStrategySpec("NYSE_OPEN_E2_ORB_G8", "core", "NYSE_OPEN", "E2", "ORB_G8", None),
    # US_DATA_1000: M2K wins VOL_RV12_N20; MNQ wins ORB_G5.
    #   ORB_G6: MNQ 6-member ROBUST (head ExpR=0.132). MGC has FDR+WF (N=317) but SINGLETON family.
    LiveStrategySpec("US_DATA_1000_E2_VOL_RV12_N20", "core", "US_DATA_1000", "E2", "VOL_RV12_N20", None),
    LiveStrategySpec("US_DATA_1000_E2_ORB_G5", "core", "US_DATA_1000", "E2", "ORB_G5", None),
    LiveStrategySpec("US_DATA_1000_E2_ORB_G6", "core", "US_DATA_1000", "E2", "ORB_G6", None),
    # TOKYO_OPEN: Multiple independent edges verified by family hash:
    #   ORB_G5_CONT — MGC 5m N=115 CORE (9-member ROBUST), MNQ CORE
    #   ORB_G5 — MGC 15m N=233+ FDR+WF, genuinely independent
    #   ORB_G6_CONT — MES N=227, FDR+WF
    LiveStrategySpec("TOKYO_OPEN_E2_ORB_G5_CONT", "core", "TOKYO_OPEN", "E2", "ORB_G5_CONT", None),
    # BH FDR exclusion: MES p_adj > 0.05; MGC and MNQ survive. @research-source bloomey_audit 2026-03-12
    LiveStrategySpec(
        "TOKYO_OPEN_E2_ORB_G5", "core", "TOKYO_OPEN", "E2", "ORB_G5", None, exclude_instruments=frozenset({"MES"})
    ),
    LiveStrategySpec("TOKYO_OPEN_E2_ORB_G6_CONT", "core", "TOKYO_OPEN", "E2", "ORB_G6_CONT", None),
    # SINGAPORE_OPEN: DIR_LONG (long-only, any ORB size) is the primary filter for MNQ.
    #   Research H5: shorts systematically negative; long side positive.
    #   DIR_LONG E2 30m head: N=686, ExpR=0.211, Sharpe_ann=1.55, FDR=True, WF=True.
    #   E2 strategies share family hash with E1 ROBUST family (17-member, PBO=0.0).
    #   ORB_G8: 15m 10-member ROBUST family head N=1111, ExpR=0.091, PBO=0.057.
    LiveStrategySpec("SINGAPORE_OPEN_E2_DIR_LONG", "core", "SINGAPORE_OPEN", "E2", "DIR_LONG", None),
    LiveStrategySpec("SINGAPORE_OPEN_E2_ORB_G8", "core", "SINGAPORE_OPEN", "E2", "ORB_G8", None),
    #   VOL_RV12_N20: MNQ 30m 12-member ROBUST family, head N=1024, ExpR=0.112, PBO=0.0
    LiveStrategySpec("SINGAPORE_OPEN_E2_VOL_RV12_N20", "core", "SINGAPORE_OPEN", "E2", "VOL_RV12_N20", None),
    # BRISBANE_1025: MNQ wins ORB_G6
    #   VOL_RV12_N20: MNQ 5m 9-member ROBUST, head ExpR=0.113, PBO=0.0
    LiveStrategySpec("BRISBANE_1025_E2_ORB_G6", "core", "BRISBANE_1025", "E2", "ORB_G6", None),
    LiveStrategySpec("BRISBANE_1025_E2_VOL_RV12_N20", "core", "BRISBANE_1025", "E2", "VOL_RV12_N20", None),
    # LONDON_METALS: MNQ wins VOL_RV12_N20 (5m, 8-member ROBUST, head ExpR=0.123) and
    #   ORB_G6_NOMON (15m head: N=1050, FDR=True, WF=True, ExpR=0.098, 3-member WHITELISTED).
    #   M2K is REGIME below (ORB_G6_CONT).
    LiveStrategySpec("LONDON_METALS_E2_VOL_RV12_N20", "core", "LONDON_METALS", "E2", "VOL_RV12_N20", None),
    # BH FDR exclusion: M2K p_adj > 0.05, high PBO; MNQ survives.
    # @research-source bloomey_audit 2026-03-12
    LiveStrategySpec(
        "LONDON_METALS_E2_ORB_G6_NOMON",
        "core",
        "LONDON_METALS",
        "E2",
        "ORB_G6_NOMON",
        None,
        exclude_instruments=frozenset({"M2K"}),
    ),
    # EUROPE_FLOW (17:00 winter / 18:00 summer Brisbane): Added 2026-03-13.
    #   MNQ: 5 years positive (2021-2025), 2026 Q1 weak (-0.02 to -0.09R on 35-45 trades)
    #   but insufficient sample to call decay. 100% WF pass, 41% FDR significant.
    #   VOL_RV12_N20: family head RR1.5 CB1: N=774, ExpR=0.117, Sharpe_ann=1.50, 7-member ROBUST, PBO=0.000
    #   ORB_G8: family head RR1.5 CB1: N=1117, ExpR=0.121, Sharpe_ann=1.56, 9-member ROBUST, PBO=0.029
    #   M2K excluded (PURGED + 0% dollar gate pass). MES excluded (zero validated).
    #   @research-source EUROPE_FLOW deep audit 2026-03-13
    LiveStrategySpec("EUROPE_FLOW_E2_VOL_RV12_N20", "core", "EUROPE_FLOW", "E2", "VOL_RV12_N20", None),
    LiveStrategySpec("EUROPE_FLOW_E2_ORB_G8", "core", "EUROPE_FLOW", "E2", "ORB_G8", None),
    # =========================================================================
    # REGIME: fitness-gated (N<100 — only trade when strategy_fitness = FIT)
    # =========================================================================
    # M2K LONDON_METALS: N=59, REGIME. ORB_G6_CONT filter (different from MNQ's VOL_RV12_N20_O15 above)
    LiveStrategySpec("LONDON_METALS_E2_ORB_G6_CONT", "regime", "LONDON_METALS", "E2", "ORB_G6_CONT", "high_vol"),
    # MGC EUROPE_FLOW: N=147 but ~20-25 trades/yr in active years, dead years (2017-2019, 2023).
    #   Institutionally regime-grade despite technically crossing CORE threshold.
    #   ORB_G4: ROBUST family, 2026 positive (+0.30R), all dollar gate pass (>1.84x).
    #   MNQ/MES/M2K excluded (MNQ no MGC-level edge, MES zero validated, M2K PURGED).
    #   @research-source EUROPE_FLOW deep audit 2026-03-13
    LiveStrategySpec(
        "EUROPE_FLOW_E2_ORB_G4",
        "regime",
        "EUROPE_FLOW",
        "E2",
        "ORB_G4",
        "high_vol",
        exclude_instruments=frozenset({"MNQ", "MES", "M2K"}),
    ),
    # NOTE: MGC TOKYO_OPEN ORB_G5_CONT loads highest-ExpR FDR-preferred variant (varies by rebuild).
    # Monitor via get_strategy_fitness if conditional gating is needed.
    # MGC TOKYO_OPEN E1: limit entry at ORB edge (vs E2 stop-market above).
    #   ORB_G5_FAST10: 30m aperture, N=151, ExpR=0.270, $24/trade, 6-member ROBUST CORE family.
    #   Previously blocked by PBO=1.0 (filter-blind CSCV). Filter-aware PBO=0.000.
    #   MGC-only — E1 not validated for MNQ/MES/M2K at this filter.
    #   @research-source PBO_FILTER_BLINDNESS.md 2026-03-13
    LiveStrategySpec(
        "TOKYO_OPEN_E1_ORB_G5_FAST10",
        "core",
        "TOKYO_OPEN",
        "E1",
        "ORB_G5_FAST10",
        None,
        exclude_instruments=frozenset({"MNQ", "MES", "M2K"}),
    ),
    # MGC SINGAPORE_OPEN: ORB_G5 adds lower-threshold days vs existing ORB_G8.
    #   N=269, ExpR=0.159, $12.27/trade, 5-member ROBUST CORE family.
    #   Previously blocked by PBO=1.0. Filter-aware PBO=0.000.
    #   MGC-only — MNQ already has SINGAPORE_OPEN via DIR_LONG and ORB_G8.
    #   @research-source PBO_FILTER_BLINDNESS.md 2026-03-13
    LiveStrategySpec(
        "SINGAPORE_OPEN_E2_ORB_G5",
        "core",
        "SINGAPORE_OPEN",
        "E2",
        "ORB_G5",
        None,
        exclude_instruments=frozenset({"MNQ", "MES", "M2K"}),
    ),
]

# =========================================================================
# Portfolio builder
# =========================================================================


def _load_best_regime_variant(
    db_path: Path,
    instrument: str,
    orb_label: str,
    entry_model: str,
    filter_type: str,
    min_expectancy_r: float = LIVE_MIN_EXPECTANCY_R,
) -> dict | None:
    """Load the best variant from validated_setups, enforcing locked RR.

    Joins family_rr_locks to restrict each (instrument, orb_label, filter_type,
    entry_model, orb_minutes, confirm_bars) to its JK-MaxSharpe-locked RR target.
    Among matching rows, picks the best by FDR significance (primary), then
    expectancy_r (tiebreaker across different orb_minutes/confirm_bars combos).

    Only returns strategies with expectancy_r >= min_expectancy_r.
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(
            """
            SELECT vs.strategy_id, vs.instrument, vs.orb_label, vs.entry_model,
                   vs.rr_target, vs.confirm_bars, vs.filter_type,
                   vs.orb_minutes,
                   vs.expectancy_r, vs.win_rate, vs.sample_size,
                   vs.sharpe_ratio, vs.max_drawdown_r,
                   vs.fdr_significant,
                   es.median_risk_points,
                   1.0 as stop_multiplier
            FROM validated_setups vs
            LEFT JOIN experimental_strategies es
              ON vs.strategy_id = es.strategy_id
            INNER JOIN family_rr_locks frl
              ON vs.instrument = frl.instrument
              AND vs.orb_label = frl.orb_label
              AND vs.filter_type = frl.filter_type
              AND vs.entry_model = frl.entry_model
              AND vs.orb_minutes = frl.orb_minutes
              AND vs.confirm_bars = frl.confirm_bars
              AND vs.rr_target = frl.locked_rr
            WHERE vs.instrument = ?
              AND vs.orb_label = ?
              AND vs.entry_model = ?
              AND vs.filter_type = ?
              AND LOWER(vs.status) = 'active'
              AND vs.expectancy_r >= ?
            ORDER BY vs.fdr_significant DESC NULLS LAST, vs.expectancy_r DESC NULLS LAST
            LIMIT 1
        """,
            [instrument, orb_label, entry_model, filter_type, min_expectancy_r],
        ).fetchall()

        if not rows:
            return None

        cols = [desc[0] for desc in con.description]
        return dict(zip(cols, rows[0], strict=False))
    finally:
        con.close()


def _load_best_experimental_variant(
    db_path: Path,
    instrument: str,
    orb_label: str,
    entry_model: str,
    filter_type: str,
) -> dict | None:
    """Load the best RR/CB variant from experimental_strategies.

    Used for HOT tier families that haven't passed full-period validation
    but are STABLE in recent rolling windows.

    NOTE: Intentionally NOT RR-locked via family_rr_locks. The HOT tier
    is dormant (no families currently use it) and experimental_strategies
    is a separate table from validated_setups. When HOT tier is activated,
    this should be revisited to enforce RR locks.
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(
            """
            SELECT strategy_id, instrument, orb_label, entry_model,
                   rr_target, confirm_bars, filter_type,
                   orb_minutes,
                   expectancy_r, win_rate, sample_size,
                   sharpe_ratio, max_drawdown_r,
                   median_risk_points,
                   1.0 as stop_multiplier
            FROM experimental_strategies
            WHERE instrument = ?
              AND orb_label = ?
              AND entry_model = ?
              AND filter_type = ?
              AND expectancy_r > 0
            ORDER BY expectancy_r DESC NULLS LAST
            LIMIT 1
        """,
            [instrument, orb_label, entry_model, filter_type],
        ).fetchall()

        if not rows:
            return None

        cols = [desc[0] for desc in con.description]
        return dict(zip(cols, rows[0], strict=False))
    finally:
        con.close()


def _check_rolling_stability(
    db_path: Path,
    instrument: str,
    orb_label: str,
    entry_model: str,
    filter_type: str,
    train_months: int = 12,
    lookback_windows: int = HOT_LOOKBACK_WINDOWS,
    min_stability: float = HOT_MIN_STABILITY,
) -> tuple[float, str]:
    """Check rolling stability for a family over recent windows.

    Returns (stability_score, note_string).
    """
    from trading_app.rolling_portfolio import (
        aggregate_rolling_performance,
        load_all_rolling_run_labels,
        load_rolling_degraded_counts,
        load_rolling_results,
        make_family_id,
    )

    all_labels = load_all_rolling_run_labels(db_path, train_months, instrument, lookback_windows)
    if not all_labels:
        return 0.0, "no rolling windows found"

    validated = load_rolling_results(db_path, train_months, instrument, run_labels=all_labels)
    degraded = load_rolling_degraded_counts(db_path, train_months, instrument, run_labels=all_labels)
    families = aggregate_rolling_performance(validated, all_labels, degraded)

    target_fid = make_family_id(orb_label, entry_model, filter_type)
    for fam in families:
        if fam.family_id == target_fid:
            if fam.weighted_stability >= min_stability:
                return fam.weighted_stability, (
                    f"STABLE ({fam.weighted_stability:.3f}, {fam.windows_passed}/{fam.windows_total} windows)"
                )
            else:
                return fam.weighted_stability, (
                    f"NOT STABLE ({fam.weighted_stability:.3f}, {fam.windows_passed}/{fam.windows_total} windows)"
                )

    return 0.0, "family not found in rolling results"


def _check_dollar_gate(variant: dict, instrument: str) -> tuple[bool, str]:
    """Check that expected dollar profit >= LIVE_MIN_EXPECTANCY_DOLLARS_MULT * RT cost.

    Returns (passes, note). If median_risk_points is unavailable, BLOCKS
    (returns False) — unknown cost adequacy must not allow trading.
    If get_cost_spec() raises, also BLOCKS — a broken cost model must not
    allow trading.
    """
    median_risk_pts = variant.get("median_risk_points")
    if median_risk_pts is None:
        log.warning(
            "dollar gate BLOCKED %s — median_risk_points is NULL (cannot verify cost adequacy)",
            variant.get("strategy_id", "unknown"),
        )
        return False, "dollar gate blocked (no median_risk_points)"
    try:
        from pipeline.cost_model import get_cost_spec

        spec = get_cost_spec(instrument)
        one_r_dollars = median_risk_pts * spec.point_value
        exp_dollars = variant["expectancy_r"] * one_r_dollars
        min_dollars = LIVE_MIN_EXPECTANCY_DOLLARS_MULT * spec.total_friction
        if exp_dollars < min_dollars:
            return False, (
                f"Exp${exp_dollars:.2f} < {LIVE_MIN_EXPECTANCY_DOLLARS_MULT}x "
                f"RT cost (${spec.total_friction:.2f} * {LIVE_MIN_EXPECTANCY_DOLLARS_MULT} = ${min_dollars:.2f})"
            )
        return True, f"Exp${exp_dollars:.2f} >= ${min_dollars:.2f} ({LIVE_MIN_EXPECTANCY_DOLLARS_MULT}x RT)"
    except Exception as exc:
        return False, f"dollar gate BLOCKED (cost spec unavailable: {exc})"


def build_live_portfolio(
    db_path: Path | None = None,
    instrument: str = "MGC",
    rolling_train_months: int = 12,
    account_equity: float = 25000.0,
    risk_per_trade_pct: float = 2.0,
    min_expectancy_r: float = LIVE_MIN_EXPECTANCY_R,
    as_of_date: date | None = None,
) -> tuple[Portfolio, list[str]]:
    """
    Build the live portfolio from LIVE_PORTFOLIO spec.

    Core tier: loads best variant from rolling eval (most recent window).
    Regime tier: loads best variant from validated_setups, then checks
    strategy_fitness -- weight=0.0 if not FIT.

    as_of_date: Reference date for seasonal gating. Defaults to today.
    """
    if as_of_date is None:
        as_of_date = date.today()

    valid_instruments = set(get_active_instruments())
    if instrument not in valid_instruments:
        raise ValueError(f"Unknown instrument '{instrument}'. Valid: {sorted(valid_instruments)}")

    if db_path is None:
        db_path = GOLD_DB_PATH

    strategies = []
    notes = []

    # --- CORE tier: try rolling validated first, fall back to validated_setups ---
    core_specs = [s for s in LIVE_PORTFOLIO if s.tier == "core"]
    if core_specs:
        rolling_strats = load_rolling_validated_strategies(
            db_path,
            instrument,
            rolling_train_months,
            min_weighted_score=STABLE_THRESHOLD,
            min_expectancy_r=min_expectancy_r,
            lookback_windows=DEFAULT_LOOKBACK_WINDOWS,
        )

        for spec in core_specs:
            if spec.exclude_instruments and instrument in spec.exclude_instruments:
                notes.append(f"SKIP: {spec.family_id} -- {instrument} excluded (BH FDR)")
                continue

            # --- Seasonal gate (hard skip — no variant loaded) ---
            if spec.active_months is not None:
                if as_of_date.month not in spec.active_months:
                    notes.append(
                        f"SEASONAL: {spec.family_id} -- month {as_of_date.month} not in {sorted(spec.active_months)}"
                    )
                    continue

            # Try rolling eval first
            match = None
            source = "rolling"
            for rs in rolling_strats:
                if (
                    rs["orb_label"] == spec.orb_label
                    and rs["entry_model"] == spec.entry_model
                    and rs["filter_type"] == spec.filter_type
                ):
                    match = rs
                    break

            # Fall back to validated_setups (best Sharpe variant meeting quality floor)
            if match is None:
                match = _load_best_regime_variant(
                    db_path,
                    instrument,
                    spec.orb_label,
                    spec.entry_model,
                    spec.filter_type,
                    min_expectancy_r=min_expectancy_r,
                )
                source = "baseline"

            if match is None:
                notes.append(f"WARN: {spec.family_id} -- no variant found")
                continue

            passes_dollar, dollar_note = _check_dollar_gate(match, instrument)
            if not passes_dollar:
                notes.append(f"SKIP: {spec.family_id} -- dollar gate failed: {dollar_note}")
                continue

            # --- Weight resolution (Carver forecast scaling / Chan half-Kelly) ---
            weight = 1.0
            weight_note = ""
            if spec.weight_override is not None:
                weight = spec.weight_override
                weight_note = f"DEMOTED (weight={weight})"

                # Auto-recovery: family rolling avg ExpR shows edge recovered?
                if (
                    spec.recovery_expr_threshold is not None
                    and source == "rolling"
                    and match.get("rolling_avg_expectancy_r", 0.0) >= spec.recovery_expr_threshold
                ):
                    weight = 1.0
                    rolling_avg = match["rolling_avg_expectancy_r"]
                    weight_note = f"RECOVERED (rolling_avg_ExpR={rolling_avg:+.3f} >= {spec.recovery_expr_threshold})"

            strategies.append(
                PortfolioStrategy(
                    strategy_id=match["strategy_id"],
                    instrument=match["instrument"],
                    orb_label=match["orb_label"],
                    entry_model=match["entry_model"],
                    rr_target=match["rr_target"],
                    confirm_bars=match["confirm_bars"],
                    filter_type=match["filter_type"],
                    orb_minutes=match.get("orb_minutes", 5),
                    expectancy_r=match["expectancy_r"],
                    win_rate=match["win_rate"],
                    sample_size=match["sample_size"],
                    sharpe_ratio=match.get("sharpe_ratio"),
                    max_drawdown_r=match.get("max_drawdown_r"),
                    median_risk_points=match.get("median_risk_points"),
                    stop_multiplier=match.get("stop_multiplier", 1.0),
                    source=source,
                    weight=weight,
                )
            )
            notes.append(
                f"CORE: {spec.family_id} -> {match['strategy_id']} "
                f"(ExpR={match['expectancy_r']:+.3f}, source={source}, "
                f"{dollar_note}, weight={weight}"
                f"{', ' + weight_note if weight_note else ''})"
            )

    # --- HOT tier: from experimental_strategies + rolling stability gate ---
    # Note: dollar gate is NOT applied to HOT tier. HOT strategies load from
    # experimental_strategies which rarely has median_risk_points populated, so
    # the gate would trivially pass-through anyway. Re-evaluate if HOT tier is
    # re-activated with strategies that have median_risk_points available.
    hot_specs = [s for s in LIVE_PORTFOLIO if s.tier == "hot"]
    for spec in hot_specs:
        if spec.exclude_instruments and instrument in spec.exclude_instruments:
            notes.append(f"SKIP: {spec.family_id} -- {instrument} excluded (BH FDR)")
            continue

        variant = _load_best_experimental_variant(
            db_path,
            instrument,
            spec.orb_label,
            spec.entry_model,
            spec.filter_type,
        )

        if variant is None:
            notes.append(f"WARN: {spec.family_id} -- no experimental variant with positive ExpR")
            continue

        # Check rolling stability gate
        weight = 1.0
        stability_note = "no gate"
        if spec.regime_gate == "rolling":
            stability_score, stability_note = _check_rolling_stability(
                db_path,
                instrument,
                spec.orb_label,
                spec.entry_model,
                spec.filter_type,
                train_months=rolling_train_months,
            )
            if stability_score < HOT_MIN_STABILITY:
                weight = 0.0
                stability_note = f"GATED OFF ({stability_note})"
            else:
                stability_note = f"ACTIVE ({stability_note})"

        strategies.append(
            PortfolioStrategy(
                strategy_id=variant["strategy_id"],
                instrument=variant["instrument"],
                orb_label=variant["orb_label"],
                entry_model=variant["entry_model"],
                rr_target=variant["rr_target"],
                confirm_bars=variant["confirm_bars"],
                filter_type=variant["filter_type"],
                orb_minutes=variant.get("orb_minutes", 5),
                expectancy_r=variant["expectancy_r"],
                win_rate=variant["win_rate"],
                sample_size=variant["sample_size"],
                sharpe_ratio=variant.get("sharpe_ratio"),
                max_drawdown_r=variant.get("max_drawdown_r"),
                median_risk_points=variant.get("median_risk_points"),
                stop_multiplier=variant.get("stop_multiplier", 1.0),
                source="hot_rolling",
                weight=weight,
            )
        )
        notes.append(
            f"HOT: {spec.family_id} -> {variant['strategy_id']} "
            f"(ExpR={variant['expectancy_r']:+.3f}, weight={weight}, {stability_note})"
        )

    # --- REGIME tier: from validated_setups + fitness gate ---
    regime_specs = [s for s in LIVE_PORTFOLIO if s.tier == "regime"]
    for spec in regime_specs:
        if spec.exclude_instruments and instrument in spec.exclude_instruments:
            notes.append(f"SKIP: {spec.family_id} -- {instrument} excluded (BH FDR)")
            continue

        variant = _load_best_regime_variant(
            db_path,
            instrument,
            spec.orb_label,
            spec.entry_model,
            spec.filter_type,
            min_expectancy_r=min_expectancy_r,
        )

        if variant is None:
            notes.append(f"WARN: {spec.family_id} -- no validated variant found")
            continue

        # Dollar gate first — no point running compute_fitness (DB query) on a
        # strategy that will be excluded for being too thin anyway.
        passes_dollar, dollar_note = _check_dollar_gate(variant, instrument)
        if not passes_dollar:
            notes.append(f"SKIP: {spec.family_id} -- dollar gate failed: {dollar_note}")
            continue

        # Check fitness gate
        weight = 1.0
        fitness_note = "no gate"
        if spec.regime_gate == "high_vol":
            try:
                fitness = compute_fitness(
                    variant["strategy_id"],
                    db_path=db_path,
                )
                if fitness.fitness_status != "FIT":
                    weight = 0.0
                    fitness_note = f"GATED OFF ({fitness.fitness_status}: {fitness.fitness_notes})"
                else:
                    fitness_note = "FIT -- gate OPEN"
            except (ValueError, duckdb.Error) as e:
                weight = 0.0
                fitness_note = f"GATED OFF (fitness error: {e})"

        strategies.append(
            PortfolioStrategy(
                strategy_id=variant["strategy_id"],
                instrument=variant["instrument"],
                orb_label=variant["orb_label"],
                entry_model=variant["entry_model"],
                rr_target=variant["rr_target"],
                confirm_bars=variant["confirm_bars"],
                filter_type=variant["filter_type"],
                orb_minutes=variant.get("orb_minutes", 5),
                expectancy_r=variant["expectancy_r"],
                win_rate=variant["win_rate"],
                sample_size=variant["sample_size"],
                sharpe_ratio=variant.get("sharpe_ratio"),
                max_drawdown_r=variant.get("max_drawdown_r"),
                median_risk_points=variant.get("median_risk_points"),
                stop_multiplier=variant.get("stop_multiplier", 1.0),
                source="baseline",
                weight=weight,
            )
        )
        notes.append(
            f"REGIME: {spec.family_id} -> {variant['strategy_id']} "
            f"(ExpR={variant['expectancy_r']:+.3f}, weight={weight}, {fitness_note})"
        )

    return Portfolio(
        name="live",
        instrument=instrument,
        strategies=strategies,
        account_equity=account_equity,
        risk_per_trade_pct=risk_per_trade_pct,
        max_concurrent_positions=3,
        max_daily_loss_r=5.0,
    ), notes


# =========================================================================
# CLI
# =========================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build live portfolio from declarative config")
    parser.add_argument("--db-path", type=Path, default=None, help="Path to gold.db")
    parser.add_argument("--instrument", default="MGC", choices=get_active_instruments())
    parser.add_argument("--rolling-train-months", type=int, default=12)
    parser.add_argument(
        "--min-expectancy-r",
        type=float,
        default=LIVE_MIN_EXPECTANCY_R,
        help=f"Min ExpR per trade to include (default {LIVE_MIN_EXPECTANCY_R})",
    )
    parser.add_argument("--output", type=str, default=None, help="Write portfolio JSON to this path")
    args = parser.parse_args()

    db_path = args.db_path or GOLD_DB_PATH

    print("Building live portfolio...")
    print(f"  DB: {db_path}")
    print(f"  Rolling window: {args.rolling_train_months}m")
    print()

    portfolio, notes = build_live_portfolio(
        db_path=db_path,
        instrument=args.instrument,
        rolling_train_months=args.rolling_train_months,
        min_expectancy_r=args.min_expectancy_r,
    )

    # Print strategy details
    print(f"{'=' * 70}")
    print("LIVE PORTFOLIO")
    print(f"{'=' * 70}")
    for note in notes:
        print(f"  {note}")
    print()

    from pipeline.cost_model import get_cost_spec

    active = [s for s in portfolio.strategies if s.weight >= 1.0]
    demoted = [s for s in portfolio.strategies if 0 < s.weight < 1]
    gated = [s for s in portfolio.strategies if s.weight == 0]

    def _exp_dollars(s) -> str:
        """Expected net dollar profit per trade.

        ExpR is already net of costs; 1R in dollars is the stop distance
        in dollar terms: 1R$ = median_risk_pts * point_value.
        Exp$ = ExpR * 1R$
        """
        if s.median_risk_points is None:
            return "   n/a"
        try:
            spec = get_cost_spec(s.instrument)
            one_r_dollars = s.median_risk_points * spec.point_value
            d = s.expectancy_r * one_r_dollars
            return f"${d:+6.2f}"
        except Exception:
            return "   n/a"

    print(f"Active strategies: {len(active)}")
    print(f"  {'Strategy':<50} {'ExpR':>6}  {'Exp$/trade':>10}  {'WR':>5}  {'N':>5}")
    print(f"  {'-' * 50} {'------':>6}  {'----------':>10}  {'-----':>5}  {'-----':>5}")
    for s in active:
        print(
            f"  {s.strategy_id:<50} {s.expectancy_r:>+6.3f}  {_exp_dollars(s):>10}  "
            f"{s.win_rate:>4.0%}  {s.sample_size:>5}"
        )

    if demoted:
        print(f"\nDemoted strategies (0 < weight < 1): {len(demoted)}")
        print(f"  {'Strategy':<50} {'Weight':>6}  {'ExpR':>6}  {'Exp$/trade':>10}")
        print(f"  {'-' * 50} {'------':>6}  {'------':>6}  {'----------':>10}")
        for s in demoted:
            print(f"  {s.strategy_id:<50} {s.weight:>6.2f}  {s.expectancy_r:>+6.3f}  {_exp_dollars(s):>10}")

    if gated:
        print(f"\nGated OFF (weight=0): {len(gated)}")
        for s in gated:
            print(f"  {s.strategy_id}: regime gate closed")

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(portfolio.to_json())
        print(f"\nWritten to {output_path}")


if __name__ == "__main__":
    main()
