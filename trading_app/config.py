"""
Strategy configuration: filters, entry models, and grid parameters.

==========================================================================
TERMINOLOGY & DEFINITIONS
==========================================================================

ORB (Opening Range Breakout):
  The high-low range formed in the first N minutes after a session opens.
  Duration is session-specific (see ORB_DURATION_MINUTES below).
  When price breaks above the ORB high or below the ORB low, it signals
  a potential directional move. All times are Australia/Brisbane (UTC+10).

ORB SESSIONS (defined in pipeline/init_db.py as ORB_LABELS):
  CME_REOPEN     - CME Globex electronic reopen 5:00 PM CT. Primary ORB session.
  TOKYO_OPEN     - Tokyo Stock Exchange open 9:00 AM JST. 15m ORB. Strong long bias.
  SINGAPORE_OPEN - SGX/HKEX open 9:00 AM SGT. MGC excluded (74% double-break). MNQ active (cross-market flow).
  LONDON_METALS  - London metals AM session 8:00 AM London. Best with E3 retrace.
  US_DATA_830    - US economic data release 8:30 AM ET.
  NYSE_OPEN      - NYSE cash open 9:30 AM ET.
  US_DATA_1000   - US 10:00 AM data (ISM/CC) + post-equity-open flow.
  COMEX_SETTLE   - COMEX gold settlement 1:30 PM ET (all instruments).
  CME_PRECLOSE   - CME equity futures pre-settlement 2:45 PM CT.
  NYSE_CLOSE     - NYSE closing bell 4:00 PM ET.

TRADING SESSIONS:
  Fixed session stat windows are in pipeline/build_daily_features.py.
  DST-aware session times (NYSE_OPEN, LONDON_METALS, etc.) are in
  pipeline/dst.py SESSION_CATALOG — those track actual market opens.

TRADING DAY:
  Runs 09:00 Brisbane to next 09:00 Brisbane (~1,440 minutes).
  Bars before 09:00 are assigned to the PREVIOUS trading day.

ENTRY MODELS (defined below as ENTRY_MODELS):
  E1 (Market-On-Next-Bar) - Enter at OPEN of the bar AFTER confirm bar.
     Fill rate ~100%. Best for momentum ORBs (CME_REOPEN, TOKYO_OPEN).
  E2 (Stop-Market) - Stop order at ORB level + N ticks slippage. Triggers on
     first bar whose range crosses the ORB level. No confirmation needed.
     Fakeouts included as trades. Industry-standard honest breakout entry (Crabel).
     Always CB1 (no confirm bars concept — stop triggers on first touch).
  E3 (Limit-At-ORB) - Place limit order at ORB level after confirm.
     Fills only if price retraces to ORB level (~96-97% on G5+ days).
     Best for retrace ORBs (LONDON_METALS, US_DATA_830) where price spikes then pulls back.

CONFIRM BARS (CB1-CB5, defined in outcome_builder.py):
  Number of consecutive 1-minute bars closing outside the ORB range
  required before entry is confirmed. Higher CB = more confirmation but
  worse entry price (market has moved further).
  CB2 optimal for CME_REOPEN/TOKYO_OPEN momentum; CB5 optimal for LONDON_METALS E3 retrace.

RR TARGETS (RR1.0-RR4.0, defined in outcome_builder.py):
  Risk/Reward ratio. Target distance = entry risk * RR target.
  Risk = |entry_price - stop_price| where stop = opposite ORB level.
  RR2.5 optimal for CME_REOPEN/TOKYO_OPEN; RR2.0 for LONDON_METALS; RR1.5 for US_DATA_830.

ORB SIZE FILTERS:
  G-filters (Greater-than): Only trade when ORB size >= N points.
    G4+ is the minimum for positive expectancy on MES/MNQ. G5/G6 increase
    per-trade edge but reduce trade count. G8+ only useful for US_DATA_830.
    MGC: G6 minimum (Feb 2026 regime shift — ATR 31→105, G4 passes 87.5%).
  L-filters (Less-than): Only trade when ORB size < N points.
    ALL L-filter strategies have negative expectancy. Do not trade.
  NO_FILTER: Trade all days regardless of ORB size.
    MGC/MES: ALL no-filter strategies have negative expectancy. Do not trade.
    MNQ: POSITIVE unfiltered at 5 CORE sessions after BH FDR (K=105,627) and WF.
    See TRADING_RULES.md ORB Size Filters table for details (audit 2026-03-24).

GRID (5,544 strategy combinations, full grid before ENABLED_SESSIONS filtering):
  E1: 12 ORBs x 6 RRs x 5 CBs x 11 filters = 3,960
  E2: 12 ORBs x 6 RRs x 1 CB x 11 filters = 792 (E2 always CB1)
  E3: 12 ORBs x 6 RRs x 1 CB x 11 filters = 792 (E3 always CB1)
  Total = 5,544 (base grid; session-specific composites expand per-session)

==========================================================================
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date
from typing import TYPE_CHECKING, Any, ClassVar

from pipeline.cost_model import COST_SPECS, get_cost_spec

if TYPE_CHECKING:
    import pandas as pd


# ──────────────────────────────────────────────────────────────────────────
# Filter self-description — AtomDescription
# ──────────────────────────────────────────────────────────────────────────
#
# Canonical support for eligibility visibility. Every StrategyFilter can
# describe itself as a list of atomic conditions via the `describe()` method.
# This is the SINGLE source of truth for "how does this filter evaluate".
# No parallel decomposition registry allowed.
#
# Design: docs/plans/2026-04-07-canonical-filter-self-description-design.md
# Rule:   .claude/rules/institutional-rigor.md — no re-encoded canonical logic
# ──────────────────────────────────────────────────────────────────────────


# Missing-value detection used by describe() implementations. Consistent
# with pandas semantics: treats None, NaN, pd.NA, NaT as missing.
def _atom_is_missing(value: Any) -> bool:
    """True if value represents missing data (None, NaN, NaT, pd.NA).

    This is the canonical missing-data check for describe() implementations.
    Built on pandas.isna() which handles Python None, float NaN, numpy NaN,
    pandas NA, and pandas NaT uniformly. Lazy-imports pandas.
    """
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        import pandas as pd

        return bool(pd.isna(value))
    except (ImportError, ValueError, TypeError):
        return False


def _atom_numeric(value: Any) -> float | None:
    """Return value as float if numeric and non-missing, else None.

    Combines _atom_is_missing with explicit numeric conversion. Used by
    describe() implementations to narrow types for comparisons — pyright
    cannot narrow through _atom_is_missing alone.
    """
    if _atom_is_missing(value):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


def is_e2_lookahead_filter(filter_type: str) -> bool:
    """True iff filter_type depends on break-bar properties unknown at E2 entry.

    Canonical source: E2_EXCLUDED_FILTER_PREFIXES / E2_EXCLUDED_FILTER_SUBSTRINGS
    (defined later in this module). Discovery, execution, drift checks, and
    eligibility messaging must all delegate here instead of re-encoding the
    membership rule.
    """
    return filter_type.startswith(E2_EXCLUDED_FILTER_PREFIXES) or any(
        sub in filter_type for sub in E2_EXCLUDED_FILTER_SUBSTRINGS
    )


def _e2_look_ahead_reason(filter_type: str) -> str | None:
    """Return E2 look-ahead exclusion reason if filter_type is E2-excluded.

    Canonical source: E2_EXCLUDED_FILTER_PREFIXES / E2_EXCLUDED_FILTER_SUBSTRINGS
    (defined later in this module). Mirrors the gate used in
    strategy_discovery.py:1161 and execution_engine.py:627 — do NOT re-encode
    the membership rule; delegate to the same constants.

    Returns a human-readable reason string if the filter is E2-excluded, or
    None if the filter is E2-safe. Used by describe() overrides on break-bar
    dependent filters (BreakSpeed, BreakBarContinues, Volume, CombinedATRVol).
    """
    # Forward reference to constants defined lower in the module.
    # Python resolves globals at call time, so this is safe as long as the
    # module has finished importing before describe() is invoked.
    if filter_type.startswith(E2_EXCLUDED_FILTER_PREFIXES):
        return (
            f"E2 look-ahead: filter_type '{filter_type}' depends on break-bar "
            f"properties unknown at E2 entry placement"
        )
    if is_e2_lookahead_filter(filter_type):
        for sub in E2_EXCLUDED_FILTER_SUBSTRINGS:
            if sub in filter_type:
                return (
                    f"E2 look-ahead: filter_type '{filter_type}' depends on break-bar "
                    f"properties unknown at E2 entry placement (substring '{sub}')"
                )
    return None


@dataclass(frozen=True)
class AtomDescription:
    """A single atomic condition produced by a filter's describe() method.

    Filters produce one or more AtomDescription records per call to
    describe(). Each atom represents one logical check against the daily
    features row — e.g. "ORB size >= 5 pts" or "cost ratio < 10%".

    The ELIGIBILITY LAYER consumes these to build its user-facing report
    with explicit statuses. All filter semantics live here — the eligibility
    builder MUST NOT re-encode any logic; it only translates AtomDescription
    to its ConditionRecord format.

    Fields:
    - name: human-readable description of the check
    - category: PRE_SESSION | INTRA_SESSION | OVERLAY | DIRECTIONAL
    - resolves_at: lifecycle event at which a PENDING atom resolves
      (STARTUP | ORB_FORMATION | BREAK_DETECTED | CONFIRM_COMPLETE | TRADE_ENTERED)
    - feature_column: the daily_features column this atom reads (already
      session-resolved; None for derived or overlay atoms)
    - observed_value: the actual value today, or None if missing/pending
    - threshold: the comparison threshold from the filter instance
    - comparator: plain-English comparator ("≥", "<", "==", etc.)
    - passes: True (PASS), False (FAIL), None (PENDING or MISSING)
    - is_data_missing: True if the required data was missing (distinct from FAIL)
    - is_not_applicable: True if this atom does not apply to the caller's
      (instrument, session, entry_model) combination
    - not_applicable_reason: if is_not_applicable, human-readable reason
    - validated_for: tuple of (instrument, session) pairs where research
      has validated this atom. Empty = applies to all combinations.
      Sourced from filter ClassVar VALIDATED_FOR. Adapter maps a non-empty
      tuple that excludes the caller's lane to NOT_APPLICABLE_INSTRUMENT.
    - last_revalidated: date of last research revalidation. Sourced from
      filter ClassVar LAST_REVALIDATED. Adapter maps stale (>180 days)
      to STALE_VALIDATION.
    - confidence_tier: PROVEN | PLAUSIBLE | LEGACY | UNKNOWN — research
      quality tier from filter ClassVar CONFIDENCE_TIER. Surfaced in the
      eligibility report so traders see provenance at a glance.
    - error_message: optional diagnostic string captured by the filter when
      it explicitly catches an exception (e.g., type mismatch on division).
      The adapter aggregates non-None error_messages into report.build_errors
      so the diagnostic trail is preserved without widening describe()'s API.
    - size_multiplier: trade sizing modifier for PASS conditions (default 1.0,
      0.5 for calendar HALF_SIZE)
    - explanation: one-sentence plain-English description
    """

    name: str
    category: str
    resolves_at: str
    passes: bool | None
    feature_column: str | None = None
    observed_value: Any = None
    threshold: Any = None
    comparator: str = ""
    is_data_missing: bool = False
    is_not_applicable: bool = False
    not_applicable_reason: str = ""
    validated_for: tuple[tuple[str, str], ...] = ()
    last_revalidated: date | None = None
    confidence_tier: str = "UNKNOWN"
    error_message: str | None = None
    size_multiplier: float = 1.0
    explanation: str = ""

# ── Noise ExpR floor per entry model ──────────────────────────────────────
# NO LONGER A HARD GATE (2026-03-21). Phase 2b removed from strategy_validator.
# Now used only for post-validation noise_risk flag computation.
#
# Canonical aggregation: p95 of pooled null survivor ExpR per instrument
# per entry model. Per-instrument floors are in NOISE_FLOOR_BY_INSTRUMENT.
# This global dict retained for backward compat with live_config / null scripts.
#
# Seed artifacts: scripts/tests/null_seeds/{.,mnq,mes}/
# Caveat: Gaussian null has sigma overshoot (MGC 2.54x, MNQ 1.23x real
# trimmed std). Bias direction on ORB noise floors is unproven.
# Block bootstrap calibration is a future workstream.
NOISE_EXPR_FLOOR: dict[str, float] = {
    "E1": 0,
    "E2": 0,
}

# NOISE_FLOOR DISABLED (2026-03-26): per-strategy null
# (scripts/tools/noise_floor_bootstrap.py) is the correct method.
# The pooled global-max floor was replaced 2026-03-24.
# Summary flag removed to prevent false alarm fatigue (55/56 false positives).
#
# Per-strategy null results (100 MNQ seeds, Mar 25 2026):
#   NYSE_CLOSE VOL_RV12_N20 RR1.0 O15: noise mean=0.055, P95=0.106, real=0.208 → p=0.011 SIGNAL
#   COMEX_SETTLE ORB_G8 RR1.0 O5:      noise mean=0.083, P95=0.125, real=0.130 → p=0.024 SIGNAL
#   SINGAPORE_OPEN ORB_G8 RR4.0 O15:   noise mean=0.089, P95=0.162, real=0.163 → p=0.053 MARGINAL
#   NYSE_OPEN X_MES_ATR60: untestable (cross-instrument filter absent in single-inst null)
#   US_DATA_1000 X_MES_ATR60: untestable (same reason)
#
# @research-source noise_floor_methodology.md (Mar 25 2026)
# @revalidated-for stratified-K event-based (2026-03-25)
NOISE_FLOOR_BY_INSTRUMENT: dict[str, dict[str, float]] = {
    "MGC": {"E1": 0, "E2": 0},  # Disabled — use per-strategy null instead
    "MES": {"E1": 0, "E2": 0},  # Disabled
    "MNQ": {"E1": 0, "E2": 0},  # Disabled
}

# Walk-forward start-date override per instrument.
# Full-sample validation (Phase A) uses ALL data. Only WF window generation
# starts from max(earliest_outcome, override_date).
#
# ── MGC REGIME LIMITATION (adversarial review 2026-03-18) ────────────────
# REGIME DEPENDENCY IS REAL AND NOT RESOLVED:
#   - All WF windows cluster in recent high-vol years regardless of mode
#     (trade-count or calendar). Insufficient trades in low-vol years.
#   - Most FDR-surviving MGC strategies use G-filters, which self-select for
#     high-vol conditions (G4 at low ATR requires large % of daily range).
#   - Full-sample validation includes low-vol years (75%-of-years-positive),
#     but WF does NOT validate OOS performance in low vol.
#
# CONSEQUENCE: MGC strategies are regime-conditional on elevated ATR. They
# may not produce edge if gold volatility returns to historical low levels.
# @research-source adversarial-review-findings 2026-03-18
# @revalidated-for E1/E2 event-based sessions (2026-03-18)
# This is acceptable IF position sizing scales with regime confidence.
# Runtime check: scripts/tools/check_regime.py flags low-ATR conditions.
# ─────────────────────────────────────────────────────────────────────────
WF_START_OVERRIDE: dict[str, date] = {
    "MGC": date(2022, 1, 1),  # Gold <$1800 pre-2022 = tiny ORBs, G4+ windows invalid
    # ── MNQ/MES micro contract launch exclusion (2026-04-09 data audit) ────
    # MNQ and MES micro contracts launched 2019-05-06. Structural data audit
    # across 5 independent variables confirms 2019 is non-representative:
    #
    #   MNQ 2019 vs 2020+:
    #     ATR:           113.7 vs 279.5 = 0.42x
    #     CME_PRECLOSE G8 pass: 39.0% vs 97.7% (monthly: Nov=5.3%, Dec=20%)
    #     EUROPE_FLOW G8 pass:  22.8% vs 83.5%
    #     COMEX_SETTLE G8 pass: 30.5% vs 93.2%
    #     NYSE_OPEN volume:     5,689 vs 30,845 = 0.16x
    #
    #   MES 2019 vs 2020+:
    #     ATR:           33.2 vs 63.9 = 0.52x
    #     NYSE_OPEN G8 pass:    10.5% vs 58.1%
    #     CME_PRECLOSE G8 pass:  1.2% vs 25.8%
    #     NYSE_OPEN volume:     6,896 vs 24,055 = 0.29x
    #
    # Structural cause: contract-launch thin liquidity + small ORBs. The
    # absolute G-filter thresholds (G5=5pts, G8=8pts) select fundamentally
    # different populations in the low-ATR 2019 regime — NOT a data-quality
    # issue but a structural microstructure difference.
    #
    # Monthly validation: Q3/Q4 2019 does NOT normalize (Nov G8=5.3%,
    # Dec=20% for MNQ CME_PRECLOSE). Jan 2020 = 71.4%, Feb = 72.2% —
    # clear improvement. 2020-01-01 is the clean boundary.
    #
    # Impact on 6 validated strategies: all have 2019 data (2-156 trades).
    # Net-positive — removes thin/noisy early WF training windows for 4/5
    # non-NYSE MNQ sessions. NYSE_OPEN (156 trades, 97% G8 pass) loses
    # valid data but WFE=2.12 has ample margin.
    #
    # NOT data-snooped: justification is structural (ATR, volume, filter
    # pass rates) — zero strategy PnL consulted.
    #
    # @research-source data-audit: 2026-04-09 session (5-variable structural analysis)
    # @revalidated-for E2 event-based sessions (2026-04-09)
    #
    # KNOWN GAP (separate task): MES G-filter thresholds (G5=5, G8=8) are
    # absolute points designed for NQ-scale ORBs. MES CME_PRECLOSE G8 passes
    # only 25.8% of days even in 2020+. This is a filter-rescaling problem,
    # NOT fixable by WF_START_OVERRIDE. Tracked separately.
    "MNQ": date(2020, 1, 1),  # Micro launch 2019-05-06, ATR 0.42x, G8 39% CME_PRECLOSE
    "MES": date(2020, 1, 1),  # Micro launch 2019-05-06, ATR 0.52x, G8 10.5% NYSE_OPEN
}

# @research-source Lopez de Prado AFML Ch.2 — information-driven sampling;
#   window by trade count for regime-spanning OOS validation where calendar
#   windows fail due to extreme ATR variation (MGC 9.2x)
# @entry-models E1/E2
# @revalidated-for E1/E2 event-based sessions (2026-03-17)
WF_TRADE_COUNT_OVERRIDE: dict[str, int] = {
    "MGC": 30,  # 30 trades per OOS window — regime-spanning
}
WF_MIN_TRAIN_TRADES: dict[str, int] = {
    "MGC": 45,  # 1.5x OOS window — stable IS estimation
}

# ── REGIME WF scaling (N=30-99) ─────────────────────────────────────────
# Strategies with sample_size < CORE_MIN_SAMPLES use smaller WF windows.
# 1/3 of CORE params. min_windows=2 → with 60% threshold, BOTH windows
# must be positive. P(noise passes 2/2) = 0.25 vs P(noise passes 2/3) = 0.50.
# Net: REGIME WF is MORE selective per-strategy than CORE WF.
# @research-source regime-validation-design 2026-03-31
# @revalidated-for E2 event-based sessions (2026-03-31)
REGIME_WF_TRADE_COUNT: dict[str, int] = {
    "MGC": 10,  # 1/3 of CORE 30
}
REGIME_WF_MIN_TRAIN_TRADES: dict[str, int] = {
    "MGC": 15,  # 1.5x OOS window (same ratio as CORE 45/30)
}
REGIME_WF_MIN_WINDOWS = 2
REGIME_WF_MIN_TRADES_PER_WINDOW = 5  # Calendar mode: 1/3 of CORE 15


@dataclass(frozen=True)
class StrategyFilter:
    """Base class for strategy filters."""

    filter_type: str
    description: str

    @property
    def requires_micro_data(self) -> bool:
        """Does this filter need REAL micro contract data in bars_1m?

        Default False — price-based filters work on any era (parent-proxy
        or real-micro) because price levels are identical across
        MNQ/NQ, MES/ES, MGC/GC etc.

        Volume-based filters (VolumeFilter, OrbVolumeFilter) MUST override
        to return True because MNQ micro volume is NOT the same as NQ
        parent volume — the selectivity signal only holds on real-micro
        data.

        CompositeFilter overrides dynamically: True iff any component
        requires it.

        Consumers (Phase 3c/3d of canonical-data-redownload):
        - Stage 3c rebuild: only rebuild era-appropriate date ranges
        - Stage 3d drift check: reject validated_setups with volume filters
          referencing trades before `data_era.micro_launch_day(instrument)`

        @rule canonical-filter-self-description
        @canonical-source pipeline/data_era.py (Phase 3a foundation, b032a03)
        """
        return False

    def to_json(self) -> str:
        """Serialize filter params to JSON."""
        return json.dumps(asdict(self))

    def matches_row(self, row: dict, orb_label: str) -> bool:
        """Check if a daily_features row matches this filter. Override in subclass."""
        return True

    def matches_df(self, df: pd.DataFrame, orb_label: str) -> pd.Series:
        """Vectorized filter: return boolean Series. Default falls back to iterrows."""
        import pandas as pd

        return pd.Series(
            [self.matches_row(row.to_dict(), orb_label) for _, row in df.iterrows()],
            index=df.index,
        )

    def describe(
        self,
        row: dict,
        orb_label: str,
        entry_model: str,
    ) -> list[AtomDescription]:
        """Return atomic descriptions of this filter's conditions.

        Default implementation: one atom derived from matches_row() result.
        This works correctly for simple filters but loses granularity for
        composites. Composite filters MUST override to return multiple atoms
        so consumers can see which component failed.

        The filter owns its decomposition — the eligibility layer MUST NOT
        re-encode comparison logic. If you need finer atoms, override here.

        Args:
            row: A daily_features row as a dict (e.g. from DataFrame.to_dict)
            orb_label: The session identifier (e.g. "CME_REOPEN")
            entry_model: The strategy's entry model ("E1" | "E2" | "E3") —
                used to mark atoms as NOT_APPLICABLE when look-ahead unsafe

        Returns:
            List of AtomDescription records. An empty list means "this filter
            has no atomic gates" (e.g. NoFilter). A single-atom list is the
            most common case.
        """
        _ = entry_model  # base class default does not use entry_model
        passes = self.matches_row(row, orb_label)
        return [
            AtomDescription(
                name=self.description,
                category="INTRA_SESSION",  # conservative default
                resolves_at="ORB_FORMATION",
                passes=passes,
                threshold=None,
                observed_value=None,
                comparator="",
                explanation=self.description,
            )
        ]


@dataclass(frozen=True)
class NoFilter(StrategyFilter):
    """Pass-through filter — all days match."""

    filter_type: str = "NO_FILTER"
    description: str = "No filter (all days)"

    def matches_row(self, row: dict, orb_label: str) -> bool:
        return True

    def matches_df(self, df: pd.DataFrame, orb_label: str) -> pd.Series:
        import pandas as pd

        return pd.Series(True, index=df.index)

    def describe(
        self,
        row: dict,
        orb_label: str,
        entry_model: str,
    ) -> list[AtomDescription]:
        """NoFilter has zero atomic gates — always eligible by filter definition.

        Overlays (calendar, ATR velocity) are added separately by the
        eligibility builder, not by this method.
        """
        _ = (row, orb_label, entry_model)
        return []


@dataclass(frozen=True)
class OrbSizeFilter(StrategyFilter):
    """Filter by ORB size ranges."""

    min_size: float | None = None
    max_size: float | None = None

    def matches_row(self, row: dict, orb_label: str) -> bool:
        size = row.get(f"orb_{orb_label}_size")
        if size is None:
            return False
        if self.min_size is not None and size < self.min_size:
            return False
        if self.max_size is not None and size >= self.max_size:
            return False
        return True

    def matches_df(self, df: pd.DataFrame, orb_label: str) -> pd.Series:
        import pandas as pd

        col = f"orb_{orb_label}_size"
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        mask = df[col].notna()
        if self.min_size is not None:
            mask = mask & (df[col] >= self.min_size)
        if self.max_size is not None:
            mask = mask & (df[col] < self.max_size)
        return mask

    def describe(
        self,
        row: dict,
        orb_label: str,
        entry_model: str,
    ) -> list[AtomDescription]:
        """Describe ORB size gate as up to two atoms (min_size, optional max_size).

        ORB size is intra-session: it resolves at ORB_FORMATION because the
        size is only known once the ORB window closes.
        """
        _ = entry_model  # size gates are entry-model agnostic
        col = f"orb_{orb_label}_size"
        observed = _atom_numeric(row.get(col))
        missing = observed is None

        atoms: list[AtomDescription] = []
        if self.min_size is not None:
            passes = None if observed is None else observed >= self.min_size
            atoms.append(
                AtomDescription(
                    name=f"ORB size >= {self.min_size:g} pts",
                    category="INTRA_SESSION",
                    resolves_at="ORB_FORMATION",
                    passes=passes,
                    feature_column=col,
                    observed_value=observed,
                    threshold=self.min_size,
                    comparator=">=",
                    is_data_missing=missing,
                    explanation=f"Skip ORBs smaller than {self.min_size:g} points.",
                )
            )
        if self.max_size is not None:
            passes = None if observed is None else observed < self.max_size
            atoms.append(
                AtomDescription(
                    name=f"ORB size < {self.max_size:g} pts",
                    category="INTRA_SESSION",
                    resolves_at="ORB_FORMATION",
                    passes=passes,
                    feature_column=col,
                    observed_value=observed,
                    threshold=self.max_size,
                    comparator="<",
                    is_data_missing=missing,
                    explanation=f"Skip ORBs larger than {self.max_size:g} points (band filter).",
                )
            )
        return atoms


@dataclass(frozen=True)
class CostRatioFilter(StrategyFilter):
    """Filter on round-trip friction share of raw ORB risk.

    Honest framing: normalized minimum-viable-trade-size screen using the
    canonical cost model. This is not a new predictive signal.
    """

    max_cost_ratio_pct: float

    def matches_row(self, row: dict, orb_label: str) -> bool:
        size = row.get(f"orb_{orb_label}_size")
        symbol = row.get("symbol")
        if size is None or symbol is None or size <= 0:
            return False
        try:
            cost_spec = get_cost_spec(symbol)
        except ValueError:
            return False
        raw_risk = size * cost_spec.point_value
        cost_ratio_pct = 100.0 * cost_spec.total_friction / (raw_risk + cost_spec.total_friction)
        return cost_ratio_pct < self.max_cost_ratio_pct

    def matches_df(self, df: pd.DataFrame, orb_label: str) -> pd.Series:
        import pandas as pd

        col = f"orb_{orb_label}_size"
        if col not in df.columns or "symbol" not in df.columns:
            return pd.Series(False, index=df.index)
        result = pd.Series(False, index=df.index)
        base_mask = df[col].notna() & df["symbol"].notna() & (df[col] > 0)
        for symbol, cost_spec in COST_SPECS.items():
            inst_mask = base_mask & (df["symbol"] == symbol)
            if not inst_mask.any():
                continue
            raw_risk = df.loc[inst_mask, col] * cost_spec.point_value
            cost_ratio_pct = 100.0 * cost_spec.total_friction / (raw_risk + cost_spec.total_friction)
            result.loc[inst_mask] = cost_ratio_pct < self.max_cost_ratio_pct
        return result

    def describe(
        self,
        row: dict,
        orb_label: str,
        entry_model: str,
    ) -> list[AtomDescription]:
        """Cost-ratio gate derived from orb size + canonical cost spec.

        Intra-session: resolves at ORB_FORMATION (size known at ORB close).
        Both missing/zero orb_size and unknown symbol surface as DATA_MISSING
        so the report shows 'data unavailable' instead of a spurious FAIL.
        """
        _ = entry_model  # cost-ratio applies to all entry models
        size_col = f"orb_{orb_label}_size"
        size = _atom_numeric(row.get(size_col))
        symbol = row.get("symbol")
        missing = size is None or symbol is None or size <= 0

        observed_ratio: float | None = None
        if not missing:
            assert size is not None  # type narrowing for pyright
            try:
                cost_spec = get_cost_spec(str(symbol))
                raw_risk = size * cost_spec.point_value
                observed_ratio = (
                    100.0 * cost_spec.total_friction / (raw_risk + cost_spec.total_friction)
                )
            except ValueError:
                missing = True
                observed_ratio = None

        passes = None if missing else observed_ratio < self.max_cost_ratio_pct  # type: ignore[operator]
        return [
            AtomDescription(
                name=f"Cost ratio < {self.max_cost_ratio_pct:g}%",
                category="INTRA_SESSION",
                resolves_at="ORB_FORMATION",
                passes=passes,
                feature_column=size_col,
                observed_value=observed_ratio,
                threshold=self.max_cost_ratio_pct,
                comparator="<",
                is_data_missing=missing,
                explanation=(
                    f"Require round-trip cost share of raw ORB risk "
                    f"< {self.max_cost_ratio_pct:g}% (minimum-viable-trade-size screen)."
                ),
            )
        ]


@dataclass(frozen=True)
class VolumeFilter(StrategyFilter):
    """Filter by relative volume at break bar (Selective Classification).

    Abstain from trading when breakout volume is below normal.
    Relative volume = break_bar_volume / median(same minute-of-day, N prior days).
    Fail-closed: if rel_vol data missing, day is ineligible.

    The rel_vol_{orb_label} key must be pre-computed and injected into the
    row dict before calling matches_row (see strategy_discovery._compute_relative_volumes).
    """

    min_rel_vol: float = 1.2
    lookback_days: int = 20

    @property
    def requires_micro_data(self) -> bool:
        """Relative-volume gate requires REAL micro contract data.

        Break-bar volume for MNQ/MES/MGC differs from NQ/ES/GC parent
        volume. Using parent-proxy data for rel_vol computation produces
        a meaningless signal (parent volumes are 10-20x micro volumes).
        Stage 3d will reject validated_setups with VolumeFilter referencing
        pre-micro-launch trades.

        @canonical-source pipeline/data_era.py
        """
        return True

    def matches_row(self, row: dict, orb_label: str) -> bool:
        rel_vol = row.get(f"rel_vol_{orb_label}")
        if rel_vol is None:
            return False  # fail-closed: no data = ineligible
        return rel_vol >= self.min_rel_vol

    def matches_df(self, df: pd.DataFrame, orb_label: str) -> pd.Series:
        import pandas as pd

        col = f"rel_vol_{orb_label}"
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        return df[col].notna() & (df[col] >= self.min_rel_vol)

    def describe(
        self,
        row: dict,
        orb_label: str,
        entry_model: str,
    ) -> list[AtomDescription]:
        """Relative-volume gate. Intra-session, resolves at BREAK_DETECTED.

        E2-excluded: rel_vol includes break-bar volume, unknown at E2 entry.
        """
        col = f"rel_vol_{orb_label}"
        if entry_model == "E2":
            reason = _e2_look_ahead_reason(self.filter_type)
            if reason is not None:
                return [
                    AtomDescription(
                        name=f"Rel volume >= {self.min_rel_vol:g}",
                        category="INTRA_SESSION",
                        resolves_at="BREAK_DETECTED",
                        passes=None,
                        feature_column=col,
                        threshold=self.min_rel_vol,
                        comparator=">=",
                        is_not_applicable=True,
                        not_applicable_reason=reason,
                        explanation=(
                            "Relative-volume gate does not apply to E2 entries — "
                            "break-bar volume is unknown at E2 order placement."
                        ),
                    )
                ]
        observed = _atom_numeric(row.get(col))
        missing = observed is None
        passes = None if missing else observed >= self.min_rel_vol
        return [
            AtomDescription(
                name=f"Rel volume >= {self.min_rel_vol:g}",
                category="INTRA_SESSION",
                resolves_at="BREAK_DETECTED",
                passes=passes,
                feature_column=col,
                observed_value=observed,
                threshold=self.min_rel_vol,
                comparator=">=",
                is_data_missing=missing,
                explanation=(
                    f"Require break-bar relative volume >= {self.min_rel_vol:g}x baseline "
                    f"(breakout conviction gate)."
                ),
            )
        ]


@dataclass(frozen=True)
class CombinedATRVolumeFilter(VolumeFilter):
    """Combined ATR regime + break-bar volume filter.

    Requires BOTH:
    1. ATR_20 percentile >= min_atr_pct (high-vol regime)
    2. Relative volume >= min_rel_vol (break-bar conviction)

    Subclasses VolumeFilter so strategy_fitness.py isinstance guard
    triggers bars_1m enrichment for rel_vol computation.

    atr_20_pct is pre-computed in daily_features (rolling 252d percentile).
    Fail-closed: if either value is missing, day is ineligible.

    @research-source research/research_vol_regime_filter.py
    """

    min_atr_pct: float = 70.0

    def matches_row(self, row: dict, orb_label: str) -> bool:
        atr_pct = row.get("atr_20_pct")
        if atr_pct is None:
            return False  # fail-closed
        rel_vol = row.get(f"rel_vol_{orb_label}")
        if rel_vol is None:
            return False  # fail-closed
        return atr_pct >= self.min_atr_pct and rel_vol >= self.min_rel_vol

    def matches_df(self, df: pd.DataFrame, orb_label: str) -> pd.Series:
        import pandas as pd

        rel_col = f"rel_vol_{orb_label}"
        has_rel = rel_col in df.columns
        has_atr = "atr_20_pct" in df.columns

        if not has_rel or not has_atr:
            return pd.Series(False, index=df.index)

        return (
            df["atr_20_pct"].notna()
            & (df["atr_20_pct"] >= self.min_atr_pct)
            & df[rel_col].notna()
            & (df[rel_col] >= self.min_rel_vol)
        )

    def describe(
        self,
        row: dict,
        orb_label: str,
        entry_model: str,
    ) -> list[AtomDescription]:
        """Combined ATR regime + rel-volume gate. Emits TWO atoms.

        - ATR-20 percentile atom: PRE_SESSION, resolves at STARTUP (E2-safe).
        - Rel-volume atom: INTRA_SESSION, resolves at BREAK_DETECTED.
          E2-excluded by filter_type prefix (ATR70_VOL). For E2 the volume
          atom is NOT_APPLICABLE but the ATR atom still evaluates normally,
          preserving granularity in the eligibility report.
        """
        # Atom 1: ATR regime (E2-safe, pre-session)
        atr_obs = _atom_numeric(row.get("atr_20_pct"))
        atr_missing = atr_obs is None
        atr_passes = None if atr_missing else atr_obs >= self.min_atr_pct
        atr_atom = AtomDescription(
            name=f"ATR percentile >= {self.min_atr_pct:g}",
            category="PRE_SESSION",
            resolves_at="STARTUP",
            passes=atr_passes,
            feature_column="atr_20_pct",
            observed_value=atr_obs,
            threshold=self.min_atr_pct,
            comparator=">=",
            is_data_missing=atr_missing,
            explanation=(
                f"Require own ATR-20 rolling percentile >= {self.min_atr_pct:g}% "
                f"(high-vol regime gate)."
            ),
        )

        # Atom 2: rel_vol (E2-excluded via filter_type prefix)
        vol_col = f"rel_vol_{orb_label}"
        if entry_model == "E2":
            reason = _e2_look_ahead_reason(self.filter_type)
            if reason is not None:
                vol_atom = AtomDescription(
                    name=f"Rel volume >= {self.min_rel_vol:g}",
                    category="INTRA_SESSION",
                    resolves_at="BREAK_DETECTED",
                    passes=None,
                    feature_column=vol_col,
                    threshold=self.min_rel_vol,
                    comparator=">=",
                    is_not_applicable=True,
                    not_applicable_reason=reason,
                    explanation=(
                        "Relative-volume half of the combined gate does not "
                        "apply to E2 entries — break-bar volume is unknown at "
                        "E2 order placement."
                    ),
                )
                return [atr_atom, vol_atom]

        vol_obs = _atom_numeric(row.get(vol_col))
        vol_missing = vol_obs is None
        vol_passes = None if vol_missing else vol_obs >= self.min_rel_vol
        vol_atom = AtomDescription(
            name=f"Rel volume >= {self.min_rel_vol:g}",
            category="INTRA_SESSION",
            resolves_at="BREAK_DETECTED",
            passes=vol_passes,
            feature_column=vol_col,
            observed_value=vol_obs,
            threshold=self.min_rel_vol,
            comparator=">=",
            is_data_missing=vol_missing,
            explanation=(
                f"Require break-bar relative volume >= {self.min_rel_vol:g}x baseline."
            ),
        )
        return [atr_atom, vol_atom]


@dataclass(frozen=True)
class OrbVolumeFilter(StrategyFilter):
    """Filter by total volume during the ORB formation window.

    Gates on aggregate contract volume across the entire ORB period (not the
    break bar). Known at ORB close — safe for pre-break decisions.
    Values from daily_features.orb_{SESSION}_volume (pre-computed in
    pipeline/build_daily_features.py). No enrichment needed.

    Thresholds are absolute (not normalized). Different sessions have different
    volume regimes — discovery tests all tiers across all sessions and finds
    the combinations where volume discriminates.

    Fail-closed: missing volume data = ineligible day.

    @research-source research/output/confluence_program/phase1_run.py
    @entry-models E2
    @revalidated-for E2
    """

    min_volume: float

    @property
    def requires_micro_data(self) -> bool:
        """Aggregate ORB-window volume requires REAL micro contract data.

        Absolute volume thresholds (e.g. 2K/4K/8K/16K contracts) are
        calibrated against real-micro trading patterns. Parent-proxy
        volumes are an order of magnitude higher — a 2K threshold on
        NQ parent data would fire every session, producing a meaningless
        signal.

        @canonical-source pipeline/data_era.py
        """
        return True

    def matches_row(self, row: dict, orb_label: str) -> bool:
        volume = row.get(f"orb_{orb_label}_volume")
        if volume is None:
            return False  # fail-closed
        return volume >= self.min_volume

    def matches_df(self, df: pd.DataFrame, orb_label: str) -> pd.Series:
        import pandas as pd

        col = f"orb_{orb_label}_volume"
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        return df[col].notna() & (df[col] >= self.min_volume)

    def describe(
        self,
        row: dict,
        orb_label: str,
        entry_model: str,
    ) -> list[AtomDescription]:
        """Aggregate ORB-window volume gate. Intra-session, resolves at ORB_FORMATION.

        E2-safe: ORB-window volume is fully observed once the ORB closes,
        before any E2 break can occur.
        """
        _ = entry_model
        col = f"orb_{orb_label}_volume"
        observed = _atom_numeric(row.get(col))
        missing = observed is None
        passes = None if missing else observed >= self.min_volume
        return [
            AtomDescription(
                name=f"ORB volume >= {self.min_volume:g}",
                category="INTRA_SESSION",
                resolves_at="ORB_FORMATION",
                passes=passes,
                feature_column=col,
                observed_value=observed,
                threshold=self.min_volume,
                comparator=">=",
                is_data_missing=missing,
                explanation=(
                    f"Require aggregate ORB-window volume >= {self.min_volume:g} "
                    f"contracts (session participation gate)."
                ),
            )
        ]


@dataclass(frozen=True)
class CrossAssetATRFilter(StrategyFilter):
    """Filter by another instrument's ATR regime.

    Reads cross_atr_{source_instrument}_pct from the row dict.
    This key is injected at discovery/fitness time by
    _inject_cross_asset_atrs() — NOT stored in daily_features schema.

    Fail-closed: if the key is absent or None, day is ineligible.

    @research-source research/research_vol_regime_filter.py

    Canonical metadata (consumed by describe() / eligibility adapter):
    - VALIDATED_FOR: MNQ at 6 US sessions per research_vol_regime_filter.
      Cross-asset filter is MNQ-only (US sessions where MES leads MNQ flow).
      Other instrument/session pairs return NOT_APPLICABLE_INSTRUMENT.
    - LAST_REVALIDATED: None (legacy, pre-Apr 2026 research)
    - CONFIDENCE_TIER: PROVEN
    """

    # ── Canonical research metadata (ClassVar) ──────────────────────────
    VALIDATED_FOR: ClassVar[tuple[tuple[str, str], ...]] = (
        ("MNQ", "CME_PRECLOSE"),
        ("MNQ", "COMEX_SETTLE"),
        ("MNQ", "US_DATA_1000"),
        ("MNQ", "NYSE_OPEN"),
        ("MNQ", "NYSE_CLOSE"),
        ("MNQ", "US_DATA_830"),
    )
    CONFIDENCE_TIER: ClassVar[str] = "PROVEN"

    source_instrument: str = "MES"
    min_pct: float = 70.0

    def matches_row(self, row: dict, orb_label: str) -> bool:
        val = row.get(f"cross_atr_{self.source_instrument}_pct")
        if val is None:
            return False  # fail-closed
        return val >= self.min_pct

    def matches_df(self, df: pd.DataFrame, orb_label: str) -> pd.Series:
        import pandas as pd

        col = f"cross_atr_{self.source_instrument}_pct"
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        return df[col].notna() & (df[col] >= self.min_pct)

    def describe(
        self,
        row: dict,
        orb_label: str,
        entry_model: str,
    ) -> list[AtomDescription]:
        """Cross-asset ATR percentile gate. Pre-session, resolves at STARTUP.

        Threads VALIDATED_FOR + CONFIDENCE_TIER through. Adapter maps
        non-validated lanes to NOT_APPLICABLE_INSTRUMENT.
        """
        _ = (orb_label, entry_model)
        col = f"cross_atr_{self.source_instrument}_pct"
        observed = _atom_numeric(row.get(col))
        missing = observed is None
        passes = None if missing else observed >= self.min_pct
        return [
            AtomDescription(
                name=f"{self.source_instrument} ATR percentile >= {self.min_pct:g}",
                category="PRE_SESSION",
                resolves_at="STARTUP",
                passes=passes,
                feature_column=col,
                observed_value=observed,
                threshold=self.min_pct,
                comparator=">=",
                is_data_missing=missing,
                validated_for=self.VALIDATED_FOR,
                confidence_tier=self.CONFIDENCE_TIER,
                explanation=(
                    f"Require {self.source_instrument} ATR-20 rolling percentile "
                    f">= {self.min_pct:g}% (cross-asset volatility regime gate)."
                ),
            )
        ]


@dataclass(frozen=True)
class ATRVelRatioFilter(StrategyFilter):
    """Filter by ATR velocity ratio (today's ATR20 / 5-day prior ATR20 average).

    Distinct from ATRVelocityFilter (line ~1793) which is an AVOID filter combining
    atr_vel_regime=Contracting + ORB compression. This class is a simple numeric
    threshold gate on atr_vel_ratio — admits days where vol is expanding.

    Reads atr_vel_ratio from daily_features. ratio > 1.0 = vol expanding.
    Pre-session (no look-ahead): atr_20 uses prior 20 days; atr_vel_ratio uses
    rows[i-5:i] prior-only slice. Verified in pipeline/build_daily_features.py
    lines 1114 + 1121-1132.

    Fail-closed: missing data means day is ineligible.

    Wave 4 Phase B T2-T8 validation (2026-04-11):
    - MNQ TOKYO_OPEN RR1.0: in_ExpR +0.188, WFE 1.42, p=0.0042 (SURVIVES)
    - MES US_DATA_1000 RR1.5: in_ExpR +0.103, WFE 0.96, p=0.044 (SURVIVES)

    @research-source scripts/research/wave4_presession_t2t8.py
    @entry-models E2
    """

    CONFIDENCE_TIER: ClassVar[str] = "PLAUSIBLE"

    min_ratio: float = 1.05

    def matches_row(self, row: dict, orb_label: str) -> bool:
        val = row.get("atr_vel_ratio")
        if val is None:
            return False
        return val >= self.min_ratio

    def matches_df(self, df: pd.DataFrame, orb_label: str) -> pd.Series:
        import pandas as pd

        if "atr_vel_ratio" not in df.columns:
            return pd.Series(False, index=df.index)
        return df["atr_vel_ratio"].notna() & (df["atr_vel_ratio"] >= self.min_ratio)

    def describe(
        self,
        row: dict,
        orb_label: str,
        entry_model: str,
    ) -> list[AtomDescription]:
        """ATR velocity expansion gate. Pre-session, resolves at STARTUP."""
        _ = (orb_label, entry_model)
        observed = _atom_numeric(row.get("atr_vel_ratio"))
        missing = observed is None
        passes = None if missing else observed >= self.min_ratio
        return [
            AtomDescription(
                name=f"ATR velocity ratio >= {self.min_ratio:g}",
                category="PRE_SESSION",
                resolves_at="STARTUP",
                passes=passes,
                feature_column="atr_vel_ratio",
                confidence_tier=self.CONFIDENCE_TIER,
                observed_value=observed,
                threshold=self.min_ratio,
                comparator=">=",
                is_data_missing=missing,
                explanation=(
                    f"Require ATR velocity ratio (today vs 5-day prior ATR) "
                    f">= {self.min_ratio:g} (volatility expansion gate)."
                ),
            )
        ]


@dataclass(frozen=True)
class OwnATRPercentileFilter(StrategyFilter):
    """Filter by the instrument's own ATR(20) rolling percentile.

    Reads atr_20_pct from daily_features (rolling 252d percentile, pre-computed).
    Fail-closed: missing data means day is ineligible.

    April 2026 hypothesis H4: MNQ_ATR_P60 as simpler alternative to X_MES_ATR60
    (CORR(MES_ATR20, MNQ_ATR20) = 0.93 — cross-asset filter is 93% redundant).

    Canonical metadata:
    - VALIDATED_FOR: empty (applies broadly — no instrument restriction
      proven; the filter is a vol-regime gate that should generalize).
    - LAST_REVALIDATED: None (recent hypothesis, no formal revalidation date)
    - CONFIDENCE_TIER: PLAUSIBLE — H4 hypothesis test result, not yet
      promoted to PROVEN status.
    """

    # ── Canonical research metadata (ClassVar) ──────────────────────────
    CONFIDENCE_TIER: ClassVar[str] = "PLAUSIBLE"

    min_pct: float = 60.0

    def matches_row(self, row: dict, orb_label: str) -> bool:
        val = row.get("atr_20_pct")
        if val is None:
            return False
        return val >= self.min_pct

    def matches_df(self, df: pd.DataFrame, orb_label: str) -> pd.Series:
        import pandas as pd

        if "atr_20_pct" not in df.columns:
            return pd.Series(False, index=df.index)
        return df["atr_20_pct"].notna() & (df["atr_20_pct"] >= self.min_pct)

    def describe(
        self,
        row: dict,
        orb_label: str,
        entry_model: str,
    ) -> list[AtomDescription]:
        """Own ATR-20 percentile gate. Pre-session, resolves at STARTUP.

        Threads CONFIDENCE_TIER (PLAUSIBLE) through.
        """
        _ = (orb_label, entry_model)
        observed = _atom_numeric(row.get("atr_20_pct"))
        missing = observed is None
        passes = None if missing else observed >= self.min_pct
        return [
            AtomDescription(
                name=f"Own ATR percentile >= {self.min_pct:g}",
                category="PRE_SESSION",
                resolves_at="STARTUP",
                passes=passes,
                feature_column="atr_20_pct",
                confidence_tier=self.CONFIDENCE_TIER,
                observed_value=observed,
                threshold=self.min_pct,
                comparator=">=",
                is_data_missing=missing,
                explanation=(
                    f"Require own-instrument ATR-20 rolling percentile "
                    f">= {self.min_pct:g}% (own-vol regime gate)."
                ),
            )
        ]


@dataclass(frozen=True)
class OvernightRangeFilter(StrategyFilter):
    """Filter by overnight range rolling percentile.

    Reads overnight_range_pct from daily_features (rolling 60d percentile, pre-computed).
    Selects days where overnight range is above a rolling threshold — high overnight
    activity predicts stronger breakout follow-through.
    Fail-closed: missing data means day is ineligible.

    April 2026 hypothesis H3: overnight_range as a session filter.
    NOTE: Prior claim of US-session specificity was WRONG — signal is Asian sessions.
    Prior WR spread claim (Q1=36.1%, Q5=62.6%) was pooled artifact (actual spread ~4.5%).
    @research-source calibration audit 2026-03-26, corrected 2026-03-27
    """

    min_pct: float = 60.0

    def matches_row(self, row: dict, orb_label: str) -> bool:
        val = row.get("overnight_range_pct")
        if val is None:
            return False
        return val >= self.min_pct

    def matches_df(self, df: pd.DataFrame, orb_label: str) -> pd.Series:
        import pandas as pd

        if "overnight_range_pct" not in df.columns:
            return pd.Series(False, index=df.index)
        return df["overnight_range_pct"].notna() & (df["overnight_range_pct"] >= self.min_pct)

    def describe(
        self,
        row: dict,
        orb_label: str,
        entry_model: str,
    ) -> list[AtomDescription]:
        """Overnight range rolling percentile gate. Pre-session, resolves at STARTUP."""
        _ = (orb_label, entry_model)
        observed = _atom_numeric(row.get("overnight_range_pct"))
        missing = observed is None
        passes = None if missing else observed >= self.min_pct
        return [
            AtomDescription(
                name=f"Overnight range percentile >= {self.min_pct:g}",
                category="PRE_SESSION",
                resolves_at="STARTUP",
                passes=passes,
                feature_column="overnight_range_pct",
                observed_value=observed,
                threshold=self.min_pct,
                comparator=">=",
                is_data_missing=missing,
                explanation=(
                    f"Require overnight range rolling percentile >= {self.min_pct:g}% "
                    f"(Asian-session activity gate)."
                ),
            )
        ]


@dataclass(frozen=True)
class GARCHForecastVolPctFilter(StrategyFilter):
    """Filter by GARCH(1,1) forecast vol rolling percentile.

    Reads ``garch_forecast_vol_pct`` from ``daily_features`` (rolling 252d
    percentile, pre-computed in ``pipeline.build_daily_features``).
    Fail-closed: missing data (None / NaN / pd.NA) means day is ineligible.

    Supports both directions:
    - ``direction="low"``: admit rows where ``garch_forecast_vol_pct <= pct_threshold``
      (quieter-than-usual vol regime).
    - ``direction="high"``: admit rows where ``garch_forecast_vol_pct >= pct_threshold``
      (noisier-than-usual vol regime).

    Wave 5 G5 deployment target: ``GARCH_VOL_PCT_LT20`` for MNQ × NYSE_OPEN
    × RR1.5, direction="low" — research finding from
    ``scripts/research/wave4_presession_t2t8.py`` (Phase B T2-T8 survivor,
    in_ExpR +0.240, WFE 1.00, p=0.042). Deployed as a rolling percentile
    rather than an absolute threshold because the ``garch_forecast_vol``
    distribution varies significantly across instruments (MNQ Q20 ≈ 0.159
    annualized vs MES Q20 ≈ 0.111) — a single absolute cutoff would either
    starve MNQ or over-trigger MES.

    Canonical metadata:
    - VALIDATED_FOR: research-provisional on MNQ (Phase B survivor); other
      instruments/sessions NOT YET validated.
    - LAST_REVALIDATED: 2026-04-11 (Wave 4 Phase B T2-T8)
    - CONFIDENCE_TIER: PLAUSIBLE — Phase B survivor with in_ExpR +0.240 and
      clean WFE but single-sample discovery; Criterion 8 OOS gate remains
      the promotion blocker (time-driven).

    @research-source scripts/research/wave4_presession_t2t8.py
    @revalidated-for 2026-04 Wave 5 Pathway B deployment
    @entry-models E2 (stop-market breakout)
    """

    # ── Canonical research metadata (ClassVar) ──────────────────────────
    CONFIDENCE_TIER: ClassVar[str] = "PLAUSIBLE"

    pct_threshold: float = 20.0
    direction: str = "low"  # "low" → <= threshold, "high" → >= threshold

    def __post_init__(self):
        if self.direction not in ("low", "high"):
            raise ValueError(
                f"GARCHForecastVolPctFilter direction must be 'low' or 'high', "
                f"got {self.direction!r}"
            )

    def matches_row(self, row: dict, orb_label: str) -> bool:
        _ = orb_label
        val = row.get("garch_forecast_vol_pct")
        if val is None:
            return False
        # pd.NA / NaN handling via _atom_numeric which returns None on missing
        coerced = _atom_numeric(val)
        if coerced is None:
            return False
        if self.direction == "low":
            return coerced <= self.pct_threshold
        return coerced >= self.pct_threshold

    def matches_df(self, df: pd.DataFrame, orb_label: str) -> pd.Series:
        _ = orb_label
        import pandas as pd

        if "garch_forecast_vol_pct" not in df.columns:
            return pd.Series(False, index=df.index)
        col = df["garch_forecast_vol_pct"]
        if self.direction == "low":
            return col.notna() & (col <= self.pct_threshold)
        return col.notna() & (col >= self.pct_threshold)

    def describe(
        self,
        row: dict,
        orb_label: str,
        entry_model: str,
    ) -> list[AtomDescription]:
        """GARCH forecast vol percentile gate. Pre-session, resolves at STARTUP."""
        _ = (orb_label, entry_model)
        observed = _atom_numeric(row.get("garch_forecast_vol_pct"))
        missing = observed is None
        if missing:
            passes = None
        elif self.direction == "low":
            passes = observed <= self.pct_threshold
        else:
            passes = observed >= self.pct_threshold
        comparator = "<=" if self.direction == "low" else ">="
        regime_word = "low-vol" if self.direction == "low" else "high-vol"
        return [
            AtomDescription(
                name=f"GARCH forecast vol percentile {comparator} {self.pct_threshold:g}",
                category="PRE_SESSION",
                resolves_at="STARTUP",
                passes=passes,
                feature_column="garch_forecast_vol_pct",
                confidence_tier=self.CONFIDENCE_TIER,
                observed_value=observed,
                threshold=self.pct_threshold,
                comparator=comparator,
                is_data_missing=missing,
                explanation=(
                    f"Require GARCH forecast vol rolling percentile "
                    f"{comparator} {self.pct_threshold:g}% ({regime_word} regime gate)."
                ),
            )
        ]


@dataclass(frozen=True)
class OvernightRangeAbsFilter(StrategyFilter):
    """Filter by absolute overnight range (points).

    Gates on the Asia-session range (09:00-17:00 Brisbane) in raw points.
    Higher overnight activity predicts stronger breakout follow-through on
    US sessions. Values from daily_features.overnight_range (pre-computed).

    WARNING — LOOK-AHEAD FOR ASIAN SESSIONS:
    overnight_range is computed from 09:00-17:00 Brisbane. Sessions starting
    INSIDE this window (CME_REOPEN, TOKYO_OPEN, BRISBANE_1025, SINGAPORE_OPEN)
    would use future price data. This filter MUST ONLY be routed to sessions
    starting AFTER 17:00 Brisbane (LONDON_METALS through NYSE_CLOSE) via
    get_filters_for_grid(). DO NOT add to BASE_GRID_FILTERS.

    Thresholds are absolute (not normalized by ATR). Different instruments
    have different overnight ranges — discovery handles the mismatch.

    Fail-closed: missing data = ineligible day.

    @research-source research/output/confluence_program/phase1_run.py
    @entry-models E2
    @revalidated-for E2
    """

    min_range: float

    def matches_row(self, row: dict, orb_label: str) -> bool:
        val = row.get("overnight_range")
        if val is None:
            return False  # fail-closed
        return val >= self.min_range

    def matches_df(self, df: pd.DataFrame, orb_label: str) -> pd.Series:
        import pandas as pd

        if "overnight_range" not in df.columns:
            return pd.Series(False, index=df.index)
        return df["overnight_range"].notna() & (df["overnight_range"] >= self.min_range)

    def describe(
        self,
        row: dict,
        orb_label: str,
        entry_model: str,
    ) -> list[AtomDescription]:
        """Absolute overnight range (points) gate. Pre-session, resolves at STARTUP.

        LOOK-AHEAD NOTE: overnight_range is 09:00-17:00 Brisbane. Caller routing
        (via get_filters_for_grid) restricts this filter to post-17:00 sessions.
        describe() trusts the routing and emits the atom as PRE_SESSION.
        """
        _ = (orb_label, entry_model)
        observed = _atom_numeric(row.get("overnight_range"))
        missing = observed is None
        passes = None if missing else observed >= self.min_range
        return [
            AtomDescription(
                name=f"Overnight range >= {self.min_range:g} pts",
                category="PRE_SESSION",
                resolves_at="STARTUP",
                passes=passes,
                feature_column="overnight_range",
                observed_value=observed,
                threshold=self.min_range,
                comparator=">=",
                is_data_missing=missing,
                explanation=(
                    f"Require absolute overnight range (09:00-17:00 Brisbane) "
                    f">= {self.min_range:g} points. Routed only to post-17:00 sessions."
                ),
            )
        ]


@dataclass(frozen=True)
class PrevDayRangeNormFilter(StrategyFilter):
    """Filter by prior day range normalized by ATR-20.

    Gates on prev_day_range / atr_20 >= min_ratio. Higher ratio means
    yesterday was more volatile relative to the recent regime.

    No lookahead: prev_day_range and atr_20 are strictly prior-day values,
    safe for ALL sessions.

    Fail-closed: missing prev_day_range or atr_20 = ineligible day.

    @research-source scripts/research/scan_presession_features.py
    @research-source scripts/research/scan_presession_t2t8.py
    @entry-models E2
    @revalidated-for E2 (Apr 2026)

    Canonical metadata (consumed by describe() / eligibility adapter):
    - VALIDATED_FOR: empty (applies all sessions — no instrument restriction)
    - LAST_REVALIDATED: 2026-04-02 (presession scan refresh)
    - CONFIDENCE_TIER: PROVEN (BH FDR survivor, year-stable)
    """

    # ── Canonical research metadata (ClassVar) ──────────────────────────
    LAST_REVALIDATED: ClassVar[date] = date(2026, 4, 2)
    CONFIDENCE_TIER: ClassVar[str] = "PROVEN"

    min_ratio: float

    def matches_row(self, row: dict, orb_label: str) -> bool:
        pdr = row.get("prev_day_range")
        atr = row.get("atr_20")
        if pdr is None or atr is None or atr <= 0:
            return False  # fail-closed
        return pdr / atr >= self.min_ratio

    def matches_df(self, df: pd.DataFrame, orb_label: str) -> pd.Series:
        import pandas as pd

        if "prev_day_range" not in df.columns or "atr_20" not in df.columns:
            return pd.Series(False, index=df.index)
        valid = df["prev_day_range"].notna() & df["atr_20"].notna() & (df["atr_20"] > 0)
        ratio = df["prev_day_range"] / df["atr_20"].replace(0, float("nan"))
        return valid & (ratio >= self.min_ratio)

    def describe(
        self,
        row: dict,
        orb_label: str,
        entry_model: str,
    ) -> list[AtomDescription]:
        """prev_day_range / atr_20 gate. Pre-session, resolves at STARTUP.

        atr_20 <= 0 is treated as DATA_MISSING (data corruption, not FAIL).
        matches_row() fails-closed on atr_20 <= 0; describe() surfaces it as
        missing so the report shows 'data unavailable' instead of a spurious
        comparison against observed=None.

        Type mismatches (e.g. string in prev_day_range column from schema
        drift) are explicitly caught and reported via the atom's
        error_message field. The eligibility adapter aggregates non-None
        error_messages into report.build_errors so the diagnostic trail is
        preserved without parallel error-tracking. Status surfaces as
        DATA_MISSING (infrastructure problem, not a trading signal).
        """
        _ = (orb_label, entry_model)
        raw_pdr = row.get("prev_day_range")
        raw_atr = row.get("atr_20")
        pdr = _atom_numeric(raw_pdr)
        atr = _atom_numeric(raw_atr)
        error_message: str | None = None
        observed: float | None = None
        missing = False

        if pdr is None or atr is None:
            # One or both raw inputs missing/non-numeric. Distinguish "field
            # absent" (missing) from "field present but wrong type" (also
            # missing, but with diagnostic message).
            missing = True
            if raw_pdr is not None and pdr is None and not _atom_is_missing(raw_pdr):
                error_message = (
                    f"PDR: type mismatch on prev_day_range — expected numeric, "
                    f"got {type(raw_pdr).__name__}({raw_pdr!r})"
                )
            elif raw_atr is not None and atr is None and not _atom_is_missing(raw_atr):
                error_message = (
                    f"PDR: type mismatch on atr_20 — expected numeric, "
                    f"got {type(raw_atr).__name__}({raw_atr!r})"
                )
        elif atr <= 0:
            missing = True
            error_message = f"PDR: invalid atr_20={atr!r} (must be > 0)"
        else:
            observed = pdr / atr

        passes: bool | None
        if missing or observed is None:
            passes = None
        else:
            passes = observed >= self.min_ratio
        return [
            AtomDescription(
                name=f"prev_day_range / atr_20 >= {self.min_ratio:g}",
                category="PRE_SESSION",
                resolves_at="STARTUP",
                passes=passes,
                feature_column="prev_day_range",
                observed_value=observed,
                threshold=self.min_ratio,
                comparator=">=",
                is_data_missing=missing,
                last_revalidated=self.LAST_REVALIDATED,
                confidence_tier=self.CONFIDENCE_TIER,
                error_message=error_message,
                explanation=(
                    f"Require prior-day range normalized by ATR-20 "
                    f">= {self.min_ratio:g} (prior-day expansion gate)."
                ),
            )
        ]


@dataclass(frozen=True)
class GapNormFilter(StrategyFilter):
    """Filter by absolute gap size normalized by ATR-20.

    Gates on abs(gap_open_points) / atr_20 >= min_ratio. Higher ratio means
    a larger overnight gap relative to the regime.

    No lookahead: gap_open_points is computed from prior close to current open,
    safe for ALL sessions.

    Fail-closed: missing gap_open_points or atr_20 = ineligible day.

    @research-source scripts/research/scan_presession_features.py
    @research-source scripts/research/scan_presession_t2t8.py
    @entry-models E2
    @revalidated-for E2 (Apr 2026)

    Canonical metadata (consumed by describe() / eligibility adapter):
    - VALIDATED_FOR: MGC CME_REOPEN only — gap-shock signal is gold-specific
      and CME-reopen-specific per scan_presession_t2t8.py
    - LAST_REVALIDATED: 2026-04-02
    - CONFIDENCE_TIER: PROVEN
    """

    # ── Canonical research metadata (ClassVar) ──────────────────────────
    VALIDATED_FOR: ClassVar[tuple[tuple[str, str], ...]] = (
        ("MGC", "CME_REOPEN"),
    )
    LAST_REVALIDATED: ClassVar[date] = date(2026, 4, 2)
    CONFIDENCE_TIER: ClassVar[str] = "PROVEN"

    min_ratio: float

    def matches_row(self, row: dict, orb_label: str) -> bool:
        gap = row.get("gap_open_points")
        atr = row.get("atr_20")
        if gap is None or atr is None or atr <= 0:
            return False  # fail-closed
        return abs(gap) / atr >= self.min_ratio

    def matches_df(self, df: pd.DataFrame, orb_label: str) -> pd.Series:
        import pandas as pd

        if "gap_open_points" not in df.columns or "atr_20" not in df.columns:
            return pd.Series(False, index=df.index)
        valid = df["gap_open_points"].notna() & df["atr_20"].notna() & (df["atr_20"] > 0)
        ratio = df["gap_open_points"].abs() / df["atr_20"].replace(0, float("nan"))
        return valid & (ratio >= self.min_ratio)

    def describe(
        self,
        row: dict,
        orb_label: str,
        entry_model: str,
    ) -> list[AtomDescription]:
        """abs(gap_open_points) / atr_20 gate. Pre-session, resolves at STARTUP.

        atr_20 <= 0 treated as DATA_MISSING (see PrevDayRangeNormFilter note).
        Type mismatches caught explicitly and reported via error_message.
        """
        _ = (orb_label, entry_model)
        raw_gap = row.get("gap_open_points")
        raw_atr = row.get("atr_20")
        gap = _atom_numeric(raw_gap)
        atr = _atom_numeric(raw_atr)
        error_message: str | None = None
        observed: float | None = None
        missing = False

        if gap is None or atr is None:
            missing = True
            if raw_gap is not None and gap is None and not _atom_is_missing(raw_gap):
                error_message = (
                    f"GAP: type mismatch on gap_open_points — expected numeric, "
                    f"got {type(raw_gap).__name__}({raw_gap!r})"
                )
            elif raw_atr is not None and atr is None and not _atom_is_missing(raw_atr):
                error_message = (
                    f"GAP: type mismatch on atr_20 — expected numeric, "
                    f"got {type(raw_atr).__name__}({raw_atr!r})"
                )
        elif atr <= 0:
            missing = True
            error_message = f"GAP: invalid atr_20={atr!r} (must be > 0)"
        else:
            observed = abs(gap) / atr

        passes: bool | None
        if missing or observed is None:
            passes = None
        else:
            passes = observed >= self.min_ratio
        return [
            AtomDescription(
                name=f"abs(gap) / atr_20 >= {self.min_ratio:g}",
                category="PRE_SESSION",
                resolves_at="STARTUP",
                passes=passes,
                feature_column="gap_open_points",
                observed_value=observed,
                threshold=self.min_ratio,
                comparator=">=",
                is_data_missing=missing,
                validated_for=self.VALIDATED_FOR,
                last_revalidated=self.LAST_REVALIDATED,
                confidence_tier=self.CONFIDENCE_TIER,
                error_message=error_message,
                explanation=(
                    f"Require absolute overnight gap normalized by ATR-20 "
                    f">= {self.min_ratio:g} (gap-shock gate)."
                ),
            )
        ]


@dataclass(frozen=True)
class DirectionFilter(StrategyFilter):
    """Filter by breakout direction (long/short only)."""

    direction: str = "long"  # "long" or "short"

    def matches_row(self, row: dict, orb_label: str) -> bool:
        break_dir = row.get(f"orb_{orb_label}_break_dir")
        if break_dir is None:
            return False  # fail-closed
        return break_dir == self.direction

    def matches_df(self, df: pd.DataFrame, orb_label: str) -> pd.Series:
        import pandas as pd

        col = f"orb_{orb_label}_break_dir"
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        return df[col] == self.direction

    def describe(
        self,
        row: dict,
        orb_label: str,
        entry_model: str,
    ) -> list[AtomDescription]:
        """Direction gate. Directional, resolves at BREAK_DETECTED.

        IMPORTANT: pre-break, the break direction is genuinely unknown (PENDING),
        not DATA_MISSING. This is distinct from the old parallel-model bug
        which marked DirectionFilter as NOT_APPLICABLE_DIRECTION unconditionally.
        The filter DOES apply; it just hasn't resolved yet.
        """
        _ = entry_model
        col = f"orb_{orb_label}_break_dir"
        raw = row.get(col)
        missing = _atom_is_missing(raw)
        passes = None if missing else (raw == self.direction)
        return [
            AtomDescription(
                name=f"Break direction == {self.direction}",
                category="DIRECTIONAL",
                resolves_at="BREAK_DETECTED",
                passes=passes,
                feature_column=col,
                observed_value=None if missing else raw,
                threshold=self.direction,
                comparator="==",
                is_data_missing=False,  # PENDING, not truly missing
                explanation=(
                    f"Strategy trades only {self.direction}-direction breaks. "
                    f"Resolves once the break direction is observed."
                ),
            )
        ]


@dataclass(frozen=True)
class CalendarSkipFilter(StrategyFilter):
    """Skip trading on calendar event days (NFP, OPEX, Friday).

    Portfolio overlay — not part of discovery grid.
    All three flags are pre-computed in daily_features.
    """

    skip_nfp: bool = True
    skip_opex: bool = True
    skip_friday_session: str | None = None  # e.g. "CME_REOPEN" or None

    def matches_row(self, row: dict, orb_label: str) -> bool:
        if self.skip_nfp and row.get("is_nfp_day"):
            return False
        if self.skip_opex and row.get("is_opex_day"):
            return False
        if self.skip_friday_session and orb_label == self.skip_friday_session:
            if row.get("is_friday"):
                return False
        return True

    def matches_df(self, df: pd.DataFrame, orb_label: str) -> pd.Series:
        import pandas as pd

        mask = pd.Series(True, index=df.index)
        if self.skip_nfp and "is_nfp_day" in df.columns:
            mask = mask & ~df["is_nfp_day"].fillna(False).astype(bool)
        if self.skip_opex and "is_opex_day" in df.columns:
            mask = mask & ~df["is_opex_day"].fillna(False).astype(bool)
        if self.skip_friday_session and orb_label == self.skip_friday_session:
            if "is_friday" in df.columns:
                mask = mask & ~df["is_friday"].fillna(False).astype(bool)
        return mask

    def describe(
        self,
        row: dict,
        orb_label: str,
        entry_model: str,
    ) -> list[AtomDescription]:
        """Calendar overlay summary. OVERLAY, resolves at STARTUP.

        NOTE: This is the filter-as-filter_type path (deprecated in discovery).
        The eligibility builder's richer calendar overlay (which can emit
        HALF_SIZE with size_multiplier=0.5) is handled separately in
        trading_app/eligibility/builder.py. This override exists to satisfy
        the drift check that every StrategyFilter is self-describing.
        """
        _ = entry_model
        passes = self.matches_row(row, orb_label)
        return [
            AtomDescription(
                name="Calendar skip overlay",
                category="OVERLAY",
                resolves_at="STARTUP",
                passes=passes,
                explanation=(
                    f"Skip trading on "
                    f"NFP={self.skip_nfp} / OPEX={self.skip_opex} / "
                    f"Friday({self.skip_friday_session or 'off'}). "
                    f"Rich overlay (HALF_SIZE, sizing) handled by the "
                    f"eligibility builder separately."
                ),
            )
        ]


@dataclass(frozen=True)
class DayOfWeekSkipFilter(StrategyFilter):
    """Skip trading on specific days of the week.

    Uses day_of_week column (0=Mon..6=Sun, Python weekday convention).
    Fail-closed: missing day_of_week means day is ineligible.

    Canonical metadata:
    - VALIDATED_FOR: empty (DOW signal applies broadly; lane routing
      decisions live in get_filters_for_grid, not here)
    - LAST_REVALIDATED: None
    - CONFIDENCE_TIER: derived from skip_days — Monday-skip (NOMON) is
      PLAUSIBLE per confluence research; Tuesday/Friday-skip (NOTUE,
      NOFRI) are LEGACY (removed from grid Mar 2026, retained for DB
      compat). The describe() method derives the tier from the actual
      skip set so each instance reports its own tier accurately.
    """

    skip_days: tuple[int, ...] = ()

    def matches_row(self, row: dict, orb_label: str) -> bool:
        dow = row.get("day_of_week")
        if dow is None:
            return False  # fail-closed
        return dow not in self.skip_days

    def matches_df(self, df: pd.DataFrame, orb_label: str) -> pd.Series:
        import pandas as pd

        if "day_of_week" not in df.columns:
            return pd.Series(False, index=df.index)
        # NaN → fail-closed (notna check), then exclude skip_days
        return df["day_of_week"].notna() & ~df["day_of_week"].isin(self.skip_days)

    def _confidence_tier(self) -> str:
        """Derive tier from skip_days. NOMON (Monday=0) = PLAUSIBLE
        per confluence research; Tuesday/Friday skips = LEGACY (removed
        from grid Mar 2026 but retained for DB compat).

        If the skip set contains both PLAUSIBLE and LEGACY days, the
        lower tier (LEGACY) wins — conservative provenance disclosure.
        """
        if not self.skip_days:
            return "UNKNOWN"
        # Friday=4, Tuesday=1 are LEGACY
        if any(d in (1, 4) for d in self.skip_days):
            return "LEGACY"
        # Monday=0 only path
        return "PLAUSIBLE"

    def describe(
        self,
        row: dict,
        orb_label: str,
        entry_model: str,
    ) -> list[AtomDescription]:
        """Day-of-week skip gate. Pre-session, resolves at STARTUP.

        day_of_week follows Python weekday convention (0=Mon..6=Sun) and
        is derived from trading_day. Fail-closed on missing DOW.

        Threads instance-derived confidence_tier through (LEGACY for
        Tuesday/Friday skips, PLAUSIBLE for Monday-only).
        """
        _ = (orb_label, entry_model)
        dow_num = _atom_numeric(row.get("day_of_week"))
        missing = dow_num is None
        dow_int = int(dow_num) if dow_num is not None else None
        passes = None if missing else (dow_int not in self.skip_days)
        skip_list = list(self.skip_days)
        return [
            AtomDescription(
                name=f"Day of week not in {skip_list}",
                category="PRE_SESSION",
                resolves_at="STARTUP",
                passes=passes,
                feature_column="day_of_week",
                observed_value=dow_int,
                threshold=skip_list,
                comparator="not in",
                is_data_missing=missing,
                confidence_tier=self._confidence_tier(),
                explanation=(
                    f"Skip trading on day_of_week {skip_list} "
                    f"(0=Mon, 6=Sun, Python weekday convention)."
                ),
            )
        ]


@dataclass(frozen=True)
class ATRVelocityFilter(StrategyFilter):
    """Skip sessions when ATR is actively contracting AND ORB compression is Neutral or Compressed.

    @research-source research_avoid_crosscheck.py (Feb 2026, E0/old sessions — STALE)
    @revalidated-for E1/E2 event-based sessions (Mar 2026):
      - MGC: CONFIRMED. 10/10 years negative at TOKYO_OPEN E1.
      - MES: AVOID is real but MES baselines already negative — redundant, not actionable
             as standalone filter. 127/293 BH survivors but effect is baseline-wide.
      - MNQ: NO SIGNAL. 0/169 BH FDR survivors across 11 sessions. Do NOT apply.
      See compressed_spring.md for full revalidation.

    Signal is fully pre-entry — both atr_vel_regime and compression_tier are
    computed from prior-days data and known at ORB close.

    Logic: skip when BOTH conditions hold:
      1. atr_vel_regime == 'Contracting'  (today ATR < 95% of prior-5-day avg)
      2. orb_{label}_compression_tier in ('Neutral', 'Compressed')
         (Expanded is OK — Contracting+Expanded has mixed/weaker signal)

    Applied to: MGC CME_REOPEN and TOKYO_OPEN only (where signal is confirmed).
    NOT applied to: MNQ (no signal), MES (baseline already negative — redundant).

    Fail-open: missing data (warm-up period) -> trade is allowed.

    Canonical metadata (consumed by describe() / eligibility adapter):
    - VALIDATED_FOR: ((MGC, CME_REOPEN), (MGC, TOKYO_OPEN)) — derived
      mechanically from apply_to_sessions × instrument restriction.
      The two sources of truth (apply_to_sessions tuple and VALIDATED_FOR
      ClassVar) MUST agree — drift check #N enforces this if added.
    - LAST_REVALIDATED: 2026-03-18 (compressed_spring revalidation)
    - CONFIDENCE_TIER: PROVEN
    """

    filter_type: str = "ATR_VEL"
    description: str = "Skip Contracting ATR × Neutral/Compressed ORB sessions"
    apply_to_sessions: tuple[str, ...] = ("CME_REOPEN", "TOKYO_OPEN")

    # ── Canonical research metadata (ClassVar) ──────────────────────────
    VALIDATED_FOR: ClassVar[tuple[tuple[str, str], ...]] = (
        ("MGC", "CME_REOPEN"),
        ("MGC", "TOKYO_OPEN"),
    )
    LAST_REVALIDATED: ClassVar[date] = date(2026, 3, 18)
    CONFIDENCE_TIER: ClassVar[str] = "PROVEN"

    def matches_row(self, row: dict, orb_label: str) -> bool:
        if orb_label not in self.apply_to_sessions:
            return True  # not a monitored session — allow trade
        vel_regime = row.get("atr_vel_regime")
        if vel_regime != "Contracting":
            return True  # only skip on contracting ATR
        comp_tier = row.get(f"orb_{orb_label}_compression_tier")
        if comp_tier is None:
            return True  # warm-up: no rolling data yet — fail-open
        return comp_tier == "Expanded"  # Neutral/Compressed → skip; Expanded → allow

    def matches_df(self, df: pd.DataFrame, orb_label: str) -> pd.Series:
        import pandas as pd

        if orb_label not in self.apply_to_sessions:
            return pd.Series(True, index=df.index)
        # Start with all True, then exclude Contracting + non-Expanded
        comp_col = f"orb_{orb_label}_compression_tier"
        is_contracting = df.get("atr_vel_regime") == "Contracting"
        if comp_col in df.columns:
            is_non_expanded = df[comp_col].notna() & (df[comp_col] != "Expanded")
        else:
            # No compression data — fail-open
            return pd.Series(True, index=df.index)
        # Skip when BOTH contracting AND non-expanded
        return ~(is_contracting & is_non_expanded)

    def describe(
        self,
        row: dict,
        orb_label: str,
        entry_model: str,
    ) -> list[AtomDescription]:
        """ATR velocity overlay. Delegates to canonical matches_row().

        CRITICAL CANONICAL DELEGATION: this method MUST call
        self.matches_row() directly for the passes bool. Do NOT re-encode
        the vel_regime/compression_tier comparison — the canonical
        matches_row has warm-up fail-open semantics (missing compression
        tier -> True, meaning PASS not DATA_MISSING). A parallel re-implementation
        would drift from canonical and mislabel warm-up days as FAIL. The
        prior parallel-model bug did exactly this.

        Non-monitored sessions (anything outside apply_to_sessions) are
        marked NOT_APPLICABLE so the eligibility report shows the overlay
        is scoped to the two MGC sessions where signal was confirmed.
        """
        _ = entry_model
        if orb_label not in self.apply_to_sessions:
            return [
                AtomDescription(
                    name="ATR velocity overlay",
                    category="OVERLAY",
                    resolves_at="ORB_FORMATION",
                    passes=True,
                    is_not_applicable=True,
                    not_applicable_reason=(
                        f"Session '{orb_label}' is not in the monitored "
                        f"set {list(self.apply_to_sessions)} — ATR velocity "
                        f"signal is confirmed only for MGC CME_REOPEN and "
                        f"MGC TOKYO_OPEN."
                    ),
                    validated_for=self.VALIDATED_FOR,
                    last_revalidated=self.LAST_REVALIDATED,
                    confidence_tier=self.CONFIDENCE_TIER,
                    explanation=(
                        "ATR velocity overlay: skip when ATR is actively "
                        "contracting AND the ORB is Neutral/Compressed."
                    ),
                )
            ]
        # Canonical delegation — warm-up fail-open is inherited.
        passes = self.matches_row(row, orb_label)
        vel_regime = row.get("atr_vel_regime")
        comp_tier = row.get(f"orb_{orb_label}_compression_tier")
        vel_str = "missing" if _atom_is_missing(vel_regime) else str(vel_regime)
        tier_str = "missing" if _atom_is_missing(comp_tier) else str(comp_tier)
        return [
            AtomDescription(
                name="Not (Contracting ATR + Neutral/Compressed ORB)",
                category="OVERLAY",
                resolves_at="ORB_FORMATION",
                passes=passes,
                feature_column="atr_vel_regime",
                observed_value=f"vel={vel_str}, tier={tier_str}",
                threshold="not (Contracting & non-Expanded)",
                comparator="!=",
                is_data_missing=False,  # warm-up is fail-open, not missing
                validated_for=self.VALIDATED_FOR,
                last_revalidated=self.LAST_REVALIDATED,
                confidence_tier=self.CONFIDENCE_TIER,
                explanation=(
                    "Skip sessions when ATR is actively contracting AND the "
                    "ORB is Neutral or Compressed. Fail-open on warm-up "
                    "(missing compression tier -> PASS, matching canonical)."
                ),
            )
        ]


# Default ATR velocity overlay — applied as portfolio-level overlay.
ATR_VELOCITY_OVERLAY = ATRVelocityFilter()


@dataclass(frozen=True)
class DoubleBreakFilter(StrategyFilter):
    """Skip days where the ORB had a double-break (both sides breached).

    Double-break = mean-reversion regime. ORB breakout = momentum strategy.
    Filtering these out selects clean momentum days only.
    """

    exclude: bool = True

    def matches_row(self, row: dict, orb_label: str) -> bool:
        db = row.get(f"orb_{orb_label}_double_break")
        if db is None:
            return True  # missing data → pass-through
        if self.exclude:
            return not db
        return db

    def matches_df(self, df: pd.DataFrame, orb_label: str) -> pd.Series:
        import pandas as pd

        col = f"orb_{orb_label}_double_break"
        if col not in df.columns:
            return pd.Series(True, index=df.index)  # missing → pass-through
        if self.exclude:
            # NaN → pass-through (True), non-double-break → True, double-break → False
            return df[col].isna() | ~df[col].astype(bool)
        else:
            return df[col].fillna(False).astype(bool)

    def describe(
        self,
        row: dict,
        orb_label: str,
        entry_model: str,
    ) -> list[AtomDescription]:
        """DoubleBreakFilter is LOOK-AHEAD and has no real-time semantics.

        double_break checks whether BOTH ORB sides were breached during the
        FULL session (i.e. after trade entry). It was removed from production
        discovery 2026-02 (6 validated strategies proved to be artifacts).
        describe() always returns NOT_APPLICABLE so the eligibility report
        surfaces the filter as dead rather than claiming it passed/failed.
        """
        _ = (row, orb_label, entry_model)
        return [
            AtomDescription(
                name="Double-break filter",
                category="INTRA_SESSION",
                resolves_at="ORB_FORMATION",
                passes=None,
                is_not_applicable=True,
                not_applicable_reason=(
                    "double_break is look-ahead: it checks whether BOTH ORB "
                    "sides were breached during the full session (after trade "
                    "entry). Cannot be used as a real-time gate. Removed from "
                    "production discovery 2026-02."
                ),
                explanation=(
                    "Deprecated filter — kept only for backward-compat with "
                    "legacy strategy ids."
                ),
            )
        ]


@dataclass(frozen=True)
class BreakSpeedFilter(StrategyFilter):
    """Filter by break delay: minutes from ORB end to first break.

    Fast breaks (low delay) indicate momentum / conviction.
    Slow breaks indicate grinding / indecision.

    Uses orb_{label}_break_delay_min from daily_features.
    Fail-closed: missing data means day is ineligible.

    Canonical metadata (consumed by describe() / eligibility adapter):
    - VALIDATED_FOR: MNQ at 5 sessions + MGC CME_REOPEN per
      break_speed_signal_retest.md (Apr 2026). Per-instrument validity
      enforces NOT_APPLICABLE_INSTRUMENT for non-validated lanes.
    - LAST_REVALIDATED: 2026-04-01 (Phase 3 retest)
    - CONFIDENCE_TIER: PROVEN
    """

    # ── Canonical research metadata (ClassVar) ──────────────────────────
    VALIDATED_FOR: ClassVar[tuple[tuple[str, str], ...]] = (
        ("MNQ", "NYSE_CLOSE"),
        ("MNQ", "NYSE_OPEN"),
        ("MNQ", "TOKYO_OPEN"),
        ("MNQ", "LONDON_METALS"),
        ("MNQ", "CME_REOPEN"),
        ("MGC", "CME_REOPEN"),
    )
    LAST_REVALIDATED: ClassVar[date] = date(2026, 4, 1)
    CONFIDENCE_TIER: ClassVar[str] = "PROVEN"

    max_delay_min: float = 5.0

    def matches_row(self, row: dict, orb_label: str) -> bool:
        delay = row.get(f"orb_{orb_label}_break_delay_min")
        if delay is None:
            return False  # fail-closed: no break or no data
        return delay <= self.max_delay_min

    def matches_df(self, df: pd.DataFrame, orb_label: str) -> pd.Series:
        import pandas as pd

        col = f"orb_{orb_label}_break_delay_min"
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        return df[col].notna() & (df[col] <= self.max_delay_min)

    def describe(
        self,
        row: dict,
        orb_label: str,
        entry_model: str,
    ) -> list[AtomDescription]:
        """Break-speed gate. Intra-session, resolves at BREAK_DETECTED.

        E2-excluded: break delay is unknown until the break actually occurs,
        but the E2 entry order must already be placed. Canonical gate via
        _e2_look_ahead_reason (E2_EXCLUDED_FILTER_SUBSTRINGS: '_FAST').

        Threads canonical ClassVar metadata into the atom so the eligibility
        adapter can apply NOT_APPLICABLE_INSTRUMENT for non-validated lanes
        and STALE_VALIDATION for aged research.
        """
        col = f"orb_{orb_label}_break_delay_min"
        if entry_model == "E2":
            reason = _e2_look_ahead_reason(self.filter_type)
            if reason is not None:
                return [
                    AtomDescription(
                        name=f"Break delay <= {self.max_delay_min:g} min",
                        category="INTRA_SESSION",
                        resolves_at="BREAK_DETECTED",
                        passes=None,
                        feature_column=col,
                        threshold=self.max_delay_min,
                        comparator="<=",
                        is_not_applicable=True,
                        not_applicable_reason=reason,
                        validated_for=self.VALIDATED_FOR,
                        last_revalidated=self.LAST_REVALIDATED,
                        confidence_tier=self.CONFIDENCE_TIER,
                        explanation=(
                            "Break-speed gate does not apply to E2 entries — "
                            "break delay is unknown at E2 order placement."
                        ),
                    )
                ]
        observed = _atom_numeric(row.get(col))
        missing = observed is None
        passes = None if missing else observed <= self.max_delay_min
        return [
            AtomDescription(
                name=f"Break delay <= {self.max_delay_min:g} min",
                category="INTRA_SESSION",
                resolves_at="BREAK_DETECTED",
                passes=passes,
                feature_column=col,
                observed_value=observed,
                threshold=self.max_delay_min,
                comparator="<=",
                is_data_missing=missing,
                validated_for=self.VALIDATED_FOR,
                last_revalidated=self.LAST_REVALIDATED,
                confidence_tier=self.CONFIDENCE_TIER,
                explanation=(
                    f"Require break within {self.max_delay_min:g} minutes of ORB end "
                    f"(momentum conviction gate)."
                ),
            )
        ]


@dataclass(frozen=True)
class BreakBarContinuesFilter(StrategyFilter):
    """Filter by break bar direction: does the break bar close in the break direction?

    A break bar that closes as a green candle (for longs) or red candle
    (for shorts) shows conviction. A reversal candle at the break point
    = weak breakout.

    Uses orb_{label}_break_bar_continues from daily_features.
    Fail-closed: missing data means day is ineligible.

    Canonical metadata (consumed by describe() / eligibility adapter):
    - VALIDATED_FOR: empty (E1-only filter, applies broadly within E1)
    - LAST_REVALIDATED: None (legacy filter, no recent revalidation)
    - CONFIDENCE_TIER: PLAUSIBLE — confluence boost in some sessions but
      not a standalone PROVEN signal. E2-incompatible (look-ahead).
    """

    # ── Canonical research metadata (ClassVar) ──────────────────────────
    CONFIDENCE_TIER: ClassVar[str] = "PLAUSIBLE"

    require_continues: bool = True

    def matches_row(self, row: dict, orb_label: str) -> bool:
        continues = row.get(f"orb_{orb_label}_break_bar_continues")
        if continues is None:
            return False  # fail-closed
        return continues == self.require_continues

    def matches_df(self, df: pd.DataFrame, orb_label: str) -> pd.Series:
        import pandas as pd

        col = f"orb_{orb_label}_break_bar_continues"
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        return df[col].notna() & (df[col] == self.require_continues)

    def describe(
        self,
        row: dict,
        orb_label: str,
        entry_model: str,
    ) -> list[AtomDescription]:
        """Break-bar continuation gate. Intra-session, resolves at CONFIRM_COMPLETE.

        E2-excluded: bar-close direction is only known once the break bar
        closes, but the E2 entry order must already be placed. Canonical gate
        via _e2_look_ahead_reason (E2_EXCLUDED_FILTER_SUBSTRINGS: '_CONT').

        Threads CONFIDENCE_TIER (PLAUSIBLE) through the atom.
        """
        col = f"orb_{orb_label}_break_bar_continues"
        if entry_model == "E2":
            reason = _e2_look_ahead_reason(self.filter_type)
            if reason is not None:
                return [
                    AtomDescription(
                        name="Break bar closes in break direction",
                        category="INTRA_SESSION",
                        resolves_at="CONFIRM_COMPLETE",
                        passes=None,
                        feature_column=col,
                        threshold=self.require_continues,
                        comparator="==",
                        is_not_applicable=True,
                        not_applicable_reason=reason,
                        confidence_tier=self.CONFIDENCE_TIER,
                        explanation=(
                            "Break-bar continuation gate does not apply to E2 entries — "
                            "bar-close direction is unknown at E2 order placement."
                        ),
                    )
                ]
        raw = row.get(col)
        missing = _atom_is_missing(raw)
        passes = None if missing else bool(raw) == self.require_continues
        return [
            AtomDescription(
                name="Break bar closes in break direction",
                category="INTRA_SESSION",
                resolves_at="CONFIRM_COMPLETE",
                passes=passes,
                feature_column=col,
                observed_value=None if missing else bool(raw),
                threshold=self.require_continues,
                comparator="==",
                is_data_missing=missing,
                confidence_tier=self.CONFIDENCE_TIER,
                explanation=(
                    "Require the break bar to close in the break direction "
                    "(conviction gate)."
                ),
            )
        ]


@dataclass(frozen=True)
class PitRangeFilter(StrategyFilter):
    """Filter by exchange pit range normalized by ATR-20.

    Gates on pit_range_atr >= min_ratio. Higher ratio means the CME pit session
    was more active relative to the recent volatility regime.

    Zero look-ahead: pit closes 21:00 UTC, CME_REOPEN starts 23:00 UTC.
    Uses prior-day pit range (shift-1 in exchange_statistics).
    Fail-closed: missing pit_range_atr = ineligible day.

    @research-source scripts/research/exchange_range_t2t8.py
    @entry-models E1, E2
    @revalidated-for E1 (Apr 2026), E2 concordance +3-4pp

    Canonical metadata (consumed by describe() / eligibility adapter):
    - VALIDATED_FOR: 3/3 instruments at CME_REOPEN per scan_presession_t2t8
    - LAST_REVALIDATED: 2026-04-04 (T1-T8 confluence pass)
    - CONFIDENCE_TIER: PROVEN (DSR-significant, BH FDR survivor)
    """

    # ── Canonical research metadata (ClassVar — not dataclass fields) ────
    VALIDATED_FOR: ClassVar[tuple[tuple[str, str], ...]] = (
        ("MGC", "CME_REOPEN"),
        ("MES", "CME_REOPEN"),
        ("MNQ", "CME_REOPEN"),
    )
    LAST_REVALIDATED: ClassVar[date] = date(2026, 4, 4)
    CONFIDENCE_TIER: ClassVar[str] = "PROVEN"

    min_ratio: float

    def matches_row(self, row: dict, orb_label: str) -> bool:
        val = row.get("pit_range_atr")
        if val is None:
            return False  # fail-closed
        return val >= self.min_ratio

    def matches_df(self, df: pd.DataFrame, orb_label: str) -> pd.Series:
        import pandas as pd

        if "pit_range_atr" not in df.columns:
            return pd.Series(False, index=df.index)
        return df["pit_range_atr"].notna() & (df["pit_range_atr"] >= self.min_ratio)

    def describe(
        self,
        row: dict,
        orb_label: str,
        entry_model: str,
    ) -> list[AtomDescription]:
        """CME pit range / atr_20 gate. Pre-session, resolves at STARTUP.

        Pit closes 21:00 UTC; CME_REOPEN starts 23:00 UTC — zero look-ahead.
        Uses the pre-computed pit_range_atr column (already normalized).

        Threads canonical ClassVar metadata (VALIDATED_FOR, LAST_REVALIDATED,
        CONFIDENCE_TIER) into the atom so the eligibility adapter can apply
        NOT_APPLICABLE_INSTRUMENT and STALE_VALIDATION mechanically.
        """
        _ = (orb_label, entry_model)
        observed = _atom_numeric(row.get("pit_range_atr"))
        missing = observed is None
        passes = None if missing else observed >= self.min_ratio
        return [
            AtomDescription(
                name=f"pit_range / atr_20 >= {self.min_ratio:g}",
                category="PRE_SESSION",
                resolves_at="STARTUP",
                passes=passes,
                feature_column="pit_range_atr",
                observed_value=observed,
                threshold=self.min_ratio,
                comparator=">=",
                is_data_missing=missing,
                validated_for=self.VALIDATED_FOR,
                last_revalidated=self.LAST_REVALIDATED,
                confidence_tier=self.CONFIDENCE_TIER,
                explanation=(
                    f"Require CME pit-session range normalized by ATR-20 "
                    f">= {self.min_ratio:g} (prior pit activity gate)."
                ),
            )
        ]


@dataclass(frozen=True)
class CompositeFilter(StrategyFilter):
    """Chain two filters: base AND overlay must both pass."""

    base: StrategyFilter
    overlay: StrategyFilter

    @property
    def requires_micro_data(self) -> bool:
        """Composite requires micro data iff ANY component does.

        Dynamic computation — a price-based base composed with a volume-based
        overlay (e.g. `ORB_G5 AND VOL_RV12_N20`) still needs real micro data
        because the overlay's volume gate is meaningless on parent-proxy data.
        Recurses naturally through nested composites via the inner filter's
        own `requires_micro_data` property.

        @canonical-source pipeline/data_era.py
        """
        return self.base.requires_micro_data or self.overlay.requires_micro_data

    def matches_row(self, row: dict, orb_label: str) -> bool:
        return self.base.matches_row(row, orb_label) and self.overlay.matches_row(row, orb_label)

    def matches_df(self, df: pd.DataFrame, orb_label: str) -> pd.Series:
        return self.base.matches_df(df, orb_label) & self.overlay.matches_df(df, orb_label)

    def describe(
        self,
        row: dict,
        orb_label: str,
        entry_model: str,
    ) -> list[AtomDescription]:
        """Composite = base AND overlay. Atoms are the UNION of component atoms.

        Zero re-encoded logic — delegates fully to base.describe() and
        overlay.describe(). Consumers see each component's atoms separately,
        preserving granularity for the eligibility report (so the user can
        see which component failed instead of a single opaque FAIL).
        """
        return [
            *self.base.describe(row, orb_label, entry_model),
            *self.overlay.describe(row, orb_label, entry_model),
        ]


# =========================================================================
# PREDEFINED FILTER SETS — full ORB size spectrum
# =========================================================================

MGC_ORB_SIZE_FILTERS = {
    # "Less than" filters — DEAD: not in discovery grid (negative ExpR, 0/1024 validated).
    # Retained for reference and test coverage only.
    "L2": OrbSizeFilter(filter_type="ORB_L2", description="ORB size < 2 points", max_size=2.0),
    "L3": OrbSizeFilter(filter_type="ORB_L3", description="ORB size < 3 points", max_size=3.0),
    "L4": OrbSizeFilter(filter_type="ORB_L4", description="ORB size < 4 points", max_size=4.0),
    "L6": OrbSizeFilter(filter_type="ORB_L6", description="ORB size < 6 points", max_size=6.0),
    "L8": OrbSizeFilter(filter_type="ORB_L8", description="ORB size < 8 points", max_size=8.0),
    # "Greater than" filters — larger ORBs (better cost absorption)
    "G2": OrbSizeFilter(filter_type="ORB_G2", description="ORB size >= 2 points", min_size=2.0),
    "G3": OrbSizeFilter(filter_type="ORB_G3", description="ORB size >= 3 points", min_size=3.0),
    "G4": OrbSizeFilter(filter_type="ORB_G4", description="ORB size >= 4 points", min_size=4.0),
    "G5": OrbSizeFilter(filter_type="ORB_G5", description="ORB size >= 5 points", min_size=5.0),
    "G6": OrbSizeFilter(filter_type="ORB_G6", description="ORB size >= 6 points", min_size=6.0),
    "G8": OrbSizeFilter(filter_type="ORB_G8", description="ORB size >= 8 points", min_size=8.0),
}

# Volume-based filters (Selective Classification — abstain on low volume)
MGC_VOLUME_FILTERS = {
    "VOL_RV12_N20": VolumeFilter(
        filter_type="VOL_RV12_N20",
        description="Relative volume >= 1.2 (20-day lookback)",
        min_rel_vol=1.2,
        lookback_days=20,
    ),
    # Higher rel_vol thresholds (Mar 2026 confluence research program).
    # Phase 1: 12/48 BH FDR survivors at q=0.05 were rel_vol features.
    # Phase 3: all walk-forward DEPLOYABLE (WFE 0.97-2.30).
    # Thresholds cover key breakpoints found across sessions:
    #   CME_PRECLOSE ~1.3, MES ~1.7, NYSE_OPEN ~2.3, NYSE_CLOSE ~2.5, O30 ~4.8
    # @research-source research/output/confluence_program/phase1_run.py
    # @entry-models E2
    # @revalidated-for E2
    "VOL_RV15_N20": VolumeFilter(
        filter_type="VOL_RV15_N20",
        description="Relative volume >= 1.5 (20-day lookback)",
        min_rel_vol=1.5,
        lookback_days=20,
    ),
    "VOL_RV20_N20": VolumeFilter(
        filter_type="VOL_RV20_N20",
        description="Relative volume >= 2.0 (20-day lookback)",
        min_rel_vol=2.0,
        lookback_days=20,
    ),
    "VOL_RV25_N20": VolumeFilter(
        filter_type="VOL_RV25_N20",
        description="Relative volume >= 2.5 (20-day lookback)",
        min_rel_vol=2.5,
        lookback_days=20,
    ),
    "VOL_RV30_N20": VolumeFilter(
        filter_type="VOL_RV30_N20",
        description="Relative volume >= 3.0 (20-day lookback)",
        min_rel_vol=3.0,
        lookback_days=20,
    ),
    "ATR70_VOL": CombinedATRVolumeFilter(
        filter_type="ATR70_VOL",
        description="ATR pct >= 70 AND rel_vol >= 1.2 (combined regime+conviction)",
        min_rel_vol=1.2,
        lookback_days=20,
        min_atr_pct=70.0,
    ),
    # ORB window volume gates (Mar 2026 confluence research program).
    # Phase 1: orb_volume had 15/48 BH FDR survivors at q=0.05 (most hits).
    # Phase 3: all walk-forward DEPLOYABLE (WFE 1.02-3.24).
    # Log-spaced tiers cover the 10x volume range across sessions:
    #   SINGAPORE_OPEN ~2K, COMEX/PRECLOSE/CLOSE ~3-5K, US_DATA ~8K, NYSE_OPEN ~25K
    # @research-source research/output/confluence_program/phase1_run.py
    # @entry-models E2
    # @revalidated-for E2
    "ORB_VOL_2K": OrbVolumeFilter(
        filter_type="ORB_VOL_2K",
        description="ORB window volume >= 2,000 contracts",
        min_volume=2000.0,
    ),
    "ORB_VOL_4K": OrbVolumeFilter(
        filter_type="ORB_VOL_4K",
        description="ORB window volume >= 4,000 contracts",
        min_volume=4000.0,
    ),
    "ORB_VOL_8K": OrbVolumeFilter(
        filter_type="ORB_VOL_8K",
        description="ORB window volume >= 8,000 contracts",
        min_volume=8000.0,
    ),
    "ORB_VOL_16K": OrbVolumeFilter(
        filter_type="ORB_VOL_16K",
        description="ORB window volume >= 16,000 contracts",
        min_volume=16000.0,
    ),
    # Standalone ATR percentile filters (Mar 2026 confluence research program).
    # Phase 1: atr_20_pct had 2 BH FDR survivors at q=0.05 (weak but additive).
    # atr_20_pct is already normalized (0-100 percentile rank) — no instrument scaling issues.
    # Separate from ATR70_VOL (which requires BOTH atr>=70 AND rel_vol>=1.2).
    # @research-source research/output/confluence_program/phase1_run.py
    # @entry-models E2
    # @revalidated-for E2
    "ATR_P30": OwnATRPercentileFilter(
        filter_type="ATR_P30",
        description="Own ATR(20) percentile >= 30",
        min_pct=30.0,
    ),
    "ATR_P50": OwnATRPercentileFilter(
        filter_type="ATR_P50",
        description="Own ATR(20) percentile >= 50",
        min_pct=50.0,
    ),
    "ATR_P70": OwnATRPercentileFilter(
        filter_type="ATR_P70",
        description="Own ATR(20) percentile >= 70",
        min_pct=70.0,
    ),
    # Wave 4 Phase B T2-T8 survivor: ATR velocity ratio (expansion gate)
    # Tested at 2026-04-11 on post-Phase-3c data. 2/11 shortlist combos survived
    # full T3+T4+T6+T7 battery with in_ExpR > 0.05 (MNQ TOKYO_OPEN RR1.0,
    # MES US_DATA_1000 RR1.5). Three thresholds registered for Blueprint
    # Gate 2 sensitivity discipline (>=3 values per dimension).
    # 1.05 matches existing atr_vel_regime "Expanding" (build_daily_features.py:1133).
    # 1.10 and 1.15 give tighter selectivity for tuning.
    # @research-source scripts/research/wave4_presession_t2t8.py
    # @entry-models E2
    "ATR_VEL_GE105": ATRVelRatioFilter(
        filter_type="ATR_VEL_GE105",
        description="ATR velocity ratio >= 1.05 (expanding vol regime)",
        min_ratio=1.05,
    ),
    "ATR_VEL_GE110": ATRVelRatioFilter(
        filter_type="ATR_VEL_GE110",
        description="ATR velocity ratio >= 1.10 (strong expansion gate)",
        min_ratio=1.10,
    ),
    "ATR_VEL_GE115": ATRVelRatioFilter(
        filter_type="ATR_VEL_GE115",
        description="ATR velocity ratio >= 1.15 (extreme expansion gate)",
        min_ratio=1.15,
    ),
    # Wave 4 Phase B T2-T8 survivor #1 (strongest): GARCH(1,1) forecast vol
    # rolling percentile gate. Tested at 2026-04-11 on post-Phase-3c data.
    # The MNQ × NYSE_OPEN × RR1.5 LOW-quintile configuration produced in_ExpR
    # +0.240, WFE 1.00, p=0.042 — the single strongest un-shipped Phase B
    # result. Deployed as rolling percentile (not absolute threshold) because
    # the garch_forecast_vol distribution varies significantly across
    # instruments (MNQ Q20 ≈ 0.159 vs MES Q20 ≈ 0.111) — a fixed absolute
    # cutoff would fragment cross-instrument behavior and drift with regime.
    # Only the LT20 ("low-vol") variant is registered for deployment;
    # research did not find a surviving HIGH-vol configuration at the Phase B
    # kill-criteria threshold.
    # @research-source scripts/research/wave4_presession_t2t8.py
    # @entry-models E2
    # @revalidated-for 2026-04 Wave 5 Pathway B deployment
    "GARCH_VOL_PCT_LT20": GARCHForecastVolPctFilter(
        filter_type="GARCH_VOL_PCT_LT20",
        description="GARCH forecast vol rolling percentile <= 20 (low-vol regime)",
        pct_threshold=20.0,
        direction="low",
    ),
}

# Cost-ratio filters (Mar 2026): normalized cost screens derived from the
# canonical friction model. These are trade-size gates, not new signals.
COST_RATIO_FILTERS = {
    "COST_LT08": CostRatioFilter(
        filter_type="COST_LT08",
        description="Round-trip friction < 8% of raw ORB risk",
        max_cost_ratio_pct=8.0,
    ),
    "COST_LT10": CostRatioFilter(
        filter_type="COST_LT10",
        description="Round-trip friction < 10% of raw ORB risk",
        max_cost_ratio_pct=10.0,
    ),
    "COST_LT12": CostRatioFilter(
        filter_type="COST_LT12",
        description="Round-trip friction < 12% of raw ORB risk",
        max_cost_ratio_pct=12.0,
    ),
    "COST_LT15": CostRatioFilter(
        filter_type="COST_LT15",
        description="Round-trip friction < 15% of raw ORB risk",
        max_cost_ratio_pct=15.0,
    ),
}

# Break quality filters (research: break_quality_deep, Feb 2026)
# Break speed: fast breaks (<= N min from ORB end) select momentum days
_BREAK_SPEED_FAST5 = BreakSpeedFilter(
    filter_type="BRK_FAST5",
    description="Break within 5 min of ORB end",
    max_delay_min=5.0,
)
_BREAK_SPEED_FAST10 = BreakSpeedFilter(
    filter_type="BRK_FAST10",
    description="Break within 10 min of ORB end",
    max_delay_min=10.0,
)
# Break bar conviction: break bar closes in break direction
_BREAK_BAR_CONTINUES = BreakBarContinuesFilter(
    filter_type="BRK_CONT",
    description="Break bar continues in break direction",
    require_continues=True,
)

# Direction filters (H5: TOKYO_OPEN session shorts are noise; long-only doubles avgR)
DIR_LONG = DirectionFilter(filter_type="DIR_LONG", description="Long breakouts only", direction="long")
DIR_SHORT = DirectionFilter(filter_type="DIR_SHORT", description="Short breakouts only", direction="short")

# Double-break filter (regime classifier: momentum vs mean-reversion)
NO_DBL_BREAK = DoubleBreakFilter(
    filter_type="NO_DBL_BREAK",
    description="Skip double-break days (clean momentum only)",
    exclude=True,
)

# MES TOKYO_OPEN band filters (H2: MES TOKYO_OPEN ORBs >= 12pt are toxic)
_MES_1000_BAND_FILTERS = {
    "ORB_G4_L12": OrbSizeFilter(
        filter_type="ORB_G4_L12",
        description="ORB size >= 4 and < 12 points",
        min_size=4.0,
        max_size=12.0,
    ),
    "ORB_G5_L12": OrbSizeFilter(
        filter_type="ORB_G5_L12",
        description="ORB size >= 5 and < 12 points",
        min_size=5.0,
        max_size=12.0,
    ),
}

# Filters included in discovery grid (active filters only)
# L-filters removed from grid (negative ExpR, 0/1024 validated). Classes retained for reference.
# G2/G3 removed (99%+ pass rate on most sessions = cosmetic, not real filtering)
_GRID_SIZE_FILTERS = {k: v for k, v in MGC_ORB_SIZE_FILTERS.items() if k in ("G4", "G5", "G6", "G8")}

# MGC-specific: G4/G5 restored (Feb 2026 correction).
# Original removal claimed "G4 passes 87.5%" but actual CME_REOPEN pass rate is 7.2%.
# The 87.5% figure was likely about TOKYO_OPEN session (15m ORB = larger ranges).
# G4 strategies are the best MGC performers (ExpR +0.44-0.54, all validated).
_MGC_GRID_SIZE_FILTERS = {k: v for k, v in MGC_ORB_SIZE_FILTERS.items() if k in ("G4", "G5", "G6", "G8")}

# Calendar skip filters (portfolio overlay, not in discovery grid)
# DEPRECATED: Blanket NFP/OPEX skip disproven (Mar 2026 research). Retained for
# per-session conditional use only. Default engine overlay is now None.
CALENDAR_SKIP_NFP_OPEX = CalendarSkipFilter(
    filter_type="CAL_SKIP_NFP_OPEX",
    description="Skip NFP + OPEX days",
    skip_nfp=True,
    skip_opex=True,
    skip_friday_session=None,
)
CALENDAR_SKIP_ALL_CME_REOPEN = CalendarSkipFilter(
    filter_type="CAL_SKIP_ALL_CME_REOPEN",
    description="Skip NFP + OPEX + Friday@CME_REOPEN",
    skip_nfp=True,
    skip_opex=True,
    skip_friday_session="CME_REOPEN",
)

# DOW skip filters (discovery grid composites, Feb 2026 research)
# NOFRI and NOTUE removed from discovery grid Mar 2026 — DOW stress test found LIKELY NOISE.
# See research/output/DOW_FILTER_STRESS_TEST.md for full analysis.
# Definitions and ALL_FILTERS entries retained for DB row compatibility (Option B).
_DOW_SKIP_FRIDAY = DayOfWeekSkipFilter(filter_type="DOW_NOFRI", description="Skip Friday", skip_days=(4,))
_DOW_SKIP_MONDAY = DayOfWeekSkipFilter(filter_type="DOW_NOMON", description="Skip Monday", skip_days=(0,))
_DOW_SKIP_TUESDAY = DayOfWeekSkipFilter(filter_type="DOW_NOTUE", description="Skip Tuesday", skip_days=(1,))

# ORB-prefixed size filters for composite construction
_GRID_SIZE_FILTERS_ORB = {f"ORB_{k}": v for k, v in _GRID_SIZE_FILTERS.items()}

# M6E (Micro EUR/USD) pip-scaled size filters.
# MGC point filters (G4=4.0 points) are meaningless for EUR/USD (price ~1.0800).
# A 5-min EUR/USD ORB = 5-30 pips = 0.0005-0.0030 in native price units.
# Filter thresholds scaled to maintain similar friction/ORB ratio to MGC G4/G6/G8.
# M6E round-trip cost ~3 pips; G4 (4 pips) is the minimum viable ORB size.
_M6E_SIZE_FILTERS: dict[str, OrbSizeFilter] = {
    "M6E_G4": OrbSizeFilter(filter_type="M6E_G4", description="ORB size >= 0.0004 (4 pips)", min_size=0.0004),
    "M6E_G6": OrbSizeFilter(filter_type="M6E_G6", description="ORB size >= 0.0006 (6 pips)", min_size=0.0006),
    "M6E_G8": OrbSizeFilter(filter_type="M6E_G8", description="ORB size >= 0.0008 (8 pips)", min_size=0.0008),
}


def _make_dow_composites(
    size_filters: Mapping[str, StrategyFilter],
    dow_filter: DayOfWeekSkipFilter,
    suffix: str,
) -> dict[str, CompositeFilter]:
    """Build CompositeFilter(size + DOW skip) for each size filter."""
    return {
        f"{key}_{suffix}": CompositeFilter(
            filter_type=f"{key}_{suffix}",
            description=f"{filt.description} + {dow_filter.description}",
            base=filt,
            overlay=dow_filter,
        )
        for key, filt in size_filters.items()
    }


def _make_dbl_composites(
    size_filters: dict[str, StrategyFilter],
    dbl_filter: DoubleBreakFilter,
    suffix: str,
) -> dict[str, CompositeFilter]:
    """Build CompositeFilter(size + double-break skip) for each size filter."""
    return {
        f"{key}_{suffix}": CompositeFilter(
            filter_type=f"{key}_{suffix}",
            description=f"{filt.description} + {dbl_filter.description}",
            base=filt,
            overlay=dbl_filter,
        )
        for key, filt in size_filters.items()
    }


def _make_break_quality_composites(
    size_filters: Mapping[str, StrategyFilter],
    bq_filter: StrategyFilter,
    suffix: str,
) -> dict[str, CompositeFilter]:
    """Build CompositeFilter(size + break quality) for each size filter."""
    return {
        f"{key}_{suffix}": CompositeFilter(
            filter_type=f"{key}_{suffix}",
            description=f"{filt.description} + {bq_filter.description}",
            base=filt,
            overlay=bq_filter,
        )
        for key, filt in size_filters.items()
    }


# Base discovery grid filters — no session-specific DOW composites.
# get_filters_for_grid() starts from this so sessions that don't declare a
# DOW rule (e.g. SINGAPORE_OPEN, US_DATA_830, NYSE_OPEN) never inherit composites from other sessions.
# Exported (no underscore) so sync tests can assert base-grid invariants.
BASE_GRID_FILTERS: dict[str, StrategyFilter] = {
    "NO_FILTER": NoFilter(),
    **{f"ORB_{k}": v for k, v in _GRID_SIZE_FILTERS.items()},
    **COST_RATIO_FILTERS,
    **MGC_VOLUME_FILTERS,
}

# Overnight range absolute filters — extracted for reuse in composites.
# LOOK-AHEAD for Asian sessions — routed to US/EU sessions only via get_filters_for_grid().
# Phase 1: 11 BH FDR survivors (q=0.05), ALL MNQ, US sessions.
# @research-source research/output/confluence_program/phase1_run.py
_OVNRNG_FILTERS: dict[str, StrategyFilter] = {
    "OVNRNG_10": OvernightRangeAbsFilter(
        filter_type="OVNRNG_10",
        description="Overnight range >= 10 points",
        min_range=10.0,
    ),
    "OVNRNG_25": OvernightRangeAbsFilter(
        filter_type="OVNRNG_25",
        description="Overnight range >= 25 points",
        min_range=25.0,
    ),
    "OVNRNG_50": OvernightRangeAbsFilter(
        filter_type="OVNRNG_50",
        description="Overnight range >= 50 points",
        min_range=50.0,
    ),
    "OVNRNG_100": OvernightRangeAbsFilter(
        filter_type="OVNRNG_100",
        description="Overnight range >= 100 points",
        min_range=100.0,
    ),
}

# Master filter registry — COMPLETE source of truth for all filter types.
# Every filter_type in validated_setups MUST be in this dict.
# New scripts look up filters here — no guessing from naming conventions.
ALL_FILTERS: dict[str, StrategyFilter] = {
    **BASE_GRID_FILTERS,
    **_M6E_SIZE_FILTERS,
    # NOFRI/NOTUE retained in registry for DB row compatibility (Option B, Mar 2026).
    # Removed from discovery grid only — see get_filters_for_grid().
    **_make_dow_composites(_GRID_SIZE_FILTERS_ORB, _DOW_SKIP_FRIDAY, "NOFRI"),
    **_make_dow_composites(_GRID_SIZE_FILTERS_ORB, _DOW_SKIP_MONDAY, "NOMON"),
    **_make_dow_composites(_GRID_SIZE_FILTERS_ORB, _DOW_SKIP_TUESDAY, "NOTUE"),
    **_make_break_quality_composites(_GRID_SIZE_FILTERS_ORB, _BREAK_SPEED_FAST5, "FAST5"),
    **_make_break_quality_composites(_GRID_SIZE_FILTERS_ORB, _BREAK_SPEED_FAST10, "FAST10"),
    **_make_break_quality_composites(_GRID_SIZE_FILTERS_ORB, _BREAK_BAR_CONTINUES, "CONT"),
    # COST_LT × FAST composites (Apr 2026). Friction gate + order flow momentum.
    # Data-verified: +0.07R lift over COST alone at NYSE_CLOSE, +0.03R at NYSE_OPEN.
    # COST_LT is a stricter ORB-size cut on MNQ (nested in G8), but better calibrated.
    # @research-source memory/break_speed_signal_retest.md
    **_make_break_quality_composites(COST_RATIO_FILTERS, _BREAK_SPEED_FAST5, "FAST5"),
    **_make_break_quality_composites(COST_RATIO_FILTERS, _BREAK_SPEED_FAST10, "FAST10"),
    # OVNRNG × FAST composites (Apr 2026). Genuinely independent dimensions:
    # overnight range (pre-session) + break speed (intra-session).
    # Data-verified: +0.08R lift over OVNRNG alone at NYSE_CLOSE, +0.04R at NYSE_OPEN.
    # @research-source memory/break_speed_signal_retest.md
    **_make_break_quality_composites(_OVNRNG_FILTERS, _BREAK_SPEED_FAST5, "FAST5"),
    **_make_break_quality_composites(_OVNRNG_FILTERS, _BREAK_SPEED_FAST10, "FAST10"),
    # Direction filters (session-specific but must be registered for portfolio lookups)
    "DIR_LONG": DIR_LONG,
    "DIR_SHORT": DIR_SHORT,
    # Cross-asset ATR filters (Mar 2026 vol-regime research)
    # MNQ sessions filtered by MES/MGC ATR regime. Added to grid via get_filters_for_grid().
    # @research-source research/research_vol_regime_filter.py
    "X_MES_ATR70": CrossAssetATRFilter(
        filter_type="X_MES_ATR70",
        description="MES ATR pct >= 70",
        source_instrument="MES",
        min_pct=70.0,
    ),
    "X_MES_ATR60": CrossAssetATRFilter(
        filter_type="X_MES_ATR60",
        description="MES ATR pct >= 60",
        source_instrument="MES",
        min_pct=60.0,
    ),
    "X_MGC_ATR70": CrossAssetATRFilter(
        filter_type="X_MGC_ATR70",
        description="MGC ATR pct >= 70",
        source_instrument="MGC",
        min_pct=70.0,
    ),
    # MES TOKYO_OPEN band filters (H2: ORBs >= 12pt are toxic)
    **_MES_1000_BAND_FILTERS,
    # Overnight range absolute filters — from extracted _OVNRNG_FILTERS dict.
    **_OVNRNG_FILTERS,
    # Pre-session volatility filters (Apr 2026 research scan).
    # VALIDATED: bigger prev_day_range/atr predicts better ORB WR (+7-9% spread).
    # Direction: HIGH_BETTER — trade when prior day was volatile, skip quiet days.
    # No lookahead: prev_day_range and atr_20 are strictly prior-day values.
    # Routed to LONDON_METALS + EUROPE_FLOW via get_filters_for_grid().
    # @research-source scripts/research/scan_presession_features.py
    # @research-source scripts/research/scan_presession_t2t8.py
    # @entry-models E2
    # @revalidated-for E2 (Apr 2026)
    "PDR_R080": PrevDayRangeNormFilter(
        filter_type="PDR_R080",
        description="prev_day_range/atr >= 0.80 (~Q55, passes ~45%)",
        min_ratio=0.80,
    ),
    "PDR_R105": PrevDayRangeNormFilter(
        filter_type="PDR_R105",
        description="prev_day_range/atr >= 1.05 (~Q60, passes ~40%)",
        min_ratio=1.05,
    ),
    "PDR_R125": PrevDayRangeNormFilter(
        filter_type="PDR_R125",
        description="prev_day_range/atr >= 1.25 (~Q75, passes ~25%)",
        min_ratio=1.25,
    ),
    # Gap size filter for CME_REOPEN MGC only (+9.2% WR spread, p=0.009).
    # Bigger gaps = stronger conviction at session open = better ORB resolution.
    # No lookahead: gap_open_points is prior close to current open.
    # Routed to CME_REOPEN via get_filters_for_grid().
    # @research-source scripts/research/scan_presession_t2t8.py
    # @entry-models E2
    # @revalidated-for E2 (Apr 2026)
    "GAP_R005": GapNormFilter(
        filter_type="GAP_R005",
        description="abs(gap)/atr >= 0.005 (~Q50, passes ~50%)",
        min_ratio=0.005,
    ),
    "GAP_R015": GapNormFilter(
        filter_type="GAP_R015",
        description="abs(gap)/atr >= 0.015 (~Q75, passes ~25%)",
        min_ratio=0.015,
    ),
    # Pit range anti-filter (Apr 2026 — exchange_range_t2t8.py).
    # Skip dead-pit days: pit_range/atr < 0.10 = bottom 20% = WR 38-39%, deeply negative ExpR.
    # VALIDATED: 3/3 instruments, BH FDR at K=320, 12-15/16 years positive.
    # Zero look-ahead: pit closes 21:00 UTC, CME_REOPEN starts 23:00 UTC.
    # Routed to CME_REOPEN only via get_filters_for_grid().
    # @research-source scripts/research/exchange_range_t2t8.py
    # @entry-models E1, E2
    # @revalidated-for E1 (Apr 2026), E2 concordance +3-4pp
    "PIT_MIN": PitRangeFilter(
        filter_type="PIT_MIN",
        description="pit_range/atr >= 0.10 (skip bottom ~20% dead-pit days)",
        min_ratio=0.10,
    ),
}

# Calendar skip overlays (NOT in discovery grid — applied at portfolio/paper_trader level)
# Wired into ExecutionEngine._arm_strategies via calendar_overlay param.
# WARNING: Blanket NFP+OPEX skip is WRONG — Mar 2026 revalidation showed effects are
# instrument×session specific (some combos BETTER on NFP/OPEX days). These filters
# exist as infrastructure but should NOT be applied universally. Per-strategy calendar
# overlays (instrument×session specific) are the correct approach.
# @research-source research/research_calendar_effects.py
# @revalidated-for E1/E2 event-based sessions (Mar 2026)
CALENDAR_OVERLAYS: dict[str, CalendarSkipFilter] = {
    "CAL_SKIP_NFP_OPEX": CALENDAR_SKIP_NFP_OPEX,
    "CAL_SKIP_ALL_CME_REOPEN": CALENDAR_SKIP_ALL_CME_REOPEN,
}


def get_filters_for_grid(instrument: str, session: str) -> dict[str, StrategyFilter]:
    """Return session-aware filter set for discovery grid.

    Starts from BASE_GRID_FILTERS so that session-specific DOW composites are
    only added when a session has a research basis for them. Sessions without a
    declared DOW rule (SINGAPORE_OPEN, US_DATA_830, NYSE_OPEN) return the plain base set.

    DOW alignment guard: All DOW filters are validated against the canonical
    Brisbane→Exchange DOW mapping (pipeline/dst.py). Sessions where Brisbane
    DOW != exchange DOW (currently only NYSE_OPEN/0030) will raise ValueError if a
    DOW filter is applied — prevents silent misalignment.

    Break quality filters (Feb 2026 research):
    - Sessions CME_REOPEN, TOKYO_OPEN, LONDON_METALS: adds break speed + conviction composites
      for each G-filter. Research basis: break_quality_deep.py showed fast breaks
      and conviction candles predict success on momentum sessions.

    - Session LONDON_METALS: adds DOW composite (skip Monday) for each G-filter
    - Session TOKYO_OPEN: adds DIR_LONG (H5)
    - MES + TOKYO_OPEN: also adds G4_L12, G5_L12 band filters (H2 confirmed)
    - MNQ + SINGAPORE_OPEN: adds DIR_LONG (Feb 2026 raw-verified: SHORT avgR=-0.247 p=0.006,
      N=236, 2yrs consistent; LONG avgR=+0.187; asymmetry +0.434R)
    - All other combos: returns BASE_GRID_FILTERS unchanged
    """
    from pipeline.dst import validate_dow_filter_alignment

    # M6E (EUR/USD): pip-scaled filters only. MGC point filters (G4=4.0) are
    # meaningless for EUR/USD — every trade would trivially pass them.
    # No DOW composites yet; add only after first discovery pass identifies sessions
    # with breakout edge. FX DOW alignment also needs separate verification.
    if instrument == "M6E":
        return {
            "NO_FILTER": NoFilter(),
            **_M6E_SIZE_FILTERS,
        }

    # MGC: same G4-G8 grid as other instruments (G4 passes only 7.2% of 0900 days).
    # Break quality composites are built from MGC-specific size filters below.
    if instrument == "MGC":
        size_filters = _MGC_GRID_SIZE_FILTERS
        size_filters_orb = {f"ORB_{k}": v for k, v in size_filters.items()}
        filters: dict[str, StrategyFilter] = {
            "NO_FILTER": NoFilter(),
            **size_filters_orb,
            **COST_RATIO_FILTERS,
            **MGC_VOLUME_FILTERS,
        }
    else:
        size_filters_orb = _GRID_SIZE_FILTERS_ORB
        filters = dict(BASE_GRID_FILTERS)

    # Break quality composites for sessions with validated break-speed WR signal.
    # Apr 2026 retest: NYSE_CLOSE (+13.1% WR spread, DSR p<0.001, WFE 106%),
    # NYSE_OPEN (+8.2%, DSR p=0.013, WFE 92%), CME_REOPEN (+9.6% MGC, DSR p=0.002).
    # TOKYO_OPEN and LONDON_METALS retained for DB compatibility (existing validated strategies).
    # @research-source memory/break_speed_signal_retest.md
    if session in ("CME_REOPEN", "TOKYO_OPEN", "LONDON_METALS", "NYSE_CLOSE", "NYSE_OPEN"):
        filters.update(_make_break_quality_composites(size_filters_orb, _BREAK_SPEED_FAST5, "FAST5"))
        filters.update(_make_break_quality_composites(size_filters_orb, _BREAK_SPEED_FAST10, "FAST10"))
        filters.update(_make_break_quality_composites(size_filters_orb, _BREAK_BAR_CONTINUES, "CONT"))

    # COST_LT × FAST composites at break-speed validated sessions (Apr 2026).
    # Data-verified interaction: LOW_COST + FAST at NYSE_CLOSE = +0.191R (N=700),
    # NYSE_OPEN = +0.109R (N=1668), CME_REOPEN MGC = +0.178R (N=585).
    # COST_LT is a stricter ORB-size cut (nested in G8 on MNQ) but better friction-calibrated.
    # Let FDR decide which instrument×session×RR combos survive — no pre-filtering.
    # @research-source memory/break_speed_signal_retest.md
    if session in ("NYSE_CLOSE", "NYSE_OPEN", "CME_REOPEN"):
        filters.update(_make_break_quality_composites(COST_RATIO_FILTERS, _BREAK_SPEED_FAST5, "FAST5"))
        filters.update(_make_break_quality_composites(COST_RATIO_FILTERS, _BREAK_SPEED_FAST10, "FAST10"))

    # NOFRI removed from CME_REOPEN grid Mar 2026 (LIKELY NOISE — DOW stress test).
    # NOMON retained for LONDON_METALS (PLAUSIBLE BUT UNPROVEN, p=0.006 permutation).
    # NOTUE removed from TOKYO_OPEN grid Mar 2026 (LIKELY NOISE — DOW stress test).
    # See research/output/DOW_FILTER_STRESS_TEST.md.
    if session == "LONDON_METALS":
        validate_dow_filter_alignment(session, _DOW_SKIP_MONDAY.skip_days)
        filters.update(_make_dow_composites(size_filters_orb, _DOW_SKIP_MONDAY, "NOMON"))
    if session == "TOKYO_OPEN":
        filters["DIR_LONG"] = DIR_LONG
    if instrument == "MES" and session == "TOKYO_OPEN":
        filters.update(_MES_1000_BAND_FILTERS)
    if instrument == "MNQ" and session == "SINGAPORE_OPEN":
        # Feb 2026: SHORT is systematic AVOID (N=236, avgR=-0.247, p=0.006, raw-verified).
        # LONG is positive (+0.187). Wire long-only to block short discovery.
        filters["DIR_LONG"] = DIR_LONG
    # Cross-asset ATR filters for MNQ US sessions (Mar 2026 vol-regime research).
    # WF-validated: MES ATR regime predicts MNQ breakout quality at US sessions.
    # Selection: sessions whose timing overlaps US equity trading hours. Excludes
    # CME_REOPEN (evening), Asia/London sessions (no US equity overlap).
    # Review this set when adding new US-hours sessions.
    # @research-source research/research_vol_regime_filter.py
    if instrument == "MNQ":
        _cross_asset_sessions = {
            "CME_PRECLOSE",
            "COMEX_SETTLE",
            "US_DATA_1000",
            "NYSE_OPEN",
            "NYSE_CLOSE",
            "US_DATA_830",
        }
        if session in _cross_asset_sessions:
            filters["X_MES_ATR70"] = ALL_FILTERS["X_MES_ATR70"]
            filters["X_MES_ATR60"] = ALL_FILTERS["X_MES_ATR60"]
            filters["X_MGC_ATR70"] = ALL_FILTERS["X_MGC_ATR70"]

    # Overnight range absolute filters for US sessions (Mar 2026 confluence research).
    # LOOK-AHEAD for Asian sessions — route ONLY to sessions starting after 17:00 Brisbane.
    # @research-source research/output/confluence_program/phase1_run.py
    _overnight_clean_sessions = {
        "LONDON_METALS",
        "EUROPE_FLOW",
        "US_DATA_830",
        "NYSE_OPEN",
        "US_DATA_1000",
        "COMEX_SETTLE",
        "CME_PRECLOSE",
        "NYSE_CLOSE",
    }
    if session in _overnight_clean_sessions:
        for ovn_key in ("OVNRNG_10", "OVNRNG_25", "OVNRNG_50", "OVNRNG_100"):
            filters[ovn_key] = ALL_FILTERS[ovn_key]

    # OVNRNG × FAST composites at sessions with both OVNRNG and break-speed validation.
    # Genuinely independent dimensions: overnight range (pre-session) + break speed (intra-session).
    # Data-verified interaction: HIGH_OVN + FAST at NYSE_CLOSE = +0.176R (N=681),
    # NYSE_OPEN = +0.136R (N=1060). CME_REOPEN excluded (OVNRNG not routed there).
    # @research-source memory/break_speed_signal_retest.md
    _ovnrng_fast_sessions = {"NYSE_CLOSE", "NYSE_OPEN"}
    if session in _overnight_clean_sessions and session in _ovnrng_fast_sessions:
        filters.update(_make_break_quality_composites(_OVNRNG_FILTERS, _BREAK_SPEED_FAST5, "FAST5"))
        filters.update(_make_break_quality_composites(_OVNRNG_FILTERS, _BREAK_SPEED_FAST10, "FAST10"))

    # Pre-session volatility filters (Apr 2026 research scan).
    # VALIDATED: bigger prev_day_range/atr predicts better ORB WR.
    # PDR: LONDON_METALS MGC (p=0.003, 9/10yr), EUROPE_FLOW MGC+MNQ (p=0.008/0.017).
    # GAP: CME_REOPEN MGC only (p=0.009, WFE=0.68).
    # No lookahead — prev_day_range/atr_20 and gap_open_points are strictly prior-day.
    # @research-source scripts/research/scan_presession_t2t8.py
    _pdr_validated = {
        ("MGC", "LONDON_METALS"),
        ("MGC", "EUROPE_FLOW"),
        ("MNQ", "EUROPE_FLOW"),
        # GC proxy — Amendment 3.1 (2026-04-10). Same underlying as MGC,
        # same price-based features. PDR is regime-robust (percentile filter).
        ("GC", "LONDON_METALS"),
        ("GC", "EUROPE_FLOW"),
        ("GC", "NYSE_OPEN"),
        ("GC", "US_DATA_1000"),
    }
    if (instrument, session) in _pdr_validated:
        for pdr_key in ("PDR_R080", "PDR_R105", "PDR_R125"):
            filters[pdr_key] = ALL_FILTERS[pdr_key]

    if (instrument in ("MGC", "GC")) and session == "CME_REOPEN":
        for gap_key in ("GAP_R005", "GAP_R015"):
            filters[gap_key] = ALL_FILTERS[gap_key]

    # ATR velocity ratio expansion gate (Wave 4 Phase B — 2026-04-11).
    # T2-T8 survivors: MNQ TOKYO_OPEN RR1.0 (in_ExpR +0.188, p=0.0042),
    # MES US_DATA_1000 RR1.5 (in_ExpR +0.103, p=0.044).
    # Threshold 1.05 matches "Expanding" regime (pipeline/build_daily_features.py:1133).
    # No lookahead: atr_vel_ratio uses rows[i-5:i] prior-only slice.
    # @research-source scripts/research/wave4_presession_t2t8.py
    _atr_vel_validated = {
        ("MNQ", "TOKYO_OPEN"),
        ("MES", "US_DATA_1000"),
    }
    if (instrument, session) in _atr_vel_validated:
        filters["ATR_VEL_GE105"] = ALL_FILTERS["ATR_VEL_GE105"]

    # Pit range anti-filter: skip dead-pit days at CME_REOPEN (Apr 2026).
    # VALIDATED: 3/3 instruments pass T1-T8, BH FDR at K=320, +17% WR spread.
    # Zero look-ahead: pit closes 21:00 UTC, CME_REOPEN starts 23:00 UTC.
    # @research-source scripts/research/exchange_range_t2t8.py
    if session == "CME_REOPEN":
        filters["PIT_MIN"] = ALL_FILTERS["PIT_MIN"]

    # REMOVED (Feb 2026): NO_DBL_BREAK / NODBL composites for SINGAPORE_OPEN.
    # double_break column is LOOK-AHEAD — computed over full session AFTER
    # trade entry. Cannot be used as a pre-entry filter. All 6 validated
    # strategies using NODBL were artifacts of hindsight bias.
    return filters


# Entry models: realistic fill assumptions for backtesting
# E0 was purged Feb 2026: limit-on-confirm had 3 structural biases (fill-on-touch,
#   fakeout exclusion, fill-bar wins). Won 33/33 combos = artifact. Replaced by E2.
# E1 = Market at next bar open after confirm (momentum entry, honest baseline)
# E2 = Stop-market at ORB level + slippage (industry-standard breakout entry, Crabel)
# E3 = Limit order at ORB level, waiting for retrace after confirm (may not fill)
ENTRY_MODELS = ["E1", "E2", "E3"]

# Entry models to skip during outcome computation and grid search.
# E3 is soft-retired (100% fill = garbage on non-retrace sessions) but kept
# in ENTRY_MODELS for schema/test compatibility. Skip at runtime to save ~14%
# rebuild time. To re-enable E3: remove "E3" from this frozenset.
SKIP_ENTRY_MODELS: frozenset[str] = frozenset({"E3"})

# E2 stop-market slippage: number of ticks beyond ORB level for fill-through.
# Default 1 = industry standard (fill-through-by-1-tick). Use 2 for stress testing.
# Tick sizes per instrument are in pipeline/cost_model.py CostSpec.tick_size.
E2_SLIPPAGE_TICKS = 1
E2_STRESS_TICKS = 2

# Filters excluded from E2 discovery grid (look-ahead for stop-market entries).
#
# E2 enters on the FIRST bar whose range touches the ORB boundary after the
# ORB window closes. On ~42-49% of break-days, this bar is a fakeout (closes
# back inside) that precedes the confirmed close-based break. Filters that
# reference break-bar properties (volume, continuation, delay) are therefore
# look-ahead for E2 — the values are not knowable at E2 entry time.
#
# Grounding: Pardo Ch.4 ("backtest must use only data available at the
# decision point"), Aronson Ch.6 (look-ahead bias proportional to predictive
# power of unavailable information).
#
# CONT: break_bar_continues (bar close direction — unknown at intra-bar touch)
# FAST: break delay/speed (break_ts unknown until confirmed close)
# VOL_RV*: relative volume = break_bar_volume / baseline (break bar not yet closed)
# ATR70_VOL: combines ATR (safe) + rel_vol (look-ahead) — hybrid contaminated
#
# E1 is NOT affected: E1 enters AFTER the break bar closes (next bar open),
# so all break-bar properties are known at E1 entry time.
E2_EXCLUDED_FILTER_PREFIXES: tuple[str, ...] = (
    "VOL_RV",  # break_bar_volume in numerator
    "ATR70_VOL",  # includes rel_vol component
)
E2_EXCLUDED_FILTER_SUBSTRINGS: tuple[str, ...] = (
    "_CONT",  # break_bar_continues
    "_FAST",  # break delay/speed
    "NOMON_CONT",  # continuation variant
)

# =========================================================================
# Stop Multipliers: tighter stop placement via MAE profiling
# =========================================================================
# Research (2026-03-06): 0.75x stop kills losers at 10-20:1 vs winners.
# 56/57 combos show shallower max DD; 30/32 profitable combos improve R/DD.
# 16/228 survive BH FDR at q=0.05. Option B (same risk reference, earlier exit):
# loss at tight stop = -stop_multiplier R (original R basis unchanged).
# @research-source: session analysis, mae_r friction-adjusted simulation
# @research-date: 2026-03-06
# @entry-models: E1, E2
# @note: Do NOT test more than 3 levels (overfitting risk per Bloomy review).
STOP_MULTIPLIERS = [1.0, 0.75]


# =========================================================================
# E2 Order Timeout: break-speed overlay via execution timing
# =========================================================================
# Instead of a daily_features filter (look-ahead for E2), this timeout
# cancels the E2 stop-market trigger if the break hasn't happened within
# N minutes of ORB completion. Mechanically identical to break_delay_min <= N.
#
# Validated per-instrument per-session on raw canonical data (orb_outcomes +
# daily_features), pipeline filter objects, BH FDR at K=40, bootstrap null
# floor, ±20% sensitivity, walk-forward, and year stability:
#
#   MNQ NYSE_OPEN FAST5:  WR spread +7.6pp, p=0.0001, WFE=0.71, 12/17yr
#   MNQ NYSE_CLOSE FAST5: WR spread +9.2pp, p=0.009,  WFE=1.50, 10/10yr
#
# NOT validated: MGC (0/8, p>0.71), MES (0/4, p>0.11).
# Keyed by (instrument, session) → timeout_minutes.
# @research-source memory/break_speed_signal_retest.md
# @entry-models E2
# @revalidated-for E2 (Apr 2026)
#
# COMPREHENSIVE RETEST (K=88, all instruments × sessions, Apr 5 2026):
#   CME_PRECLOSE: 7/9 BH, +15pp. Only 8-10% slow after filters = +6R/8yr.
#   NYSE_OPEN: 3/9 BH, +6pp. 26% slow, -58.6R removable = +19R/8yr.
#   NYSE_CLOSE: slow trades PROFITABLE (+0.043R) — DO NOT timeout.
#   COMEX_SETTLE/EUROPE_FLOW: slow profitable — DO NOT timeout.
#   TOKYO/SINGAPORE: no signal. MGC/MES: no signal.
# DORMANT: +25R/8yr = +3R/yr on current portfolio. Not worth complexity.
# Enable when portfolio scales or broker supports native GTD stops.
E2_ORDER_TIMEOUT: dict[tuple[str, str], float] = {
    # ("MNQ", "CME_PRECLOSE"): 5.0,
    # ("MNQ", "NYSE_OPEN"): 5.0,
}


def apply_tight_stop(outcomes: list[dict], stop_multiplier: float, cost_spec) -> list[dict]:
    """Apply tight stop simulation to a list of outcome dicts (Option B).

    For each trade, checks whether max adverse excursion (mae_r) exceeded
    stop_multiplier * raw_stop_distance. If so, the trade is "killed" —
    pnl_r becomes -stop_multiplier and outcome becomes "loss".

    Uses per-trade friction-adjusted thresholds because mae_r includes
    friction in its denominator (risk_in_dollars = raw_risk + friction).

    Args:
        outcomes: list of outcome dicts with keys: mae_r, pnl_r, outcome,
                  entry_price, stop_price
        stop_multiplier: fraction of stop distance (e.g. 0.75)
        cost_spec: CostSpec from pipeline.cost_model

    Returns:
        New list of outcome dicts with adjusted pnl_r and outcome.
        Original list is NOT mutated.
    """
    if stop_multiplier >= 1.0:
        return outcomes  # No-op for standard stop

    adjusted = []
    for o in outcomes:
        mae_r = o.get("mae_r")
        entry_price = o.get("entry_price")
        stop_price = o.get("stop_price")

        # Skip trades without required fields
        if mae_r is None or entry_price is None or stop_price is None:
            adjusted.append(o)
            continue

        risk_pts = abs(entry_price - stop_price)
        if risk_pts <= 0:
            adjusted.append(o)
            continue

        # Per-trade friction-adjusted threshold
        raw_risk_d = risk_pts * cost_spec.point_value
        risk_d = raw_risk_d + cost_spec.total_friction
        max_adv_pts = mae_r * risk_d / cost_spec.point_value

        killed = max_adv_pts >= stop_multiplier * risk_pts

        if killed:
            new_o = dict(o)
            new_o["pnl_r"] = round(-stop_multiplier, 4)
            # Killed winners/scratches become losses
            if new_o.get("outcome") != "loss":
                new_o["outcome"] = "loss"
            adjusted.append(new_o)
        else:
            adjusted.append(o)

    return adjusted


# =========================================================================
# Variable Aperture: session-specific ORB duration (minutes)
# =========================================================================
# Research (scripts/analyze_mgc_15m_orb.py, 2026-02-13):
#   CME_REOPEN: 5m is OPTIMAL (ExpR +0.399, Sharpe 0.248). 15m/30m destroy edge.
#   TOKYO_OPEN: 15m BETTER than 5m (ExpR +0.206, Sharpe 0.122, N=133 at G6+).
#         5m baseline near-zero at G6+. 15m gives 2x trades + better edge.
#   SINGAPORE_OPEN: 5m ORB; double-break filter applied in discovery.
#   LONDON_METALS: 5m is OPTIMAL (ExpR +0.227, Sharpe 0.198). 15m/30m crush edge.
#   US_DATA_830: All negative at all windows.
#   NYSE_OPEN: All negative at all windows.
ORB_DURATION_MINUTES: dict[str, int] = {
    "CME_REOPEN": 5,  # CME Globex electronic reopen 5:00 PM CT
    "TOKYO_OPEN": 15,  # Tokyo Stock Exchange open 9:00 AM JST (15m ORB)
    "SINGAPORE_OPEN": 5,  # SGX/HKEX open 9:00 AM SGT
    "LONDON_METALS": 5,  # London metals AM session 8:00 AM London
    "EUROPE_FLOW": 5,  # European flow adjacent to London metals (7AM London winter / 9AM summer)
    "US_DATA_830": 5,  # US economic data release 8:30 AM ET
    "NYSE_OPEN": 5,  # NYSE cash open 9:30 AM ET
    "US_DATA_1000": 5,  # US 10:00 AM data (ISM/CC) + post-equity-open flow
    "COMEX_SETTLE": 5,  # COMEX gold settlement 1:30 PM ET
    "CME_PRECLOSE": 5,  # CME equity futures pre-settlement 2:45 PM CT
    "NYSE_CLOSE": 5,  # NYSE closing bell 4:00 PM ET
    "BRISBANE_1025": 5,  # Fixed 10:25 AM Brisbane (session discovery 2026-03-01)
}

# =========================================================================
# Tradeable instruments (research-validated)
# =========================================================================
# MCL (Micro Crude Oil): PERMANENTLY NO-GO for breakout strategies.
#   Tested: 5m/15m/30m ORBs, all sessions, NYMEX-focused, breakout + fade.
#   Oil is structurally mean-reverting (47-80% double break). No edge exists.
#   See memory/mcl_research.md for full scientific validation.
# Live strategy counts change after every rebuild — query live_config, do not hardcode.
# M2K (Micro Russell): DEAD for ORB (0/18 families survive null test, Mar 2026).
# Canonical source: pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS
# Do NOT hardcode — import from the single source of truth.
from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS  # noqa: E402

TRADEABLE_INSTRUMENTS = list(ACTIVE_ORB_INSTRUMENTS)

# Timed early exit: DISABLED (2026-03-18 OOS validation NO-GO).
# See EARLY_EXIT_MINUTES below for full rationale.
# None = no early exit for that session.
# E3 retrace window: max minutes after confirm bar to wait for retrace fill.
# None = unbounded (scan to trading day end). Value set from audit results.
# @research-source: research/research_e3_fill_timing.py
# @research-date: 2026-02-25
# @entry-models: E1, E2, E3
# @note: E3 is soft-retired (RETIRED status). Value only matters for historical outcomes.
E3_RETRACE_WINDOW_MINUTES: int | None = 60  # Audit: 4/12 sessions show stale inflation >0.1R

EARLY_EXIT_MINUTES: dict[str, int | None] = {
    # T80 time-stop: DISABLED (2026-03-18).
    # OOS paired portfolio replay (scripts/research/test_t80_oos.py) showed T80
    # does not improve walk-forward performance for any instrument:
    #   MGC: delta=-0.012, p=0.71 (inconclusive)
    #   MNQ: delta=-0.008, p=0.02 (T80 HURTS, 1/9 windows better)
    #   MES: delta=+0.005, p=0.56 (inconclusive)
    # Sessions with strongest in-sample BH evidence (LONDON_METALS 27 survivors,
    # SINGAPORE_OPEN 24 survivors) still hurt OOS. Original research had zero MNQ
    # BH survivors yet T80 was applied to MNQ — now confirmed harmful.
    # Decision: remove from execution. Discovery uses raw outcome/pnl_r.
    # ts_outcome/ts_pnl_r columns remain in DB schema (historical record).
    # @research-source: research/research_winner_speed.py (original)
    # @oos-validation: scripts/research/test_t80_oos.py (2026-03-18)
    # @decision: NO-GO — in-sample finding did not survive OOS validation
    "CME_REOPEN": None,
    "TOKYO_OPEN": None,
    "SINGAPORE_OPEN": None,
    "LONDON_METALS": None,
    "EUROPE_FLOW": None,
    "US_DATA_830": None,
    "NYSE_OPEN": None,
    "US_DATA_1000": None,
    "COMEX_SETTLE": None,
    "CME_PRECLOSE": None,
    "NYSE_CLOSE": None,
    "BRISBANE_1025": None,
}

# Session exit modes: how each session manages target/stop after entry.
# "fixed_target" = set-and-forget (target + stop, no modification)
# "ib_conditional" = IB-aware (hold target until IB resolves, then adapt)
SESSION_EXIT_MODE: dict[str, str] = {
    "CME_REOPEN": "fixed_target",
    "TOKYO_OPEN": "fixed_target",  # IB-conditional disabled — not validated in outcome_builder, creates parity gap
    "SINGAPORE_OPEN": "fixed_target",
    "LONDON_METALS": "fixed_target",
    "EUROPE_FLOW": "fixed_target",
    "US_DATA_830": "fixed_target",
    "NYSE_OPEN": "fixed_target",
    "US_DATA_1000": "fixed_target",
    "COMEX_SETTLE": "fixed_target",
    "CME_PRECLOSE": "fixed_target",
    "NYSE_CLOSE": "fixed_target",
    "BRISBANE_1025": "fixed_target",
}

# IB (Initial Balance) = first 120 minutes from 09:00 Brisbane (23:00 UTC).
# Used by TOKYO_OPEN session for IB-conditional exits.
IB_DURATION_MINUTES = 120

# Hold duration when IB breaks aligned with trade direction (TOKYO_OPEN session).
# Trade holds with stop only (no target) for this many hours after entry.
HOLD_HOURS = 7

# =========================================================================
# Strategy classification thresholds (FIX5 rules)
# =========================================================================
# CORE: enough samples for standalone portfolio weight
# REGIME: conditional overlay / signal only (not standalone)
# INVALID: fails min-sample, stress, or robustness
# @research-source: Lopez de Prado AFML Ch.11 — min sample for statistical power
# under multiple testing. 100 chosen to ensure BH FDR at alpha=0.05 with
# estimated correlation ρ̂≈0.3 (intra-session). 30 chosen as minimum for
# meaningful regime-conditional signal.
# @revalidated-for: E2 event-based sessions (2026-03-10)
CORE_MIN_SAMPLES = 100
REGIME_MIN_SAMPLES = 30

# Walk-forward efficiency floor. WFE = OOS_performance / IS_performance.
# WFE < 0.50 = strategy lost >50% of edge out-of-sample → likely overfit.
# @research-source: Pardo "Design, Testing, Optimization of Trading Systems";
#   RESEARCH_RULES.md L59: "WFE > 50% = strategy likely real. < 50% = likely overfit."
#   quant-audit-protocol.md Step 5: "T3: WFE < 0.50 → KILL → OVERFIT"
# @revalidated-for: E2 event-based (2026-04-02). Impact: 8/210 strategies killed.
MIN_WFE = 0.50

# Sessions excluded from portfolio fitness monitoring, PER INSTRUMENT.
# MGC SINGAPORE_OPEN: 74% double-break rate, mean-reverting structure, no edge.
# MNQ SINGAPORE_OPEN: HAS edge (17 ROBUST strategies, all CORE tier, all WF passed).
# The exclusion is MGC-specific, not blanket.
# @research-source: session analysis, double-break frequency audit
# @revalidated-for: E2 event-based (2026-03-10)
EXCLUDED_FROM_FITNESS: dict[str, set[str]] = {
    "MGC": {"SINGAPORE_OPEN"},
}


def get_excluded_sessions(instrument: str) -> set[str]:
    """Return sessions excluded from fitness for a specific instrument.

    Callers MUST pass the instrument to get per-instrument exclusions.
    Returns empty set if no exclusions exist for the instrument.
    """
    return EXCLUDED_FROM_FITNESS.get(instrument, set())


def classify_strategy(sample_size: int) -> str:
    """Classify a strategy by sample size per FIX5 rules.

    Returns 'CORE', 'REGIME', or 'INVALID'.
    """
    if sample_size >= CORE_MIN_SAMPLES:
        return "CORE"
    elif sample_size >= REGIME_MIN_SAMPLES:
        return "REGIME"
    else:
        return "INVALID"


# =========================================================================
# Portfolio overlay invariant (FIX5 rules)
# =========================================================================
# A valid trade day requires BOTH conditions:
#   1) A break occurred (outcome exists in orb_outcomes)
#   2) The strategy's filter_type makes the day eligible
# orb_outcomes contains ALL break-days regardless of filter.
# Overlay must ONLY write pnl_r on eligible days (series == 0.0).
# Low trade counts under strict filters (G6/G8) are EXPECTED, not bugs.


# =========================================================================
# Shared warning generation for AI query interfaces
# =========================================================================

_WARNING_RULES = {
    "NO_FILTER": "NO_FILTER: MGC/MES negative expectancy. MNQ positive unfiltered at 5 CORE sessions (audit 2026-03-24).",
    "ORB_L": "L-filter (less-than) strategies have negative expectancy -- house wins.",
}


def generate_strategy_warnings(df) -> list[str]:
    """Generate auto-warnings based on query result content.

    Used by mcp_server and query_agent to flag dangerous filter types
    and insufficient sample sizes.
    """
    warnings: list[str] = []
    if df is None or (hasattr(df, "empty") and df.empty):
        return warnings

    if "filter_type" in df.columns:
        for ft in df["filter_type"].unique():
            if ft == "NO_FILTER":
                warnings.append(_WARNING_RULES["NO_FILTER"])
            elif str(ft).startswith("ORB_L"):
                warnings.append(_WARNING_RULES["ORB_L"])

    if "sample_size" in df.columns:
        small = (df["sample_size"] < REGIME_MIN_SAMPLES).sum()
        if small > 0:
            warnings.append(f"{small} result(s) have sample_size < {REGIME_MIN_SAMPLES} (INVALID -- not tradeable).")
        regime = ((df["sample_size"] >= REGIME_MIN_SAMPLES) & (df["sample_size"] < CORE_MIN_SAMPLES)).sum()
        if regime > 0:
            warnings.append(
                f"{regime} result(s) have sample_size {REGIME_MIN_SAMPLES}-{CORE_MIN_SAMPLES - 1} "
                f"(REGIME -- conditional overlay only, not standalone)."
            )

    return list(set(warnings))
