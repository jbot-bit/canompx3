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
  0900 - Brisbane market open (23:00 UTC prev day). Primary US-session ORB.
         Largest ORBs in current regime. Best edge with E1 momentum entry.
  1000 - 1 hour after Brisbane open (00:00 UTC). Secondary US-session ORB.
         Uses 15m ORB (variable aperture). Strong long bias; IB-conditional exits.
  1100 - 2 hours after Brisbane open (01:00 UTC). Inconsistent edge.
  1800 - GLOBEX open / London close (08:00 UTC). Best with E3 retrace entry.
         Price often spikes through ORB then retraces, rewarding limit orders.
  2300 - Overnight session (13:00 UTC). Only works with G8+ (very large ORBs).
  0030 - Late overnight (14:30 UTC). No edge found; negative across all settings.

TRADING SESSIONS:
  Fixed session stat windows are in pipeline/build_daily_features.py.
  DST-aware session times (US_EQUITY_OPEN, LONDON_OPEN, etc.) are in
  pipeline/dst.py SESSION_CATALOG — those track actual market opens.

TRADING DAY:
  Runs 09:00 Brisbane to next 09:00 Brisbane (~1,440 minutes).
  Bars before 09:00 are assigned to the PREVIOUS trading day.

ENTRY MODELS (defined below as ENTRY_MODELS):
  E1 (Market-On-Next-Bar) - Enter at OPEN of the bar AFTER confirm bar.
     Fill rate ~100%. Best for momentum ORBs (0900, 1000).
  E3 (Limit-At-ORB) - Place limit order at ORB level after confirm.
     Fills only if price retraces to ORB level (~96-97% on G5+ days).
     Best for retrace ORBs (1800, 2300) where price spikes then pulls back.

CONFIRM BARS (CB1-CB5, defined in outcome_builder.py):
  Number of consecutive 1-minute bars closing outside the ORB range
  required before entry is confirmed. Higher CB = more confirmation but
  worse entry price (market has moved further).
  CB2 optimal for 0900/1000 momentum; CB5 optimal for 1800 E3 retrace.

RR TARGETS (RR1.0-RR4.0, defined in outcome_builder.py):
  Risk/Reward ratio. Target distance = entry risk * RR target.
  Risk = |entry_price - stop_price| where stop = opposite ORB level.
  RR2.5 optimal for 0900/1000; RR2.0 for 1800; RR1.5 for 2300.

ORB SIZE FILTERS:
  G-filters (Greater-than): Only trade when ORB size >= N points.
    G4+ is the minimum for positive expectancy. G5/G6 increase per-trade
    edge but reduce trade count. G8+ only useful for 2300.
  L-filters (Less-than): Only trade when ORB size < N points.
    ALL L-filter strategies have negative expectancy. Do not trade.
  NO_FILTER: Trade all days regardless of ORB size.
    ALL no-filter strategies have negative expectancy. Do not trade.

GRID (2,808 strategy combinations, full grid before ENABLED_SESSIONS filtering):
  E1: 13 ORBs x 6 RRs x 5 CBs x 6 filters = 2,340
  E3: 13 ORBs x 6 RRs x 1 CB x 6 filters = 468 (E3 always CB1)
  Total = 2,808

==========================================================================
"""

from dataclasses import dataclass, asdict
import json


@dataclass(frozen=True)
class StrategyFilter:
    """Base class for strategy filters."""

    filter_type: str
    description: str

    def to_json(self) -> str:
        """Serialize filter params to JSON."""
        return json.dumps(asdict(self))

    def matches_row(self, row: dict, orb_label: str) -> bool:
        """Check if a daily_features row matches this filter. Override in subclass."""
        return True


@dataclass(frozen=True)
class NoFilter(StrategyFilter):
    """Pass-through filter — all days match."""

    filter_type: str = "NO_FILTER"
    description: str = "No filter (all days)"

    def matches_row(self, row: dict, orb_label: str) -> bool:
        return True


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

    def matches_row(self, row: dict, orb_label: str) -> bool:
        rel_vol = row.get(f"rel_vol_{orb_label}")
        if rel_vol is None:
            return False  # fail-closed: no data = ineligible
        return rel_vol >= self.min_rel_vol


@dataclass(frozen=True)
class DirectionFilter(StrategyFilter):
    """Filter by breakout direction (long/short only)."""

    direction: str = "long"  # "long" or "short"

    def matches_row(self, row: dict, orb_label: str) -> bool:
        break_dir = row.get(f"orb_{orb_label}_break_dir")
        if break_dir is None:
            return False  # fail-closed
        return break_dir == self.direction


@dataclass(frozen=True)
class CalendarSkipFilter(StrategyFilter):
    """Skip trading on calendar event days (NFP, OPEX, Friday).

    Portfolio overlay — not part of discovery grid.
    All three flags are pre-computed in daily_features.
    """

    skip_nfp: bool = True
    skip_opex: bool = True
    skip_friday_session: str | None = None  # e.g. "0900" or None

    def matches_row(self, row: dict, orb_label: str) -> bool:
        if self.skip_nfp and row.get("is_nfp_day"):
            return False
        if self.skip_opex and row.get("is_opex_day"):
            return False
        if self.skip_friday_session and orb_label == self.skip_friday_session:
            if row.get("is_friday"):
                return False
        return True


@dataclass(frozen=True)
class DayOfWeekSkipFilter(StrategyFilter):
    """Skip trading on specific days of the week.

    Uses day_of_week column (0=Mon..6=Sun, Python weekday convention).
    Fail-closed: missing day_of_week means day is ineligible.
    """

    skip_days: tuple[int, ...] = ()

    def matches_row(self, row: dict, orb_label: str) -> bool:
        dow = row.get("day_of_week")
        if dow is None:
            return False  # fail-closed
        return dow not in self.skip_days


@dataclass(frozen=True)
class ATRVelocityFilter(StrategyFilter):
    """Skip sessions when ATR is actively contracting AND ORB compression is Neutral or Compressed.

    Research (Feb 2026 — research_avoid_crosscheck.py):
      Contracting×Neutral:   9/9 sessions 100% negative, median avgR=-0.372R
      Contracting×Compressed: 10/10 sessions 100% negative, median avgR=-0.362R
      MES 1000 anchor: 5/5 years negative, BH-sig p=0.0022
      MGC 1000 E1: 10/10 years (2016-2025) negative

    Signal is fully pre-entry — both atr_vel_regime and compression_tier are
    computed from prior-days data and known at ORB close (10:05 AM).

    Logic: skip when BOTH conditions hold:
      1. atr_vel_regime == 'Contracting'  (today ATR < 95% of prior-5-day avg)
      2. orb_{label}_compression_tier in ('Neutral', 'Compressed')
         (Expanded is OK — Contracting+Expanded has mixed/weaker signal)

    Applied to: sessions 0900 and 1000 (where the signal is confirmed).
    NOT applied to: 1800, 1100, 0030, 2300 (insufficient evidence or exception).
    Exception: MNQ 1800 is positive in the contracting regime — explicitly excluded
               by default apply_to_sessions.

    Fail-open: missing data (warm-up period) → trade is allowed.
    """
    filter_type: str = "ATR_VEL"
    description: str = "Skip Contracting ATR × Neutral/Compressed ORB sessions"
    apply_to_sessions: tuple[str, ...] = ("0900", "1000")

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


@dataclass(frozen=True)
class BreakSpeedFilter(StrategyFilter):
    """Filter by break delay: minutes from ORB end to first break.

    Fast breaks (low delay) indicate momentum / conviction.
    Slow breaks indicate grinding / indecision.

    Uses orb_{label}_break_delay_min from daily_features.
    Fail-closed: missing data means day is ineligible.
    """

    max_delay_min: float = 5.0

    def matches_row(self, row: dict, orb_label: str) -> bool:
        delay = row.get(f"orb_{orb_label}_break_delay_min")
        if delay is None:
            return False  # fail-closed: no break or no data
        return delay <= self.max_delay_min


@dataclass(frozen=True)
class BreakBarContinuesFilter(StrategyFilter):
    """Filter by break bar direction: does the break bar close in the break direction?

    A break bar that closes as a green candle (for longs) or red candle
    (for shorts) shows conviction. A reversal candle at the break point
    = weak breakout.

    Uses orb_{label}_break_bar_continues from daily_features.
    Fail-closed: missing data means day is ineligible.
    """

    require_continues: bool = True

    def matches_row(self, row: dict, orb_label: str) -> bool:
        continues = row.get(f"orb_{orb_label}_break_bar_continues")
        if continues is None:
            return False  # fail-closed
        return continues == self.require_continues


@dataclass(frozen=True)
class CompositeFilter(StrategyFilter):
    """Chain two filters: base AND overlay must both pass."""

    base: StrategyFilter
    overlay: StrategyFilter

    def matches_row(self, row: dict, orb_label: str) -> bool:
        return (self.base.matches_row(row, orb_label)
                and self.overlay.matches_row(row, orb_label))


# =========================================================================
# PREDEFINED FILTER SETS — full ORB size spectrum
# =========================================================================

MGC_ORB_SIZE_FILTERS = {
    # "Less than" filters — smaller ORBs
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

# Direction filters (H5: 1000 session shorts are noise; long-only doubles avgR)
DIR_LONG = DirectionFilter(
    filter_type="DIR_LONG", description="Long breakouts only", direction="long"
)
DIR_SHORT = DirectionFilter(
    filter_type="DIR_SHORT", description="Short breakouts only", direction="short"
)

# Double-break filter (regime classifier: momentum vs mean-reversion)
NO_DBL_BREAK = DoubleBreakFilter(
    filter_type="NO_DBL_BREAK",
    description="Skip double-break days (clean momentum only)",
    exclude=True,
)

# MES 1000 band filters (H2: MES 1000 ORBs >= 12pt are toxic)
_MES_1000_BAND_FILTERS = {
    "ORB_G4_L12": OrbSizeFilter(
        filter_type="ORB_G4_L12", description="ORB size >= 4 and < 12 points",
        min_size=4.0, max_size=12.0,
    ),
    "ORB_G5_L12": OrbSizeFilter(
        filter_type="ORB_G5_L12", description="ORB size >= 5 and < 12 points",
        min_size=5.0, max_size=12.0,
    ),
}

# Filters included in discovery grid (active filters only)
# L-filters removed from grid (negative ExpR, 0/1024 validated). Classes retained for reference.
# G2/G3 removed (99%+ pass rate on most sessions = cosmetic, not real filtering)
_GRID_SIZE_FILTERS = {k: v for k, v in MGC_ORB_SIZE_FILTERS.items()
                      if k in ("G4", "G5", "G6", "G8")}

# Calendar skip filters (portfolio overlay, not in discovery grid)
CALENDAR_SKIP_NFP_OPEX = CalendarSkipFilter(
    filter_type="CAL_SKIP_NFP_OPEX",
    description="Skip NFP + OPEX days",
    skip_nfp=True, skip_opex=True, skip_friday_session=None,
)
CALENDAR_SKIP_ALL_0900 = CalendarSkipFilter(
    filter_type="CAL_SKIP_ALL_0900",
    description="Skip NFP + OPEX + Friday@0900",
    skip_nfp=True, skip_opex=True, skip_friday_session="0900",
)

# DOW skip filters (discovery grid composites, Feb 2026 research)
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
    size_filters: dict[str, StrategyFilter],
    dow_filter: DayOfWeekSkipFilter,
    suffix: str,
) -> dict[str, CompositeFilter]:
    """Build CompositeFilter(size + DOW skip) for each size filter."""
    return {
        f"{key}_{suffix}": CompositeFilter(
            filter_type=f"{key}_{suffix}",
            description=f"{filt.description} + {dow_filter.description}",
            base=filt, overlay=dow_filter,
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
            base=filt, overlay=dbl_filter,
        )
        for key, filt in size_filters.items()
    }


def _make_break_quality_composites(
    size_filters: dict[str, StrategyFilter],
    bq_filter: StrategyFilter,
    suffix: str,
) -> dict[str, CompositeFilter]:
    """Build CompositeFilter(size + break quality) for each size filter."""
    return {
        f"{key}_{suffix}": CompositeFilter(
            filter_type=f"{key}_{suffix}",
            description=f"{filt.description} + {bq_filter.description}",
            base=filt, overlay=bq_filter,
        )
        for key, filt in size_filters.items()
    }


# Base discovery grid filters — no session-specific DOW composites.
# get_filters_for_grid() starts from this so sessions that don't declare a
# DOW rule (e.g. 1100, 2300, 0030) never inherit composites from other sessions.
# Exported (no underscore) so sync tests can assert base-grid invariants.
BASE_GRID_FILTERS: dict[str, StrategyFilter] = {
    "NO_FILTER": NoFilter(),
    **{f"ORB_{k}": v for k, v in _GRID_SIZE_FILTERS.items()},
    **MGC_VOLUME_FILTERS,
}

# Master filter registry — base + all DOW composites + break quality composites + M6E.
# Portfolio.py looks up filters by filter_type key from this registry.
# Count: 1 (NO_FILTER) + 4 (ORB G4-G8) + 1 (VOL) + 12 (DOW) + 8 (BRK_FAST5/CONT) + 3 (M6E) = 29
ALL_FILTERS: dict[str, StrategyFilter] = {
    **BASE_GRID_FILTERS,
    **_M6E_SIZE_FILTERS,
    **_make_dow_composites(_GRID_SIZE_FILTERS_ORB, _DOW_SKIP_FRIDAY,  "NOFRI"),
    **_make_dow_composites(_GRID_SIZE_FILTERS_ORB, _DOW_SKIP_MONDAY,  "NOMON"),
    **_make_dow_composites(_GRID_SIZE_FILTERS_ORB, _DOW_SKIP_TUESDAY, "NOTUE"),
    **_make_break_quality_composites(_GRID_SIZE_FILTERS_ORB, _BREAK_SPEED_FAST5, "FAST5"),
    **_make_break_quality_composites(_GRID_SIZE_FILTERS_ORB, _BREAK_SPEED_FAST10, "FAST10"),
    **_make_break_quality_composites(_GRID_SIZE_FILTERS_ORB, _BREAK_BAR_CONTINUES, "CONT"),
}

# Calendar skip overlays (NOT in discovery grid — applied at portfolio/paper_trader level)
# Wired into ExecutionEngine._arm_strategies via calendar_overlay param.
CALENDAR_OVERLAYS: dict[str, CalendarSkipFilter] = {
    "CAL_SKIP_NFP_OPEX": CALENDAR_SKIP_NFP_OPEX,
    "CAL_SKIP_ALL_0900": CALENDAR_SKIP_ALL_0900,
}


def get_filters_for_grid(instrument: str, session: str) -> dict[str, StrategyFilter]:
    """Return session-aware filter set for discovery grid.

    Starts from BASE_GRID_FILTERS so that session-specific DOW composites are
    only added when a session has a research basis for them. Sessions without a
    declared DOW rule (1100, 2300, 0030) return the plain base set.

    DOW alignment guard: All DOW filters are validated against the canonical
    Brisbane→Exchange DOW mapping (pipeline/dst.py). Sessions where Brisbane
    DOW != exchange DOW (currently only 0030) will raise ValueError if a
    DOW filter is applied — prevents silent misalignment.

    Break quality filters (Feb 2026 research):
    - Sessions "0900", "1000", "1800": adds break speed + conviction composites
      for each G-filter. Research basis: break_quality_deep.py showed fast breaks
      and conviction candles predict success on momentum sessions.

    - Session "0900": adds DOW composite (skip Friday) for each G-filter
    - Session "1800": adds DOW composite (skip Monday) for each G-filter
    - Session "1000": adds DIR_LONG (H5) + DOW composite (skip Tuesday) for each G-filter
    - MES + "1000": also adds G4_L12, G5_L12 band filters (H2 confirmed)
    - MNQ + "1100": adds DIR_LONG (Feb 2026 raw-verified: SHORT avgR=-0.247 p=0.006,
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

    filters = dict(BASE_GRID_FILTERS)

    # Break quality composites for momentum sessions (0900, 1000, 1800)
    if session in ("0900", "1000", "1800"):
        filters.update(_make_break_quality_composites(
            _GRID_SIZE_FILTERS_ORB, _BREAK_SPEED_FAST5, "FAST5"))
        filters.update(_make_break_quality_composites(
            _GRID_SIZE_FILTERS_ORB, _BREAK_SPEED_FAST10, "FAST10"))
        filters.update(_make_break_quality_composites(
            _GRID_SIZE_FILTERS_ORB, _BREAK_BAR_CONTINUES, "CONT"))

    if session == "0900":
        validate_dow_filter_alignment(session, _DOW_SKIP_FRIDAY.skip_days)
        filters.update(_make_dow_composites(_GRID_SIZE_FILTERS_ORB, _DOW_SKIP_FRIDAY, "NOFRI"))
    if session == "1800":
        validate_dow_filter_alignment(session, _DOW_SKIP_MONDAY.skip_days)
        filters.update(_make_dow_composites(_GRID_SIZE_FILTERS_ORB, _DOW_SKIP_MONDAY, "NOMON"))
    if session == "1000":
        validate_dow_filter_alignment(session, _DOW_SKIP_TUESDAY.skip_days)
        filters["DIR_LONG"] = DIR_LONG
        filters.update(_make_dow_composites(_GRID_SIZE_FILTERS_ORB, _DOW_SKIP_TUESDAY, "NOTUE"))
    if instrument == "MES" and session == "1000":
        filters.update(_MES_1000_BAND_FILTERS)
    if instrument == "MNQ" and session == "1100":
        # Feb 2026: SHORT is systematic AVOID (N=236, avgR=-0.247, p=0.006, raw-verified).
        # LONG is positive (+0.187). Wire long-only to block short discovery.
        filters["DIR_LONG"] = DIR_LONG
    # REMOVED (Feb 2026): NO_DBL_BREAK / NODBL composites for 1100.
    # double_break column is LOOK-AHEAD — computed over full session AFTER
    # trade entry. Cannot be used as a pre-entry filter. All 6 validated
    # strategies using NODBL were artifacts of hindsight bias.
    # See research: cross-session context (0900 dbl resolved + 1000 early dbl)
    # is the legitimate version of this idea.
    return filters


# Entry models: realistic fill assumptions for backtesting
# E0 = Limit at ORB level ON the confirm bar itself (always fills for CB1; partial for CB2+)
# E1 = Market at next bar open after confirm (momentum entry)
# E3 = Limit order at ORB level, waiting for retrace after confirm (may not fill)
# E2 was removed: identical to E1 on 1-minute bars (same days, same N, same WR)
# See entry_rules.py for implementation: detect_confirm() + resolve_entry()
ENTRY_MODELS = ["E0", "E1", "E3"]

# =========================================================================
# Variable Aperture: session-specific ORB duration (minutes)
# =========================================================================
# Research (scripts/analyze_mgc_15m_orb.py, 2026-02-13):
#   0900: 5m is OPTIMAL (ExpR +0.399, Sharpe 0.248). 15m/30m destroy edge.
#   1000: 15m BETTER than 5m (ExpR +0.206, Sharpe 0.122, N=133 at G6+).
#         5m baseline near-zero at G6+. 15m gives 2x trades + better edge.
#   1100: 5m ORB; double-break filter applied in discovery.
#   1800: 5m is OPTIMAL (ExpR +0.227, Sharpe 0.198). 15m/30m crush edge.
#   2300: All negative at all windows.
#   0030: All negative at all windows.
ORB_DURATION_MINUTES: dict[str, int] = {
    "0900": 5,
    "1000": 15,
    "1100": 5,              # Double-break filter applied in discovery
    "1130": 5,              # HK/SG equity open 9:30 AM HKT
    "1800": 5,
    "2300": 5,
    "0030": 5,
    # Dynamic sessions (DST-aware, resolved per-day by pipeline/dst.py)
    "CME_OPEN": 5,         # CME Globex electronic open 5:00 PM CT
    "US_EQUITY_OPEN": 5,   # NYSE cash open 09:30 ET (MES, MNQ)
    "US_DATA_OPEN": 5,     # Econ data release 08:30 ET (MGC)
    "LONDON_OPEN": 5,      # London metals 08:00 LT (MGC)
    "US_POST_EQUITY": 5,   # US post-equity-open 10:00 AM ET
    "CME_CLOSE": 5,        # CME equity futures pre-close 2:45 PM CT
}

# =========================================================================
# Tradeable instruments (research-validated)
# =========================================================================
# MCL (Micro Crude Oil): PERMANENTLY NO-GO for breakout strategies.
#   Tested: 5m/15m/30m ORBs, all sessions, NYMEX-focused, breakout + fade.
#   Oil is structurally mean-reverting (47-80% double break). No edge exists.
#   See memory/mcl_research.md for full scientific validation.
# MNQ (Micro Nasdaq): WEAK edge (~half MGC). Only 2 years data. Held lightly.
TRADEABLE_INSTRUMENTS = ["MGC"]

# Timed early exit: kill losers at N minutes after fill.
# Research (artifacts/EARLY_EXIT_RULES.md, G4+ filter):
#   0900: 15 min -> +26% Sharpe, 38% tighter MaxDD (only 24% recover)
#   1000: 30 min -> 3.7x Sharpe, 35% tighter MaxDD (only 12-18% recover)
#   Other sessions: no benefit
# Rule: At N minutes after fill, if bar close vs entry is negative, exit at bar close.
# None = no early exit for that session.
# E3 retrace window: max minutes after confirm bar to wait for retrace fill.
# None = unbounded (scan to trading day end). Value set from audit results.
# See research/research_e3_fill_timing.py for the analysis backing this.
E3_RETRACE_WINDOW_MINUTES: int | None = 60  # Audit: 4/12 sessions show stale inflation >0.1R

EARLY_EXIT_MINUTES: dict[str, int | None] = {
    "0900": 15,
    "1000": 30,
    "1100": None,
    "1130": None,
    "1800": None,
    "2300": None,
    "0030": None,
    # Dynamic sessions: no early exit until validated
    "CME_OPEN": None,
    "US_EQUITY_OPEN": None,
    "US_DATA_OPEN": None,
    "LONDON_OPEN": None,
    "US_POST_EQUITY": None,
    "CME_CLOSE": None,
}

# Session exit modes: how each session manages target/stop after entry.
# "fixed_target" = set-and-forget (target + stop, no modification)
# "ib_conditional" = IB-aware (hold target until IB resolves, then adapt)
SESSION_EXIT_MODE: dict[str, str] = {
    "0900": "fixed_target",
    "1000": "ib_conditional",
    "1100": "fixed_target",
    "1130": "fixed_target",
    "1800": "fixed_target",
    "2300": "fixed_target",
    "0030": "fixed_target",
    # Dynamic sessions: fixed_target until IB research is done
    "CME_OPEN": "fixed_target",
    "US_EQUITY_OPEN": "fixed_target",
    "US_DATA_OPEN": "fixed_target",
    "LONDON_OPEN": "fixed_target",
    "US_POST_EQUITY": "fixed_target",
    "CME_CLOSE": "fixed_target",
}

# IB (Initial Balance) = first 120 minutes from 09:00 Brisbane (23:00 UTC).
# Used by 1000 session for IB-conditional exits.
IB_DURATION_MINUTES = 120

# Hold duration when IB breaks aligned with trade direction (1000 session).
# Trade holds with stop only (no target) for this many hours after entry.
HOLD_HOURS = 7

# =========================================================================
# Strategy classification thresholds (FIX5 rules)
# =========================================================================
# CORE: enough samples for standalone portfolio weight
# REGIME: conditional overlay / signal only (not standalone)
# INVALID: fails min-sample, stress, or robustness
CORE_MIN_SAMPLES = 100
REGIME_MIN_SAMPLES = 30


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
