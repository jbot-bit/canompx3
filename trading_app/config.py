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
    ALL no-filter strategies have negative expectancy. Do not trade.

GRID (3,024 strategy combinations, full grid before ENABLED_SESSIONS filtering):
  E1: 12 ORBs x 6 RRs x 5 CBs x 6 filters = 2,160
  E2: 12 ORBs x 6 RRs x 1 CB x 6 filters = 432 (E2 always CB1)
  E3: 12 ORBs x 6 RRs x 1 CB x 6 filters = 432 (E3 always CB1)
  Total = 3,024 (base grid; session-specific composites expand per-session)

==========================================================================
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

# ── Noise ExpR floor per entry model (adversarial validation 2026-03-18) ──
# Null test (2 seeds: 42, 99) ran the full pipeline on random-walk bars.
# Noise survivors set a floor: any production strategy below this is
# indistinguishable from chance.
#
# E2 (stop-market) has structural near-breakeven on noise (-0.004R avg),
# so noise easily produces E2 strategies with apparent edge.
# E1 (limit entry) costs real money immediately (-0.118R avg on noise),
# so noise rarely produces E1 survivors.
#
# Floors are rounded UP from the noise max across both seeds:
#   E2 noise max = 0.2379 → floor 0.24
#   E1 noise max = 0.0461 → floor 0.05
#
# @research-source null_test_seed42 + null_test_seed99 (adversarial validation 2026-03-18)
# @entry-models E1, E2
# @revalidated-for E1/E2 event-based sessions (2026-03-18)
NOISE_EXPR_FLOOR: dict[str, float] = {
    "E1": 0.05,
    "E2": 0.24,
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

    def matches_df(self, df: pd.DataFrame, orb_label: str) -> pd.Series:
        """Vectorized filter: return boolean Series. Default falls back to iterrows."""
        import pandas as pd

        return pd.Series(
            [self.matches_row(row.to_dict(), orb_label) for _, row in df.iterrows()],
            index=df.index,
        )


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

    def matches_df(self, df: pd.DataFrame, orb_label: str) -> pd.Series:
        import pandas as pd

        col = f"rel_vol_{orb_label}"
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        return df[col].notna() & (df[col] >= self.min_rel_vol)


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


@dataclass(frozen=True)
class CrossAssetATRFilter(StrategyFilter):
    """Filter by another instrument's ATR regime.

    Reads cross_atr_{source_instrument}_pct from the row dict.
    This key is injected at discovery/fitness time by
    _inject_cross_asset_atrs() — NOT stored in daily_features schema.

    Fail-closed: if the key is absent or None, day is ineligible.

    @research-source research/research_vol_regime_filter.py
    """

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

    def matches_df(self, df: pd.DataFrame, orb_label: str) -> pd.Series:
        import pandas as pd

        if "day_of_week" not in df.columns:
            return pd.Series(False, index=df.index)
        # NaN → fail-closed (notna check), then exclude skip_days
        return df["day_of_week"].notna() & ~df["day_of_week"].isin(self.skip_days)


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
    """

    filter_type: str = "ATR_VEL"
    description: str = "Skip Contracting ATR × Neutral/Compressed ORB sessions"
    apply_to_sessions: tuple[str, ...] = ("CME_REOPEN", "TOKYO_OPEN")

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

    def matches_df(self, df: pd.DataFrame, orb_label: str) -> pd.Series:
        import pandas as pd

        col = f"orb_{orb_label}_break_delay_min"
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        return df[col].notna() & (df[col] <= self.max_delay_min)


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

    def matches_df(self, df: pd.DataFrame, orb_label: str) -> pd.Series:
        import pandas as pd

        col = f"orb_{orb_label}_break_bar_continues"
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        return df[col].notna() & (df[col] == self.require_continues)


@dataclass(frozen=True)
class CompositeFilter(StrategyFilter):
    """Chain two filters: base AND overlay must both pass."""

    base: StrategyFilter
    overlay: StrategyFilter

    def matches_row(self, row: dict, orb_label: str) -> bool:
        return self.base.matches_row(row, orb_label) and self.overlay.matches_row(row, orb_label)

    def matches_df(self, df: pd.DataFrame, orb_label: str) -> pd.Series:
        return self.base.matches_df(df, orb_label) & self.overlay.matches_df(df, orb_label)


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
    "ATR70_VOL": CombinedATRVolumeFilter(
        filter_type="ATR70_VOL",
        description="ATR pct >= 70 AND rel_vol >= 1.2 (combined regime+conviction)",
        min_rel_vol=1.2,
        lookback_days=20,
        min_atr_pct=70.0,
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
    size_filters: dict[str, StrategyFilter],
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
    size_filters: dict[str, StrategyFilter],
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
    **MGC_VOLUME_FILTERS,
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
            **MGC_VOLUME_FILTERS,
        }
    else:
        size_filters_orb = _GRID_SIZE_FILTERS_ORB
        filters = dict(BASE_GRID_FILTERS)

    # Break quality composites for momentum sessions (CME_REOPEN, TOKYO_OPEN, LONDON_METALS)
    if session in ("CME_REOPEN", "TOKYO_OPEN", "LONDON_METALS"):
        filters.update(_make_break_quality_composites(size_filters_orb, _BREAK_SPEED_FAST5, "FAST5"))
        filters.update(_make_break_quality_composites(size_filters_orb, _BREAK_SPEED_FAST10, "FAST10"))
        filters.update(_make_break_quality_composites(size_filters_orb, _BREAK_BAR_CONTINUES, "CONT"))

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
# MNQ (Micro Nasdaq): ~5 years data, 14 live strategies, 8 sessions.
# MES (Micro S&P 500): 4-5 live strategies, CME_PRECLOSE dominant.
# M2K (Micro Russell): 4-5 live strategies, NYSE_OPEN/US_DATA_1000/LONDON_METALS.
# Canonical source: pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS
# Do NOT hardcode — import from the single source of truth.
from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS  # noqa: E402

TRADEABLE_INSTRUMENTS = list(ACTIVE_ORB_INSTRUMENTS)

# Timed early exit: kill losers at N minutes after fill.
# Research (artifacts/EARLY_EXIT_RULES.md, P5b winner speed profiling):
#   CME_REOPEN: 15 min -> +26% Sharpe, 38% tighter MaxDD (T80=38m, only 24% recover)
#   TOKYO_OPEN: 30 min -> 3.7x Sharpe, 35% tighter MaxDD (T80=32m, only 12-18% recover)
#   LONDON_METALS: 30 min -> T80=36m, avg_r_after=-0.339R (worst dead-chop penalty)
#   Other sessions: no benefit
# Rule: At N minutes after fill, if bar close vs entry is negative, exit at bar close.
# None = no early exit for that session.
# E3 retrace window: max minutes after confirm bar to wait for retrace fill.
# None = unbounded (scan to trading day end). Value set from audit results.
# @research-source: research/research_e3_fill_timing.py
# @research-date: 2026-02-25
# @entry-models: E1, E2, E3
# @note: E3 is soft-retired (RETIRED status). Value only matters for historical outcomes.
E3_RETRACE_WINDOW_MINUTES: int | None = 60  # Audit: 4/12 sessions show stale inflation >0.1R

EARLY_EXIT_MINUTES: dict[str, int | None] = {
    # T80 = time by which 80% of winners hit target. Trades past T80 are net
    # negative (P5b, BH FDR confirmed). At threshold: if MTM < 0, exit.
    # Per-session (not per-instrument). Where instruments differ, uses max
    # T80 (more patient) to avoid cutting slower instruments' winners.
    #
    # @research-source: research/research_winner_speed.py
    # @research-date: 2026-03-04 (6 new sessions); 2026-02-20 (original 5)
    # @entry-models: E0, E1, E2, E3
    # @revalidated-for: E2 (2026-03-03, bar-level P5b simulation)
    # @revalidation-result: ALL sessions net-positive at current thresholds.
    #   E2 T80 differs (CME_REOPEN: 110m agg vs 38m config) but rule is
    #   robust — 76-94% of MTM-negative trades at threshold never recover.
    #   Future optimization: CME_REOPEN 25m saves ~47% more R than 38m.
    # @note: New 6 sessions use RR=1.0 T80 max-across-instruments from
    #   winner_speed_summary.csv (2026-03-04 run, all 4 instruments).
    #   Pending P5b bar-level simulation for fine-tuning.
    "CME_REOPEN": 38,  # E2 T80=110m(agg), config checks early, NetR=+905R
    "TOKYO_OPEN": 39,  # E2 T80=63m(agg), NetR=+1207R
    "SINGAPORE_OPEN": 31,  # E2 T80=65m(agg), NetR=+1114R
    "LONDON_METALS": 36,  # E2 T80=60m(agg), NetR=+937R
    "EUROPE_FLOW": None,  # No T80 research yet — trades run to target/stop
    "US_DATA_830": 49,  # RR1.0 T80: MGC=46, MNQ=49, MES=44, M2K=45
    "NYSE_OPEN": 59,  # RR1.0 T80: MGC=28, MNQ=48, MES=43, M2K=59
    "US_DATA_1000": 45,  # RR1.0 T80: MGC=45, MNQ=40, MES=41, M2K=38
    "COMEX_SETTLE": 39,  # RR1.0 T80: MGC=39, MNQ=26, MES=24, M2K=27
    "CME_PRECLOSE": 16,  # E2 T80=13m(agg), closest to optimal, NetR=+607R
    "NYSE_CLOSE": 111,  # RR1.0 T80: MNQ=111, MES=110, M2K=108 (no MGC)
    "BRISBANE_1025": 26,  # RR1.0 T80: MNQ=26, M2K=24 (MNQ/M2K only)
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
    "NO_FILTER": "NO_FILTER strategies have negative expectancy -- house wins.",
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
