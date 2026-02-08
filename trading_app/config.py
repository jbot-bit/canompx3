"""
Strategy configuration: filters, entry models, and grid parameters.

==========================================================================
TERMINOLOGY & DEFINITIONS
==========================================================================

ORB (Opening Range Breakout):
  The high-low range of the first 5 minutes after a session opens.
  When price breaks above the ORB high or below the ORB low, it signals
  a potential directional move. All times are Australia/Brisbane (UTC+10).

ORB SESSIONS (defined in pipeline/init_db.py as ORB_LABELS):
  0900 - Brisbane market open (23:00 UTC prev day). Primary US-session ORB.
         Largest ORBs in current regime. Best edge with E1 momentum entry.
  1000 - 1 hour after Brisbane open (00:00 UTC). Secondary US-session ORB.
         Strong long bias; short side is negative expectancy.
  1100 - 2 hours after Brisbane open (01:00 UTC). Inconsistent edge.
  1800 - GLOBEX open / London close (08:00 UTC). Best with E3 retrace entry.
         Price often spikes through ORB then retraces, rewarding limit orders.
  2300 - Overnight session (13:00 UTC). Only works with G8+ (very large ORBs).
  0030 - Late overnight (14:30 UTC). No edge found; negative across all settings.

TRADING SESSIONS (defined in pipeline/build_daily_features.py):
  Asia:   09:00-17:00 Brisbane (23:00-07:00 UTC)
  London: 18:00-23:00 Brisbane (08:00-13:00 UTC)
  NY:     23:00-02:00 Brisbane (13:00-16:00 UTC, crosses midnight local)

TRADING DAY:
  Runs 09:00 Brisbane to next 09:00 Brisbane (~1,440 minutes).
  Bars before 09:00 are assigned to the PREVIOUS trading day.

ENTRY MODELS (defined below as ENTRY_MODELS):
  E1 (Market-On-Next-Bar) - Enter at OPEN of the bar AFTER confirm bar.
     Fill rate ~100%. Best for momentum ORBs (0900, 1000).
  E2 (Market-On-Confirm-Close) - Enter at CLOSE of the confirm bar itself.
     Fill rate 100%. Similar to E1 but slightly worse fill on average.
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

GRID (6,480 strategy combinations):
  6 ORBs x 6 RRs x 5 CBs x 12 filters x 3 entry models = 6,480

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

# Master filter registry (all filters by filter_type key)
ALL_FILTERS: dict[str, StrategyFilter] = {
    "NO_FILTER": NoFilter(),
    **{f"ORB_{k}": v for k, v in MGC_ORB_SIZE_FILTERS.items()},
}

# Entry models: realistic fill assumptions for backtesting
# E1 = Market at next bar open after confirm (momentum entry)
# E2 = Market at confirm bar close (immediate entry)
# E3 = Limit order at ORB level, waiting for retrace (better price, may not fill)
# See entry_rules.py for implementation: detect_confirm() + resolve_entry()
ENTRY_MODELS = ["E1", "E2", "E3"]
