"""
Filter definitions for strategy discovery.

Filters select subsets of trading days based on market conditions:
  - OrbSizeFilter: ORB size ranges (e.g., < 4 points, >= 3 points)
  - NoFilter: Pass-through (all days)

Each filter is a frozen dataclass for hashability and serialization.
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
