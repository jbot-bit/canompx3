"""Canonical promotion-time trade-window recomputation for validated strategies.

The validator and drift layer both need the same answer to the question:
"what are the actual traded days for this strategy under canonical computed
facts?" This module centralizes that answer so promotion-time provenance and
post-promotion audits cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trading_app.config import ALL_FILTERS, CompositeFilter, CrossAssetATRFilter, VolumeFilter
from trading_app.strategy_discovery import (
    _build_filter_day_sets,
    _compute_relative_volumes,
    _inject_cross_asset_atrs,
    _load_daily_features,
    _load_outcomes_bulk,
)


@dataclass(frozen=True)
class StrategyTradeWindow:
    """Canonical traded-day window for one strategy."""

    first_trade_day: date | None
    last_trade_day: date | None
    trade_day_count: int


def _contains_filter_type(filter_obj, klass: type) -> bool:
    """Return True if filter_obj or any composite component is instance of klass."""
    if isinstance(filter_obj, klass):
        return True
    if isinstance(filter_obj, CompositeFilter):
        return _contains_filter_type(filter_obj.base, klass) or _contains_filter_type(filter_obj.overlay, klass)
    return False


class StrategyTradeWindowResolver:
    """Recompute strategy trade windows from canonical computed facts.

    Caches loaded `daily_features`, enriched feature views, and exact-lane
    outcomes within one DB connection so validator/drift can resolve multiple
    strategies without re-querying the same facts repeatedly.
    """

    def __init__(self, con):
        self.con = con
        self._features_cache: dict[tuple[str, int], list[dict]] = {}
        self._filter_day_cache: dict[tuple[str, int, str, str], set[date]] = {}
        self._outcomes_cache: dict[tuple[str, int, str, str], dict] = {}

    def _get_features(self, instrument: str, orb_minutes: int) -> list[dict]:
        key = (instrument, orb_minutes)
        if key not in self._features_cache:
            self._features_cache[key] = _load_daily_features(
                self.con,
                instrument,
                orb_minutes,
                None,
                None,
            )
        return self._features_cache[key]

    def _get_filter_days(
        self,
        instrument: str,
        orb_minutes: int,
        orb_label: str,
        filter_type: str,
    ) -> set[date]:
        cache_key = (instrument, orb_minutes, orb_label, filter_type)
        if cache_key in self._filter_day_cache:
            return self._filter_day_cache[cache_key]

        filter_obj = ALL_FILTERS[filter_type]
        features = self._get_features(instrument, orb_minutes)

        if _contains_filter_type(filter_obj, VolumeFilter):
            _compute_relative_volumes(
                self.con,
                features,
                instrument,
                [orb_label],
                {filter_type: filter_obj},
            )

        if _contains_filter_type(filter_obj, CrossAssetATRFilter):
            _inject_cross_asset_atrs(
                self.con,
                features,
                instrument,
                {filter_type: filter_obj},
            )

        filter_days = _build_filter_day_sets(
            features,
            [orb_label],
            {filter_type: filter_obj},
        )[(filter_type, orb_label)]
        self._filter_day_cache[cache_key] = filter_days
        return filter_days

    def _get_grouped_outcomes(
        self,
        instrument: str,
        orb_minutes: int,
        orb_label: str,
        entry_model: str,
    ) -> dict:
        cache_key = (instrument, orb_minutes, orb_label, entry_model)
        if cache_key not in self._outcomes_cache:
            self._outcomes_cache[cache_key] = _load_outcomes_bulk(
                self.con,
                instrument,
                orb_minutes,
                [orb_label],
                [entry_model],
                holdout_date=None,
                start_date=None,
            )
        return self._outcomes_cache[cache_key]

    def resolve(
        self,
        *,
        instrument: str,
        orb_label: str,
        orb_minutes: int,
        entry_model: str,
        rr_target: float,
        confirm_bars: int,
        filter_type: str,
    ) -> StrategyTradeWindow:
        """Return canonical trade window for one strategy definition."""
        filter_days = self._get_filter_days(instrument, orb_minutes, orb_label, filter_type)
        grouped_outcomes = self._get_grouped_outcomes(
            instrument,
            orb_minutes,
            orb_label,
            entry_model,
        )
        grouped = grouped_outcomes.get((orb_label, entry_model, rr_target, confirm_bars), [])
        traded_days = sorted(o["trading_day"] for o in grouped if o["trading_day"] in filter_days)

        if not traded_days:
            return StrategyTradeWindow(first_trade_day=None, last_trade_day=None, trade_day_count=0)

        return StrategyTradeWindow(
            first_trade_day=traded_days[0],
            last_trade_day=traded_days[-1],
            trade_day_count=len(traded_days),
        )
