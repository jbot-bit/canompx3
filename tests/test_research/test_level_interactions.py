"""Tests for research.lib.level_interactions."""

import pandas as pd

from research.lib.level_interactions import classify_level_interaction, resolve_level_reference


def _bars(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["ts_utc", "open", "high", "low", "close"]).assign(
        ts_utc=lambda df: pd.to_datetime(df["ts_utc"], utc=True)
    )


class TestResolveLevelReference:
    def test_resolve_prev_day_high(self):
        row = {"prev_day_high": 101.25}
        ref = resolve_level_reference(row, "prev_day_high", target_session="TOKYO_OPEN")
        assert ref.price == 101.25
        assert ref.unavailable_reason is None

    def test_resolve_pivot_from_prev_day_fields(self):
        row = {
            "prev_day_high": 110.0,
            "prev_day_low": 100.0,
            "prev_day_close": 106.0,
        }
        ref = resolve_level_reference(row, "pivot", target_session="TOKYO_OPEN")
        assert ref.price == 105.33333333333333
        assert ref.unavailable_reason is None

    def test_fail_closed_when_level_not_safe_for_session(self):
        row = {"overnight_high": 104.0}
        ref = resolve_level_reference(row, "overnight_high", target_session="TOKYO_OPEN")
        assert ref.price is None
        assert ref.unavailable_reason == "level_not_safe_for_session"

    def test_fail_closed_when_unsupported(self):
        ref = resolve_level_reference({}, "vah", target_session="TOKYO_OPEN")
        assert ref.price is None
        assert ref.unavailable_reason == "unsupported_level"


class TestClassifyLevelInteraction:
    def test_touch_only_from_below(self):
        bars = _bars(
            [
                ("2026-04-19T00:00:00Z", 99.0, 99.8, 98.9, 99.5),
                ("2026-04-19T00:01:00Z", 99.5, 100.0, 99.2, 100.0),
            ]
        )
        event = classify_level_interaction(
            bars,
            level_name="prev_day_high",
            level_price=100.0,
            reference_side="below",
        )
        assert event.interaction_kind == "touch_only"
        assert event.swept is False
        assert event.reclaimed is False
        assert event.bar_index == 1

    def test_wick_fail_from_below(self):
        bars = _bars(
            [
                ("2026-04-19T00:00:00Z", 99.2, 99.7, 99.1, 99.4),
                ("2026-04-19T00:01:00Z", 99.4, 100.6, 99.3, 99.7),
            ]
        )
        event = classify_level_interaction(
            bars,
            level_name="prev_day_high",
            level_price=100.0,
            reference_side="below",
            sweep_epsilon=0.25,
        )
        assert event.interaction_kind == "wick_fail"
        assert event.swept is True
        assert event.reclaimed is False
        assert event.bar_index == 1

    def test_close_through_and_reclaim_from_below(self):
        bars = _bars(
            [
                ("2026-04-19T00:00:00Z", 99.0, 99.7, 98.9, 99.4),
                ("2026-04-19T00:01:00Z", 99.4, 100.8, 99.3, 100.4),
                ("2026-04-19T00:02:00Z", 100.4, 100.5, 99.2, 99.6),
            ]
        )
        event = classify_level_interaction(
            bars,
            level_name="prev_day_high",
            level_price=100.0,
            reference_side="below",
            sweep_epsilon=0.25,
            reclaim_lookahead_bars=2,
        )
        assert event.interaction_kind == "close_through"
        assert event.swept is True
        assert event.reclaimed is True
        assert event.reclaim_bar_index == 2

    def test_close_through_without_reclaim(self):
        bars = _bars(
            [
                ("2026-04-19T00:00:00Z", 101.0, 101.1, 100.4, 100.6),
                ("2026-04-19T00:01:00Z", 100.6, 100.8, 99.1, 99.5),
                ("2026-04-19T00:02:00Z", 99.5, 99.8, 99.0, 99.2),
            ]
        )
        event = classify_level_interaction(
            bars,
            level_name="prev_day_low",
            level_price=100.0,
            reference_side="above",
            sweep_epsilon=0.25,
            reclaim_lookahead_bars=1,
        )
        assert event.interaction_kind == "close_through"
        assert event.swept is True
        assert event.reclaimed is False

    def test_fail_closed_on_missing_bar_columns(self):
        bars = pd.DataFrame({"open": [1.0], "high": [2.0], "low": [0.5]})
        event = classify_level_interaction(
            bars,
            level_name="prev_day_high",
            level_price=1.5,
            reference_side="below",
        )
        assert event.interaction_kind is None
        assert event.unavailable_reason == "missing_bar_columns"
