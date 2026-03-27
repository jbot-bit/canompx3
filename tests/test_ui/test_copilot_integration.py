"""Integration tests for the co-pilot state machine + briefing cards."""

from datetime import date, datetime
from zoneinfo import ZoneInfo

BRISBANE = ZoneInfo("Australia/Brisbane")


class TestFullCopilotFlow:
    """End-to-end tests that don't require Streamlit runtime."""

    def test_state_machine_all_times_of_day(self):
        """Verify state machine returns valid state for every hour."""
        from ui.session_helpers import get_app_state

        for hour in range(24):
            now = datetime(2026, 3, 6, hour, 30, tzinfo=BRISBANE)
            state = get_app_state(now)
            assert state.name in (
                "WEEKEND",
                "IDLE",
                "APPROACHING",
                "ALERT",
                "OVERNIGHT",
            ), f"Invalid state at {hour}:30 — got {state.name}"

    def test_briefing_cards_build_without_error(self):
        """Briefing card builder should not crash."""
        from ui.session_helpers import build_session_briefings

        briefings = build_session_briefings()
        assert len(briefings) > 0, "Should have at least one briefing"
        for b in briefings:
            assert b.session, "Briefing missing session"
            assert b.instrument, "Briefing missing instrument"
            assert b.rr_target > 0, f"{b.session} {b.instrument} missing rr_target"
            assert len(b.conditions) > 0 or b.direction_note, f"{b.session} {b.instrument} has no conditions"

    def test_no_state_crash_on_dst_transition_day(self):
        """State machine must not crash on US DST transition."""
        from ui.session_helpers import get_app_state

        # Mar 8 2026 = US spring forward
        for hour in range(24):
            now = datetime(2026, 3, 8, hour, 0, tzinfo=BRISBANE)
            state = get_app_state(now)
            assert state.name in (
                "WEEKEND",
                "IDLE",
                "APPROACHING",
                "ALERT",
                "OVERNIGHT",
            )

    def test_weekend_state_saturday_sunday(self):
        """Saturday and Sunday should always be WEEKEND."""
        from ui.session_helpers import get_app_state

        sat = datetime(2026, 3, 7, 12, 0, tzinfo=BRISBANE)
        sun = datetime(2026, 3, 8, 12, 0, tzinfo=BRISBANE)
        assert get_app_state(sat).name == "WEEKEND"
        assert get_app_state(sun).name == "WEEKEND"

    def test_sessions_always_have_time(self):
        """Every upcoming session must have a valid datetime."""
        from ui.session_helpers import get_upcoming_sessions

        now = datetime(2026, 3, 6, 8, 0, tzinfo=BRISBANE)
        upcoming = get_upcoming_sessions(now)
        for _name, dt in upcoming:
            assert dt.tzinfo is not None
            assert dt > now
            assert (dt - now).total_seconds() < 36 * 3600

    def test_filter_translator_covers_all_live_filters(self):
        """Every filter in LIVE_PORTFOLIO should translate to non-empty English."""
        from trading_app.live_config import LIVE_PORTFOLIO
        from ui.session_helpers import filter_to_english

        all_filters = set(s.filter_type for s in LIVE_PORTFOLIO)
        for f in all_filters:
            english = filter_to_english(f)
            assert english, f"Filter {f} translated to empty string"
            assert english != f or f in ("DIR_LONG", "DIR_SHORT"), f"Filter {f} fell through to raw name: {english}"
