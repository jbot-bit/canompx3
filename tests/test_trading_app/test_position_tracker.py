"""Tests for PositionTracker state machine."""

from datetime import UTC, datetime, timedelta

from trading_app.live.position_tracker import PositionState, PositionTracker


class TestStateTransitions:
    def test_entry_sent_creates_pending(self):
        tracker = PositionTracker()
        record = tracker.on_entry_sent("S1", "long", 100.0, order_id=42)
        assert record.state == PositionState.PENDING_ENTRY
        assert record.engine_entry_price == 100.0
        assert record.entry_order_id == 42

    def test_entry_filled_transitions_to_entered(self):
        tracker = PositionTracker()
        tracker.on_entry_sent("S1", "long", 100.0, order_id=42)
        record = tracker.on_entry_filled("S1", 100.5)
        assert record.state == PositionState.ENTERED
        assert record.fill_entry_price == 100.5
        assert record.entry_slippage == 0.5

    def test_exit_sent_transitions_to_pending_exit(self):
        tracker = PositionTracker()
        tracker.on_entry_sent("S1", "long", 100.0)
        tracker.on_entry_filled("S1", 100.0)
        record = tracker.on_exit_sent("S1", exit_order_id=99)
        assert record.state == PositionState.PENDING_EXIT
        assert record.exit_order_id == 99

    def test_exit_filled_removes_position(self):
        tracker = PositionTracker()
        tracker.on_entry_sent("S1", "long", 100.0)
        tracker.on_entry_filled("S1", 100.0)
        tracker.on_exit_sent("S1")
        record = tracker.on_exit_filled("S1", 105.0)
        assert record.fill_exit_price == 105.0
        assert tracker.get("S1") is None  # removed

    def test_signal_entry_goes_straight_to_entered(self):
        tracker = PositionTracker()
        record = tracker.on_signal_entry("S1", 100.0, "long")
        assert record.state == PositionState.ENTERED
        assert record.engine_entry_price == 100.0
        assert record.fill_entry_price is None


class TestBestEntryPrice:
    def test_prefers_fill(self):
        tracker = PositionTracker()
        tracker.on_entry_sent("S1", "long", 100.0)
        tracker.on_entry_filled("S1", 100.5)
        assert tracker.best_entry_price("S1", 0.0) == 100.5

    def test_falls_back_to_engine(self):
        tracker = PositionTracker()
        tracker.on_entry_sent("S1", "long", 100.0)
        assert tracker.best_entry_price("S1", 0.0) == 100.0

    def test_falls_back_to_fallback(self):
        tracker = PositionTracker()
        assert tracker.best_entry_price("UNKNOWN", 42.0) == 42.0

    def test_fill_price_zero_not_falsy(self):
        """fill_entry_price=0.0 must not fall through to engine_entry_price — guards is-None fix (PT1)."""
        tracker = PositionTracker()
        tracker.on_entry_sent("S1", "long", 100.0)
        tracker.on_entry_filled("S1", 0.0)
        assert tracker.best_entry_price("S1", 99.0) == 0.0


class TestStaleDetection:
    def test_fresh_pending_not_stale(self):
        tracker = PositionTracker()
        tracker.on_entry_sent("S1", "long", 100.0)
        assert tracker.stale_positions(timeout_seconds=300) == []

    def test_old_pending_is_stale(self):
        tracker = PositionTracker()
        tracker.on_entry_sent("S1", "long", 100.0)
        # Backdate the state_changed_at
        record = tracker.get("S1")
        record.state_changed_at = datetime.now(UTC) - timedelta(seconds=600)
        stale = tracker.stale_positions(timeout_seconds=300)
        assert len(stale) == 1
        assert stale[0].strategy_id == "S1"

    def test_entered_not_stale(self):
        tracker = PositionTracker()
        tracker.on_entry_sent("S1", "long", 100.0)
        tracker.on_entry_filled("S1", 100.0)
        # Even if old, ENTERED is not a pending state
        record = tracker.get("S1")
        record.state_changed_at = datetime.now(UTC) - timedelta(seconds=600)
        assert tracker.stale_positions(timeout_seconds=300) == []


class TestEdgeCases:
    def test_exit_fill_unknown_strategy(self):
        tracker = PositionTracker()
        result = tracker.on_exit_filled("UNKNOWN", 100.0)
        assert result is None

    def test_entry_fill_unknown_strategy(self):
        tracker = PositionTracker()
        result = tracker.on_entry_filled("UNKNOWN", 100.0)
        assert result is None

    def test_double_entry_rejected_when_pending(self):
        tracker = PositionTracker()
        tracker.on_entry_sent("S1", "long", 100.0)
        # Second entry should be REJECTED (active position)
        result = tracker.on_entry_sent("S1", "short", 200.0)
        assert result is None
        # Original position unchanged
        assert tracker.get("S1").direction == "long"
        assert tracker.get("S1").engine_entry_price == 100.0

    def test_double_entry_rejected_when_entered(self):
        tracker = PositionTracker()
        tracker.on_entry_sent("S1", "long", 100.0)
        tracker.on_entry_filled("S1", 100.5)
        # Reject entry when already ENTERED
        result = tracker.on_entry_sent("S1", "short", 200.0)
        assert result is None
        assert tracker.get("S1").state == PositionState.ENTERED

    def test_entry_allowed_over_stale_pending_exit(self):
        tracker = PositionTracker()
        tracker.on_entry_sent("S1", "long", 100.0)
        tracker.on_entry_filled("S1", 100.0)
        tracker.on_exit_sent("S1", exit_order_id=99)
        assert tracker.get("S1").state == PositionState.PENDING_EXIT
        # Entry over stale PENDING_EXIT should be ALLOWED (failed rollover)
        record = tracker.on_entry_sent("S1", "short", 200.0)
        assert record is not None
        assert record.direction == "short"
        assert record.state == PositionState.PENDING_ENTRY

    def test_signal_entry_rejected_when_active(self):
        tracker = PositionTracker()
        tracker.on_signal_entry("S1", 100.0, "long")
        # Second signal entry should be REJECTED
        result = tracker.on_signal_entry("S1", 200.0, "short")
        assert result is None
        assert tracker.get("S1").direction == "long"

    def test_active_positions(self):
        tracker = PositionTracker()
        tracker.on_entry_sent("S1", "long", 100.0)
        tracker.on_signal_entry("S2", 200.0, "short")
        assert len(tracker.active_positions()) == 2

    def test_pop_removes(self):
        tracker = PositionTracker()
        tracker.on_entry_sent("S1", "long", 100.0)
        record = tracker.pop("S1")
        assert record is not None
        assert tracker.get("S1") is None
