"""Tests for discipline UI components — debrief card, cooling screen, priming."""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_render_pending_debriefs_no_exits(tmp_path):
    """No exit signals -> no debrief cards rendered."""
    from ui.discipline import render_pending_debriefs
    signals_path = tmp_path / "signals.jsonl"
    debriefs_path = tmp_path / "debriefs.jsonl"
    signals_path.write_text("")
    with patch("ui.discipline.st") as mock_st:
        render_pending_debriefs(signals_path=signals_path, debriefs_path=debriefs_path)
        mock_st.form.assert_not_called()


def test_render_pending_debriefs_shows_form(tmp_path):
    """Exit signal without debrief -> form rendered."""
    from ui.discipline import render_pending_debriefs
    signals_path = tmp_path / "signals.jsonl"
    debriefs_path = tmp_path / "debriefs.jsonl"
    signal = {
        "ts": "2026-03-06T23:15:00Z",
        "instrument": "MGC",
        "type": "SIGNAL_EXIT",
        "strategy_id": "MGC_CME_REOPEN_E2_CB1_G4_RR2.5",
        "price": 3245.50,
    }
    signals_path.write_text(json.dumps(signal) + "\n")
    with patch("ui.discipline.st") as mock_st:
        mock_form = MagicMock()
        mock_st.form.return_value.__enter__ = MagicMock(return_value=mock_form)
        mock_st.form.return_value.__exit__ = MagicMock(return_value=False)
        # Prevent the submit branch from executing (MagicMock is truthy)
        mock_st.form_submit_button.return_value = False
        render_pending_debriefs(signals_path=signals_path, debriefs_path=debriefs_path)
        mock_st.form.assert_called_once()


def test_check_cooling_returns_false_when_not_active():
    from ui.discipline import check_cooling
    with patch("ui.discipline.st") as mock_st:
        mock_st.session_state = {}
        assert check_cooling() is False


def test_check_cooling_returns_true_when_active():
    from ui.discipline import check_cooling
    from datetime import datetime, timezone, timedelta
    with patch("ui.discipline.st") as mock_st:
        until = (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat()
        mock_st.session_state = {"cooling_until": until, "cooling_mode": "hard"}
        assert check_cooling() is True


def test_render_pre_session_priming_shows_commitment(tmp_path):
    from ui.discipline import render_pre_session_priming
    debriefs_path = tmp_path / "debriefs.jsonl"
    state_path = tmp_path / "state.jsonl"
    with patch("ui.discipline.st") as mock_st:
        mock_st.session_state = {}
        mock_st.button.return_value = False
        render_pre_session_priming(
            session="CME_REOPEN",
            strategies=[],
            debriefs_path=debriefs_path,
            state_path=state_path,
        )
        mock_st.button.assert_called()
