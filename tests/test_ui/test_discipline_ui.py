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
