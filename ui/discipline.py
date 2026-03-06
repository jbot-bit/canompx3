"""Discipline Coach UI components — debrief card, cooling screen, pre-session priming.

Pure Streamlit rendering. Reads live_signals.jsonl (written by orchestrator),
writes to data/trade_debriefs.jsonl and data/discipline_state.jsonl.
No orchestrator or execution engine changes.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from ui.discipline_data import (
    ADHERENCE_VALUES,
    DEVIATION_TRIGGERS,
    DEBRIEFS_PATH,
    STATE_PATH,
    append_debrief,
    get_pending_debriefs,
    is_cooling_active,
    trigger_cooling,
)

# Signals file — same path as copilot.py
_SIGNALS_FILE = Path(__file__).parent.parent / "live_signals.jsonl"


# -- Debrief card ----------------------------------------------------------


def render_pending_debriefs(
    *,
    signals_path: Path = _SIGNALS_FILE,
    debriefs_path: Path = DEBRIEFS_PATH,
) -> None:
    """Render debrief forms for any exit signals without a matching debrief."""
    pending = get_pending_debriefs(
        signals_path=signals_path, debriefs_path=debriefs_path
    )
    if not pending:
        return

    st.markdown("**Post-Trade Debrief**")

    for exit_signal in pending:
        strategy_id = exit_signal.get("strategy_id", "unknown")
        exit_ts = exit_signal.get("ts", "")
        exit_price = exit_signal.get("price", "")
        instrument = exit_signal.get("instrument", "")

        form_key = f"debrief_{strategy_id}_{exit_ts}"

        with st.form(key=form_key):
            st.markdown(f"**{strategy_id}** exited @ {exit_price}")

            # Layer 2: Adherence classification
            adherence = st.radio(
                "How did you execute?",
                options=list(ADHERENCE_VALUES),
                format_func=lambda x: x.replace("_", " ").title(),
                horizontal=True,
                key=f"adh_{form_key}",
            )

            # Layer 3: Deviation trigger (conditional — shown always in form,
            # but only saved when adherence != followed)
            deviation_trigger = st.selectbox(
                "What caused the deviation?",
                options=[None] + list(DEVIATION_TRIGGERS),
                format_func=lambda x: (x or "---").replace("_", " ").title(),
                key=f"dev_{form_key}",
            )

            # Layer 4: Emotional temperature
            emotional_temp = st.slider(
                "Emotional temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="0.0 = calm, 1.0 = hot",
                key=f"emo_{form_key}",
            )

            # Layer 5: Letter to future self (conditional)
            letter = st.text_area(
                "Letter to future self (optional)",
                placeholder="What do you want to remind yourself next time?",
                key=f"letter_{form_key}",
            )

            submitted = st.form_submit_button("Save Debrief")

            if submitted:
                record = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "trading_day": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "instrument": instrument,
                    "strategy_id": strategy_id,
                    "signal_exit_ts": exit_ts,
                    "exit_price": exit_price,
                    "adherence": adherence,
                    "deviation_trigger": deviation_trigger if adherence != "followed" else None,
                    "emotional_temp": emotional_temp,
                    "letter_to_future_self": letter if letter else None,
                }
                append_debrief(record, path=debriefs_path)
                st.success("Debrief saved.")
                st.rerun()
