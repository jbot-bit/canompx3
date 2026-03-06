"""Discipline Coach UI components — debrief card, cooling screen, pre-session priming.

Pure Streamlit rendering. Reads live_signals.jsonl (written by orchestrator),
writes to data/trade_debriefs.jsonl and data/discipline_state.jsonl.
No orchestrator or execution engine changes.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from ui.discipline_data import (
    ADHERENCE_VALUES,
    DEVIATION_TRIGGERS,
    DEBRIEFS_PATH,
    STATE_PATH,
    append_debrief,
    append_discipline_event,
    compute_adherence_stats,
    cooling_remaining_seconds,
    get_latest_letter,
    get_pending_debriefs,
    is_cooling_active,
    override_cooling,
    trigger_cooling,
)

# Signals file — same path as copilot.py
_SIGNALS_FILE = Path(__file__).parent.parent / "live_signals.jsonl"


# -- Debrief card ----------------------------------------------------------


def render_pending_debriefs(
    *,
    signals_path: Path = _SIGNALS_FILE,
    debriefs_path: Path = DEBRIEFS_PATH,
    state_path: Path = STATE_PATH,
) -> None:
    """Render debrief forms for any exit signals without a matching debrief."""
    pending = get_pending_debriefs(signals_path=signals_path, debriefs_path=debriefs_path)
    if not pending:
        return

    # Auto-trigger cooling on losing exits (if not already cooling)
    for ex in pending:
        pnl_r = ex.get("pnl_r")
        if pnl_r is not None and pnl_r < 0 and not is_cooling_active(st.session_state):
            trigger_cooling(
                st.session_state,
                pnl_r=pnl_r,
                consecutive_losses=1,  # simplified for MVP
                session_pnl_r=pnl_r,
                state_path=state_path,
            )
            break  # one trigger per render cycle

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


# -- Cooling period --------------------------------------------------------

_TRADING_QUOTES = [
    "The goal of a successful trader is to make the best trades. Money is secondary.",
    "It's not whether you're right or wrong, but how much you make when right and lose when wrong.",
    "The market can stay irrational longer than you can stay solvent.",
    "In trading, the impossible happens about twice a year.",
    "The elements of good trading are: cutting losses, cutting losses, and cutting losses.",
    "Trade what you see, not what you think.",
    "Discipline is the bridge between goals and accomplishment.",
    "The best trade is the one you didn't take.",
]


def check_cooling(
    *,
    state_path: Path = STATE_PATH,
) -> bool:
    """Check if cooling period is active. If active, render cooling screen.

    Returns True if cooling is active (caller should skip signal rendering).
    """
    if not is_cooling_active(st.session_state):
        return False

    remaining = cooling_remaining_seconds(st.session_state)
    mode = st.session_state.get("cooling_mode", "hard")

    # Progress bar
    progress = 1.0 - (remaining / 90.0)
    st.progress(min(progress, 1.0), text=f"Cooling: {int(remaining)}s remaining")

    # Cooling content
    st.markdown("**Take a breath.**")
    quote = random.choice(_TRADING_QUOTES)
    st.markdown(f'*"{quote}"*')
    st.caption("Wait for the signal. The plan is the edge.")

    # Soft mode: override button after 15s
    if mode == "soft" and remaining < 75:  # 90 - 15 = 75
        if st.button("Override cooling", type="secondary"):
            override_cooling(st.session_state, state_path=state_path)
            st.rerun()

    return mode == "hard" or remaining >= 75


def render_cooling_settings() -> None:
    """Render cooling mode toggle in sidebar."""
    current = st.session_state.get("cooling_mode", "hard")
    mode = st.radio(
        "Cooling mode",
        options=["hard", "soft"],
        index=0 if current == "hard" else 1,
        format_func=lambda x: f"{x.title()} (90s {'non-dismissable' if x == 'hard' else 'dismissable after 15s'})",
        key="cooling_mode_radio",
        horizontal=True,
    )
    st.session_state["cooling_mode"] = mode


# -- Pre-session priming ---------------------------------------------------


def render_pre_session_priming(
    *,
    session: str,
    strategies: list,
    debriefs_path: Path = DEBRIEFS_PATH,
    state_path: Path = STATE_PATH,
) -> None:
    """Render pre-session priming card: stats, plan, commitment, letter from past self.

    Args:
        session: Session name (e.g. "CME_REOPEN")
        strategies: List of PortfolioStrategy for this session
    """
    st.markdown("**Pre-Session Priming**")

    # Pattern stats
    stats = compute_adherence_stats(path=debriefs_path, session=session)
    if stats["total"] > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Adherence",
                f"{stats['adherence_rate']:.0%}",
                help=f"{stats['followed']}/{stats['total']} signals followed",
            )
        with col2:
            st.metric("Avg R (followed)", f"{stats['avg_r_followed']:+.2f}")
        with col3:
            st.metric("Deviation cost", f"${stats['deviation_cost_dollars']:,.0f}")
    else:
        st.caption("No debrief history yet for this session.")

    # Today's plan
    if strategies:
        st.markdown("**Today's Plan**")
        for s in strategies:
            st.markdown(
                f"- **{s.instrument}** {s.entry_model} CB{s.confirm_bars} "
                f"{s.filter_type} RR{s.rr_target} ({s.orb_minutes}m ORB)"
            )
        st.caption("Action rule: Execute within 60s of signal.")

    # Commitment button
    committed_key = f"committed_{session}_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
    already_committed = st.session_state.get(committed_key, False)

    if already_committed:
        st.success("Committed to the plan.")
    else:
        if st.button("I commit to following the plan", type="primary", key=f"commit_{session}"):
            st.session_state[committed_key] = True
            append_discipline_event(
                "commitment",
                {"session": session},
                path=state_path,
            )
            st.rerun()

    # Letter from past self
    letter = get_latest_letter(session=session, path=debriefs_path)
    if letter:
        st.markdown("---")
        st.markdown("**Letter from your past self:**")
        st.info(f'"{letter["text"]}"')
        ts_str = letter.get("ts", "")
        if ts_str:
            try:
                dt = datetime.fromisoformat(ts_str)
                st.caption(f"Written {dt.strftime('%b %d')} after {letter.get('strategy_id', '')}")
            except ValueError:
                pass
