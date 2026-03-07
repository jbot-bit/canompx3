# ui/copilot.py
"""
Trading Co-Pilot — single-page operational dashboard.

Renders the appropriate view based on the current app state:
  WEEKEND     — "Markets closed" with next Monday info
  IDLE        — Completed sessions recap, calm "next session" text
  APPROACHING — Countdown + briefing cards appearing
  ALERT       — Full briefing cards, start session button
  OVERNIGHT   — Dimmed overnight session list
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from datetime import time as dt_time
from pathlib import Path

import streamlit as st

from pipeline.dst import SESSION_CATALOG
from ui.db_reader import (
    get_previous_trading_day,
    get_prior_day_atr,
    get_today_completed_sessions,
)
from ui.discipline import (
    check_cooling,
    render_cooling_settings,
    render_pending_debriefs,
    render_pre_session_priming,
)
from ui.session_helpers import (
    AWAKE_END,
    AWAKE_START,
    BRISBANE,
    AppState,
    SessionBriefing,
    build_session_briefings,
    get_app_state,
    get_refresh_seconds,
    get_upcoming_sessions,
)

# Signals file written by SessionOrchestrator
_SIGNALS_FILE = Path(__file__).parent.parent / "live_signals.jsonl"
_STOP_FILE = Path(__file__).parent.parent / "live_session.stop"


# ── Cached data ──────────────────────────────────────────────────────────────


@st.cache_data(ttl=300)
def _cached_briefings() -> list[SessionBriefing]:
    """Cache briefing cards for 5 minutes."""
    return build_session_briefings()


@st.cache_data(ttl=300)
def _cached_atr(instrument: str) -> float | None:
    return get_prior_day_atr(instrument)


# ── Header bar ───────────────────────────────────────────────────────────────


def _render_header(now: datetime, state: AppState) -> None:
    """Top bar: date, time, session dots."""
    # Format times
    bris_time = now.strftime("%I:%M %p").lstrip("0")
    # ET = Brisbane - 15h (EST) or -14h (EDT). Approximate.
    et_offset = timedelta(hours=-15) if now.month >= 11 or now.month <= 2 else timedelta(hours=-14)
    et_time = (now + et_offset).strftime("%I:%M %p").lstrip("0")
    day_str = now.strftime("%a %d %b %Y")

    col_date, col_time = st.columns([2, 3])
    with col_date:
        st.markdown(f"### {day_str}")
    with col_time:
        st.markdown(f"**{bris_time} Brisbane** &nbsp;&nbsp; ({et_time} ET)")

    # Session dot strip — only sessions with live portfolio strategies
    briefings = _cached_briefings()
    active_sessions = sorted(set(b.session for b in briefings), key=lambda s: _session_sort_key(s, now))

    if active_sessions and not state.name == "WEEKEND":
        upcoming_names = {name for name, _ in get_upcoming_sessions(now)}
        dots = []
        for session in active_sessions:
            if session == state.next_session:
                dots.append(f":blue[**{session}**]")
            elif session in upcoming_names:
                dots.append(f":gray[{session}]")
            else:
                dots.append(f":gray[~~{session}~~]")  # completed
        st.caption(" &bull; ".join(dots))

    st.divider()


def _session_sort_key(session: str, now: datetime) -> float:
    """Sort sessions by their datetime relative to now."""
    if session not in SESSION_CATALOG:
        return 9999.0
    today = now.date()
    tomorrow = today + timedelta(days=1)
    for cal_day in [today, tomorrow]:
        h, m = SESSION_CATALOG[session]["resolver"](cal_day)
        dt = datetime.combine(cal_day, dt_time(h, m), tzinfo=BRISBANE)
        if dt > now - timedelta(hours=12):
            return dt.timestamp()
    return 9999.0


# ── State renderers ──────────────────────────────────────────────────────────


def _render_weekend(state: AppState) -> None:
    """Markets closed view."""
    st.markdown(
        "<h1 style='text-align:center; color:#888; margin-top:80px;'>Markets Closed</h1>",
        unsafe_allow_html=True,
    )
    if state.next_monday:
        st.markdown(
            f"<p style='text-align:center; color:#666; font-size:1.3rem;'>"
            f"Next trading day: <b>Monday {state.next_monday.strftime('%d %b')}</b>"
            f" &mdash; CME_REOPEN 9:00 AM</p>",
            unsafe_allow_html=True,
        )

    # Show last trading day summary
    st.divider()
    _render_previous_day_summary(state.trading_day)


def _render_idle(state: AppState, now: datetime) -> None:
    """Long gap between sessions — show recap + next session (no countdown)."""
    # Completed sessions recap
    _render_today_summary(state.trading_day, now)

    st.markdown("---")

    # Next session — calm, not urgent
    if state.next_session and state.next_session_dt:
        session_time = state.next_session_dt.strftime("%I:%M %p").lstrip("0")
        hours = int(state.minutes_to_next // 60) if state.minutes_to_next else 0
        mins = int(state.minutes_to_next % 60) if state.minutes_to_next else 0

        if hours > 0:
            gap_str = f"{hours}h {mins}m"
        else:
            gap_str = f"{mins}m"

        st.markdown(
            f"<p style='text-align:center; color:#888; font-size:1.1rem; margin-top:30px;'>"
            f"Next: <b>{state.next_session}</b> &middot; {session_time}"
            f" &mdash; in {gap_str}</p>",
            unsafe_allow_html=True,
        )
        if state.then_session and state.then_session_dt:
            then_time = state.then_session_dt.strftime("%I:%M %p").lstrip("0")
            st.markdown(
                f"<p style='text-align:center; color:#666; font-size:0.9rem;'>"
                f"then {state.then_session} &middot; {then_time}</p>",
                unsafe_allow_html=True,
            )


def _render_approaching(state: AppState, now: datetime) -> None:
    """15-60 min to session — countdown appears, briefing cards start showing."""
    mins = int(state.minutes_to_next) if state.minutes_to_next else 0
    session_time = state.next_session_dt.strftime("%I:%M %p").lstrip("0") if state.next_session_dt else ""

    st.markdown(
        f"<h2 style='text-align:center;'>{state.next_session} &middot; {session_time}</h2>"
        f"<h1 style='text-align:center; font-size:3rem;'>in {mins} minutes</h1>",
        unsafe_allow_html=True,
    )
    if state.then_session and state.then_session_dt:
        then_time = state.then_session_dt.strftime("%I:%M %p").lstrip("0")
        st.markdown(
            f"<p style='text-align:center; color:#888;'>then {state.then_session} &middot; {then_time}</p>",
            unsafe_allow_html=True,
        )

    st.divider()

    # Pre-session priming
    briefings = _cached_briefings()
    session_strategies = [b for b in briefings if b.session == state.next_session]
    render_pre_session_priming(session=state.next_session, strategies=session_strategies)
    st.divider()

    # Show briefing cards for next session
    _render_briefing_cards(state.next_session, now)

    # Start session button
    _render_session_controls()


def _render_alert(state: AppState, now: datetime) -> None:
    """<15 min — full briefing, urgent styling."""
    mins = int(state.minutes_to_next) if state.minutes_to_next else 0
    session_time = state.next_session_dt.strftime("%I:%M %p").lstrip("0") if state.next_session_dt else ""

    st.markdown(
        f"<h2 style='text-align:center; color:#ff6b6b;'>"
        f"{state.next_session} &middot; {session_time}</h2>"
        f"<h1 style='text-align:center; font-size:4rem; color:#ff6b6b;'>"
        f"in {mins} min</h1>",
        unsafe_allow_html=True,
    )
    if state.then_session and state.then_session_dt:
        then_time = state.then_session_dt.strftime("%I:%M %p").lstrip("0")
        st.markdown(
            f"<p style='text-align:center; color:#888;'>then {state.then_session} &middot; {then_time}</p>",
            unsafe_allow_html=True,
        )

    st.divider()

    # Pre-session priming
    briefings = _cached_briefings()
    session_strategies = [b for b in briefings if b.session == state.next_session]
    render_pre_session_priming(session=state.next_session, strategies=session_strategies)
    st.divider()

    # Briefing cards — full detail
    _render_briefing_cards(state.next_session, now)

    # Start session button
    _render_session_controls()


def _render_overnight(state: AppState, now: datetime) -> None:
    """Next session is outside awake hours."""
    # Show today's recap first
    _render_today_summary(state.trading_day, now)

    st.markdown("---")

    # Overnight session list (dimmed)
    st.markdown(
        "<p style='text-align:center; color:#666; margin-top:30px;'>Overnight sessions</p>",
        unsafe_allow_html=True,
    )
    upcoming = get_upcoming_sessions(now)
    overnight = [(n, dt) for n, dt in upcoming if not (AWAKE_START <= dt.hour < AWAKE_END)]
    daytime = [(n, dt) for n, dt in upcoming if AWAKE_START <= dt.hour < AWAKE_END]

    if overnight:
        for name, dt in overnight[:6]:
            t = dt.strftime("%I:%M %p").lstrip("0")
            st.markdown(f"<p style='text-align:center; color:#555;'>{name} &middot; {t}</p>", unsafe_allow_html=True)

    if daytime:
        next_day_session = daytime[0]
        t = next_day_session[1].strftime("%I:%M %p").lstrip("0")
        st.markdown(
            f"<p style='text-align:center; color:#888; font-size:1.2rem; margin-top:20px;'>"
            f"Next morning: <b>{next_day_session[0]}</b> &middot; {t}</p>",
            unsafe_allow_html=True,
        )

    # Previous day summary
    st.divider()
    _render_previous_day_summary(state.trading_day)


# ── Briefing cards ───────────────────────────────────────────────────────────


def _render_briefing_cards(session: str, now: datetime) -> None:
    """Render instrument briefing cards for a session."""
    briefings = _cached_briefings()
    session_briefings = [b for b in briefings if b.session == session]

    if not session_briefings:
        st.info(f"No strategies configured for {session}")
        return

    for b in session_briefings:
        with st.container(border=True):
            # Instrument header
            instrument_names = {
                "MGC": "Micro Gold",
                "MNQ": "Micro Nasdaq",
                "MES": "Micro S&P",
                "M2K": "Micro Russell",
            }
            full_name = instrument_names.get(b.instrument, b.instrument)

            st.markdown(f"**{b.instrument}** &middot; {full_name}")
            st.markdown("")

            # Qualifying conditions
            if b.conditions:
                conditions_text = " OR ".join(b.conditions)
                st.markdown(f"**IF:** {conditions_text}")
            if b.direction_note:
                st.markdown(f"**Direction:** {b.direction_note}")

            # Entry instruction + RR
            st.markdown(f"**THEN:** {b.entry_instruction}")
            st.markdown(f"**Target:** {b.rr_target:.1f}x risk")

            # ATR context
            atr = _cached_atr(b.instrument)
            if atr is not None:
                # Find min G-threshold from conditions
                g_thresholds = []
                for c in b.conditions:
                    if c.startswith("ORB >= "):
                        try:
                            pts = int(c.split(">=")[1].split("pts")[0].strip())
                            g_thresholds.append(pts)
                        except (ValueError, IndexError):
                            pass
                if g_thresholds:
                    min_g = min(g_thresholds)
                    likelihood = "common" if atr > min_g * 3 else "typical" if atr > min_g * 1.5 else "marginal"
                    st.caption(f"Prior day ATR: {atr:.1f}pts — {min_g}pt ORB is {likelihood}")
                else:
                    st.caption(f"Prior day ATR: {atr:.1f}pts")

            # Strategy count
            st.caption(f"{b.strategy_count} strategies merged &middot; {b.orb_minutes}m ORB")


# ── Day summary ──────────────────────────────────────────────────────────────


def _render_today_summary(trading_day: date | None, now: datetime) -> None:
    """Show completed sessions for today's trading day."""
    if trading_day is None:
        return

    st.markdown("**Today**")

    briefings = _cached_briefings()
    active_sessions = sorted(set(b.session for b in briefings), key=lambda s: _session_sort_key(s, now))
    upcoming_names = {name for name, _ in get_upcoming_sessions(now)}

    for session in active_sessions:
        if session in SESSION_CATALOG:
            h, m = SESSION_CATALOG[session]["resolver"](now.date())
            t = f"{h % 12 or 12}:{m:02d} {'AM' if h < 12 else 'PM'}"
        else:
            t = ""

        if session in upcoming_names:
            st.markdown(f":gray[{session} &middot; {t} — waiting]")
        else:
            st.markdown(f":gray[~~{session} &middot; {t} — done~~]")


def _render_previous_day_summary(trading_day: date | None) -> None:
    """Show last trading day results."""
    if trading_day is None:
        return
    prev = get_previous_trading_day(trading_day)
    if prev is None:
        return

    results = get_today_completed_sessions(prev)
    if not results:
        st.caption(f"Last trading day ({prev}): no data")
        return

    total_r = sum(r.get("pnl_r", 0) or 0 for r in results)
    st.caption(f"Last trading day ({prev}): {len(results)} outcomes, {total_r:+.1f}R total")


# ── Session controls ─────────────────────────────────────────────────────────


def _render_session_controls() -> None:
    """Start/stop live session buttons."""
    proc = st.session_state.get("live_proc")
    is_running = proc is not None and proc.poll() is None

    if is_running:
        inst = st.session_state.get("live_instrument", "?")
        mode = st.session_state.get("live_mode_short", "signal-only")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(f"Session RUNNING — {inst} [{mode}] pid={proc.pid}")
        with col2:
            if st.button("Stop", type="secondary"):
                _stop_session()
                st.rerun()
    else:
        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            instrument = st.selectbox(
                "Instrument",
                ["MGC", "MNQ", "MES", "M2K"],
                key="copilot_instrument",
                label_visibility="collapsed",
            )
        with col2:
            if st.button("Start Signal-Only", type="primary"):
                _start_session(instrument, signal_only=True)
                st.rerun()
        with col3:
            if st.button("Start Demo"):
                _start_session(instrument, signal_only=False)
                st.rerun()


def _start_session(instrument: str, signal_only: bool) -> None:
    """Launch a live session subprocess."""
    if _SIGNALS_FILE.exists():
        try:
            with open(_SIGNALS_FILE, "w"):
                pass
        except OSError:
            pass
    _STOP_FILE.unlink(missing_ok=True)

    flag = "--signal-only" if signal_only else "--demo"
    cmd = [
        sys.executable,
        "scripts/run_live_session.py",
        "--instrument",
        instrument,
        flag,
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=Path(__file__).parent.parent,
    )
    st.session_state["live_proc"] = proc
    st.session_state["live_instrument"] = instrument
    st.session_state["live_mode_short"] = "signal-only" if signal_only else "demo"


def _stop_session() -> None:
    """Gracefully stop a live session."""
    proc = st.session_state.get("live_proc")
    if proc:
        _STOP_FILE.touch()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _STOP_FILE.unlink(missing_ok=True)
            proc.kill()
    st.session_state.pop("live_proc", None)


# ── Signal log ───────────────────────────────────────────────────────────────


def _render_signal_log() -> None:
    """Show recent signals from live_signals.jsonl."""
    if not _SIGNALS_FILE.exists():
        return

    try:
        raw = _SIGNALS_FILE.read_text(encoding="utf-8").strip()
        if not raw:
            return
        lines = [ln for ln in raw.split("\n") if ln.strip()]
        records = [json.loads(ln) for ln in lines[-20:]]  # last 20
    except Exception:
        return

    if not records:
        return

    st.markdown("**Live Signals**")
    for r in reversed(records):
        ts = r.get("ts", "")
        try:
            ts_display = datetime.fromisoformat(ts).astimezone(BRISBANE).strftime("%I:%M %p").lstrip("0")
        except (ValueError, TypeError):
            ts_display = ts

        event_type = r.get("type", "")
        instrument = r.get("instrument", "")
        strategy = r.get("strategy_id", "")
        price = r.get("price", "")

        colors = {
            "SIGNAL_ENTRY": "green",
            "ORDER_ENTRY": "green",
            "SIGNAL_EXIT": "orange",
            "ORDER_EXIT": "orange",
            "REJECT": "red",
            "SESSION_START": "blue",
        }
        color = colors.get(event_type, "gray")
        st.markdown(f":{color}[{ts_display}] {event_type} {instrument} {strategy} {price}")


# ── Main render ──────────────────────────────────────────────────────────────


def render() -> None:
    """Main entry point — renders the co-pilot dashboard."""
    now = datetime.now(BRISBANE)
    state = get_app_state(now)

    _render_header(now, state)

    # Sidebar — discipline settings
    with st.sidebar:
        render_cooling_settings()

    # State-dependent main area
    if state.name == "WEEKEND":
        _render_weekend(state)
    elif state.name == "OVERNIGHT":
        _render_overnight(state, now)
    elif state.name == "IDLE":
        _render_idle(state, now)
    elif state.name == "APPROACHING":
        _render_approaching(state, now)
    elif state.name == "ALERT":
        _render_alert(state, now)

    # Signal log (always, if running)
    proc = st.session_state.get("live_proc")
    if proc is not None and proc.poll() is None:
        st.divider()
        # Cooling check — may block signal rendering in hard mode
        cooling_active = check_cooling()
        if not cooling_active:
            _render_signal_log()
        # Debrief cards for any unprocessed exits
        render_pending_debriefs()

    # Adaptive refresh
    refresh = get_refresh_seconds(
        minutes_to_next=state.minutes_to_next or 999,
        is_weekend=state.name == "WEEKEND",
    )
    # Cap sleep at 5s to keep UI responsive — Streamlit blocks during sleep
    time.sleep(min(refresh, 5))
    st.rerun()
