"""
Live Trading Monitor page.

Two modes available from the UI:
  - Signal Only: watch ⚡ signals, trade manually on Tradovate/TradingView
  - Demo:        auto-place paper orders on Tradovate DEMO account

For LIVE (real money) orders, use the terminal:
    python scripts/run_live_session.py --instrument MGC --live
"""
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

# Signals file written by SessionOrchestrator, read here
_SIGNALS_FILE = Path(__file__).parent.parent.parent / "live_signals.jsonl"

# Stop-file: DataFeed's heartbeat checks for this and shuts down gracefully.
# On Windows, proc.terminate() calls TerminateProcess() (hard kill) which skips
# all finally blocks and post_session() cleanup. Writing this file instead
# gives the session a chance to close positions before exiting.
_STOP_FILE = Path(__file__).parent.parent.parent / "live_session.stop"

# Colour map for event type badges
_TYPE_COLOURS = {
    "SESSION_START": "blue",
    "SIGNAL_ENTRY": "green",
    "SIGNAL_EXIT": "orange",
    "ORDER_ENTRY": "green",
    "ORDER_EXIT": "orange",
    "ORDER_SCRATCH": "orange",
    "REJECT": "red",
}


def render() -> None:
    st.header("Live Trading Monitor")
    st.caption(
        "Signal-only and Demo modes only. "
        "For live real-money orders use the terminal: "
        "`python scripts/run_live_session.py --instrument MGC --live`"
    )

    proc: subprocess.Popen | None = st.session_state.get("live_proc")
    is_running = proc is not None and proc.poll() is None

    # ── Config (locked while session is running) ──────────────────────────────
    col_inst, col_mode = st.columns(2)
    with col_inst:
        instrument = st.selectbox(
            "Instrument",
            ["MGC", "MNQ", "MES", "M2K"],
            disabled=is_running,
        )
    with col_mode:
        mode_label = st.selectbox(
            "Mode",
            ["Signal Only  (no orders placed — safest)", "Demo  (paper orders on Tradovate)"],
            disabled=is_running,
        )

    # ── Start / Stop / Status ─────────────────────────────────────────────────
    col_start, col_stop, col_status = st.columns([1, 1, 4])

    with col_start:
        if st.button("▶  Start", disabled=is_running, type="primary", width="stretch"):
            _start_session(instrument, mode_label)
            st.rerun()

    with col_stop:
        if st.button("⏹  Stop", disabled=not is_running, type="secondary", width="stretch"):
            _stop_session()
            st.rerun()

    with col_status:
        if is_running:
            inst = st.session_state.get("live_instrument", instrument)
            mode = st.session_state.get("live_mode_short", "signal-only")
            st.success(f"● RUNNING — {inst}  [{mode}]   pid={proc.pid}")
        else:
            exit_code = proc.returncode if proc else None
            if exit_code is not None and exit_code != 0:
                st.error(f"● STOPPED  (exit code {exit_code})")
            else:
                st.info("● STOPPED")

    st.divider()

    # ── Signal log ────────────────────────────────────────────────────────────
    _render_signal_log()

    # ── Auto-refresh while session is live ────────────────────────────────────
    if is_running:
        time.sleep(2)
        st.rerun()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _truncate_file(path: Path) -> None:
    """Truncate a file to zero bytes without deleting it.

    Windows raises PermissionError if you unlink() a file while another process
    has it open. Truncating avoids that — the session process opens/closes the
    signals file per-write so the window is tiny, but truncation is safer.
    """
    try:
        with open(path, "w"):
            pass
    except OSError:
        pass


def _start_session(instrument: str, mode_label: str) -> None:
    is_signal_only = mode_label.startswith("Signal")
    flag = "--signal-only" if is_signal_only else "--demo"
    mode_short = "signal-only" if is_signal_only else "demo"

    # Clear previous signal log and any leftover stop file
    if _SIGNALS_FILE.exists():
        _truncate_file(_SIGNALS_FILE)
    _STOP_FILE.unlink(missing_ok=True)

    cmd = [
        sys.executable,
        "scripts/run_live_session.py",
        "--instrument", instrument,
        flag,
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,  # logs go to terminal where Streamlit was launched
        stderr=subprocess.DEVNULL,
    )
    st.session_state["live_proc"] = proc
    st.session_state["live_instrument"] = instrument
    st.session_state["live_mode_short"] = mode_short


def _stop_session() -> None:
    proc: subprocess.Popen | None = st.session_state.get("live_proc")
    if proc:
        # Write the stop file so DataFeed shuts down gracefully (closes positions,
        # runs post_session). On Windows, proc.terminate() is a hard kill that
        # bypasses all cleanup — the stop file is the safe alternative.
        _STOP_FILE.touch()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            # Graceful shutdown timed out — hard kill as last resort
            _STOP_FILE.unlink(missing_ok=True)
            proc.kill()
    st.session_state.pop("live_proc", None)


def _render_signal_log() -> None:
    col_head, col_clear = st.columns([5, 1])
    with col_head:
        st.subheader("Signal Log")
    with col_clear:
        if st.button("Clear", width="stretch") and _SIGNALS_FILE.exists():
            _truncate_file(_SIGNALS_FILE)
            st.rerun()

    if not _SIGNALS_FILE.exists():
        st.info("No signals yet. Start a session to begin streaming.")
        return

    try:
        raw = _SIGNALS_FILE.read_text(encoding="utf-8").strip()
        lines = [ln for ln in raw.split("\n") if ln.strip()]
        records = [json.loads(ln) for ln in lines]
    except Exception as exc:
        st.error(f"Error reading signal log: {exc}")
        return

    if not records:
        st.info("Session started — waiting for the first ORB signal...")
        return

    # Show most recent first, cap at 100 rows
    records = records[-100:][::-1]

    rows = []
    for r in records:
        ts_raw = r.get("ts", "")
        try:
            ts = datetime.fromisoformat(ts_raw).astimezone(timezone.utc).strftime("%H:%M:%S UTC")
        except ValueError:
            ts = ts_raw

        event_type = r.get("type", "")
        colour = _TYPE_COLOURS.get(event_type, "gray")
        badge = f":{colour}[{event_type}]"

        rows.append({
            "Time": ts,
            "Event": badge,
            "Instrument": r.get("instrument", ""),
            "Strategy": r.get("strategy_id", "—"),
            "Direction": r.get("direction", "—"),
            "Price": r.get("price", "—"),
            "Contract": r.get("contract", "—"),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)
