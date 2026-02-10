"""
Market State Viewer page.

Pick a date to see ORB snapshots, session signals, and strategy scores.
"""

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH
from ui.db_reader import get_daily_features


def render():
    st.header("Market State Viewer")

    # Date picker
    selected_date = st.date_input(
        "Trading Day",
        value=date(2025, 6, 15),
        min_value=date(2016, 2, 1),
        max_value=date(2026, 2, 4),
    )

    orb_minutes = st.selectbox("ORB Minutes", [5, 15, 30], index=0)

    if st.button("Load Market State"):
        with st.spinner("Loading..."):
            _load_and_display(selected_date, orb_minutes)

    # Auto-load if date changes
    if "last_ms_date" in st.session_state and st.session_state.last_ms_date != selected_date:
        _load_and_display(selected_date, orb_minutes)
    st.session_state.last_ms_date = selected_date


def _load_and_display(selected_date: date, orb_minutes: int):
    """Load MarketState and daily_features for display."""
    date_str = selected_date.isoformat()

    # Daily features (direct DB read)
    features = get_daily_features(date_str, orb_minutes)
    if features is None:
        st.warning(f"No daily_features found for {date_str} (orb_minutes={orb_minutes})")
        return

    # Key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Bar Count (1m)", int(features.get("bar_count_1m", 0)))
    with col2:
        rsi = features.get("rsi_14_at_0900")
        st.metric("RSI-14 at 0900", f"{rsi:.1f}" if rsi is not None else "N/A")
    with col3:
        st.metric("ORB Minutes", orb_minutes)

    # ORB snapshots table
    st.subheader("ORB Snapshots")
    orb_labels = ["0900", "1000", "1100", "1800", "2300", "0030"]
    orb_rows = []
    for label in orb_labels:
        prefix = f"orb_{label}_"
        high = features.get(f"{prefix}high")
        low = features.get(f"{prefix}low")
        size = features.get(f"{prefix}size")
        break_dir = features.get(f"{prefix}break_dir")
        outcome = features.get(f"{prefix}outcome")
        orb_rows.append({
            "ORB": label,
            "High": f"{high:.2f}" if high is not None else "-",
            "Low": f"{low:.2f}" if low is not None else "-",
            "Size": f"{size:.2f}" if size is not None else "-",
            "Break Dir": break_dir or "-",
            "Outcome": outcome or "-",
        })

    orb_df = pd.DataFrame(orb_rows)
    st.dataframe(orb_df, use_container_width=True, hide_index=True)

    # Session stats
    st.subheader("Session Ranges")
    session_rows = []
    for session in ["asia", "london", "ny"]:
        high = features.get(f"session_{session}_high")
        low = features.get(f"session_{session}_low")
        if high is not None and low is not None:
            session_rows.append({
                "Session": session.upper(),
                "High": f"{high:.2f}",
                "Low": f"{low:.2f}",
                "Range": f"{(high - low):.2f}",
            })
    if session_rows:
        st.dataframe(pd.DataFrame(session_rows), use_container_width=True, hide_index=True)

    # MarketState (full object with signals)
    st.subheader("MarketState Signals")
    try:
        from trading_app.market_state import MarketState
        ms = MarketState.from_trading_day(
            trading_day=selected_date,
            db_path=str(GOLD_DB_PATH),
            orb_minutes=orb_minutes,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Reversal Active:** {ms.signals.reversal_active}")
        with col2:
            st.markdown(f"**Chop Detected:** {ms.signals.chop_detected}")
        with col3:
            st.markdown(f"**Continuation:** {ms.signals.continuation}")

        if ms.signals.prior_outcomes:
            st.markdown("**Prior Outcomes:**")
            st.json(ms.signals.prior_outcomes)

        # Strategy scores if portfolio loaded
        if "portfolio" in st.session_state:
            st.subheader("Strategy Scores (Context-Adjusted)")
            portfolio = st.session_state["portfolio"]
            scores = ms.score_strategies(portfolio.strategies)
            if scores:
                scores_df = pd.DataFrame([
                    {"strategy_id": k, "score": f"{v:.3f}"}
                    for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)
                ])
                st.dataframe(scores_df, use_container_width=True, hide_index=True)
            else:
                st.info("No scores computed (check portfolio).")

    except Exception as e:
        st.warning(f"MarketState load failed: {e}")
        st.caption("Daily features displayed above are still valid.")
