"""
Portfolio Overview page.

Shows validated strategy summary, equity curve, correlation heatmap,
and position sizing calculator.
"""

import sys
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH
from pipeline.cost_model import get_cost_spec
from trading_app.portfolio import (
    build_portfolio,
    build_strategy_daily_series,
    correlation_matrix,
    compute_position_size,
)
from ui.db_reader import get_validated_strategies


@st.cache_data(ttl=300)
def _cached_daily_series(_db_path, strategy_ids: tuple):
    """Cached wrapper for build_strategy_daily_series (avoids re-query on every rerun)."""
    return build_strategy_daily_series(_db_path, list(strategy_ids))


@st.cache_data(ttl=300)
def _cached_correlation(_db_path, strategy_ids: tuple):
    """Cached wrapper for correlation_matrix."""
    return correlation_matrix(_db_path, list(strategy_ids))


def render():
    st.header("Portfolio Overview")

    # --- Strategy Summary ---
    st.subheader("Validated Strategies")
    df = get_validated_strategies()
    if df.empty:
        st.warning("No validated strategies found in DB.")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Strategies", len(df))
    with col2:
        st.metric("Avg ExpR", f"{df['expectancy_r'].mean():.3f}")
    with col3:
        st.metric("Avg Win Rate", f"{df['win_rate'].mean():.1%}")
    with col4:
        st.metric("Avg Sharpe", f"{df['sharpe_ratio'].mean():.3f}" if "sharpe_ratio" in df else "N/A")

    # By session
    st.markdown("**By Session**")
    session_summary = df.groupby("orb_label").agg(
        count=("strategy_id", "count"),
        avg_expr=("expectancy_r", "mean"),
        avg_wr=("win_rate", "mean"),
    ).reset_index()
    st.dataframe(session_summary, use_container_width=True, hide_index=True)

    # By entry model
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**By Entry Model**")
        em_summary = df.groupby("entry_model").agg(
            count=("strategy_id", "count"),
            avg_expr=("expectancy_r", "mean"),
        ).reset_index()
        st.dataframe(em_summary, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**By Filter Type**")
        ft_summary = df.groupby("filter_type").agg(
            count=("strategy_id", "count"),
            avg_expr=("expectancy_r", "mean"),
        ).reset_index()
        st.dataframe(ft_summary, use_container_width=True, hide_index=True)

    # --- Build Portfolio ---
    st.subheader("Portfolio Builder")
    col1, col2, col3 = st.columns(3)
    with col1:
        max_strats = st.slider("Max strategies", 5, 50, 20)
        include_nested = st.checkbox("Include nested ORB", value=False)
    with col2:
        min_expr = st.slider("Min ExpR", 0.0, 0.5, 0.10, 0.01)
        max_per_orb = st.slider("Max per ORB", 1, 10, 5)
    with col3:
        account_equity = st.number_input("Account equity ($)", 5000, 500000, 25000, 5000)
        risk_pct = st.slider("Risk per trade (%)", 0.5, 5.0, 2.0, 0.5)

    if st.button("Build Portfolio"):
        with st.spinner("Building portfolio..."):
            try:
                portfolio = build_portfolio(
                    db_path=GOLD_DB_PATH,
                    max_strategies=max_strats,
                    min_expectancy_r=min_expr,
                    max_per_orb=max_per_orb,
                    account_equity=account_equity,
                    risk_per_trade_pct=risk_pct,
                    include_nested=include_nested,
                )
                st.session_state["portfolio"] = portfolio
                st.success(f"Built portfolio with {len(portfolio.strategies)} strategies")
            except Exception as e:
                st.error(f"Portfolio build failed: {e}")

    # --- Equity Curve ---
    if "portfolio" in st.session_state:
        portfolio = st.session_state["portfolio"]
        strategy_ids = [s.strategy_id for s in portfolio.strategies]

        st.subheader("Equity Curve (Cumulative R)")
        with st.spinner("Loading daily series..."):
            try:
                series_df, stats = _cached_daily_series(GOLD_DB_PATH, tuple(strategy_ids))
                # Sum across strategies for combined equity
                combined = series_df.sum(axis=1).cumsum()
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=combined.index, y=combined.values,
                    mode="lines", name="Combined",
                    line=dict(color="gold", width=2),
                ))
                fig.update_layout(
                    xaxis_title="Trading Day",
                    yaxis_title="Cumulative R",
                    template="plotly_dark",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Stats
                total_r = combined.iloc[-1] if len(combined) > 0 else 0
                max_dd = (combined - combined.cummax()).min()
                st.markdown(f"**Total R:** {total_r:.1f} | **Max Drawdown:** {max_dd:.1f}R")
            except Exception as e:
                st.error(f"Failed to build equity curve: {e}")

        # --- Correlation Heatmap ---
        st.subheader("Strategy Correlation")
        if len(strategy_ids) >= 2:
            with st.spinner("Computing correlations..."):
                try:
                    corr = _cached_correlation(GOLD_DB_PATH, tuple(strategy_ids))
                    fig = px.imshow(
                        corr,
                        color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1,
                        aspect="auto",
                    )
                    fig.update_layout(
                        template="plotly_dark",
                        height=500,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Correlation failed: {e}")
        else:
            st.info("Need at least 2 strategies for correlation.")

        # --- Position Sizing ---
        st.subheader("Position Sizing")
        cost_spec = get_cost_spec("MGC")
        risk_points = st.number_input("Risk (points)", 1.0, 50.0, 5.0, 0.5)
        contracts = compute_position_size(account_equity, risk_pct, risk_points, cost_spec)
        risk_dollars = risk_points * cost_spec.point_value + cost_spec.total_friction
        st.markdown(
            f"**Contracts:** {contracts} | "
            f"**Risk/trade:** ${risk_dollars:.2f} | "
            f"**Friction:** ${cost_spec.total_friction:.2f}"
        )
