"""
Strategy Explorer page.

Filterable, sortable table of validated strategies with detail view.
"""


import streamlit as st


from ui.db_reader import get_validated_strategies, query_df

def render():
    st.header("Strategy Explorer")

    df = get_validated_strategies()
    if df.empty:
        st.warning("No validated strategies found.")
        return

    # Filters
    st.subheader("Filters")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        sessions = ["All"] + sorted(df["orb_label"].unique().tolist())
        selected_session = st.selectbox("Session", sessions)

    with col2:
        entry_models = ["All"] + sorted(df["entry_model"].unique().tolist())
        selected_em = st.selectbox("Entry Model", entry_models)

    with col3:
        filter_types = ["All"] + sorted(df["filter_type"].unique().tolist())
        selected_ft = st.selectbox("Filter Type", filter_types)

    with col4:
        min_expr = st.number_input("Min ExpR", 0.0, 1.0, 0.0, 0.01)

    # Apply filters
    filtered = df.copy()
    if selected_session != "All":
        filtered = filtered[filtered["orb_label"] == selected_session]
    if selected_em != "All":
        filtered = filtered[filtered["entry_model"] == selected_em]
    if selected_ft != "All":
        filtered = filtered[filtered["filter_type"] == selected_ft]
    if min_expr > 0:
        filtered = filtered[filtered["expectancy_r"] >= min_expr]

    st.markdown(f"**{len(filtered)}** strategies match filters")

    # Display columns
    display_cols = [
        "strategy_id", "orb_label", "entry_model", "rr_target",
        "confirm_bars", "filter_type", "expectancy_r", "win_rate",
        "sample_size", "sharpe_ratio", "max_drawdown_r",
    ]
    available_cols = [c for c in display_cols if c in filtered.columns]

    # Sort
    sort_col = st.selectbox("Sort by", available_cols, index=available_cols.index("expectancy_r") if "expectancy_r" in available_cols else 0)
    sort_asc = st.checkbox("Ascending", value=False)
    filtered = filtered.sort_values(sort_col, ascending=sort_asc)

    # Table
    st.dataframe(
        filtered[available_cols],
        use_container_width=True,
        hide_index=True,
        height=400,
    )

    # Detail view
    st.subheader("Strategy Detail")
    strategy_ids = filtered["strategy_id"].tolist()
    if strategy_ids:
        selected_id = st.selectbox("Select strategy", strategy_ids)
        row = filtered[filtered["strategy_id"] == selected_id].iloc[0]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ExpR", f"{row['expectancy_r']:.4f}")
            st.metric("Win Rate", f"{row['win_rate']:.2%}")
        with col2:
            st.metric("Sample Size", int(row["sample_size"]))
            st.metric("Sharpe", f"{row.get('sharpe_ratio', 0):.3f}")
        with col3:
            st.metric("Max DD (R)", f"{row.get('max_drawdown_r', 0):.2f}")
            st.metric("RR Target", f"{row['rr_target']:.1f}")

        # Yearly results if available
        if "yearly_results" in row and row["yearly_results"]:
            st.markdown("**Yearly Results**")
            try:
                import json
                yearly = json.loads(row["yearly_results"]) if isinstance(row["yearly_results"], str) else row["yearly_results"]
                st.json(yearly)
            except Exception:
                st.text(str(row["yearly_results"]))

        # Outcomes for this strategy
        st.markdown("**Recent Outcomes**")
        try:
            outcomes_df = query_df(f"""
                SELECT trading_day, orb_label, entry_model, rr_target, confirm_bars,
                       direction, entry_price, pnl_r, outcome
                FROM orb_outcomes
                WHERE orb_label = '{row["orb_label"]}'
                  AND entry_model = '{row["entry_model"]}'
                  AND rr_target = {row["rr_target"]}
                  AND confirm_bars = {row["confirm_bars"]}
                ORDER BY trading_day DESC
                LIMIT 20
            """)
            if not outcomes_df.empty:
                st.dataframe(outcomes_df, use_container_width=True, hide_index=True)
            else:
                st.info("No outcomes found for this strategy.")
        except Exception as e:
            st.error(f"Failed to load outcomes: {e}")
    else:
        st.info("No strategies match the current filters.")
