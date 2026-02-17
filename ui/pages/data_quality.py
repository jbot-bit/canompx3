"""
Data Quality page.

Row counts, date coverage, bar distribution, gap days, contract timeline.
"""


import plotly.express as px
import streamlit as st


from ui.db_reader import (
    get_table_counts,
    get_date_ranges,
    get_bars_per_day,
    get_gap_days,
    get_contract_timeline,
)

@st.cache_data(ttl=300)
def _cached_bars_per_day():
    return get_bars_per_day()

@st.cache_data(ttl=300)
def _cached_contract_timeline():
    return get_contract_timeline()

def render():
    st.header("Data Quality")

    # Row counts
    st.subheader("Table Row Counts")
    counts = get_table_counts()
    cols = st.columns(3)
    for i, (table, count) in enumerate(counts.items()):
        with cols[i % 3]:
            display = f"{count:,}" if count >= 0 else "missing"
            st.metric(table, display)

    # Date ranges
    st.subheader("Date Coverage")
    ranges = get_date_ranges()
    for table, r in ranges.items():
        if r:
            st.markdown(f"**{table}:** {r.get('min', '?')} to {r.get('max', '?')}")

    # Bars per day distribution
    st.subheader("Bars Per Day Distribution")
    bars_df = _cached_bars_per_day()
    if not bars_df.empty:
        fig = px.histogram(
            bars_df,
            x="bar_count_1m",
            nbins=50,
            title="1-Minute Bar Count Distribution",
        )
        fig.update_layout(
            xaxis_title="Bar Count",
            yaxis_title="Number of Days",
            template="plotly_dark",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Days", len(bars_df))
        with col2:
            st.metric("Median Bars", int(bars_df["bar_count_1m"].median()))
        with col3:
            st.metric("Min Bars", int(bars_df["bar_count_1m"].min()))
        with col4:
            st.metric("Max Bars", int(bars_df["bar_count_1m"].max()))

    # Gap days
    st.subheader("Gap Days (< 500 bars)")
    gaps_df = get_gap_days()
    if gaps_df.empty:
        st.success("No gap days detected.")
    else:
        st.warning(f"{len(gaps_df)} days with low bar counts")
        st.dataframe(gaps_df, use_container_width=True, hide_index=True, height=200)

    # Contract timeline
    st.subheader("Contract Roll Timeline")
    contracts_df = _cached_contract_timeline()
    if not contracts_df.empty:
        st.dataframe(contracts_df, use_container_width=True, hide_index=True, height=300)

        # Timeline chart
        fig = px.timeline(
            contracts_df.rename(columns={
                "first_seen": "Start",
                "last_seen": "Finish",
                "source_symbol": "Contract",
            }),
            x_start="Start",
            x_end="Finish",
            y="Contract",
            title="Contract Usage Timeline",
        )
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No contract data available.")
