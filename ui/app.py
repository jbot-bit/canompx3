"""
Canompx3 Dashboard -- Main Streamlit app.

Launch: streamlit run ui/app.py
"""

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="Canompx3 Dashboard",
    page_icon="Au",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Page imports (deferred to avoid slow startup)
from ui.pages import portfolio, strategies, market_state, data_quality
from ui.chat import render_chat

# Sidebar navigation
st.sidebar.title("Canompx3")
st.sidebar.caption("Gold (MGC) Trading Research")

page = st.sidebar.radio(
    "Navigate",
    ["Portfolio", "Strategies", "Market State", "Data Quality"],
    index=0,
)

# Render selected page
if page == "Portfolio":
    portfolio.render()
elif page == "Strategies":
    strategies.render()
elif page == "Market State":
    market_state.render()
elif page == "Data Quality":
    data_quality.render()

# AI Chat (always visible in sidebar below navigation)
render_chat()
