"""
Canompx3 Trading Co-Pilot.

Single-page operational dashboard. Shows what to trade, when, and why.

Launch: streamlit run ui/app.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

st.set_page_config(
    page_title="Trading Co-Pilot",
    page_icon="Au",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from ui.copilot import render

render()
