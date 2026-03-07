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

# Gold/dark terminal theme — matches pipeline dashboard aesthetic
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@400;500;600;700;800&display=swap');

    /* Base typography */
    html, body, [class*="css"] {
        font-family: 'JetBrains Mono', monospace;
        color: #e0dcd0;
    }
    h1, h2, h3, h4, h5, h6,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Syne', sans-serif !important;
        letter-spacing: -0.02em;
        color: #c9a227 !important;
    }

    /* Main background */
    .stApp, [data-testid="stAppViewContainer"] {
        background: #0a0a0a;
    }
    [data-testid="stSidebar"] {
        background: #0d0d0d;
        border-right: 1px solid #1a1a1a;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }

    /* Cards and containers */
    [data-testid="stExpander"],
    .stAlert {
        border: 1px solid #1f1f1f !important;
        border-radius: 8px !important;
        background: #0f0f0f !important;
    }
    [data-testid="stExpander"]:hover {
        border-color: #c9a22740 !important;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: #0f0f0f;
        border: 1px solid #1f1f1f;
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        color: #c9a227 !important;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'JetBrains Mono', monospace !important;
        color: #888 !important;
        text-transform: uppercase;
        font-size: 0.7rem !important;
        letter-spacing: 0.08em;
    }

    /* Buttons */
    .stButton > button {
        font-family: 'JetBrains Mono', monospace !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-size: 0.8rem;
        border-radius: 4px;
        transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"] {
        background: #c9a227 !important;
        color: #0a0a0a !important;
        border: none !important;
        font-weight: 700;
    }
    .stButton > button[kind="primary"]:hover {
        background: #d4af37 !important;
        box-shadow: 0 0 20px #c9a22740;
    }
    .stButton > button[kind="secondary"] {
        background: transparent !important;
        border: 1px solid #333 !important;
        color: #888 !important;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: #c9a227 !important;
        color: #c9a227 !important;
    }

    /* Dividers */
    hr, [data-testid="stDivider"] {
        border-color: #1a1a1a !important;
    }

    /* Radio buttons and selects */
    .stRadio > label, .stSelectbox > label {
        color: #888 !important;
        text-transform: uppercase;
        font-size: 0.75rem !important;
        letter-spacing: 0.08em;
    }

    /* Success/Info/Warning boxes */
    .stSuccess {
        background-color: #0f1a0f !important;
        border-left: 4px solid #2d5a2d !important;
    }
    .stInfo {
        background-color: #0f0f1a !important;
        border-left: 4px solid #c9a227 !important;
    }
    .stWarning {
        background-color: #1a1500 !important;
        border-left: 4px solid #c9a227 !important;
    }

    /* Captions */
    .stCaption, [data-testid="stCaption"] {
        color: #555 !important;
        font-size: 0.75rem !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #0a0a0a;
    }
    ::-webkit-scrollbar-thumb {
        background: #333;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #c9a227;
    }

    /* Gold accent for links */
    a { color: #c9a227 !important; }
    a:hover { color: #d4af37 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

render()
