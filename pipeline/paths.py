"""
Canonical paths for the MGC data pipeline.

All path constants are defined here to ensure consistency across the codebase.
"""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Database
GOLD_DB_PATH = PROJECT_ROOT / "gold.db"

# Data directories
DBN_DIR = PROJECT_ROOT / "dbn"
OHLCV_DIR = PROJECT_ROOT / "OHLCV_MGC_FULL"

# Default DBN file
DEFAULT_DBN_FILE = OHLCV_DIR / "glbx-mdp3-20100912-20260203.ohlcv-1m.dbn.zst"
