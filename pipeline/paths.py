"""
Canonical paths for the MGC data pipeline.

All path constants are defined here to ensure consistency across the codebase.
"""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Load .env from project root (populates DUCKDB_PATH and API keys into os.environ)
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass  # python-dotenv not installed — rely on shell env

# Database — single canonical DB:
#   <project>/gold.db    = canonical DB used by all code (pipeline, research, live trading)
#   C:/db/gold.db        = auto-synced scratch copy (drift check #37 copies canonical when stale)
#   DUCKDB_PATH env var  = override (set in .env)
#
# All scripts default to GOLD_DB_PATH. The scratch copy exists for isolation during
# long-running writes but is never the default.
# If DUCKDB_PATH points to a non-existent file → warns and falls back to project root.
import os as _os
import sys as _sys

def _resolve_db_path() -> Path:
    if "DUCKDB_PATH" in _os.environ:
        candidate = Path(_os.environ["DUCKDB_PATH"])
        if candidate.exists():
            return candidate
        print(
            f"[DB] WARNING: DUCKDB_PATH={candidate} does not exist — "
            f"falling back to project gold.db",
            file=_sys.stderr,
        )
    return PROJECT_ROOT / "gold.db"

GOLD_DB_PATH = _resolve_db_path()

# Data directories
DBN_DIR = PROJECT_ROOT / "dbn"
OHLCV_DIR = PROJECT_ROOT / "OHLCV_MGC_FULL"

# Daily DBN files directory (1,559 individual daily .dbn.zst files)
DAILY_DBN_DIR = PROJECT_ROOT / "DB" / "GOLD_DB_FULLSIZE"

# Default DBN file (single concatenated file — may not exist if using daily files)
DEFAULT_DBN_FILE = OHLCV_DIR / "glbx-mdp3-20100912-20260203.ohlcv-1m.dbn.zst"
