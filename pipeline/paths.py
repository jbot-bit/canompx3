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

# Database — resolution order:
#   1. DUCKDB_PATH env var (explicit shell override ONLY — never set in .env)
#   2. gold.db in project root — canonical production DB
#
# SCRATCH DB WORKFLOW (for long-running writes):
#   cp gold.db C:/db/gold.db
#   export DUCKDB_PATH=C:/db/gold.db   # shell only, never .env
#   python trading_app/outcome_builder.py ...
#   cp C:/db/gold.db gold.db && unset DUCKDB_PATH && rm C:/db/gold.db
#
# If DUCKDB_PATH points to a non-existent file → warns and falls back to project root.
# .env must NOT contain DUCKDB_PATH — it is for API keys only.
import os as _os
import sys as _sys

def _resolve_db_path() -> Path:
    if "DUCKDB_PATH" in _os.environ:
        candidate = Path(_os.environ["DUCKDB_PATH"])
        if candidate.exists():
            print(f"[DB] Using scratch override: {candidate}", file=_sys.stderr)
            return candidate
        print(
            f"[DB] WARNING: DUCKDB_PATH={candidate} does not exist — scratch was deleted? "
            f"Falling back to project gold.db",
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
