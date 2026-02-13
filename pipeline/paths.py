"""
Canonical paths for the MGC data pipeline.

All path constants are defined here to ensure consistency across the codebase.
"""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Database — resolution order:
#   1. DUCKDB_PATH env var (explicit override)
#   2. local_db/gold.db (NTFS junction to C:\db — fast local disk, not OneDrive-synced)
#   3. gold.db (project root fallback — OneDrive path, slow for writes)
import os as _os

def _resolve_db_path() -> Path:
    if "DUCKDB_PATH" in _os.environ:
        return Path(_os.environ["DUCKDB_PATH"])
    local = PROJECT_ROOT / "local_db" / "gold.db"
    if local.exists():
        return local
    return PROJECT_ROOT / "gold.db"

GOLD_DB_PATH = _resolve_db_path()

# Data directories
DBN_DIR = PROJECT_ROOT / "dbn"
OHLCV_DIR = PROJECT_ROOT / "OHLCV_MGC_FULL"

# Daily DBN files directory (1,559 individual daily .dbn.zst files)
DAILY_DBN_DIR = PROJECT_ROOT / "DB" / "GOLD_DB_FULLSIZE"

# Default DBN file (single concatenated file — may not exist if using daily files)
DEFAULT_DBN_FILE = OHLCV_DIR / "glbx-mdp3-20100912-20260203.ohlcv-1m.dbn.zst"
