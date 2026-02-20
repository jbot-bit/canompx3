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
#   1. DUCKDB_PATH env var (explicit override, or from .env)
#   2. local_db/gold.db (NTFS junction to C:\db — alternative local path)
#   3. gold.db (project root fallback)
import os as _os

def _resolve_db_path() -> Path:
    if "DUCKDB_PATH" in _os.environ:
        return Path(_os.environ["DUCKDB_PATH"])
    local = PROJECT_ROOT / "local_db" / "gold.db"
    try:
        if local.exists():
            return local
    except OSError:
        pass  # WinError 448: untrusted mount point (sandbox/CI)
    return PROJECT_ROOT / "gold.db"

GOLD_DB_PATH = _resolve_db_path()

if "DUCKDB_PATH" in _os.environ:
    import sys as _sys
    print(f"[DB] Using override: {_os.environ['DUCKDB_PATH']}", file=_sys.stderr)

# Data directories
DBN_DIR = PROJECT_ROOT / "dbn"
OHLCV_DIR = PROJECT_ROOT / "OHLCV_MGC_FULL"

# Daily DBN files directory (1,559 individual daily .dbn.zst files)
DAILY_DBN_DIR = PROJECT_ROOT / "DB" / "GOLD_DB_FULLSIZE"

# Default DBN file (single concatenated file — may not exist if using daily files)
DEFAULT_DBN_FILE = OHLCV_DIR / "glbx-mdp3-20100912-20260203.ohlcv-1m.dbn.zst"
