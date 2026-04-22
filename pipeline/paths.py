"""
Canonical paths for the MGC data pipeline.

All path constants are defined here to ensure consistency across the codebase.
"""

from pathlib import Path
import subprocess

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
#   DUCKDB_PATH env var  = override for genuine test databases only
#
# C:/db/gold.db scratch copy is DEPRECATED (Mar 24 2026). It caused stale-data
# bugs: terminals reading 88-min-old data, reporting strategies as alive that
# were killed by fresh FDR. DUCKDB_PATH pointing to C:/db/gold.db is blocked.
# If DUCKDB_PATH points to a non-existent file → warns and falls back to project root.
import os as _os
import sys as _sys


def _canonical_repo_root() -> Path:
    """Resolve the shared repo root when running from a git worktree.

    In the main checkout, this is just PROJECT_ROOT. In detached worktrees,
    ``PROJECT_ROOT / "gold.db"`` often does not exist even though the canonical
    database lives at the shared repository root. We resolve the git common dir
    and step back to the shared root when possible.
    """
    try:
        common_dir = subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return PROJECT_ROOT

    common_path = Path(common_dir)
    if not common_path.is_absolute():
        common_path = (PROJECT_ROOT / common_path).resolve()
    if common_path.name == ".git":
        return common_path.parent
    if common_path.parent.name == "worktrees":
        return common_path.parent.parent.parent
    return PROJECT_ROOT


def _resolve_db_path() -> Path:
    """Resolve the canonical DB path.

    The scratch DB at C:/db/gold.db is DEPRECATED — it caused stale-data bugs
    across multiple sessions (Mar 24 2026: decisions made on 88-min-stale data,
    strategies reported as alive that were actually killed by fresh FDR).
    Any DUCKDB_PATH pointing to the scratch copy is rejected with a loud warning.
    """
    _SCRATCH = Path("C:/db/gold.db")
    if "DUCKDB_PATH" in _os.environ:
        candidate = Path(_os.environ["DUCKDB_PATH"])
        if candidate.resolve() == _SCRATCH.resolve():
            print(
                "[DB] BLOCKED: DUCKDB_PATH points to deprecated scratch DB C:/db/gold.db. "
                "Using canonical project gold.db instead. Remove DUCKDB_PATH or point it "
                "to a real override.",
                file=_sys.stderr,
            )
        elif candidate.exists():
            return candidate
        else:
            print(
                f"[DB] WARNING: DUCKDB_PATH={candidate} does not exist — falling back to project gold.db",
                file=_sys.stderr,
            )
    canonical = _canonical_repo_root() / "gold.db"
    if canonical.exists():
        return canonical
    return PROJECT_ROOT / "gold.db"


GOLD_DB_PATH = _resolve_db_path()

# Data directories
DBN_DIR = PROJECT_ROOT / "dbn"
OHLCV_DIR = PROJECT_ROOT / "OHLCV_MGC_FULL"

# Daily DBN files directory (1,559 individual daily .dbn.zst files)
DAILY_DBN_DIR = PROJECT_ROOT / "DB" / "GOLD_DB_FULLSIZE"

# Default DBN file (single concatenated file — may not exist if using daily files)
DEFAULT_DBN_FILE = OHLCV_DIR / "glbx-mdp3-20100912-20260203.ohlcv-1m.dbn.zst"

# Trace logs directory (structured JSON audit/research traces)
TRACES_DIR = PROJECT_ROOT / "logs" / "traces"

# Live trading journal — separate DB to avoid write contention with gold.db
LIVE_JOURNAL_DB_PATH = PROJECT_ROOT / "live_journal.db"
