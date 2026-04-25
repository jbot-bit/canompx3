"""
Canonical paths for the MGC data pipeline.

All path constants are defined here to ensure consistency across the codebase.
"""

import subprocess as _subprocess
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
#   DUCKDB_PATH env var  = override for genuine test databases only
#
# C:/db/gold.db scratch copy is DEPRECATED (Mar 24 2026). It caused stale-data
# bugs: terminals reading 88-min-old data, reporting strategies as alive that
# were killed by fresh FDR. DUCKDB_PATH pointing to C:/db/gold.db is blocked.
# If DUCKDB_PATH points to a non-existent file → warns and falls back to project root.
import os as _os
import sys as _sys


def _discover_git_common_root(project_root: Path) -> Path | None:
    """Return the shared git root when running inside a linked worktree.

    `Path(__file__).parent.parent` points at the current checkout. In a linked
    worktree that checkout intentionally does not carry the canonical `gold.db`;
    the shared DB lives in the common repo root next to `.git/`.
    """
    try:
        result = _subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
    except Exception:
        return None

    stdout = result.stdout.strip()
    if not stdout:
        return None

    common_dir = Path(stdout)
    if not common_dir.is_absolute():
        common_dir = (project_root / common_dir).resolve()
    if common_dir.name != ".git":
        return None

    common_root = common_dir.parent
    if not (common_root / "pipeline").exists():
        return None
    return common_root


def _default_canonical_db(project_root: Path) -> Path:
    """Return the best canonical DB candidate for this checkout."""
    local_db = project_root / "gold.db"
    if local_db.exists():
        return local_db

    common_root = _discover_git_common_root(project_root)
    if common_root is not None:
        common_db = common_root / "gold.db"
        if common_db.exists():
            if common_root != project_root:
                print(
                    f"[DB] INFO: local worktree has no gold.db; using shared canonical DB at {common_db}",
                    file=_sys.stderr,
                )
            return common_db

    return local_db


def _resolve_db_path(project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve the canonical DB path. ALWAYS returns project root gold.db.

    The scratch DB at C:/db/gold.db is DEPRECATED — it caused stale-data bugs
    across multiple sessions (Mar 24 2026: decisions made on 88-min-stale data,
    strategies reported as alive that were actually killed by fresh FDR).
    Any DUCKDB_PATH pointing to the scratch copy is rejected with a loud warning.
    """
    _SCRATCH = Path("C:/db/gold.db")
    default_db = _default_canonical_db(project_root)
    if "DUCKDB_PATH" in _os.environ:
        candidate = Path(_os.environ["DUCKDB_PATH"]).expanduser()
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
    return default_db


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
