"""State persistence — JSON file for cooling/session state across restarts.

Atomic write (temp file + rename) prevents corruption on power loss.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_STATE_PATH = Path(__file__).parent.parent / "data" / "session_state.json"


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_state(path: Path | None = None) -> dict[str, Any]:
    """Load persisted state from JSON file. Returns empty dict if missing/corrupt."""
    p = path or DEFAULT_STATE_PATH
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Failed to load state from %s: %s", p, exc)
        return {}


def save_state(state: dict[str, Any], path: Path | None = None) -> bool:
    """Atomically write state to JSON file. Returns True on success."""
    p = path or DEFAULT_STATE_PATH
    _ensure_dir(p)
    try:
        # Write to temp file first, then os.replace (atomic on both POSIX and Windows)
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp", prefix="state_")
        try:
            to_write = {**state, "_saved_at": datetime.now(UTC).isoformat()}
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(to_write, fh, indent=2, default=str)
            os.replace(tmp, str(p))
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise
        return True
    except OSError as exc:
        log.error("Failed to save state to %s: %s", p, exc)
        return False


def load_cooling_state(path: Path | None = None) -> dict[str, Any]:
    """Load just the cooling sub-state."""
    state = load_state(path)
    return state.get("cooling", {})


def save_cooling_state(cooling: dict[str, Any], path: Path | None = None) -> bool:
    """Merge cooling state into the persisted state file."""
    state = load_state(path)
    state["cooling"] = cooling
    return save_state(state, path)


def load_commitment_state(path: Path | None = None) -> dict[str, Any]:
    """Load the commitment checklist sub-state."""
    state = load_state(path)
    return state.get("commitment", {"items": {}, "date": None})


def save_commitment_state(commitment: dict[str, Any], path: Path | None = None) -> bool:
    """Merge commitment state into the persisted state file."""
    state = load_state(path)
    state["commitment"] = commitment
    return save_state(state, path)
