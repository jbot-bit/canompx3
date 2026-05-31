#!/usr/bin/env python3
"""Read-only worktree launch classifier for START_WORKTREE.bat.

Hooks cannot auto-ENTER a worktree (they fire inside an already-running
session). This classifier is the brain behind a thin launcher .bat: given a
descriptor, it computes the canonical worktree path/branch (mirroring
`scripts/tools/new_session.sh`) and decides whether launching there is safe:

  NEW          worktree does not exist yet      -> launcher will `git worktree add`
  REUSE_CLEAN  exists, clean, no live peer       -> launcher will launch into it
  REFUSE_HOT   exists AND (dirty OR a live peer)  -> launcher refuses (exit 3)

Lease/peer semantics are NOT re-implemented here (institutional-rigor §4). They
delegate to the canonical module `scripts/tools/worktree_guard.py --status
--json`, reading `lease_present` + `peer_live`. Dirty-check is the independent
second guard, so a lease subprocess failure fails open to not-hot WITHOUT losing
protection (dirty still refuses).

Classification is never an error: `main()` always returns 0. The launcher reads
the decision from stdout (`--json`) or the printed token.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

NEW = "NEW"
REUSE_CLEAN = "REUSE_CLEAN"
REFUSE_HOT = "REFUSE_HOT"

_GIT_TIMEOUT = 5
_GUARD_TIMEOUT = 8


def _repo_root() -> Path:
    """Top-level of the repo containing this script (the canonical main checkout)."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            check=False,
        )
        if r.returncode == 0 and r.stdout.strip():
            return Path(r.stdout.strip())
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass
    # Fallback: scripts/tools/<file> -> parents[2] == repo root.
    return Path(__file__).resolve().parents[2]


def compute_wt_path(descriptor: str, repo_root: Path | None = None) -> tuple[Path, str]:
    """Return (worktree_path, branch) for a descriptor — mirrors new_session.sh.

    worktree = <repo_parent>/<repo_name>-<descriptor>
    branch   = session/<user>-<descriptor>
    """
    root = repo_root if repo_root is not None else _repo_root()
    wt_path = root.parent / f"{root.name}-{descriptor}"
    user = _username()
    branch = f"session/{user}-{descriptor}"
    return wt_path, branch


def _username() -> str:
    import getpass

    try:
        return getpass.getuser().strip() or "user"
    except (OSError, KeyError):
        return "user"


def _is_dirty(wt: Path) -> bool:
    """True if the worktree has tracked-file changes (untracked ignored, like
    new_session.sh's collision concern). Fail-open to NOT dirty on git error."""
    try:
        r = subprocess.run(
            ["git", "-C", str(wt), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            check=False,
        )
        if r.returncode != 0:
            return False
        return bool(r.stdout.strip())
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return False


def _lease_hot(wt: Path) -> bool:
    """True iff a LIVE peer session holds the worktree lease.

    Delegates entirely to canonical `scripts/tools/worktree_guard.py --status
    --json`. hot == lease_present AND peer_live. Any subprocess/parse failure
    -> NOT hot (fail-open); the dirty-check remains an independent guard.
    """
    guard = _repo_root() / "scripts" / "tools" / "worktree_guard.py"
    if not guard.exists():
        return False
    try:
        r = subprocess.run(
            [sys.executable, str(guard), "--status", "--json", "--cwd", str(wt)],
            capture_output=True,
            text=True,
            timeout=_GUARD_TIMEOUT,
            check=False,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return False
        snap = json.loads(r.stdout)
    except (subprocess.SubprocessError, FileNotFoundError, OSError, json.JSONDecodeError, ValueError):
        return False
    return bool(snap.get("lease_present") and snap.get("peer_live"))


def classify(wt: Path) -> str:
    """NEW / REUSE_CLEAN / REFUSE_HOT for a candidate worktree path."""
    if not wt.exists():
        return NEW
    if _is_dirty(wt) or _lease_hot(wt):
        return REFUSE_HOT
    return REUSE_CLEAN


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="worktree_launch_preflight",
        description="Classify whether launching Claude into a worktree is safe.",
    )
    parser.add_argument("--descriptor", required=True, help="Short branch/dir descriptor")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args(argv)

    wt_path, branch = compute_wt_path(args.descriptor)
    decision = classify(wt_path)

    if args.json:
        print(
            json.dumps(
                {
                    "descriptor": args.descriptor,
                    "worktree_path": str(wt_path),
                    "branch": branch,
                    "decision": decision,
                }
            )
        )
    else:
        # Stable, parseable lines for the .bat (DECISION first token of line 1).
        print(f"DECISION={decision}")
        print(f"WTPATH={wt_path}")
        print(f"BRANCH={branch}")
    return 0  # classification is never an error


if __name__ == "__main__":
    sys.exit(main())
