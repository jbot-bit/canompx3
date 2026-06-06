"""Content-hash cache for drift checks (proof-of-honesty, narrow first cut).

A drift check that reads a known, enumerable set of input files can cache its
PASS verdict keyed on ``sha256(label + content-hash of every declared dep)``.
When any declared input changes, the key changes, the cache misses, and the
check runs for real. Honesty is preserved *by construction*: the key is a pure
function of the check's actual inputs.

Hard invariants (each enforced by tests in test_drift_cache.py):

1. Fail-closed: ANY error reading/parsing the cache, or hashing a dep, returns
   a MISS so the caller runs the real check. There is no path where an error
   yields a blind PASS.
2. Only PASS (empty-violation) verdicts are ever written. A FAIL is never
   cached, so a subsequent fix always re-verifies from scratch.
3. Opt-in only: this module caches nothing on its own. The caller decides which
   labels have a declared dep set; absent a dep set, nothing is cached.

This is deliberately minimal — ONE check is wired in check_drift.py first. The
mechanism is proven before any speed-harvest expansion to the slow checks.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Cache version — bump to invalidate every cached entry on a key-format change.
_CACHE_VERSION = "1"


def _cache_dir() -> Path | None:
    """Resolve the cache directory under .git/ (never tracked by git).

    Returns None if no usable location can be resolved — callers treat None as
    "caching disabled" and always run the real check (fail-closed).
    """
    git = PROJECT_ROOT / ".git"
    try:
        if git.is_dir():
            d = git / ".drift-cache"
            d.mkdir(parents=True, exist_ok=True)
            return d
        common = _git_common_dir()
        if common is not None:
            d = common / ".drift-cache"
            d.mkdir(parents=True, exist_ok=True)
            return d
    except OSError:
        return None
    return None


def _git_common_dir() -> Path | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    common = Path(r.stdout.strip())
    if not common.is_absolute():
        common = PROJECT_ROOT / common
    try:
        return common.resolve()
    except OSError:
        return None


def _hash_dep(path: Path) -> str:
    """Content hash of a single dependency file. Raises on any read failure so
    the caller's try/except converts it to a MISS (fail-closed)."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def cache_key(label: str, dep_paths: list[str]) -> str | None:
    """Compute the content-hash cache key for ``label`` over ``dep_paths``.

    Returns None (→ MISS) if any dep is missing or unreadable — a check whose
    inputs we cannot fully hash must always run for real.
    """
    parts = [_CACHE_VERSION, label]
    try:
        for rel in sorted(dep_paths):
            p = (PROJECT_ROOT / rel).resolve()
            parts.append(f"{rel}:{_hash_dep(p)}")
    except OSError:
        return None
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return digest


def read_pass(label: str, key: str | None) -> bool:
    """Return True iff a cached PASS exists for (label, key).

    Fail-closed: a None key, missing file, unreadable file, or any parse error
    returns False so the caller runs the real check.
    """
    if key is None:
        return False
    d = _cache_dir()
    if d is None:
        return False
    f = d / f"{_safe_name(label)}.json"
    try:
        payload = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    # A stale or mismatched key is a miss. Only an exact key match with the
    # PASS marker is a hit.
    return payload.get("key") == key and payload.get("verdict") == "PASS"


def write_pass(label: str, key: str | None, violations: list[str]) -> None:
    """Persist a PASS for (label, key). No-op unless ``violations`` is empty and
    ``key`` is non-None. FAIL verdicts are never written.

    Any write error is swallowed: a failure to cache must never break the check
    run, and a missing cache entry simply means the next run does the real work.
    """
    if key is None or violations:
        return
    d = _cache_dir()
    if d is None:
        return
    f = d / f"{_safe_name(label)}.json"
    try:
        f.write_text(json.dumps({"key": key, "verdict": "PASS"}), encoding="utf-8")
    except OSError:
        return


def _safe_name(label: str) -> str:
    """Stable filename for a check label (the key itself guarantees correctness;
    this only needs to be a collision-resistant, filesystem-safe name)."""
    return hashlib.sha256(label.encode("utf-8")).hexdigest()[:32]
