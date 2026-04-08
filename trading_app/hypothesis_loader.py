"""
Hypothesis registry loader (READ-SIDE) — Phase 4 Stage 4.0.

Provides read-only access to pre-registered hypothesis files in
``docs/audit/hypotheses/`` for the strategy_validator's institutional gates
(criteria 1, 2, 4, 5 of the locked criteria in
``docs/institutional/pre_registered_criteria.md``).

This module is the READ side. It loads files, computes their content SHA, and
extracts metadata fields. The WRITE side (discovery integration that stamps
the SHA on every experimental_strategies row and enforces the file as a
pre-facto enumeration constraint) is added in Phase 4 Stage 4.1 on top of
this same loader.

Authority chain
---------------

- Registry rules: ``docs/audit/hypotheses/README.md``
- Schema template: ``docs/institutional/hypothesis_registry_template.md``
- Locked criteria: ``docs/institutional/pre_registered_criteria.md``

Single-use rule
---------------

A pre-registered hypothesis file is single-use. Re-running the same file
silently doubles the multiple-testing family. The validator and discovery
both check ``experimental_strategies`` for prior usage of a SHA before
allowing it; that check lives at the call sites, not in this module.

Bias control
------------

This module is intentionally pure-IO. It does NOT query
``validated_setups``, ``edge_families``, ``live_config``, or any deployment
artifact. It reads only the hypothesis YAML files. This separation keeps
the hypothesis-file authoring and lookup paths uncontaminated by prior
deployment knowledge.

@research-source: docs/audit/hypotheses/README.md
@research-source: docs/institutional/pre_registered_criteria.md
@canonical-source: trading_app/hypothesis_loader.py
@revalidated-for: Phase 4 Stage 4.0 (2026-04-08)
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path
from typing import Any

import yaml

# The canonical hypothesis registry directory, anchored to the repo root.
# Discovered at module load time so the validator can locate files without
# the caller having to pass paths.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_HYPOTHESIS_DIR = _REPO_ROOT / "docs" / "audit" / "hypotheses"


class HypothesisLoaderError(Exception):
    """Raised when a hypothesis file cannot be loaded or fails schema validation."""


def hypothesis_dir() -> Path:
    """Return the canonical hypothesis registry directory.

    Used by tests and the discovery write-side integration. Read-only —
    never write to this path from this module.
    """
    return _HYPOTHESIS_DIR


def compute_file_sha(path: Path) -> str:
    """Compute the deterministic content SHA of a hypothesis file.

    Uses sha256 of the raw file bytes. The SHA is the lock identifier for
    the pre-registration: it changes if anyone edits the file after commit,
    making post-hoc tampering detectable.

    Parameters
    ----------
    path
        Filesystem path to a hypothesis YAML file.

    Returns
    -------
    str
        The hex digest of the file's sha256 hash.

    Raises
    ------
    HypothesisLoaderError
        If the file does not exist or cannot be read.
    """
    if not path.is_file():
        raise HypothesisLoaderError(
            f"Hypothesis file not found: {path}. "
            f"See docs/audit/hypotheses/README.md for the registry workflow."
        )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_hypothesis_file_by_sha(sha: str, search_dir: Path | None = None) -> Path | None:
    """Find a hypothesis YAML file in the registry whose content SHA matches.

    The validator stores a SHA in ``experimental_strategies.hypothesis_file_sha``
    rather than a path because the SHA is content-derived and tamper-evident.
    This function reverses the lookup: scan the registry directory, compute
    each file's SHA, return the matching path or None.

    Parameters
    ----------
    sha
        Hex digest sha256 to search for. Case-insensitive comparison.
    search_dir
        Override directory for testing. Defaults to the canonical
        ``docs/audit/hypotheses/`` location.

    Returns
    -------
    Path | None
        The matching file path, or None if no file in the directory matches.
        Returns None if the directory does not exist (legitimate before any
        hypothesis has been committed).

    Notes
    -----
    Linear scan over the directory. The registry is expected to contain at
    most a few dozen files over the project's lifetime. Premature optimisation
    rejected.
    """
    target = sha.lower()
    directory = search_dir if search_dir is not None else _HYPOTHESIS_DIR
    if not directory.is_dir():
        return None
    for entry in sorted(directory.glob("*.yaml")):
        try:
            entry_sha = compute_file_sha(entry).lower()
        except HypothesisLoaderError:
            continue
        if entry_sha == target:
            return entry
    return None


# Required top-level keys per the template schema. Any missing key raises
# HypothesisLoaderError. Additional optional keys are silently allowed.
_REQUIRED_METADATA_KEYS = {
    "name",
    "date_locked",
    "holdout_date",
    "total_expected_trials",
}

_REQUIRED_TOP_LEVEL_KEYS = {
    "metadata",
    "hypotheses",
}


def load_hypothesis_metadata(path: Path) -> dict[str, Any]:
    """Parse a hypothesis YAML file and validate its schema, returning metadata.

    Parameters
    ----------
    path
        Filesystem path to a hypothesis YAML file.

    Returns
    -------
    dict[str, Any]
        A dictionary with at least these keys (others passed through):
        - ``sha``: the file content SHA (hex digest)
        - ``path``: the absolute path
        - ``name``: short slug from metadata.name
        - ``date_locked``: ISO date or datetime
        - ``holdout_date``: a ``datetime.date`` object
        - ``total_expected_trials``: int (the MinBTL trial budget)
        - ``has_theory``: bool — True iff at least one hypothesis cites a
          ``theory_citation`` field (used by the Chordia gate to pick the
          3.00 vs 3.79 threshold)
        - ``hypotheses``: the raw hypotheses list
        - ``metadata``: the raw metadata dict

    Raises
    ------
    HypothesisLoaderError
        On any of: file missing, YAML parse error, missing required top-level
        key, missing required metadata key, total_expected_trials not an int,
        holdout_date not parseable as a date.
    """
    if not path.is_file():
        raise HypothesisLoaderError(
            f"Hypothesis file not found: {path}. "
            f"See docs/audit/hypotheses/README.md for the registry workflow."
        )
    try:
        raw_text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise HypothesisLoaderError(f"Hypothesis file is not valid YAML: {path}: {exc}") from exc
    except OSError as exc:
        raise HypothesisLoaderError(f"Cannot read hypothesis file {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise HypothesisLoaderError(
            f"Hypothesis file {path} top-level must be a mapping, got {type(data).__name__}."
        )

    missing_top = _REQUIRED_TOP_LEVEL_KEYS - data.keys()
    if missing_top:
        raise HypothesisLoaderError(
            f"Hypothesis file {path} missing required top-level keys: {sorted(missing_top)}. "
            f"See docs/institutional/hypothesis_registry_template.md for the schema."
        )

    metadata = data["metadata"]
    if not isinstance(metadata, dict):
        raise HypothesisLoaderError(f"Hypothesis file {path} 'metadata' must be a mapping.")

    missing_meta = _REQUIRED_METADATA_KEYS - metadata.keys()
    if missing_meta:
        raise HypothesisLoaderError(
            f"Hypothesis file {path} missing required metadata keys: {sorted(missing_meta)}. "
            f"See docs/institutional/hypothesis_registry_template.md for the schema."
        )

    declared_n = metadata["total_expected_trials"]
    if not isinstance(declared_n, int) or declared_n < 1:
        raise HypothesisLoaderError(
            f"Hypothesis file {path} metadata.total_expected_trials must be a positive int, got {declared_n!r}."
        )

    holdout_raw = metadata["holdout_date"]
    holdout_parsed = _coerce_to_date(holdout_raw)
    if holdout_parsed is None:
        raise HypothesisLoaderError(
            f"Hypothesis file {path} metadata.holdout_date must be a YYYY-MM-DD string or date, got {holdout_raw!r}."
        )

    hypotheses = data["hypotheses"]
    if not isinstance(hypotheses, list) or len(hypotheses) == 0:
        raise HypothesisLoaderError(
            f"Hypothesis file {path} 'hypotheses' must be a non-empty list."
        )

    # has_theory is True iff at least one hypothesis cites a theory_citation
    # field. Used by the Chordia gate to pick the 3.00 (with theory) vs 3.79
    # (without theory) threshold per Criterion 4.
    has_theory = False
    for h in hypotheses:
        if isinstance(h, dict) and h.get("theory_citation"):
            has_theory = True
            break

    return {
        "sha": compute_file_sha(path),
        "path": str(path),
        "name": metadata["name"],
        "date_locked": metadata["date_locked"],
        "holdout_date": holdout_parsed,
        "total_expected_trials": declared_n,
        "has_theory": has_theory,
        "hypotheses": hypotheses,
        "metadata": metadata,
    }


def load_hypothesis_by_sha(sha: str, search_dir: Path | None = None) -> dict[str, Any] | None:
    """Convenience: find a file by SHA and return its parsed metadata.

    Parameters
    ----------
    sha
        Hex digest sha256 of the file content.
    search_dir
        Override directory for testing.

    Returns
    -------
    dict[str, Any] | None
        Parsed metadata dict (per ``load_hypothesis_metadata``) or None if
        no file with the matching SHA exists in the search directory.
    """
    path = find_hypothesis_file_by_sha(sha, search_dir=search_dir)
    if path is None:
        return None
    return load_hypothesis_metadata(path)


def _coerce_to_date(value: Any) -> date | None:
    """Coerce a YAML date-ish value to a ``datetime.date`` or None.

    YAML's safe_load may return either a ``datetime.date`` (if the source is
    bare ``2026-01-01``) or a ``str`` (if the source is quoted ``"2026-01-01"``).
    Both are accepted; everything else returns None.
    """
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


__all__ = [
    "HypothesisLoaderError",
    "compute_file_sha",
    "find_hypothesis_file_by_sha",
    "hypothesis_dir",
    "load_hypothesis_by_sha",
    "load_hypothesis_metadata",
]
