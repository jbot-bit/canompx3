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
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

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


# -----------------------------------------------------------------------------
# Phase 4 Stage 4.1 additions — canonical MinBTL, ScopePredicate, Mode A check
# -----------------------------------------------------------------------------
#
# These functions are the write-side complements to the read-side loader
# primitives above. They are ALL pure functions of a metadata dict — no DB,
# no subprocess, no deployment-artifact reads. The loader's bias-control
# invariant (see module docstring § "Bias control") is preserved.
#
# Write-side gates that DO require side effects (git cleanliness, single-use
# check against experimental_strategies) live in
# ``trading_app.phase_4_discovery_gates``, not here.
#
# Canonical authority: docs/institutional/pre_registered_criteria.md
# § Criterion 2 (MinBTL), Amendment 2.7 (Mode A sacred boundary).

# MinBTL bounds — LOCKED per Criterion 2. Changing these requires a new
# amendment to pre_registered_criteria.md. Bailey et al 2013 Theorem 1:
# MinBTL = 2·Ln[N] / E[max_N]².
_MINBTL_CLEAN_BOUND: int = 300
_MINBTL_PROXY_BOUND: int = 2000


def enforce_minbtl_bound(
    meta: dict[str, Any],
    on_proxy_data: bool = False,
) -> tuple[str | None, str | None]:
    """Enforce Criterion 2 (MinBTL) on a hypothesis file's declared trial count.

    This is the CANONICAL implementation. Two call sites:

    - ``trading_app.strategy_validator._check_criterion_2_minbtl`` — delegates
      here so the validator and discovery paths share the same bounds. Stage
      4.0 originally inlined ``bound = 2000 if on_proxy_data else 300``; Stage
      4.1 moves that logic here to eliminate parallel implementations.
    - ``trading_app.strategy_discovery.main`` — called at CLI parse time
      before enumeration begins so a mis-sized hypothesis file fails loud.

    Parameters
    ----------
    meta
        Hypothesis file metadata dict, typically produced by
        ``load_hypothesis_metadata``. Must contain ``total_expected_trials``
        as a positive int. ``data_source_mode`` and ``data_source_disclosure``
        are consulted only when ``on_proxy_data`` is True.
    on_proxy_data
        When True, the proxy-extended bound (``_MINBTL_PROXY_BOUND``) applies.
        The caller is responsible for deciding this based on the scope of the
        hypothesis file (e.g., does it include pre-micro-launch trading days?).
        When True, this function additionally requires
        ``metadata.data_source_mode == "proxy"`` and a non-empty
        ``metadata.data_source_disclosure`` string, per Criterion 2's
        "explicit data-source disclosure" clause.

    Returns
    -------
    tuple[str | None, str | None]
        ``(None, None)`` on pass. ``("REJECTED", reason_str)`` on fail.
        Tuple shape matches the Stage 4.0 pre-flight gate convention in
        ``strategy_validator.py`` so the validator can delegate with zero
        call-site churn.

    Raises
    ------
    HypothesisLoaderError
        If ``total_expected_trials`` is missing or malformed. This indicates
        a broken hypothesis file — loader-level integrity failure, not a
        soft Criterion 2 rejection.
    """
    declared = meta.get("total_expected_trials")
    if not isinstance(declared, int) or declared < 1:
        raise HypothesisLoaderError(
            f"total_expected_trials must be a positive int, got {declared!r}. "
            f"Hypothesis file is malformed — cannot evaluate Criterion 2."
        )

    if on_proxy_data:
        # Proxy mode is an opt-in that requires explicit disclosure per the
        # locked text of Criterion 2: "N <= 2,000 pre-registered trials on
        # proxy-extended data with explicit data-source disclosure".
        source_meta = meta.get("metadata", {}) if isinstance(meta.get("metadata"), dict) else {}
        mode = source_meta.get("data_source_mode")
        disclosure = source_meta.get("data_source_disclosure")
        if mode != "proxy":
            return (
                "REJECTED",
                "criterion_2: proxy-data bound (2000) requested but "
                "metadata.data_source_mode != 'proxy' — proxy use requires "
                "explicit opt-in per Criterion 2 locked text",
            )
        if not isinstance(disclosure, str) or not disclosure.strip():
            return (
                "REJECTED",
                "criterion_2: proxy-data bound (2000) requested but "
                "metadata.data_source_disclosure is missing or empty — "
                "Criterion 2 locked text requires explicit data-source disclosure",
            )
        bound = _MINBTL_PROXY_BOUND
        mode_label = "proxy-extended"
    else:
        bound = _MINBTL_CLEAN_BOUND
        mode_label = "clean-MNQ"

    if declared > bound:
        return (
            "REJECTED",
            f"criterion_2: MinBTL bound exceeded — declared {declared} trials "
            f"> {mode_label} bound {bound} (pre_registered_criteria.md "
            f"§ Criterion 2, Bailey et al 2013 Theorem 1)",
        )
    return (None, None)


def check_mode_a_consistency(meta: dict[str, Any]) -> None:
    """Validate a hypothesis file's ``holdout_date`` against Amendment 2.7.

    Under Mode A (Amendment 2.7, 2026-04-08), the sacred holdout window begins
    at ``HOLDOUT_SACRED_FROM``. A hypothesis file declaring ``holdout_date``
    AFTER the sacred-from date would implicitly permit discovery to consume
    sacred data, which is banned.

    Parameters
    ----------
    meta
        Hypothesis file metadata dict from ``load_hypothesis_metadata``. Must
        contain ``holdout_date`` as a ``datetime.date``.

    Raises
    ------
    HypothesisLoaderError
        If ``holdout_date`` is missing, malformed, or strictly greater than
        ``HOLDOUT_SACRED_FROM``. Error message cites Amendment 2.7 and the
        canonical source.
    """
    holdout = meta.get("holdout_date")
    if holdout is None:
        raise HypothesisLoaderError(
            "metadata.holdout_date is required for Mode A consistency check. "
            "See docs/audit/hypotheses/README.md and Amendment 2.7."
        )
    # Normalize datetime → date for comparison (HOLDOUT_SACRED_FROM is a date).
    # YAML safe_load may return either; both are accepted here.
    if isinstance(holdout, datetime):
        holdout_cmp = holdout.date()
    elif isinstance(holdout, date):
        holdout_cmp = holdout
    else:
        raise HypothesisLoaderError(
            f"metadata.holdout_date must be a date or datetime, got {type(holdout).__name__}"
        )
    if holdout_cmp > HOLDOUT_SACRED_FROM:
        raise HypothesisLoaderError(
            f"Amendment 2.7 violation: hypothesis holdout_date "
            f"{holdout_cmp.isoformat()} is after the sacred window boundary "
            f"{HOLDOUT_SACRED_FROM.isoformat()}. Mode A requires holdout_date "
            f"<= sacred-from. Canonical source: trading_app.holdout_policy."
        )


@dataclass(frozen=True)
class HypothesisScope:
    """Immutable scope bundle for a single hypothesis within a registry file.

    Stores the per-hypothesis filter_type + scope dimensions as frozensets of
    primitive types. The dataclass is frozen so instances are hashable and
    safe to share across threads. Bundling is preserved — a
    ``(session, filter_type, em, rr, cb, stop)`` tuple must match one
    HypothesisScope's ALL dimensions simultaneously, not a flat union across
    hypotheses.

    This prevents cross-pollination: if hypothesis 1 declares
    ``OVNRNG + EUROPE_FLOW`` and hypothesis 2 declares ``ORB_G + CME_REOPEN``,
    the combo ``OVNRNG + CME_REOPEN`` is REJECTED (neither hypothesis
    declared it) rather than ACCEPTED (flat union would allow it).
    """

    filter_type: str
    instruments: frozenset[str]
    sessions: frozenset[str]
    rr_targets: frozenset[float]
    entry_models: frozenset[str]
    confirm_bars: frozenset[int]
    stop_multipliers: frozenset[float]
    expected_trial_count: int

    def accepts(
        self,
        *,
        orb_label: str,
        filter_type: str,
        entry_model: str,
        rr_target: float,
        confirm_bars: int,
        stop_multiplier: float,
    ) -> bool:
        """Check whether this hypothesis's scope accepts a specific combo.

        All six dimensions must match simultaneously. Instrument is NOT
        checked here because ScopePredicate.extract_scope_predicate has
        already filtered to hypotheses for a specific instrument.
        """
        return (
            filter_type == self.filter_type
            and orb_label in self.sessions
            and entry_model in self.entry_models
            and rr_target in self.rr_targets
            and confirm_bars in self.confirm_bars
            and stop_multiplier in self.stop_multipliers
        )


@dataclass(frozen=True)
class ScopePredicate:
    """Per-hypothesis scope predicate built from a hypothesis registry file.

    Wraps a tuple of ``HypothesisScope`` bundles, each corresponding to one
    hypothesis in the file (filtered to those declaring the current
    instrument). The ``accepts`` method returns True if any bundle accepts
    the combo — OR across hypotheses, but per-bundle AND across dimensions.

    Immutable and hashable. Construct via ``extract_scope_predicate``.
    """

    hypotheses: tuple[HypothesisScope, ...]
    instrument: str
    total_declared_trials: int

    def accepts(
        self,
        *,
        orb_label: str,
        filter_type: str,
        entry_model: str,
        rr_target: float,
        confirm_bars: int,
        stop_multiplier: float,
    ) -> bool:
        """True iff at least one hypothesis scope bundle accepts the combo."""
        return any(
            h.accepts(
                orb_label=orb_label,
                filter_type=filter_type,
                entry_model=entry_model,
                rr_target=rr_target,
                confirm_bars=confirm_bars,
                stop_multiplier=stop_multiplier,
            )
            for h in self.hypotheses
        )

    def allowed_sessions(self) -> frozenset[str]:
        """Union of sessions across all hypotheses. Early-exit helper."""
        result: set[str] = set()
        for h in self.hypotheses:
            result |= h.sessions
        return frozenset(result)

    def allowed_filter_types(self) -> frozenset[str]:
        """Set of allowed filter_types. Early-exit helper."""
        return frozenset(h.filter_type for h in self.hypotheses)

    def allowed_entry_models(self) -> frozenset[str]:
        """Union of entry_models across all hypotheses. Early-exit helper."""
        result: set[str] = set()
        for h in self.hypotheses:
            result |= h.entry_models
        return frozenset(result)

    def allowed_rr_targets(self) -> frozenset[float]:
        """Union of rr_targets across all hypotheses. Early-exit helper."""
        result: set[float] = set()
        for h in self.hypotheses:
            result |= h.rr_targets
        return frozenset(result)

    def allowed_confirm_bars(self) -> frozenset[int]:
        """Union of confirm_bars across all hypotheses. Early-exit helper."""
        result: set[int] = set()
        for h in self.hypotheses:
            result |= h.confirm_bars
        return frozenset(result)

    def allowed_stop_multipliers(self) -> frozenset[float]:
        """Union of stop_multipliers across all hypotheses. Early-exit helper."""
        result: set[float] = set()
        for h in self.hypotheses:
            result |= h.stop_multipliers
        return frozenset(result)


def extract_scope_predicate(
    meta: dict[str, Any],
    *,
    instrument: str,
) -> ScopePredicate:
    """Build a ScopePredicate from a hypothesis file's metadata for one instrument.

    Iterates ``meta['hypotheses']``, filters to hypotheses whose
    ``scope.instruments`` contains the given instrument, validates each
    hypothesis's scope block shape, and constructs ``HypothesisScope``
    instances from them.

    Parameters
    ----------
    meta
        Hypothesis file metadata dict from ``load_hypothesis_metadata``.
    instrument
        Single instrument the predicate is being built for. Required keyword
        argument to prevent positional arg confusion.

    Returns
    -------
    ScopePredicate
        Immutable predicate containing only hypotheses that apply to this
        instrument. The instrument field is stamped for later
        consistency-checking by the caller.

    Raises
    ------
    HypothesisLoaderError
        If ``meta['hypotheses']`` is missing/empty, any hypothesis has a
        malformed scope block, or ZERO hypotheses declare the given
        instrument (instrument not in scope for this file).
    """
    hypotheses_raw = meta.get("hypotheses")
    if not isinstance(hypotheses_raw, list) or not hypotheses_raw:
        raise HypothesisLoaderError(
            "Hypothesis file has no hypotheses list or it is empty. "
            "Cannot build scope predicate."
        )

    filtered: list[HypothesisScope] = []
    total_trials = 0

    for idx, h in enumerate(hypotheses_raw):
        if not isinstance(h, dict):
            raise HypothesisLoaderError(
                f"Hypothesis #{idx} must be a mapping, got {type(h).__name__}"
            )

        scope = h.get("scope")
        if not isinstance(scope, dict):
            raise HypothesisLoaderError(
                f"Hypothesis #{idx} missing or malformed 'scope' block"
            )

        instruments_raw = scope.get("instruments")
        if not isinstance(instruments_raw, list) or not instruments_raw:
            raise HypothesisLoaderError(
                f"Hypothesis #{idx} scope.instruments must be a non-empty list"
            )

        if instrument not in instruments_raw:
            continue  # not for this instrument — skip

        filter_block = h.get("filter")
        if not isinstance(filter_block, dict):
            raise HypothesisLoaderError(
                f"Hypothesis #{idx} missing or malformed 'filter' block"
            )
        filter_type = filter_block.get("type")
        if not isinstance(filter_type, str) or not filter_type:
            raise HypothesisLoaderError(
                f"Hypothesis #{idx} filter.type must be a non-empty string"
            )

        sessions_raw = scope.get("sessions")
        if not isinstance(sessions_raw, list) or not sessions_raw:
            raise HypothesisLoaderError(
                f"Hypothesis #{idx} scope.sessions must be a non-empty list"
            )

        rr_raw = scope.get("rr_targets")
        if not isinstance(rr_raw, list) or not rr_raw:
            raise HypothesisLoaderError(
                f"Hypothesis #{idx} scope.rr_targets must be a non-empty list"
            )

        em_raw = scope.get("entry_models")
        if not isinstance(em_raw, list) or not em_raw:
            raise HypothesisLoaderError(
                f"Hypothesis #{idx} scope.entry_models must be a non-empty list"
            )

        cb_raw = scope.get("confirm_bars")
        if not isinstance(cb_raw, list) or not cb_raw:
            raise HypothesisLoaderError(
                f"Hypothesis #{idx} scope.confirm_bars must be a non-empty list"
            )

        stop_raw = scope.get("stop_multipliers")
        if not isinstance(stop_raw, list) or not stop_raw:
            raise HypothesisLoaderError(
                f"Hypothesis #{idx} scope.stop_multipliers must be a non-empty list"
            )

        expected = h.get("expected_trial_count")
        if not isinstance(expected, int) or expected < 1:
            raise HypothesisLoaderError(
                f"Hypothesis #{idx} expected_trial_count must be a positive int"
            )

        try:
            rr_targets = frozenset(float(x) for x in rr_raw)
            confirm_bars = frozenset(int(x) for x in cb_raw)
            stop_multipliers = frozenset(float(x) for x in stop_raw)
        except (TypeError, ValueError) as exc:
            raise HypothesisLoaderError(
                f"Hypothesis #{idx} scope contains non-numeric values: {exc}"
            ) from exc

        filtered.append(
            HypothesisScope(
                filter_type=filter_type,
                instruments=frozenset(instruments_raw),
                sessions=frozenset(sessions_raw),
                rr_targets=rr_targets,
                entry_models=frozenset(em_raw),
                confirm_bars=confirm_bars,
                stop_multipliers=stop_multipliers,
                expected_trial_count=expected,
            )
        )
        total_trials += expected

    if not filtered:
        raise HypothesisLoaderError(
            f"Hypothesis file declares no hypotheses for instrument "
            f"{instrument!r}. Check per-hypothesis scope.instruments lists."
        )

    return ScopePredicate(
        hypotheses=tuple(filtered),
        instrument=instrument,
        total_declared_trials=total_trials,
    )


__all__ = [
    "HypothesisLoaderError",
    "HypothesisScope",
    "ScopePredicate",
    "check_mode_a_consistency",
    "compute_file_sha",
    "enforce_minbtl_bound",
    "extract_scope_predicate",
    "find_hypothesis_file_by_sha",
    "hypothesis_dir",
    "load_hypothesis_by_sha",
    "load_hypothesis_metadata",
]
