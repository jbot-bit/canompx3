"""Append-only fast-lane trial ledger (Stage 2A.2).

Universe-of-trials accounting per Bailey-Lopez de Prado 2014 § 3 ("a backtest
where the researcher has not controlled for the extent of the search involved
in his or her finding is worthless"). Every fast-lane / Pathway-A prereg
execution appends one entry; entries are never deleted or mutated. The
``structural_hash`` field (computed by ``fast_lane_structural_hash``) is the
de-dup criterion the 2A.3 scanner consumes to enforce sibling-retest /
graveyard / K-overrun suppression.

Design grounding:
  docs/runtime/stages/2026-05-20-fast-lane-anti-fp-trial-provenance.md
Implementation grounding:
  docs/runtime/stages/2026-05-20-fast-lane-anti-fp-2a2-ledger-digest.md

Capital-class boundary (CLAUDE.md § Source-of-Truth Chain Rule + design § Hard
Constraints): the writer REFUSES to append any entry whose ``prereg_path``
resolves to ``validated_setups`` / ``chordia_audit_log.yaml`` /
``lane_allocation.json`` / ``trading_app/live/``. The fast-lane ledger is for
candidate triage only, never validation / deployment state.

This module is pure (no ``__main__``); 2A.3's scanner is the call site that
populates the ledger. 2A.2 ships the writer + reader + capital-class refusal
+ the Check #169 invariant (append-only + holdout sentinel).
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Canonical sentinel values per design grounding § "Hard Constraints" +
# § "Trial Ledger Schema". Both are inlined here intentionally because the
# ledger writer must reject any entry that disagrees -- this is the
# holdout-boundary sentinel Check #169 enforces. The constants ARE the gate.
HOLDOUT_POLICY_SENTINEL = "mode_A"
HOLDOUT_SACRED_FROM_SENTINEL = "2026-01-01"

# Capital-class path substrings (case-insensitive). Any ``prereg_path`` whose
# lower-cased string contains one of these is refused at write time. The list
# mirrors CLAUDE.md § Database Location, § Source-of-Truth Chain Rule, and the
# design grounding's § "Files NOT to TOUCH" boundary.
#
# Scope honesty: this is a *substring* check, not a full canonical-path
# resolver. It catches the documented set of write-attempt vectors --
# repo-root-relative POSIX paths, repo-root-relative Windows paths, absolute
# paths that happen to traverse the canonical directory names. It does NOT
# catch hypothetical bypasses such as a UNC path on Windows, a symlink under
# a sibling name, or a typo like "validatedsetups" without the underscore.
# Those are out of scope because the fast-lane runner is the only documented
# call site -- there is no adversarial-input surface today. Future call sites
# that accept operator-supplied paths must either (a) resolve to a canonical
# absolute path first, or (b) extend this list with the new shape.
_CAPITAL_CLASS_FORBIDDEN_SUBSTRINGS = (
    "validated_setups",
    "chordia_audit_log.yaml",
    "lane_allocation.json",
    "trading_app/live/",
    "trading_app\\live\\",
)

# Schema version for the on-disk YAML file. Bump on any structural change to
# LedgerEntry; older readers must fail-closed if they see a higher version.
LEDGER_SCHEMA_VERSION = 1


class CapitalClassWriteRefused(Exception):
    """Raised when an append targets a capital-class file path."""


class LedgerAppendOnlyViolation(Exception):
    """Raised when an append would mutate or reorder existing entries."""


@dataclass(frozen=True)
class LedgerEntry:
    """One fast-lane prereg execution.

    Stage 2A.2 ships the minimum-required fields. Stage 2A.3 wires the
    scanner / ranker / bridge to populate the optional K-lineage + outcome
    fields. Holdout sentinels are required from day one because the
    capital-class boundary is the only reason the ledger is safe to keep.
    """

    run_id: str
    run_timestamp_utc: str  # ISO 8601 with explicit 'Z' (UTC)
    prereg_path: str
    prereg_sha: str
    structural_hash: str  # 16-hex from fast_lane_structural_hash
    template_version: str
    testing_mode: str  # "family" | "individual"
    pathway: str  # "A" | "B"
    K_declared: int
    holdout_policy: str = HOLDOUT_POLICY_SENTINEL
    holdout_sacred_from: str = HOLDOUT_SACRED_FROM_SENTINEL
    # 2A.3-populated optional surface; default-empty so 2A.2 callers don't
    # need to know about it yet.
    k_lineage: dict[str, Any] = field(default_factory=dict)
    n_hat: float | None = None
    upstream_provenance: dict[str, Any] = field(default_factory=dict)
    outcome: dict[str, Any] = field(default_factory=dict)


def _validate_capital_class_boundary(prereg_path: str) -> None:
    """Raise CapitalClassWriteRefused if prereg_path crosses the boundary."""
    # Runtime fail-closed guard per institutional-rigor.md § 6; annotation
    # type-narrows so Pyright flags as unreachable/unnecessary -- ignored
    # because public API must not trust caller type-discipline.
    if not isinstance(prereg_path, str):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(  # pyright: ignore[reportUnreachable]
            f"fast_lane_trial_ledger: prereg_path must be str, got {type(prereg_path).__name__}"
        )
    lowered = prereg_path.lower()
    for forbidden in _CAPITAL_CLASS_FORBIDDEN_SUBSTRINGS:
        if forbidden in lowered:
            raise CapitalClassWriteRefused(
                f"fast_lane_trial_ledger: REFUSED write -- prereg_path "
                f"{prereg_path!r} contains capital-class substring "
                f"{forbidden!r}. The fast-lane ledger is candidate-triage "
                f"only; capital-class state lives in its own audit log."
            )


def _validate_structural_hash(value: str) -> None:
    if not isinstance(value, str):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(  # pyright: ignore[reportUnreachable]
            f"fast_lane_trial_ledger: structural_hash must be str, got {type(value).__name__}"
        )
    if len(value) != 16:
        raise ValueError(f"fast_lane_trial_ledger: structural_hash must be 16 hex chars, got {len(value)} ({value!r})")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"fast_lane_trial_ledger: structural_hash {value!r} is not hex") from exc


def _parse_utc_ts(value: str) -> _dt.datetime:
    """Normalise a UTC ISO 8601 string (Z or +00:00 suffix) to an aware datetime.

    Used for monotonicity comparisons so mixed-suffix pairs compare correctly.
    Raises ValueError on malformed input — callers may propagate or wrap.
    """
    return _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def _validate_iso8601_utc(value: str) -> None:
    if not isinstance(value, str):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(  # pyright: ignore[reportUnreachable]
            f"fast_lane_trial_ledger: run_timestamp_utc must be str, got {type(value).__name__}"
        )
    # Accept either Z-suffix or +00:00 offset; reject naive timestamps so the
    # append-only timestamp comparison in Check #169 is unambiguous.
    if not (value.endswith("Z") or value.endswith("+00:00")):
        raise ValueError(
            f"fast_lane_trial_ledger: run_timestamp_utc must be UTC (end with 'Z' or '+00:00'); got {value!r}"
        )
    try:
        _parse_utc_ts(value)
    except ValueError as exc:
        raise ValueError(f"fast_lane_trial_ledger: run_timestamp_utc {value!r} is not ISO 8601") from exc


def _validate_holdout_sentinels(entry_holdout_policy: str, entry_sacred_from: str) -> None:
    if entry_holdout_policy != HOLDOUT_POLICY_SENTINEL:
        raise ValueError(
            f"fast_lane_trial_ledger: holdout_policy must be {HOLDOUT_POLICY_SENTINEL!r}; got {entry_holdout_policy!r}"
        )
    if entry_sacred_from != HOLDOUT_SACRED_FROM_SENTINEL:
        raise ValueError(
            f"fast_lane_trial_ledger: holdout_sacred_from must be "
            f"{HOLDOUT_SACRED_FROM_SENTINEL!r}; got {entry_sacred_from!r}"
        )


def _entry_to_yaml_dict(entry: LedgerEntry) -> dict[str, Any]:
    """Render a LedgerEntry as a YAML-safe dict in canonical field order."""
    return {
        "run_id": entry.run_id,
        "run_timestamp_utc": entry.run_timestamp_utc,
        "prereg_path": entry.prereg_path,
        "prereg_sha": entry.prereg_sha,
        "structural_hash": entry.structural_hash,
        "template_version": entry.template_version,
        "testing_mode": entry.testing_mode,
        "pathway": entry.pathway,
        "K_declared": entry.K_declared,
        "holdout_policy": entry.holdout_policy,
        "holdout_sacred_from": entry.holdout_sacred_from,
        "k_lineage": dict(entry.k_lineage),
        "n_hat": entry.n_hat,
        "upstream_provenance": dict(entry.upstream_provenance),
        "outcome": dict(entry.outcome),
    }


def read_ledger(ledger_path: Path) -> dict[str, Any]:
    """Load the ledger YAML. Returns a dict with schema_version + entries.

    Fails fast on missing banner, missing schema_version, or schema mismatch.
    """
    if not ledger_path.exists():
        raise FileNotFoundError(f"fast_lane_trial_ledger: ledger file {ledger_path} does not exist")
    raw = ledger_path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError(f"fast_lane_trial_ledger: {ledger_path} top-level must be a dict")
    if data.get("do_not_hand_edit") is not True:
        raise ValueError(f"fast_lane_trial_ledger: {ledger_path} missing `do_not_hand_edit: true` banner")
    if data.get("schema_version") != LEDGER_SCHEMA_VERSION:
        raise ValueError(
            f"fast_lane_trial_ledger: {ledger_path} schema_version "
            f"{data.get('schema_version')!r} != expected "
            f"{LEDGER_SCHEMA_VERSION}"
        )
    entries = data.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError(f"fast_lane_trial_ledger: {ledger_path} `entries` must be a list")
    return data


def append_trial_ledger_entry(
    ledger_path: Path,
    entry: LedgerEntry,
) -> None:
    """Append one entry to the on-disk ledger. Idempotent on run_id.

    Raises:
        CapitalClassWriteRefused: entry.prereg_path crosses capital-class
            boundary.
        LedgerAppendOnlyViolation: entry.run_id already exists, or the
            entry's run_timestamp_utc is older than the most recent entry's.
        ValueError / TypeError: on any malformed field per the validators.
        FileNotFoundError: ledger file does not exist (caller must land the
            skeleton first; 2A.2 stage ships the skeleton).
    """
    _validate_capital_class_boundary(entry.prereg_path)
    _validate_structural_hash(entry.structural_hash)
    _validate_iso8601_utc(entry.run_timestamp_utc)
    _validate_holdout_sentinels(entry.holdout_policy, entry.holdout_sacred_from)

    if not isinstance(entry.run_id, str) or not entry.run_id.strip():
        raise ValueError(f"fast_lane_trial_ledger: run_id must be non-empty str, got {entry.run_id!r}")
    if entry.pathway not in {"A", "B"}:
        raise ValueError(f"fast_lane_trial_ledger: pathway must be 'A' or 'B', got {entry.pathway!r}")
    if entry.testing_mode not in {"family", "individual"}:
        raise ValueError(
            f"fast_lane_trial_ledger: testing_mode must be 'family' or 'individual', got {entry.testing_mode!r}"
        )
    if not isinstance(entry.K_declared, int) or isinstance(entry.K_declared, bool):
        raise TypeError(f"fast_lane_trial_ledger: K_declared must be int, got {type(entry.K_declared).__name__}")
    if entry.K_declared < 1:
        raise ValueError(f"fast_lane_trial_ledger: K_declared must be >= 1, got {entry.K_declared}")

    data = read_ledger(ledger_path)
    entries: list[dict[str, Any]] = list(data.get("entries", []))

    # Append-only invariants: dup run_id forbidden; new timestamp must be
    # >= last entry's timestamp. Both also enforced by Check #169 across the
    # whole file (timestamp monotonicity); the writer enforces them on the
    # incoming entry so a runtime bypass cannot land a bad row.
    seen_run_ids = {row.get("run_id") for row in entries}
    if entry.run_id in seen_run_ids:
        raise LedgerAppendOnlyViolation(
            f"fast_lane_trial_ledger: run_id {entry.run_id!r} already "
            f"present in {ledger_path.name}; ledger is append-only -- "
            f"emit a new run_id rather than overwriting."
        )
    if entries:
        last_ts = entries[-1].get("run_timestamp_utc")
        if isinstance(last_ts, str):
            try:
                if _parse_utc_ts(last_ts) > _parse_utc_ts(entry.run_timestamp_utc):
                    raise LedgerAppendOnlyViolation(
                        f"fast_lane_trial_ledger: incoming run_timestamp_utc "
                        f"{entry.run_timestamp_utc!r} is older than last entry's "
                        f"{last_ts!r}; ledger requires monotonic non-decreasing "
                        f"timestamps."
                    )
            except ValueError:
                # Malformed timestamp in an existing entry — writer already
                # validates on write, so this should never happen on a clean
                # ledger. Treat as non-comparable and let the check catch it.
                pass

    entries.append(_entry_to_yaml_dict(entry))
    data["entries"] = entries

    ledger_path.write_text(
        _dump_ledger_yaml(data),
        encoding="utf-8",
    )


def _dump_ledger_yaml(data: dict[str, Any]) -> str:
    """Render the ledger dict with banner first + stable ordering."""
    # Emit the banner + schema_version explicitly so order survives a round
    # trip; PyYAML's default dump alphabetises top-level keys which would
    # reorder `do_not_hand_edit` after `entries`.
    banner = "do_not_hand_edit: true\n"
    schema = f"schema_version: {data['schema_version']}\n"
    body = yaml.safe_dump(
        {"entries": data.get("entries", [])},
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=False,
    )
    return banner + schema + body


__all__ = [
    "HOLDOUT_POLICY_SENTINEL",
    "HOLDOUT_SACRED_FROM_SENTINEL",
    "LEDGER_SCHEMA_VERSION",
    "CapitalClassWriteRefused",
    "LedgerAppendOnlyViolation",
    "LedgerEntry",
    "append_trial_ledger_entry",
    "read_ledger",
    "_parse_utc_ts",
]
