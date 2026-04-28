"""Manifest dataclass + JSON serializer for evidence packs.

Pure data; no I/O. Every value-bearing leaf carries provenance siblings
(<field>_source, <field>_as_of) per design doc § 6. Serializer emits a
deterministic, sorted-key JSON document so two runs against the same git
SHA + DB fingerprint produce byte-identical bytes.

Verdict values (locked):
  PASS, CONDITIONAL, KILL, INCOMPLETE_EVIDENCE, LANE_NOT_SUPPORTED_BY_POOLED.

Gate status values (locked):
  PASS, FAIL, UNCOMPUTED, CROSS_CHECK_ONLY.

Contamination registry status (locked):
  PRESENT, MISSING.

Contamination overall status (locked):
  CLEAN, TAINTED, UNCOMPUTED.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

VERDICT_VALUES = frozenset(
    {
        "PASS",
        "CONDITIONAL",
        "KILL",
        "INCOMPLETE_EVIDENCE",
        "LANE_NOT_SUPPORTED_BY_POOLED",
    }
)

GATE_STATUS_VALUES = frozenset({"PASS", "FAIL", "UNCOMPUTED", "CROSS_CHECK_ONLY"})

REGISTRY_STATUS_VALUES = frozenset({"PRESENT", "MISSING"})

CONTAMINATION_STATUS_VALUES = frozenset({"CLEAN", "TAINTED", "UNCOMPUTED"})


@dataclass(frozen=True)
class Provenance:
    """Provenance siblings for any value-bearing leaf."""

    source: str
    as_of: str  # ISO-8601


@dataclass(frozen=True)
class GateResult:
    name: str
    status: str  # PASS | FAIL | UNCOMPUTED | CROSS_CHECK_ONLY
    value: str | float | int | None
    threshold: str | float | int | None
    source: str
    note: str = ""

    def __post_init__(self) -> None:
        if self.status not in GATE_STATUS_VALUES:
            raise ValueError(f"GateResult.status invalid: {self.status!r}")


@dataclass(frozen=True)
class Query:
    label: str
    sql_text: str
    sql_sha256: str
    rerun_command: str


@dataclass(frozen=True)
class Contamination:
    registry_paths: tuple[str, ...]  # sorted glob results, may be empty
    registry_status: str  # PRESENT | MISSING
    hits: tuple[str, ...]  # tainted commit hashes matching the result
    status: str  # CLEAN | TAINTED | UNCOMPUTED
    expected_glob: str
    note: str = ""

    def __post_init__(self) -> None:
        if self.registry_status not in REGISTRY_STATUS_VALUES:
            raise ValueError(f"Contamination.registry_status invalid: {self.registry_status!r}")
        if self.status not in CONTAMINATION_STATUS_VALUES:
            raise ValueError(f"Contamination.status invalid: {self.status!r}")
        if self.registry_status == "MISSING" and self.status != "UNCOMPUTED":
            raise ValueError(
                "Contamination invariant: registry_status=MISSING requires "
                f"status=UNCOMPUTED, got status={self.status!r}"
            )


@dataclass(frozen=True)
class Manifest:
    pack_version: str
    slug: str
    run_iso8601: str
    git_sha: str | None
    db_fingerprint: str | None
    db_path: str
    holdout_date: str  # YYYY-MM-DD
    hypothesis: dict[str, Any]
    result: dict[str, Any]
    validated_setups: dict[str, Any] | None
    tables_used: tuple[str, ...]
    is_oos_split: dict[str, Any]
    k_framings: dict[str, Any]
    gates: tuple[GateResult, ...]
    queries: tuple[Query, ...]
    contamination: Contamination
    verdict: str  # PASS | CONDITIONAL | KILL | INCOMPLETE_EVIDENCE | LANE_NOT_SUPPORTED_BY_POOLED
    verdict_reasons: tuple[str, ...]
    provenance: dict[str, Provenance] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.verdict not in VERDICT_VALUES:
            raise ValueError(f"Manifest.verdict invalid: {self.verdict!r}")
        if self.verdict != "PASS" and not self.verdict_reasons:
            raise ValueError(f"verdict={self.verdict!r} requires non-empty verdict_reasons")
        # Tables-used must be a subset of canonical layers
        canonical = {"bars_1m", "daily_features", "orb_outcomes"}
        unknown = set(self.tables_used) - canonical
        if unknown:
            raise ValueError(
                f"tables_used contains non-canonical layers: {sorted(unknown)}; "
                "discovery-truth must come from canonical layers per RESEARCH_RULES.md"
            )


def to_json_bytes(manifest: Manifest) -> bytes:
    """Deterministic, sorted-key JSON serialization of a Manifest.

    Two runs against the same git SHA + DB fingerprint produce byte-identical
    output. Acceptance criterion #1.
    """

    payload = asdict(manifest)
    return json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        default=_json_default,
    ).encode("utf-8")


def _json_default(obj: Any) -> Any:
    # dataclasses.asdict already unwraps nested dataclasses; this only fires
    # for unexpected types (tuple is JSON-native via json.dumps; date types
    # not currently used).
    raise TypeError(f"non-serializable type {type(obj).__name__}")


def walk_value_leaves(payload: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any]]:
    """Yield (path, value) for every scalar leaf in a JSON-shaped payload.

    Used by acceptance criterion #12: every value-bearing leaf in the manifest
    must carry a *_source and *_as_of sibling. Test walks the manifest dict
    and asserts that property holds for fields registered in
    Manifest.provenance.
    """

    out: list[tuple[tuple[str, ...], Any]] = []
    if isinstance(payload, dict):
        for k, v in payload.items():
            out.extend(walk_value_leaves(v, path + (str(k),)))
    elif isinstance(payload, (list, tuple)):
        for i, v in enumerate(payload):
            out.extend(walk_value_leaves(v, path + (f"[{i}]",)))
    else:
        out.append((path, payload))
    return out
