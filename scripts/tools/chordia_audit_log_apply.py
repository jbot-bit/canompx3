"""Apply reviewed Chordia audit-log proposals without touching live allocation.

The evidence factory and replay bridge deliberately emit proposal-only YAML.
This tool is the narrow review/apply step: it validates a proposal, skips
existing strategy_ids idempotently, and inserts reviewed audit rows into
``docs/runtime/chordia_audit_log.yaml``. It never mutates live allocation or
``validated_setups``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT_LOG = _REPO_ROOT / "docs" / "runtime" / "chordia_audit_log.yaml"
VALID_VERDICTS = frozenset({"PASS_CHORDIA", "PASS_PROTOCOL_A", "FAIL_CHORDIA", "FAIL_BOTH", "PARK"})


@dataclass(frozen=True)
class ApplyResult:
    proposal_path: str
    audit_log_path: str
    reviewed: bool
    write: bool
    applied_count: int
    skipped_existing_count: int
    strategy_ids_applied: tuple[str, ...]
    strategy_ids_skipped_existing: tuple[str, ...]
    live_mutation: bool = False
    validated_setups_mutation: bool = False


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return payload


def _existing_strategy_ids(audit_log: dict[str, Any]) -> set[str]:
    audits = audit_log.get("audits") or []
    if not isinstance(audits, list):
        raise ValueError("audit log 'audits' must be a list.")
    out: set[str] = set()
    for row in audits:
        if isinstance(row, dict) and row.get("strategy_id"):
            out.add(str(row["strategy_id"]))
    return out


def _validate_proposal(proposal: dict[str, Any]) -> list[dict[str, Any]]:
    if proposal.get("proposal_only") is not True:
        raise ValueError("proposal must have proposal_only: true.")
    audits = proposal.get("audits") or []
    if not isinstance(audits, list):
        raise ValueError("proposal 'audits' must be a list.")
    validated: list[dict[str, Any]] = []
    seen_strategy_ids: set[str] = set()
    for idx, row in enumerate(audits, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"proposal audit row {idx} must be a mapping.")
        strategy_id = str(row.get("strategy_id") or "").strip()
        verdict = str(row.get("verdict") or "").strip().upper()
        if not strategy_id:
            raise ValueError(f"proposal audit row {idx} has no strategy_id.")
        if verdict not in VALID_VERDICTS:
            raise ValueError(f"proposal audit row {idx} has invalid verdict {verdict!r}.")
        if not row.get("audit_date"):
            raise ValueError(f"proposal audit row {idx} has no audit_date.")
        if not row.get("source_result"):
            raise ValueError(f"proposal audit row {idx} has no source_result.")
        if strategy_id in seen_strategy_ids:
            raise ValueError(f"proposal audit row {idx} has duplicate strategy_id {strategy_id!r}.")
        seen_strategy_ids.add(strategy_id)
        validated.append(dict(row, strategy_id=strategy_id, verdict=verdict))
    return validated


def _format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _format_entry(row: dict[str, Any], *, reviewed_by: str) -> str:
    lines = [
        f"  - strategy_id: {row['strategy_id']}",
        f"    audit_date: {_format_scalar(row['audit_date'])}",
        f"    verdict: {row['verdict']}",
    ]
    if "has_theory" in row:
        lines.append(f"    has_theory: {_format_scalar(row['has_theory'])}")
    if row.get("t_stat") is not None:
        lines.append(f"    t_stat: {_format_scalar(row.get('t_stat'))}")
    lines.append("    threshold: 3.79")
    if row.get("sample_size") is not None:
        lines.append(f"    sample_size: {_format_scalar(row.get('sample_size'))}")
    lines.extend(
        [
            "    note: |",
            "      Reviewed Chordia evidence-factory proposal.",
            "      No theory grant claimed; strict no-theory threshold t>=3.79 applies.",
            f"      Reviewed by: {reviewed_by}.",
            f"      Result MD: `{row['source_result']}`",
        ]
    )
    return "\n".join(lines)


def _insert_after_audits_key(text: str, block: str) -> str:
    marker = "audits:\n"
    idx = text.find(marker)
    if idx < 0:
        raise ValueError("audit log has no top-level 'audits:' key.")
    insert_at = idx + len(marker)
    prefix = text[:insert_at]
    suffix = text[insert_at:]
    return f"{prefix}\n{block}\n{suffix}"


def apply_reviewed_proposal(
    *,
    proposal_path: Path,
    audit_log_path: Path = DEFAULT_AUDIT_LOG,
    reviewed: bool = False,
    reviewed_by: str = "agent-reviewed",
    write: bool = False,
) -> ApplyResult:
    proposal = _load_yaml(proposal_path)
    proposal_rows = _validate_proposal(proposal)
    audit_log = _load_yaml(audit_log_path)
    existing = _existing_strategy_ids(audit_log)

    skipped = tuple(row["strategy_id"] for row in proposal_rows if row["strategy_id"] in existing)
    rows_to_apply = [row for row in proposal_rows if row["strategy_id"] not in existing]
    applied = tuple(row["strategy_id"] for row in rows_to_apply)

    if write and not reviewed:
        raise ValueError("--write requires --reviewed.")
    if write and rows_to_apply:
        entries = "\n\n".join(_format_entry(row, reviewed_by=reviewed_by) for row in rows_to_apply)
        original = audit_log_path.read_text(encoding="utf-8")
        audit_log_path.write_text(_insert_after_audits_key(original, entries), encoding="utf-8", newline="\n")

    return ApplyResult(
        proposal_path=str(proposal_path),
        audit_log_path=str(audit_log_path),
        reviewed=reviewed,
        write=write,
        applied_count=len(applied),
        skipped_existing_count=len(skipped),
        strategy_ids_applied=applied,
        strategy_ids_skipped_existing=skipped,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", required=True, type=Path)
    parser.add_argument("--audit-log", default=DEFAULT_AUDIT_LOG, type=Path)
    parser.add_argument("--reviewed", action="store_true")
    parser.add_argument("--reviewed-by", default="agent-reviewed")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)

    result = apply_reviewed_proposal(
        proposal_path=args.proposal,
        audit_log_path=args.audit_log,
        reviewed=args.reviewed,
        reviewed_by=args.reviewed_by,
        write=args.write,
    )
    if args.format == "json":
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    else:
        mode = "WRITE" if result.write else "DRY_RUN"
        print(
            f"{mode}: apply={result.applied_count}, skip_existing={result.skipped_existing_count}, live_mutation=false"
        )
        for sid in result.strategy_ids_applied:
            print(f"APPLY {sid}")
        for sid in result.strategy_ids_skipped_existing:
            print(f"SKIP_EXISTING {sid}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
