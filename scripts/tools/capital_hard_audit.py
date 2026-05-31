#!/usr/bin/env python3
"""Hard capital-path auditor with evidence and framing gates."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from scripts.tools.live_readiness_report import build_live_readiness_report  # noqa: E402
from scripts.tools.project_pulse import PulseItem, build_pulse  # noqa: E402

VERDICTS = ("CLEAR", "ACCEPT_WITH_RISK", "VERIFY_MORE", "BLOCK")
EVIDENCE_CLASSES = ("MEASURED", "INFERRED", "UNSUPPORTED", "SKIPPED_WITH_RESIDUAL_RISK")
ROLE_CHOICES = (
    "standalone",
    "filter",
    "conditioner",
    "allocator",
    "replacement",
    "diagnostic",
    "shadow-only",
)
FINALITY_TOKENS = (
    "DEAD_FOR_ORB",
    "NOT_DEPLOYABLE",
    "NO_GO",
    "dead for orb",
    "dead",
    "no-go",
    "not deployable",
)


@dataclass(frozen=True)
class ClaimEvidence:
    path: str
    evidence_class: str
    alternative_framings: list[str]
    disconfirming_checks: list[str]
    unchecked_scope: list[str]
    accepted_risks: list[str]
    finality_claims: list[str]


def _git_context(root: Path) -> dict[str, Any]:
    def run(*args: str) -> tuple[int, str]:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=str(root),
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return 127, ""
        return result.returncode, (result.stdout or "").strip()

    head_rc, head = run("rev-parse", "--short", "HEAD")
    branch_rc, branch = run("branch", "--show-current")
    status_rc, status = run("status", "--porcelain")
    detached = branch_rc == 0 and not branch
    return {
        "head": head if head_rc == 0 else None,
        "branch": branch if branch_rc == 0 and branch else None,
        "detached": detached,
        "dirty": bool(status),
        "available": head_rc == 0 or branch_rc == 0 or status_rc == 0,
    }


def _canonical_role_alternatives(role: str) -> list[str]:
    baseline = [role]
    if role == "standalone":
        baseline.extend(["filter", "allocator"])
    elif role == "shadow-only":
        baseline.extend(["diagnostic", "allocator"])
    elif role == "allocator":
        baseline.extend(["filter", "shadow-only"])
    elif role == "filter":
        baseline.extend(["allocator", "diagnostic"])
    elif role == "replacement":
        baseline.extend(["allocator", "standalone"])
    else:
        baseline.extend(["filter", "shadow-only"])
    seen: list[str] = []
    for item in baseline:
        if item not in seen:
            seen.append(item)
    return seen


def _extract_section_lines(lines: list[str], heading_markers: tuple[str, ...]) -> list[str]:
    collecting = False
    collected: list[str] = []
    for raw in lines:
        line = raw.strip()
        lower = line.lower()
        if any(marker in lower for marker in heading_markers):
            collecting = True
            continue
        if collecting and line.startswith("#"):
            break
        if collecting and line:
            collected.append(line.lstrip("-* ").strip())
    return collected


def _classify_claim(path: Path) -> ClaimEvidence:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    lower = text.lower()
    alternative_framings = _extract_section_lines(
        lines,
        ("alternative framing", "alternative framings checked", "anti-pigeonhole", "anti-tunnel"),
    )
    disconfirming_checks = _extract_section_lines(
        lines,
        ("what would falsify", "disconfirming check", "stop condition"),
    )
    unchecked_scope = _extract_section_lines(lines, ("unchecked scope",))
    accepted_risks = _extract_section_lines(lines, ("limitations", "accepted risk", "accepted risks"))
    finality_claims = [token for token in FINALITY_TOKENS if token.lower() in lower]

    if disconfirming_checks and unchecked_scope:
        evidence_class = "MEASURED"
    elif "pytest" in lower or "check_drift" in lower or "verified" in lower or "canonical evidence" in lower:
        evidence_class = "INFERRED"
    else:
        evidence_class = "UNSUPPORTED"

    return ClaimEvidence(
        path=str(path),
        evidence_class=evidence_class,
        alternative_framings=alternative_framings,
        disconfirming_checks=disconfirming_checks,
        unchecked_scope=unchecked_scope,
        accepted_risks=accepted_risks,
        finality_claims=finality_claims,
    )


def _pulse_blockers(pulse_items: list[PulseItem]) -> list[str]:
    blockers: list[str] = []
    for item in pulse_items:
        if item.category == "broken" or (
            item.source in {"followup_coverage", "queue_reconciliation"} and item.category == "unactioned"
        ):
            blockers.append(f"{item.source}: {item.summary}")
    return blockers


def _framing_defects(
    *,
    role: str,
    decision_target: str,
    claims: list[ClaimEvidence],
    alternative_framings: list[str],
    unchecked_scope: list[str],
) -> list[str]:
    defects: list[str] = []
    lower_target = decision_target.lower()
    requires_alt = role in {"standalone", "filter", "allocator", "replacement"} or any(
        token in lower_target for token in ("dead", "deploy", "promote", "live", "expansion")
    )
    if requires_alt and len(alternative_framings) < 2:
        defects.append("Alternative framing review is incomplete for this capital decision.")
    if not unchecked_scope:
        defects.append("Unchecked scope is missing, so narrowed conclusions could overclaim.")
    for claim in claims:
        if claim.finality_claims and len(alternative_framings) < 2:
            defects.append(f"Claim {claim.path} asserts finality without alternative framing coverage.")
    return defects


def build_capital_hard_audit(
    *,
    decision_target: str,
    role: str,
    object_unit: str,
    horizon: str,
    profile_id: str = "topstep_50k_mnq_auto",
    claim_paths: list[Path] | None = None,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    git = _git_context(root)
    readiness = build_live_readiness_report(profile_id=profile_id, db_path=GOLD_DB_PATH)
    pulse = build_pulse(
        root, db_path=GOLD_DB_PATH, fast=True, skip_drift=True, skip_tests=True, tool_name="capital_hard_audit"
    )
    claims = [_classify_claim(path) for path in (claim_paths or []) if path.exists()]

    readiness_gate = readiness.get("strict_zero_warn") or {}
    blockers = list(readiness_gate.get("blockers") or [])
    blockers.extend(_pulse_blockers(pulse.items))
    if git.get("detached"):
        blockers.append("Git context is detached; capital-path decisions require explicit branch context.")

    warnings = list(readiness_gate.get("warnings") or [])
    alternative_framings = [role]
    for claim in claims:
        alternative_framings.extend(claim.alternative_framings)
    normalized_alternatives: list[str] = []
    for item in alternative_framings:
        value = item.strip()
        if value and value not in normalized_alternatives:
            normalized_alternatives.append(value)

    disconfirming_checks: list[str] = []
    unchecked_scope: list[str] = []
    accepted_risks: list[str] = []
    for claim in claims:
        disconfirming_checks.extend(claim.disconfirming_checks)
        unchecked_scope.extend(claim.unchecked_scope)
        accepted_risks.extend(claim.accepted_risks)

    if not disconfirming_checks:
        disconfirming_checks = [
            "strict-zero-warn must stay green for the exact profile under review",
            "pulse must not carry unresolved broken/follow-up blockers for this decision path",
        ]

    framing_defects = _framing_defects(
        role=role,
        decision_target=decision_target,
        claims=claims,
        alternative_framings=normalized_alternatives,
        unchecked_scope=unchecked_scope,
    )

    evidence_pack: list[dict[str, Any]] = [
        {
            "source": "live_readiness_report",
            "evidence_class": "MEASURED" if readiness_gate.get("green") is not None else "UNSUPPORTED",
            "artifact": f"profile={readiness.get('profile_id')}",
        },
        {
            "source": "project_pulse",
            "evidence_class": "MEASURED",
            "artifact": f"git={pulse.git_head}",
        },
    ]
    for claim in claims:
        evidence_pack.append(
            {
                "source": "claim",
                "evidence_class": claim.evidence_class,
                "artifact": claim.path,
            }
        )

    if blockers:
        verdict = "BLOCK"
    elif framing_defects:
        verdict = "VERIFY_MORE"
    elif warnings:
        verdict = "ACCEPT_WITH_RISK" if role == "shadow-only" else "VERIFY_MORE"
        if not accepted_risks:
            accepted_risks = warnings.copy()
    else:
        verdict = "CLEAR"

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "decision_target": decision_target,
        "verdict": verdict,
        "object": {
            "unit": object_unit,
            "horizon": horizon,
            "role": role,
        },
        "git_context": git,
        "profile_id": readiness.get("profile_id"),
        "blockers": blockers,
        "warnings": warnings,
        "accepted_risks": accepted_risks,
        "framing_defects": framing_defects,
        "alternative_framings": normalized_alternatives or _canonical_role_alternatives(role),
        "disconfirming_checks": disconfirming_checks,
        "unchecked_scope": unchecked_scope,
        "evidence_pack": evidence_pack,
        "evidence_classes": list(EVIDENCE_CLASSES),
        "claims_reviewed": [claim.path for claim in claims],
        "pulse_summary": {
            "git_head": pulse.git_head,
            "git_branch": pulse.git_branch,
            "broken_count": len(pulse.broken),
            "decaying_count": len(pulse.decaying),
        },
        "strict_zero_warn": readiness_gate,
    }


def render_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True)


def render_text(report: dict[str, Any]) -> str:
    lines = [
        f"Capital Hard Audit | verdict={report['verdict']} | target={report['decision_target']}",
        f"Object: unit={report['object']['unit']} horizon={report['object']['horizon']} role={report['object']['role']}",
        f"Git: branch={report['git_context'].get('branch') or 'DETACHED'} head={report['git_context'].get('head') or 'unknown'}",
        "",
        "Blockers:",
    ]
    lines.extend(f"- {item}" for item in (report["blockers"] or ["none"]))
    lines.append("")
    lines.append("Alternative framings checked:")
    lines.extend(f"- {item}" for item in report["alternative_framings"])
    lines.append("")
    lines.append("What would falsify this verdict:")
    lines.extend(f"- {item}" for item in report["disconfirming_checks"])
    lines.append("")
    lines.append("Unchecked scope:")
    lines.extend(f"- {item}" for item in (report["unchecked_scope"] or ["none recorded"]))
    if report["framing_defects"]:
        lines.append("")
        lines.append("Framing defects:")
        lines.extend(f"- {item}" for item in report["framing_defects"])
    if report["accepted_risks"]:
        lines.append("")
        lines.append("Accepted risks:")
        lines.extend(f"- {item}" for item in report["accepted_risks"])
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-target", required=True)
    parser.add_argument("--role", choices=ROLE_CHOICES, required=True)
    parser.add_argument("--object-unit", required=True)
    parser.add_argument("--horizon", required=True)
    parser.add_argument("--profile-id", default="topstep_50k_mnq_auto")
    parser.add_argument("--claim-path", action="append", default=[])
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_capital_hard_audit(
        decision_target=args.decision_target,
        role=args.role,
        object_unit=args.object_unit,
        horizon=args.horizon,
        profile_id=args.profile_id,
        claim_paths=[Path(item) for item in args.claim_path],
    )
    print(render_json(report) if args.format == "json" else render_text(report))
    return 0 if report["verdict"] in {"CLEAR", "ACCEPT_WITH_RISK"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
