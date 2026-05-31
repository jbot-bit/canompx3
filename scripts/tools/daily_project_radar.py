#!/usr/bin/env python3
"""Read-only daily project radar: hard risk scan plus bounded opportunity inbox."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tools import daily_bug_scan  # noqa: E402

VERDICT_ORDER = {"CLEAR": 0, "ACCEPT_WITH_RISK": 1, "VERIFY_MORE": 2, "FIX_REQUIRED": 3, "BLOCK": 4}
CAPITAL_PATH_PREFIXES = (
    "docs/audit/hypotheses/",
    "docs/audit/results/",
    "pipeline/",
    "research/",
    "trading_app/live/",
)
CAPITAL_PATH_TOKENS = (
    "broker",
    "execution",
    "order",
    "profile",
    "prop_",
    "risk",
    "run_live_session",
    "webhook",
)
NO_GO_TOKENS = (
    "nr7",
    "nr4",
    "ibs",
    "rsi",
    "macd",
    "bollinger",
    "ema20",
    "breakeven",
    "pyramiding",
    "non-orb",
)
OFFICIAL_SOURCE_WATCHLIST = (
    "CME trading hours and economic calendar",
    "FRED release API",
    "OpenAI changelog",
    "Anthropic release notes",
    "arXiv API",
)
TOOLING_SOURCE_WATCHLIST = (
    "OpenAI changelog, deprecations, docs MCP, usage/cost, and cost optimization docs",
    "Anthropic API release notes, usage/cost API, and Claude Code cost docs",
    "Codex and Claude Code MCP/plugin/tooling capability changes",
    "repo-local .codex/.claude routing, token hygiene, dependency, and automation surfaces",
)
LANE_NAMES = ("risk", "opportunity", "ai_tooling")


@dataclass(frozen=True)
class SentinelFinding:
    path: str
    summary: str
    evidence_class: str = "MEASURED"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _normalize(path: str) -> str:
    return path.replace("\\", "/")


def is_capital_impact_path(path: str) -> bool:
    normalized = _normalize(path).lower()
    return any(normalized.startswith(prefix) for prefix in CAPITAL_PATH_PREFIXES) or any(
        token in normalized for token in CAPITAL_PATH_TOKENS
    )


def _build_capital_hard_audit(**kwargs: Any) -> dict[str, Any]:
    from scripts.tools import capital_hard_audit

    return capital_hard_audit.build_capital_hard_audit(**kwargs)


def _all_candidate_paths(scan_packet: daily_bug_scan.ScanPacket) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for candidate in scan_packet.candidate_commits:
        for path in [*candidate.touched_code_paths, *candidate.touched_test_paths]:
            normalized = _normalize(path)
            if normalized not in seen:
                seen.add(normalized)
                paths.append(normalized)
    return paths


def _expected_test_path(path: str) -> str | None:
    normalized = _normalize(path)
    source = Path(normalized)
    stem = source.stem
    if normalized.startswith("scripts/tools/") and normalized.endswith(".py"):
        return f"tests/test_tools/test_{stem}.py"
    if normalized.startswith("trading_app/") and normalized.endswith(".py") and len(source.parts) == 2:
        return f"tests/test_trading_app/test_{stem}.py"
    if normalized.startswith("pipeline/") and normalized.endswith(".py") and len(source.parts) == 2:
        return f"tests/test_pipeline/test_{stem}.py"
    return None


def targeted_behavioral_sentinels(root: Path, paths: list[str]) -> list[dict[str, str]]:
    findings: list[SentinelFinding] = []
    for raw in paths:
        normalized = _normalize(raw)
        if normalized.startswith("tests/"):
            continue
        path = root / normalized
        expected_test = _expected_test_path(normalized)
        if expected_test and not (root / expected_test).exists():
            findings.append(
                SentinelFinding(
                    path=normalized,
                    summary=f"Missing companion test: expected {expected_test}",
                    evidence_class="INFERRED",
                )
            )
        if not path.exists() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lower = text.lower()
        if re.search(r"\.?\\+\.venv\\+scripts\\+python\.exe", lower):
            findings.append(SentinelFinding(path=normalized, summary="Hardcoded repo Python command"))
        deprecated_db_patterns = ("c:/db/" + "gold.db", "c:\\db\\" + "gold.db")
        if any(pattern in lower for pattern in deprecated_db_patterns):
            findings.append(SentinelFinding(path=normalized, summary="Hardcoded deprecated gold.db path"))
        for idx, line in enumerate(lower.splitlines()):
            if not re.search(r"except\s+exception", line):
                continue
            local_block = "\n".join(lower.splitlines()[idx + 1 : idx + 4])
            if re.search(r"^\s*(return\s+(true|\[\]|\{\}|0)|pass)\b", local_block, re.MULTILINE):
                findings.append(
                    SentinelFinding(path=normalized, summary="Exception path may return success or pass silently")
                )
                break
        if path.suffix.lower() in {".md", ".txt"}:
            has_readiness_claim = re.search(r"\b(clear|done|green|ready|safe)\b", lower)
            has_evidence = re.search(r"\b(pytest|check_drift|command|evidence|verified)\b", lower)
            if has_readiness_claim and not has_evidence:
                findings.append(SentinelFinding(path=normalized, summary="Unsupported readiness language in doc claim"))
    return [asdict(finding) for finding in findings]


def _false_negative_sample(scan_packet: daily_bug_scan.ScanPacket) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    if scan_packet.skipped_commits:
        skipped = scan_packet.skipped_commits[0]
        samples.append(
            {
                "source": "skipped_commit",
                "sha": skipped.sha,
                "subject": skipped.subject,
                "reason": skipped.reason,
                "sampled_paths": skipped.touched_paths[:5],
            }
        )
    return samples


def _confidence_limiter(
    *,
    skipped_surfaces: list[str],
    false_negative_sample: list[dict[str, Any]],
    silence_ledger: list[dict[str, Any]] | None = None,
) -> str:
    if any(item.get("downgrades_verdict") for item in silence_ledger or []):
        return "downgrading silence ledger entry"
    if not false_negative_sample:
        return "no false-negative sample available"
    if skipped_surfaces:
        return "skipped daily surfaces"
    return "none"


def build_lane_audit(
    *,
    lane: str,
    checked_surfaces: list[str],
    skipped_surfaces: list[str],
    excluded_but_relevant_surfaces: list[str],
    false_negative_sample: list[dict[str, Any]],
    counter_framings: list[str],
    disconfirming_checks: list[str],
    unchecked_scope: list[str],
    residual_risk: list[str],
    silence_ledger: list[dict[str, Any]] | None = None,
    self_check_findings: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    audit = {
        "lane": lane,
        "checked_surfaces": checked_surfaces,
        "skipped_surfaces": skipped_surfaces,
        "excluded_but_relevant_surfaces": excluded_but_relevant_surfaces,
        "false_negative_sample": false_negative_sample,
        "counter_framings": counter_framings,
        "disconfirming_checks": disconfirming_checks,
        "what_would_falsify_verdict": disconfirming_checks,
        "unchecked_scope": unchecked_scope,
        "residual_risk": residual_risk,
        "silence_ledger": silence_ledger or [],
        "confidence_limiter": _confidence_limiter(
            skipped_surfaces=skipped_surfaces,
            false_negative_sample=false_negative_sample,
            silence_ledger=silence_ledger,
        ),
        "self_check_findings": self_check_findings or [],
    }
    if extra:
        audit.update(extra)
    return audit


def _audit_of_auditor(
    *,
    scan_packet: daily_bug_scan.ScanPacket,
    capital_paths: list[str],
    capital_audit: dict[str, Any],
    sentinel_findings: list[dict[str, str]],
) -> dict[str, Any]:
    skipped_surfaces = [
        "full behavioral audit skipped in daily mode; targeted sentinels used instead",
        "full test suite skipped in daily mode; targeted scan/report only",
    ]
    if not capital_paths:
        skipped_surfaces.append("capital_hard_audit skipped: no capital-impact paths detected")
    false_negative_sample = _false_negative_sample(scan_packet)
    unchecked_scope = [
        "does not prove absence of bugs outside commit window",
        "does not execute broad pytest suite",
        "does not run full behavioral audit by default",
    ]
    if not false_negative_sample:
        unchecked_scope.append("no skipped commit was available for false-negative sampling")
    return build_lane_audit(
        lane="risk",
        checked_surfaces=[
            "daily_bug_scan",
            "git_context",
            "working_tree_candidates",
            "capital_impact_classifier",
            "targeted_behavioral_sentinels",
        ],
        skipped_surfaces=skipped_surfaces,
        excluded_but_relevant_surfaces=[
            "full CI status not queried by daily local scanner",
            "runtime logs and broker telemetry not read in daily risk lane",
        ],
        false_negative_sample=false_negative_sample,
        counter_framings=[
            "tooling-only changes may still affect capital indirectly through automation",
            "no capital path detected does not prove no deploy-readiness risk",
            "a skipped doc-only commit may still contain unsupported readiness claims",
        ],
        disconfirming_checks=[
            "a skipped commit contains production code missed by path classification",
            "a changed path has no targeted or companion test coverage",
            "capital_hard_audit reports BLOCK for capital-impact changes",
            "full behavioral audit finds a fail-open or silent-success pattern",
        ],
        unchecked_scope=unchecked_scope,
        residual_risk=[
            "static-only verification can miss runtime failures",
            "bounded daily scan can miss low-frequency or cross-commit regressions",
        ],
        self_check_findings=sentinel_findings,
        extra={"capital_audit_status": capital_audit.get("status", capital_audit.get("verdict", "unknown"))},
    )


def _max_verdict(*verdicts: str) -> str:
    return max(verdicts, key=lambda item: VERDICT_ORDER.get(item, 0))


def build_risk_lane(root: Path, scan_packet: daily_bug_scan.ScanPacket) -> dict[str, Any]:
    paths = _all_candidate_paths(scan_packet)
    capital_paths = [path for path in paths if is_capital_impact_path(path)]
    sentinel_findings = targeted_behavioral_sentinels(root, paths)
    capital_audit: dict[str, Any]
    if capital_paths:
        try:
            capital_audit = _build_capital_hard_audit(
                decision_target="daily radar capital-impact review",
                role="diagnostic",
                object_unit="changed-path set",
                horizon="daily",
                root=root,
            )
            capital_audit["status"] = capital_audit.get("verdict", "unknown")
        except Exception as exc:  # pragma: no cover - defensive wrapper for CLI robustness
            capital_audit = {"status": "BLOCKED", "verdict": "BLOCK", "error": str(exc)}
    else:
        capital_audit = {"status": "skipped", "reason": "no capital-impact paths detected"}

    audit = _audit_of_auditor(
        scan_packet=scan_packet,
        capital_paths=capital_paths,
        capital_audit=capital_audit,
        sentinel_findings=sentinel_findings,
    )

    verdict = "CLEAR"
    if scan_packet.verification.mode == "blocked":
        verdict = _max_verdict(verdict, "BLOCK")
    elif scan_packet.verification.mode != "full":
        verdict = _max_verdict(verdict, "VERIFY_MORE")
    if sentinel_findings:
        verdict = _max_verdict(verdict, "FIX_REQUIRED")
    if scan_packet.omitted_candidate_count:
        verdict = _max_verdict(verdict, "VERIFY_MORE")
    if not audit["false_negative_sample"]:
        verdict = _max_verdict(verdict, "VERIFY_MORE")
    if capital_audit.get("verdict") == "BLOCK":
        verdict = _max_verdict(verdict, "BLOCK")
    elif capital_audit.get("verdict") in {"VERIFY_MORE", "ACCEPT_WITH_RISK"}:
        verdict = _max_verdict(verdict, capital_audit["verdict"])
    if verdict == "CLEAR" and audit["skipped_surfaces"]:
        verdict = "ACCEPT_WITH_RISK"

    return {
        "verdict": verdict,
        "scan": asdict(scan_packet),
        "capital_impact_paths": capital_paths,
        "capital_audit": capital_audit,
        "targeted_behavioral_findings": sentinel_findings,
        "lane_audit": audit,
        "audit_of_auditor": audit,
    }


def _fingerprint(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def classify_idea(raw: dict[str, Any]) -> dict[str, Any]:
    source_type = str(raw.get("source_type") or "unknown")
    source = str(raw.get("source") or "unknown")
    claim = str(raw.get("claim") or "").strip()
    mechanism = str(raw.get("mechanism") or "").strip()
    lower_claim = claim.lower()
    matched_no_go = next((token for token in NO_GO_TOKENS if token in lower_claim), None)
    if matched_no_go:
        disposition = "REJECT"
        rationale = f"Matches known NO-GO registry term: {matched_no_go}"
    elif not mechanism:
        disposition = "PARK"
        rationale = "Missing mechanism; numbers or novelty alone are not enough"
    elif source_type == "community":
        disposition = "PARK"
        rationale = "Unofficial community source; cannot become project truth without corroboration"
    else:
        disposition = "ESCALATE_TO_RESEARCH_PLAN"
        rationale = "Mechanism-bearing official/literature signal; requires preregistered repo research before testing"
    return {
        "source_type": source_type,
        "source": source,
        "claim": claim,
        "mechanism": mechanism,
        "role": str(raw.get("role") or "unassigned"),
        "disposition": disposition,
        "rationale": rationale,
        "source_fingerprint": _fingerprint(source_type, source, claim),
    }


def build_opportunity_radar(
    *,
    external_items: list[dict[str, Any]] | None = None,
    max_external_items: int = 10,
    max_ideas: int = 3,
) -> dict[str, Any]:
    bounded_items = (external_items or [])[:max_external_items]
    ideas = [classify_idea(item) for item in bounded_items][:max_ideas]
    ignored_sample = [idea for idea in ideas if idea["disposition"] in {"PARK", "REJECT"}][:1]
    unchecked_scope = [
        "does not auto-backtest ideas",
        "does not query DB or mutate research state",
        "does not treat community or novelty claims as project truth",
    ]
    if not ignored_sample:
        unchecked_scope.append("no parked/rejected idea was available for false-negative sampling")
    lane_audit = build_lane_audit(
        lane="opportunity",
        checked_surfaces=[
            "bounded source items",
            "known NO-GO token screen",
            "mechanism presence",
            "source authority class",
        ],
        skipped_surfaces=[
            "full literature review skipped in daily mode",
            "repo-standard preregistered research protocol skipped until escalation",
        ],
        excluded_but_relevant_surfaces=[
            "canonical DB edge scans",
            "local literature corpus beyond supplied source items",
            "community/forum corroboration outside bounded source items",
        ],
        false_negative_sample=ignored_sample,
        counter_framings=[
            "not standalone may still be useful as a filter or allocator",
            "not deployable today may still be a research-plan seed",
            "one rejected route does not prove a mechanism class is dead",
        ],
        disconfirming_checks=[
            "a PARK idea contains a clear mechanism and official/literature source",
            "a REJECT idea is not actually covered by the NO-GO registry",
            "an ESCALATE idea lacks a bounded preregistered research path",
        ],
        unchecked_scope=unchecked_scope,
        residual_risk=[
            "bounded opportunity screening can miss mechanism-bearing weak signals",
            "static source classification can understate hidden local relevance",
        ],
    )
    return {
        "mode": "report_only",
        "watched_sources": list(OFFICIAL_SOURCE_WATCHLIST),
        "idea_cards": ideas,
        "source_fingerprints": [idea["source_fingerprint"] for idea in ideas],
        "lane_audit": lane_audit,
        "limits": {
            "max_external_items": max_external_items,
            "max_ideas": max_ideas,
            "auto_backtest": False,
            "db_writes": False,
            "deploy_changes": False,
        },
    }


def build_ai_tooling_lane(
    *,
    external_items: list[dict[str, Any]] | None = None,
    skipped_sources: list[dict[str, Any]] | None = None,
    max_source_items: int = 10,
    max_cards: int = 3,
) -> dict[str, Any]:
    from scripts.tools import ai_tooling_leverage

    return ai_tooling_leverage.build_ai_tooling_leverage(
        external_items=external_items,
        skipped_sources=skipped_sources,
        max_source_items=max_source_items,
        max_cards=max_cards,
    )


def _lane_manifest(report: dict[str, Any]) -> dict[str, Any]:
    lanes: dict[str, dict[str, Any]] = {}
    for lane_name in LANE_NAMES:
        lane_payload = report.get(lane_name) or {}
        lane_audit = lane_payload.get("lane_audit") if isinstance(lane_payload, dict) else None
        lanes[lane_name] = {
            "included": lane_name in report,
            "read_only": True,
            "has_lane_audit": bool(lane_audit),
            "confidence_limiter": lane_audit.get("confidence_limiter") if lane_audit else "not_checked",
        }
    return {
        "lanes": lanes,
        "strong_recommendations_require_lane_audit": True,
        "source_caps": {
            "daily_max_external_items": "bounded by CLI args",
            "daily_max_idea_cards": "bounded by CLI args",
            "daily_max_tooling_cards": "bounded by CLI args",
        },
        "anti_silence_contract": [
            "checked surfaces",
            "skipped surfaces",
            "excluded but relevant surfaces",
            "false-negative sample",
            "counter-framings",
            "disconfirming checks",
            "unchecked scope",
            "residual risk",
        ],
    }


def build_daily_radar(
    *,
    root: Path = PROJECT_ROOT,
    lane: str = "all",
    since: str | None = None,
    hours: int = 24,
    base_ref: str = "origin/main",
    include_local_head: bool = False,
    max_commits: int = 5,
    max_external_items: int = 10,
    max_ideas: int = 3,
    external_items: list[dict[str, Any]] | None = None,
    ai_tooling_items: list[dict[str, Any]] | None = None,
    ai_tooling_skipped_sources: list[dict[str, Any]] | None = None,
    max_tooling_cards: int = 3,
    scan_packet: daily_bug_scan.ScanPacket | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "generated_at": _now_iso(),
        "lane": lane,
        "read_only": True,
    }
    if lane in {"risk", "all"}:
        packet = scan_packet or daily_bug_scan.build_scan_packet(
            root=root,
            since=since,
            hours=hours,
            base_ref=base_ref,
            include_local_head=include_local_head,
            include_working_tree=True,
            max_commits=max_commits,
        )
        report["risk"] = build_risk_lane(root, packet)
    if lane in {"opportunity", "all"}:
        report["opportunity"] = build_opportunity_radar(
            external_items=external_items,
            max_external_items=max_external_items,
            max_ideas=max_ideas,
        )
    if lane in {"ai_tooling", "all"}:
        report["ai_tooling"] = build_ai_tooling_lane(
            external_items=ai_tooling_items,
            skipped_sources=ai_tooling_skipped_sources,
            max_source_items=max_external_items,
            max_cards=max_tooling_cards,
        )
    report["lane_manifest"] = _lane_manifest(report)
    return report


def render_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True, default=str)


def render_text(report: dict[str, Any]) -> str:
    lines = [f"Daily Project Radar | generated_at={report['generated_at']} | lane={report['lane']}"]
    risk = report.get("risk")
    if risk:
        lines.append(f"Risk verdict: {risk['verdict']}")
        findings = risk.get("targeted_behavioral_findings") or []
        lines.append(
            f"Risk candidates: shown={len(risk['scan']['candidate_commits'])} total={risk['scan']['total_candidate_count']}"
        )
        lines.append(f"Capital impact paths: {len(risk['capital_impact_paths'])}")
        if findings:
            lines.append("Targeted behavioral findings:")
            lines.extend(f"- {item['path']}: {item['summary']}" for item in findings[:5])
        audit = risk["audit_of_auditor"]
        lines.append("Unchecked scope:")
        lines.extend(f"- {item}" for item in audit["unchecked_scope"][:5])
    opportunity = report.get("opportunity")
    if opportunity:
        lines.append(f"Opportunity mode: {opportunity['mode']} | ideas={len(opportunity['idea_cards'])}")
        for idea in opportunity["idea_cards"]:
            lines.append(f"- {idea['disposition']}: {idea['claim']} ({idea['source']})")
        lines.append(f"Opportunity confidence limiter: {opportunity['lane_audit']['confidence_limiter']}")
    ai_tooling = report.get("ai_tooling")
    if ai_tooling:
        lines.append(f"AI/tooling mode: {ai_tooling['mode']} | cards={len(ai_tooling['cards'])}")
        for card in ai_tooling["cards"]:
            lines.append(f"- {card['disposition']}: {card['claim']} ({card['vendor']})")
        lines.append(f"AI/tooling confidence limiter: {ai_tooling['lane_audit']['confidence_limiter']}")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=("risk", "opportunity", "ai_tooling", "all"), default="all")
    parser.add_argument("--since")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--base-ref", default="origin/main")
    parser.add_argument("--include-local-head", action="store_true")
    parser.add_argument("--max-commits", type=int, default=5)
    parser.add_argument("--max-external-items", type=int, default=10)
    parser.add_argument("--max-ideas", type=int, default=3)
    parser.add_argument("--max-tooling-cards", type=int, default=3)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_daily_radar(
        root=PROJECT_ROOT,
        lane=args.lane,
        since=args.since,
        hours=args.hours,
        base_ref=args.base_ref,
        include_local_head=args.include_local_head,
        max_commits=args.max_commits,
        max_external_items=args.max_external_items,
        max_ideas=args.max_ideas,
        max_tooling_cards=args.max_tooling_cards,
    )
    print(render_json(report) if args.format == "json" else render_text(report))
    risk = report.get("risk")
    if risk and risk["verdict"] in {"BLOCK", "FIX_REQUIRED", "VERIFY_MORE"}:
        return 2
    ai_tooling = report.get("ai_tooling") or {}
    if any(card.get("disposition") == "BLOCKING_CHANGE" for card in ai_tooling.get("cards", [])):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
