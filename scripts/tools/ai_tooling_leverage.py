#!/usr/bin/env python3
"""Bias-resistant AI/tooling leverage radar.

The module is intentionally read-only. It classifies bounded source items into
local project leverage cards; it does not fetch broad web pages, write caches,
change configs, or run live-capital actions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DISPOSITIONS = ("ADOPT_NOW", "EVALUATE", "WATCH", "IGNORE", "BLOCKING_CHANGE")
IMPACT_TYPES = (
    "cost_reduction",
    "audit_quality",
    "automation_unlock",
    "context_hygiene",
    "model_quality",
    "latency_reduction",
    "security_or_privacy",
    "breaking_change",
)
OFFICIAL_DOMAINS = {
    "openai": ("openai.com", "developers.openai.com", "platform.openai.com", "help.openai.com"),
    "codex": ("openai.com", "developers.openai.com", "platform.openai.com", "help.openai.com"),
    "anthropic": ("anthropic.com", "docs.anthropic.com", "platform.claude.com", "support.claude.com"),
    "claude": ("anthropic.com", "docs.anthropic.com", "platform.claude.com", "support.claude.com"),
}
BREAKING_TOKENS = ("deprecat", "sunset", "shut down", "shutdown", "retire", "legacy", "migration")
COST_TOKENS = ("cost", "spend", "usage", "token", "pricing", "cache", "cached", "batch", "flex", "cheap")
AUDIT_TOKENS = ("audit", "eval", "trace", "review", "quality", "verification", "grader")
AUTOMATION_TOKENS = ("agent", "tool", "mcp", "connector", "codex", "claude code", "automation", "workflow")
CONTEXT_TOKENS = ("context", "compact", "prompt", "docs mcp", "cache retention")
SECURITY_TOKENS = ("security", "privacy", "rbac", "key", "auth", "allowlist", "data residency")
LATENCY_TOKENS = ("latency", "realtime", "priority", "stream")
MODEL_TOKENS = ("model", "gpt", "opus", "sonnet", "haiku", "codex")
STRONG_DISPOSITIONS = {"ADOPT_NOW", "EVALUATE"}


@dataclass(frozen=True)
class ToolingLeverageCard:
    source_url: str
    source_fingerprint: str
    vendor: str
    published_at: str
    claim: str
    local_repo_touchpoints: list[str]
    role: str
    impact_type: str
    disposition: str
    evidence_class: str
    alternative_framings_checked: list[str]
    acceptance_check: str
    disconfirming_check: str
    risks_if_adopted: list[str]
    risks_if_ignored: list[str]
    expires_at: str


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _fingerprint(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def _norm(value: Any) -> str:
    return str(value or "").strip()


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if str(value).strip() else []


def _is_official_source(vendor: str, source_url: str, source_type: str) -> bool:
    if source_type != "official":
        return False
    host = urlparse(source_url).netloc.lower()
    domains = OFFICIAL_DOMAINS.get(vendor.lower(), ())
    return any(host == domain or host.endswith(f".{domain}") for domain in domains)


def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token in text for token in tokens)


def _infer_impact_type(claim: str, role: str) -> str:
    text = f"{claim} {role}".lower()
    if _contains_any(text, BREAKING_TOKENS):
        return "breaking_change"
    if _contains_any(text, COST_TOKENS):
        return "cost_reduction"
    if _contains_any(text, SECURITY_TOKENS):
        return "security_or_privacy"
    if _contains_any(text, CONTEXT_TOKENS):
        return "context_hygiene"
    if _contains_any(text, AUDIT_TOKENS):
        return "audit_quality"
    if _contains_any(text, LATENCY_TOKENS):
        return "latency_reduction"
    if _contains_any(text, AUTOMATION_TOKENS):
        return "automation_unlock"
    if _contains_any(text, MODEL_TOKENS):
        return "model_quality"
    return "automation_unlock"


def _alternative_framings(impact_type: str, role: str) -> list[str]:
    base = [
        f"primary role tested: {role or 'unassigned'}",
        f"could instead be a {impact_type} signal",
        "could be irrelevant without local repo touchpoints",
    ]
    if impact_type != "security_or_privacy":
        base.append("could introduce security or privacy risk if adopted blindly")
    if impact_type != "breaking_change":
        base.append("could hide a migration blocker if ignored")
    return base


def _acceptance_check(disposition: str, official: bool, impact_type: str) -> str:
    if not official:
        return "Confirm with official vendor documentation before any adoption or project-truth claim."
    if disposition == "BLOCKING_CHANGE":
        return "Search repo for affected API/model/tool usage and open a targeted migration or blocker ticket."
    if disposition == "IGNORE":
        return "No action unless a future official source links this item to a local repo touchpoint."
    return f"Run a bounded local eval proving {impact_type} benefit against current repo workflow before adoption."


def _disconfirming_check(disposition: str) -> str:
    if disposition == "IGNORE":
        return "A concrete repo path depends on this capability or a current workflow cost/error traces to it."
    if disposition == "BLOCKING_CHANGE":
        return "Repo search proves no affected API/model/tool usage and vendor docs show no migration deadline."
    return "Local benchmark shows no cost, audit-quality, latency, or automation improvement versus current workflow."


def _base_disposition(*, official: bool, impact_type: str, role: str, touchpoints: list[str]) -> str:
    if role == "not_relevant" or (not touchpoints and impact_type not in {"breaking_change", "security_or_privacy"}):
        return "IGNORE"
    if not official:
        return "WATCH"
    if impact_type == "breaking_change":
        return "BLOCKING_CHANGE"
    return "EVALUATE"


def _downgrade_for_silence(disposition: str, has_downgrading_silence: bool) -> str:
    if has_downgrading_silence and disposition in STRONG_DISPOSITIONS:
        return "WATCH"
    return disposition


def classify_tooling_item(raw: dict[str, Any], *, has_downgrading_silence: bool = False) -> dict[str, Any]:
    source_type = _norm(raw.get("source_type") or "unknown").lower()
    vendor = _norm(raw.get("vendor") or "unknown").lower()
    source_url = _norm(raw.get("source_url") or raw.get("source") or "unknown")
    claim = _norm(raw.get("claim"))
    role = _norm(raw.get("role") or "unassigned").lower()
    published_at = _norm(raw.get("published_at") or "unknown")
    touchpoints = _as_list(raw.get("local_repo_touchpoints"))
    official = _is_official_source(vendor, source_url, source_type)
    impact_type = _infer_impact_type(claim, role)
    disposition = _base_disposition(official=official, impact_type=impact_type, role=role, touchpoints=touchpoints)
    disposition = _downgrade_for_silence(disposition, has_downgrading_silence)
    evidence_class = "MEASURED" if official else "UNSUPPORTED"
    if source_type == "official" and not official:
        evidence_class = "INFERRED"
    expires_at = (datetime.now(UTC) + timedelta(days=30)).date().isoformat()
    card = ToolingLeverageCard(
        source_url=source_url,
        source_fingerprint=_fingerprint(source_type, vendor, source_url, claim),
        vendor=vendor,
        published_at=published_at,
        claim=claim,
        local_repo_touchpoints=touchpoints,
        role=role,
        impact_type=impact_type,
        disposition=disposition,
        evidence_class=evidence_class,
        alternative_framings_checked=_alternative_framings(impact_type, role),
        acceptance_check=_acceptance_check(disposition, official, impact_type),
        disconfirming_check=_disconfirming_check(disposition),
        risks_if_adopted=[
            "workflow churn without measured local improvement",
            "new vendor defaults may weaken existing audit or privacy assumptions",
        ],
        risks_if_ignored=[
            "missed cost, context, or automation efficiency",
            "missed deprecation or migration blocker",
        ],
        expires_at=expires_at,
    )
    return asdict(card)


def _normalize_skipped_sources(skipped_sources: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    ledger: list[dict[str, Any]] = []
    for item in skipped_sources or []:
        ledger.append(
            {
                "source": _norm(item.get("source") or "unknown"),
                "reason": _norm(item.get("reason") or "not specified"),
                "impact": _norm(item.get("impact") or "unknown impact"),
                "downgrades_verdict": bool(item.get("downgrades_verdict")),
            }
        )
    return ledger


def _lane_audit(
    *,
    bounded_count: int,
    card_count: int,
    negative_controls: list[dict[str, Any]],
    silence_ledger: list[dict[str, Any]],
) -> dict[str, Any]:
    downgrading_silence = [item for item in silence_ledger if item["downgrades_verdict"]]
    unchecked_scope = [
        "does not perform broad vendor documentation crawl in daily mode",
        "does not run local model/tool benchmarks automatically",
        "does not change repo config, deployments, DB state, or live trading paths",
    ]
    if not negative_controls:
        unchecked_scope.append("no ignored item was available for false-negative sampling")
    confidence_limiter = "none"
    if downgrading_silence:
        confidence_limiter = "high-impact source skipped"
    elif not negative_controls:
        confidence_limiter = "no negative control available"
    return {
        "checked_surfaces": [
            "bounded source items",
            "official vendor domain gate",
            "local repo touchpoint claims",
            "role-sensitive framing",
            "negative controls",
            "silence ledger",
        ],
        "skipped_surfaces": [
            "live network fetch skipped unless caller supplies source items",
            "full pricing/model benchmark skipped in daily mode",
            "external community/forum sweep skipped in daily mode",
        ],
        "excluded_but_relevant_surfaces": [
            "vendor pricing tables not supplied as source items",
            "local API usage/cost exports not queried",
            "MCP/plugin release feeds beyond supplied source items",
        ],
        "false_negative_sample": negative_controls[:1],
        "counter_framings": [
            "new capability may be useful only as a cost reducer",
            "new capability may be a migration or security risk rather than an upgrade",
            "apparently irrelevant item may matter if a hidden local dependency exists",
        ],
        "disconfirming_checks": [
            "repo search finds no local touchpoint for the claimed capability",
            "official docs contradict the supplied source item",
            "bounded local eval fails to beat current workflow cost or audit quality",
        ],
        "unchecked_scope": unchecked_scope,
        "residual_risk": [
            "daily caps can miss low-frequency vendor deprecations",
            "classification is static and cannot prove runtime performance",
        ],
        "confidence_limiter": confidence_limiter,
        "item_counts": {
            "bounded_source_items": bounded_count,
            "cards": card_count,
            "negative_controls": len(negative_controls),
            "downgrading_silences": len(downgrading_silence),
        },
    }


def build_ai_tooling_leverage(
    *,
    external_items: list[dict[str, Any]] | None = None,
    skipped_sources: list[dict[str, Any]] | None = None,
    max_source_items: int = 10,
    max_cards: int = 3,
) -> dict[str, Any]:
    silence_ledger = _normalize_skipped_sources(skipped_sources)
    has_downgrading_silence = any(item["downgrades_verdict"] for item in silence_ledger)
    bounded_items = (external_items or [])[:max_source_items]
    cards: list[dict[str, Any]] = []
    negative_controls: list[dict[str, Any]] = []
    for item in bounded_items:
        card = classify_tooling_item(item, has_downgrading_silence=has_downgrading_silence)
        if card["disposition"] == "IGNORE":
            negative_controls.append(card)
            continue
        if len(cards) < max_cards:
            cards.append(card)
    return {
        "mode": "report_only",
        "read_only": True,
        "generated_at": _now_iso(),
        "cards": cards,
        "negative_controls": negative_controls[:1],
        "source_fingerprints": [card["source_fingerprint"] for card in [*cards, *negative_controls[:1]]],
        "silence_ledger": silence_ledger,
        "lane_audit": _lane_audit(
            bounded_count=len(bounded_items),
            card_count=len(cards),
            negative_controls=negative_controls,
            silence_ledger=silence_ledger,
        ),
        "limits": {
            "max_source_items": max_source_items,
            "max_cards": max_cards,
            "cheap_local_probes": 0,
            "network_fetch": False,
            "db_writes": False,
            "deploy_changes": False,
            "live_trading_changes": False,
        },
    }


def render_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-source-items", type=int, default=10)
    parser.add_argument("--max-cards", type=int, default=3)
    parser.add_argument("--format", choices=("json",), default="json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_ai_tooling_leverage(max_source_items=args.max_source_items, max_cards=args.max_cards)
    print(render_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
