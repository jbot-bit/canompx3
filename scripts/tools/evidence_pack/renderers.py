"""Markdown + JSON renderers for evidence packs.

Per design doc § 5 + § 10: text-only outputs in PR1. The ``--html`` flag is
reserved but inert; an HTMLRenderer hook is provided as a stub that raises
NotImplementedError to make the deferred state explicit.
"""

from __future__ import annotations

from .manifest import Contamination, GateResult, Manifest, to_json_bytes


def render_manifest_json(manifest: Manifest) -> bytes:
    """Deterministic JSON manifest. Acceptance criterion #1."""

    return to_json_bytes(manifest)


def render_gate_table_json(manifest: Manifest) -> bytes:
    """Twelve-row gate table as a flat JSON array."""

    import json

    rows = [
        {
            "name": g.name,
            "status": g.status,
            "value": g.value,
            "threshold": g.threshold,
            "source": g.source,
            "note": g.note,
        }
        for g in manifest.gates
    ]
    return json.dumps(rows, sort_keys=True, indent=2, ensure_ascii=False).encode("utf-8")


def render_decision_card_md(manifest: Manifest) -> str:
    """One-page Markdown decision card.

    Order mirrors docs/audit/deploy_readiness/2026-04-15-sgp-momentum-deploy-
    readiness.md so auditors do not need to learn a new layout.
    """

    parts: list[str] = []
    parts.append(f"# Evidence Pack — {manifest.slug}")
    parts.append("")

    # Banners (red first, then amber). Verdict drives the red band.
    if manifest.verdict in {"INCOMPLETE_EVIDENCE", "LANE_NOT_SUPPORTED_BY_POOLED"}:
        parts.append(f"> 🛑 **{manifest.verdict}** — verdict downgraded by fail-closed gate.")
        for reason in manifest.verdict_reasons:
            parts.append(f">   - {reason}")
        parts.append("")
    if manifest.contamination.status == "TAINTED":
        parts.append(
            "> 🛑 **CONTAMINATION** — result MD commit appears in the "
            "E2 look-ahead contamination registry. Tainted hits: "
            f"{', '.join(manifest.contamination.hits) or '(unspecified)'}."
        )
        parts.append("")
    if manifest.contamination.registry_status == "MISSING":
        parts.append(
            f"> ⚠️ **CONTAMINATION REGISTRY MISSING** — sorted recursive glob "
            f"`{manifest.contamination.expected_glob}` returned no match. "
            "Contamination check reported as UNCOMPUTED. Verdict not "
            "downgraded. The canonical registry exists on unmerged research "
            "branches `research/2026-04-28-phase-d-mes-europe-flow-pathway-b` "
            "and `research/mnq-unfiltered-high-rr-family` at commit "
            "`96bba7a7` as of 2026-04-28."
        )
        parts.append("")

    # Decision card body
    parts.append("## Decision")
    parts.append("")
    parts.append(f"- **Verdict:** {manifest.verdict}")
    parts.append(f"- **Slug:** {manifest.slug}")
    parts.append(f"- **Generated:** {manifest.run_iso8601}")
    parts.append(f"- **Repo SHA:** `{manifest.git_sha or '(unknown)'}`")
    parts.append(f"- **DB fingerprint:** `{manifest.db_fingerprint or '(unknown)'}`")
    parts.append(f"- **DB path:** `{manifest.db_path}`")
    parts.append(f"- **Holdout:** `{manifest.holdout_date}`")
    parts.append("")

    # Gate table
    parts.append("## Twelve-criteria gate table")
    parts.append("")
    parts.append("| # | Criterion | Status | Value | Threshold | Source |")
    parts.append("|---|---|---|---|---|---|")
    for i, g in enumerate(manifest.gates, start=1):
        parts.append(f"| {i} | {g.name} | {g.status} | {g.value} | {g.threshold} | `{g.source}` |")
    parts.append("")

    # Hypothesis + result + validated_setups
    parts.append("## Sources")
    parts.append("")
    parts.append("**Hypothesis (pre-reg):**")
    for k, v in sorted(manifest.hypothesis.items()):
        parts.append(f"- `{k}`: {v}")
    parts.append("")
    parts.append("**Result MD:**")
    for k, v in sorted(manifest.result.items()):
        parts.append(f"- `{k}`: {v}")
    parts.append("")
    if manifest.validated_setups:
        parts.append("**Validated setups row:**")
        for k, v in sorted(manifest.validated_setups.items()):
            parts.append(f"- `{k}`: {v}")
        parts.append("")

    parts.append("## Reproducibility")
    parts.append("")
    parts.append(f"- Tables used: {', '.join(manifest.tables_used) or '(none)'}")
    parts.append(f"- Queries: {len(manifest.queries)} recorded (see `report.md`).")
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append(
        f"*Pack version {manifest.pack_version}. Derivative artefact only — "
        "decision-ledger entries cite canonical sources, not this pack.*"
    )
    parts.append("")
    return "\n".join(parts)


def render_report_md(manifest: Manifest) -> str:
    """Full pack report. Decision card + per-query SQL/SHA/rerun."""

    parts: list[str] = [render_decision_card_md(manifest), "", "## Queries"]
    parts.append("")
    if not manifest.queries:
        parts.append("_No queries recorded for this pack._")
        parts.append("")
    else:
        for q in manifest.queries:
            parts.append(f"### {q.label}")
            parts.append("")
            parts.append(f"- **SHA-256:** `{q.sql_sha256}`")
            parts.append(f"- **Rerun:** `{q.rerun_command}`")
            parts.append("")
            parts.append("```sql")
            parts.append(q.sql_text.rstrip())
            parts.append("```")
            parts.append("")
    return "\n".join(parts)


def render_html(manifest: Manifest) -> str:
    """Reserved for PR4. Inert in PR1 per design doc § 13."""

    raise NotImplementedError("HTML rendering is reserved for PR4 (see design doc § 13). PR1 ships text-only outputs.")


__all__ = [
    "render_manifest_json",
    "render_gate_table_json",
    "render_decision_card_md",
    "render_report_md",
    "render_html",
]


# Compile-time defensive: keep the import-graph honest so unused-import drift
# checks see why these are referenced.
_ = (Contamination, GateResult)
