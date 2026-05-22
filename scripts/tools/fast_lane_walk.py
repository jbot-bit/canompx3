"""Fast-lane walk orchestrator — read-only end-to-end chain composer.

See ``docs/specs/fast_lane_state_graph.md`` for the canonical fast-lane chain.

Stage 4 of ``docs/plans/2026-05-19-fast-lane-pipeline-connective-tissue-design.md``.

Composes the existing fast-lane writers into one operator-facing command:

1. ``fast_lane_promote_queue.main(["--write"])`` — refresh PROMOTE queue cache.
2. ``cherry_pick_ranker.main(["--write", "--write-journal"])`` — rank QUEUED entries + append journal.
3. ``cherry_pick_journal_enricher.main([])`` — backfill heavyweight verdicts (read-only path).
4. ``fast_lane_status.main(["--write"])`` — rebuild status roll-up.

Then renders an awareness Markdown report to stdout (Connector 4 of the
design): counts-per-stage table, top-3 stalled strategies, ERROR roll-up,
and a next-operator-action footer naming exactly one strategy_id and one
stage to act on.

Capital-class boundary
----------------------
This orchestrator NEVER mutates ``chordia_audit_log.yaml``,
the lane allocation file, ``validated_setups``, or anything under
``trading_app/live/``. It composes scripts that already enforce that
boundary individually (Checks #157, #160, #161, #168). Walk inherits.

Stage 3 (age-staleness view) deferred
-------------------------------------
The roll-up's ``age_days`` field already answers the "what is stale"
question at the strategy_id level. Per-stage transition ages
(queued / ranked / bridged / grounded / enriched) require a separate
writer (Stage 3) that has no documented operator consumer today. The
state-graph node ``fast_lane_age_staleness`` retains ``proposed: true``;
this orchestrator works against the Stage 2A.2 roll-up alone.

Exit codes
----------
0   success
1   argparse / I/O / unhandled exception
2   any composed writer returned non-zero (matches
    ``fast_lane_promote_queue.py`` convention)
"""

from __future__ import annotations

import argparse
import io
import sys
from collections.abc import Callable
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = REPO_ROOT / "docs" / "runtime"
STATUS_ROLLUP_PATH = RUNTIME_DIR / "fast_lane_status.yaml"

SCHEMA_VERSION = 2

SUPPRESSED_QUEUE_STATUSES = {
    "SUPPRESSED_GRAVEYARD",
    "SUPPRESSED_DUPLICATE_ACTIVE",
    "SUPPRESSED_SIBLING_RETEST",
    "SUPPRESSED_BANNED_ENTRY_MODEL",
    "SUPPRESSED_E2_LOOKAHEAD",
    "SUPPRESSED_K_OVERRUN",
}


# ----------------------------------------------------------------------
# Step composition
# ----------------------------------------------------------------------


def _default_step_table() -> list[tuple[str, Callable[[list[str]], int], list[str]]]:
    """Default (label, main, argv) triples for the four chain steps.

    Imports are deferred so the orchestrator can be imported / tested
    without immediately pulling the heavy upstream dependency graph.
    """
    from scripts.research import (
        cherry_pick_journal_enricher,
        cherry_pick_ranker,
        fast_lane_promote_queue,
    )
    from scripts.tools import fast_lane_status

    return [
        ("promote_queue", fast_lane_promote_queue.main, ["--write"]),
        ("cherry_pick_ranker", cherry_pick_ranker.main, ["--write", "--write-journal"]),
        ("journal_enricher", cherry_pick_journal_enricher.main, []),
        ("status_rollup", fast_lane_status.main, ["--write"]),
    ]


def _dry_run_argv(label: str, argv: list[str]) -> list[str]:
    """Return the non-mutating argv for a composed dry-run step."""
    effective_argv = [a for a in argv if not a.startswith("--write")]
    if label == "promote_queue" and "--no-ledger-append" not in effective_argv:
        effective_argv.append("--no-ledger-append")
    return effective_argv


def run_chain(
    steps: list[tuple[str, Callable[[list[str]], int], list[str]]] | None = None,
    *,
    dry_run: bool = False,
) -> tuple[int, list[dict[str, Any]]]:
    """Invoke each composed writer in order; collect (label, rc, captured_stdout).

    Returns
    -------
    (overall_rc, results)
        overall_rc = 2 if any step returned non-zero; 0 otherwise.
        results = list of {"step", "rc", "stdout"} dicts in execution order.
    """
    if steps is None:
        steps = _default_step_table()
    results: list[dict[str, Any]] = []
    overall_rc = 0
    for label, fn, argv in steps:
        effective_argv = list(argv)
        if dry_run:
            effective_argv = _dry_run_argv(label, effective_argv)
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                rc = fn(effective_argv)
            except SystemExit as exc:
                rc = int(exc.code) if isinstance(exc.code, int) else 1
            except Exception as exc:
                rc = 1
                buf.write(f"\n[orchestrator] {label} raised {type(exc).__name__}: {exc}\n")
        if rc != 0:
            overall_rc = 2
        results.append({"step": label, "rc": rc, "stdout": buf.getvalue(), "argv": effective_argv})
    return overall_rc, results


# ----------------------------------------------------------------------
# Report rendering (Connector 4)
# ----------------------------------------------------------------------


def _load_rollup(path: Path = STATUS_ROLLUP_PATH) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError):
        return None


def _counts_per_stage(entries: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in entries:
        stage = e.get("current_stage")
        if isinstance(stage, str):
            counts[stage] = counts.get(stage, 0) + 1
    return counts


def _is_fast_lane_automation_actionable(entry: dict[str, Any]) -> bool:
    stage = entry.get("current_stage")
    if not isinstance(stage, str):
        return False
    if stage == "ERROR":
        return True
    if entry.get("lineage_class") == "DIRECT_HEAVYWEIGHT":
        return False
    if stage == "HEAVYWEIGHT_COMPLETE":
        return entry.get("next_action_token") == "run_cherry_pick_journal_enricher"
    terminal = {"REVOKED", "PARKED", "REJECTED_OOS_UNPOWERED", *SUPPRESSED_QUEUE_STATUSES}
    return stage not in terminal


def _top_stalled(entries: list[dict[str, Any]], k: int = 3) -> list[dict[str, Any]]:
    """Top-K Fast Lane entries by age_days. Excludes terminal and direct-heavyweight rows."""
    actionable = [e for e in entries if _is_fast_lane_automation_actionable(e) and e.get("current_stage") != "ERROR"]
    actionable.sort(key=lambda e: (-(e.get("age_days") or 0), e.get("strategy_id") or ""))
    return actionable[:k]


def _error_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in entries if e.get("current_stage") == "ERROR"]


def _blocked_fast_lane_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocked = [
        e
        for e in entries
        if e.get("lineage_class") == "FAST_LANE" and e.get("blocker_class") not in (None, "NONE")
    ]
    blocked.sort(key=lambda e: (e.get("blocker_class") or "", e.get("strategy_id") or ""))
    return blocked


def _direct_heavyweight_backlog(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    backlog = [e for e in entries if e.get("lineage_class") == "DIRECT_HEAVYWEIGHT"]
    backlog.sort(key=lambda e: (-(e.get("age_days") or 0), e.get("strategy_id") or ""))
    return backlog


def _next_operator_action(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick exactly one strategy_id + stage for the operator to act on next.

    Priority (per design § 2.5 Connector 4):
    1. Highest-age PROMOTE_QUEUED with no bridge draft (=> run ranker).
    2. Highest-age RANKED with no bridge draft (=> run bridge).
    3. Highest-age BRIDGED (=> author grounded sibling OR accept strict-t).
    4. Highest-age GROUNDED (=> operator promotes draft to active).
    5. Highest-age HEAVYWEIGHT_PENDING (=> run strict-unlock).
    6. Highest-age HEAVYWEIGHT_COMPLETE without journal verdict (=> enricher).

    Returns None when nothing is actionable.
    """
    by_stage: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        stage = e.get("current_stage")
        if not isinstance(stage, str):
            continue
        if not _is_fast_lane_automation_actionable(e):
            continue
        by_stage.setdefault(stage, []).append(e)

    def _oldest(stage: str) -> dict[str, Any] | None:
        lst = by_stage.get(stage) or []
        if not lst:
            return None
        return max(lst, key=lambda e: (e.get("age_days") or 0, e.get("strategy_id") or ""))

    for stage in (
        "ERROR",
        "PROMOTE_QUEUED",
        "RANKED",
        "BRIDGED",
        "GROUNDED",
        "HEAVYWEIGHT_PENDING",
        "HEAVYWEIGHT_COMPLETE",
    ):
        candidate = _oldest(stage)
        if candidate is not None:
            return candidate
    return None


def render_report(
    rollup: dict[str, Any] | None,
    chain_results: list[dict[str, Any]],
    *,
    today: date | None = None,
) -> str:
    """Render the awareness Markdown report.

    Five Connector 4 surfaces:
    1. Banner (schema_version, generated_at, source).
    2. Chain step status (rc per step).
    3. Counts per stage.
    4. Top-3 stalled.
    5. ERROR roll-up.
    6. Next-operator-action footer.
    """
    today_d = today if today is not None else date.today()
    lines: list[str] = []
    lines.append(f"# Fast-lane walk report — {today_d.isoformat()}")
    lines.append("")
    lines.append(f"schema_version: {SCHEMA_VERSION}")
    lines.append("source: scripts/tools/fast_lane_walk.py")
    lines.append("")

    lines.append("## Chain steps")
    lines.append("")
    lines.append("| step | rc |")
    lines.append("|---|---|")
    for r in chain_results:
        lines.append(f"| {r['step']} | {r['rc']} |")
    lines.append("")

    if rollup is None or not isinstance(rollup, dict):
        lines.append("**ERROR:** status roll-up missing or unparseable; no further surfaces rendered.")
        return "\n".join(lines) + "\n"

    entries = rollup.get("entries") or []
    if not isinstance(entries, list):
        entries = []

    counts = _counts_per_stage(entries)
    lines.append("## Counts per stage")
    lines.append("")
    lines.append("| stage | n |")
    lines.append("|---|---|")
    for stage in sorted(counts):
        lines.append(f"| {stage} | {counts[stage]} |")
    lines.append(f"| **total** | **{sum(counts.values())}** |")
    lines.append("")

    stalled = _top_stalled(entries, k=3)
    lines.append("## Top-3 stalled (actionable stages only)")
    lines.append("")
    if stalled:
        lines.append("| strategy_id | stage | age_days | next_action |")
        lines.append("|---|---|---|---|")
        for e in stalled:
            lines.append(
                f"| {e.get('strategy_id', '?')} | {e.get('current_stage', '?')} "
                f"| {e.get('age_days', 0)} | {e.get('next_action_token', '?')} |"
            )
    else:
        lines.append("_None — every entry is in a terminal stage._")
    lines.append("")

    errors = _error_entries(entries)
    lines.append("## ERROR roll-up")
    lines.append("")
    if errors:
        lines.append("| strategy_id | upstream_artifact |")
        lines.append("|---|---|")
        for e in errors:
            lines.append(f"| {e.get('strategy_id', '?')} | {e.get('upstream_artifact_path') or '?'} |")
    else:
        lines.append("_zero ERROR entries — chain is internally consistent._")
    lines.append("")

    blocked = _blocked_fast_lane_entries(entries)
    lines.append("## Blocked Fast Lane candidates")
    lines.append("")
    if blocked:
        lines.append("| strategy_id | stage | blocker_class | primary_blocker | evidence |")
        lines.append("|---|---|---|---|---|")
        for e in blocked:
            evidence = e.get("blocker_evidence") or {}
            if isinstance(evidence, dict):
                evidence_bits = ", ".join(str(k) for k in sorted(evidence)[:4]) or "-"
            else:
                evidence_bits = "-"
            lines.append(
                f"| {e.get('strategy_id', '?')} | {e.get('current_stage', '?')} "
                f"| {e.get('blocker_class', '?')} | {e.get('primary_blocker') or '-'} "
                f"| {evidence_bits} |"
            )
    else:
        lines.append("_zero blocked Fast Lane candidates._")
    lines.append("")

    direct_backlog = _direct_heavyweight_backlog(entries)
    lines.append("## Direct heavyweight backlog")
    lines.append("")
    if direct_backlog:
        lines.append("| strategy_id | stage | age_days | next_action |")
        lines.append("|---|---|---|---|")
        for e in direct_backlog[:10]:
            lines.append(
                f"| {e.get('strategy_id', '?')} | {e.get('current_stage', '?')} "
                f"| {e.get('age_days', 0)} | {e.get('next_action_token', '?')} |"
            )
    else:
        lines.append("_zero direct heavyweight backlog entries._")
    lines.append("")

    nxt = _next_operator_action(entries)
    lines.append("## Next operator action")
    lines.append("")
    if nxt is not None:
        lines.append(
            f"**strategy_id:** `{nxt.get('strategy_id', '?')}`  \n"
            f"**stage:** `{nxt.get('current_stage', '?')}`  \n"
            f"**next_action:** `{nxt.get('next_action_token', '?')}`  \n"
            f"**age_days:** {nxt.get('age_days', 0)}"
        )
    else:
        lines.append("`NO_FAST_LANE_ACTIONABLE` — no queued, ranked, bridged, pending, or error Fast Lane entry.")
    lines.append("")

    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fast_lane_walk",
        description=(
            "End-to-end read-only walk over the fast-lane chain. "
            "Composes existing writers; emits an awareness Markdown "
            "report to stdout. Returns 2 on any upstream non-zero rc."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Strip --write flags from every composed writer; chain runs in non-mutating mode. Caches are NOT refreshed."
        ),
    )
    p.add_argument(
        "--write-report",
        action="store_true",
        help=(
            "Also write the rendered report to "
            "docs/runtime/fast_lane_walk_<date>.md. Without this flag the "
            "report goes only to stdout."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    overall_rc, results = run_chain(dry_run=args.dry_run)
    rollup = _load_rollup()
    today_d = date.today()
    report = render_report(rollup, results, today=today_d)
    sys.stdout.write(report)
    if args.write_report:
        out = RUNTIME_DIR / f"fast_lane_walk_{today_d.isoformat()}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        try:
            display = str(out.relative_to(REPO_ROOT))
        except ValueError:
            display = str(out)
        sys.stdout.write(f"\nwrote report: {display}\n")
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
