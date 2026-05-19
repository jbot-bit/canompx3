"""Cherry-pick journal enricher — closes the feedback loop from heavyweight verdicts back to ranker entries.

When the cherry-pick ranker writes a journal entry, ``heavyweight_verdict``,
``t_observed_post_clustered_se``, and ``lesson_label`` are deliberately null —
no heavyweight Chordia replay has run yet. Once a strict-unlock result MD
lands in ``docs/audit/results/``, this enricher walks the journal, matches
each open entry to the most recent result MD for the same ``strategy_id``,
and fills in the post-decision fields.

Read-only over ``docs/audit/results/`` and ``docs/runtime/promote_queue.yaml``;
write-only to ``docs/runtime/cherry_pick_journal.yaml``. No mutation of any
capital-class file (``chordia_audit_log.yaml``, ``lane_allocation.json``,
``validated_setups``, ``trading_app/live/*``) — those are operator-only.

Doctrine grounding
------------------
- Plan: ``C:/Users/joshd/.claude/plans/or-linknin-them-togehr-delegated-gizmo.md``
- Stage: ``docs/runtime/stages/2026-05-19-cherry-pick-journal-feedback-loop.md``
- Methodology: ``.claude/rules/backtesting-methodology.md`` § RULE 3.3
  (OOS power floor — the journal's ``oos_power_tier`` is the upstream filter
  that justifies whether a verdict is interpretable)
- Class-bug defense: ``feedback_canonical_inline_copy_parity_bug_class.md``
  — verdict strings + result MD path patterns are inlined here, registered
  in ``pipeline.canonical_inline_copies`` only when n>=3 (per
  ``feedback_n3_same_class_doctrine_threshold.md``).
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "docs" / "audit" / "results"
JOURNAL_PATH = REPO_ROOT / "docs" / "runtime" / "cherry_pick_journal.yaml"

# Heavyweight result MDs are produced by research/chordia_strict_unlock_v1.py
# and stamped with this name pattern. Capturing it as a constant keeps the
# enricher's scope tight: not every result MD under docs/audit/results/ is a
# heavyweight Chordia verdict candidate.
HEAVYWEIGHT_FILENAME_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}-.*?-chordia-(?:unlock|heavyweight)-v\d+\.md$"
)

# Verdict line from the result MD header: ``**MEASURED verdict:** `PASS_CHORDIA```.
# Tolerant of either backtick wrapping or bare token following the colon.
_VERDICT_RE = re.compile(
    r"\*\*MEASURED verdict:\*\*\s*`?(?P<verdict>[A-Z_]+)`?",
)

# Strategy ID line from the result MD title: ``# Chordia strict unlock audit — MNQ_...``.
# Tolerant of an em-dash or hyphen separator before the strategy_id token.
_TITLE_RE = re.compile(
    r"^#\s+Chordia\s+(?:strict\s+unlock|heavyweight)\s+audit\s*[—-]\s*(?P<sid>[A-Z0-9_.]+)",
    re.MULTILINE,
)

# IS-row of the split-summary table — used to extract the clustered-SE t-stat
# the strict runner emits. The header row format is canonical:
#   | Split | N_universe | N_fired | Fire% | Scratch | Null non-scratch
#   | ExpR | Policy EV/opp | Sharpe | t | p_two |
# We anchor on ``| IS |`` and pull the 10th cell (1-indexed) as ``t``.
_IS_ROW_RE = re.compile(
    r"^\|\s*IS\s*\|"
    r"\s*[-+0-9.nNaA]+\s*\|"  # N_universe
    r"\s*[-+0-9.nNaA]+\s*\|"  # N_fired
    r"\s*[-+0-9.nNaA]+%?\s*\|"  # Fire%
    r"\s*[-+0-9.nNaA]+\s*\|"  # Scratch
    r"\s*[-+0-9.nNaA]+\s*\|"  # Null non-scratch
    r"\s*[-+0-9.eEnNaA]+\s*\|"  # ExpR
    r"\s*[-+0-9.eEnNaA]+\s*\|"  # Policy EV/opp
    r"\s*[-+0-9.eEnNaA]+\s*\|"  # Sharpe
    r"\s*(?P<t>[-+0-9.eEnNaA]+)\s*\|",
    re.MULTILINE,
)

# Map raw verdict tokens to the journal's controlled vocabulary. Anything
# unknown lands as ``UNKNOWN`` rather than guessing — feedback_chordia_oos_park_vs_unverified_power_floor.md
# class trap (PARK from runner can be UNVERIFIED in disguise).
VERDICT_MAP: dict[str, str] = {
    "PASS_CHORDIA": "PASS_CHORDIA",
    "FAIL_STRICT": "FAIL_STRICT",
    "FAIL_CHORDIA": "FAIL_STRICT",
    "PARK": "PARK",
    "DEFERRED_NOT_RUN": "DEFERRED_NOT_RUN",
}


@dataclass(frozen=True)
class HeavyweightOutcome:
    """Parsed verdict from a heavyweight Chordia strict-unlock result MD."""

    strategy_id: str
    verdict: str
    t_clustered: float | None
    result_md_path: str  # repo-relative


def _parse_float_or_none(s: str) -> float | None:
    s = s.strip().lower()
    if s in {"nan", "", "n/a"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_heavyweight_result(md_path: Path) -> HeavyweightOutcome | None:
    """Extract verdict + clustered-SE t from a strict-unlock result MD.

    Returns None when the file lacks a parseable strategy_id, verdict, or
    IS row — fail-quiet on the parse side because not every MD under
    docs/audit/results/ is shaped like a Chordia heavyweight verdict (memos,
    rollups, audit-of-audit docs).
    """
    if not md_path.exists():
        return None
    text = md_path.read_text(encoding="utf-8")

    title = _TITLE_RE.search(text)
    if title is None:
        return None

    verdict_m = _VERDICT_RE.search(text)
    if verdict_m is None:
        return None
    raw_verdict = verdict_m.group("verdict")
    mapped = VERDICT_MAP.get(raw_verdict, "UNKNOWN")

    is_row = _IS_ROW_RE.search(text)
    t_clustered = _parse_float_or_none(is_row.group("t")) if is_row else None

    rel = (
        md_path.relative_to(REPO_ROOT)
        if md_path.is_relative_to(REPO_ROOT)
        else md_path
    )
    return HeavyweightOutcome(
        strategy_id=title.group("sid"),
        verdict=mapped,
        t_clustered=t_clustered,
        result_md_path=str(rel).replace("\\", "/"),
    )


def collect_heavyweight_outcomes(results_dir: Path) -> dict[str, HeavyweightOutcome]:
    """Map strategy_id -> most recent parsed outcome under results_dir.

    "Most recent" = lexicographically last filename (canonical date prefix
    ``YYYY-MM-DD-...``). If two MDs share the same strategy_id, the later
    date wins.
    """
    if not results_dir.exists():
        return {}
    outcomes: dict[str, HeavyweightOutcome] = {}
    for md in sorted(results_dir.glob("*.md")):
        if not HEAVYWEIGHT_FILENAME_PATTERN.match(md.name):
            continue
        parsed = parse_heavyweight_result(md)
        if parsed is None:
            continue
        # Later filename (lex order) overwrites earlier — preserves latest verdict.
        outcomes[parsed.strategy_id] = parsed
    return outcomes


def _derive_lesson_label(
    entry: dict[str, Any], outcome: HeavyweightOutcome
) -> str:
    """Best-effort categorical lesson based on entry context + heavyweight verdict.

    Returns a short canonical token suitable for ``lesson_label`` when the
    enricher fills in a verdict. Operator may override via hand-edit.

    The mapping is deliberately coarse — long-form prose lessons live in
    ``cherry_pick_journal.md``, not in this enricher.
    """
    verdict = outcome.verdict
    if verdict == "PASS_CHORDIA":
        return "T_HELD_AFTER_CLUSTERED_SE"
    if verdict == "FAIL_STRICT":
        # Headroom-driven failure trumps everything else if we entered with no
        # headroom. The deflation_headroom signal in components is the operator's
        # honest pre-test expectation; reflect that back in the lesson.
        comps = entry.get("components") or {}
        if comps.get("deflation_headroom", 0.0) == 0.0:
            return "T_DEFLATION_KILLED_AT_NO_HEADROOM"
        return "T_DEFLATION_KILLED_DESPITE_HEADROOM"
    if verdict == "PARK":
        # See feedback_chordia_oos_park_vs_unverified_power_floor.md — PARK
        # on underpowered OOS is methodologically UNVERIFIED, not a kill.
        if entry.get("oos_power_tier") in (
            "NA_NO_OOS",
            "NA_N_BELOW_FLOOR",
            "STATISTICALLY_USELESS",
        ):
            return "PARK_BUT_OOS_UNDERPOWERED"
        return "PARK_OOS_SIGN_FLIP"
    if verdict == "DEFERRED_NOT_RUN":
        return "DEFERRED_AWAITING_OOS_ACCRUAL"
    return "VERDICT_UNRECOGNIZED"


def enrich_entries(
    journal: dict[str, Any],
    outcomes: dict[str, HeavyweightOutcome],
) -> list[dict[str, Any]]:
    """Fill in heavyweight_verdict / t_observed / lesson_label for matching entries.

    Returns the list of entry dicts that were mutated (for stdout reporting).
    Mutation is in-place on the journal payload.
    """
    mutated: list[dict[str, Any]] = []
    for entry in journal.get("entries", []):
        if not isinstance(entry, dict):
            continue
        if entry.get("heavyweight_verdict") not in (None, "DEFERRED_NOT_RUN"):
            # Already resolved by a prior enricher run or operator hand-edit.
            continue
        sid = entry.get("strategy_id")
        if not isinstance(sid, str):
            continue
        outcome = outcomes.get(sid)
        if outcome is None:
            continue
        # Don't downgrade an existing DEFERRED_NOT_RUN to itself.
        if (
            entry.get("heavyweight_verdict") == "DEFERRED_NOT_RUN"
            and outcome.verdict == "DEFERRED_NOT_RUN"
        ):
            continue

        entry["heavyweight_verdict"] = outcome.verdict
        entry["t_observed_post_clustered_se"] = (
            round(outcome.t_clustered, 4)
            if outcome.t_clustered is not None
            else None
        )
        # Only auto-populate lesson_label when it's null — preserve hand-edits.
        if entry.get("lesson_label") in (None, ""):
            entry["lesson_label"] = _derive_lesson_label(entry, outcome)
        mutated.append(entry)
    return mutated


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cherry_pick_journal_enricher",
        description=(
            "Enrich docs/runtime/cherry_pick_journal.yaml entries with heavyweight "
            "Chordia verdicts parsed from docs/audit/results/. Read-only by default; "
            "--write applies updates."
        ),
    )
    p.add_argument(
        "--journal",
        type=Path,
        default=JOURNAL_PATH,
        help="Path to cherry_pick_journal.yaml (default: docs/runtime/cherry_pick_journal.yaml).",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Path to heavyweight result MDs (default: docs/audit/results/).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be updated without writing. Default behaviour.",
    )
    p.add_argument(
        "--write",
        action="store_true",
        help="Persist mutations back to the journal file.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if not args.journal.exists():
        print(f"Journal not found at {args.journal} — nothing to enrich.")
        return 0

    journal_text = args.journal.read_text(encoding="utf-8")
    journal = yaml.safe_load(journal_text) or {}
    if not isinstance(journal, dict) or not isinstance(journal.get("entries"), list):
        print(f"Journal at {args.journal} is malformed — refusing to write.", file=sys.stderr)
        return 2

    outcomes = collect_heavyweight_outcomes(args.results_dir)
    mutated = enrich_entries(journal, outcomes)

    if not mutated:
        print("No entries to enrich (every open entry either lacks a matching result MD or is already resolved).")
        return 0

    for e in mutated:
        print(
            f"iter={e.get('iter')} strategy_id={e.get('strategy_id')} "
            f"-> heavyweight_verdict={e.get('heavyweight_verdict')} "
            f"t_clustered={e.get('t_observed_post_clustered_se')} "
            f"lesson_label={e.get('lesson_label')}"
        )

    if args.write and not args.dry_run:
        args.journal.write_text(
            yaml.safe_dump(journal, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        rel = (
            args.journal.relative_to(REPO_ROOT)
            if args.journal.is_relative_to(REPO_ROOT)
            else args.journal
        )
        print(f"\nWrote {len(mutated)} update(s) to {rel}")
    else:
        print(f"\nDRY RUN — {len(mutated)} update(s) NOT persisted. Pass --write to apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
