"""Tests for scripts/tools/fast_lane_walk.py (Stage 2A.3 — chain orchestrator)."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path
from typing import Any

from typing import Callable

from scripts.tools.fast_lane_walk import (
    SCHEMA_VERSION,
    _counts_per_stage,
    _error_entries,
    _next_operator_action,
    _top_stalled,
    main,
    render_report,
    run_chain,
)


# ----------------------------------------------------------------------
# Pure-function tests (Connector 4 surfaces)
# ----------------------------------------------------------------------


def _entry(sid: str, stage: str, age: int = 0, action: str = "noop") -> dict[str, Any]:
    return {
        "strategy_id": sid,
        "current_stage": stage,
        "age_days": age,
        "next_action_token": action,
        "upstream_artifact_path": None,
        "downstream_artifact_path": None,
        "observed_at": {},
    }


def test_counts_per_stage_basic() -> None:
    entries = [
        _entry("A", "ACTIVE_PREREG"),
        _entry("B", "ACTIVE_PREREG"),
        _entry("C", "REVOKED"),
        _entry("D", "HEAVYWEIGHT_COMPLETE"),
    ]
    counts = _counts_per_stage(entries)
    assert counts == {"ACTIVE_PREREG": 2, "REVOKED": 1, "HEAVYWEIGHT_COMPLETE": 1}


def test_top_stalled_excludes_terminal_stages() -> None:
    entries = [
        _entry("OLD_REVOKED", "REVOKED", age=999),
        _entry("OLD_PARKED", "PARKED", age=999),
        _entry("OLD_ERROR", "ERROR", age=999),
        _entry("OLD_REJECTED", "REJECTED_OOS_UNPOWERED", age=999),
        _entry("YOUNG_ACTIVE", "ACTIVE_PREREG", age=1),
        _entry("OLD_ACTIVE", "ACTIVE_PREREG", age=10),
    ]
    out = _top_stalled(entries, k=3)
    assert [e["strategy_id"] for e in out] == ["OLD_ACTIVE", "YOUNG_ACTIVE"]
    # Terminal entries excluded even with extreme age.


def test_top_stalled_sorts_descending_age_then_strategy_id() -> None:
    entries = [
        _entry("Z", "ACTIVE_PREREG", age=5),
        _entry("A", "ACTIVE_PREREG", age=5),
        _entry("M", "ACTIVE_PREREG", age=5),
    ]
    out = _top_stalled(entries, k=3)
    # All same age — tiebreak by strategy_id ascending.
    assert [e["strategy_id"] for e in out] == ["A", "M", "Z"]


def test_error_entries_filter() -> None:
    entries = [_entry("A", "ACTIVE_PREREG"), _entry("B", "ERROR")]
    out = _error_entries(entries)
    assert len(out) == 1 and out[0]["strategy_id"] == "B"


def test_next_operator_action_priority_promote_queued_first() -> None:
    """PROMOTE_QUEUED outranks RANKED outranks BRIDGED etc."""
    entries = [
        _entry("OLD_BRIDGED", "BRIDGED", age=100),
        _entry("YOUNG_PROMOTE", "PROMOTE_QUEUED", age=1),
        _entry("OLD_RANKED", "RANKED", age=50),
    ]
    nxt = _next_operator_action(entries)
    assert nxt is not None and nxt["strategy_id"] == "YOUNG_PROMOTE"


def test_next_operator_action_within_stage_oldest_wins() -> None:
    entries = [
        _entry("YOUNG", "PROMOTE_QUEUED", age=1),
        _entry("OLD", "PROMOTE_QUEUED", age=10),
        _entry("MID", "PROMOTE_QUEUED", age=5),
    ]
    nxt = _next_operator_action(entries)
    assert nxt is not None and nxt["strategy_id"] == "OLD"


def test_next_operator_action_returns_none_when_only_terminal() -> None:
    entries = [
        _entry("A", "REVOKED"),
        _entry("B", "PARKED"),
        _entry("C", "REJECTED_OOS_UNPOWERED"),
    ]
    assert _next_operator_action(entries) is None


# ----------------------------------------------------------------------
# render_report
# ----------------------------------------------------------------------


def test_render_report_contains_all_five_surfaces() -> None:
    entries = [
        _entry("MNQ_X", "PROMOTE_QUEUED", age=3, action="run_cherry_pick_ranker"),
        _entry("MNQ_Y", "ACTIVE_PREREG", age=1, action="run_fast_lane"),
        _entry("MNQ_E", "ERROR", age=0, action="operator_resolve_error"),
    ]
    rollup = {"schema_version": 1, "entries": entries}
    chain_results = [
        {"step": "promote_queue", "rc": 0, "stdout": "", "argv": ["--write"]},
        {"step": "status_rollup", "rc": 0, "stdout": "", "argv": ["--write"]},
    ]
    out = render_report(rollup, chain_results, today=date(2026, 5, 19))
    assert "Fast-lane walk report — 2026-05-19" in out
    assert "## Chain steps" in out
    assert "| promote_queue | 0 |" in out
    assert "## Counts per stage" in out
    assert "PROMOTE_QUEUED" in out and "ACTIVE_PREREG" in out and "ERROR" in out
    assert "## Top-3 stalled" in out
    assert "MNQ_X" in out  # actionable, age=3
    assert "## ERROR roll-up" in out
    assert "MNQ_E" in out
    assert "## Next operator action" in out
    assert "MNQ_X" in out  # highest-priority next action


def test_render_report_handles_missing_rollup() -> None:
    chain_results = [{"step": "promote_queue", "rc": 1, "stdout": "", "argv": []}]
    out = render_report(None, chain_results, today=date(2026, 5, 19))
    assert "ERROR" in out
    assert "missing or unparseable" in out
    # Still shows chain step status above the fold.
    assert "promote_queue" in out


def test_render_report_zero_actionable_says_so() -> None:
    rollup = {"entries": [_entry("A", "REVOKED")]}
    chain_results = [{"step": "status_rollup", "rc": 0, "stdout": "", "argv": []}]
    out = render_report(rollup, chain_results, today=date(2026, 5, 19))
    assert "no actionable entry" in out


# ----------------------------------------------------------------------
# run_chain
# ----------------------------------------------------------------------


def test_run_chain_propagates_non_zero_rc() -> None:
    def good(argv: list[str]) -> int:
        return 0

    def bad(argv: list[str]) -> int:
        return 1

    steps = [("good", good, []), ("bad", bad, []), ("good_again", good, [])]
    overall_rc, results = run_chain(steps=steps)
    assert overall_rc == 2  # any non-zero => 2
    rcs = [r["rc"] for r in results]
    assert rcs == [0, 1, 0]


def test_run_chain_catches_exception_as_rc_1() -> None:
    def boom(argv: list[str]) -> int:
        raise RuntimeError("kaboom")

    steps = [("boom", boom, [])]
    overall_rc, results = run_chain(steps=steps)
    assert overall_rc == 2
    assert results[0]["rc"] == 1
    assert "RuntimeError" in results[0]["stdout"]
    assert "kaboom" in results[0]["stdout"]


def test_run_chain_dry_run_strips_write_flags() -> None:
    seen_argv: list[list[str]] = []

    def capture(argv: list[str]) -> int:
        seen_argv.append(list(argv))
        return 0

    steps = [
        ("s1", capture, ["--write"]),
        ("s2", capture, ["--write", "--write-journal", "--other"]),
        ("s3", capture, []),
    ]
    overall_rc, _ = run_chain(steps=steps, dry_run=True)
    assert overall_rc == 0
    assert seen_argv == [[], ["--other"], []]


def test_run_chain_zero_on_all_pass() -> None:
    pass_fn: Callable[[list[str]], int] = lambda argv: 0
    steps: list[tuple[str, Callable[[list[str]], int], list[str]]] = [
        ("a", pass_fn, []),
        ("b", pass_fn, []),
    ]
    overall_rc, results = run_chain(steps=steps)
    assert overall_rc == 0
    assert all(r["rc"] == 0 for r in results)


# ----------------------------------------------------------------------
# Idempotency + main() smoke
# ----------------------------------------------------------------------


def test_main_dry_run_does_not_mutate_capital_class_files(tmp_path: Path) -> None:
    """Smoke: main(['--dry-run']) returns 0/2 and emits a report; capital-class
    state is greppable in the writer source — not asserted here as that's
    Check #168's job (already covered by Stage 2A.2).
    """
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["--dry-run"])
    out = buf.getvalue()
    assert rc in (0, 2)
    assert "Fast-lane walk report" in out
    assert f"schema_version: {SCHEMA_VERSION}" in out


def test_main_two_back_to_back_dry_runs_match_modulo_date() -> None:
    """Acceptance #6: idempotency — back-to-back dry-runs produce identical
    Markdown modulo the date stamp.
    """

    def _run() -> str:
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["--dry-run"])
        return buf.getvalue()

    first = _run()
    second = _run()

    def _strip_date(text: str) -> str:
        lines = []
        for line in text.splitlines():
            if line.startswith("# Fast-lane walk report"):
                lines.append("# Fast-lane walk report — <date>")
            else:
                lines.append(line)
        return "\n".join(lines)

    assert _strip_date(first) == _strip_date(second)


def test_main_writes_optional_report_file(tmp_path: Path, monkeypatch: Any) -> None:
    """Acceptance: --write-report emits a Markdown file at the canonical path."""
    from scripts.tools import fast_lane_walk as mod

    monkeypatch.setattr(mod, "RUNTIME_DIR", tmp_path)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["--dry-run", "--write-report"])
    assert rc in (0, 2)
    written = list(tmp_path.glob("fast_lane_walk_*.md"))
    assert len(written) == 1, written
    body = written[0].read_text(encoding="utf-8")
    assert "Fast-lane walk report" in body
