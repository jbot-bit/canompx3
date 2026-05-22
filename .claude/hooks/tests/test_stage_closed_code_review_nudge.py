"""Tests for `.claude/hooks/stage-closed-code-review-nudge.py`.

PostToolUse(Edit|Write) nudge. Drives the hook via subprocess + stdin JSON
just as Claude Code invokes it. Each test materializes a stage file in a
tempdir and points the hook at it via the standard PostToolUse event shape
(``tool_input.file_path``). Asserts on (returncode == 0, stdout non-empty vs
empty), since the hook never blocks — it just decides whether to emit a
nudge line to stdout (additionalContext).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / "stage-closed-code-review-nudge.py"


def _run_hook(
    file_path: Path,
    scratch_dir: Path,
) -> subprocess.CompletedProcess:
    payload = json.dumps(
        {"tool_name": "Edit", "tool_input": {"file_path": str(file_path)}}
    )
    env = {**os.environ, "STAGE_REVIEW_SCRATCH_DIR": str(scratch_dir)}
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=payload,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
        env=env,
    )


@pytest.fixture
def stages_dir(tmp_path: Path) -> Path:
    d = tmp_path / "docs" / "runtime" / "stages"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def scratch_dir(tmp_path: Path) -> Path:
    d = tmp_path / "scratch"
    d.mkdir()
    return d


def _write_stage(
    stages_dir: Path,
    slug: str,
    *,
    mode: str,
    scope_lock: list[str],
    body_extra: str = "",
) -> Path:
    """Write a minimal stage file with YAML front-matter."""
    scope_yaml = "\n".join(f"  - {p}" for p in scope_lock)
    content = (
        f"---\n"
        f"task: |\n"
        f"  Test stage for {slug}\n"
        f"mode: {mode}\n"
        f"scope_lock:\n{scope_yaml}\n"
        f"\n"
        f"## Blast Radius\n"
        f"- {scope_lock[0] if scope_lock else 'noop'} — test fixture\n"
        f"\n"
        f"{body_extra}\n"
        f"---\n"
    )
    path = stages_dir / f"{slug}.md"
    path.write_text(content, encoding="utf-8")
    return path


def test_closed_capital_clean_body_emits_nudge(stages_dir: Path, scratch_dir: Path):
    stage = _write_stage(
        stages_dir,
        "2026-05-23-test-closed",
        mode="CLOSED",
        scope_lock=["trading_app/live/foo.py"],
    )
    res = _run_hook(stage, scratch_dir)
    assert res.returncode == 0, res.stderr
    assert "stage-closed code-review nudge" in res.stdout
    assert "2026-05-23-test-closed" in res.stdout
    assert "trading_app/live/foo.py" in res.stdout
    assert "/capital-review" in res.stdout  # truth-layer routes to capital-review


def test_closed_pipeline_only_routes_to_code_review(stages_dir: Path, scratch_dir: Path):
    stage = _write_stage(
        stages_dir,
        "2026-05-23-pipeline-closed",
        mode="CLOSED",
        scope_lock=["pipeline/check_drift.py"],
    )
    res = _run_hook(stage, scratch_dir)
    assert res.returncode == 0
    assert "[stage-closed code-review nudge]" in res.stdout
    assert "/code-review" in res.stdout
    assert "/capital-review" not in res.stdout


def test_closed_noncapital_scope_emits_nothing(stages_dir: Path, scratch_dir: Path):
    stage = _write_stage(
        stages_dir,
        "2026-05-23-docs-only",
        mode="CLOSED",
        scope_lock=["docs/runtime/foo.md", ".claude/hooks/foo.py"],
    )
    res = _run_hook(stage, scratch_dir)
    assert res.returncode == 0
    assert res.stdout.strip() == ""


def test_implementation_mode_emits_nothing(stages_dir: Path, scratch_dir: Path):
    stage = _write_stage(
        stages_dir,
        "2026-05-23-in-flight",
        mode="IMPLEMENTATION",
        scope_lock=["trading_app/live/foo.py"],
    )
    res = _run_hook(stage, scratch_dir)
    assert res.returncode == 0
    assert res.stdout.strip() == ""


def test_fresh_global_marker_suppresses(stages_dir: Path, scratch_dir: Path):
    (scratch_dir / ".code-review-ts").touch()
    stage = _write_stage(
        stages_dir,
        "2026-05-23-marker-test",
        mode="CLOSED",
        scope_lock=["trading_app/live/foo.py"],
    )
    res = _run_hook(stage, scratch_dir)
    assert res.returncode == 0
    assert res.stdout.strip() == ""


def test_body_mentions_review_suppresses(stages_dir: Path, scratch_dir: Path):
    stage = _write_stage(
        stages_dir,
        "2026-05-23-already-reviewed",
        mode="CLOSED",
        scope_lock=["trading_app/live/foo.py"],
        body_extra="Code-review: dispatched evidence-auditor, verdict PASS.",
    )
    res = _run_hook(stage, scratch_dir)
    assert res.returncode == 0
    assert res.stdout.strip() == ""


def test_non_stage_file_ignored(scratch_dir: Path, tmp_path: Path):
    """Edits to non-stage files (e.g., pipeline/foo.py) should not trigger nudge."""
    not_a_stage = tmp_path / "pipeline" / "foo.py"
    not_a_stage.parent.mkdir(parents=True)
    not_a_stage.write_text("# noop\n", encoding="utf-8")
    res = _run_hook(not_a_stage, scratch_dir)
    assert res.returncode == 0
    assert res.stdout.strip() == ""


def test_marker_idempotency_back_to_back(stages_dir: Path, scratch_dir: Path):
    """Two events on the same closed stage emit only one nudge."""
    stage = _write_stage(
        stages_dir,
        "2026-05-23-idempotency",
        mode="CLOSED",
        scope_lock=["trading_app/live/foo.py"],
    )
    res1 = _run_hook(stage, scratch_dir)
    assert res1.returncode == 0
    assert "stage-closed code-review nudge" in res1.stdout

    res2 = _run_hook(stage, scratch_dir)
    assert res2.returncode == 0
    assert res2.stdout.strip() == "", "second invocation should be marker-suppressed"


def test_malformed_yaml_fails_open(stages_dir: Path, scratch_dir: Path):
    """Unreadable / weird stage file → exit 0, no nudge."""
    stage = stages_dir / "2026-05-23-malformed.md"
    stage.write_text("not even close to yaml :: garbage\n@@@\n", encoding="utf-8")
    res = _run_hook(stage, scratch_dir)
    assert res.returncode == 0
    assert res.stdout.strip() == ""


def test_missing_file_fails_open(stages_dir: Path, scratch_dir: Path):
    """File path that doesn't exist on disk → exit 0, no nudge."""
    missing = stages_dir / "2026-05-23-missing.md"
    res = _run_hook(missing, scratch_dir)
    assert res.returncode == 0
    assert res.stdout.strip() == ""
