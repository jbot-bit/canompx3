"""CLI tests for scripts/tools/work_queue.py — specifically the render-handoff
`--write` footgun guard added 2026-05-17.

Background: `render-handoff --write` replaces HANDOFF.md with the thin queue
baton, deleting all session-prose blocks ("Last Session", "This Session", etc).
A pulse warning (`handoff_queue_mismatch`) used to recommend the command in
its `detail:` field — an operator following the hint silently lost ~65 lines
of session memory. `--force` is now required to confirm the destruction.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = PROJECT_ROOT / "scripts" / "tools" / "work_queue.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location("_work_queue_cli_under_test", CLI_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed_queue(root: Path) -> None:
    path = root / "docs" / "runtime" / "action-queue.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "updated_at: 2026-05-17T00:00:00+00:00",
                "items:",
                "  - id: first",
                "    title: First thing",
                "    class: research",
                "    status: ready",
                "    priority: P1",
                "    close_before_new_work: true",
                "    owner_hint: codex",
                "    last_verified_at: 2026-05-17",
                "    freshness_sla_days: 2",
                "    next_action: Do first",
                "    exit_criteria: Finish first",
                "    blocked_by: []",
                "    decision_refs: []",
                "    evidence_refs: []",
                "    notes_ref: docs/runtime/stages/first.md",
                "    override_note:",
            ]
        ),
        encoding="utf-8",
    )


class TestRenderHandoffWriteFootgun:
    def test_write_without_force_refuses_and_preserves_handoff(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _seed_queue(tmp_path)
        handoff = tmp_path / "HANDOFF.md"
        original = "# HANDOFF\n\n## Last Session\n- precious session prose worth preserving\n"
        handoff.write_text(original, encoding="utf-8")

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["work_queue.py", "render-handoff", "--write"])

        cli = _load_cli_module()
        rc = cli.main()

        assert rc == 2, "expected non-zero exit when --write used without --force"
        assert handoff.read_text(encoding="utf-8") == original, "HANDOFF.md must not be overwritten"

        captured = capsys.readouterr()
        assert "force" in captured.err.lower(), "stderr must mention --force escape hatch"

    def test_write_with_force_overwrites_handoff(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _seed_queue(tmp_path)
        handoff = tmp_path / "HANDOFF.md"
        original = "# old\n"
        handoff.write_text(original, encoding="utf-8")

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            ["work_queue.py", "render-handoff", "--write", "--force"],
        )

        cli = _load_cli_module()
        rc = cli.main()

        assert rc == 0, "--write --force must succeed (escape hatch preserved)"
        assert handoff.read_text(encoding="utf-8") != original, "HANDOFF.md must be rewritten with --force"
        assert "First thing" in handoff.read_text(encoding="utf-8"), "rewritten HANDOFF must contain queue render"

    def test_render_handoff_reconfigures_cp1252_stdout_before_printing_unicode(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _seed_queue(tmp_path)
        stdout_bytes = io.BytesIO()
        cp1252_stdout = io.TextIOWrapper(stdout_bytes, encoding="cp1252", errors="strict")

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["work_queue.py", "render-handoff", "--summary", "fresh → current"])
        monkeypatch.setattr(sys, "stdout", cp1252_stdout)

        cli = _load_cli_module()
        rc = cli.main()
        sys.stdout.flush()

        assert rc == 0
        assert "Cross-Tool Session Baton" in stdout_bytes.getvalue().decode("utf-8")
