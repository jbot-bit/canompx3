"""Violation-injection tests for four high-blast-radius drift checks.

Integration audit (2026-06-09) found these registered drift checks had no direct
test references, so "check exists" could be mistaken for "check is violation-
tested" (integrity-guardian.md § 11 / institutional-rigor.md § 11 — never trust a
check without known-violation injection). The four are capital-/safety-adjacent:

  - ``check_c1_kill_switch_guards_intact``     — broker-entry kill-switch race (C1)
  - ``check_brain_consuming_hooks_registered`` — silent dead-hook (data-loss path)
  - ``check_preflight_launcher_modes``         — live preflight launcher modes
  - ``check_action_queue_loads_cleanly``       — /orient + /next work-queue schema

Each check gets (1) a pass-case asserting the real repo files are clean today and
(2) a violation-case proving the check actually catches injected breakage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import pipeline.check_drift as cd
from pipeline.check_drift import (
    check_action_queue_loads_cleanly,
    check_brain_consuming_hooks_registered,
    check_c1_kill_switch_guards_intact,
    check_preflight_launcher_modes,
)

# --------------------------------------------------------------------------- #
# check_c1_kill_switch_guards_intact
# --------------------------------------------------------------------------- #
# Violation: _handle_event lost its kill-switch guard entirely → C1 race re-opens.
_BAD_ORCHESTRATOR_NO_GUARD = """\
class SessionOrchestrator:
    async def _on_bar(self, bar):
        if self._kill_switch_fired:
            return
        await self._dispatch(bar)

    async def _handle_event(self, event):
        await self._submit(event)
"""

# Violation: guard present but BLANKET (no ENTRY discriminator) → breaks EOD
# wind-down (iter 178 audit do-not-touch).
_BAD_ORCHESTRATOR_BLANKET = """\
class SessionOrchestrator:
    async def _on_bar(self, bar):
        if self._kill_switch_fired:
            return
        await self._dispatch(bar)

    async def _handle_event(self, event):
        if self._kill_switch_fired:
            return
        await self._submit(event)
"""


def _write_orchestrator(root: Path, body: str) -> None:
    target = root / "live" / "session_orchestrator.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")


def test_c1_kill_switch_passes_on_real_repo() -> None:
    assert check_c1_kill_switch_guards_intact() == []


def test_c1_kill_switch_catches_removed_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_orchestrator(tmp_path, _BAD_ORCHESTRATOR_NO_GUARD)
    monkeypatch.setattr(cd, "TRADING_APP_DIR", tmp_path)
    violations = check_c1_kill_switch_guards_intact()
    assert violations, "removed _handle_event kill-switch guard must be caught"
    assert any("_handle_event" in v for v in violations)


def test_c1_kill_switch_catches_blanket_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_orchestrator(tmp_path, _BAD_ORCHESTRATOR_BLANKET)
    monkeypatch.setattr(cd, "TRADING_APP_DIR", tmp_path)
    violations = check_c1_kill_switch_guards_intact()
    assert violations, "blanket (non-ENTRY-scoped) guard must be caught"
    assert any("ENTRY" in v for v in violations)


def test_c1_kill_switch_catches_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cd, "TRADING_APP_DIR", tmp_path)  # no session_orchestrator.py
    violations = check_c1_kill_switch_guards_intact()
    assert violations and any("missing" in v for v in violations)


# --------------------------------------------------------------------------- #
# check_brain_consuming_hooks_registered
# --------------------------------------------------------------------------- #
_REQUIRED_HOOKS = (
    "worktree_guard.py",
    "worktree-destroy-guard.py",
    "mcp-git-guard.py",
    "branch-flip-guard.py",
)


def _write_settings(root: Path, hook_names: tuple[str, ...]) -> None:
    import json

    settings_dir = root / ".claude"
    settings_dir.mkdir(parents=True, exist_ok=True)
    hooks = [{"hooks": [{"command": f"python C:/x/.claude/hooks/{name}"}]} for name in hook_names]
    (settings_dir / "settings.json").write_text(json.dumps({"hooks": {"PostToolUse": hooks}}), encoding="utf-8")


def test_brain_hooks_passes_on_real_repo() -> None:
    assert check_brain_consuming_hooks_registered() == []


def test_brain_hooks_catches_unregistered_hook(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Drop mcp-git-guard.py — the exact silent-unwiring class this check exists for.
    present = tuple(h for h in _REQUIRED_HOOKS if h != "mcp-git-guard.py")
    _write_settings(tmp_path, present)
    monkeypatch.setattr(cd, "PROJECT_ROOT", tmp_path)
    violations = check_brain_consuming_hooks_registered()
    assert violations, "an unregistered required hook must be caught"
    assert any("mcp-git-guard.py" in v for v in violations)


def test_brain_hooks_catches_missing_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cd, "PROJECT_ROOT", tmp_path)  # no .claude/settings.json
    violations = check_brain_consuming_hooks_registered()
    assert violations and any("missing" in v for v in violations)


def test_brain_hooks_rejects_substring_only_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A longer filename CONTAINING a required name as a substring must NOT satisfy
    # the requirement (the check matches a path segment, preceded by / or \).
    spoofed = tuple(f"x-{h}" if h == "branch-flip-guard.py" else h for h in _REQUIRED_HOOKS)
    _write_settings(tmp_path, spoofed)
    monkeypatch.setattr(cd, "PROJECT_ROOT", tmp_path)
    violations = check_brain_consuming_hooks_registered()
    assert any("branch-flip-guard.py" in v for v in violations)


# --------------------------------------------------------------------------- #
# check_preflight_launcher_modes
# --------------------------------------------------------------------------- #
# The check reads four infra scripts under scripts/infra/ and requires each to
# carry an explicit `--claim ... --mode ...` token, plus windows_agent_launch.py
# must pass `"--mode", mode` through. We exercise it via monkeypatched PROJECT_ROOT.
_LAUNCHER_TOKENS = {
    "scripts/infra/codex-project.sh": "--claim codex --mode mutating",
    "scripts/infra/codex-project-search.sh": "--claim codex-search --mode read-only",
    "scripts/infra/wsl-env.sh": "--claim wsl-shell --mode read-only",
    "scripts/infra/claude-worktree.sh": "--claim claude --mode mutating",
}


def _write_launchers(root: Path, *, omit: str | None = None) -> None:
    for rel, token in _LAUNCHER_TOKENS.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        body = "" if rel == omit else f"#!/bin/bash\nrun_preflight {token}\n"
        path.write_text(body, encoding="utf-8")
    win = root / "scripts" / "infra" / "windows_agent_launch.py"
    win.write_text('args = ["--claim", claim, "--mode", mode]\n', encoding="utf-8")


def test_preflight_launcher_modes_passes_on_real_repo() -> None:
    assert check_preflight_launcher_modes() == []


def test_preflight_launcher_modes_catches_missing_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_launchers(tmp_path, omit="scripts/infra/wsl-env.sh")
    monkeypatch.setattr(cd, "PROJECT_ROOT", tmp_path)
    violations = check_preflight_launcher_modes()
    assert violations, "a launcher missing its preflight mode token must be caught"
    assert any("wsl-env.sh" in v for v in violations)


def test_preflight_launcher_modes_catches_windows_passthrough_drop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_launchers(tmp_path)
    win = tmp_path / "scripts" / "infra" / "windows_agent_launch.py"
    win.write_text("args = []  # dropped --mode passthrough\n", encoding="utf-8")
    monkeypatch.setattr(cd, "PROJECT_ROOT", tmp_path)
    violations = check_preflight_launcher_modes()
    assert any("windows_agent_launch.py" in v for v in violations)


# --------------------------------------------------------------------------- #
# check_action_queue_loads_cleanly
# --------------------------------------------------------------------------- #
# The check calls ``load_queue()`` with NO args. ``load_queue``'s ``root`` default
# is bound to PROJECT_ROOT at definition time, so patching the module constant
# afterward does NOT redirect it. We therefore test two honest seams:
#   (a) the check passes on the real (clean) repo queue, AND
#   (b) the strict-schema contract the check depends on — ``load_queue(root=...)``
#       rejecting drifted YAML — is proven directly, plus a monkeypatched
#       ``load_queue`` proving the check surfaces a raised error fail-closed.


def _write_action_queue(root: Path, yaml_text: str) -> None:
    qpath = root / "docs" / "runtime" / "action-queue.yaml"
    qpath.parent.mkdir(parents=True, exist_ok=True)
    qpath.write_text(yaml_text, encoding="utf-8")


def test_action_queue_passes_on_real_repo() -> None:
    assert check_action_queue_loads_cleanly() == []


def test_load_queue_rejects_extra_field(tmp_path: Path) -> None:
    """The strict-schema contract (extra='forbid') the drift check relies on."""
    from pipeline.work_queue import load_queue

    _write_action_queue(
        tmp_path,
        "items:\n  - id: t1\n    title: bad\n    status: open\n    queue_class: audit\n    bogus_extra_field: nope\n",
    )
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
        load_queue(root=tmp_path)


def test_load_queue_rejects_bad_status(tmp_path: Path) -> None:
    from pipeline.work_queue import load_queue

    _write_action_queue(
        tmp_path,
        "items:\n  - id: t1\n    title: bad-status\n    status: not_a_real_status\n    queue_class: audit\n",
    )
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
        load_queue(root=tmp_path)


def test_action_queue_check_fails_closed_on_load_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If load_queue raises, the check must surface a violation (fail-closed),
    not swallow it and report clean."""
    import pipeline.work_queue as wq

    def _boom(*_a, **_k):
        raise ValueError("synthetic schema drift")

    monkeypatch.setattr(wq, "load_queue", _boom)
    violations = check_action_queue_loads_cleanly()
    assert violations, "a raised load_queue error must fail closed as a violation"
    assert any("does NOT validate" in v or "synthetic schema drift" in v for v in violations)
