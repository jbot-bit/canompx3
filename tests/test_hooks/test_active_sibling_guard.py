"""Pressure tests for `_active_sibling_lines()` in `.claude/hooks/session-start.py`.

The mtime-based active-worktree gate hard-blocks a second session in the SAME
worktree (the .git/index-corruption case) and warns on a hot SIBLING worktree
(the sanctioned parallel pattern). PID liveness is deliberately NOT used — it is
unreliable across Windows process trees, which is why the existing PID lock
under-reports. mtime of a dirty tracked file is the trustworthy signal.

Subprocess-free by design (see test_session_start_mutex.py module docstring for
the CI KeyboardInterrupt rationale): `_git` is stubbed and `os.stat` is
monkeypatched so no test spawns git or touches real worktrees.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = pytest.mark.timeout(0)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOOK_PATH = PROJECT_ROOT / ".claude" / "hooks" / "session-start.py"

NOW = 1_780_000_000.0  # fixed clock for deterministic age math
SELF = "/repo/main"
SIB = "/repo/sibling"


def _load_hook(
    monkeypatch: pytest.MonkeyPatch,
    *,
    worktrees: list[str],
    current: str,
    dirty: dict[str, list[str]],
) -> ModuleType:
    """Load the hook with `_git` stubbed to a synthetic worktree/status table.

    `worktrees` — paths returned by `git worktree list --porcelain`.
    `current`   — path `git rev-parse --show-toplevel` returns.
    `dirty`     — map of worktree path -> list of dirty tracked rel-paths
                  (what `git status --porcelain --untracked-files=no` yields).
    """
    spec = importlib.util.spec_from_file_location("session_start_hook_active", HOOK_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def _fake_git(args: list[str], timeout: int = 5) -> tuple[int, str]:
        if args[:2] == ["worktree", "list"]:
            return 0, "\n".join(f"worktree {w}\nHEAD 0000000\n" for w in worktrees)
        if args[:2] == ["rev-parse", "--show-toplevel"]:
            return 0, current
        # status --porcelain --untracked-files=no for a -C <path>
        if args[:1] == ["-C"] and "status" in args:
            wt = args[1]
            rels = dirty.get(wt, [])
            return 0, "\n".join(f" M {r}" for r in rels)
        return 1, ""

    monkeypatch.setattr(mod, "_git", _fake_git, raising=True)
    monkeypatch.delenv("CLAUDE_ALLOW_CONCURRENT", raising=False)
    return mod


def _patch_mtimes(monkeypatch: pytest.MonkeyPatch, mod: ModuleType, mtimes: dict[str, float]) -> None:
    """Make `os.stat(full_path).st_mtime` return controlled values.

    `mtimes` keys are full paths (worktree + os.sep + rel). Missing keys raise
    OSError to simulate a file that vanished mid-scan.
    """
    def _fake_stat(path, *a, **k):  # type: ignore[no-untyped-def]
        key = str(path).replace("\\", "/")
        if key in mtimes:
            return SimpleNamespace(st_mtime=mtimes[key])
        raise OSError(f"no such file (simulated): {key}")

    monkeypatch.setattr(mod.os, "stat", _fake_stat, raising=True)


def _f(wt: str, rel: str) -> str:
    return os.path.join(wt, rel).replace("\\", "/")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_no_siblings_no_block(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_hook(monkeypatch, worktrees=[SELF], current=SELF, dirty={})
    lines, block = mod._active_sibling_lines(now_epoch=NOW)
    assert block is False
    assert lines == []


def test_stale_dirty_sibling_does_not_block_or_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sibling dirty but last edited 6h ago — abandoned, not active."""
    mod = _load_hook(
        monkeypatch, worktrees=[SELF, SIB], current=SELF, dirty={SIB: ["HANDOFF.md"]}
    )
    _patch_mtimes(monkeypatch, mod, {_f(SIB, "HANDOFF.md"): NOW - 6 * 3600})
    lines, block = mod._active_sibling_lines(now_epoch=NOW)
    assert block is False
    assert lines == []  # outside window -> not even a warning


def test_active_sibling_warns_but_does_not_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """A DIFFERENT hot worktree is the sanctioned parallel pattern: warn, allow."""
    mod = _load_hook(
        monkeypatch, worktrees=[SELF, SIB], current=SELF, dirty={SIB: ["a.py"]}
    )
    _patch_mtimes(monkeypatch, mod, {_f(SIB, "a.py"): NOW - 120})  # 2 min ago
    lines, block = mod._active_sibling_lines(now_epoch=NOW)
    assert block is False
    assert any("other worktree" in ln for ln in lines)
    assert any(SIB in ln for ln in lines)


def test_active_SAME_worktree_hard_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """THIS tree edited 2 min ago with no live PID lock = corruption risk -> BLOCK."""
    mod = _load_hook(
        monkeypatch, worktrees=[SELF], current=SELF, dirty={SELF: ["live.py"]}
    )
    _patch_mtimes(monkeypatch, mod, {_f(SELF, "live.py"): NOW - 120})
    lines, block = mod._active_sibling_lines(now_epoch=NOW)
    assert block is True
    assert any("BLOCKED" in ln for ln in lines)
    assert any("CLAUDE_ALLOW_CONCURRENT" in ln for ln in lines)


def test_override_env_skips_block(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_hook(
        monkeypatch, worktrees=[SELF], current=SELF, dirty={SELF: ["live.py"]}
    )
    _patch_mtimes(monkeypatch, mod, {_f(SELF, "live.py"): NOW - 120})
    monkeypatch.setenv("CLAUDE_ALLOW_CONCURRENT", "1")
    lines, block = mod._active_sibling_lines(now_epoch=NOW)
    assert block is False
    assert lines == []


def test_future_mtime_clock_skew_does_not_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """A file 'modified in the future' is garbage — never block on it."""
    mod = _load_hook(
        monkeypatch, worktrees=[SELF], current=SELF, dirty={SELF: ["live.py"]}
    )
    _patch_mtimes(monkeypatch, mod, {_f(SELF, "live.py"): NOW + 5 * 3600})
    lines, block = mod._active_sibling_lines(now_epoch=NOW)
    assert block is False
    assert lines == []


def test_vanished_file_mid_scan_fails_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dirty path reported by git but file gone when stat'd (rename/delete) -> skip."""
    mod = _load_hook(
        monkeypatch, worktrees=[SELF, SIB], current=SELF, dirty={SIB: ["gone.py"]}
    )
    _patch_mtimes(monkeypatch, mod, {})  # every stat raises OSError
    lines, block = mod._active_sibling_lines(now_epoch=NOW)
    assert block is False
    assert lines == []


def test_git_failure_fails_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """`git worktree list` failing must never wedge session start."""
    mod = _load_hook(monkeypatch, worktrees=[SELF], current=SELF, dirty={})

    def _broken_git(args: list[str], timeout: int = 5) -> tuple[int, str]:
        return 1, ""

    monkeypatch.setattr(mod, "_git", _broken_git, raising=True)
    lines, block = mod._active_sibling_lines(now_epoch=NOW)
    assert block is False
    assert lines == []


def test_self_path_normalised_not_treated_as_sibling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backslash/forward-slash + trailing-slash variants of the current path
    must resolve to SELF (hard block), not a sibling (warn-only)."""
    weird_self = "/repo/main/"  # trailing slash variant
    mod = _load_hook(
        monkeypatch,
        worktrees=[weird_self],
        current=SELF,
        dirty={weird_self: ["live.py"]},
    )
    _patch_mtimes(monkeypatch, mod, {_f(weird_self, "live.py"): NOW - 60})
    lines, block = mod._active_sibling_lines(now_epoch=NOW)
    assert block is True  # recognised as same worktree despite trailing slash


def test_boundary_just_inside_window_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Edited just under 15 min ago -> still active -> block (same tree)."""
    mod = _load_hook(
        monkeypatch, worktrees=[SELF], current=SELF, dirty={SELF: ["x.py"]}
    )
    _patch_mtimes(monkeypatch, mod, {_f(SELF, "x.py"): NOW - (15 * 60 - 5)})
    _, block = mod._active_sibling_lines(now_epoch=NOW)
    assert block is True


def test_boundary_just_outside_window_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Edited just over 15 min ago -> stale -> pass."""
    mod = _load_hook(
        monkeypatch, worktrees=[SELF], current=SELF, dirty={SELF: ["x.py"]}
    )
    _patch_mtimes(monkeypatch, mod, {_f(SELF, "x.py"): NOW - (15 * 60 + 5)})
    lines, block = mod._active_sibling_lines(now_epoch=NOW)
    assert block is False
    assert lines == []
