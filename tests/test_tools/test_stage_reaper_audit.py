"""Tests for scripts/tools/stage_reaper_audit.py — the stale-stage classifier.

Covers all four classifications and the hard safety gates:
- DONE_SAFE only when scope is git-tracked, quiet > recency, no peer dirty.
- LIVE_OR_CONTESTED on recent commit, peer-dirty, or non-IMPLEMENTATION mode.
- UNVERIFIABLE on missing git history or unparseable scope.
- CLOSED passed through untouched.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    import sys

    spec = importlib.util.spec_from_file_location(
        "stage_reaper_audit", REPO_ROOT / "scripts" / "tools" / "stage_reaper_audit.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass can resolve cls.__module__ (else the
    # module isn't in sys.modules and dataclasses raises AttributeError).
    sys.modules["stage_reaper_audit"] = mod
    spec.loader.exec_module(mod)
    return mod


sra = _load_module()


def test_help_output_survives_narrow_console_encoding() -> None:
    """`--help` must not crash on Windows consoles with cp1252-style output."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "tools" / "stage_reaper_audit.py"), "--help"],
        cwd=REPO_ROOT,
        env={"PYTHONIOENCODING": "cp1252"},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "DONE_SAFE" in result.stdout


def _stage(tmp_path: Path, name: str, mode: str, scope: list[str]) -> Path:
    body = [
        "---",
        f'task: "{name}"',
        f"mode: {mode}",
        "scope_lock:",
        *[f"  - {p}" for p in scope],
        "blast_radius: |",
        "  ## Blast Radius",
        "  - synthetic test stage, no real blast radius beyond this fixture file",
        "---",
        "",
        "## STATUS",
        "synthetic",
    ]
    f = tmp_path / name
    f.write_text("\n".join(body), encoding="utf-8")
    return f


@pytest.fixture
def parsers():
    return sra._load_stage_parsers()


def test_closed_stage_is_passed_through(tmp_path, parsers):
    pf, psl = parsers
    f = _stage(tmp_path, "closed.md", "CLOSED", ["scripts/tools/foo.py"])
    v = sra.classify_stage(f, pf, psl, [], 48.0, REPO_ROOT)
    assert v.classification == "CLOSED"


def test_design_mode_never_auto_archived(tmp_path, parsers, monkeypatch):
    pf, psl = parsers
    # Even with a quiet, tracked scope file, DESIGN must hold.
    monkeypatch.setattr(sra, "_git_last_commit_age_hours", lambda root, sp: 9999.0)
    f = _stage(tmp_path, "design.md", "DESIGN", ["pipeline/dst.py"])
    v = sra.classify_stage(f, pf, psl, [], 48.0, REPO_ROOT)
    assert v.classification == "LIVE_OR_CONTESTED"
    assert "IMPLEMENTATION/TRIVIAL" in v.reasons[0]


def test_no_scope_lock_is_unverifiable(tmp_path, parsers):
    pf, psl = parsers
    f = _stage(tmp_path, "noscope.md", "IMPLEMENTATION", [])
    v = sra.classify_stage(f, pf, psl, [], 48.0, REPO_ROOT)
    assert v.classification == "UNVERIFIABLE"
    assert "no scope_lock" in v.reasons[0]


def test_missing_git_history_is_unverifiable(tmp_path, parsers, monkeypatch):
    pf, psl = parsers
    monkeypatch.setattr(sra, "_git_last_commit_age_hours", lambda root, sp: None)
    f = _stage(tmp_path, "newfile.md", "IMPLEMENTATION", ["scripts/tools/brand_new.py"])
    v = sra.classify_stage(f, pf, psl, [], 48.0, REPO_ROOT)
    assert v.classification == "UNVERIFIABLE"
    assert "no git history" in v.reasons[0]


def test_recent_commit_is_live_or_contested(tmp_path, parsers, monkeypatch):
    pf, psl = parsers
    monkeypatch.setattr(sra, "_git_last_commit_age_hours", lambda root, sp: 3.0)
    f = _stage(tmp_path, "recent.md", "IMPLEMENTATION", ["scripts/tools/hot.py"])
    v = sra.classify_stage(f, pf, psl, [], 48.0, REPO_ROOT)
    assert v.classification == "LIVE_OR_CONTESTED"
    assert "recent commit" in v.reasons[0]


def test_protected_path_tagged_in_reason(tmp_path, parsers, monkeypatch):
    pf, psl = parsers
    monkeypatch.setattr(sra, "_git_last_commit_age_hours", lambda root, sp: 3.0)
    f = _stage(tmp_path, "live.md", "IMPLEMENTATION", ["trading_app/live/broker.py"])
    v = sra.classify_stage(f, pf, psl, [], 48.0, REPO_ROOT)
    assert v.classification == "LIVE_OR_CONTESTED"
    assert "PROTECTED" in v.reasons[0]


def test_peer_dirty_is_live_or_contested(tmp_path, parsers, monkeypatch):
    pf, psl = parsers
    # Quiet scope, but a peer worktree is dirty on it.
    monkeypatch.setattr(sra, "_git_last_commit_age_hours", lambda root, sp: 9999.0)
    monkeypatch.setattr(sra, "_peer_dirty_on", lambda peers, scope: ["peerwt:scripts/tools/x.py"])
    f = _stage(tmp_path, "peer.md", "IMPLEMENTATION", ["scripts/tools/x.py"])
    v = sra.classify_stage(f, pf, psl, [Path("/fake/peer")], 48.0, REPO_ROOT)
    assert v.classification == "LIVE_OR_CONTESTED"
    assert "peer dirty" in v.reasons[0]


def test_quiet_tracked_no_peer_is_done_safe(tmp_path, parsers, monkeypatch):
    pf, psl = parsers
    monkeypatch.setattr(sra, "_git_last_commit_age_hours", lambda root, sp: 9999.0)
    monkeypatch.setattr(sra, "_peer_dirty_on", lambda peers, scope: [])
    f = _stage(tmp_path, "done.md", "IMPLEMENTATION", ["scripts/daily_refresh.bat"])
    v = sra.classify_stage(f, pf, psl, [], 48.0, REPO_ROOT)
    assert v.classification == "DONE_SAFE"


def test_one_recent_scope_file_holds_whole_stage(tmp_path, parsers, monkeypatch):
    pf, psl = parsers
    # First file quiet, second recent → whole stage held.
    ages = {"scripts/tools/a.py": 9999.0, "scripts/tools/b.py": 2.0}
    monkeypatch.setattr(sra, "_git_last_commit_age_hours", lambda root, sp: ages[sp])
    monkeypatch.setattr(sra, "_peer_dirty_on", lambda peers, scope: [])
    f = _stage(tmp_path, "mixed.md", "IMPLEMENTATION", ["scripts/tools/a.py", "scripts/tools/b.py"])
    v = sra.classify_stage(f, pf, psl, [], 48.0, REPO_ROOT)
    assert v.classification == "LIVE_OR_CONTESTED"
