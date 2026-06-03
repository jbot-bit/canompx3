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


def test_peer_dirty_cache_marks_matching_scope_only():
    cache = sra.AuditCache(peer_dirty_hits={"scripts/tools/x.py": ["peer:scripts/tools/x.py"]})
    assert sra._peer_dirty_on_cached(cache, ["scripts/tools/x.py"]) == ["peer:scripts/tools/x.py"]
    assert sra._peer_dirty_on_cached(cache, ["scripts/tools/y.py"]) == []


def test_peer_dirty_cache_directory_scope_matches_files_beneath():
    # A directory scope entry (trailing `/`) must catch a peer-dirty file BENEATH
    # it — restoring the old git-pathspec prefix semantics. A plain exact-match
    # cache would MISS this (the UNSAFE direction: a contested stage could reach
    # DONE_SAFE). Regression guard for the pathspec->exact-match narrowing.
    cache = sra.AuditCache(peer_dirty_hits={"pipeline/foo.py": ["peer:pipeline/foo.py"]})
    assert sra._peer_dirty_on_cached(cache, ["pipeline/"]) == ["peer:pipeline/foo.py"]
    # A non-matching directory yields nothing.
    assert sra._peer_dirty_on_cached(cache, ["trading_app/"]) == []
    # A directory entry that is a string-prefix of a sibling FILE name must not
    # over-match across the boundary (`pipeline/` ⊄ `pipelinex/foo.py`).
    cache2 = sra.AuditCache(peer_dirty_hits={"pipelinex/foo.py": ["peer:pipelinex/foo.py"]})
    assert sra._peer_dirty_on_cached(cache2, ["pipeline/"]) == []
    # Exact file entries are unaffected by the directory branch.
    assert sra._peer_dirty_on_cached(cache, ["pipeline/foo.py"]) == ["peer:pipeline/foo.py"]


def test_porcelain_paths_modified_file_with_arrow_in_name_is_not_a_rename():
    # A file literally named with " -> " that is MODIFIED (status ` M`, no R/C)
    # must NOT be misread as a rename — only R/C status triggers the split.
    # Locks the safe-direction handling of the " -> " heuristic.
    assert sra._porcelain_paths(" M a -> b.md") == ["a -> b.md"]


def test_porcelain_paths_extracts_both_sides_of_rename():
    # Plain lines yield one path.
    assert sra._porcelain_paths(" M docs/runtime/stages/foo.md") == ["docs/runtime/stages/foo.md"]
    assert sra._porcelain_paths("?? scripts/new.py") == ["scripts/new.py"]
    assert sra._porcelain_paths("A  added.py") == ["added.py"]
    assert sra._porcelain_paths("UU both.md") == ["both.md"]
    # Rename/copy lines yield BOTH old and new (git porcelain v1: `old -> new`).
    assert sra._porcelain_paths("R  a/old.md -> a/archive/old.md") == ["a/old.md", "a/archive/old.md"]
    assert sra._porcelain_paths("C  src.py -> copy.py") == ["src.py", "copy.py"]
    # R/C can sit in EITHER status column per the spec (X=index, Y=work tree).
    assert sra._porcelain_paths(" R old.md -> new.md") == ["old.md", "new.md"]
    assert sra._porcelain_paths("MR staged.md -> renamed.md") == ["staged.md", "renamed.md"]
    # C-quoted paths (whitespace/specials) have their wrapping quotes stripped.
    assert sra._porcelain_paths('R  "old name.md" -> "new name.md"') == ["old name.md", "new name.md"]
    assert sra._porcelain_paths(' M "sp ace.md"') == ["sp ace.md"]
    # Malformed / empty lines yield nothing.
    assert sra._porcelain_paths("") == []
    assert sra._porcelain_paths("XY") == []


def test_peer_dirty_cache_detects_renamed_scope_file():
    # A peer that git-mv'd a scope file emits `R old -> new`. The reaper MUST
    # still see the OLD path as dirty (hands off) — missing it would let a
    # contested stage be reaped. Regression guard for the pathspec->exact-match
    # narrowing introduced with AuditCache.
    porcelain_line = "R  docs/runtime/stages/foo.md -> docs/runtime/stages/archive/foo.md"
    dirty: dict[str, list[str]] = {}
    for path in sra._porcelain_paths(porcelain_line):
        dirty.setdefault(path, []).append(f"peer:{path}")
    cache = sra.AuditCache(peer_dirty_hits=dirty)
    assert sra._peer_dirty_on_cached(cache, ["docs/runtime/stages/foo.md"]) == ["peer:docs/runtime/stages/foo.md"]
    assert sra._peer_dirty_on_cached(cache, ["docs/runtime/stages/archive/foo.md"]) == [
        "peer:docs/runtime/stages/archive/foo.md"
    ]


def test_commit_age_cache_calls_git_once(monkeypatch):
    cache = sra.AuditCache()
    calls = []

    def fake_age(root, rel_path):
        calls.append(rel_path)
        return 100.0

    monkeypatch.setattr(sra, "_git_last_commit_age_hours", fake_age)
    assert sra._git_last_commit_age_hours_cached(cache, REPO_ROOT, "scripts/tools/x.py") == 100.0
    assert sra._git_last_commit_age_hours_cached(cache, REPO_ROOT, "scripts/tools/x.py") == 100.0
    assert calls == ["scripts/tools/x.py"]


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
