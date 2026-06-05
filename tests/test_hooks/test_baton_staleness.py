"""Tests for the baton staleness detector in .claude/hooks/_memory_capture.py.

The 4th check of the auto-memory-capture loop: on SessionStart, re-verify
falsifiable "not-on-origin / unmerged" git claims in memory batons against live
git. Fire ONLY on a git-PROVEN contradiction (baton says unmerged, git proves
merged). Never on prose alone; never on a genuinely-unmerged SHA. Fail-open.

Covers:
- PROVEN STALE: baton claims `<sha> NOT on origin/main`, but sha IS an ancestor
  of origin/main -> flagged (this is exactly this-session's f1fd7a90 class).
- GENUINELY UNMERGED: baton claims local-only, sha is NOT on origin/main ->
  silent (claim is still true).
- NO ADJACENT SHA: "unmerged" prose with no SHA on the line -> silent
  (proximity guard; no false positive).
- UNKNOWN SHA: a fabricated/GC'd sha -> silent (cannot prove merged).
- _sha_on_origin_main is strict: True only for a real ancestor.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = PROJECT_ROOT / ".claude" / "hooks" / "_memory_capture.py"


def _load_helper() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_memory_capture", HELPER_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repo_with_origin(tmp_path: Path) -> tuple[Path, str, str]:
    """A repo with origin/main set; returns (repo, merged_sha, unmerged_sha)."""
    repo = tmp_path / "wt"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "--initial-branch=main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "a.txt").write_text("1")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "c1"], cwd=repo, check=True)
    merged_sha = _run(repo, "rev-parse", "HEAD")
    # Make origin/main point at this commit (no real remote needed).
    subprocess.run(["git", "update-ref", "refs/remotes/origin/main", merged_sha], cwd=repo, check=True)
    # A second commit NOT reachable from origin/main.
    (repo / "b.txt").write_text("2")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "c2"], cwd=repo, check=True)
    unmerged_sha = _run(repo, "rev-parse", "HEAD")
    return repo, merged_sha, unmerged_sha


def _patch_paths(mod: ModuleType, repo: Path, memory: Path) -> None:
    mod.PROJECT_ROOT = repo  # _git runs cwd=PROJECT_ROOT
    mod.MEMORY_DIR = memory


def test_proven_stale_claim_is_flagged(repo_with_origin, tmp_path):
    repo, merged_sha, _ = repo_with_origin
    mem = tmp_path / "memory"
    mem.mkdir()
    # The exact this-session failure shape.
    (mem / "baton.md").write_text(f"- `{merged_sha[:8]}` (Stage-1 guard commit) is NOT on origin/main — unmerged.\n")
    mod = _load_helper()
    _patch_paths(mod, repo, mem)
    findings = mod.scan_stale_batons()
    assert len(findings) == 1
    assert findings[0]["file"] == "baton.md"
    assert merged_sha[:8] in findings[0]["shas"]


def test_genuinely_unmerged_claim_is_silent(repo_with_origin, tmp_path):
    repo, _, unmerged_sha = repo_with_origin
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "baton.md").write_text(f"- `{unmerged_sha[:8]}` committed local-only, NOT on origin/main yet.\n")
    mod = _load_helper()
    _patch_paths(mod, repo, mem)
    assert mod.scan_stale_batons() == []  # claim is TRUE -> no nag


def test_unmerged_prose_without_adjacent_sha_is_silent(repo_with_origin, tmp_path):
    repo, merged_sha, _ = repo_with_origin
    mem = tmp_path / "memory"
    mem.mkdir()
    # SHA present, but on a DIFFERENT line from the not-merged phrasing.
    (mem / "baton.md").write_text(
        f"Discussion of unmerged work in the abstract.\nSeparately, commit {merged_sha[:8]} did something.\n"
    )
    mod = _load_helper()
    _patch_paths(mod, repo, mem)
    assert mod.scan_stale_batons() == []  # proximity guard


def test_unknown_sha_is_silent(repo_with_origin, tmp_path):
    repo, _, _ = repo_with_origin
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "baton.md").write_text("- `deadbeef` is NOT on origin/main.\n")
    mod = _load_helper()
    _patch_paths(mod, repo, mem)
    assert mod.scan_stale_batons() == []  # cannot prove merged -> silent


def test_sha_on_origin_main_is_strict(repo_with_origin, tmp_path):
    repo, merged_sha, unmerged_sha = repo_with_origin
    mod = _load_helper()
    _patch_paths(mod, repo, tmp_path / "memory")
    assert mod._sha_on_origin_main(merged_sha) is True
    assert mod._sha_on_origin_main(unmerged_sha) is False
    assert mod._sha_on_origin_main("deadbeef") is False


def test_scan_fail_open_on_missing_memory_dir(repo_with_origin, tmp_path):
    repo, _, _ = repo_with_origin
    mod = _load_helper()
    _patch_paths(mod, repo, tmp_path / "does-not-exist")
    assert mod.scan_stale_batons() == []


# --- Tier 2: recent type:project batons asserting live status ---------------


def _write_baton(mem: Path, name: str, type_: str, desc: str) -> Path:
    p = mem / name
    p.write_text(f"---\nname: {name[:-3]}\ndescription: {desc}\nmetadata:\n  type: {type_}\n---\nbody\n")
    return p


def test_tier2_flags_recent_project_baton_with_live_status(repo_with_origin, tmp_path):
    repo, _, _ = repo_with_origin
    mem = tmp_path / "memory"
    mem.mkdir()
    _write_baton(mem, "project_x.md", "project", "RESUME: NEXT = Stage 3 wire cap")
    mod = _load_helper()
    _patch_paths(mod, repo, mem)
    assert mod.scan_live_project_batons() == ["project_x.md"]


def test_tier2_ignores_feedback_type(repo_with_origin, tmp_path):
    repo, _, _ = repo_with_origin
    mem = tmp_path / "memory"
    mem.mkdir()
    # Feedback file that legitimately discusses 'unmerged' — must NOT surface.
    _write_baton(mem, "feedback_y.md", "feedback", "never leave work unmerged; NOT on origin")
    mod = _load_helper()
    _patch_paths(mod, repo, mem)
    assert mod.scan_live_project_batons() == []


def test_tier2_ignores_settled_project_baton(repo_with_origin, tmp_path):
    repo, _, _ = repo_with_origin
    mem = tmp_path / "memory"
    mem.mkdir()
    _write_baton(mem, "project_done.md", "project", "DONE and landed; no further action")
    mod = _load_helper()
    _patch_paths(mod, repo, mem)
    assert mod.scan_live_project_batons() == []


def test_memory_dir_resolves_home_based_not_repo_based(tmp_path, monkeypatch):
    """Regression: the default MEMORY_DIR must derive a HOME-based slug path,
    not a repo-local one (the latter is a near-empty decoy on this machine).

    Exercises the REAL resolver against a fake home so a misconfigured default
    can't pass by being patched away (the gap that hid the original bug)."""
    mod = _load_helper()
    fake_root = tmp_path / "proj"
    fake_root.mkdir()
    monkeypatch.setattr(mod, "PROJECT_ROOT", fake_root)
    fake_home = tmp_path / "home"
    slug = str(fake_root).replace(":", "-").replace("\\", "-").replace("/", "-")
    home_mem = fake_home / ".claude" / "projects" / slug / "memory"
    home_mem.mkdir(parents=True)
    monkeypatch.setattr(mod.Path, "home", staticmethod(lambda: fake_home))
    assert mod._resolve_memory_dir() == home_mem


def test_tier2_ignores_old_project_baton(repo_with_origin, tmp_path):
    import os
    import time

    repo, _, _ = repo_with_origin
    mem = tmp_path / "memory"
    mem.mkdir()
    p = _write_baton(mem, "project_old.md", "project", "RESUME: NEXT = something")
    old = time.time() - (200 * 3600)  # ~8 days, well past the 72h window
    os.utime(p, (old, old))
    mod = _load_helper()
    _patch_paths(mod, repo, mem)
    assert mod.scan_live_project_batons() == []
