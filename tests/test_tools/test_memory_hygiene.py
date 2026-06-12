"""Tests for scripts/tools/memory_hygiene.py — read-only budget + baton tiers.

The tool is loaded via importlib (it imports `_memory_capture.py` BY PATH at
module load, binding `MEMORY_DIR` / `MEMORY_MD` / `_sha_on_origin_main` as
module globals). To test deterministically we rebind those globals on a fresh
load and redirect the underlying git calls to a throwaway repo.

Why patch BOTH the tool AND `_CAP`:
- `budget_report()` reads `MEMORY_MD` (tool global) — patch the tool.
- `tier_baton()` globs `MEMORY_DIR` (tool global) and calls
  `_sha_on_origin_main` (bound from `_CAP`), which runs `git merge-base` at
  `_CAP.PROJECT_ROOT`. Redirecting that to a tmp repo with a synthetic
  `refs/remotes/origin/main` makes ancestor checks deterministic — the same
  `update-ref` idiom as tests/test_hooks/test_baton_staleness.py.

Covers:
- byte-cutoff first-dropped-line math (HARD_BYTE_CUTOFF binds before line 200)
- 200-line truncation point (line cutoff binds when bytes stay small)
- whole-index-loads (under both budgets -> first_dropped is None)
- over-long-line flag (> 200 chars; headers/blanks skipped)
- READY: primary SHA on origin/main AND no open-work marker
- LANDED-BUT-OPEN: primary SHA merged BUT an open-work marker present
- UNVERIFIED: unmerged primary SHA, and no-SHA baton
- primary-SHA anchoring (DONE+PUSHED anchor beats first-quoted SHA)
- clear block: COMMENTED, READY-only, excludes LBO/UNVERIFIED
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = PROJECT_ROOT / "scripts" / "tools" / "memory_hygiene.py"


def _load_tool() -> ModuleType:
    """Fresh load of the tool (and its embedded `_CAP`) so globals are rebindable."""
    spec = importlib.util.spec_from_file_location("memory_hygiene_under_test", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repo_with_origin(tmp_path: Path) -> tuple[Path, str, str]:
    """A git repo with origin/main set; returns (repo, merged_sha, unmerged_sha)."""
    repo = tmp_path / "wt"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "--initial-branch=main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "a.txt").write_text("1")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "c1"], cwd=repo, check=True)
    merged_sha = _run(repo, "rev-parse", "HEAD")
    subprocess.run(["git", "update-ref", "refs/remotes/origin/main", merged_sha], cwd=repo, check=True)
    (repo / "b.txt").write_text("2")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "c2"], cwd=repo, check=True)
    unmerged_sha = _run(repo, "rev-parse", "HEAD")
    return repo, merged_sha, unmerged_sha


def _bind_memory(mod: ModuleType, mem_dir: Path) -> None:
    """Point the tool's budget + baton reads at a tmp memory dir."""
    mod.MEMORY_DIR = mem_dir
    mod.MEMORY_MD = mem_dir / "MEMORY.md"


def _bind_git(mod: ModuleType, repo: Path) -> None:
    """Redirect the underlying `_sha_on_origin_main` git calls to `repo`."""
    mod._CAP.PROJECT_ROOT = repo


# --------------------------------------------------------------------------- #
# Budget report
# --------------------------------------------------------------------------- #
def test_missing_memory_md_reports_not_exists(tmp_path):
    mod = _load_tool()
    _bind_memory(mod, tmp_path)  # no MEMORY.md written
    report = mod.budget_report()
    assert report["exists"] is False
    assert report["path"].endswith("MEMORY.md")


def test_whole_index_loads_within_both_budgets(tmp_path):
    mod = _load_tool()
    _bind_memory(mod, tmp_path)
    (tmp_path / "MEMORY.md").write_text("\n".join(f"- line {i}" for i in range(50)))
    report = mod.budget_report()
    assert report["exists"] is True
    assert report["over_line_budget"] is False
    assert report["over_byte_budget"] is False
    assert report["first_dropped"] is None


def test_line_cutoff_binds_when_bytes_stay_small(tmp_path):
    mod = _load_tool()
    _bind_memory(mod, tmp_path)
    # 250 tiny lines: well under 25 KB, but over the 200-line budget.
    (tmp_path / "MEMORY.md").write_text("\n".join("x" for _ in range(250)))
    report = mod.budget_report()
    assert report["over_line_budget"] is True
    # First dropped line is the one past the 200-line budget, by "lines".
    assert report["first_dropped"] == {"line": mod.LINE_BUDGET + 1, "by": "lines"}


def test_byte_cutoff_binds_before_line_cutoff(tmp_path):
    mod = _load_tool()
    _bind_memory(mod, tmp_path)
    # 100 fat lines blow past 25 KB before line 200. write_bytes (NOT write_text)
    # so the on-disk newline is a deterministic 1-byte LF — write_text would let
    # Windows text-mode translate \n -> \r\n (302 B/line), making expected_line
    # platform-dependent. The tool reads RAW bytes, so the test must pin them.
    fat = b"y" * 300
    (tmp_path / "MEMORY.md").write_bytes(b"\n".join(fat for _ in range(100)))
    report = mod.budget_report()
    assert report["over_byte_budget"] is True
    dropped = report["first_dropped"]
    assert dropped is not None
    assert dropped["by"] == "bytes"
    # Interior line = 301 bytes (300 + LF); cutoff at 25,000 -> (25000//301)+1.
    expected_line = (mod.HARD_BYTE_CUTOFF // 301) + 1
    assert dropped["line"] == expected_line
    assert dropped["line"] < mod.LINE_BUDGET  # bytes bound first, not lines


def test_overlong_lines_flagged_headers_and_blanks_skipped(tmp_path):
    mod = _load_tool()
    _bind_memory(mod, tmp_path)
    long_line = "- " + ("z" * (mod.OVERLONG_LINE_CHARS + 5))  # > guideline
    header = "## " + ("h" * (mod.OVERLONG_LINE_CHARS + 5))  # long but a header
    content = "\n".join(["- short", "", header, long_line])
    (tmp_path / "MEMORY.md").write_text(content)
    report = mod.budget_report()
    flagged_lines = {o["line"] for o in report["overlong_lines"]}
    assert 4 in flagged_lines  # the long bullet
    assert 3 not in flagged_lines  # header skipped despite length
    assert 1 not in flagged_lines  # short bullet
    assert 2 not in flagged_lines  # blank


# --------------------------------------------------------------------------- #
# Baton tiering
# --------------------------------------------------------------------------- #
def test_ready_merged_sha_no_open_marker(repo_with_origin, tmp_path):
    repo, merged_sha, _ = repo_with_origin
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "project_x.md").write_text(f"DONE+PUSHED — landed on origin/main `{merged_sha}`. All tests pass.")
    mod = _load_tool()
    _bind_memory(mod, mem)
    _bind_git(mod, repo)
    result = mod.tier_baton(mem / "project_x.md")
    assert result["tier"] == "READY"
    assert result["primary_sha"] == merged_sha
    assert result["primary_merged"] is True
    assert result["has_open_marker"] is False


def test_landed_but_open_marker_present(repo_with_origin, tmp_path):
    repo, merged_sha, _ = repo_with_origin
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "project_y.md").write_text(
        f"DONE+PUSHED `{merged_sha}`. ▶ NEXT = wire the dashboard selector.",
        encoding="utf-8",
    )
    mod = _load_tool()
    _bind_memory(mod, mem)
    _bind_git(mod, repo)
    result = mod.tier_baton(mem / "project_y.md")
    assert result["tier"] == "LANDED-BUT-OPEN"
    assert result["primary_merged"] is True
    assert result["has_open_marker"] is True


def test_unverified_unmerged_primary_sha(repo_with_origin, tmp_path):
    repo, _, unmerged_sha = repo_with_origin
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "project_z.md").write_text(f"WIP — committed local-only `{unmerged_sha}`, not pushed yet.")
    mod = _load_tool()
    _bind_memory(mod, mem)
    _bind_git(mod, repo)
    result = mod.tier_baton(mem / "project_z.md")
    assert result["tier"] == "UNVERIFIED"
    assert result["primary_merged"] is False
    assert result["merged_shas"] == []


def test_unverified_no_sha(repo_with_origin, tmp_path):
    repo, _, _ = repo_with_origin
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "project_nosha.md").write_text("A design idea with no commit yet. ▶ TODO.", encoding="utf-8")
    mod = _load_tool()
    _bind_memory(mod, mem)
    _bind_git(mod, repo)
    result = mod.tier_baton(mem / "project_nosha.md")
    assert result["tier"] == "UNVERIFIED"
    assert result["primary_sha"] is None
    assert result["shas"] == []


def test_primary_sha_anchor_beats_first_quoted(repo_with_origin, tmp_path):
    """The DONE+PUSHED-anchored SHA is primary even when an earlier SHA appears."""
    repo, merged_sha, unmerged_sha = repo_with_origin
    mem = tmp_path / "memory"
    mem.mkdir()
    # Earlier (first) SHA is the unmerged one; the anchored headline SHA is merged.
    (mem / "project_anchor.md").write_text(
        f"Recovery ref `{unmerged_sha}` kept for safety.\nDONE+PUSHED origin/main=`{merged_sha}`."
    )
    mod = _load_tool()
    _bind_memory(mod, mem)
    _bind_git(mod, repo)
    result = mod.tier_baton(mem / "project_anchor.md")
    assert result["primary_sha"] == merged_sha
    assert result["tier"] == "READY"
    # Both SHAs are surfaced as evidence; only the merged one is in merged_shas.
    assert unmerged_sha in result["shas"]
    assert merged_sha in result["merged_shas"]
    assert unmerged_sha not in result["merged_shas"]


def test_tier_all_sorts_ready_first(repo_with_origin, tmp_path):
    repo, merged_sha, unmerged_sha = repo_with_origin
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "project_unverified.md").write_text(f"local-only `{unmerged_sha}`.")
    (mem / "project_ready.md").write_text(f"DONE+PUSHED `{merged_sha}`.")
    mod = _load_tool()
    _bind_memory(mod, mem)
    _bind_git(mod, repo)
    rows = mod.tier_all_batons()
    assert [r["tier"] for r in rows] == ["READY", "UNVERIFIED"]


# --------------------------------------------------------------------------- #
# Clear block — COMMENTED, READY-only
# --------------------------------------------------------------------------- #
def test_clear_block_is_fully_commented_and_ready_only(repo_with_origin, tmp_path):
    repo, merged_sha, unmerged_sha = repo_with_origin
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "project_ready.md").write_text(f"DONE+PUSHED `{merged_sha}`.")
    (mem / "project_lbo.md").write_text(f"DONE+PUSHED `{merged_sha}`. ▶ NEXT more.", encoding="utf-8")
    (mem / "project_unverified.md").write_text(f"local `{unmerged_sha}`.")
    mod = _load_tool()
    _bind_memory(mod, mem)
    _bind_git(mod, repo)
    batons = mod.tier_all_batons()
    block = mod._render_clear_block(batons)
    # Every non-empty line is a comment — nothing executes if pasted.
    for line in block.splitlines():
        if line.strip():
            assert line.lstrip().startswith("#"), f"non-comment line leaked: {line!r}"
    # Only the READY baton's rm appears; LBO/UNVERIFIED excluded.
    assert "project_ready.md" in block
    assert "project_lbo.md" not in block
    assert "project_unverified.md" not in block


def test_clear_block_empty_when_no_ready(repo_with_origin, tmp_path):
    repo, _, unmerged_sha = repo_with_origin
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "project_unverified.md").write_text(f"local `{unmerged_sha}`.")
    mod = _load_tool()
    _bind_memory(mod, mem)
    _bind_git(mod, repo)
    batons = mod.tier_all_batons()
    block = mod._render_clear_block(batons)
    assert "nothing safe to clear" in block


# --------------------------------------------------------------------------- #
# CRLF byte-count regression (the real MEMORY.md on Windows uses CRLF).
# Grounded against the official cutoff "first 200 lines OR 25KB" — the loader
# sees RAW on-disk bytes, so a CRLF `\r` must be counted. The pre-fix tool
# stripped `\r` via splitlines(), UNDER-reporting size = false-PASS direction.
# Fixtures write explicit bytes so the CRLF survives regardless of platform.
# --------------------------------------------------------------------------- #
def test_byte_count_matches_raw_on_disk_crlf(tmp_path):
    mod = _load_tool()
    _bind_memory(mod, tmp_path)
    md = tmp_path / "MEMORY.md"
    md.write_bytes(b"\r\n".join(b"- line %d" % i for i in range(40)) + b"\r\n")
    report = mod.budget_report()
    # The tool must report the file as it sits on disk — \r included.
    assert report["bytes"] == len(md.read_bytes())


def test_first_dropped_line_crlf_byte_cutoff(tmp_path):
    mod = _load_tool()
    _bind_memory(mod, tmp_path)
    md = tmp_path / "MEMORY.md"
    # Few FAT CRLF lines so the BYTE cutoff (25 KB) is crossed well before the
    # 200-line cutoff — otherwise the line branch would mask the byte branch.
    fat = b"y" * 200  # 202 bytes/line incl. \r\n
    n_lines = 150  # 150 * 202 = 30,300 > HARD_BYTE_CUTOFF, in < 200 lines
    md.write_bytes(b"\r\n".join(fat for _ in range(n_lines)) + b"\r\n")
    report = mod.budget_report()
    dropped = report["first_dropped"]
    assert dropped is not None
    assert dropped["by"] == "bytes"
    assert dropped["line"] < mod.LINE_BUDGET  # bytes bound first
    # CRLF line is 202 bytes; cutoff at 25,000 -> ceil-style boundary.
    expected_line = (mod.HARD_BYTE_CUTOFF // 202) + 1
    assert dropped["line"] == expected_line
