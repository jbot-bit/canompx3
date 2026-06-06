"""Tests for the autopilot post-build diff review tool.

Locks in:
  - dedupe-by-hash skips an identical hunk on a second pass,
  - a high-risk (Tier-B) file gets the FULL diff while a normal file gets a
    capped snippet,
  - risk labels are correct,
  - commit_safe is False when a high-risk file is present.

These tests drive `review.review()` with a stubbed `_git` so they need no real
repo state, no network, and no DB.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "autopilot"))

import review_diff as review  # noqa: E402


def _make_diff(path: str, n_added: int) -> str:
    body = "\n".join(f"+line {i}" for i in range(n_added))
    return f"@@ -0,0 +1,{n_added} @@\n{body}"


@pytest.fixture
def stub_git(monkeypatch):
    """Stub the git-touching helpers with a controllable in-memory diff set."""
    state = {"files": {}}  # path -> (diff_text, added, removed)

    def set_files(files: dict):
        state["files"] = files

    monkeypatch.setattr(review, "_changed_files", lambda base: list(state["files"].keys()))
    monkeypatch.setattr(review, "_file_diff", lambda base, p: state["files"][p][0])
    monkeypatch.setattr(review, "_numstat", lambda base, p: (state["files"][p][1], state["files"][p][2]))
    return set_files


def test_risk_labels_and_commit_safety(stub_git):
    stub_git(
        {
            "docs/notes.md": (_make_diff("docs/notes.md", 3), 3, 0),
            "trading_app/prop_profiles.py": (_make_diff("trading_app/prop_profiles.py", 2), 2, 0),
        }
    )
    out = review.review("HEAD", set())
    by_path = {f["path"]: f for f in out["files"]}
    assert by_path["docs/notes.md"]["tier"] == "A"
    assert by_path["trading_app/prop_profiles.py"]["tier"] == "B"
    assert "trading_app/prop_profiles.py" in out["high_risk"]
    assert out["commit_safe"] is False


def test_commit_safe_when_no_high_risk(stub_git):
    stub_git({"docs/a.md": (_make_diff("docs/a.md", 2), 2, 0)})
    out = review.review("HEAD", set())
    assert out["commit_safe"] is True
    assert out["high_risk"] == []


def test_dedupe_by_hash_across_passes(stub_git):
    stub_git({"docs/a.md": (_make_diff("docs/a.md", 4), 4, 0)})
    seen: set[str] = set()
    first = review.review("HEAD", seen)
    assert first["new_hunks"] == 1
    # Second pass with the SAME seen set + identical diff -> nothing new.
    second = review.review("HEAD", seen)
    assert second["new_hunks"] == 0
    assert second["files"] == []


def test_untracked_new_file_is_reviewed(tmp_path, monkeypatch):
    """A brand-new (untracked) file must be picked up — git diff HEAD hides it,
    but the runner commits with `git add -A`, so review MUST see it. Uses a real
    temp file + stubbed git plumbing to exercise the synthesize-diff path."""
    new_file = tmp_path / "brand_new.md"
    new_file.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    monkeypatch.setattr(review, "_git", lambda args: "")  # no tracked changes
    monkeypatch.setattr(review, "_untracked_files", lambda: ["brand_new.md"])
    monkeypatch.setattr(review, "PROJECT_ROOT", tmp_path)

    out = review.review("HEAD", set())
    paths = {f["path"]: f for f in out["files"]}
    assert "brand_new.md" in paths
    assert paths["brand_new.md"]["added"] == 3
    assert "+alpha" in paths["brand_new.md"]["snippet"]
    assert out["new_hunks"] >= 1


def test_high_risk_gets_full_diff_normal_gets_snippet(stub_git):
    big = 50  # > SNIPPET_MAX_LINES
    stub_git(
        {
            "docs/big.md": (_make_diff("docs/big.md", big), big, 0),
            "pipeline/dst.py": (_make_diff("pipeline/dst.py", big), big, 0),
        }
    )
    out = review.review("HEAD", set())
    by_path = {f["path"]: f for f in out["files"]}
    # Normal file snippet is capped.
    normal_snip = by_path["docs/big.md"]["snippet"]
    assert "more lines)" in normal_snip
    assert normal_snip.count("\n") <= review.SNIPPET_MAX_LINES + 1
    # High-risk file gets the full diff (no truncation marker, all lines).
    risk_snip = by_path["pipeline/dst.py"]["snippet"]
    assert "more lines)" not in risk_snip
    assert risk_snip.count("+line") == big
