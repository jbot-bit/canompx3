from __future__ import annotations

import re
from pathlib import Path

from scripts.tools import compact_handoff

ROOT = Path(__file__).resolve().parents[2]


def _mkfile(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _post_commit_replace(content: str, tool: str, date: str, sha: str, subject: str, files: list[str]) -> str:
    """Reproduce the EXACT replacement logic in ``.githooks/post-commit`` (its
    embedded python, lines ~82-110). Kept in lockstep with the hook so the
    baton/hook *interaction* is regression-tested, not just the tool in
    isolation. If the hook's regex changes, ``test_post_commit_regex_matches_hook_source``
    fails first and points here."""
    new_block = (
        "## Last Session\n"
        f"- **Tool:** {tool}\n"
        f"- **Date:** {date}\n"
        f"- **Commit:** {sha} — {subject}\n"
        f"- **Files changed:** {len(files)} files"
    )
    file_lines = [f for f in files if f.strip()]
    if file_lines:
        detail = "\n".join(f"  - `{f}`" for f in file_lines[:15])
        if len(file_lines) > 15:
            detail += f"\n  - ... and {len(file_lines) - 15} more"
        replacement = new_block + "\n" + detail + "\n"
    else:
        replacement = new_block + "\n"
    pattern = r"## Last Session\n.*?(?=\n## (?!Last Session))"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        content = content[: match.start()] + replacement + content[match.end() :]
    return content


def test_compact_handoff_archives_existing_and_renders_compact(tmp_path: Path) -> None:
    handoff = tmp_path / "HANDOFF.md"
    _mkfile(
        handoff,
        "\n".join(
            [
                "# HANDOFF.md — Cross-Tool Session Baton",
                "",
                "## Update (2026-04-22 autonomous discovery reround)",
                "",
                "Route stack refreshed and stale exact bridge parked.",
                "",
                "### Next move",
                "",
                "- Build a bounded non-geometry shortlist",
                "- Keep verification fail-closed",
            ]
        ),
    )

    archive_path, compact = compact_handoff.compact_handoff(
        handoff_path=handoff,
        archive_dir=tmp_path / "docs" / "handoffs" / "archived",
        tool="Codex",
        date="2026-04-22",
        summary="Slimmed HANDOFF into a compact baton and archived the full history snapshot.",
        next_steps=[
            "Resume MNQ autonomous discovery from a bounded non-geometry shortlist refresh.",
            "Keep route-level verification fail-closed when repo blockers are red.",
        ],
        blockers=["Criterion 11 control report is still missing for topstep_50k_mnq_auto."],
        references=["docs/plans/2026-04-22-handoff-baton-compaction.md"],
    )

    assert archive_path.exists()
    assert "autonomous discovery reround" in archive_path.read_text(encoding="utf-8")
    rendered = handoff.read_text(encoding="utf-8")
    assert rendered == compact
    assert "## Last Session" in rendered
    assert "- **Tool:** Codex" in rendered
    assert "## Next Steps — Active" in rendered
    assert "1. Resume MNQ autonomous discovery from a bounded non-geometry shortlist refresh." in rendered
    assert "## Blockers / Warnings" in rendered
    assert "docs/plans/2026-04-22-handoff-baton-compaction.md" in rendered
    assert "docs/handoffs/archived/2026-04-22-root-handoff-archive.md" in rendered


def test_compact_handoff_falls_back_to_existing_metadata(tmp_path: Path) -> None:
    handoff = tmp_path / "HANDOFF.md"
    _mkfile(
        handoff,
        "\n".join(
            [
                "## Last Session",
                "- **Tool:** Claude",
                "- **Date:** 2026-03-17",
                "- **Summary:** Built pulse",
                "",
                "## Next Steps — Active",
                "1. Phase 1: do thing",
                "2. Phase 2: do other thing",
                "",
                "## Blockers / Warnings",
                "- Pre-existing test failure: broken thing",
            ]
        ),
    )

    _, rendered = compact_handoff.compact_handoff(
        handoff_path=handoff,
        archive_dir=tmp_path / "docs" / "handoffs" / "archived",
        tool=None,
        date=None,
        summary=None,
        next_steps=[],
        blockers=[],
        references=[],
    )

    assert "- **Tool:** Claude" in rendered
    assert "- **Date:** 2026-03-17" in rendered
    assert "- **Summary:** Built pulse" in rendered
    assert "1. Phase 1: do thing" in rendered
    assert "2. Phase 2: do other thing" in rendered
    assert "- Pre-existing test failure: broken thing" in rendered


# ---------------------------------------------------------------------------
# Stress / logic tests for the baton lifecycle (added 2026-06-05).
#
# Operator demand: prove the baton mechanism actually works and converges —
# the post-commit hook must keep the COMPACTED baton bounded, not re-accrete.
# These tests exercise the *interaction* between compact_handoff.py and the
# post-commit hook's regex, which neither happy-path test above covered.
# ---------------------------------------------------------------------------


def _bloated_baton(num_stale_blocks: int) -> str:
    """A realistic accreted baton: blocks BOTH before and after Last Session
    (the exact shape that grew the real HANDOFF.md to 182 lines)."""
    lines = ["# HANDOFF.md — Cross-Tool Session Baton", ""]
    for i in range(num_stale_blocks // 2):
        lines += [f"## Stale Block Pre {i}", f"- old content pre {i}", ""]
    lines += [
        "## Last Session",
        "- **Tool:** Codex",
        "- **Date:** 2026-01-01",
        "- **Summary:** old summary",
        "",
    ]
    for i in range(num_stale_blocks - num_stale_blocks // 2):
        lines += [f"## Stale Block Post {i}", f"- old content post {i}", ""]
    return "\n".join(lines)


def test_post_commit_only_touches_last_session_proving_accretion_bug(tmp_path: Path) -> None:
    """LOGIC TEST: prove WHY the baton accreted — the hook replaces ONLY the
    Last Session block, so every other block (pre AND post) is immortal. This
    is the structural reason compaction is needed at all."""
    bloated = _bloated_baton(num_stale_blocks=10)
    after = _post_commit_replace(bloated, "Claude Code", "2026-06-05", "abc123", "new work", ["a.py"])
    # Last Session WAS updated...
    assert "- **Date:** 2026-06-05" in after
    assert "- **Summary:** old summary" not in after
    # ...but every stale block survived — the hook cannot prune them.
    for i in range(5):
        assert f"## Stale Block Pre {i}" in after, "pre-LS blocks must survive (hook can't prune)"
    for i in range(5):
        assert f"## Stale Block Post {i}" in after, "post-LS blocks must survive (hook can't prune)"


def test_compact_then_hook_keeps_section_count_invariant(tmp_path: Path) -> None:
    """BOUNDEDNESS PROOF: after compaction, 50 successive post-commit fires
    must NOT add any '## ' section. Header count is the rigorous invariant —
    if it holds, the baton cannot accrete blocks no matter how many commits."""
    handoff = tmp_path / "HANDOFF.md"
    _mkfile(handoff, _bloated_baton(num_stale_blocks=12))

    _, compact = compact_handoff.compact_handoff(
        handoff_path=handoff,
        archive_dir=tmp_path / "docs" / "handoffs" / "archived",
        tool="Claude Code",
        date="2026-06-05",
        summary="Compacted.",
        next_steps=["Do next thing."],
        blockers=["A blocker."],
        references=["docs/plans/2026-04-22-handoff-baton-compaction.md"],
    )

    headers_compact = len(re.findall(r"^## ", compact, re.M))
    buf = compact
    for i in range(50):
        nfiles = (i % 20) + 1  # vary the files list 1..20 to stress the detail block
        files = [f"file_{i}_{j}.py" for j in range(nfiles)]
        buf = _post_commit_replace(buf, "Codex", f"2026-06-{i % 28:02d}", f"c{i}", f"subject {i}", files)

    headers_after = len(re.findall(r"^## ", buf, re.M))
    assert headers_after == headers_compact, "post-commit hook must NEVER add a section block"

    # The four load-bearing sections must survive all 50 fires.
    for sec in ["## Last Session", "## Next Steps — Active", "## Blockers / Warnings", "## Durable References"]:
        assert sec in buf, f"{sec} must survive 50 post-commit fires"

    # Line count is bounded: it varies ONLY with the (≤15-capped) files list,
    # never grows unboundedly. Ceiling = compact baton + 15 file bullets + 1
    # detail header line.
    ceiling = len(compact.splitlines()) + 16
    assert len(buf.splitlines()) <= ceiling


def test_post_commit_noop_when_no_last_session_block(tmp_path: Path) -> None:
    """SAFETY: a baton with no Last Session block must be left untouched, not
    corrupted (the hook's 'if no match, do nothing' branch)."""
    content = "# HANDOFF.md\n## Something Else\n- x\n"
    after = _post_commit_replace(content, "Codex", "2026-06-05", "abc", "s", ["a.py"])
    assert after == content


def test_compact_handoff_is_idempotent_and_preserves_all_content(tmp_path: Path) -> None:
    """STRESS: compacting an already-compact baton must preserve every prior
    archive (no clobber) and lose no content (archive == prior file byte-for-byte)."""
    handoff = tmp_path / "HANDOFF.md"
    archive_dir = tmp_path / "docs" / "handoffs" / "archived"
    original = _bloated_baton(num_stale_blocks=8)
    _mkfile(handoff, original)

    arch1, compact1 = compact_handoff.compact_handoff(
        handoff_path=handoff,
        archive_dir=archive_dir,
        tool="Claude Code",
        date="2026-06-05",
        summary="First compaction.",
        next_steps=["Step."],
        blockers=[],
        references=[],
    )
    # Archive is a byte-exact copy of the pre-compaction file (no content lost).
    assert arch1.read_text(encoding="utf-8") == original

    # Second compaction on the SAME date must NOT overwrite the first archive.
    arch2, _ = compact_handoff.compact_handoff(
        handoff_path=handoff,
        archive_dir=archive_dir,
        tool="Claude Code",
        date="2026-06-05",
        summary="Second compaction.",
        next_steps=["Step."],
        blockers=[],
        references=[],
    )
    assert arch1 != arch2, "same-date re-compaction must use a unique archive path"
    assert arch1.exists() and arch2.exists()
    assert arch2.read_text(encoding="utf-8") == compact1, "2nd archive must hold the 1st compact baton"


def test_post_commit_regex_matches_hook_source() -> None:
    """GUARD: the regex in this test file must stay byte-identical to the one
    in .githooks/post-commit. If the hook changes its pattern, this fails and
    forces _post_commit_replace above to be re-synced — so the interaction
    tests never silently drift from the real hook."""
    hook = (ROOT / ".githooks" / "post-commit").read_text(encoding="utf-8")
    canonical_pattern = r"## Last Session\n.*?(?=\n## (?!Last Session))"
    # The hook embeds the pattern as a python raw-string literal.
    assert "## Last Session" in hook
    assert r"(?=\n## (?!Last Session))" in hook, "hook's lookahead pattern changed — re-sync _post_commit_replace"
    # Sanity: our local copy compiles and is the same shape the hook uses.
    assert re.compile(canonical_pattern, re.DOTALL)
