"""Tests for the stat-claim literature-anchor gate in check_claim_hygiene.

The gate makes the seven-sins / institutional-rigor literature canon ENFORCED at
commit time: a staged doc making a Sharpe/significance/edge claim must carry a
literature anchor / criterion / result path, or be tagged MEASURED/UNSUPPORTED.

Scope is staged-only and repo-wide across docs/audit/, docs/institutional/,
docs/plans/, research/. Historical (unstaged) docs are grandfathered — the
pre-commit only ever passes changed files to the checker, so "old docs ignored"
is enforced by the harness, and "outside scope ignored" is enforced by
_in_stat_claim_scope (tested here directly).
"""

from __future__ import annotations

from pathlib import Path

from scripts.tools import check_claim_hygiene as cch

# A doc body that asserts a Sharpe NUMBER (fires _STAT_CLAIM_PATTERN).
_CLAIM_BODY = "# Finding\n\nThe US lane shows a Sharpe = 1.82 over the window.\n"


def _write(root: Path, rel: str, body: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


def _scoped(root: Path, body: str) -> Path:
    """Write a doc UNDER an in-scope folder, with PROJECT_ROOT patched to root."""
    return _write(root, "docs/audit/results/2026-06-05-x.md", body)


def test_staged_stat_claim_without_anchor_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(cch, "PROJECT_ROOT", tmp_path)
    doc = _scoped(tmp_path, _CLAIM_BODY)
    issues = cch.check_stat_claim_anchor(doc)
    assert issues, "ungrounded Sharpe claim must FAIL"
    assert "no literature anchor" in issues[0]


def test_same_claim_with_literature_anchor_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(cch, "PROJECT_ROOT", tmp_path)
    body = _CLAIM_BODY + "\nGrounded in `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md`.\n"
    doc = _scoped(tmp_path, body)
    assert cch.check_stat_claim_anchor(doc) == [], "literature-cited claim must PASS"


def test_claim_with_named_canon_author_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(cch, "PROJECT_ROOT", tmp_path)
    body = _CLAIM_BODY + "\nClears the Chordia (2018) t>=3.79 bound.\n"
    doc = _scoped(tmp_path, body)
    assert cch.check_stat_claim_anchor(doc) == [], "named-canon-author claim must PASS"


def test_claim_with_executed_result_path_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(cch, "PROJECT_ROOT", tmp_path)
    body = _CLAIM_BODY + "\nReproduced from `docs/audit/results/2026-06-04-foo.md`.\n"
    doc = _scoped(tmp_path, body)
    assert cch.check_stat_claim_anchor(doc) == [], "executed-result-path claim must PASS"


def test_unsupported_tag_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(cch, "PROJECT_ROOT", tmp_path)
    body = _CLAIM_BODY + "\nUNSUPPORTED: not yet verified against local literature.\n"
    doc = _scoped(tmp_path, body)
    assert cch.check_stat_claim_anchor(doc) == [], "UNSUPPORTED-tagged claim must PASS"


def test_measured_tag_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(cch, "PROJECT_ROOT", tmp_path)
    body = _CLAIM_BODY + "\nMEASURED: computed directly from the trade book.\n"
    doc = _scoped(tmp_path, body)
    assert cch.check_stat_claim_anchor(doc) == [], "MEASURED-tagged claim must PASS"


def test_methodology_prose_without_number_does_not_fire(tmp_path, monkeypatch):
    """No false positive: a design doc that merely mentions Sharpe with no number."""
    monkeypatch.setattr(cch, "PROJECT_ROOT", tmp_path)
    body = "# Plan\n\nWe should compute the Sharpe ratio and a deflated variant later.\n"
    doc = _scoped(tmp_path, body)
    assert cch.check_stat_claim_anchor(doc) == [], "prose-without-number must not fire"


def test_doc_outside_scope_is_ignored(tmp_path, monkeypatch):
    """A README / src doc outside the scoped folders is not subject to the gate."""
    monkeypatch.setattr(cch, "PROJECT_ROOT", tmp_path)
    doc = _write(tmp_path, "README.md", _CLAIM_BODY)
    assert cch._in_stat_claim_scope(doc) is False
    # main() only calls check_stat_claim_anchor when _in_stat_claim_scope is True,
    # so an out-of-scope ungrounded claim does not block.


def test_in_scope_detection_repo_wide(tmp_path, monkeypatch):
    monkeypatch.setattr(cch, "PROJECT_ROOT", tmp_path)
    for rel in (
        "docs/audit/results/a.md",
        "docs/institutional/b.md",
        "docs/plans/c.md",
        "research/d.md",
    ):
        p = _write(tmp_path, rel, "x")
        assert cch._in_stat_claim_scope(p) is True, f"{rel} should be in scope"
    # non-md and out-of-scope
    assert cch._in_stat_claim_scope(_write(tmp_path, "docs/audit/e.txt", "x")) is False
    assert cch._in_stat_claim_scope(_write(tmp_path, "src/f.md", "x")) is False


def test_main_blocks_ungrounded_and_passes_grounded(tmp_path, monkeypatch):
    """End-to-end through main(): exit 1 ungrounded, exit 0 grounded."""
    monkeypatch.setattr(cch, "PROJECT_ROOT", tmp_path)
    bad = _scoped(tmp_path, _CLAIM_BODY)
    assert cch.main([str(bad)]) == 1
    good = _write(
        tmp_path,
        "docs/institutional/ok.md",
        _CLAIM_BODY + "\nUNSUPPORTED pending extract.\n",
    )
    assert cch.main([str(good)]) == 0
