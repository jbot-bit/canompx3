"""Tests for the stat-claim literature-anchor gate in check_claim_hygiene.

The gate makes the seven-sins / institutional-rigor literature canon ENFORCED at
commit time: a staged doc making a Sharpe/significance/edge claim must carry a
literature anchor / criterion / result path, or be tagged MEASURED/UNSUPPORTED.

Scope is staged-only and repo-wide across docs/audit/, docs/institutional/,
docs/plans/, research/. Two layers enforce this:
  * The pre-commit [7/8] stage greps the staged set for those four folder
    prefixes and passes every matching .md to the checker (the harness layer —
    "old/unstaged docs ignored" because only changed files are passed).
  * check_claim_hygiene.main() routes each passed path by _in_stat_claim_scope,
    so "outside-scope ignored" holds even if the harness over-delivers (tested
    here directly via test_doc_outside_scope_is_ignored).
The wiring contract (hook delivers all four prefixes) is asserted by
test_precommit_feeds_all_scope_folders.
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
    body = (
        _CLAIM_BODY + "\nGrounded in `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md`.\n"
    )
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


def test_precommit_feeds_all_scope_folders():
    """Wiring contract: the pre-commit [7/8] stage must deliver ALL four scope
    folders to the checker, not just docs/audit/results/. Regression guard for
    the gap where 3 of 4 _STAT_CLAIM_DIRS were dead at commit time because the
    hook over-filtered to docs/audit/results/ only.

    Parses the committed grep pattern out of .githooks/pre-commit and asserts a
    representative path under each scope folder matches it. This proves the
    harness reaches every folder _in_stat_claim_scope accepts — closing the
    "function works but the hook never calls it" sin.
    """
    import re

    project_root = Path(__file__).resolve().parents[2]
    hook = (project_root / ".githooks" / "pre-commit").read_text(encoding="utf-8")

    # Pull the grep -E '...' pattern from the [7/8] STAGED_RESULT_DOCS assignment.
    m = re.search(r"STAGED_RESULT_DOCS=\$\(echo \"\$STAGED_ALL\" \| grep -E '([^']+)'", hook)
    assert m, "could not locate the [7/8] claim-hygiene grep pattern in .githooks/pre-commit"
    pattern = re.compile(m.group(1))

    # Every folder the checker accepts must be reachable by the hook's filter.
    representative = {
        ("docs", "audit"): "docs/audit/results/2026-06-05-x.md",
        ("docs", "institutional"): "docs/institutional/mechanism_priors.md",
        ("docs", "plans"): "docs/plans/2026-06-05-x.md",
        ("research",): "research/some_scan_v1.md",
    }
    for prefix in cch._STAT_CLAIM_DIRS:
        path = representative[prefix]
        assert pattern.search(path), f"{path} (scope {prefix}) is NOT delivered by the pre-commit grep"

    # And a non-scope .md must NOT be delivered (no over-broad blocking).
    assert not pattern.search("src/foo.md")
    assert not pattern.search("README.md")
