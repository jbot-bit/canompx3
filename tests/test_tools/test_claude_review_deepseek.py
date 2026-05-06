"""Tests for scripts/tools/claude_review_deepseek.py.

Focus: the canonical-source allowlist closes the capital-class review
bypass (PR #247 follow-up). A doc-only or <5-line change to
docs/runtime/lane_allocation.json (or any other allowlisted file) MUST
trigger Claude review, not silently skip.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tools.claude_review_deepseek import (  # noqa: E402
    _CANONICAL_SOURCE_ALLOWLIST,
    _diff_files,
    _diff_is_doc_only,
    _diff_touches_canonical_source,
    main,
)


# Diff fragments small enough to be self-contained and obviously valid.

DIFF_LANE_ALLOCATION_TINY = """\
diff --git a/docs/runtime/lane_allocation.json b/docs/runtime/lane_allocation.json
index abc123..def456 100644
--- a/docs/runtime/lane_allocation.json
+++ b/docs/runtime/lane_allocation.json
@@ -1,3 +1,3 @@
 {
-  "rebalanced_at": "2026-05-03T14:00:00Z"
+  "rebalanced_at": "2026-05-07T14:00:00Z"
 }
"""

DIFF_COST_MODEL_TINY = """\
+++ b/pipeline/cost_model.py
@@ +1 @@
+MNQ_FRICTION = 0.20
"""

DIFF_PURE_DOC_MD = """\
diff --git a/docs/audit/results/2026-05-07-foo.md b/docs/audit/results/2026-05-07-foo.md
new file mode 100644
index 0000000..abc123
--- /dev/null
+++ b/docs/audit/results/2026-05-07-foo.md
@@ -0,0 +1,5 @@
+# Audit result
+
+Findings: none.
+
+End.
"""

DIFF_README_TINY = """\
+++ b/README.md
@@ +1 @@
+new
"""

DIFF_LANE_ALLOCATION_DELETE = """\
diff --git a/docs/runtime/lane_allocation.json b/docs/runtime/lane_allocation.json
deleted file mode 100644
index abc123..0000000
--- a/docs/runtime/lane_allocation.json
+++ /dev/null
@@ -1,3 +0,0 @@
-{
-  "key": "value"
-}
"""

DIFF_NEW_FILE_NON_CANONICAL = """\
diff --git a/scripts/tools/new_helper.py b/scripts/tools/new_helper.py
new file mode 100644
index 0000000..abc123
--- /dev/null
+++ b/scripts/tools/new_helper.py
@@ -0,0 +1,3 @@
+def helper():
+    return 1
+
"""


# ---------- _diff_touches_canonical_source ----------


def test_diff_touches_canonical_source_lane_allocation():
    """A 3-line modify of lane_allocation.json must register as canonical."""
    assert _diff_touches_canonical_source(DIFF_LANE_ALLOCATION_TINY) is True


def test_diff_touches_canonical_source_cost_model():
    """A 1-line edit to pipeline/cost_model.py must register as canonical."""
    assert _diff_touches_canonical_source(DIFF_COST_MODEL_TINY) is True


def test_diff_touches_canonical_source_pure_doc_does_not():
    """An audit-result .md edit must NOT register as canonical."""
    assert _diff_touches_canonical_source(DIFF_PURE_DOC_MD) is False


def test_diff_touches_canonical_source_lane_allocation_delete():
    """Deleting lane_allocation.json must register as canonical (highest impact)."""
    assert _diff_touches_canonical_source(DIFF_LANE_ALLOCATION_DELETE) is True


def test_diff_touches_canonical_source_new_non_canonical_file():
    """A new helper script outside the allowlist must NOT register."""
    assert _diff_touches_canonical_source(DIFF_NEW_FILE_NON_CANONICAL) is False


def test_diff_files_skips_dev_null():
    """Diff parser must not pollute the file set with /dev/null."""
    files = _diff_files(DIFF_LANE_ALLOCATION_DELETE)
    assert files == {"docs/runtime/lane_allocation.json"}


# ---------- main() integration: canonical override ----------


def test_canonical_source_overrides_doc_only_skip(monkeypatch, capsys):
    """A doc-only diff that includes lane_allocation.json must reach Claude
    review (mocked to BLOCK), not short-circuit on the doc-only path."""
    monkeypatch.setenv("OPENCODE_AGENT_ACTIVE", "1")
    with patch(
        "scripts.tools.claude_review_deepseek._staged_diff",
        return_value=DIFF_LANE_ALLOCATION_TINY,
    ):
        rc = main_argv(["--mock", "--rubric-fail"])
    assert rc == 1, "lane_allocation.json edit must BLOCK on rubric-fail mock"
    err = capsys.readouterr().err
    assert "BLOCK" in err


def test_canonical_source_overrides_small_diff_skip(monkeypatch, capsys):
    """A 2-line edit to pipeline/cost_model.py must reach Claude review,
    not short-circuit on the <5-line check."""
    monkeypatch.setenv("OPENCODE_AGENT_ACTIVE", "1")
    # The diff is intentionally <5 lines total. Pre-fix this would exit 0;
    # post-fix it must BLOCK because cost_model is allowlisted.
    assert len(DIFF_COST_MODEL_TINY.splitlines()) < 5
    with patch(
        "scripts.tools.claude_review_deepseek._staged_diff",
        return_value=DIFF_COST_MODEL_TINY,
    ):
        rc = main_argv(["--mock", "--rubric-fail"])
    assert rc == 1, "cost_model.py edit must BLOCK on rubric-fail mock"


# ---------- main() integration: pre-existing skip paths preserved ----------


def test_doc_only_md_still_skips_when_no_canonical_source(monkeypatch):
    """A pure audit-result .md edit must still skip (pre-fix behavior preserved)."""
    monkeypatch.setenv("OPENCODE_AGENT_ACTIVE", "1")
    with patch(
        "scripts.tools.claude_review_deepseek._staged_diff",
        return_value=DIFF_PURE_DOC_MD,
    ):
        rc = main_argv([])
    assert rc == 0, "pure .md edit must still skip"


def test_small_pure_md_diff_still_skips(monkeypatch):
    """A 2-line README.md tweak must skip via small-diff path."""
    monkeypatch.setenv("OPENCODE_AGENT_ACTIVE", "1")
    assert len(DIFF_README_TINY.splitlines()) < 5
    with patch(
        "scripts.tools.claude_review_deepseek._staged_diff",
        return_value=DIFF_README_TINY,
    ):
        rc = main_argv([])
    assert rc == 0, "tiny README.md edit must skip"


def test_skip_when_agent_inactive(monkeypatch):
    """No OPENCODE_AGENT_ACTIVE => exit 0 immediately (belt-and-braces)."""
    monkeypatch.delenv("OPENCODE_AGENT_ACTIVE", raising=False)
    with patch(
        "scripts.tools.claude_review_deepseek._staged_diff",
        return_value=DIFF_LANE_ALLOCATION_TINY,
    ):
        rc = main_argv([])
    assert rc == 0


# ---------- Allowlist contract ----------


@pytest.mark.parametrize(
    "path",
    [
        # Capital-affecting docs/runtime structured data (production reads):
        "docs/runtime/lane_allocation.json",
        "docs/runtime/chordia_audit_log.yaml",
        # Pipeline canonical sources:
        "pipeline/cost_model.py",
        "pipeline/dst.py",
        "pipeline/asset_configs.py",
        "pipeline/paths.py",
        # Trading-app canonical sources:
        "trading_app/prop_profiles.py",
        "trading_app/lane_ctl.py",
        "trading_app/config.py",
        "trading_app/holdout_policy.py",
        "trading_app/eligibility/builder.py",
        "trading_app/entry_rules.py",
    ],
)
def test_canonical_source_path_in_allowlist(path):
    """Lock the contract at the test layer alongside the drift check."""
    assert path in _CANONICAL_SOURCE_ALLOWLIST, (
        f"{path} must remain in the canonical-source allowlist; "
        f"removing it would silently re-open the bypass closed by this PR."
    )


# ---------- helper ----------


def main_argv(argv: list[str]) -> int:
    """Invoke claude_review_deepseek.main() with a synthetic argv. The
    module reads sys.argv via argparse; patch it for the call."""
    with patch.object(sys, "argv", ["claude_review_deepseek"] + argv):
        return main()
