"""Tests for scripts/tools/commit_message_content_gate.py + .githooks/commit-msg.

Covers the hollow-commit prevention gate (2026-06-09 Stage 2). The gate BLOCKS a
commit whose message asserts a ``${...}`` placeholder that appears in ZERO ADDED
lines of the staged diff (the dd63be8b class). Conservative — calibrated to 0
false-blocks over real history.

Verifies, at the pure-function layer and the CLI/subprocess layer:
- hollow commit (placeholder in msg, 0 in additions)         -> BLOCK
- honest commit (placeholder present in additions)           -> allow
- no placeholder in message (the common case)                -> allow
- override marker [hollow-ack]                                -> allow
- GAP-2: placeholder only in a CONTEXT or REMOVED line        -> BLOCK (not present in additions)
- GAP-2: ``+++`` file-header line never counts as an addition
- empty / whitespace / malformed message / missing arg        -> fail-OPEN
- the real dd63be8b mechanism                                 -> BLOCK
- the live .githooks/commit-msg hook under core.hooksPath      -> blocks a real commit
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "scripts" / "tools" / "commit_message_content_gate.py"
HOOK_PATH = PROJECT_ROOT / ".githooks" / "commit-msg"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("commit_message_content_gate", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gate = _load_module()


# --- diff fixtures ----------------------------------------------------------

# A staged diff that ADDS the placeholder (honest commit).
_DIFF_ADDS_PLACEHOLDER = (
    "diff --git a/.claude/settings.json b/.claude/settings.json\n"
    "index 111..222 100644\n"
    "--- a/.claude/settings.json\n"
    "+++ b/.claude/settings.json\n"
    "@@ -1,2 +1,2 @@\n"
    '-      "command": "python C:/x/.claude/hooks/foo.py"\n'
    '+      "command": "python ${CLAUDE_PROJECT_DIR}/.claude/hooks/foo.py"\n'
)

# A staged diff that does NOT add the placeholder anywhere (hollow commit).
_DIFF_NO_PLACEHOLDER = (
    "diff --git a/README.md b/README.md\n"
    "index 111..222 100644\n"
    "--- a/README.md\n"
    "+++ b/README.md\n"
    "@@ -1 +1,2 @@\n"
    " existing line\n"
    "+a brand new line with no placeholder\n"
)

# The placeholder appears ONLY in a context line and a REMOVED line — never added.
_DIFF_PLACEHOLDER_CONTEXT_AND_REMOVED_ONLY = (
    "diff --git a/x.txt b/x.txt\n"
    "index 111..222 100644\n"
    "--- a/x.txt\n"
    "+++ b/x.txt\n"
    "@@ -1,3 +1,2 @@\n"
    " context uses ${CLAUDE_PROJECT_DIR} here\n"
    "-removed line also ${CLAUDE_PROJECT_DIR}\n"
    "+a replacement line without it\n"
)


# --- pure-function: extract_literal_tokens ----------------------------------


class TestExtract:
    def test_extracts_placeholder(self) -> None:
        assert gate.extract_literal_tokens("fix: use ${CLAUDE_PROJECT_DIR}") == ["${CLAUDE_PROJECT_DIR}"]

    def test_dedupes_and_order_stable(self) -> None:
        msg = "use ${A} then ${B} then ${A} again"
        assert gate.extract_literal_tokens(msg) == ["${A}", "${B}"]

    def test_no_placeholder_returns_empty(self) -> None:
        assert gate.extract_literal_tokens("plain subject, backtick `foo`, path a/b.py") == []

    def test_empty_message(self) -> None:
        assert gate.extract_literal_tokens("") == []

    def test_placeholder_in_comment_line_ignored(self) -> None:
        # git #-comment lines are not authored claims — must NOT be extracted.
        msg = "real subject\n# a comment mentioning ${CLAUDE_PROJECT_DIR}\n"
        assert gate.extract_literal_tokens(msg) == []

    def test_placeholder_below_scissors_ignored(self) -> None:
        # commit -v appends the diff below the scissors line; not authored content.
        msg = (
            "subject\n"
            "# ------------------------ >8 ------------------------\n"
            "diff --git a/x b/x\n"
            "+uses ${CLAUDE_PROJECT_DIR}\n"
        )
        assert gate.extract_literal_tokens(msg) == []

    def test_authored_placeholder_above_comments_still_extracted(self) -> None:
        msg = "fix: use ${CLAUDE_PROJECT_DIR}\n# instructional comment\n"
        assert gate.extract_literal_tokens(msg) == ["${CLAUDE_PROJECT_DIR}"]


# --- pure-function: added_lines (GAP-2 core) --------------------------------


class TestAddedLines:
    def test_keeps_added_strips_marker(self) -> None:
        out = gate.added_lines(_DIFF_ADDS_PLACEHOLDER)
        assert "${CLAUDE_PROJECT_DIR}" in out
        # the leading + is stripped
        assert not out.startswith("+")

    def test_excludes_plus_plus_plus_header(self) -> None:
        # +++ b/file headers must NOT count as added content
        diff = "+++ b/some/path.py\n+real added line\n"
        out = gate.added_lines(diff)
        assert "some/path.py" not in out
        assert "real added line" in out

    def test_context_and_removed_lines_excluded(self) -> None:
        out = gate.added_lines(_DIFF_PLACEHOLDER_CONTEXT_AND_REMOVED_ONLY)
        # placeholder lives only in context/removed -> not in added haystack
        assert "${CLAUDE_PROJECT_DIR}" not in out

    def test_empty_diff(self) -> None:
        assert gate.added_lines("") == ""


# --- pure-function: evaluate ------------------------------------------------


class TestEvaluate:
    def test_hollow_commit_blocks(self) -> None:
        block, unmatched = gate.evaluate("fix: switch to ${CLAUDE_PROJECT_DIR}", _DIFF_NO_PLACEHOLDER)
        assert block is True
        assert unmatched == ["${CLAUDE_PROJECT_DIR}"]

    def test_honest_commit_allows(self) -> None:
        block, unmatched = gate.evaluate("fix: switch to ${CLAUDE_PROJECT_DIR}", _DIFF_ADDS_PLACEHOLDER)
        assert block is False
        assert unmatched == []

    def test_no_token_allows(self) -> None:
        block, unmatched = gate.evaluate("chore: tidy README", _DIFF_NO_PLACEHOLDER)
        assert block is False
        assert unmatched == []

    def test_override_marker_allows(self) -> None:
        block, _ = gate.evaluate("fix: ${CLAUDE_PROJECT_DIR} [hollow-ack] (prose only)", _DIFF_NO_PLACEHOLDER)
        assert block is False

    def test_gap2_context_or_removed_only_blocks(self) -> None:
        # The hollow case: msg claims the placeholder, but it only appears in a
        # context/removed line, never added. MUST block.
        block, unmatched = gate.evaluate(
            "fix: introduce ${CLAUDE_PROJECT_DIR}",
            _DIFF_PLACEHOLDER_CONTEXT_AND_REMOVED_ONLY,
        )
        assert block is True
        assert unmatched == ["${CLAUDE_PROJECT_DIR}"]

    def test_empty_message_fails_open(self) -> None:
        assert gate.evaluate("", _DIFF_NO_PLACEHOLDER) == (False, [])

    def test_whitespace_message_fails_open(self) -> None:
        assert gate.evaluate("   \n  ", _DIFF_NO_PLACEHOLDER) == (False, [])

    def test_comment_only_placeholder_does_not_block(self) -> None:
        # The PROBE-4 false-block: placeholder appears ONLY in a #-comment line.
        # After git-cleanup stripping it is not an authored claim -> allow.
        msg = "real subject\n# explains ${CLAUDE_PROJECT_DIR} usage\n"
        block, unmatched = gate.evaluate(msg, _DIFF_NO_PLACEHOLDER)
        assert block is False
        assert unmatched == []

    def test_verbose_diff_dump_does_not_block(self) -> None:
        # commit -v appends the diff below the scissors line. A placeholder in the
        # REMOVED side of that dump must not be treated as an authored claim.
        msg = (
            "fix: rename foo\n"
            "# Please enter the commit message for your changes.\n"
            "# ------------------------ >8 ------------------------\n"
            "diff --git a/s.json b/s.json\n"
            "-  python ${CLAUDE_PROJECT_DIR}/x\n"
            "+  python /abs/x\n"
        )
        block, unmatched = gate.evaluate(msg, _DIFF_NO_PLACEHOLDER)
        assert block is False
        assert unmatched == []

    def test_authored_claim_with_comments_still_blocks(self) -> None:
        # The placeholder IS in the authored subject (hollow) AND comments follow.
        msg = "fix: introduce ${CLAUDE_PROJECT_DIR}\n# instructional comment\n"
        block, unmatched = gate.evaluate(msg, _DIFF_NO_PLACEHOLDER)
        assert block is True
        assert unmatched == ["${CLAUDE_PROJECT_DIR}"]

    def test_real_incident_dd63be8b_class(self) -> None:
        # The actual dd63be8b subject vs a hollow (no-substitution) diff.
        msg = "fix(hooks): resolve coordination hooks via ${CLAUDE_PROJECT_DIR} not hardcoded main path"
        block, unmatched = gate.evaluate(msg, _DIFF_NO_PLACEHOLDER)
        assert block is True
        assert unmatched == ["${CLAUDE_PROJECT_DIR}"]


# --- CLI layer: main() ------------------------------------------------------


def _run_cli(message: str, staged_diff: str, tmp_path: Path) -> int:
    msg_file = tmp_path / "COMMIT_EDITMSG"
    msg_file.write_text(message, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(MODULE_PATH), str(msg_file)],
        input=staged_diff,
        capture_output=True,
        text=True,
    )
    return proc.returncode


class TestCli:
    def test_cli_blocks_hollow(self, tmp_path: Path) -> None:
        rc = _run_cli("fix: ${CLAUDE_PROJECT_DIR}", _DIFF_NO_PLACEHOLDER, tmp_path)
        assert rc == 1

    def test_cli_allows_honest(self, tmp_path: Path) -> None:
        rc = _run_cli("fix: ${CLAUDE_PROJECT_DIR}", _DIFF_ADDS_PLACEHOLDER, tmp_path)
        assert rc == 0

    def test_cli_allows_no_token(self, tmp_path: Path) -> None:
        rc = _run_cli("chore: tidy", _DIFF_NO_PLACEHOLDER, tmp_path)
        assert rc == 0

    def test_cli_allows_override(self, tmp_path: Path) -> None:
        rc = _run_cli("fix: ${X} [hollow-ack]", _DIFF_NO_PLACEHOLDER, tmp_path)
        assert rc == 0

    def test_cli_missing_arg_fails_open(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(MODULE_PATH)],
            input="",
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0

    def test_cli_unreadable_message_file_fails_open(self, tmp_path: Path) -> None:
        missing = tmp_path / "does-not-exist"
        proc = subprocess.run(
            [sys.executable, str(MODULE_PATH), str(missing)],
            input="",
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0


# --- live hook layer: .githooks/commit-msg under a real repo ----------------


def _init_repo(repo: Path) -> None:
    subprocess.run(["git", "init", "-q", "--initial-branch=main"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True, capture_output=True)
    # Point hooksPath at the repo's own .githooks and copy the real hook + module in.
    hooks_dir = repo / ".githooks"
    hooks_dir.mkdir()
    (hooks_dir / "commit-msg").write_text(HOOK_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    os.chmod(hooks_dir / "commit-msg", 0o755)
    tools_dir = repo / "scripts" / "tools"
    tools_dir.mkdir(parents=True)
    (tools_dir / "commit_message_content_gate.py").write_text(MODULE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    subprocess.run(["git", "config", "core.hooksPath", ".githooks"], cwd=repo, check=True, capture_output=True)


def _commit(repo: Path, message: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "commit", "-m", message],
        cwd=repo,
        capture_output=True,
        text=True,
    )


@pytest.mark.skipif(not HOOK_PATH.exists(), reason="commit-msg hook not yet written")
class TestLiveHook:
    def test_live_hollow_commit_blocked(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        # Stage a file that does NOT contain the placeholder.
        (tmp_path / "f.txt").write_text("no placeholder here\n", encoding="utf-8")
        subprocess.run(["git", "add", "f.txt"], cwd=tmp_path, check=True, capture_output=True)
        r = _commit(tmp_path, "fix: switch to ${CLAUDE_PROJECT_DIR}")
        assert r.returncode != 0, f"expected BLOCK, got rc=0\n{r.stderr}"
        assert "HOLLOW-COMMIT BLOCKED" in (r.stderr + r.stdout)

    def test_live_honest_commit_allowed(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        (tmp_path / "f.txt").write_text("uses ${CLAUDE_PROJECT_DIR} for real\n", encoding="utf-8")
        subprocess.run(["git", "add", "f.txt"], cwd=tmp_path, check=True, capture_output=True)
        r = _commit(tmp_path, "fix: switch to ${CLAUDE_PROJECT_DIR}")
        assert r.returncode == 0, f"expected allow, got rc!=0\n{r.stderr}"

    def test_live_no_token_allowed(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        (tmp_path / "f.txt").write_text("hello\n", encoding="utf-8")
        subprocess.run(["git", "add", "f.txt"], cwd=tmp_path, check=True, capture_output=True)
        r = _commit(tmp_path, "chore: add greeting")
        assert r.returncode == 0, f"expected allow, got rc!=0\n{r.stderr}"

    def test_live_override_allowed(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        (tmp_path / "f.txt").write_text("no placeholder\n", encoding="utf-8")
        subprocess.run(["git", "add", "f.txt"], cwd=tmp_path, check=True, capture_output=True)
        r = _commit(tmp_path, "fix: ${CLAUDE_PROJECT_DIR} mentioned in prose [hollow-ack]")
        assert r.returncode == 0, f"expected allow with ack, got rc!=0\n{r.stderr}"

    def test_live_comment_only_placeholder_not_blocked(self, tmp_path: Path) -> None:
        # Regression for the PROBE-4 false-block: a #-comment carrying the
        # placeholder must NOT block when the authored subject does not claim it.
        # Use -F with a file whose body has a comment line (git keeps #-lines in
        # the file at commit-msg time; the gate must strip them).
        _init_repo(tmp_path)
        (tmp_path / "f.txt").write_text("hello\n", encoding="utf-8")
        subprocess.run(["git", "add", "f.txt"], cwd=tmp_path, check=True, capture_output=True)
        msg_file = tmp_path / "msg.txt"
        msg_file.write_text(
            "chore: tidy\n# note: relates to ${CLAUDE_PROJECT_DIR} work\n",
            encoding="utf-8",
        )
        r = subprocess.run(
            ["git", "commit", "--cleanup=strip", "-F", str(msg_file)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, f"comment-only placeholder false-blocked\n{r.stderr}"
