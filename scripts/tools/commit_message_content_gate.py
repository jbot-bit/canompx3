#!/usr/bin/env python3
"""Hollow-commit prevention gate — message-vs-staged-content literal-token check.

Closes the 2026-06-09 n=1 *hollow commit* gap: commit ``dd63be8b`` carried a
correct MESSAGE but a STALE staged SNAPSHOT — the real fixes sat in the working
tree while old content was staged, and nothing verified the staged blob matched
what the message claimed. The worktree mutex (``worktree_guard.py``) does NOT
catch this: it is a single-session commit-time integrity gap, unrelated to peers.

This module is the pure, testable core invoked by ``.githooks/commit-msg``. The
hook is git's purpose-built message-validation seam — it fires after the message
is composed but BEFORE the commit object is finalized, so a BLOCK here *prevents*
the hollow commit rather than detecting it after the fact ("prevention > cure").

The rule (deliberately conservative — minimize false-blocks):
  A commit message asserting a ``${...}`` shell/CC placeholder whose literal text
  appears ZERO times in the ADDED lines of the staged diff is a hollow commit
  -> BLOCK.

Why ONLY ``${...}`` placeholders (calibration, not assumption): the plan proposed
four token classes (placeholders, backtick-fenced tokens, repo file paths, quoted
CLI flags). Replaying the rule over the last 80–120 real commits (Phase 0
calibration) MEASURED the false-block rate:
  - 4-class rule: 8 false-blocks / 80 commits (10%) — far over target.
  - ``${...}``-placeholder-ONLY: 0 false-blocks / 120 commits, AND still catches
    the real ``dd63be8b`` incident (``${CLAUDE_PROJECT_DIR}`` in the subject, 0
    occurrences in staged additions -> BLOCK).
The other three classes fire on legitimate EXPLANATORY PROSE: messages routinely
name related/moved file paths not changed in THIS diff, and reference removed or
behavioral code symbols (`last_exc`, `raise SystemExit`, `all_outcomes=[]`) that
never appear in added lines. Only a ``${VAR}`` placeholder in a message reliably
means "I am introducing this placeholder" — the one class with a clean signal. A
10% false-block rate would train the operator to reflexively override, which is
functionally no gate at all. See
memory/feedback_hollow_commit_gate_placeholder_only_calibration_2026_06_09.md.

Match target = ADDED lines only (``^+`` excluding the ``^+++`` file header). A
hollow commit's stale snapshot can mention the token in a context (`` ``) or
removed (``-``) line; searching the whole diff would false-PASS. Proven against
``dd63be8b``: full-diff grep of a context-only token = 1 (false-PASS);
added-lines-only grep = 0 (correct).

Fail-OPEN everywhere: any parse error, empty input, or ambiguity -> allow. A
commit guard must err toward NOT blocking legit work (false-BLOCK = annoyance +
override; false-PASS = the status quo we already tolerate — capital-guard
fail-direction doctrine).

Override: a message containing the literal marker ``[hollow-ack]`` skips the gate.

Known limit (by design, not a defect): catches the ``${...}``-placeholder
hollow-commit sub-case only. A message asserting only quantitative prose (e.g.
"32 occurrences"), or a backtick/path/flag token, is NOT caught — those classes
have no clean low-false-block preventive signal (calibration above). The real
``dd63be8b`` incident IS caught via its ``${CLAUDE_PROJECT_DIR}`` placeholder.
"""

from __future__ import annotations

import re
import sys

# Override marker: a message containing this literal skips the gate entirely.
OVERRIDE_MARKER = "[hollow-ack]"

# --- Token extraction -------------------------------------------------------
#
# ONE high-confidence literal-token class survived Phase-0 calibration: ${...}
# shell/CC placeholders. (Backtick/path/flag classes false-blocked ~10% of real
# commits — see module docstring + the calibration memory file. They are
# deliberately NOT implemented; adding them back would re-introduce the measured
# false-block rate.) GROUP 1 is the token whose literal presence in the staged
# additions we then require.

# ${...} placeholders, e.g. ${CLAUDE_PROJECT_DIR}. The whole ${...} is the token
# (its literal text must appear in the diff additions). The name allows nested
# default-expansion forms (${VAR:-x}) by stopping at the first non-name char and
# requiring the closing brace, so ${VAR} and ${VAR:-default} both yield ${VAR...}.
_RE_PLACEHOLDER = re.compile(r"(\$\{[A-Za-z_][A-Za-z0-9_]*\})")

# git's `commit -v` scissors line — everything from here down is the appended
# diff, NOT part of the authored message. git strips it during cleanup, but only
# AFTER the commit-msg hook runs, so the gate must strip it itself.
_SCISSORS = "# ------------------------ >8 ------------------------"


def strip_git_comments(message: str) -> str:
    """Replicate git's default message cleanup BEFORE token extraction.

    At commit-msg time git has NOT yet applied `cleanup` — the raw
    ``COMMIT_EDITMSG`` still carries ``#``-prefixed comment/instruction lines and,
    under ``commit -v``, the entire staged diff appended below a scissors line.
    A ``${...}`` placeholder mentioned in a comment or in that appended diff is
    NOT a claim the author made about the change, so scanning it would false-block
    (proven by execution: a ``# comment with ${X}`` fired the gate). This strips
    both, matching git's own ``cleanup=strip`` + scissors semantics, so the gate
    only ever inspects the authored message body.
    """
    if not message:
        return ""
    out: list[str] = []
    for line in message.splitlines():
        if line.rstrip() == _SCISSORS:
            break  # everything below is the verbose-diff dump — stop
        if line.startswith("#"):
            continue  # git comment / instruction line — not authored content
        out.append(line)
    return "\n".join(out)


def extract_literal_tokens(message: str) -> list[str]:
    """Return the de-duplicated ``${...}`` placeholder tokens in ``message``.

    Operates on the comment-stripped message (see :func:`strip_git_comments`):
    ``#`` lines and the ``commit -v`` diff dump are NOT authored claims. Order-
    stable (first-seen order) for deterministic, testable output. Tokens are
    matched verbatim — the exact substring required to appear in the staged
    additions.
    """
    message = strip_git_comments(message)
    if not message:
        return []
    seen: dict[str, None] = {}
    for m in _RE_PLACEHOLDER.finditer(message):
        token = m.group(1).strip()
        if token and token not in seen:
            seen[token] = None
    return list(seen.keys())


# --- Diff handling ----------------------------------------------------------


def added_lines(staged_diff: str) -> str:
    """Return only the ADDED content of a unified diff.

    Keeps lines starting with a single ``+`` (the addition marker) but EXCLUDES
    ``+++`` file headers — those name files, not added content, and would
    false-MATCH a path token. The leading ``+`` is stripped so a token at the
    start of an added line still matches.
    """
    if not staged_diff:
        return ""
    out: list[str] = []
    for line in staged_diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            out.append(line[1:])
    return "\n".join(out)


def find_unmatched_tokens(message: str, staged_diff: str) -> list[str]:
    """Tokens asserted in ``message`` that appear ZERO times in staged additions.

    Returns the (order-stable) list of literal tokens the message claims but the
    added diff lines do not contain. An empty list means the commit is honest (or
    has no literal tokens to check). Override and empty-input handling live in
    :func:`evaluate`; this is the pure rule.
    """
    tokens = extract_literal_tokens(message)
    if not tokens:
        return []
    haystack = added_lines(staged_diff)
    return [t for t in tokens if t not in haystack]


# --- Top-level decision -----------------------------------------------------


def evaluate(message: str, staged_diff: str) -> tuple[bool, list[str]]:
    """Decide whether to BLOCK.

    Returns ``(block, unmatched_tokens)``. ``block`` is True iff the commit is a
    literal-token hollow commit. Fail-OPEN: missing message, the override marker,
    or no literal tokens all return ``(False, [])``.
    """
    if not message or not message.strip():
        return (False, [])
    if OVERRIDE_MARKER in message:
        return (False, [])
    unmatched = find_unmatched_tokens(message, staged_diff)
    return (bool(unmatched), unmatched)


def _block_message(unmatched: list[str]) -> str:
    quoted = ", ".join(repr(t) for t in unmatched)
    return (
        "\n"
        "============================ HOLLOW-COMMIT BLOCKED ============================\n"
        f"  The commit message asserts {len(unmatched)} literal token(s) that appear in\n"
        "  ZERO added lines of the staged diff:\n"
        f"      {quoted}\n"
        "\n"
        "  This is the dd63be8b class: the message claims a change the staged content\n"
        "  does not contain (stale snapshot — real fixes likely still in the working\n"
        "  tree). Stage the actual change, or amend the message to match what is staged.\n"
        "\n"
        "  Override (rare, legitimate — e.g. the token is genuinely prose-only):\n"
        f"      add the marker {OVERRIDE_MARKER} anywhere in the commit message.\n"
        "==============================================================================\n"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``commit_message_content_gate.py <message-file>``.

    Reads the commit message from the file git passes as ``$1`` and the staged
    diff from stdin (the hook pipes ``git diff --cached``). Exit 1 = BLOCK,
    exit 0 = allow. Any error fails OPEN (exit 0).
    """
    argv = sys.argv[1:] if argv is None else argv
    try:
        if not argv:
            return 0  # no message file -> fail-open
        try:
            with open(argv[0], encoding="utf-8", errors="replace") as fh:
                message = fh.read()
        except OSError:
            return 0  # unreadable message file -> fail-open
        staged_diff = sys.stdin.read() if not sys.stdin.isatty() else ""
        block, unmatched = evaluate(message, staged_diff)
        if block:
            sys.stderr.write(_block_message(unmatched))
            return 1
        return 0
    except BaseException:  # fail-OPEN on anything unexpected
        return 0


if __name__ == "__main__":
    sys.exit(main())
