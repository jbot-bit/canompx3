#!/usr/bin/env python3
"""Worktree-guard PreToolUse hook: BLOCK index-mutating Edit/Write/MultiEdit/Bash
when another live Claude session holds the lease for the SAME worktree.

## Hook contract (Anthropic Claude Code official)

  - PreToolUse event payload arrives on stdin as JSON.
  - exit 0 -> allow the tool call
  - exit 2 -> BLOCK; stderr is shown to the user verbatim
  - exit anything else -> error, treated as allow

Source: Anthropic Claude Code documentation, "Hooks" section, PreToolUse +
exit-code semantics. Mirrors the structural template of every sibling hook
(`stage-gate-guard.py`, `pre-edit-guard.py`, `judgment-review-soft-block.py`).

## Decision logic

  1. Resolve the worktree the tool ACTUALLY ran in from the event payload's
     `cwd` field (`_branch_state.invoking_cwd`) — NOT `Path.cwd()`. The hook
     command in settings.json is hardcoded to the main checkout, so the hook
     *process* cwd is always the main checkout regardless of which worktree the
     tool ran in. Reading `event["cwd"]` scopes the lease lookup to the correct
     worktree; otherwise a session correctly isolated in its OWN worktree gets
     false-blocked by the MAIN checkout's live peer (n=1 live-reproduced
     2026-06-03 — the F1 bug this fix removes). Mirrors the F4-A cwd-scoping fix
     already shipped for the branch-flip guards via the same canonical helper.
  2. Only ENFORCE the lease for operations that can corrupt this worktree's git
     index — Edit/Write/MultiEdit targeting paths INSIDE the repo, and mutating
     `git` Bash subcommands. Read-only Bash (status/log/diff/grep/ls/python …)
     and Edit/Write to paths OUTSIDE the repo (e.g. the user's `memory/` dir)
     can never corrupt the shared index, so they ALLOW even under a peer lease.
     This is the F2/F5 "blocks too broadly" fix: the prior hook blocked EVERY
     Bash command (froze harmless diagnostics and the operator's own recovery)
     and every outside-repo write.
  3. If `acquire()` returns "acquired" / "refreshed" / "reclaimed" -> ALLOW.
  4. If it returns "blocked" -> a DIFFERENT live session holds the lease for
     this exact worktree AND the op is index-mutating. Print a structured BLOCK
     message that LEADS with the heartbeat age + a live/stale verdict (the real
     decision signal), marks the recorded PID advisory, and gives the exact safe
     next action (F4). Exit 2.
  5. If "skipped" (not in a git repo) or "error" (FS transient) -> ALLOW.
     Fail-open per `.claude/rules/branch-flip-protection.md` § "Fail-safe
     guarantee" — a guard that can't reason about state must not block.

## Auto-clear of stale leases

Handled inside `scripts/tools/worktree_guard.py:acquire()`. Ownership is keyed
on the holder's (session_id, ppid) + heartbeat — NOT an OS file-lock (the old
lock was held by an ephemeral hook subprocess and provided no exclusion; see the
canonical module docstring, n=2 incident 2026-05-29/30). A peer's lease is
reclaimed only when the holder is provably gone: its ppid is dead AND its
heartbeat is stale (older than STALE_HEARTBEAT_SECONDS = 90s). A live ppid keeps
the lease no matter how stale the heartbeat — a single long-running tool call
(e.g. `check_drift.py` at ~180s) legitimately stops refreshing it, and stealing
the lease mid-call would let two sessions write. The operator's manual recovery
path remains `--force-release`.

## Why liveness is trustworthy (PID is advisory)

The canonical module's `_peer_is_live` treats a fresh heartbeat in the shared
`.claude-heartbeats/` directory as the AUTHORITATIVE liveness signal, with the
recorded ppid only an advisory cross-check (Windows `OpenProcess` can read a
live peer's ppid as momentarily dead). So a BLOCK means "a peer is beating in
THIS tree right now", not "a recorded PID looked alive". The BLOCK message
reflects that: heartbeat age first, PID flagged advisory.

## Canonical-source delegation (institutional-rigor §4)

ALL lease state I/O delegates to `scripts/tools/worktree_guard.py`; cwd
resolution delegates to `_branch_state.invoking_cwd` (the canonical helper the
branch-flip guards already use). This hook only classifies the op, shapes the
stderr message, and selects the exit code. Drift parity check
`check_worktree_guard_lease_path_parity` guards against future inlining.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOOKS_DIR = Path(__file__).resolve().parent
CANONICAL_MODULE = PROJECT_ROOT / "scripts" / "tools" / "worktree_guard.py"

# Git subcommands that mutate the index / refs / working tree and therefore need
# the worktree mutex. A read-only git subcommand (status/log/diff/show/branch/
# rev-parse/…) is SAFE under a peer lease. Conservative: anything not in this
# set is treated as read-only-safe (fail-OPEN — never block harmless
# inspection). The corruption class this guard exists for is concurrent
# index/ref writes, which all live in these verbs.
_GIT_MUTATING_SUBCOMMANDS = frozenset(
    {
        "add",
        "commit",
        "merge",
        "rebase",
        "reset",
        "checkout",
        "switch",
        "restore",
        "stash",
        "rm",
        "mv",
        "apply",
        "am",
        "cherry-pick",
        "revert",
        "clean",
        "pull",
        "fetch",
        "push",
        "tag",
        "branch",
        "worktree",
        "gc",
        "update-index",
        "update-ref",
        "filter-branch",
        "filter-repo",
        "notes",
        "submodule",
    }
)


def _load_canonical():
    """Import `scripts/tools/worktree_guard.py` by absolute path.

    Direct `from scripts.tools.worktree_guard import ...` would work but the
    sibling hooks (`judgment-review-soft-block.py`, `pre-edit-guard.py`) use
    the same importlib pattern so we stay consistent. Fail-open on any error.
    """
    spec = importlib.util.spec_from_file_location("worktree_guard", CANONICAL_MODULE)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except BaseException:
        return None
    return module


def _invoking_cwd(event: dict) -> str | None:
    """Resolve the worktree the tool ran in from `event["cwd"]`.

    Delegates to the canonical `_branch_state.invoking_cwd` so the cwd-scoping
    logic lives in exactly one place (institutional-rigor §4 — the branch-flip
    guards use the same helper). Fail-open to None on any import/parse error ->
    caller falls back to `Path.cwd()` (the historical, less-correct behaviour).
    """
    try:
        spec = importlib.util.spec_from_file_location("_branch_state", HOOKS_DIR / "_branch_state.py")
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.invoking_cwd(event)
    except BaseException:
        return None


def _matched_tool(event: dict) -> bool:
    """The settings.json matcher already filters by tool name; this is belt-
    and-braces against future matcher misconfig."""
    tool = event.get("tool_name") or event.get("tool", "")
    return tool in {"Edit", "Write", "MultiEdit", "Bash"}


def _bash_command(event: dict) -> str:
    ti = event.get("tool_input")
    if isinstance(ti, dict):
        cmd = ti.get("command")
        if isinstance(cmd, str):
            return cmd
    return ""


def _bash_is_index_mutating(command: str) -> bool:
    """True iff a Bash command could write THIS worktree's git index/refs.

    Heuristic, fail-OPEN-to-safe-allow: only clear index writers return True. A
    `git <verb>` whose first subcommand is in `_GIT_MUTATING_SUBCOMMANDS` is
    mutating. Everything else (read-only git, ls, cat, grep, python, pytest,
    ruff, …) is allowed even under a peer lease — it cannot corrupt the shared
    index, which is the only thing the worktree mutex protects. A missed
    mutating command degrades to a single slipped-through op for THAT command,
    never a false block (the pre-commit shared-state guard is the backstop for
    actual commits). The previous hook blocked ALL Bash, freezing read-only
    diagnostics (n=1 2026-06-03: a read-only `pwd`/`git rev-parse` was blocked).
    """
    if not command:
        return False
    # Split on common shell separators so `cd x && git commit …` is caught.
    for seg in re.split(r"&&|\|\||;|\||\n", command):
        toks = seg.strip().split()
        if not toks:
            continue
        i = 0
        # Skip leading env-var assignments (FOO=bar git …).
        while i < len(toks) and ("=" in toks[i] and not toks[i].startswith("-")):
            i += 1
        if i >= len(toks):
            continue
        base = toks[i].rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if base in ("git", "git.exe"):
            # Global git flags that consume the FOLLOWING token as their value
            # (e.g. `git -C <path> reset`). Skip both the flag and its argument
            # so the next token isn't misread as the subcommand.
            value_flags = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path", "--super-prefix"}
            rest = toks[i + 1:]
            j = 0
            while j < len(rest):
                t = rest[j]
                if t in value_flags:
                    j += 2  # skip flag + its value
                    continue
                if t.startswith("-"):
                    j += 1  # boolean/`--flag=value` global flag
                    continue
                return t in _GIT_MUTATING_SUBCOMMANDS  # first subcommand decides
            return False
    return False


def _edit_target(event: dict) -> str:
    ti = event.get("tool_input")
    if isinstance(ti, dict):
        for key in ("file_path", "path", "notebook_path"):
            v = ti.get(key)
            if isinstance(v, str) and v:
                return v
    return ""


def _path_inside_repo(target: str, worktree_root) -> bool:
    """True iff `target` resolves to a path inside `worktree_root`.

    ALLOW Edit/Write to paths OUTSIDE the repo (F5) — the user's `memory/` dir
    lives under `~/.claude/…`, not the repo, so such writes can never corrupt
    the git index. Conservative when containment is unprovable: a relative path
    with no known root is treated as inside (preserve the block); an absolute
    path with no known root is treated as outside.
    """
    if not target:
        return True  # unknown target -> conservative (treat as inside)
    try:
        tp = Path(target)
        if not tp.is_absolute() and worktree_root is not None:
            tp = worktree_root / tp
        tp = tp.resolve()
    except (OSError, ValueError, RuntimeError):
        return True
    if worktree_root is None:
        return not Path(target).is_absolute()
    try:
        wt = Path(worktree_root).resolve()
        tp.relative_to(wt)
        return True
    except (OSError, ValueError, RuntimeError):
        return False


def _op_is_index_mutating(event: dict, worktree_root) -> bool:
    """Classify whether this tool call can corrupt the shared git index.

    - Bash: only mutating git subcommands.
    - Edit/Write/MultiEdit: only when the target path is INSIDE the repo.

    Anything else is index-safe and must ALLOW even under a peer lease.
    """
    tool = event.get("tool_name") or event.get("tool", "")
    if tool == "Bash":
        return _bash_is_index_mutating(_bash_command(event))
    if tool in {"Edit", "Write", "MultiEdit"}:
        return _path_inside_repo(_edit_target(event), worktree_root)
    return False


def _emit_block(peer_lease: dict | None, lease_module, cwd: Path) -> None:
    """Structured stderr BLOCK message.

    Leads with the heartbeat-age + live/stale verdict (the actual decision
    signal), marks the recorded PID advisory, and ends with the exact safe next
    action — including that read-only commands and outside-repo writes are NOT
    blocked, so the operator can diagnose freely (F4).
    """
    peer = peer_lease or {}
    hb = peer.get("iso_heartbeat") or peer.get("iso_started") or ""
    age = lease_module._iso_age_seconds(hb)
    age_str = f"{age:.0f}s ago" if isinstance(age, (int, float)) else "unknown"
    stale_floor = getattr(lease_module, "STALE_HEARTBEAT_SECONDS", 90)
    verdict = (
        "LIVE (peer beating now)"
        if isinstance(age, (int, float)) and age < stale_floor
        else "heartbeat stale — liveness confirmed via peer beat / live ppid"
    )
    pid = peer.get("pid", "?")
    ppid = peer.get("ppid", "?")
    session_id = peer.get("session_id", "?")
    worktree = peer.get("worktree", "?")
    branch = peer.get("branch", "?")
    lp = lease_module.lease_path(cwd)

    lines = [
        "",
        "  ====================================================================",
        "  BLOCKED: a live peer Claude session holds THIS worktree's lease and",
        "  this is an index-mutating op (two writers corrupt the git index).",
        "  --------------------------------------------------------------------",
        f"  Liveness:     heartbeat {age_str}  ->  {verdict}",
        f"  Peer session: {session_id}",
        f"  Peer PID:     {pid} (ppid {ppid})  [advisory — heartbeat is authoritative]",
        f"  Worktree:     {worktree}",
        f"  Branch:       {branch}",
        f"  Lease sidecar:{lp}",
        "  --------------------------------------------------------------------",
        "  NOTE: read-only commands (status/log/diff/grep/ls/python) and writes",
        "  OUTSIDE the repo are NOT blocked — diagnose freely.",
        "",
        "  Resolutions (pick one):",
        "    1. Switch to the other Claude session and continue there.",
        "    2. Spawn a fresh worktree for parallel work. From Windows Explorer/",
        "       Terminal (NOT this blocked Bash tool), run the launcher that",
        "       opens a new Claude in its own worktree:",
        "         START_WORKTREE.bat <descriptor>",
        "       (scripts/tools/new_session.sh does the same, but it runs through",
        "        the Bash tool this guard is blocking - use it only once unblocked.)",
        "    3. If the peer session is CONFIRMED gone (LIVE above means it is NOT",
        "       gone — process exited, not just quiet), force-release and retry:",
        "         python scripts/tools/worktree_guard.py --force-release",
        "    4. Override for this session only (set in the harness env, then",
        "       restart the session):  WORKTREE_GUARD_BYPASS=1",
        "  ====================================================================",
        "",
    ]
    print("\n".join(lines), file=sys.stderr)


def main() -> int:
    # Read the hook event (fail-open on malformed JSON — never block what we
    # can't parse).
    try:
        event = json.load(sys.stdin) if not sys.stdin.isatty() else {}
    except (json.JSONDecodeError, ValueError):
        return 0

    if event and not _matched_tool(event):
        return 0  # matcher misconfig safety

    lease_module = _load_canonical()
    if lease_module is None:
        return 0  # canonical module unavailable -> fail-open

    # Bypass entirely when the env-var test seam is engaged (pytest / operator
    # override). Read from the harness environment; an in-command `export` does
    # NOT reach this subprocess.
    if os.environ.get("WORKTREE_GUARD_BYPASS") == "1":
        return 0

    # F1: resolve the worktree the tool ACTUALLY ran in from the event payload
    # `cwd`, falling back to the hook-process cwd. The hook command in
    # settings.json is hardcoded to the main checkout, so `Path.cwd()` is always
    # the main checkout — blind to an isolated worktree, which is what
    # false-blocked correctly-isolated sessions.
    invoked = _invoking_cwd(event)
    cwd = Path(invoked) if invoked else Path.cwd()

    # Resolve the worktree root for the index-containment test (Edit/Write).
    try:
        worktree_root = lease_module.resolve_worktree_root(cwd)
    except BaseException:
        worktree_root = None

    # F2/F5: only index-mutating ops need the mutex. Read-only Bash and
    # outside-repo writes ALLOW unconditionally — they cannot corrupt the
    # shared git index, the only thing this guard protects.
    if not _op_is_index_mutating(event, worktree_root):
        return 0

    # Identity = (session_id, ppid). session_id from the event payload is the
    # stable per-session key; ppid (this hook subprocess's parent == the Claude
    # session process) is the liveness anchor the canonical module probes. Both
    # are passed so a SECOND session in this worktree is recognised as a
    # different owner and blocked — the failure the old OS-lock model missed.
    session_id = event.get("session_id", "") or os.environ.get("CLAUDE_SESSION_ID", "")

    try:
        status_str, payload, _ = lease_module.acquire(cwd, session_id=session_id)
    except BaseException:
        # Any unexpected failure in the canonical module -> fail-open. This
        # mirrors the fail-open contract in branch-flip-protection.md.
        return 0

    if status_str == "blocked":
        _emit_block(payload, lease_module, cwd)
        return 2

    # acquired / refreshed / reclaimed / skipped / error -> allow
    return 0


if __name__ == "__main__":
    sys.exit(main())
