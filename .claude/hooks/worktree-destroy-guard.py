#!/usr/bin/env python3
"""Worktree-destroy-guard PreToolUse hook: BLOCK `git worktree remove` /
`git branch -D` when the TARGET tree/branch carries work-at-risk.

## Why this hook exists (the target-blindness gap)

The sibling mutex hook `worktree_guard.py` resolves the lease on the INVOKING
tree (`acquire(cwd, ...)`) — it answers *"is the EXECUTOR's own tree
lease-blocked?"*. But `git worktree remove <victim>` runs IN the executor's tree
while DESTROYING a DIFFERENT tree. That hook never inspects the victim's lease OR
its unpushed/dirty work, so a tree with unpushed commits / real edits
(NEEDS_FINISH) — or even one with a live peer beating in it (LIVE) — can be
`worktree remove`-d by another terminal whenever the executor's own tree isn't
locked → silent, unrecoverable data loss (the ~40-baton multi-terminal-collision
pain). `fleet_state` is the only surface that classifies the TARGET; this guard
consults it.

## Hook contract (Anthropic Claude Code official)

  - PreToolUse event payload arrives on stdin as JSON.
  - exit 0 -> allow ; exit 2 -> BLOCK (stderr shown verbatim) ; else -> allow.

## Decision logic

  1. Only fire on `git worktree remove [--force] <path>` and
     `git branch -D/-d/--delete <name>`. Any other Bash command -> exit 0.
  2. Load the canonical brain (`scripts/tools/fleet_state.py`). If it is
     genuinely unavailable (import failure) -> exit 0 (the ONLY uncertainty path
     that allows — see polarity note).
  3. Resolve the TARGET worktree's `WorktreeState` via `fleet_state`.
     `classification in {NEEDS_FINISH, LIVE}` -> BLOCK.
  4. ALWAYS run a decision-time live unpushed re-check
     (`git -C <target> rev-list <branch> --not --remotes`) and BLOCK on any
     unpushed commit — NEVER trust the fleet_state `unpushed` int alone, which
     can be a swallowed-`except` 0 (`fleet_state.py:_ahead_behind_unpushed`) or
     computed against a stale `origin/main`.
  5. Otherwise (HOLLOW / MERGED / HEALTHY / STALE, no unpushed) -> exit 0.

## Fail-open polarity is INVERTED for a destruction guard (load-bearing)

A *mutex* guard fails OPEN so it never freezes legitimate work. A *destruction*
guard failing open would PERMIT an irreversible `worktree remove`. So:

  - parse error / unknown command shape -> CONSULT the brain anyway
    (conservative), never silent-allow.
  - target unresolvable by fleet_state (None) -> do NOT silently allow; fall
    through to the direct unpushed probe and BLOCK if it finds unpushed work.
  - ONLY a genuine `fleet_state` import failure exits 0 (and logs the gap to
    stderr as a one-liner so the lost coverage is visible).

## Canonical-source delegation (institutional-rigor §4)

Liveness / hollow / classification all delegate to `fleet_state` (the brain),
which itself delegates to `worktree_guard._peer_is_live`. This hook only parses
the command target, runs the decision-time unpushed re-check, shapes the BLOCK
message, and selects the exit code. The unpushed re-check is NOT a re-encoded
liveness oracle — it is a hardened cross-check of a single int the brain itself
documents as lossy.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FLEET_STATE_MODULE = PROJECT_ROOT / "scripts" / "tools" / "fleet_state.py"

# git global flags that consume the FOLLOWING token as their value (so the next
# token isn't misread as the subcommand). Mirrors worktree_guard.py:214.
_VALUE_FLAGS = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path", "--super-prefix"}


def _bash_command(event: dict) -> str:
    ti = event.get("tool_input")
    if isinstance(ti, dict):
        cmd = ti.get("command")
        if isinstance(cmd, str):
            return cmd
    return ""


def _git_argv(seg: str) -> list[str] | None:
    """Return the git arg vector AFTER global flags for a single command segment,
    or None if the segment is not a `git ...` invocation.

    Skips leading env assignments (FOO=bar git ...) and value-consuming global
    flags (git -C <path> ...). Mirrors the worktree_guard.py tokenizer.
    """
    try:
        toks = [_strip_wrapping_quotes(tok) for tok in shlex.split(seg.strip(), posix=False)]
    except ValueError:
        toks = seg.strip().split()
    if not toks:
        return None
    i = 0
    while i < len(toks) and ("=" in toks[i] and not toks[i].startswith("-")):
        i += 1
    if i >= len(toks):
        return None
    # Case-insensitive: Windows resolves `GIT`/`Git`/`git.EXE` to the same
    # binary, so a case-exact match silently bypasses the guard (audit finding,
    # 2026-06-06). Lowercasing closes that bypass on the case-insensitive FS.
    base = toks[i].rsplit("/", 1)[-1].rsplit("\\", 1)[-1].lower()
    if base not in ("git", "git.exe"):
        return None
    rest = toks[i + 1:]
    j = 0
    while j < len(rest):
        t = rest[j]
        if t in _VALUE_FLAGS:
            j += 2
            continue
        if t.startswith("-"):
            j += 1
            continue
        break
    return rest[j:]


def _strip_wrapping_quotes(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {'"', "'"}:
        return token[1:-1]
    return token


def _parse_destroy_target(command: str) -> tuple[str | None, str | None]:
    """Extract (op, target) from a destruction command, else (None, None).

    op is "worktree" for `git worktree remove <path>` or "branch" for
    `git branch -D/-d/--delete <name>`. target is the path or branch name.
    Only these two destruction shapes are matched; everything else -> (None, None).
    """
    if not command:
        return None, None
    for seg in re.split(r"&&|\|\||;|\||\n", command):
        argv = _git_argv(seg)
        if not argv:
            continue
        sub = argv[0]
        args = argv[1:]
        if sub == "worktree":
            # `git worktree remove [--force] <path>`
            if args and args[0] == "remove":
                for a in args[1:]:
                    if a.startswith("-"):
                        continue  # --force / -f
                    return "worktree", a
                return "worktree", None  # remove with no positional (let git error)
        elif sub == "branch":
            # `git branch -D|-d|--delete [--force] <name>` — a DELETE form only.
            # NOTE: `git branch -D -r origin/feat` deletes a REMOTE-TRACKING ref,
            # not local work — fleet_state has no bare-name match, so it resolves
            # to None and the unpushed probe runs `rev-list origin/feat --not
            # --remotes` which is 0 by definition (the ref IS a remote). That
            # correctly ALLOWS: deleting a remote-tracking ref risks no local
            # work. We do not special-case `-r`; the data-flow already handles it.
            is_delete = any(a in ("-D", "-d", "--delete") for a in args)
            if is_delete:
                for a in args:
                    if a.startswith("-"):
                        continue
                    return "branch", a
                return "branch", None
    return None, None


# Sentinel: distinguishes "the brain could not be loaded at all" (genuine
# unavailability -> the ONLY allow-on-uncertainty path) from "the brain loaded
# but the target is not a known worktree" (-> fall through to the unpushed probe,
# never silent-allow). A bare None means the latter.
BRAIN_UNAVAILABLE = object()


def _load_fleet_state():
    """Import scripts/tools/fleet_state.py by absolute path. None on any failure
    (the ONLY genuine-unavailability path that allows).

    fleet_state.py uses absolute package imports (``from scripts.tools import
    worktree_guard``), so PROJECT_ROOT must be on sys.path before exec — the hook
    is invoked with an arbitrary cwd (the executor's tree), not the project root.
    """
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    spec = importlib.util.spec_from_file_location("fleet_state", FLEET_STATE_MODULE)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    # MUST register in sys.modules BEFORE exec_module: fleet_state defines a
    # @dataclass, and CPython's dataclass machinery does
    # `sys.modules.get(cls.__module__).__dict__` during class creation — which
    # raises AttributeError on None if the module isn't registered yet. (The
    # sibling worktree_guard hook gets away without this only because its
    # canonical module has no dataclass.) Caught live by the end-to-end repro.
    sys.modules["fleet_state"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop("fleet_state", None)
        return None
    return module


def _resolve_target_path(op: str, target: str | None) -> Path | None:
    """Resolve a destruction target to a worktree path for the unpushed probe.

    - worktree op: the target IS a path.
    - branch op: a branch is only directly probe-able if it currently has a
      worktree; otherwise we cannot `git -C` it. Return None -> the unpushed
      probe is skipped for that case (state-based verdict still applies).
    """
    if op == "worktree" and target:
        try:
            return Path(target)
        except (OSError, ValueError):
            return None
    return None


def _resolve_target_state(op: str, target: str | None):
    """Find the TARGET's WorktreeState via the brain. None if not classifiable.

    Matches a worktree op by resolved path; a branch op by branch name.
    """
    fs = _load_fleet_state()
    if fs is None:
        return BRAIN_UNAVAILABLE
    try:
        states = fs.fleet_state()
    except BaseException:
        # Loaded but the query itself failed — treat as unavailable (we cannot
        # classify, so allow only via the genuine-unavailability path).
        return BRAIN_UNAVAILABLE
    if not target:
        return None
    if op == "worktree":
        try:
            want = Path(target).resolve()
        except (OSError, ValueError):
            want = None
        for s in states:
            try:
                if want is not None and Path(s.path).resolve() == want:
                    return s
            except (OSError, ValueError):
                continue
        return None
    if op == "branch":
        for s in states:
            if s.branch == target:
                return s
    return None


def _live_unpushed_count(op: str, target: str | None, state) -> int:
    """Decision-time live unpushed re-check — NEVER trust state.unpushed alone.

    Resolves the target's own checkout and runs
    `git -C <target> rev-list <branch> --not --remotes --count`. Returns the
    count, or 0 when the probe cannot run (no resolvable checkout / git failure).
    Conservative: a probe that errors returns 0 (the state-based verdict is the
    backstop), but a probe that SUCCEEDS with >0 forces a BLOCK even past a clean
    classification.
    """
    target_path = _resolve_target_path(op, target)
    branch = None
    if state is not None:
        branch = getattr(state, "branch", None)
    if op == "branch" and target:
        branch = target

    # Choose a cwd to run git in. For a worktree target, the tree itself; if it
    # is already gone, fall back to the project root (the branch ref still lives
    # in the shared object store).
    cwd: Path
    if target_path is not None and target_path.exists():
        cwd = target_path
    else:
        cwd = PROJECT_ROOT

    rev = branch if branch else "HEAD"
    try:
        proc = subprocess.run(
            ["git", "-C", str(cwd), "rev-list", rev, "--not", "--remotes", "--count"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if proc.returncode != 0:
            return 0
        return int(proc.stdout.strip() or "0")
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0


def _emit_block(op: str, target: str | None, state, unpushed: int) -> None:
    cls = getattr(state, "classification", None) if state is not None else None
    reasons = list(getattr(state, "reasons", []) or []) if state is not None else []
    if unpushed > 0 and not any("unpush" in r.lower() for r in reasons):
        reasons.append(f"{unpushed} unpushed commit(s) (decision-time re-check)")
    what = "worktree remove" if op == "worktree" else "branch -D"
    lines = [
        "",
        "  ====================================================================",
        f"  BLOCKED: `git {what}` would destroy work-at-risk.",
        "  --------------------------------------------------------------------",
        f"  Target:         {target or '(unparsed)'}",
        f"  Classification: {cls or 'UNCLASSIFIED — fell through to unpushed probe'}",
        f"  Live unpushed:  {unpushed} commit(s) (git rev-list --not --remotes)",
    ]
    for r in reasons:
        lines.append(f"    - {r}")
    lines += [
        "  --------------------------------------------------------------------",
        "  This is the fleet-state brain refusing to reap a tree/branch that",
        "  still holds unpushed commits, real uncommitted work, or a live peer.",
        "",
        "  Resolutions (pick one):",
        "    1. Finish/push the work first (commit + push), then re-run.",
        "    2. If the work is genuinely disposable, verify with:",
        "         python scripts/tools/fleet_state.py",
        "       and only then force-remove from a context this guard allows.",
        "    3. If a live peer holds it, switch to that session instead.",
        "  ====================================================================",
        "",
    ]
    print("\n".join(lines), file=sys.stderr)


def main() -> None:
    """Hook entry point. Calls ``sys.exit`` with the verdict (0 allow / 2 block)
    so the contract is uniform whether invoked as a subprocess or called
    directly in-process (the test harness asserts ``pytest.raises(SystemExit)``,
    mirroring mcp-git-guard.py)."""
    try:
        event = json.load(sys.stdin) if not sys.stdin.isatty() else {}
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)  # malformed event -> allow (cannot identify a destruction)

    if event.get("tool_name") not in (None, "Bash"):
        sys.exit(0)
    command = _bash_command(event)
    op, target = _parse_destroy_target(command)
    if op is None:
        sys.exit(0)  # not a destruction command

    # FAST PROBE FIRST (audit finding 2026-06-06): the single `git rev-list`
    # unpushed re-check is ~10-50ms and needs no fleet_state. Run it BEFORE the
    # slower per-worktree `fleet_state()` classification so the most common
    # at-risk signal (unpushed commits) is caught even if the full brain query
    # would run long. A hook that exceeds its wall-clock timeout is killed and
    # treated as exit-0 (allow) by Claude Code — for a destruction guard that is
    # a silent-allow bypass, so the cheapest decisive check must complete first.
    # It does NOT depend on `state` (a worktree target probes its own HEAD; a
    # branch target probes the named ref), so it is safe to run pre-classify.
    unpushed = _live_unpushed_count(op, target, None)
    if unpushed > 0:
        _emit_block(op, target, None, unpushed)
        sys.exit(2)

    state = _resolve_target_state(op, target)

    # Genuine brain unavailability is the ONLY uncertainty path that allows.
    if state is BRAIN_UNAVAILABLE:
        print(
            "  [worktree-destroy-guard] fleet_state unavailable — destruction "
            "NOT guarded this call (coverage gap; unpushed re-check already "
            "ran clean).",
            file=sys.stderr,
        )
        sys.exit(0)

    cls = getattr(state, "classification", None) if state is not None else None
    if cls in ("NEEDS_FINISH", "LIVE"):
        _emit_block(op, target, state, unpushed)
        sys.exit(2)

    # state None + no unpushed -> nothing provably at risk -> allow.
    sys.exit(0)


if __name__ == "__main__":
    main()
