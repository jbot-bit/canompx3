#!/usr/bin/env python3
"""fleet_state.py — THE canonical fleet-state resolver (the brain).

ONE read-only function, `fleet_state(root)`, returns the SINGLE correct answer
for every git worktree of this repo: who is live in it, whether it carries real
uncommitted work, whether it has shipped, whether it is hollow, and a single
classification. Every guard / hook / tool is meant to read THIS instead of
re-deriving liveness from its own (conflicting) source.

WHY THIS EXISTS — the split-brain class (root cause #1, 7 incidents):
Three+ surfaces each answer "is a peer live in tree X" from a DIFFERENT signal:
  - worktree_guard.py        -> ppid-alive + fresh heartbeat
  - session-start.py         -> tracked-file mtime / .beat files / ppid
With 4 terminals these oracles DISAGREE, so guards block, force-remove, or miss
each other. The corpus's own lesson (`feedback_precommit_reflock_race`) was:
*"don't add a third oracle — feed the ONE reliable signal (heartbeat) into the
gate."* That was applied to the commit-gate and reaper but never centralized.
This module centralizes it: liveness is resolved ONLY via the canonical
`worktree_guard._peer_is_live` / `_fresh_peer_heartbeat` path — heartbeat-
authoritative, with ppid as advisory. No liveness logic is re-encoded here; we
IMPORT the canonical implementation (institutional-rigor §4 — delegate, never
re-encode).

READ-ONLY. No network, no index writes, no destructive ops. Safe to run from
inside a leased worktree. Output: a list of `WorktreeState` objects, or `--json`.

This is STAGE 1 of the fleet-state-brain plan: the resolver + its tests only.
Pointing the guards at it (Stage 2), awareness hooks (Stage 3), and the plan
anchor (Stage 4) are SEPARATE, later stages — this file changes no existing
behavior on its own (nothing imports it yet).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Canonical liveness + enumeration sources. We import the REAL implementations so
# there is exactly one definition of "live" / "the worktree list" in the repo.
# A defensive import guard keeps fleet_state runnable (degraded) even if a
# sibling module is mid-refactor — it never silently lies, it records the gap.
_IMPORT_ERRORS: list[str] = []

try:
    from scripts.tools import worktree_guard as _wg  # type: ignore
except Exception as exc:  # pragma: no cover - exercised only on broken checkout
    _wg = None  # type: ignore
    _IMPORT_ERRORS.append(f"worktree_guard import failed: {exc!r}")

try:
    from scripts.tools import worktree_manager as _wm  # type: ignore
except Exception as exc:  # pragma: no cover
    _wm = None  # type: ignore
    _IMPORT_ERRORS.append(f"worktree_manager import failed: {exc!r}")

try:
    from scripts.tools import stale_work_radar as _radar  # type: ignore
except Exception as exc:  # pragma: no cover
    _radar = None  # type: ignore
    _IMPORT_ERRORS.append(f"stale_work_radar import failed: {exc!r}")

from scripts.tools._worktree_churn import is_churn_path
from scripts.tools._worktree_hollow import _classify_porcelain_line, classify_hollow

# ── Classification vocabulary ────────────────────────────────────────────────
# A worktree is exactly ONE of these. Order of precedence is encoded in
# `_classify` — LIVE wins (never reap a live peer), then HOLLOW (reapable),
# then MERGED, then NEEDS_FINISH (unpushed real work), then HEALTHY/STALE.
CLASS_LIVE = "LIVE"  # a peer session is beating in this tree right now
CLASS_HOLLOW = "HOLLOW"  # gutted (deletion-dominated) — reapable
CLASS_MERGED = "MERGED"  # branch merged into base, no unpushed/real-dirty work
CLASS_NEEDS_FINISH = "NEEDS_FINISH"  # unpushed commits or real uncommitted work
CLASS_HEALTHY = "HEALTHY"  # checked out, clean, nothing at risk
CLASS_STALE = "STALE"  # not live, not merged, old/no-signal — candidate to review


@dataclass
class WorktreeState:
    """The single canonical state record for one worktree."""

    path: str
    branch: str | None
    head: str | None
    is_current: bool  # this is the worktree fleet_state was invoked from
    live: bool  # a DIFFERENT live session is in this tree (heartbeat-authoritative)
    dirty_paths: int  # uncommitted changed paths (raw porcelain count)
    real_dirty_paths: int  # uncommitted paths EXCLUDING operational churn
    hollow: bool  # gutted/deletion-dominated working tree
    ahead: int  # commits ahead of base
    behind: int  # commits behind base
    unpushed: int  # commits not on the remote tip
    merged_into_base: bool
    classification: str
    reasons: list[str] = field(default_factory=list)


def _run_git(args: list[str], cwd: Path, timeout: float = 10.0) -> tuple[int, str]:
    """Read-only git invocation. Returns (rc, stdout). Fail-soft to (1, '')."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout
    except (OSError, subprocess.SubprocessError):
        return 1, ""


def _porcelain(path: Path) -> str:
    """`git status --porcelain` for a worktree (empty string on any failure)."""
    rc, out = _run_git(["status", "--porcelain"], cwd=path)
    return out if rc == 0 else ""


def _count_dirty(porcelain: str) -> tuple[int, int, int]:
    """(total_changed_paths, real_changed_paths, real_TRACKED_NON_DELETION_paths).

    All three exclude operational churn via the canonical predicate
    (`_worktree_churn.is_churn_path` — single source, never an inline copy):
      - total      : every changed path (incl. churn) — raw porcelain count.
      - real       : non-churn changed paths (deletions INCLUDED — a deletion is
                     still a real change line).
      - real_nondel: the WORK-AT-RISK signal for a HOLLOW tree — non-churn,
                     NON-DELETION, TRACKED changes (staged/worktree add/mod/rename
                     of a tracked path). It deliberately EXCLUDES two things that
                     are NOT work-at-risk:
                       * deletions (a hollow tree's mass `D` lines are the gutting,
                         not work — gating on them would defeat the HOLLOW class:
                         a 300-deletion tree has real=300 but real_nondel=0);
                       * UNTRACKED (`??`) paths — generated runtime scaffolding
                         (`.claude/`, `.codex/`, `.canompx3-runtime/`) that every
                         worktree accretes and that the poisoning tree `...2026042`
                         still carries after being gutted. Counting `??` as
                         work-at-risk would flip that genuinely-reapable tree to
                         NEEDS_FINISH and BLOCK the cleanup this plan exists to do
                         (verified live, 2026-06-06: its 8 non-del lines are all
                         `?? .claude/`-class runtime dirs). A genuinely new source
                         file the operator wants kept would be `git add`-ed (→
                         tracked `A`), and Stage 2's reaper still requires operator
                         confirmation as the backstop.
    The deletion vs non-deletion split DELEGATES to
    `_worktree_hollow._classify_porcelain_line` (single definition, not
    re-encoded). The tracked-vs-untracked split reads the porcelain `XY` status
    field directly — `??` is git's canonical untracked marker.
    Rename lines (`R  old -> new`) are counted as one path on the new name.
    """
    total = 0
    real = 0
    real_nondel = 0
    for raw in porcelain.splitlines():
        line = raw.rstrip("\n")
        if not line.strip() or len(line) < 4:
            continue
        total += 1
        status = line[:2]  # porcelain XY status field
        path_part = line[3:].strip()
        # Rename arrow — judge the destination path.
        if " -> " in path_part:
            path_part = path_part.split(" -> ", 1)[1].strip()
        if is_churn_path(path_part):
            continue
        real += 1
        # Work-at-risk = non-deletion (canonical hollow classifier) AND tracked
        # (status != '??'). Untracked scaffolding is runtime junk, not lost work.
        if _classify_porcelain_line(line) != "del" and status != "??":
            real_nondel += 1
    return total, real, real_nondel


def _ahead_behind_unpushed(branch: str | None, base: str) -> tuple[int, int, int]:
    """Delegate to stale_work_radar's canonical divergence helpers.

    Returns (ahead, behind, unpushed). (0,0,0) when the radar is unavailable or
    the branch is detached — we never guess divergence locally.
    """
    if _radar is None or not branch:
        return 0, 0, 0
    try:
        ahead, behind = _radar.ahead_behind(branch, base)
        unpushed, has_remote = _radar.unpushed_count(branch)
        # Local-only branch: every ahead commit is unpushed (radar's own rule).
        if not has_remote:
            unpushed = ahead
        return ahead, behind, unpushed
    except Exception:
        return 0, 0, 0


def _is_merged(branch: str | None, base: str) -> bool:
    if _radar is None or not branch:
        return False
    try:
        return _radar.is_merged(branch, base)
    except Exception:
        return False


def _peer_live_in(path: Path, exclude_session_id: str) -> bool:
    """Heartbeat-authoritative liveness for one worktree, via the CANONICAL guard.

    We read the worktree's lease and ask `worktree_guard._peer_is_live` — the
    same function the PreToolUse mutex uses — so fleet_state and the gate can
    never disagree about who is live. A fresh peer `.beat` in this tree is an
    independent liveness FACT (the Windows ppid-flaps-dead fix). Fail-soft to
    False (no evidence -> not live); over-reporting live would block reaping a
    genuinely-dead tree.
    """
    if _wg is None:
        return False
    try:
        lease = _wg.read_lease(path)
        # No lease sidecar: fall back to a pure heartbeat check for THIS tree.
        # _fresh_peer_heartbeat reads .claude-heartbeats/ and matches the beat's
        # cwd to this worktree root — a sibling's beat does not count.
        if not lease:
            return bool(
                _wg._fresh_peer_heartbeat(
                    path,
                    exclude_session_id=exclude_session_id,
                    window_seconds=_wg.BLOCKING_HEARTBEAT_WINDOW_SECONDS,
                )
            )
        return bool(_wg._peer_is_live(lease, cwd=path, exclude_session_id=exclude_session_id))
    except Exception:
        return False


def _classify(
    *,
    live: bool,
    hollow: bool,
    merged: bool,
    real_dirty: int,
    real_nondel_dirty: int,
    unpushed: int,
    behind: int,
) -> tuple[str, list[str]]:
    """The single precedence ladder. Returns (class, reasons).

    `real_dirty` counts non-churn changed paths INCLUDING deletions; in a hollow
    tree it is dominated by deletion noise and is NOT a work-at-risk signal.
    `real_nondel_dirty` counts non-churn NON-DELETION paths — the actual
    work-at-risk signal for a hollow tree (a genuine add/mod among the deletions).
    """
    reasons: list[str] = []

    # LIVE always wins — a live peer is never reapable, never "stale".
    if live:
        reasons.append("live peer heartbeat in this tree")
        return CLASS_LIVE, reasons

    # SAFETY INVARIANT (Stage 2 inherits this) — work-at-risk OUTRANKS hollow.
    # A gutted working tree (deletion-dominated) is normally reapable, BUT if it
    # ALSO carries (a) commits not yet on the remote, or (b) a genuine
    # non-deletion edit hiding among the deletions, reaping it destroys
    # unrecoverable work. Two distinct risks:
    #   - unpushed commits live in the local ref (the deletion-noise that makes
    #     the tree "look hollow" says NOTHING about them).
    #   - real_nondel_dirty is a non-churn ADD/MOD/RENAME — true uncommitted work.
    #     We must NOT gate on `real_dirty` here: it INCLUDES the mass deletions
    #     (a `D` line is a real change), so every hollow tree has real_dirty≈del
    #     and gating on it would defeat the HOLLOW class entirely (verified:
    #     a 300-deletion tree has real_dirty=300). Only NON-deletion real work
    #     counts as at-risk inside a hollow tree.
    # Classify NEEDS_FINISH so Stage 2's reaper never deletes a tree with
    # shippable commits OR live edits. Without this, a HOLLOW verdict would make
    # the tree reap-eligible and silently lose that work.
    if hollow and (unpushed > 0 or real_nondel_dirty > 0):
        if unpushed > 0:
            reasons.append(f"gutted working tree BUT {unpushed} unpushed commit(s) — must finish/push, NOT reap")
        if real_nondel_dirty > 0:
            reasons.append(f"gutted working tree BUT {real_nondel_dirty} real non-deletion edit(s) — NOT reap")
        return CLASS_NEEDS_FINISH, reasons

    # HOLLOW next — a gutted tree (mass deletions, no unpushed, no real
    # non-deletion edits) is reapable. The work-at-risk cases were handled above.
    if hollow:
        reasons.append("deletion-dominated working tree (gutted) — reapable")
        return CLASS_HOLLOW, reasons

    # Real uncommitted work or unpushed commits = work at risk -> NEEDS_FINISH.
    if unpushed > 0 or real_dirty > 0:
        if unpushed > 0:
            reasons.append(f"{unpushed} unpushed commit(s)")
        if real_dirty > 0:
            reasons.append(f"{real_dirty} real uncommitted path(s)")
        return CLASS_NEEDS_FINISH, reasons

    # Nothing at risk. Merged -> safe to prune; else clean/healthy.
    if merged:
        reasons.append("merged into base — safe to prune")
        return CLASS_MERGED, reasons

    # Clean, not merged, not live: healthy if reasonably current, else stale.
    if behind > _STALE_BEHIND_THRESHOLD:
        reasons.append(f"{behind} commits behind base, clean, not live")
        return CLASS_STALE, reasons
    reasons.append("clean checked-out worktree")
    return CLASS_HEALTHY, reasons


# A clean, not-live, not-merged tree this far behind base is treated as STALE
# (a candidate to review/refresh) rather than HEALTHY. Mirrors the radar's
# rebase-debt threshold so the two surfaces agree.
_STALE_BEHIND_THRESHOLD = 100


def fleet_state(
    root: Path = PROJECT_ROOT,
    *,
    base: str = "origin/main",
    exclude_session_id: str = "",
) -> list[WorktreeState]:
    """Resolve the canonical state of every worktree of this repo.

    READ-ONLY. `exclude_session_id` is the caller's own session id (so the
    caller's own beat never counts as a "peer"); empty is safe (no self-exclude).
    """
    if _wm is None:
        # Cannot even enumerate worktrees — return empty rather than guess.
        return []

    try:
        infos = _wm.list_worktrees(root)
    except Exception:
        return []

    current_root = root.resolve()
    states: list[WorktreeState] = []
    for info in infos:
        wt_path = Path(info.path)
        try:
            wt_resolved = wt_path.resolve()
        except OSError:
            wt_resolved = wt_path
        is_current = wt_resolved == current_root

        porcelain = _porcelain(wt_path)
        total_dirty, real_dirty, real_nondel_dirty = _count_dirty(porcelain)
        hollow = classify_hollow(porcelain).is_hollow

        branch = info.branch.removeprefix("refs/heads/") if info.branch else None
        ahead, behind, unpushed = _ahead_behind_unpushed(branch, base)
        merged = _is_merged(branch, base)

        # A live check on the CURRENT tree would just detect our own session; the
        # canonical helpers already self-exclude via session id, but skip anyway
        # for clarity — the current tree is never a "peer".
        live = False if is_current else _peer_live_in(wt_path, exclude_session_id)

        classification, reasons = _classify(
            live=live,
            hollow=hollow,
            merged=merged,
            real_dirty=real_dirty,
            real_nondel_dirty=real_nondel_dirty,
            unpushed=unpushed,
            behind=behind,
        )

        states.append(
            WorktreeState(
                path=str(wt_path),
                branch=branch,
                head=info.head,
                is_current=is_current,
                live=live,
                dirty_paths=total_dirty,
                real_dirty_paths=real_dirty,
                hollow=hollow,
                ahead=ahead,
                behind=behind,
                unpushed=unpushed,
                merged_into_base=merged,
                classification=classification,
                reasons=reasons,
            )
        )
    return states


def render_table(states: list[WorktreeState]) -> str:
    lines = [f"Fleet State — {len(states)} worktree(s)", ""]
    if _IMPORT_ERRORS:
        lines.append("WARN (degraded — some canonical sources unavailable):")
        lines.extend(f"  - {e}" for e in _IMPORT_ERRORS)
        lines.append("")
    header = f"{'CLASS':<13} {'A/B':>9} {'UNPUSH':>6} {'DIRTY':>6} {'BRANCH':<40} PATH"
    lines.append(header)
    lines.append("-" * len(header))
    for s in states:
        ab = f"{s.ahead}/{s.behind}"
        dirty = f"{s.real_dirty_paths}/{s.dirty_paths}"
        marker = "*" if s.is_current else " "
        br = (s.branch or "(detached)")[:40]
        lines.append(f"{s.classification:<13} {ab:>9} {s.unpushed:>6} {dirty:>6} {br:<40} {marker}{s.path}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Canonical read-only fleet-state resolver (single liveness oracle).")
    parser.add_argument("--base", default="origin/main", help="Base ref for divergence (default origin/main)")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--session-id", default="", help="Caller's own session id (self-exclude from peer-live checks)")
    args = parser.parse_args(argv)

    states = fleet_state(base=args.base, exclude_session_id=args.session_id)

    if args.json:
        payload = {
            "worktrees": [asdict(s) for s in states],
            "import_errors": _IMPORT_ERRORS,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(render_table(states))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
