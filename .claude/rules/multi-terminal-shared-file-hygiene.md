# Multi-Terminal Shared-File Hygiene

**Load-policy:** auto-injected when touching files under `docs/runtime/` or `docs/audit/` (excluding `docs/runtime/stages/` and `docs/audit/hypotheses/drafts/`, which are author-owned).

**Authority:** triggered by `feedback_multi_terminal_shared_file_thrash_2026_05_21.md` n=1 incident. Two concurrent Claude terminals running scanner + drift against the same `docs/runtime/fast_lane_trial_ledger.yaml` produced interleaved appends; one terminal committed entries framed as "scanner smoke runs" that were actually peer-terminal drift-loop side effects. Working tree then accumulated +210 lines of additional peer-drift noise on top of the commit.

---

## The rule

**Before staging or committing any file under `docs/runtime/` or `docs/audit/`, run the three-check coordination protocol.**

Specifically:
- High-traffic generated-state files (`fast_lane_trial_ledger.yaml`, `promote_queue.yaml`, `fast_lane_graveyard_digest.yaml`, `lane_allocation.json`, `chordia_audit_log.yaml`, `validated_setups` exports) are written by many tools (scanners, drift checks, validators) from multiple terminals.
- Drift-check runs can themselves trigger append-only writes through the scanner code path — meaning a `python pipeline/check_drift.py` in one terminal mutates the same file another terminal is committing.
- Without coordination, terminals interleave appends and one terminal commits entries authored by another, losing provenance.

---

## The three-check coordination protocol

Before `git add <shared-state-path>` or `git commit` with shared-state paths staged:

### Check 1 — sibling-commit drift

```bash
git fetch origin
git log <head_at_start>..HEAD --oneline
```

Where `<head_at_start>` = the `head_at_start` field in `<git-dir>/.claude.pid` written by `session-start.py`. If new commits appear, a peer terminal advanced HEAD during this session — re-read the affected files before committing.

### Check 2 — peer scope_lock claims

```bash
grep -l "<target-path>" docs/runtime/stages/*.md
```

Any matching stage file whose `mode` is not CLOSED/DONE is claiming exclusive write on the target path from a (potentially different) session. Confirm with the user before proceeding.

### Check 3 — sibling worktree heat

```bash
git worktree list --porcelain
# for each sibling worktree:
git -C <sibling-wt> status --porcelain -- <target-path>
```

Any sibling worktree with a dirty status on the same file path indicates parallel in-flight work. Pause and coordinate.

---

## Enforcement

The PreToolUse hook `shared-state-commit-guard.py` runs the three checks automatically before any `git add` / `git commit` that touches `docs/runtime/` or `docs/audit/`. On a hit, it prints a structured WARN block and exits 2 (BLOCK).

To override after manual verification, append `# --shared-state-ack` as a trailing comment to the bash command — the hook strips this flag-marker and proceeds.

```bash
git commit -m "..."   # --shared-state-ack
```

Fail-safe design: every read error, missing lock, parse failure, or non-git context exits 0 (pass). The guard never blocks a session it can't read.

---

## When you can skip the protocol

The hook auto-skips paths under:
- `docs/runtime/stages/` — stage files claim their own scope_lock; convention is one-terminal-per-stage.
- `docs/audit/hypotheses/drafts/` — draft preregs are author-owned.

If you are committing only files in those subpaths, the hook does not fire.

---

## Threshold for further hardening

Per `feedback_n3_same_class_doctrine_threshold.md`:

- **n=1** (THIS incident, 2026-05-21): feedback file + per-instance hook (this file + `shared-state-commit-guard.py`).
- **n=2** (future, different shared file): add a drift check enforcing that no shared-state file is staged alongside an unfinished sibling-session lock.
- **n=3+**: registry of all canonical shared-state files + meta-check verifying every entry has a sibling-coordination guard.

---

## Related

- `feedback_multi_terminal_shared_file_thrash_2026_05_21.md` — n=1 incident report
- `.claude/rules/parallel-session-isolation.md` — worktree-level rule (one Claude per worktree)
- `.claude/rules/branch-flip-protection.md` — branch-level rule
- `.claude/hooks/shared-state-commit-guard.py` — this rule's enforcement
- `.claude/hooks/session-start.py` § `_session_lock_lines()` — writes `head_at_start` field
