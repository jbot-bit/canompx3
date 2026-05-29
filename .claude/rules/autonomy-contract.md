# Autonomy Contract — act on reversible work, interrupt only when it matters

**Load-policy:** referenced from CLAUDE.md § Default Mode. Read on demand when
deciding whether an action needs operator confirmation.

**Authority:** the operator wants Claude to "get all the shit done" and check in
only when a human is genuinely needed — capital, schema, destructive, or truly
ambiguous design. This contract names that posture explicitly so it is
load-bearing, not implicit. It does NOT relax any existing safety gate; it sits
on top of `workflow-preferences.md` (trivial tier), `subagent-budget.md` (cost),
and `institutional-rigor.md` (the adversarial-audit gate stays mandatory).

---

## Tier A — act silently, report after

Proceed without asking, then report what changed in ≤3 lines. Eligible work:

- Config / docs / rule files (`.claude/rules/`, `docs/` non-canonical).
- `/resources` grounding: index rebuilds, citation lookups, extract reads.
- Permission-graveyard pruning (literal perms subsumed by a wildcard).
- Reversible refactors that meet the **trivial tier** in `workflow-preferences.md`
  (no `pipeline/` / `trading_app/` production logic, no schema/canonical-source
  change, < 100 net diff lines, verification lands in the same change).
- Read-only research, queries, audits, blast-radius mapping.
- Git `commit` / `push` / `merge` of already-completed, verified work (these mean
  "execute" per `workflow-preferences.md` § Git Operations).

## Tier B — always stop and surface first

Never act autonomously on these — present the decision and wait:

- **Capital paths:** anything under `trading_app/live/`, broker / execution
  engine, session orchestrator, lane activation, `--demo` / `--live` / strict-zero
  flips, risk-limit / kill-switch changes.
- **Schema / canonical-source changes:** `gold.db` schema, `pipeline/` canonical
  modules (`dst.py`, `cost_model.py`, `asset_configs.py`, `paths.py`,
  `holdout_policy.py`), entry-model / filter logic, drift-check semantics.
- **Destructive ops:** `rm -rf`, `git reset --hard`, `git clean -fd`, history
  rewrites, dropping/truncating tables, deleting non-self-authored files.
- **Genuinely ambiguous design forks:** more than one defensible architecture and
  the choice changes what gets built (use AskUserQuestion).
- **External / outward-facing actions:** publishing, posting, sending, anything
  that leaves the machine.
- **Promotion / deployment decisions:** moving a strategy toward live, flipping
  config that affects what trades.

When in doubt between A and B, it is B. Reversibility and capital-distance are
the test: if undoing it is cheap and no capital/schema is touched, Tier A; else
Tier B.

---

## The cheap + rigorous clause (load-bearing)

Autonomy is granted ONLY when the action is **both**:

1. **Cheap** — no unbudgeted subagent fan-out. Honor `.claude/rules/subagent-budget.md`:
   max one subagent per turn, inline when the alternative is < 5 reads / < 3 greps,
   prefer prompt-time cues + cached indexes (e.g. `resources/INDEX.md`) over fresh
   cold-context loads. Stop spawning past ~100K tokens.

2. **Rigorous** — fail-closed verification, `check_drift.py` passes, dead code
   swept, self-review before any claim-of-done, and the adversarial-audit gate
   intact on capital / truth-layer paths (`institutional-rigor.md` § 2).

If an autonomous action would be **expensive OR skip rigor**, it is NOT
autonomous — stop and surface it. This is the guardrail that keeps "act without
asking" from degrading into token-burn or band-aids. Speed is never bought with
rigor; cheapness is never bought by skipping verification.

---

## Reporting after Tier A work

Report is terse and factual (per `workflow-preferences.md` § Response Style): what
changed, the verification result (drift count, test pass/fail with output), and
anything that turned out to need a Tier B decision. No cheerleading, no
narration of obvious process.

---

## Related

- `.claude/rules/workflow-preferences.md` — trivial tier, git-ops-just-execute,
  response style, implementation gating.
- `.claude/rules/subagent-budget.md` — the cost half of the cheap clause.
- `.claude/rules/institutional-rigor.md` — the rigor half; adversarial-audit gate.
- `.claude/rules/shell-canon.md` — bash-canonical (reduces the error class that
  makes autonomous shell work risky).
