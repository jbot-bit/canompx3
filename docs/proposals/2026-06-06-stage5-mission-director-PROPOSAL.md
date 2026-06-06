# PROPOSAL — Fleet-State Brain Stage 5: Mission-Director / NORTH_STAR re-anchor

**Status:** PROPOSAL for the fleet-state-brain owner to review/approve. NOT merged,
NOT built. Authored by a peer session (capital lane) at operator request.
**For plan:** `docs/runtime/active_plan.md` / `C:/Users/joshd/.claude/plans/b-full-planned-out-lucky-hummingbird.md`
**Operator verdict (2026-06-06):** "Make this a Stage 5 mission-director / NORTH_STAR
scan, NOT a new competing tool. The fleet-state brain already owns coordination;
standalone tooling would recreate split-brain risk. Official docs first, then public
examples, then repo reality. No cherry-picking."

---

## Why (the gap this closes)

The fleet-state brain solves **worktree drift** (who's live/dirty/hollow). It does NOT
solve **mission drift** — root cause #3 in the master plan ("forget the original
plan"). Symptom, observed live this session: the operator had to ask "what's our
overall mission and are we following it?" mid-flight; 4 terminals coordinated for hours
toward a push with no shared goal anchor. There is no active director — `project_pulse`
is passive (must be invoked); `decision_governor` is per-decision; nothing surfaces the
mission unprompted every session.

This Stage adds the **mission-director**: the "overseer that watches and directs,"
built on the brain, not beside it.

## Source-grounded design (5 steal / 5 reject)

**Official Claude hooks docs** (code.claude.com/docs/en/hooks-guide): hooks give
"deterministic control... ensuring certain actions always happen rather than relying on
the LLM to choose." SessionStart fires on start/resume/clear/compact and can inject
`hookSpecificOutput.additionalContext` (model-visible).

**STEAL:**
1. **Persistent human-written anchor that survives context reset** (Ralph Loop
   `tasks.json` pattern) → `NORTH_STAR.md` (operator-owned: mission + phase + the ONE
   next deliverable). Already drafted + operator-confirmed 2026-06-06 at repo root.
2. **SessionStart `additionalContext` re-anchor** (official) → surface NORTH_STAR +
   `project_pulse --fast` + `fleet_state()` summary every session/clear/compact.
3. **Plan-approval before capital code** (official across Agent Teams/OMC/OmO) → make
   it a standing surfaced rule, not occasional.
4. **Worktree-isolation ENFORCED** (official + git docs) → already Stage 2; the director
   surfaces "you are in canonical root, spawn a worktree" when it detects it.
5. **Human authors/approves the anchor; agent only drafts** (MEASURED: dev-written
   context +4% success vs agent-generated −3% / +20% cost) → NORTH_STAR is operator-owned.

**REJECT:**
1. Multi-agent frameworks (Sisyphus/Forge, 11-agent fleets) — "$24K tokens on personal
   projects." We need an anchor, not a swarm.
2. Multi-model/provider routing — only worth it at 5+ parallel agents.
3. Cloud-async as the primary spine — it's for overnight backlog, a side lane.
4. Agent-generated mission docs — measured WORSE. Human-owned only.
5. **A new standalone director tool** — operator's #1 constraint + this session's
   hardest lesson. Must consume `fleet_state()`, add no parallel system.

## Proposed Stage 5 scope (for the owner to refine)

- **Consume, don't rebuild:** read `NORTH_STAR.md` + call `fleet_state()` +
  `project_pulse` — re-encode nothing.
- **One SessionStart cue** (extends the existing session-start additionalContext path):
  prints `[MISSION] <one-line goal> | [PHASE] <current> | [NEXT] <single deliverable> |
  [FLEET] <n live / n dirty>`. Drift-arrest line if the current branch/files diverge
  from NORTH_STAR's "next deliverable."
- **NORTH_STAR is operator-owned** — the hook READS it, never writes it.
- **No blocking** — advisory/legibility per the "enforce-at-chokepoint, nag-locally"
  doctrine already in the master plan's lit grounding. The hard safety stays the
  existing guards.
- **Registration drift-check** so the cue can't silently fall out of settings.json
  (same gap that left `mcp-git-guard.py` orphaned).

## What's already done (so the owner doesn't redo it)
- `NORTH_STAR.md` drafted + operator-confirmed (repo root, 2026-06-06).
- Source scan complete (this file).

## Sources
- Official: code.claude.com/docs/en/hooks-guide ; git-scm.com/docs/git-reset
- htdocs.dev/posts/from-conductor-to-orchestrator-a-practical-guide-to-multi-agent-coding-in-2026
- beam.ai/agentic-insights/multi-agent-orchestration-patterns-production
