# NORTH_STAR — the mission, current phase, single next thing

> **DRAFT — operator must approve.** This is the human-owned mission anchor (per the
> measured rule: developer-written context = +4% success; agent-generated = −3%).
> Claude drafts/suggests; **the operator edits and owns this file.** Re-surfaced
> every session by the mission-director (fleet-state brain Stage 5).
> _Operator-confirmed mission + phase: 2026-06-06._

---

## The mission (why this project exists)

Run an **automated ORB futures bot that trades real prop-firm capital SAFELY**, then
**scale the banked income** — without ever risking an account through a silent
safety gap.

Two halves, in order:
1. **SAFE TO LIVE** — the bot can arm real capital on `topstep_50k_mnq_auto` with
   every survival/routing/exposure gate provably bound to live behavior.
2. **SCALE THE INCOME** — lift the `max_contracts=1` clamp (per the firm's 5/10/15
   balance-gated lot ladder) so realistic banked cash goes from **~$2,500/yr (1 micro,
   withdrawal-bound) → ~$10k–17.5k/acct → ~$50k–86k/yr across 5 copy-traded accounts.**

Edge discovery (the ORB pipeline on MGC/MNQ/MES) is the *engine*; safe-live + scale is
the *mission*. We are past "find an edge," now in "deploy it safely and scale it."

## Current phase (confirm/correct each session)

**Phase: SAFE-TO-LIVE, nearly there.** As of 2026-06-06, origin/main `bfa10afe`:
- The 4 Codex capital-safety bugs (naked-position, account-routing, D-3 parity, stale
  fingerprint) are **fixed and on origin**.
- C is an **interim placeholder** (`assert max_contracts==1`) — it enforces "don't
  scale until DD math is honest," it does NOT yet enable scaling.
- **Nothing is armed.** Live-arming is operator-only (`.\start_bot.bat live` + CONFIRM).

## The single next deliverable (ONE thing — don't drift)

**Real Stage-1 DD-scaling** (replaces the C placeholder): make `account_survival.py`
project drawdown at the contract count the live engine WOULD trade — by **delegating to
the shared vol-sizer (no re-encode)** — so multi-lot lanes pass C11 legitimately.
This is the gate that unlocks the entire income half.

**Hard prereqs before that ships** (do not skip):
- Bracket-parity audit `9b3fc530` CLOSED.
- Edge-at-scale slippage MEASURED at 5–15 lots (currently UNMEASURED).
- MLL→$0-post-payout lot-ladder shrink modeled in `report_max_takehome.py`.

Plan: `docs/plans/2026-06-06-clamp-lift-d3-seam-income-scope.md`.

## Standing rules (how we work — so we stop thrashing)

1. **One session per worktree.** Never destructive git ops in the canonical tree.
   Spawn a worktree (`scripts/tools/new_session.sh`).
2. **Plan-approval before capital code.** Bounded plan + operator GO first (10s review
   catches 80% of errors).
3. **Operator owns this file + arming.** Agent drafts; operator approves. Agent never arms.
4. **No re-encoding canonical logic.** Delegate to the one source (vol-sizer, gates).
5. **Assume parallel.** Other terminals are always live; check before touching shared state.

## What this file is NOT
Not a roadmap of features (see git history), not a task queue (see
`docs/runtime/action-queue.yaml`), not status (see `project_pulse.py`). It is the **one
durable answer to "what are we doing and what's the next thing"** — pinned so no session
re-derives it.
