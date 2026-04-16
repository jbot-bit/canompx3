# TopStep Canonical Compliance Fixes — Implementation Plan

**Date:** 2026-04-08
**Branch:** `topstep-canonical-info`
**Source audit:** [`docs/audit/2026-04-08-topstep-canonical-audit.md`](../audit/2026-04-08-topstep-canonical-audit.md)
**Canonical corpus:** [`docs/research-input/topstep/`](../research-input/topstep/)

## Mandate

User directive 2026-04-08: *"do the proper scope fix... always the proper thorough institutional grounded way. no bias or skipping."*

Per `.claude/rules/institutional-rigor.md`:
- Each stage = one logical fix = one commit
- Each fix has `@canonical-source` annotation pointing to the exact verbatim quote in `docs/research-input/topstep/<file>.md`
- Each fix has at least one test (unit or integration)
- Drift check passes after each commit
- No production code change without first reading the affected file end-to-end (blast radius)

## Current deployment state (verified 2026-04-08)

- TopStep account `20092334` connected, equity $103,034.09
- Bot is running in `signal`/`demo` mode (per `live_journal.db.live_trades.session_mode`)
- Only 2 lifetime trades, both MNQ long, last activity 2026-04-06
- HWM tracker `data/state/account_hwm_20092334.json` shows `dd_limit_dollars=$2,000` (50K profile setting)
- **No real Express Funded Account is connected** — F-1, F-2, F-2b are LATENT until live mode activation

This means we have time to fix things properly before any rule break occurs in production.

## Fix scope (Scope C — full institutional)

| Stage | Finding | Severity | Type | Touches |
|---|---|---|---|---|
| 1 | F-9 + F-10 + F-11 annotations | INFO | Documentation | `prop_profiles.py`, `prop_firm_policies.py` |
| 2 | F-4 MNQ/MES commissions | HIGH | One-line bumps + annotations | `pipeline/cost_model.py` |
| 3 | F-6 5-XFA aggregate cap | MED | Startup assertion | `trading_app/pre_session_check.py` |
| 4 | F-5 HWM freeze formula for XFA | MED | New field + conditional | `prop_profiles.py`, `session_orchestrator.py` |
| 5 | F-2b CopyOrderRouter shadow asymmetry | HIGH | Reconciliation logic | `copy_order_router.py`, `session_orchestrator.py` |
| 6 | F-2 Hedging guard | BLOCKER | New RiskManager check | `risk_manager.py` |
| 7 | F-1 Scaling Plan enforcer | BLOCKER | New module | `trading_app/topstep_scaling_plan.py` (new), `risk_manager.py` |
| 8 | Drift check for canonical annotations | meta | New drift check | `pipeline/check_drift.py` |

**F-3 (LFA DLL):** DEFERRED — no LFA today. Stub LFA-aware profile fields in Stage 4 along with F-5 since both touch the same area.

**F-7 (Trade Copier):** REVISED in audit — replaced by F-2b. No standalone fix needed.

**F-8 (Vol-event risk adjustments):** Decision deferred. Current state: bot doesn't poll TopstepX for risk adjustments. Recommended path is operator runbook entry, not automation. Will document but not code.

**F-12 (VPS):** Hosting decision, not code. User confirmed home hosting → no fix needed.

**F-13 (Account stacking), F-14 (30-day inactivity), F-15 (Back2Funded):** Operational notes only. Document in MEMORY.md (already done in canonical-corpus commit).

## Per-stage gate

Before starting each stage:
1. Read all files in the "Touches" column end-to-end via Read tool
2. `git status --short` clean (or known intentional state)
3. Articulate the exact change in a comment block here

After each stage:
1. Run `python pipeline/check_drift.py` — must pass
2. Run targeted tests for the touched module
3. `git diff` self-review against the canonical citation
4. Commit with message: `fix(topstep): F-X <title> [stage N/8]`

## Stage 1 — Annotations (F-9 + F-10 + F-11) [stage 1/8]

**Goal:** Add `@canonical-source` comment blocks to existing fields. Zero behavior change. Establishes the citation pattern for all subsequent stages.

**Touches:**
- `trading_app/prop_profiles.py:248` — annotate the DLL exemption claim with `topstep_dll_article.md` URL + verbatim quote
- `trading_app/prop_firm_policies.py:36-91` — annotate every TopStep PAYOUT_POLICIES field with `topstep_payout_policy.txt` and `topstep_xfa_parameters.txt` quotes
- `trading_app/prop_firm_policies.py:49` — mark `additional_days_after_payout=5` for Standard as `@inferred` with note that this is not in canonical

**Tests:** none required (pure documentation). Drift check passes by default.

**Acceptance:** `git diff` shows only added comment lines, no code changes.

**Commit:** `docs(topstep): F-9 F-10 F-11 annotate verified payout/DLL fields with canonical sources`

## Stage 2 — F-4 Commissions [stage 2/8]

**Goal:** Bump MNQ and MES `commission_rt` from $1.24 to $1.42 (TopStep Rithmic rate, more conservative than Tradovate $1.34). Add `@canonical-source` to all COST_SPECS entries.

**Canonical:** `docs/research-input/topstep/topstep_xfa_commissions.md` lines 56-184 (Tradovate + Rithmic tables).

**Touches:** `pipeline/cost_model.py:90-107` (MNQ + MES).

**Why Rithmic not Tradovate:** Conservative bias. The exact TopstepX rate isn't documented in the article we have. Rithmic is the higher of the two known rates, so using it as the default cannot UNDER-estimate friction. When the user transitions to Tradeify (Tradovate) we can override per-firm.

**Blast radius:** All backtest results that use COST_SPECS will shift very slightly. ExpR for MNQ-heavy strategies will drop by ~$0.18 per round-trip × N trades. For a 200-trade/yr strategy with 0.4R win rate, that's ~$36/yr — small but real.

**Tests:**
- `tests/test_pipeline/test_cost_model.py` — assert MNQ.commission_rt == 1.42 and MES.commission_rt == 1.42
- Run `python pipeline/check_drift.py` — confirm no drift checks regress

**Acceptance:** Drift check passes. Cost model test passes. ExpR delta on validated_setups is < 5% (verified via spot-check on 1 active strategy).

**Commit:** `fix(cost-model): F-4 MNQ/MES commission_rt to canonical TopStep Rithmic rate ($1.42)`

## Stage 3 — F-6 5-XFA aggregate cap [stage 3/8]

**Goal:** Startup assertion that the sum of `copies` across active TopStep profiles is ≤ 5.

**Canonical:** `docs/research-input/topstep/topstep_xfa_parameters.txt:35,222` ("You can have up to 5 active Express Funded Accounts at the same time.")

**Touches:** `trading_app/pre_session_check.py` (likely; will Read first to confirm location).

**Tests:**
- Unit test: profile set with sum=5 passes; profile set with sum=6 fails fast.
- Integration: `pre_session_check` invocation against current profiles passes.

**Acceptance:** Test passes. Drift check passes.

**Commit:** `fix(pre-session): F-6 enforce 5-XFA aggregate cap across active TopStep profiles`

## Stage 4 — F-5 HWM freeze formula for XFA [stage 4/8]

**Goal:** Differentiate Trading Combine vs Express Funded Account semantics in HWM freeze calculation. Add `is_express_funded: bool` field to `AccountProfile`. Use `tier.max_dd + 100` for XFA, `account_size + tier.max_dd + 100` for TC.

**Canonical:** `docs/research-input/topstep/topstep_mll_article.md:60-66`:
> "Express Funded Accounts work the same way, but start at a $0 balance. For a $50,000 Express Funded Account, your Maximum Loss Limit starts at -$2,000 and trails upward as your balance grows. Once your balance reaches $2,000, the Maximum Loss Limit stays at $0."

**Touches:**
- `trading_app/prop_profiles.py` — add `is_express_funded: bool = True` to `AccountProfile` dataclass; default True for new XFA-style profiles, False for Trading Combine practice profiles
- `trading_app/live/session_orchestrator.py:407-409` — branch on `prof.is_express_funded` for the freeze calculation

**Stub for F-3 LFA:** Also add `is_live_funded: bool = False` field for future LFA support. Don't wire it up yet — just reserve the slot.

**Blast radius:** Existing AccountProfile instances need explicit `is_express_funded` value. Default True is correct for `topstep_50k_mnq_auto`, `topstep_50k_type_a`, `topstep_100k_type_a`, `topstep_50k`, `bulenox_50k`. The current Tradeify profiles also default True. The current TopStep practice account (20092334) is technically a TC — but the bot's `topstep_50k_mnq_auto` profile is XFA-shaped, so the field should be True.

**Tests:**
- Unit test: `freeze_at_balance` returns `max_dd + 100` for XFA, `account_size + max_dd + 100` for TC.
- Integration: HWM tracker initializes correctly for both types.

**Acceptance:** Tests pass. Drift check passes.

**Commit:** `fix(hwm): F-5 correct HWM freeze formula for Express Funded Account semantics`

## Stage 5 — F-2b Shadow failure asymmetry [stage 5/8]

**Goal:** Eliminate the silent state divergence when `CopyOrderRouter` shadow submissions fail.

**Canonical:** Bot risk per `docs/research-input/topstep/topstep_cross_account_hedging.md:273-288` ("You remain fully responsible for all activity across your accounts, including positions created through automated trading systems.") + the actual `copy_order_router.py:57-71` log-and-continue pattern.

**Decision pending blast radius read:** One of these two approaches:

**Option A — Fail-fatal-rollback:** If any shadow submit fails, immediately try to flatten the primary entry. If primary flatten fails too, halt the bot and require manual intervention. Strict but safe.

**Option B — Periodic reconciliation:** Continue current log-and-continue behavior but add a periodic position reconciliation in `session_orchestrator.py` that polls each account's positions every N seconds and halts if divergence is detected. More forgiving of transient broker errors.

Recommendation: **Option B** because transient broker rate-limiting is common and Option A would over-halt. Reconciliation interval = 60 seconds. Halt on divergence.

**Touches:**
- `trading_app/live/copy_order_router.py` — minor: tag failed shadows in a per-router state map
- `trading_app/live/session_orchestrator.py` — add `_reconcile_copies` async task that polls positions and compares
- `trading_app/risk_manager.py` — new halt reason `COPY_DIVERGENCE`

**Tests:**
- Unit test: divergence detector returns True when one account has a position the other doesn't
- Integration: simulated shadow failure triggers halt within 1 reconcile cycle

**Acceptance:** Tests pass. Drift check passes. Manual review of reconcile logic for race conditions.

**Commit:** `fix(copy-router): F-2b detect cross-copy position divergence and halt on detection`

## Stage 6 — F-2 Hedging guard [stage 6/8]

**Goal:** Pre-trade gate refusing entries opposite an existing open position on the same instrument within the same account.

**Canonical:** `docs/research-input/topstep/topstep_cross_account_hedging.md:55-71,317`:
> "Cross-account hedging occurs when you hold opposite positions across multiple accounts at the same time."
> "Yes! You can trade the same instrument across multiple accounts. What's prohibited is holding opposite positions simultaneously."

**Touches:**
- `trading_app/risk_manager.py` — new check in `can_enter()`: scan `active_trades` for any position with same instrument and opposite direction; if found, refuse entry with reason `OPPOSITE_INSTRUMENT_DIRECTION`
- New helper to determine instrument from strategy_id (or pass instrument explicitly)

**Tests:**
- `tests/test_trading_app/test_risk_manager.py` (likely exists; will Read first)
  - case: long MNQ open, attempt short MNQ → refused
  - case: short MNQ open, attempt long MNQ → refused
  - case: long MNQ open, attempt long MNQ → allowed (covered by other limits)
  - case: long MNQ open, attempt long MGC → allowed (different instrument)
  - case: long MNQ open, MNQ exited, attempt short MNQ → allowed

**Acceptance:** All 5 tests pass. Drift check passes.

**Commit:** `fix(risk-manager): F-2 reject opposite-direction entries on same instrument (cross-account hedging guard)`

## Stage 7 — F-1 Scaling Plan enforcer [stage 7/8]

**Goal:** New module `trading_app/topstep_scaling_plan.py` that maps current XFA balance → max mini-equivalent lots using the canonical ladder. Wire into RiskManager pre-trade check.

**Canonical:**
- Ladder image: `docs/research-input/topstep/images/xfa_scaling_chart.png` (visually parsed)
- Rules: `docs/research-input/topstep/topstep_scaling_plan_article.md`
- 10:1 ratio: `topstep_scaling_plan_article.md:78-83`
- No intra-day scaling: `topstep_scaling_plan_article.md:68`
- 10s grace period (manual fat-finger only): `topstep_scaling_plan_article.md:104`

**Module structure:**

```python
# trading_app/topstep_scaling_plan.py
"""TopStep XFA Scaling Plan enforcer.

@canonical-source docs/research-input/topstep/images/xfa_scaling_chart.png
@canonical-source docs/research-input/topstep/topstep_scaling_plan_article.md
@scraped 2026-04-08
"""

from dataclasses import dataclass

# Ladder values parsed from canonical PNG (50K/100K/150K accounts).
# Each tuple is (min_balance, max_lots).
SCALING_PLAN_LADDER: dict[int, list[tuple[float, int]]] = {
    50_000:  [(0.0, 2), (1500.0, 3), (2000.0, 5)],
    100_000: [(0.0, 3), (1500.0, 4), (2000.0, 5), (3000.0, 10)],
    150_000: [(0.0, 3), (1500.0, 4), (2000.0, 5), (3000.0, 10), (4500.0, 15)],
}

def max_lots_for_xfa(account_size: int, eod_balance: float) -> int:
    """Return the max mini-equivalent lots allowed for an XFA at this balance.

    @canonical-source docs/research-input/topstep/images/xfa_scaling_chart.png
    """
    ladder = SCALING_PLAN_LADDER[account_size]
    max_lots = ladder[0][1]
    for threshold, lots in ladder:
        if eod_balance >= threshold:
            max_lots = lots
    return max_lots

def micros_to_minis(micros: int) -> int:
    """Convert micro contracts to mini-equivalent (TopstepX 10:1 ratio).

    @canonical-source docs/research-input/topstep/topstep_scaling_plan_article.md:78-83
    """
    return (micros + 9) // 10  # ceiling
```

**Touches:**
- New file: `trading_app/topstep_scaling_plan.py`
- `trading_app/risk_manager.py` — new pre-trade check: get current XFA balance from HWM tracker, compute `max_lots_for_xfa`, check that `current_open_lots + new_lots ≤ max_lots`
- Possibly `trading_app/live/session_orchestrator.py` — wire HWM tracker → risk manager so risk manager can read EOD balance

**Tests:**
- Unit test: `max_lots_for_xfa(50_000, 0)` == 2
- Unit test: `max_lots_for_xfa(50_000, 1499)` == 2
- Unit test: `max_lots_for_xfa(50_000, 1500)` == 3
- Unit test: `max_lots_for_xfa(50_000, 2000)` == 5
- Unit test: `max_lots_for_xfa(100_000, 0)` == 3
- Unit test: `max_lots_for_xfa(100_000, 3000)` == 10
- Unit test: `max_lots_for_xfa(150_000, 4500)` == 15
- Unit test: `micros_to_minis(10)` == 1, `micros_to_minis(15)` == 2, `micros_to_minis(20)` == 2
- Integration: RiskManager rejects 5-lane entry on Day-1 50K XFA (balance=0)

**Open question — net position calculation:** Per `topstep_scaling_plan_article.md:71`, the article links to a separate article on net position calculation across simultaneous long and short positions. I have NOT fetched that article yet. For Stage 7, the conservative interpretation is **gross exposure** (sum of all open positions, regardless of direction). This may overestimate the lot count vs the canonical "net" rule, but errs on the safe side. Need to verify post-implementation by fetching `https://intercom.help/topstep-llc/en/articles/8284209` and adjusting if needed.

**Acceptance:** All 9+ tests pass. Drift check passes. Manual integration test: simulate a fresh 50K XFA Day-1 and confirm bot rejects 5th simultaneous lane entry.

**Commit:** `fix(scaling-plan): F-1 enforce TopStep XFA Scaling Plan ladder per-day`

## Stage 8 — Drift check for canonical annotations [stage 8/8]

**Goal:** New `pipeline/check_drift.py` check that scans `trading_app/` and `pipeline/` for `@canonical-source` comments and verifies the referenced files exist in `docs/research-input/topstep/` (or `docs/institutional/literature/`).

**Touches:** `pipeline/check_drift.py` (add new check function and register in main loop).

**Tests:**
- Inject a stale `@canonical-source` reference into a temp file → drift check fails
- Remove the stale reference → drift check passes

**Acceptance:** Drift check passes against the new annotations from Stages 1-7. Total drift check count increments by 1.

**Commit:** `feat(drift-check): verify @canonical-source annotations point to existing files`

## Re-scrape cadence

After all stages land, schedule a quarterly re-scrape (next: 2026-07-08) to detect TopStep policy changes. The drift check from Stage 8 will surface any annotation that points to a re-scraped file with mismatched content (assuming we add verbatim-text comparison, which is Stage 9 — out of scope for this plan).

## Risk and rollback

Each stage is one commit on `topstep-canonical-info`. If any stage fails its acceptance criteria or breaks an unrelated test, `git revert` that single commit and address the regression. Stages are independent except: Stage 4 depends on Stage 1 (annotation pattern), Stage 7 depends on Stage 4 (HWM tracker exposing balance) and Stage 6 (RiskManager test scaffolding), Stage 8 depends on Stages 1-7 (annotations to verify).

## Estimated commit count

8 stages = 8 commits + this plan doc = 9 commits total.

## Status tracker

Update this file as stages complete:

```
Stage 1 — Annotations (F-9 + F-10 + F-11)            [PLANNED]
Stage 2 — F-4 Commissions                            [PLANNED]
Stage 3 — F-6 5-XFA cap                              [PLANNED]
Stage 4 — F-5 HWM XFA freeze                         [PLANNED]
Stage 5 — F-2b Shadow asymmetry                      [PLANNED]
Stage 6 — F-2 Hedging guard                          [PLANNED]
Stage 7 — F-1 Scaling Plan enforcer                  [PLANNED]
Stage 8 — Drift check for annotations                [PLANNED]
```
