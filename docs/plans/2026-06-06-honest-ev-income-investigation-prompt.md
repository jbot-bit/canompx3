# Honest-EV Income Investigation — Resume Prompt (2026-06-06)

Paste the fenced block below into a fresh session after `/clear`. It is a
READ-ONLY investigation prompt. Paths verified to exist 2026-06-06; the
anti-look-ahead and binding-constraint rules encode the specific traps this
session caught (1.15yr selection-window, `validated_setups` derived-layer,
WITHDRAWAL-bound take-home, re-encoding the existing `simulate_withdrawals`).

## Session-verified facts the next session inherits (don't re-derive)
- Canonical DD anchor reconciled LIVE this session: **$1,535.22 ≤ $1,800 budget,
  gate=PASS** (deployed `topstep_50k_mnq_auto`, 3-lane MNQ book, 6.99yr window).
  `$2,038.84` is the UNCAPPED baseline — NEVER the live-capped anchor.
- The model (`scripts/reports/report_max_takehome.py`) ALREADY has
  `simulate_withdrawals` (balance / 5-winning-day gate / 50%-rule / MLL→$0 flag)
  and `greedy_size` (lot-cap). EXTEND, do not rebuild.
- Clamped headline (1 micro): take-home **$2,472–$2,523/yr, binds=WITHDRAWAL on
  all tiers** → account size is moot at 1 micro.
- Unclamped ceiling: 50K→$10,392 / 100K→$13,070 / 150K→$17,468 (×5 copy ≈
  $51k/$64k/$86k) — BUT annualized off a **1.15yr newest-lane window** (selection
  bias) and slippage at 4–15 lots is **UNMEASURED**. Bind is STILL WITHDRAWAL,
  not EDGE, even at full lot caps.
- The three model holes (all confirmed in code): lots don't drop on balance fall;
  sizing doesn't shrink after MLL→$0; consistency-40% rule absent.
- D-3 seam (verified): survival gate reads ZERO sizing inputs — it's structurally
  blind to contract count, hardcoded 1 micro. Not "two values to sync."

## THE PROMPT

```
You are in C:/Users/joshd/canompx3. Canonical shell is bash.

MISSION
Find the maximum honest EV path to bank real money from our ORB edge — via prop
firms or a better alternative. "Bank" = cash actually withdrawn after the
profit split, fees, and every firm rule. Not gross backtest PnL.

START FROM THE QUESTION: Are we asking the wrong question?
Then answer, in order:
  1. What did we actually test (vs. what we think we tested)?
  2. What question should actually decide income?
  3. What is the highest-EV next test?
  4. What is the best concrete way to bank cash?

DO NOT ASSUME — every one of these must be VERIFIED against an official source
or labelled ASSUMPTION with a range:
  - any Topstep rule (payout cadence, per-request cap, lifetime cap, MLL/floor,
    trailing-DD behavior, scaling-plan lots, copy-trade permission, account count)
  - slippage at >1 contract
  - that D-3 / clamp-lift is the best next move (it may be plumbing for an
    upside we haven't validated — say so if the data shows it)
  - that bigger account size earns more (this session's model showed take-home
    is WITHDRAWAL-bound and size-moot at 1 micro — re-verify, don't inherit)

READ FIRST (these paths are verified to exist 2026-06-06):
  - memory: project_max_takehome_model_and_clamp_lift_scope_2026_06_06.md
  - docs/research-input/topstep/topstep_payout_economics_2026-06-06.md  (sourcing)
  - docs/plans/2026-06-06-clamp-lift-d3-seam-income-scope.md            (scope)
  - scripts/reports/report_max_takehome.py     (the model — ALREADY has a
    withdrawal state machine `simulate_withdrawals` + lot-cap `greedy_size`;
    EXTEND it, do not re-encode it)
  - trading_app/account_survival.py             (canonical survival/DD truth)
  - trading_app/portfolio.py                    (canonical sizer:
    compute_position_size_vol_scaled / compute_vol_scalar)
  - trading_app/execution_engine.py             (live sizing: _compute_contracts)
  - trading_app/prop_portfolio.py               (book builder)
  - trading_app/topstep_scaling_plan.py         (SCALING_PLAN_LADDER, max_lots_for_xfa)
  - trading_app/prop_profiles.py                (AccountProfile, lane defs, ACCOUNT_TIERS)
  - trading_app/prop_firm_policies.py           (payout policies — NOTE: this file
    has a KNOWN STALE flat $5k/$6k cap; the per-tier truth is in the sourcing doc)

VERIFY CURRENT TRUTH (fail-closed):
  - Re-run the canonical survival loader (evaluate_profile_survival, write_state=False)
    and use ITS live DD as the anchor. Do NOT hardcode $1,535.22 — assert the
    fresh run reproduces it within tolerance; if it differs, the model is STALE,
    report and stop.
  - The $2,038.84 figure is the UNCAPPED baseline — NEVER use it as the live-capped
    anchor (that mislabel already poisoned prior C11 batons).
  - Mark any docs/memory/derived-layer value that conflicts with a fresh canonical
    run as STALE. Code + gold.db + canonical loaders are truth; docs are not.

ANTI-LOOK-AHEAD / ANTI-BIAS (mandatory — this is where the prior numbers lied):
  - The "unclamped ceiling" in the model annualizes off a 1.15yr window of the
    NEWEST lanes — that is selection bias toward favorable recent data. Any income
    ceiling MUST be reported on the SAME ~6.99yr deployed-book window as the
    headline, or explicitly flagged as short-window-directional and NOT banked.
  - `load_lanes` reads validated_setups — a DERIVED layer banned for truth-finding
    (research-truth-protocol.md). Ceiling lanes must trace to canonical
    orb_outcomes + applied filters, not a stale validated_setups snapshot.
  - No strategy P&L without its filter applied. No future bar as a predictor.
  - Holdout: 2026 data is sacred OOS — do not let it leak into any IS comparison.

CORE TESTS (read-only):

  1. FULL-HISTORY RE-WINDOW (highest priority — falsifies the whole clamp-lift thesis):
     Re-run the lot-capped ceiling on the SAME ~6.99yr deployed-book window, not
     the short 1.15yr newer-lane window. Does the ~4x upside ($2.5k -> $10-17k)
     survive full history, or is it a short-window artifact? One-function change
     to the existing model — reuse `book` over `days`, do not load the 18-lane
     newest-universe.

  2. EXTEND the existing withdrawal sim with its three documented-but-unenforced
     frictions (all currently flags/absent in simulate_withdrawals):
       a. lots DROP when a withdrawal pushes balance below a ladder threshold
          (today greedy_size sizes ONCE, statically, at the top-tier cap)
       b. sizing SHRINKS after MLL->$0 (today it's a printed flag only)
       c. the consistency 40%-largest-day rule GATES payouts (today absent)
     Output banked cash/month and /year. All three can only LOWER the ceiling —
     report how much.

  3. SLIPPAGE-AT-SIZE: edge survival at 1x/1.5x/2x/3x cost, for 4/5/10/15 lots.
     Slippage at >1 contract is UNMEASURED — label it ASSUMPTION with a range and
     show the break-even slippage that kills the clamp-lift upside.

  4. MULTI-ACCOUNT EV — only with an official source for: copy-trade permission,
     max concurrent accounts, and one-account-breach-vs-all risk. Model 1-5
     accounts with correlated-failure and total banked-after-cost. If the rule
     source is missing -> NEED SOURCE, stop that test.

  5. BETTER-FIRM COMPARISON: check OUR configured firms first (prop_profiles.py:
     Topstep deployed; Tradeify/Bulenox present but payout mechanics NOT modeled).
     Compare payout rules, DD rules, lot caps, fees, copy-trade, cadence, EV/month.
     If Topstep is not best for THIS edge, say so plainly. Any firm number without
     an official source -> ASSUMPTION or NEED SOURCE.

OUTPUT FORMAT:
  - Best opportunity
  - Biggest blocker
  - Biggest miss
  - Current framing trap
  - Best next test
  - Best firm / path
  - Expected banked cash/MONTH after costs (with window + the binding constraint named)
  - What to STOP wasting time on
  - What to do next (one concrete action)

RULES:
  - READ-ONLY until I explicitly say GO. No edits to pipeline/ trading_app/ scripts/.
  - No fake precision. Annualized-off-short-window numbers are DIRECTIONAL, say so.
  - No assumption without an "ASSUMPTION:" label and a range.
  - Missing official rule source -> stop and report "NEED SOURCE: <what>".
  - Prefer banked cash/month over gross PnL everywhere.
  - If D-3 / clamp-lift is NOT the highest-EV next action, say so directly and
    name what is.
  - Name the BINDING constraint on every income number (WITHDRAWAL vs EDGE vs
    SLIPPAGE vs CADENCE) — an unbound number is incomplete.
```

## Recommended run order (cost-aware)
Test 1 alone (~15 min, read-only) falsifies-or-confirms the clamp-lift thesis.
Run it FIRST and let its result decide whether Tests 2–5 are worth the context.
Tests 4 & 5 will hit NEED SOURCE for Tradeify/Bulenox (payout mechanics unmodeled,
`cap=None`) unless official policy docs are on disk.
