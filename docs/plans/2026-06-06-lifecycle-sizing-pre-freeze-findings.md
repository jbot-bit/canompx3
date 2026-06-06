# Lifecycle Sizing (Pre-Freeze De-Risk → Post-Freeze Expand) — Findings

**Date:** 2026-06-06 · **Status:** READ-ONLY research, adversarially red-teamed.
**Author note:** all numbers DIRECTIONAL (simplified path model, not the canonical
`simulate_survival`); gross-of-withdrawal-throttle unless stated. NOT banked cash.
**Probe:** `.claude/scratch/prefreeze_probe.py` (throwaway scratch).

---

## The idea (operator, 2026-06-06)

Topstep's Maximum Loss Limit (MLL) is a **trailing floor that FREEZES** once the
account reaches a fixed profit threshold. Before freeze the floor trails up under
you (a drawdown from a high can bust you while still in profit). After freeze the
floor locks → DD largely stops being the binding survival constraint.

→ So: **trade SMALL until frozen (survive the trailing phase), then size UP
(spend size where it's safe).** "Dynamic / lifecycle sizing." Operator also raised
"higher risk from the start" and "halt some lanes until we pass" as variants.

## Canonical grounding (verified, not memory)

`trading_app/account_survival.py` + `get_profile('topstep_50k_mnq_auto')`:
- `dd_type = "eod_trailing"`, `dd_limit_dollars = $2,000`, `starting_balance = $0`
- **`freeze_at_balance = $2,100`** → floor locks at $2,000 once balance hits +$2,100.
- Freeze is implemented (`frozen` branch, lines ~728–745). So the gate's DD already
  bakes in freeze; that's why `topstep_50k_mnq_auto` passes at DD $1,535 ≤ $1,800.

Deployed book = 3 MNQ lanes (COMEX_SETTLE OVNRNG_100, US_DATA_1000 VWAP_MID,
TOKYO_OPEN COST_LT08), 1144 common-window days, 6.99yr, mean $18.77/day, 1-micro
total $21,474.

## Result — variant ranking (Monte Carlo, 30k paths, deployed book)

| Variant | P(bust pre-freeze) | P(reach freeze) | Median post-freeze PnL (gross) |
|---|---|---|---|
| De-risk x1 → expand (15 lots) | 1.2% | 98.8% | $32,718 ← **ILLEGAL on 50k (see correction)** |
| De-risk x1 → expand (5 lots, 50k-legal) | 1.1% | 98.9% | **$12,218** (P bust_POST 1.99%) |
| De-risk x1 → expand (3 lots) | 1.2% | 98.8% | $7,311 (P bust_POST 0.17%) |
| Baseline static x1 | 1.2% | 98.8% | $7,292 |
| Higher-risk x3 from start | 21.6% | 78.4% | $26,311 |
| Max-risk x5 from start | 32.9% | 67.1% | $43,078 |

**Winner: de-risk-then-expand, lot-cap-honest.** "Higher risk from start" is
strictly dominated — it pays a 21–33% bust toll to skip a phase that's ~1% bust at
x1. The pre-freeze phase is nearly free to survive; spend size AFTER the floor locks.

## THE BIAS I CAUGHT (anti-pigeonhole — load-bearing)

The headline $32,718 used **x5 per lane = 15 total lots**, which **violates the 50k
scaling-plan cap of 5 lots** (15 lots is 150k territory). Honest 50k post-freeze =
**5 lots → ~$12,218 gross**, and P(bust_POST) rises to **1.99%** — freeze does NOT
make a fat book invincible; a 5-lot book can still breach the *locked* floor
($100 = FREEZE−MLL) on a bad day. Optimizing "size up after freeze" while silently
breaking the lot cap AND under-counting post-freeze bust = exactly the pigeonholing
the investigation prompt warned against. **Honest 50k lift ≈ 1.7× ($7.3k→$12.2k),
not 4.5×.** The 15-lot $32k applies to 150k accounts only.

## Adversarial red-team (all PASSED)

1. **Seed sensitivity** (5 seeds): bust 1.06–1.22%, post-PnL $32.4k–32.7k. Stable.
2. **Block bootstrap** (preserves losing streaks, block=1/5/10/20d): P(bust_pre)
   0.78–1.31%. No explosion → result is NOT a day-ordering / IID artifact.
3. **Actual chronological path** (no resampling): froze on day 125, **worst
   pre-freeze DD $299** (vs $2,000 floor = 6.7× margin), never busted, x5 post = $96k.
   Real history is kinder than the resampled cohort.

## HONEST CAVEATS (do not bank these numbers)

1. Probe is a SIMPLIFIED path model (daily-resampled book PnL, EOD trailing-freeze
   approximation). Canonical `simulate_survival` uses intraday min/max deltas. The
   DIRECTION is robust; the exact $ is not bankable. Re-run through canonical before action.
2. Numbers are GROSS post-freeze PnL — BEFORE 90/10 split, 50%-of-balance rule,
   per-request cap ($2k/$3k/$5k), 5-winning-day cadence. Withdrawal throttle WILL
   compress $12k materially (every prior tier was WITHDRAWAL-bound). Test 2 pending.
3. **D-3 seam is directly in this path.** Lifecycle sizing REQUIRES the live engine
   to size on `frozen` state, but the survival gate reads ZERO sizing inputs
   (hardcoded 1 micro). Arming lifecycle sizing without closing D-3 = gate validates
   a 1-micro book while engine trades a 5-lot post-freeze book = silent
   capital-under-protection. HARD blocker.
4. `max_contracts=1` clamp blocks any size >1 in the live engine today.

## Firm-selection angle (NEED SOURCE)

"Some firms have a moving floor, some don't" (operator) — CORRECT and a real lever.
A **static floor from day 1** (no trailing) would make even the pre-freeze phase
free → x1 bust toward 0%, size up from day 1. Tradeify/Bulenox are in
`prop_profiles.py` but payout/floor mechanics are NOT modeled (`cap=None`).
→ **NEED SOURCE: Tradeify + Bulenox floor type (trailing-freeze vs static-EOD vs
static-day-1) and per-tier payout caps** before any better-firm comparison.

## TEST 2 — withdrawal-bound banked take-home (real path, lot-cap honest)

Lifecycle policy (pre-freeze 3 lots → post-freeze tier lot cap) run THROUGH the
withdrawal state machine (90/10 split, per-request cap, 50%-rule, 5-winning-day
gate, $30 fee). **Binds=WITHDRAWAL on every tier** (cadence still the ultimate throttle):

| Tier | Post-freeze lot cap | Lifecycle take-home/yr | vs clamped 1-micro ($2,536) | x5 copy |
|---|---|---|---|---|
| 50k | 5 | $4,014 | 1.6× | $20,069 |
| 100k | 10 | $8,156 | 3.2× | $40,778 |
| **150k** | **15** | **$12,104** | **4.8×** | **$60,519** |

## ⚠⚠ CORRECTION (2026-06-06, operator-caught) — BUFFER-AWARE numbers SUPERSEDE Test 2 above

Test 2 had a real ERROR: it generated PnL at a STATIC post-freeze lot count for the
whole window AND banked withdrawals on top — but Topstep policy makes these mutually
exclusive. Official rules (fetched 2026-06-06, help.topstep.com 8284233/8284215/8284204):
- XFA balance STARTS at $0 (buying power = acct size); payouts are REAL money.
- **MLL→$0 after first payout** (permanent). To bring MLL to $0 you must first EARN the
  full MLL distance as un-withdrawable buffer: 50k=$2,000 / 100k=$3,000 / **150k=$4,500**.
- Withdraw before MLL=$0 → buffer FORFEITED anyway. After: floor = remaining capital;
  a payout that drops a scaling level CUTS your contract count.

**Buffer-aware honest take-home** (build MLL→$0 first, keep reserve=MLL dist, harvest excess;
`.claude/scratch/buffer_aware_takehome.py`):

| Tier | Build days to MLL→$0 | Honest take-home/yr | % of inflated Test-2 | ×5 copy |
|---|---|---|---|---|
| 50k | 126 (~6mo) | **$2,844** | 71% | $14,220 |
| 100k | 191 (~9mo) | **$4,776** | 59% | $23,880 |
| **150k** | **195 (~9mo)** | **$7,230** | **60%** | **$36,150** |

**Buffer is a REGRESSIVE tax on size:** 150k must build $4,500 dead buffer vs 50k's $2,000,
over a longer build → keeps only 60% of naive vs 50k's 71%. Incremental 150k-vs-50k
= **+$4,386/yr (was +$8,090 — roughly HALVED)**. Still positive, ~20× payback on +$200 cost,
but the $58k×5 headline is now **$36k×5**, before slippage@15 + simplified-path haircut.
Phase-2 harvest (reserve=MLL dist) is a MODELING CHOICE, not optimized. Scaling-plan
downgrade-on-withdrawal NOT yet modeled (would shave 150k further). Operator intuition
("6k+ before anything, worse on 150k") = CONFIRMED.

## 150k WORTH-IT verdict — YES (conditionally) [PRE-BUFFER-FIX; see correction above]

- **+$8,090/yr take-home vs 50k for +$200 one-time eval cost = 40× yr-1 payback, ~free after.**
  Funded XFA has no recurring monthly; only the higher Combine sub during ~2mo eval.
- **Mechanism = the LOT CAP, not the edge.** Same 3-lane MNQ edge, same ~1% pre-freeze
  bust. 150k legally holds 15 post-freeze lots vs 5 on the 50k → 3× safe size once frozen.
- **Exploit:** 1 micro/lane until +$2,100 freeze (~96 days, worst hist DD $299) → scale
  to 15 lots → ×5 sanctioned copy-trade ≈ $58k/yr net.
- **CONDITIONAL on:** D-3 seam closed + clamp lifted + slippage@15lots measured +
  150k profile created. Until then $12k is a PROJECTION; deployable 50k-clamped reality ≈ $2.5k.
- Account size is MOOT only UNDER the clamp (inherited finding was right for that regime);
  lifecycle sizing is what makes bigger accounts worth it.

## Next actions (gated)

1. **[READ-ONLY] Test 2** — layer the withdrawal state machine onto the lifecycle
   policy → banked cash/yr (the number that actually matters). Do BEFORE any build.
2. **[READ-ONLY] NEED SOURCE** — competitor floor mechanics (above).
3. **[Tier B, needs GO + adversarial audit]** — close D-3 seam + lift clamp +
   add `frozen`-aware sizing to the live engine. NOT before 1 & 2.

## Related
- `docs/plans/2026-06-06-honest-ev-income-investigation-prompt.md` — parent prompt.
- `project_max_takehome_model_and_clamp_lift_scope_2026_06_06.md` — D-3 seam, clamp.
- `scripts/reports/report_max_takehome.py` — the withdrawal/lot-cap model to extend.
