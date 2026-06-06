# D-3 Sizing-Seam Close — Stage 1 (DESIGN / SPEC)

**Date:** 2026-06-07
**Mode:** DESIGN — operator GO received for Stage 1 only. Tier B (touches the canonical
capital-survival gate). Stages 2/3/4 remain separately gated.
**Parent scope:** `docs/plans/2026-06-06-clamp-lift-d3-seam-income-scope.md`
**Audited:** `/check` institutional audit run against canonical code BEFORE this spec
(findings folded in — see "Audit-grounded facts").

---

## 1. Goal

Make the account-survival gate project 90-day drawdown at the contract count the **live
execution engine would actually trade**, instead of a hardcoded 1 micro. This closes the
"D-3 seam" — the gap where the survival gate certifies DD-safety at 1 micro while the engine
is free to size larger.

**Stage 1 is gate-only and creates ZERO new capital exposure:**
- It touches the survival **gate** (`account_survival.py`), never the executor.
- It does **not** raise `max_contracts` — that stays 1, so **no live verdict changes today**.
- It makes the gate's DD math *honest for whatever cap is later set*, so Stage 3 (the actual
  clamp lift) becomes provable rather than blocked.

Non-goal (later, still gated): the scaling-plan lot ladder as config (Stage 2), raising the
clamp (Stage 3, operator `--live` GO + adversarial audit + bracket audit `9b3fc530`), income
re-model (Stage 4).

## 2. Audit-grounded facts (the build must respect these)

All verified against canonical code / `gold.db` PRAGMA on 2026-06-07:

1. **The vol-scaled sizer is already a shared module.** `portfolio.py:218`
   `compute_vol_scalar(atr_20, median_atr_20)` and `portfolio.py:240`
   `compute_position_size_vol_scaled(equity, risk_pct, risk_points, cost, vol_scalar)`.
   The execution engine merely calls them (`execution_engine.py:287`). The survival gate
   must import and call the **same** functions — no re-encode.

2. **`atr_20` is discarded before the survival sim sees it.** `_load_strategy_outcomes`
   (`strategy_fitness.py:454`) returns `orb_outcomes` rows only; `daily_features` (which
   carries `atr_20`) is loaded solely to compute `eligible_days`, then dropped
   (`strategy_fitness.py:450-451`). So each `outcome` in `_load_lane_trade_paths`
   (`account_survival.py:427`) has `entry_price`/`stop_price`/`pnl_r`/`mae_r` but **no ATR**.
   ➜ A naive implementation would feed nulls to `compute_vol_scalar`, silently get
   `vol_scalar=1.0`, and ship flat-linear sizing while believing it shipped engine-faithful.
   **This must be prevented.**

3. **`median_atr_20` is NOT a stored column.** It is a runtime **trailing 252-day rolling
   median** of `atr_20` (`SELECT MEDIAN(atr_20) FROM daily_features` over a lookback).
   Canonical computers: `paper_trader._get_median_atr_20` (`paper_trader.py:132`) and
   `session_orchestrator.py:1216-1228` (explicit comment: "median_atr_20 is NOT in
   daily_features"). `gold.db` PRAGMA: `orb_outcomes` has neither ATR; `daily_features` has
   `atr_20` (99.9% non-null, 35441/35463, 22 NULL) and **no** `median_atr_20`.

4. **The engine fails OPEN on missing vol inputs.** `execution_engine.py:276-285`: if
   `atr_20<=0 or median_atr_20<=0`, `vol_scalar=1.0` with a logged warning. The survival
   sim must mirror this fallback **explicitly and logged** — never a silent default.

5. **A handcuff guard already exists.** `_assert_single_micro_sizing`
   (`account_survival.py:836`) currently fails the C11 gate closed whenever
   `max_contracts != 1`. Stage 1 converts it from a blunt handcuff into a real parity proof.

## 3. Design decisions (locked with operator)

| # | Decision | Rationale |
|---|----------|-----------|
| Fidelity | **Engine-faithful** per-trade vol-scaled sizing | only version that truly closes the seam |
| Equity | **Fixed starting equity** (constant per trade) | conservative: sim never de-risks during DD → models the LARGEST size the engine would trade → DD overstated, never falsely lenient; deterministic, no path-dependent feedback loop |
| Lot cap | **Reuse existing `PortfolioStrategy.max_contracts`** (=1 today) | does not front-run Stage 2's ladder; gate becomes size-aware for whatever cap is later set; Stage 1 alone changes no verdict |
| Median | **Canonical trailing 252d median** via `paper_trader._get_median_atr_20`, point-in-time as-of each trade day | no re-encode; trailing (not full-history) avoids leaking future volatility into past sizing (look-ahead) |

## 4. Architecture & data flow

One canonical sizing source, two consumers (engine + survival sim).

```
orb_outcomes rows  ─┐
                    ├─► sim-local helper queries daily_features for {trading_day: atr_20}
daily_features ─────┘   (own batched query — NOT feat_dicts, which are discarded) +
                        trailing median via canonical _get_median_atr_20 (as-of day)
                              │
                              ▼
   IF size_model is None (correlation/allocation/daily-pnl callers):
                    contracts_per_trade = 1 ; pnl_dollars UNCHANGED   # byte-identical to today
   ELSE (survival caller only — SizingContext from build_profile_portfolio):
     vol_equity = ctx.account_equity        # Portfolio.account_equity NOTIONAL (>0) — vol-risk sizing input
     assert vol_equity > 0                  # fail-closed: zero-equity DD projection is never a PASS
     cap_balance = ctx.cap_balance          # EOD/notional balance for the XFA LOT CAP (see note below)
     for each trade:  vol_scalar = compute_vol_scalar(atr_20, median_atr_20)   # portfolio.py
                      n = compute_position_size_vol_scaled(vol_equity, risk_pct, risk_points, cost, vol_scalar)
                      n = min(n, ctx.max_contracts,
                              max_lots_for_xfa(ctx.account_size, cap_balance))  # canonical ladder, = 1 today
                      pnl_dollars *= n ; mae_dollars *= n ; contracts = n ; lots = lots_for_position(instr, n)
   # NOTE (3rd-audit): TWO equity concepts — do NOT conflate.
   #   vol_equity  = Portfolio.account_equity notional  → drives vol-risk size (engine: execution_engine.py:267)
   #   cap_balance = balance for the XFA lot ladder      → max_lots_for_xfa canonically wants EOD broker balance
   #                 (topstep_scaling_plan.py:96 "use EOD balance, not live equity"; XFA starts $0 & grows).
   #   For a HISTORICAL survival sim there is no live EOD balance per day. Stage-1 default: cap_balance =
   #   ctx.account_equity (notional) so the cap is non-binding at cap=1 today (no verdict change). Modelling
   #   the growing-balance lot ladder per simulated day is Stage 4's job (it needs the balance↔lots↔withdrawal
   #   coupling already flagged unmodeled). Stage 1 MUST document this so it is not mistaken for engine-exact.
                              │
                              ▼
   existing _max_observed_rolling_drawdown aggregation (UNCHANGED)
```

### Injection boundary — REVISED after 2nd audit + real-data run (LOAD-BEARING)

**The prior draft put scaling inside `_load_lane_trade_paths`. That is WRONG and is replaced.**

Audit finding (grounded): `_load_lane_trade_paths` pnl is NOT survival-private. Its sibling
`_load_lane_daily_pnl` (`account_survival.py:368`) sums the same `TradePath.pnl_dollars` and is
consumed by **correlation + allocation**: `lane_correlation.py:190,195`, `lane_allocator.py:693`
(via `_cached`), `scripts/research/portfolio_correlation_audit.py:63`. Scaling there would silently
alter `_pearson` (`lane_correlation.py:204-207`) and allocation. Additionally, `_load_lane_trade_paths`
receives only `strategy_id` and has NO access to `max_contracts`/`account_equity`/`risk_per_trade_pct`
(absent from `validated_setups` — PRAGMA-confirmed; they live on the `Portfolio`/`PortfolioStrategy`
object, `portfolio.py:86,101,102`), so scaling there is also unbuildable without a signature change —
the very change that leaks into correlation/allocation.

**Real-data severity check (gold.db, not synthetic, not metadata):** ran the actual deployed book
(all GC — same instrument, per Carver Ch.11 single-instrument premise,
`carver_2015_ch11_portfolios.md:37`) through `_load_lane_daily_pnl` + real `atr_20` vol-scalars.
28 lane pairs: mean rho delta 0.021, max 0.120. Max occurred on an already-uncorrelated, low-overlap
pair (n=74, rho 0.000→−0.120); allocation-relevant high-rho pairs (+0.78, n=1373+) moved 0.001–0.004.
The allocator gates `rho < 0.70` (`lane_allocator.py:976` `RHO_REJECT_THRESHOLD`); the real rho
population is bimodal (~0.0 or ~0.78–1.0), none near 0.70 with a material delta → **distortion cannot
flip an accept/reject on today's book.** So the harm is decision-immaterial *today*, but the design must
still fail-safe by construction (a future cross-instrument lane reintroduces divergent per-day scalars).

**REVISED mechanism — opt-in `size_model`, default None (today's behavior). CORRECTED by 3rd
audit for the express-funded equity fail-open + canonical-lot reinvention:**

- `_load_lane_trade_paths(con, strategy_id, *, …, size_model: SizingContext | None = None)`. When
  `size_model is None` (every existing caller — correlation, allocation, the daily-pnl path), behavior
  is byte-identical to today: `contracts_per_trade = 1`, raw `pnl_dollars`. **No blast radius.**

- ONLY the survival-private caller `_load_profile_daily_scenarios` (`:567-574`) passes a
  `SizingContext` (a thin struct, NOT a new ladder — see below) built from
  `build_profile_portfolio(profile_id)` (imported `:40`, used `:853`): it carries
  `account_equity`, `risk_per_trade_pct`, the per-`strategy_id` `max_contracts`, AND the
  profile's `account_size` (for the XFA cap).

- **EQUITY SOURCE — corrected (3rd audit, CRITICAL).** The sizer's `account_equity` MUST be the
  **`Portfolio.account_equity` notional** (`portfolio.py:101`, default 25000.0 `:473`) — the SAME
  value the live engine sizes from (`execution_engine.py:267` `equity = self.portfolio.account_equity`).
  It MUST NOT be `SurvivalRules.starting_balance`, because for express-funded (XFA) profiles
  `starting_balance = 0.0` (`account_survival.py:625`; `prop_profiles.py:121` "XFA accounts start at
  $0 broker equity"; `:627` `is_express_funded=True` for Tradeify 50k). Feeding 0.0 to
  `compute_position_size_vol_scaled` returns 0 contracts (it guards `risk_points/vol_scalar` but
  NOT `account_equity<=0`, `portfolio.py:255-258`) → DD projected at $0 → gate **falsely PASSES**.
  The live engine never sizes vol-risk off the $0 broker balance — only the XFA *lot cap* uses live
  balance. So "fixed equity" = `Portfolio.account_equity`, not the starting balance.

- **LOT CAP — reuse canonical, do NOT invent.** The contract clamp is `min(vol_scaled,
  max_contracts, max_lots_for_xfa(account_size, equity_for_cap))`. `max_lots_for_xfa`
  (`topstep_scaling_plan.py:88`) is ALREADY the balance-gated ladder, already imported in
  `account_survival.py:51` and used at `:636,:720`, and is the SAME cap the live risk manager
  applies (`risk_manager.py:260`). The earlier draft's invented `SizeModel.max_contracts_for`
  is dropped — it would re-encode this canonical ladder (forbidden). `SizingContext` only
  *transports* values; the cap math stays in `max_lots_for_xfa`.

- **EQUITY GUARD (fail-closed).** The parity guard fails closed if the equity reaching the sizer
  is `<= 0` — a zero-risk DD projection is never a valid PASS. This closes the express fail-open
  by construction even if a future caller mis-wires equity.

- The sizer call stays INSIDE `_load_lane_trade_paths` (where `risk_points = abs(entry-stop)` is derived
  at `:438` — `TradePath` does NOT expose `risk_points`/`entry`/`stop`, confirmed `:189-209`, so caller-
  side sizing is not possible without widening TradePath; keeping it in-function is the smaller diff).
- This makes "survival-private" TRUE by the arg, not by hopeful call-graph reasoning: correlation/
  allocation pass no `size_model` ⇒ provably unscaled, pinned by a test (§6-g).

### Components (each independently testable)

1. **Sim-local ATR enrichment** (NEW helper inside `account_survival.py`, called from
   `_load_lane_trade_paths`). **AMENDED (audit M2/M4/M5):** `_load_strategy_outcomes`
   **discards** its `feat_dicts` before returning, and it is shared by 5 other callers
   (`deployability.py:455`, `lane_correlation.py:90`, `sprt_monitor.py:124`,
   `sr_monitor.py:143`, `walkforward.py:106`) — so enrichment MUST NOT be added there
   (would contaminate all of them) and `feat_dicts` are NOT in memory at the sim loop.
   Instead, the sim-local helper issues its **own** `daily_features` query for
   `(instrument, orb_minutes=5)` over the trade window and builds a
   `{trading_day: atr_20}` map. *Boundary:* input = (con, instrument, trade-day set);
   output = atr-by-day map. The "no SQL on the hot path" wording from the prior draft was
   wrong and is removed — one batched query per lane (not per trade) is the cost.
   The `NO_FILTER` path (`_load_strategy_outcomes:498-506`, `feat_dicts=[]`) is irrelevant
   here because this helper queries `daily_features` directly, independent of the filter path.

2. **Trailing median provider** (delegate to `paper_trader._get_median_atr_20`): returns the
   252d trailing median of `atr_20` as of a trading day. *Boundary:* input = (con, instrument,
   trading_day); output = float median. **Reused, not reimplemented.** Audit-confirmed
   look-ahead-safe (`paper_trader.py:143` `trading_day < ?`).

3. **Sim sizer call** (in `_load_lane_trade_paths`, GATED by `size_model`, replacing
   `contracts_per_trade = 1` at `account_survival.py:424` ONLY when `size_model is not None`):
   per-trade `n` via the shared `portfolio.py` sizer using `ctx.account_equity` (the **notional**,
   not starting_balance) + day vol_scalar, clamped to
   `min(ctx.max_contracts, max_lots_for_xfa(ctx.account_size, equity))` — the **canonical** ladder,
   not a re-encode. *Boundary:* input = trade row + ATR + SizingContext; output = `n`, scaled pnl/mae.
   `risk_points` is derived in-loop at `account_survival.py:438` (`abs(entry-stop)`) — confirmed available.
   `size_model is None` ⇒ untouched `contracts_per_trade = 1` path (every non-survival caller).

4. **Parity guard** (rewrite of `_assert_single_micro_sizing`): assert the sim's per-trade
   contract resolution equals the shared sizer's output for matched inputs. **AMENDED (audit
   M3):** a NULL/≤0 `atr_20` on a priced day is NOT a parity failure — it is the same
   logged `vol_scalar=1.0` fallback the engine takes (`execution_engine.py:276-285`), so the
   gate still PASSES (parity is preserved: both sides fall back identically). Fail closed ONLY
   when ATR is *structurally* unobtainable (column/table missing, query error, enrichment
   raised) — i.e. parity cannot be evaluated at all. Mirrors existing `:854` "any builder
   error fails closed".

5. **Drift check** (new, capital-class, `requires_db`): assert the survival sim and execution
   engine resolve contract count through the **same** canonical `portfolio.py` helper, and
   that median_atr_20 comes from the canonical provider — catches a future re-fork of the seam.

6. **Dead-field cleanup** (audit M1): `SurvivalRules.contracts_per_trade_micro`
   (`account_survival.py:130,653`) is set but never read by `simulate_survival` — PnL is
   pre-scaled at `TradePath` construction. Stage 1 adds a code comment marking it vestigial
   (do NOT remove it — out of scope, and a removal touches the rules dataclass), so a future
   implementer does not mistake it for a second sizing knob.

## 5. Error handling / fallbacks (all fail-LOUD, never silent)

| Condition | Behavior |
|-----------|----------|
| `atr_20` NULL or ≤0 for a trade day (22 such rows: MNQ/MES/MGC × 2019-05-06 & 2022-06-13; MNQ+MES are live-book) | `vol_scalar=1.0`, **logged warning**, counted in a fallback tally surfaced in the report. **Parity PRESERVED — gate still PASSES** (engine takes the identical fallback). NOT a fail-close. |
| trailing median unobtainable (<lookback history) | `vol_scalar=1.0`, logged; parity preserved (engine does same) |
| ATR column/table missing, query raises, or enrichment errors (structural) | parity guard **fails closed** — gate blocks (parity cannot be evaluated at all) |
| **Express-funded profile (`is_express_funded=True`, `starting_balance=0.0`)** — Tradeify 50k `prop_profiles.py:627` | size off `Portfolio.account_equity` NOTIONAL (>0), NOT starting_balance. If the equity reaching the sizer is ≤0 → **parity guard FAILS CLOSED** (3rd-audit fix: a $0-equity DD=$0 projection must never PASS). |
| `risk_points<=0` for a trade | sizer returns 0 for that trade (existing `compute_position_size_vol_scaled:255`); benign — a zero-risk trade has no DD contribution |

## 6. Testing (TDD — RED before GREEN)

```bash
# RED first — proves vol-scaling is actually wired (not silently 1.0):
python -m pytest tests/test_trading_app/test_account_survival.py -k "sizing or vol or d3 or parity" -q
# Regression — DD byte-identical at max_contracts=1 (no verdict change today):
python -m pytest tests/test_trading_app/test_account_survival.py -k "survival" -q
# Capital gate reconciles + new drift check passes:
python -u pipeline/check_drift.py --fast --quiet --skip-crg-advisory 2>&1 | grep -iE "survival|sizing|C11|parity"
```

**Oracle grounded by real measurement (2026-06-07, 4th audit — engine run, 10k paths, seed=7,
horizon=90, as_of=2026-06-01, profiles topstep_50k_mnq_auto + tradeify_50k).** The earlier
"~2× DD" oracle was FALSIFIED: there are TWO DD metrics with different scaling laws — see (a)/(a2).

Required assertions:
- **(a) scaling wired — LINEAR metric:** `_max_observed_rolling_drawdown` (the strict-gate input,
  `account_survival.py:894/:901`) scales **≈ ×n** in the per-trade contract count — MEASURED ×2.00
  exactly at cap=2 on BOTH profiles (1986→3973, 2445→4891). It is a pure historical replay with no
  breaches, so linearity is tight. Assert `rolling_dd(cap=n) ≈ n·rolling_dd(cap=1)` within a small
  tolerance (set tolerance from a multi-seed run at TDD time, not hardcoded). RED before, GREEN after.
- **(a2) scaling wired — NON-LINEAR metric (do NOT assert ×n):** the Monte-Carlo `p95_max_dd` scales
  only **×1.18–1.32** (1448→1704; 1666→2202) because paths breach and `break` early (`:741/:756`),
  truncating observed DD. Assert only that it INCREASES with n; never that it is linear.
- **(a3) the gate's real safety response — operational_pass_prob MONOTONE DECREASING:** MEASURED
  0.9929→0.1023 (topstep) and 0.9834→0.7057 (tradeify) at cap=2. This is the true seam-close proof:
  once the sim sizes like the engine, the gate correctly FAILS CLOSED at unsafe size. Assert
  `operational_pass_prob(cap=n)` strictly decreasing in n; pin that cap=2 flips `operational_gate_pass`
  False on the live profile (0.10 ≪ min_survival_probability). Breach driver MEASURED: daily_loss
  0→0.873 (topstep) / trailing_dd 0.017→0.294 (tradeify).
- **(b) zero verdict change at cap=1:** at `max_contracts=1`, **every** `DailyScenario` field
  (`total_pnl_dollars`, `min/max_balance_delta_dollars`, `max_open_lots`) is byte-identical to today,
  and projected DD reconciles to the known live DD ≈ $1,535.22 vs $1,800 budget.
- **(b2) scale-at-construction is NECESSARY (MEASURED):** `worst_min_delta` scaled ×2.00 (−354→−709;
  −520→−1039), and that drove the daily_loss breach 0→0.873. Pnl-only scaling would leave
  `min_balance_delta_dollars` unscaled → that breach never fires → silent under-protection. Therefore
  scaling MUST occur at `_scenario_from_trade_paths`/per-trade construction (`:462/:524-537`), pinned
  by asserting `min_balance_delta_dollars(cap=2) == 2·(cap=1)`.
- **(c) sparse fallback:** a day with NULL `atr_20` logs the fallback and uses `vol_scalar=1.0`,
  and **the gate still PASSES** (parity preserved — engine takes the same fallback). (audit M3)
- **(d) canonical median:** median comes from `paper_trader._get_median_atr_20` (mock asserts
  the call — no parallel SQL).
- **(e) look-ahead pin:** median is **trailing/point-in-time**, not full-history (a trade on
  day D must not see ATR after D).
- **(f) parity guard fail-closed ONLY on structural loss:** ATR column/table missing or query
  raises ⇒ guard returns `(False, …)`. NULL atr on a priced day does NOT trip it (covered by c).
- **(g) blast-radius pin (audit M2 + 2nd audit):** assert (i) enrichment lives in the sim-local
  helper and `_load_strategy_outcomes` return shape is UNCHANGED (its 6 other callers still get
  plain outcome dicts with no `atr_20`), AND (ii) `_load_lane_daily_pnl` output is **byte-identical
  pre/post** change (it passes no `size_model` ⇒ raw 1-contract pnl ⇒ correlation/allocation
  untouched). This is the load-bearing separation that makes "survival-private" true.
- **(h) express-funded does NOT trivially pass (3rd-audit CRITICAL):** a `is_express_funded=True`
  profile (`starting_balance=0.0`) with `size_model` set projects **non-zero** DD (sizes off
  `Portfolio.account_equity` notional, not the $0 balance). RED proof: wiring `starting_balance`
  as equity → DD=$0 → gate PASSES (the bug); GREEN: notional equity → real DD.
- **(i) equity guard fail-closed:** equity `<=0` reaching the sizer ⇒ parity guard returns
  `(False, …)` (never a $0-DD PASS).
- **(j) canonical lot cap:** the clamp delegates to `max_lots_for_xfa` (mock/assert call), not a
  re-encoded ladder; a balance below a tier threshold lowers the cap as the canonical function dictates.
- **(j2) max_open_lots steps at the mini-lot BOUNDARY, not per-contract (MEASURED):** for MNQ micros,
  `lots_for_position('MNQ', n) = 1` for n up to ≥10 (measured), so `max_open_lots` stayed **1 at cap=2**
  and `scaling_breach_prob` was **0.0000** — the `max_lots_for_xfa` breach is a DEAD branch for micro
  instruments and only bites when contracts cross a mini-lot boundary. Do NOT assert `max_open_lots`
  scales with n; assert it steps only at the `lots_for_position` boundary (4th-audit over-statement
  corrected: contract scaling ≠ lot scaling for micros).
- **(k) one-query-per-lane:** the sim-local ATR map is built with a single batched
  `daily_features` query per lane, not per-trade (perf + correctness of the as-of map).

## 7. Hard boundaries / risks (report, do not bypass)

- **No clamp lift, no arming, no executor change** in Stage 1.
- Capital / canonical path ⇒ `/check` apply-if-safe = STOP-and-audit; edits land only via the
  TDD plan with verification in the same change.
- **New cross-module coupling:** survival sim → `paper_trader._get_median_atr_20`. The drift
  check (component 5) must pin this so a future refactor of either sizer keeps gate↔engine
  parity (don't reintroduce the seam one layer down).
- **Look-ahead is the sharpest risk:** the historical median must be trailing as-of each trade
  day. Test (e) is mandatory, not optional. *Mitigation confirmed:* the canonical
  `_get_median_atr_20` is ALREADY trailing — `paper_trader.py:140` uses `trading_day < ?` (strictly
  before) over a 504-calendar-day / ~252-trading-day window. Reusing it inherits look-ahead
  protection for free; this is the core reason to delegate rather than re-encode.
- **No circular import:** `account_survival` already imports `portfolio`/`strategy_fitness`;
  `paper_trader` does not import `account_survival` — reuse of `_get_median_atr_20` is clean.
- **Blast radius (CORRECTED by 2nd audit + real data):** `_load_lane_trade_paths` has 2 direct
  callers (`:368` `_load_lane_daily_pnl`, `:568` `_load_profile_daily_scenarios`), but `:368` is
  NOT internal — it feeds correlation/allocation (`lane_correlation.py:190,195`,
  `lane_allocator.py:693`, `portfolio_correlation_audit.py:63`). The opt-in `size_model` arg
  (default None) confines scaling to the `:568` caller; all others get raw pnl. Real-data run
  confirmed even if scaling DID leak, distortion is decision-immaterial on the all-GC book
  (max rho delta 0.120 on an uncorrelated n=74 pair; high-rho pairs move 0.001–0.004; allocator
  threshold rho<0.70 never crossed). `TradePath.pnl_dollars` is summed once into
  `DailyScenario.total_pnl_dollars` (`:462`); MC samples that, no re-multiply ⇒ no double-scaling.
  `_assert_single_micro_sizing` has 1 caller (`evaluate_profile_survival:906`) + 3 pinning tests
  (`test_account_survival.py:922/939/952`) that go RED and must be rewritten.
- **`SurvivalRules.contracts_per_trade_micro` is vestigial** (set, never read by
  `simulate_survival`). Stage 1 comments it as dead; does not remove (out of scope).
- Not yet executed end-to-end: the "~2× DD" expectation is analytic until the RED test runs.
