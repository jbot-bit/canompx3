# C11 Solution Matrix — Attribution, Candidate Universe, and the Drawdown-Throttle Finding

**Date:** 2026-06-04
**Author:** Claude (read-only analysis session)
**Profile under test:** `topstep_50k_mnq_auto` (MNQ, 3 daily lanes)
**Branch / worktree:** `c11-attribution-analysis` @ `de1f9089` in `C:\Users\joshd\canompx3-c11-attribution`
**DB:** shared canonical `C:\Users\joshd\canompx3\gold.db` (read-only)

---

## Scope

This audit answers: **why does `topstep_50k_mnq_auto` fail Criterion 11 (strict 90-day
drawdown), and what is the smallest edge-preserving fix?** It covers (1) baseline
reproduction, (2) gate-correctness verification against the official Topstep rule,
(3) attribution of the failing drawdown window, (4) the full candidate-fix universe,
(5) edge recomputed in R-multiples, and (6) a causal drawdown-throttle test with an
overfitting audit.

**Out of scope (deliberately):** any production-code edit, config change, commit,
or live action. This is analysis only. No 2026-holdout tuning (the gate uses full
history 2019–2026; the failing window is 2022, not 2026 — verified).

---

## Decision / Verdict

**CONTINUE — but pivot the fix class.** Caps, stop-tightening, lane-removal, and
lane de-correlation are **refuted** for this failure mode by direct evidence. The
C11 failure is a **shared adverse-regime grind (Jul–Oct 2022)**, not a fat-tail,
not a single-lane defect, not a correlation spike. The only mechanism that addresses
it is a **drawdown-responsive participation throttle**, which clears C11 against both
the self-imposed $1,600 and the real $2,000 budget with ~11% edge loss and survives
an overfitting audit (engages in 5 distinct historical episodes, not just 2022,
no-lookahead).

**This is a finding to bring to a pre-registered implementation decision, not an
approval to ship.** See Limitations.

---

## 1. Baseline reproduction (gate truth)

Canonical path (`account_survival._load_profile_daily_scenarios` +
`_max_observed_rolling_drawdown`), no monkeypatch:

| quantity | value |
|---|---|
| max 90d DD | **$2,038.84** (matches documented baseline exactly) |
| daily-loss breach days | 0 |
| source window | 2019-05-31 → 2026-06-04 (2049 days) |
| budget 0.80× ($1,600) | **FAIL** |
| budget 1.00× ($2,000, real MLL) | **FAIL by $38.84** |
| MC operational gate | PASS (per prior runs, ~99.7%) |

C11 fails **only** on the strict-DD-magnitude leg. The failure margin against the
*real* Topstep MLL is 1.9%.

## 2. Gate correctness — verified against the official Topstep rule

Two of my initial hypotheses (the gate ignores the daily-loss halt; the gate
measures the wrong DD object) were **tested and falsified**:

- **Daily-loss-halt modeling gap — FALSIFIED.** Zero days in 7 years close worse
  than the −$450 daily-loss limit, so flooring daily PnL at the halt changes the
  rolling DD by **$0.00**.
- **Wrong-DD-object (intraday) — FALSIFIED.** Verified live against
  help.topstep.com: Topstep's Maximum Loss Limit **trails on end-of-day balance**
  (rises with EOD growth, never down, locks at start balance). The gate's
  close-to-close (EOD) measurement is therefore the **correct** object. (Nuance:
  *liquidation* is monitored in real time on net P&L incl. unrealized — so the gate,
  using EOD troughs, is if anything mildly *lenient* on the intraday liquidation
  leg, never harsh. This argues the $2,038 is a floor, not an inflation.)

**Conclusion: the C11 strict-DD gate measures the right quantity.** The remaining
open calibration question is the *statistic* ("worst single 90-day window over 7
years") and the *threshold* ($1,600 vs $2,000), reported below but not unilaterally
resolved — the $1,600 is an operator risk knob.

## 3. Attribution — what drives the $2,038 window

The worst 90-day window is **Jul 27 – Oct 25, 2022** (the 2022 bear grind), a
**78-day peak→trough leg** with no single catastrophic day (worst day −$247):

| lane | N | W | L | WR | PnL in leg |
|---|---|---|---|---|---|
| COMEX_SETTLE | 26 | 7 | 19 | 27% | −$337 |
| US_DATA_1000 | 36 | 9 | 27 | 25% | −$1,255 |
| TOKYO_OPEN | 20 | 5 | 15 | 25% | −$313 |
| **book** | 82 | 21 | 61 | **26%** | **−$1,905** |

Every lane's win-rate collapses from ~43% lifetime to ~25% in this window. It is a
**shared regime**, not a lane defect. US_DATA carries 66% of the loss because it has
the most trades (highest N), not because it is the worst lane.

## 4. Edge in R-multiples (Step 3 correctness fix)

The dollar-proxy ExpR misranked the lanes. Recomputed in R (mean `pnl_r`):

| lane | N | **ExpR (R)** | WR | Sharpe (R) | ExpR$ proxy (misleading) |
|---|---|---|---|---|---|
| COMEX_SETTLE | 539 | 0.106 | 41.2% | 1.65 | $5.98 |
| US_DATA_1000 | 828 | 0.182 | 43.1% | 2.70 | $19.32 |
| TOKYO_OPEN | 460 | 0.179 | 44.1% | 2.72 | $9.57 |

**US_DATA and TOKYO are tied-best in true edge** (0.182 vs 0.179 R). The $-proxy's
2× US-over-TOKYO gap was a position-size artifact (US has wider stops → larger $
risk per R). **"US is the best lane" is a $-illusion** — it is tied-best and
highest-volume, which is why it dominates the absolute-$ drawdown.

## 5. Candidate-fix universe (complete, before narrowing)

| Fix class | Tested? | Evidence | Edge-preserving? | Knowable pre-entry? | Lookahead/contam? | Status |
|---|---|---|---|---|---|---|
| Uniform ORB cap | YES | banked cliff table; non-monotonic (80pt → $2,068 WORSE, ≤60pt → $1,525) | NO (60pt strips $27→$18 ExpR band) | yes | no | **REFUTED** (kills winning variance; non-monotonic) |
| Per-lane / US-only cap | partial | cap is execution filter (`account_survival.py:370`) | NO at clearing values | yes | no | **REFUTED** (same mechanism) |
| Stop multiplier tighten | inferred | losses are normal-sized full-RR losses in a grind, not fat-risk | NO | yes | no | **REFUTED** (tighter stop loses *more* often in chop) |
| Lane removal | YES (attribution) | grind is portfolio-wide; removing US drops tied-best lane + shifts `common_start` window | NO | n/a | no | **REFUTED** |
| Lane de-correlation | YES | daily rho ≈ 0 lifetime AND in 2022 (−0.20/+0.40n9/+0.15); vindicates 2026-04-20 audit | n/a | n/a | no | **REFUTED** (lanes already independent daily) |
| Regime gating (feature-based) | NO | `daily_features.atr_vel_regime` + double-break exist | potentially | yes | risk: fitting a 2022 detector | **NEEDS MATRIX** (overfit risk; not pursued — throttle dominates) |
| Session-specific cap | implied | subsumed by per-lane cap result | NO | yes | no | **REFUTED** |
| Dynamic cap by ORB %ile | NO | percentile of *future* distribution = lookahead unless trailing | maybe | only if trailing | **risk** | **INVALID if non-trailing** |
| Fixed micro sizing | n/a | already 1-contract micro per lane | n/a | yes | no | **N/A (already minimal)** |
| Volatility sizing (Carver) | NO | Carver Ch9-10 grounded; would scale by ATR | yes | yes | no | **UNTESTED** (Stage-2 roadmap; heavier) |
| **Drawdown throttle** | **YES** | clears C11, ~11% edge loss, 5 episodes, no-lookahead | **YES (89% kept)** | **yes (own equity)** | **no** | **CANDIDATE — survives audit** |
| Daily-loss throttle | partial | 0 days breach −$450, so inert here | n/a | yes | no | **INERT** (no binding days) |
| Allocation sequencing | NO | reorders lane start dates | no edge effect | n/a | no | **UNTESTED (low EV)** |
| Do-nothing / shadow-monitor | n/a | accept C11 fail, paper-trade | n/a | n/a | no | **FALLBACK option** |

## 6. The drawdown-throttle matrix (the fix that works)

Causal rule: walk EOD equity; when drawdown-from-running-peak ≥ `trigger`, scale the
**next** day's participation by `factor` until equity recovers within `recover` of
peak. Decision for day *t* uses equity through *t−1* only (no-lookahead, verified
structurally).

| trigger | factor | max90d DD | $1,600 | $2,000 | total $ | edge loss | active days throttled |
|---|---|---|---|---|---|---|---|
| baseline | — | $2,038.84 | FAIL | FAIL | $23,412 | 0% | 0% |
| $600 | 0.5 | $1,323 | PASS | PASS | $20,173 | 13.8% | 32.6% |
| **$800** | **0.5** | **$1,459** | **PASS** | **PASS** | **$20,917** | **10.7%** | 29.7% |
| $1,000 | 0.5 | $1,522 | PASS | PASS | $20,544 | 12.3% | 24.8% |
| $1,200 | 0.5 | $1,695 | FAIL | PASS | $21,396 | 8.6% | 15.9% |
| $800 | 0.0 (full halt) | $844 | PASS | PASS | $3,979 | **83%** | 78.5% |

**Findings:**
- **Half-size (`factor=0.5`) dominates full-halt (`factor=0.0`).** Full halt clears
  DD trivially but craters return 83% — it also sits out the *recovery*. Half-size
  keeps ~89% of edge.
- **Robust band, not a knife-edge.** `factor=0.5` clears $1,600 across triggers
  $600/$800/$1,000; only $1,200 misses $1,600 (still clears $2,000). Recommended
  cell: **trigger=$800, factor=0.5** ($1,459 DD, 10.7% edge loss).

### Overfitting audit (the make-or-break test)

For the recommended cell (trigger=$800, factor=0.5):
- **Engages in 5 distinct drawdown episodes across 6 years**: 2020-12→2021-07,
  2022-03→04, 2022-08→2023-06, 2025-03→08, 2026-01→02. It responds to the
  *condition* (any sustained drawdown), **not** the 2022 *date* — the signature of
  a general rule, not a fitted detector.
- **Does not displace the problem:** worst window after throttle is still the 2022
  window, *attenuated* $2,038 → $1,459 (not pushed into a new window).
- **No-lookahead:** scale chosen from peak/balance computed *before* today's PnL.

## 7. Adversarial interpretation

1. **Root cause of C11 failure:** a **shared adverse-regime grind** (2022), where
   all three lanes' edge simultaneously turned negative (ExpR +0.18 → −0.20, WR 43%
   → 25%) over 78 days. **Not** ORB size, **not** stop distance, **not** lane
   correlation (lanes stay rho≈0), **not** sizing, **not** one lane.
2. **Highest-EV fix after edge preservation:** **drawdown throttle, trigger ≈ $800,
   factor 0.5** — clears both budgets, keeps ~89% of edge, robust across a trigger
   band, general across 5 episodes.
3. **Fake-greens (clear C11 but destroy value or cheat):** ORB cap ≤60pt (clears
   DD by deleting the $27-ExpR winning band — flagged in plan); full-halt throttle
   factor=0.0 (clears DD, −83% return); any regime *detector* fit to the 2022 dates
   (would be a parameter rescue / lookahead).
4. **Untested, must NOT be dismissed:** Carver volatility-targeting position sizing
   (Ch9-10) — a principled size-by-ATR rule could subsume the throttle and is the
   Stage-2 roadmap; it was not tested here (heavier, separate work).
5. **Final status: CONTINUE.** The throttle is the recommended path; pre-register it
   before implementation.

---

## Files read (canonical)

- `trading_app/account_survival.py` (`_load_lane_trade_paths:320`, `_load_profile_daily_scenarios:472`, `_max_observed_rolling_drawdown:749`, `evaluate_profile_survival:767`, `_historical_daily_loss_breach_days:737`, `simulate_survival` daily-loss halt `:669`)
- `trading_app/prop_profiles.py` (`topstep_50k_mnq_auto:681`, `get_profile_lane_definitions:1191`, `load_allocation_lanes`)
- `trading_app/live/session_orchestrator.py` (daily-loss breaker `:598`)
- `docs/institutional/literature/carver_2015_ch11_portfolios.md`, `carver_2015_volatility_targeting_position_sizing.md`
- `resources/prop-firm-official-rules.md`; live: help.topstep.com MLL article (WebFetch)
- `docs/audit/2026-04-11-criterion-11-f1-false-alarm.md` (referenced)

## Commands run

All via `C:/Users/joshd/canompx3/.venv/Scripts/python.exe` from the worktree, against
shared `gold.db`. Analysis scripts (read-only, outside repo) in `C:\Users\joshd\c11_matrix\`:
`repro_baseline.py`, `test_daily_loss_halt_in_dd.py`, `locate_worst_window.py`,
`reconcile_and_rmultiple.py`, `throttle_test.py`.

## Verification

- Baseline $2,038.84 reproduced exactly via canonical path (no re-encoding of the DD walk).
- Topstep MLL rule verified against the live official source, not memory.
- R-multiple recompute mirrors `account_survival.py:358-389` filter logic identically.
- Throttle no-lookahead verified structurally; overfitting audit = 5 episodes across 6 years.

## Limitations

- **UNSUPPORTED until pre-registered + adversarial-audit-gated:** the throttle is an
  in-sample fit over 2019–2026. Although it survives the multi-episode overfitting
  audit, the recommended `trigger=$800/factor=0.5` was *selected* by clearing C11 —
  parameter-selection bias remains. A pre-registered, walk-forward / OOS-honest
  validation is required before any implementation. **Do not ship from this doc.**
- The throttle was tested as a post-hoc transform of the canonical EOD scenario
  series. A faithful implementation must enforce the same scale **live** in
  `SessionOrchestrator` AND model it in the gate (`account_survival`), or the
  gate-vs-live divergence (memory-documented) reappears. The fix must change a
  **single source** both consume.
- The $1,600 budget is an operator-chosen risk knob (not a Topstep rule). Results are
  reported against both $1,600 and the real $2,000 MLL; the threshold choice is the
  operator's, not resolved here.
- n per lane in the 2022 window is 20–36 — small. The attribution is directional, not
  a powered per-lane estimate.
- Carver volatility-targeting sizing was **not** tested; it may be a superior
  principled alternative to the throttle and is the Stage-2 roadmap item.

## Smallest implementation plan (when/if approved — NOT executed here)

1. Pre-register the throttle hypothesis (trigger/factor/recover) with OOS-honest
   acceptance criteria; estimate k-budget (Bailey MinBTL — only a handful of cells
   tested, well under the 300-trial bound).
2. Add a single canonical drawdown-throttle parameter set consumed by **both**
   `account_survival` (gate) and `SessionOrchestrator` (live) — reuse the existing
   account-equity/HWM tracker (`account_hwm_tracker.py`) rather than new state.
3. Walk-forward validate; confirm 2026 holdout untouched.
4. Adversarial-audit gate (capital path) before arming.
