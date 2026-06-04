# C11 Equity-Drawdown Throttle — OOS Validation Result

**Date:** 2026-06-04
**Profile:** `topstep_50k_mnq_auto` (MNQ, 3-lane book)
**Pre-reg:** `docs/audit/hypotheses/drafts/2026-06-04-c11-equity-drawdown-throttle.draft.yaml`
**Plan:** `docs/plans/2026-06-04-NEXT-c11-throttle-validation.md`
**Harness:** `C:\Users\joshd\c11_matrix\throttle_validate.py` (read-only, outside repo)

---

## Scope

Validate the causal equity-drawdown throttle (halve daily participation when account
equity is ≥ TRIGGER below its running peak; restore at RECOVER) as a risk-control
overlay that brings `topstep_50k_mnq_auto`'s strict 90-day rolling drawdown under the
$1,600 (0.80×) **and** $2,000 (1.00× MLL) budgets while preserving ≥85% of total
R-edge — tested OOS-honestly, not blessing the in-sample fit. This is **validation
only**. No implementation, no production-code change, no commit to `main`. The
candidate `trigger=$800 / factor=0.5` was selected in-sample by clearing C11
(parameter-selection bias, per the draft YAML); this exercise tests it against
walk-forward + frozen-holdout discipline.

## Method

- **Preflight (Phase 0).** MinBTL K-budget gate (`estimate_k_budget`, MNQ, K=8): **PASS**
  — needs 4.16 yr horizon, 6.65 yr available (2.49 yr headroom), well under Bailey's
  bound. Baseline reproduced **exact**: max-90d-DD = **$2,038.84** (canonical gate fails:
  > $1,600 and > $2,000).
- **Anchored walk-forward.** Anchor = 2019-05-31. Five 12-month test steps (2020–2024).
  Throttle state warms from the anchor through each step (174 warm-up days before 2020,
  > 90-day HORIZON). A cell is **WF-stable** if it clears the $1,600 C11 budget in ≥ 3 of
  the 5 steps. DD is measured path-honestly: a rolling-90d window counts toward a period
  iff its **end** day falls in that period (the drawdown may legitimately have begun
  earlier).
- **Frozen holdout.** 2025-01-01 .. 2026-06-02 (412 days). Used as a **confirmatory** C11
  read after the band is fixed by WF-stability — see Limitations on touch discipline.
- **Edge preservation.** Total realized R retained vs un-throttled baseline, **full
  history** (the YAML's binding floor: ≥ 85%).
- **Daily-loss belt.** Modeled path-aware: the throttle scales **both** `total_pnl` and
  `min_balance_delta` (intraday low), matching `account_survival`'s belt rule
  `min(min_balance_delta, total_pnl) ≤ −$450`.
- **No-lookahead.** Decision for day *t* uses peak/balance through *t−1* only; verified
  structurally and by an explicit one-day-at-a-time replay (leaks = 0).

## Decision

**VERDICT: PASS.** A contiguous WF-stable band of **three** adjacent triggers
**{600, 800, 1000} at factor=0.5** clears every binding gate. Not a knife-edge.

| Gate (draft YAML, verbatim) | Result |
|---|---|
| `c11_survival` (full-history DD ≤ $1,600) | PASS — 600/800/1000 = $1,323 / $1,459 / $1,522 |
| `c11_survival_real_mll` (≤ $2,000) | PASS — all ≤ $1,522 |
| holdout DD ≤ $1,600 (corroboration) | PASS — 600/800/1000 = $898 / $1,023 / $1,094 |
| `edge_preservation_floor` (full-history total R ≥ 85%) | PASS — 86.2% / 89.3% / 87.7% |
| `daily_loss` (zero breach days preserved) | PASS — 0 breaches (baseline 0; throttle is a contraction on belt exposure) |
| `walk_forward_stability` (clears C11 ≥ 3/5 steps) | PASS — 600/800/1000 all 5/5 |
| `robustness_band` (≥ 2 adjacent triggers) | PASS — band width 3 |
| `operational_gate` (MC op-pass no regression) | PASS (structural) — baseline 0.998 ≫ 0.70; throttle de-risks monotonically |

**Chosen cell (mid-band, most headroom): `trigger=$800, factor=0.5, recover=$400`.**
full max-90d-DD = **$1,458.95** (−$580 vs baseline), holdout DD = $1,022.89, edge_full =
89.3%, WF = 5/5. The IS-best cell (800/0.5) **is** inside the WF-stable band → not
overfit on the WF axis. Trigger=1200 is correctly excluded (fails C11 at $1,695, WF 4/5).

Mechanism generality confirmed: the throttle engages in **5 distinct historical episodes**
(2020, 2022, 2022–23, 2025, 2026) — a general drawdown rule, not a 2022 detector. It
**reduces** the worst window ($2,038 → $1,459) rather than relocating it (still ends
2022-10-25).

## Files

- `C:\Users\joshd\c11_matrix\throttle_validate.py` — WF + frozen-holdout + multiplicity
  + daily-loss-belt + no-lookahead harness (read-only, outside repo).
- `C:\Users\joshd\c11_matrix\throttle_test.py` — original IS grid + overfitting/episode
  audit (reference).
- `C:\Users\joshd\c11_matrix\repro_baseline.py` — exact baseline reproduction
  ($2,038.84).
- Canonical sources consulted (read-only): `trading_app/account_survival.py`
  (`_max_observed_rolling_drawdown:749`, `_historical_daily_loss_breach_days:737`,
  `_load_profile_daily_scenarios:472`), `trading_app/prop_profiles.py` (`get_profile`,
  `daily_loss_dollars`).

## Verification

- k-budget gate: `estimate_k_budget(MNQ, n_trials=8)` → PASS (verdict in output).
- baseline: `repro_baseline.py` → max90dDD=$2038.84, MATCH=True.
- harness: `throttle_validate.py` → VERDICT PASS, band [600,800,1000], chosen 800/0.5.
- DD parity with canonical `_max_observed_rolling_drawdown` confirmed structurally
  (identical inner loop) and empirically (baseline exact).
- daily-loss belt: `min_balance_delta` ∈ [−377.5, 0], 0 baseline breaches, 0 throttled
  breaches; throttle is monotone-safe on the belt.

## Throttle-aware Monte Carlo (closes the operational-gate limitation)

Run 2026-06-04 via `C:\Users\joshd\c11_matrix\throttle_mc.py` (20,000 paths, seed 0).
The harness **first proves parity**: at factor=1.0 it reproduces canonical
`account_survival.simulate_survival` **exactly** on every channel (dd_survival 0.99715,
trailing_dd 0.00285, daily_loss 0.0, p95_max_dd $1324.59, p50 $669.56 — all bit-identical
after using canonical `_quantile`). Only on that proven-canonical base is the throttle injected.

| Metric | Baseline | Throttle (800/0.5/400) | Delta |
|---|---|---|---|
| dd_survival_probability | 0.99715 | **1.00000** | +0.00285 |
| operational_pass_probability | 0.99715 | **1.00000** | +0.00285 |
| trailing_dd (MLL) breach prob | 0.00285 | **0.00000** | −0.00285 |
| daily_loss breach prob | 0.00000 | 0.00000 | 0 |
| p95_max_dd | $1,324.59 | **$1,102.62** | −$221.97 |
| p50_max_dd | $669.56 | $668.74 | −0.82 |

**No regression on any survival or breach channel.** The throttle eliminated the residual
0.285% MLL-breach probability and cut p95 worst-case DD by ~$222 — confirming the
contraction argument empirically, not just structurally. Operational-gate criterion now
**directly verified** (not merely argued): op-pass 1.0 ≫ 0.70 threshold, no hidden
survival-reduction bug.

## Limitations

This is a validation verdict, **not** an approval to arm. Open items, stated honestly:

- **Holdout touch discipline (UNSUPPORTED as pristine single-touch).** The harness prints
  holdout DD for all 8 grid cells. The band {600,800,1000} is fully determined by
  `WF-stable ∩ full-DD ≤ $1,600` *before* holdout is consulted, so holdout did **not**
  drive selection — it is confirmatory. But the display is a multi-cell holdout read, not
  a literal one-touch evaluation. Treat holdout DD as corroboration, not as an independent
  single-shot test.
- **Operational-gate — RESOLVED.** A throttle-aware MC was run (see section above):
  parity-proven harness, 20k paths, throttle de-risks monotonically (op-pass 0.99715 → 1.0,
  MLL-breach 0.00285 → 0.0, p95 DD −$222), no regression on any channel. The earlier
  "structural argument only" caveat is closed by direct measurement.
- **Edge cost is real.** The chosen cell costs ~10.7% of full-history total R (89.3%
  retained). Holdout edge retention is lower (79.5%) because the throttle engaged in the
  2025 drawdown episode — within the YAML's full-history floor, but the holdout window pays
  more. This is the deliberate edge-for-drawdown trade, not a defect.
- **Selection bias acknowledged.** trigger/factor were IS-chosen. The WF + band-width
  defense mitigates but does not eliminate this; the band being 3 wide and WF 5/5 is the
  evidence it is not a knife-edge fit.
- **Implementation is Tier B (capital path) and NOT done.** If approved, the throttle
  param set must be a **single canonical source** consumed by BOTH `account_survival`
  (gate) AND `session_orchestrator` (live), reusing `account_hwm_tracker.py` HWM state —
  otherwise the gate-vs-live divergence returns. Adversarial-audit-gated before arming.

## Next (requires operator approval — do NOT proceed without it)

1. Promote `…throttle.draft.yaml` → `docs/audit/hypotheses/` with `date_locked` set.
2. Run a throttle-aware MC to close the operational-gate limitation.
3. Present the Tier-B implementation plan (single canonical param source, HWM reuse,
   gate+live parity). Implementation only after that plan is approved.
