# MNQ endogenous-RR via IS-grid optimization — v1

**Pre-reg:** `docs/audit/hypotheses/2026-04-27-mnq-mfe-distribution-endogenous-rr-v1.yaml`
**Runner:** `research/mnq_mfe_distribution_endogenous_rr_v1.py`
**Scratch policy:** realized-eod (Stage 5 fix landed; rebuild verified at >=99.5% population per `pipeline/check_drift.py::check_orb_outcomes_scratch_pnl`).

**Scope:** does the IS-grid-optimal RR target produce a measurably higher IS ExpR than fixed RR=1.5 on the 4 sessions whose v1-high-RR cells sign-flipped under realized-EOD MTM (NYSE_OPEN, US_DATA_1000, CME_PRECLOSE, COMEX_SETTLE) x (5m, 15m)?

**Outcome (verdict):** 0 of 7 cells pass H1 (BH-FDR q < 0.05, t >= +3.0, optimal_RR != baseline). H2 (descriptive): 6/7 cells have optimal RR > baseline 1.5; 0/7 equal baseline.

## Per-cell table

| Session | Apt | N_IS | N_OOS | Optimal RR | ExpR opt | ExpR @ RR=1.5 | t_IS | p_one | q_BH | OOS opt | OOS @ 1.5 | H1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CME_PRECLOSE | 5 | 1643 | 69 | 4.0 | +0.1309 | +0.0982 | +1.54 | 0.0621 | 0.0869 | -0.3226 | -0.1648 | fail |
| NYSE_OPEN | 5 | 1719 | 73 | 4.0 | +0.1726 | +0.0988 | +2.35 | 0.0095 | 0.0332 | +0.1925 | +0.0997 | fail |
| US_DATA_1000 | 5 | 1718 | 72 | 4.0 | +0.1504 | +0.0985 | +1.56 | 0.0593 | 0.0869 | +0.0602 | -0.0053 | fail |
| CME_PRECLOSE | 15 | 1131 | 53 | 4.0 | +0.0881 | +0.0691 | +1.76 | 0.0390 | 0.0869 | +0.1133 | +0.1001 | fail |
| COMEX_SETTLE | 15 | 1657 | 69 | 1.0 | +0.0382 | +0.0265 | +0.72 | 0.2343 | 0.2343 | -0.0136 | -0.0811 | fail |
| NYSE_OPEN | 15 | 1715 | 73 | 4.0 | +0.1533 | +0.1027 | +2.40 | 0.0082 | 0.0332 | +0.2142 | +0.2384 | fail |
| US_DATA_1000 | 15 | 1717 | 72 | 4.0 | +0.1577 | +0.1203 | +1.44 | 0.0749 | 0.0874 | +0.1349 | +0.1922 | fail |

## ExpR by RR per cell (IS only)

| Session | Apt | RR=1.0 | RR=1.5 | RR=2.0 | RR=2.5 | RR=3.0 | RR=4.0 |
|---|---|---:|---:|---:|---:|---:|---:|
| CME_PRECLOSE | 5 | +0.0760 | +0.0982 | +0.1194 | +0.1247 | +0.1230 | +0.1309 **opt** |
| NYSE_OPEN | 5 | +0.0777 | +0.0988 | +0.1080 | +0.1263 | +0.1396 | +0.1726 **opt** |
| US_DATA_1000 | 5 | +0.0864 | +0.0985 | +0.1110 | +0.1363 | +0.1495 | +0.1504 **opt** |
| CME_PRECLOSE | 15 | +0.0562 | +0.0691 | +0.0717 | +0.0757 | +0.0860 | +0.0881 **opt** |
| COMEX_SETTLE | 15 | +0.0382 **opt** | +0.0265 | +0.0107 | -0.0028 | -0.0045 | -0.0225 |
| NYSE_OPEN | 15 | +0.0895 | +0.1027 | +0.1319 | +0.1462 | +0.1466 | +0.1533 **opt** |
| US_DATA_1000 | 15 | +0.0921 | +0.1203 | +0.1308 | +0.1353 | +0.1304 | +0.1577 **opt** |


## Verdict and follow-on

**Resolved verdict: KILL** — endogenous-RR refuted at locked Chordia-with-theory threshold (t >= +3.0). 0 of 7 cells pass H1.

### Honest reading of the negative result

H2 (descriptive) is striking: **6 of 7 cells have IS-optimal RR = 4.0** (the highest grid value), 1 cell has optimal = 1.0 (COMEX_SETTLE 15m, mean-reverting profile). This descriptive pattern is consistent with the Stage 6 sign-flip discovery: post-confirmed-break drift is positive on these sessions and goes far beyond RR=1.5.

But the **lift over baseline RR=1.5 is small relative to noise**. On the strongest cells (NYSE_OPEN 5m and 15m), t-stat is +2.35 and +2.40 — passes BH-FDR q<0.05 but fails Chordia t>=+3.0. The locked criterion is conjunctive (BH + Chordia); H1 fails.

OOS evidence is mixed:
- NYSE_OPEN 5m, US_DATA_1000 5m: optimal RR > baseline OOS (consistent with IS direction)
- NYSE_OPEN 15m, US_DATA_1000 15m, CME_PRECLOSE 5m: baseline > optimal OOS (sign-flipped on holdout)

The OOS sign-flip on the strongest IS cells (NYSE_OPEN 15m: opt OOS +0.21 vs baseline +0.24; US_DATA_1000 15m: opt +0.13 vs baseline +0.19) is a yellow flag — institutional rigor says do not chase the IS-optimal RR without OOS confirmation. Per `feedback_oos_power_floor.md`, the OOS sample N=27-73 trades has limited power; the OOS sign-flip is consistent with both noise AND a real overfit. Distinguishing requires more OOS data — not available under Mode A discipline.

### What this DOES support

- The Stage-5 canonical fix is correctly producing realized-EOD MTM. The data is now usable for RR-optimization research that was institutionally impossible pre-fix.
- The descriptive direction of H2 (high-RR optimal on sign-flipped sessions) is consistent with the mechanism_priors.md hypothesis about post-break trend continuation.
- The KILL is a **rigor verdict**, not a "the hypothesis is wrong" verdict. Endogenous RR may still be right; we just lack power to prove it at strict significance with current sample sizes.

### Follow-on routing

- **No deployment.** No allocator change.
- **No Pathway B K=1.** The K=1 framework requires a specific cell with theory-grounded optimal RR; H1 didn't qualify any cell.
- **Future work (deferred):**
  - Vol-adaptive RR (per-trade RR conditional on `garch_atr_ratio`) — separate Stage 2 pre-reg.
  - Wider grid (sub-integer RR like 1.25, 1.75, etc.) — separate pre-reg with new MinBTL accounting.
  - Cross-instrument replication (MES, MGC) — independent pre-regs.
  - Direct CPCV walk-forward on optimal-RR-per-fold — under Lopez de Prado 2020 framework.

## Mechanism note

Stage 6 found scratch_mean_R = +0.9955 on MNQ NYSE_OPEN 15m RR=4.0 — empirical evidence that
post-confirmed-break drift is positive on these sessions. Carver Ch 9-10 grounds the
hypothesis that optimal RR is a function of signal strength; the result above shows whether
that varies enough across sessions to warrant per-lane RR tuning.

## Reproduction

```bash
python research/mnq_mfe_distribution_endogenous_rr_v1.py
```

## Limitations

- Discretized RR grid: only 6 candidate RR values. True optimum could lie between grid points.
- Cell-level RR optimization: this is in-sample selection of a single parameter per cell. Bailey
  et al 2013 MinBTL bound 0.46 years (well within available 7-year IS window) — not at risk of overfit.
- Direction-pooled: longs and shorts averaged. Per-direction analysis deferred to follow-up.
- Vol-adaptive (per-trade RR conditional on regime): out of scope; deferred to Stage-2 pre-reg.
- OOS is descriptive only. Mode A holdout is sacred; do not tune against OOS.

## Cross-references

- Pre-reg: `docs/audit/hypotheses/2026-04-27-mnq-mfe-distribution-endogenous-rr-v1.yaml`
- Stage 5 fix: `trading_app/outcome_builder.py` commit 68ee35f8
- Stage 6 sign-flip table: `docs/audit/results/2026-04-27-canonical-scratch-fix-downstream-impact.md`
- Criterion 13 scratch policy: `docs/institutional/pre_registered_criteria.md`
- Mechanism prior: `docs/institutional/mechanism_priors.md` § 11.5
