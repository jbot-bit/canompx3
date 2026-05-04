# FDR Methodology Specification

**Status:** ACTIVE
**Last audit:** 2026-03-30
**Authority:** RESEARCH_RULES.md (methodology), this spec (implementation details)

---

## Method: Benjamini-Hochberg FDR, Session-Stratified

### Why BH FDR (not Bonferroni, not BHY, not FDP-StepM)

BH FDR is the primary multiple testing correction for strategy validation.

**Grounding (local PDFs in `resources/`):**
- BH 1995 (Theorem 1, journal p.293 / PDF p.6): Proven FDR control for independent test statistics.
- BH 1995 (journal p.297-298 / PDF p.10-11, Fig. 1): Power advantage over Bonferroni grows with m and fraction of true signals.
- Two Million Trading Strategies (p.4-5): "FDR-BH and FDP-StepM methods do not lead to a mechanical increase in statistical thresholds" as N grows. BH has best power properties among tested MHT methods.
- Two Million (p.31-32): "The thresholds are not very dependent on N for FDR and FDP methods."
- Harvey & Liu Backtesting (journal p.22 / PDF p.11): BHY with 300 tests at 240 monthly observations and 10% annual vol raises hurdle from 4.4%/yr to 7.4%/yr.

**Why not Bonferroni:** Threshold scales linearly with K. At K=5000+, kills everything.
**Why not BHY:** More conservative than BH under dependency. BH controls FDR under PRDS (positive regression dependency on each subset) per Benjamini-Yekutieli 2001 — from training memory, not verified against local PDF (BY 2001 is not in `resources/`). We assume PRDS holds because strategies within a session share the same underlying price path and directional exposure, but this has not been formally demonstrated. The conservative K inflation (2-3x) provides an additional safety margin regardless.
**Why not FDP-StepM:** Two Million paper (p.5-6) actually prefers FDP-StepM over BH for correlated strategies. We accept BH's weaker guarantees under correlation because: (a) K inflation provides a conservative buffer, (b) FDP-StepM requires bootstrap implementation we don't have, (c) BH is adaptive to fraction of true rejections (Two Million p.33).

### Why Session-Stratified K

K is computed per session (orb_label), not globally.

**Grounding:**
- Sessions are pre-specified by exchange events (CME settlement, NYSE open), not by data outcomes.
- Efron separate-class model: stratified FDR is valid and more powerful when group structure is pre-specified on logical grounds.
- RESEARCH_RULES.md: "instrument/family K for promotion decisions."
- Empirical validation (2026-03-30 audit): 111 strategies would be killed by global K=79,162 that survive session K. These are valid strata victims — sessions have genuinely different market microstructure.

**Implementation:** `strategy_validator.py` lines 1248-1278. Pool is built from `experimental_strategies WHERE is_canonical = TRUE AND instrument IN (ACTIVE_ORB_INSTRUMENTS)`, grouped by `orb_label`. BH applied per session with `total_tests = len(pool)`.

### What Counts Toward K

- Every canonical strategy with a non-null p-value in the session pool.
- `is_canonical` deduplication collapses same-trade-day-hash strategies into one representative (highest filter specificity). Aliases don't count.
- Dead instruments (M2K, MBT, M6E, SIL) are excluded from the pool via `instrument IN (ACTIVE_ORB_INSTRUMENTS)`.
- E1 and E2 both count toward K even when they share trade-day-hashes (909 shared hashes, r=0.96, MAD=0.047R). This inflates K conservatively. All 488 validated strategies are E2.
- Stop multiplier variants (S0.75, S1.0) count separately — different stop distances produce distinct P&L series and p-values (Harvey & Liu 2015). Confirmed: 500/500 pairs share trade_day_hash (same entry days, different P&L).
- RR target variants count separately — same entry days, different take-profit. Confirmed: 500/500 pairs share hash (Aronson p.282: parameter optimization).

**Conservative direction:** K is inflated ~2-3x beyond the number of truly independent tests. This means the BH correction is STRICTER than necessary. Fewer false positives, possibly killing real edge. Safe direction for capital allocation.

---

## FST / DSR: Informational Only (NOT a Gate)

### What FST Does

False Strategy Theorem (Bailey & de Prado 2014): computes SR0 = expected maximum Sharpe ratio from K independent noise trials. If observed SR < SR0, the strategy is explainable by chance.

### Why It Kills Everything

**Empirical (2026-03-30 audit):**
- Best validated Sharpe: 2.954 (MNQ CME_PRECLOSE)
- FST SR0 at K=3634 (per-session): 3.648
- FST SR0 at K=480 (reduced): 3.092
- **0/488 pass FST at ANY per-session K**
- At K=228, SR0=2.93. Best SR=2.954 barely exceeds SR0 in point estimate, but DSR probability (which accounts for sampling variance via skewness/kurtosis) remains below 0.95 — the strategy does not pass DSR even at this K.

### Why We Don't Gate On It

1. **N_eff uncertainty:** DSR requires N_eff (effective independent trials). True N_eff is unknown without ONC algorithm (Lopez de Prado). With V[SR]=0.047: N_eff=5 → 4 pass, N_eff=10 → 0 pass. The result is hypersensitive to a parameter we cannot estimate. (`dsr.py` lines 27-35 document this.)

2. **Different null hypothesis:** BH tests H0: mean(pnl_r) = 0 via t-test. FST tests H0: SR is consistent with max-SR-of-K-noise-trials. These are different questions. The t-test asks "is the mean nonzero?" FST asks "could the BEST Sharpe among K be this high by luck?" BH handles selection bias via FDR adjustment; FST handles it by inflating the benchmark.

3. **Return structure:** Our fixed-RR binary returns produce platykurtic distributions (kurtosis_excess ≈ -1.7). Sharpe ratio is a poor discriminator for bimodal {-1, +RR} outcomes. The t-test on mean return is more appropriate. Empirically verified (2026-03-30 query): Sharpe-based p-value using Lo (2002) standard error formula (from training memory — Lo 2002 "The Statistics of Sharpe Ratios" not in `resources/`) agrees with t-test p-value on all 488 strategies (0 divergent at α=0.05, max |delta| = 0.035, corr(delta, kurtosis) = -0.26).

### What DSR Scores Mean

- `dsr_score` in `validated_setups`: probability that observed SR exceeds noise expectation.
- Average: 0.016. Max: 0.787. Only 4/488 above 0.50. None above 0.95.
- These scores are INFORMATIONAL — monitor for trends, do not gate on them.

---

## Honest Counts (2026-03-30, verified empirically)

| Level | Count | What It Measures |
|-------|-------|-----------------|
| Validated strategies | 488 | Raw count including all parameter variants (RR × stop × filter) |
| Unique trade-day-hashes | 171 | Distinct entry patterns (after canonical dedup) |
| Family hashes | 172 | Validated_setups grouping |
| Edge families (non-purged) | 88 | 84 MNQ + 3 MES + 1 MGC |
| **Independent bets (J>0.7 clustering)** | **7** | **5 MNQ + 1 MES + 1 MGC** (verified 2026-03-30, replaces prior "14" claim) |
| Independent clusters (J<0.3) | 1 | All MNQ streams share >46% of trade days — too strict for ORB |

**Prior "14 independent bets" claim (portfolio_reconstruction_audit) was FALSE.** Empirical J>0.7 clustering on 84 non-purged MNQ families gives 5 clusters (sizes: 68, 7, 4, 3, 2). The mega-cluster of 68 families shares >70% of trade days — these are parameter variants of one core signal.

**Use 7 independent bets for DD budget and portfolio sizing. Use 88 edge families for diversification reporting. Never cite 488 as "independent edges."**

---

## Known K Inflation Sources (Conservative)

1. **E1/E2 duplication:** 909 shared hashes. r=0.96, MAD=0.047R. Same signal counted twice. All validated are E2.
2. **Parameter variants:** 6 RR × 2 stop = up to 12 strategies per unique trade stream. These are parameter optimization (Aronson p.282).
3. **Correlated ORB_G filters:** ORB_G4/G5/G6/G8 filter subsets of the same ORB size distribution. Different trade days but Jaccard 0.46-1.0.

All inflate K in the safe direction (overly strict). No false positives leak through. Some real edge may be killed.

---

## Audit

Run: `python scripts/tools/audit_fdr_integrity.py`
13 checks covering BH correctness, K drift, ghost strategies, gate consistency, hash accuracy, parameter redundancy, Jaccard independence, session vs global K, cross-instrument K, stop/RR independence, honest counts.

**Critical:** The audit filters on `ACTIVE_ORB_INSTRUMENTS` to match validator behavior (`strategy_validator.py:1255`).

## discovery_k Freeze

`discovery_k` and `discovery_date` are frozen on first write. Subsequent validator runs update `fdr_significant` and `fdr_adjusted_p` (which change as the canonical pool grows) but preserve the original K and date for audit trail integrity. Implemented 2026-03-30.
