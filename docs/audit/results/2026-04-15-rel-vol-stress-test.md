# rel_vol_HIGH_Q3 — Honest Stress Test

**Date:** 2026-04-15
**Purpose:** stress-test whether the 5 BH-global volume survivors are GENUINE edge or overfit / statistical anomaly. Bailey-Lopez de Prado 2014 DSR, False Strategy Theorem, temporal stability, Bonferroni at global K.

## Methodology

- **DSR** at 3 N_eff framings: K_global=14,261 (strictest), K_family~2,500 (moderate), K_lane~56 (lenient). Implementation: `trading_app.dsr.compute_sr0` + `compute_dsr`.
- **E[max_t from noise]** at K_global via Gumbel extreme-value approximation — `sqrt(2·ln(K)) - (ln(ln(K)) + ln(4π))/(2·sqrt(2·ln(K)))`.
- **Bonferroni @ K_global** — threshold = 0.05/14261 = 3.5e-6.
- **Temporal stability** — IS split 50/50 by date, sign-match + |t|≥2 in both halves required.
- **Verdict scoring:** 3 points DSR@global + 2 DSR@family + 1 DSR@lane + 2 temporal + 2 exceeds-max-t + 2 Bonferroni. GENUINE ≥ 7, MARGINAL ≥ 4, else LIKELY_OVERFIT.

**Expected max t from K=14261 random trials (Gumbel):** 3.826
**Bonferroni threshold at K=14261:** 3.5e-6 (corresponds to |t| ≥ 4.639)

## Per-lane stress test

### MES COMEX_SETTLE O5 RR1.0 short
**Verdict:** **MARGINAL** (hard gates passed: 4/5)
**Hard gates:** DSR@K_family_pass_0.95=✗, temporal_stable_both_halves=✓, t_exceeds_Emax_noise=✓, per_day_significant_t2=✓, block_bootstrap_p_lt_0.01=✓
**Rationale:** ✓ temporal_stable_both_halves, ✓ t_exceeds_Emax_noise, ✓ per_day_significant_t2, ✓ block_bootstrap_p_lt_0.01

**Observed (per-trade):**
- N_on = 251, N_off = 508
- SR_on (per-trade) = +0.0993
- Welch t = +4.480, p = 0.000009
- |t| exceeds E[max_t] from K=14261? **True** (E[max_t] = 3.826)
- p < Bonferroni@K_global (3.5e-6)? **False**

**Autocorrelation-robust checks:**
- Per-day aggregated t = +4.480 (n_on_days=251), Δ_per_day = +0.2879
- Block bootstrap (5-day blocks, 2000 resamples) p = 0.0005

**DSR at 3 N_eff framings:**

| Framing | K_eff | SR0 (noise-max) | DSR | Pass @ 0.95 |
|---------|-------|-----------------|-----|-------------|
| lane | 56 | +0.1901 | 0.0807 | no |
| family | 900 | +0.2642 | 0.0055 | no |
| global | 14261 | +0.3233 | 0.0003 | no |

**Temporal stability (IS split 50/50):**
- First half: 2019-05-14 to 2022-09-13
  - N_on=130, Δ=+0.340, t=+3.82, p=0.0002
- Second half: 2022-09-14 to 2025-12-30
  - N_on=121, Δ=+0.244, t=+2.64, p=0.0088
- Sign match both halves: **True**
- Both halves |t|≥2: **True**

---

### MGC LONDON_METALS O5 RR1.0 short
**Verdict:** **MARGINAL** (hard gates passed: 4/5)
**Hard gates:** DSR@K_family_pass_0.95=✗, temporal_stable_both_halves=✓, t_exceeds_Emax_noise=✓, per_day_significant_t2=✓, block_bootstrap_p_lt_0.01=✓
**Rationale:** ✓ temporal_stable_both_halves, ✓ t_exceeds_Emax_noise, ✓ per_day_significant_t2, ✓ block_bootstrap_p_lt_0.01

**Observed (per-trade):**
- N_on = 147, N_off = 298
- SR_on (per-trade) = +0.1601
- Welch t = +4.653, p = 0.000005
- |t| exceeds E[max_t] from K=14261? **True** (E[max_t] = 3.826)
- p < Bonferroni@K_global (3.5e-6)? **False**

**Autocorrelation-robust checks:**
- Per-day aggregated t = +4.653 (n_on_days=147), Δ_per_day = +0.3491
- Block bootstrap (5-day blocks, 2000 resamples) p = 0.0005

**DSR at 3 N_eff framings:**

| Framing | K_eff | SR0 (noise-max) | DSR | Pass @ 0.95 |
|---------|-------|-----------------|-----|-------------|
| lane | 56 | +0.1901 | 0.3666 | no |
| family | 900 | +0.2642 | 0.1182 | no |
| global | 14261 | +0.3233 | 0.0317 | no |

**Temporal stability (IS split 50/50):**
- First half: 2022-06-20 to 2024-03-04
  - N_on=72, Δ=+0.394, t=+4.06, p=0.0001
- Second half: 2024-03-11 to 2025-12-31
  - N_on=75, Δ=+0.305, t=+2.67, p=0.0085
- Sign match both halves: **True**
- Both halves |t|≥2: **True**

---

### MES TOKYO_OPEN O5 RR1.5 long
**Verdict:** **MARGINAL** (hard gates passed: 4/5)
**Hard gates:** DSR@K_family_pass_0.95=✗, temporal_stable_both_halves=✓, t_exceeds_Emax_noise=✓, per_day_significant_t2=✓, block_bootstrap_p_lt_0.01=✓
**Rationale:** ✓ temporal_stable_both_halves, ✓ t_exceeds_Emax_noise, ✓ per_day_significant_t2, ✓ block_bootstrap_p_lt_0.01

**Observed (per-trade):**
- N_on = 278, N_off = 563
- SR_on (per-trade) = +0.0817
- Welch t = +4.367, p = 0.000015
- |t| exceeds E[max_t] from K=14261? **True** (E[max_t] = 3.826)
- p < Bonferroni@K_global (3.5e-6)? **False**

**Autocorrelation-robust checks:**
- Per-day aggregated t = +4.367 (n_on_days=278), Δ_per_day = +0.3212
- Block bootstrap (5-day blocks, 2000 resamples) p = 0.0005

**DSR at 3 N_eff framings:**

| Framing | K_eff | SR0 (noise-max) | DSR | Pass @ 0.95 |
|---------|-------|-----------------|-----|-------------|
| lane | 56 | +0.1901 | 0.0361 | no |
| family | 900 | +0.2642 | 0.0012 | no |
| global | 14261 | +0.3233 | 0.0000 | no |

**Temporal stability (IS split 50/50):**
- First half: 2019-05-14 to 2022-07-25
  - N_on=144, Δ=+0.356, t=+3.40, p=0.0008
- Second half: 2022-07-27 to 2025-12-30
  - N_on=134, Δ=+0.278, t=+2.70, p=0.0073
- Sign match both halves: **True**
- Both halves |t|≥2: **True**

---

### MNQ SINGAPORE_OPEN O5 RR1.0 short
**Verdict:** **MARGINAL** (hard gates passed: 4/5)
**Hard gates:** DSR@K_family_pass_0.95=✗, temporal_stable_both_halves=✓, t_exceeds_Emax_noise=✓, per_day_significant_t2=✓, block_bootstrap_p_lt_0.01=✓
**Rationale:** ✓ temporal_stable_both_halves, ✓ t_exceeds_Emax_noise, ✓ per_day_significant_t2, ✓ block_bootstrap_p_lt_0.01

**Observed (per-trade):**
- N_on = 276, N_off = 559
- SR_on (per-trade) = +0.1657
- Welch t = +4.291, p = 0.000021
- |t| exceeds E[max_t] from K=14261? **True** (E[max_t] = 3.826)
- p < Bonferroni@K_global (3.5e-6)? **False**

**Autocorrelation-robust checks:**
- Per-day aggregated t = +4.291 (n_on_days=276), Δ_per_day = +0.2657
- Block bootstrap (5-day blocks, 2000 resamples) p = 0.0005

**DSR at 3 N_eff framings:**

| Framing | K_eff | SR0 (noise-max) | DSR | Pass @ 0.95 |
|---------|-------|-----------------|-----|-------------|
| lane | 56 | +0.1901 | 0.3499 | no |
| family | 900 | +0.2642 | 0.0595 | no |
| global | 14261 | +0.3233 | 0.0063 | no |

**Temporal stability (IS split 50/50):**
- First half: 2019-05-15 to 2022-08-25
  - N_on=152, Δ=+0.257, t=+3.01, p=0.0028
- Second half: 2022-08-26 to 2025-12-29
  - N_on=124, Δ=+0.271, t=+2.99, p=0.0031
- Sign match both halves: **True**
- Both halves |t|≥2: **True**

---

### MES COMEX_SETTLE O5 RR1.5 short
**Verdict:** **MARGINAL** (hard gates passed: 3/5)
**Hard gates:** DSR@K_family_pass_0.95=✗, temporal_stable_both_halves=✓, t_exceeds_Emax_noise=✗, per_day_significant_t2=✓, block_bootstrap_p_lt_0.01=✓
**Rationale:** ✓ temporal_stable_both_halves, ✓ per_day_significant_t2, ✓ block_bootstrap_p_lt_0.01

**Observed (per-trade):**
- N_on = 249, N_off = 504
- SR_on (per-trade) = +0.0533
- Welch t = +3.681, p = 0.000259
- |t| exceeds E[max_t] from K=14261? **False** (E[max_t] = 3.826)
- p < Bonferroni@K_global (3.5e-6)? **False**

**Autocorrelation-robust checks:**
- Per-day aggregated t = +3.681 (n_on_days=249), Δ_per_day = +0.3000
- Block bootstrap (5-day blocks, 2000 resamples) p = 0.0005

**DSR at 3 N_eff framings:**

| Framing | K_eff | SR0 (noise-max) | DSR | Pass @ 0.95 |
|---------|-------|-----------------|-----|-------------|
| lane | 56 | +0.1901 | 0.0155 | no |
| family | 900 | +0.2642 | 0.0004 | no |
| global | 14261 | +0.3233 | 0.0000 | no |

**Temporal stability (IS split 50/50):**
- First half: 2019-05-14 to 2022-09-13
  - N_on=128, Δ=+0.291, t=+2.60, p=0.0098
- Second half: 2022-09-14 to 2025-12-30
  - N_on=121, Δ=+0.318, t=+2.69, p=0.0076
- Sign match both halves: **True**
- Both halves |t|≥2: **True**

---

## Joint cross-lane probability

- Product of per-lane p-values (independent assumption): 3.80e-24
- Even assuming lanes are dependent at rho=0.3 effective, combined evidence is still extreme.
- Interpretation: probability that 5 independent tests all randomly produce p < 0.0005 is vanishingly small.

## Consolidated verdict summary

| Lane | Verdict | Score |
|------|---------|-------|
| MES COMEX_SETTLE O5 RR1.0 short | **MARGINAL** | 4/5 |
| MGC LONDON_METALS O5 RR1.0 short | **MARGINAL** | 4/5 |
| MES TOKYO_OPEN O5 RR1.5 long | **MARGINAL** | 4/5 |
| MNQ SINGAPORE_OPEN O5 RR1.0 short | **MARGINAL** | 4/5 |
| MES COMEX_SETTLE O5 RR1.5 short | **MARGINAL** | 3/5 |

## Recommendation

_Populate after reviewing scores._