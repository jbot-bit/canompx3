# Pre-velocity × atr_vel_regime interaction — descriptive

**Date:** 2026-04-26
**Pre-reg:** `docs/audit/hypotheses/2026-04-26-pre-velocity-vol-regime-interaction-descriptive.yaml` (committed at `c8129a47`)
**Script:** `research/audit_pre_velocity_vol_regime_interaction.py`
**Classification:** DESCRIPTIVE DIAGNOSTIC (RULE 10 carve-out — read-only)

---

## Mechanism prior (locked before run)

Chan 2013 Ch 7 stop-cascade (verbatim p.155-156, `docs/institutional/literature/chan_2013_ch7_intraday_momentum.md`): stop-density-driven momentum scales with regime volatility. Predicted sign: interaction = Δ_ExpR_E − Δ_ExpR_C **> 0** (one-sided).

## Coverage

- Cells attempted: 108
- Cells with results emitted: 108
- Cells adequately powered (each sub-cell N ≥ 30): **79**
- Cells underpowered: 29

## Headline

- Adequately-powered cells: **79**
- Cells with raw one-sided p<0.05: **3** (vs ~3.95 expected under null at one-sided α=0.05 → within Poisson noise)
- BH-FDR survivors at q=0.05, **K_within_scan=79**: **0**

### K honesty audit (added 2026-04-26 per K-rigor review)

- The prereg locked **K_global=89** (matching PR #119's adequately-powered count). This scan's stricter 4-sub-cell power floor (N≥30 in *each* of E_aligned/E_opposed/C_aligned/C_opposed) dropped 10 cells, shrinking the actual BH-FDR pool to **K=79**. The 10 dropped cells had at least one sub-cell with N<30 (mostly MGC sessions with thin Expanding+Contracting splits).
- BH-FDR was applied at K=79 (within-scan), which is standard practice but slightly less conservative than the locked K=89. **Result is null at either K.**
- **Cross-test cumulative K** is the more rigorous standard per Harvey-Liu: PR #119 (binary alignment, K=89) + this scan (alignment × regime interaction, K=79) test reframings of the same underlying pre-velocity-edge hypothesis on the same canonical data. Cumulative K = **168**. Top |t|=1.86 (max in this scan) does not approach significance at K=168 either. **Result is null at every defensible K.**

### Direction-consistency note (descriptive only — does NOT promote anything)

Among the 25 highest-|t| powered cells, 24/25 show *positive* interaction (mechanism-predicted sign per Chan stop-cascade). This is direction-suggestive but statistically null — consistent with either (a) a true small effect (~+0.10R interaction) below detection at current sample sizes, or (b) directional priors weakly visible in noise. Neither interpretation supports promotion.



### No BH-FDR survivors at K_global

**Verdict: NULL.** Pre-velocity × atr_vel_regime interaction is not detectable as a universal effect in the canonical 12 × 3 × 3 cross-section under Mode A IS. Per the prereg kill criterion, this descriptive closes the cluster.


## Per-instrument BH-FDR (K_family)

| Instrument | N_powered | p<0.05 (raw, one-sided) | BH-FDR survivors @ q=0.05 |
|---|---|---|---|
| MNQ | 29 | 1 | 0 |
| MES | 26 | 1 | 0 |
| MGC | 24 | 1 | 0 |


## Top |t| cells (descriptive only, no claim)

| Instrument | Session | O | N_E_a | N_E_o | N_C_a | N_C_o | Δ_E | Δ_C | interaction | t | p_one | df |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MES | EUROPE_FLOW | 15 | 182 | 177 | 183 | 215 | +0.138R | -0.150R | +0.288R | +1.86 | 0.0317 | 725 |
| MGC | LONDON_METALS | 30 | 80 | 86 | 85 | 95 | +0.392R | -0.049R | +0.441R | +1.84 | 0.0336 | 334 |
| MNQ | TOKYO_OPEN | 5 | 166 | 187 | 197 | 212 | +0.302R | +0.023R | +0.279R | +1.74 | 0.0414 | 725 |
| MES | EUROPE_FLOW | 5 | 179 | 180 | 192 | 206 | +0.093R | -0.106R | +0.199R | +1.36 | 0.0869 | 723 |
| MES | EUROPE_FLOW | 30 | 181 | 177 | 179 | 215 | +0.113R | -0.084R | +0.198R | +1.23 | 0.1101 | 726 |
| MNQ | TOKYO_OPEN | 30 | 177 | 173 | 198 | 206 | +0.115R | -0.087R | +0.202R | +1.18 | 0.1186 | 731 |
| MNQ | COMEX_SETTLE | 15 | 165 | 152 | 171 | 180 | +0.174R | -0.022R | +0.196R | +1.08 | 0.1408 | 653 |
| MNQ | BRISBANE_1025 | 15 | 182 | 173 | 215 | 193 | +0.196R | +0.032R | +0.164R | +1.01 | 0.1560 | 734 |
| MGC | US_DATA_830 | 5 | 68 | 84 | 78 | 95 | +0.232R | -0.011R | +0.243R | +1.01 | 0.1571 | 297 |
| MES | TOKYO_OPEN | 15 | 177 | 182 | 196 | 193 | +0.080R | -0.075R | +0.155R | +0.99 | 0.1621 | 733 |
| MGC | TOKYO_OPEN | 5 | 87 | 77 | 88 | 96 | +0.149R | -0.046R | +0.195R | +0.95 | 0.1713 | 330 |
| MNQ | SINGAPORE_OPEN | 5 | 162 | 193 | 196 | 209 | +0.140R | +0.003R | +0.137R | +0.87 | 0.1931 | 718 |
| MES | COMEX_SETTLE | 30 | 152 | 129 | 160 | 162 | +0.160R | +0.001R | +0.159R | +0.87 | 0.1932 | 574 |
| MES | TOKYO_OPEN | 30 | 185 | 169 | 192 | 192 | +0.045R | -0.091R | +0.136R | +0.84 | 0.2014 | 722 |
| MES | CME_PRECLOSE | 5 | 115 | 143 | 145 | 165 | +0.133R | -0.009R | +0.143R | +0.79 | 0.2159 | 504 |
| MGC | COMEX_SETTLE | 30 | 54 | 41 | 41 | 50 | -0.081R | -0.308R | +0.227R | +0.76 | 0.2249 | 168 |
| MES | TOKYO_OPEN | 5 | 178 | 181 | 188 | 201 | +0.121R | +0.013R | +0.108R | +0.73 | 0.2322 | 728 |
| MGC | LONDON_METALS | 15 | 91 | 77 | 86 | 95 | -0.001R | -0.172R | +0.171R | +0.73 | 0.2331 | 333 |
| MNQ | CME_PRECLOSE | 5 | 121 | 138 | 163 | 145 | +0.314R | +0.176R | +0.138R | +0.71 | 0.2378 | 529 |
| MES | US_DATA_830 | 5 | 153 | 193 | 187 | 200 | +0.174R | +0.064R | +0.109R | +0.70 | 0.2408 | 680 |
| MNQ | COMEX_SETTLE | 30 | 145 | 119 | 136 | 148 | +0.136R | +0.007R | +0.129R | +0.64 | 0.2614 | 526 |
| MES | SINGAPORE_OPEN | 15 | 174 | 177 | 210 | 185 | +0.052R | -0.042R | +0.094R | +0.61 | 0.2713 | 714 |
| MES | US_DATA_1000 | 15 | 187 | 159 | 193 | 195 | +0.204R | +0.100R | +0.104R | +0.60 | 0.2729 | 703 |
| MES | LONDON_METALS | 15 | 173 | 185 | 194 | 201 | +0.023R | -0.064R | +0.088R | +0.55 | 0.2921 | 727 |
| MGC | EUROPE_FLOW | 5 | 74 | 91 | 93 | 92 | +0.050R | -0.059R | +0.108R | +0.52 | 0.3026 | 327 |


## Underpowered cells (RULE 3.2)

- Count: 29
- Excluded from K_global pool. Listed in CSV companion.
