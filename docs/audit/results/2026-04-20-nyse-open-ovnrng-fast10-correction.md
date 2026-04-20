# CORRECTION — MNQ NYSE_OPEN 5m E2 RR1.0 CB1 OVNRNG_50_FAST10

**Parent pre-reg:** `docs/audit/hypotheses/2026-04-20-nyse-open-ovnrng-fast10-v1.yaml` (locked SHA `a46ef923`)
**Parent results:** `docs/audit/results/2026-04-20-nyse-open-ovnrng-fast10-confirm.md`
**Parent PR:** #44
**Correction authored:** 2026-04-20 ~12:50 UTC
**Revised verdict:** `MISCLASSIFIED`

---

## Two material corrections to PR #44's verdict

### Correction 1 — SCALE ARTIFACT (simple arithmetic on canonical data)

MNQ tripled in price over the IS window (~7,884 → ~25,084). Median overnight_range went from 38 points to 167 points (4.4×). `OVNRNG_50` uses an ABSOLUTE-points threshold, so its meaning changed era-over-era.

**Fire rate by year** on the MNQ NYSE_OPEN 5m E2 RR1.0 CB1 universe:

| Year | MNQ avg price | Median ovn_range | OVNRNG_50 fire | OVN_50_FAST10 fire |
|------|---------------|-----------------|----------------|-------------------|
| 2019 | 7,884 | 38 | 23.7% | **21.3%** |
| 2020 | 10,298 | 84 | 83.8% | 72.7% |
| 2021 | 14,469 | 64 | 70.0% | 60.1% |
| 2022 | 12,781 | 96 | 92.2% | 82.0% |
| 2023 | 14,247 | 57 | 59.5% | 53.2% |
| 2024 | 19,234 | 72 | 76.7% | 66.9% |
| 2025 | 22,639 | 110 | 96.0% | 83.4% |
| **2026 OOS** | 25,084 | 167 | 98.6% | **90.1%** |

Fire rate went from 21% (selective) to 90% (near pass-through). Classic scale-artifact signature.

**Per-year lift over unfiltered baseline:**

| Year | Unfilt ExpR | Filt ExpR | Lift | Fire% |
|------|-------------|-----------|------|-------|
| 2019 | +0.007 | +0.141 | **+0.134** | 21.3% |
| 2020 | +0.019 | +0.048 | +0.028 | 72.7% |
| 2021 | +0.064 | +0.167 | +0.103 | 60.1% |
| 2022 | +0.110 | +0.095 | **−0.015** | 82.0% |
| 2023 | +0.006 | +0.094 | +0.088 | 53.2% |
| 2024 | +0.198 | +0.253 | +0.055 | 66.9% |
| 2025 | +0.134 | +0.166 | +0.032 | 83.4% |
| **2026** | +0.136 | +0.137 | **+0.001** | **90.1%** |

The celebrated "OOS +0.137R dir_match" is **essentially 100% the unfiltered baseline**. The filter added **+0.001R** in 2026. In 2022 it HURT (−0.015R).

**ATR-normalized OVNRNG quintiles** (on IS, shows the real shape):

| Bin | ovn/atr | N | WR | ExpR |
|-----|---------|---|----|----|
| Q1 lo | 0.17 | 339 | 56.6% | +0.085 |
| Q2 | 0.24 | 338 | 55.9% | +0.078 |
| Q3 | 0.31 | 338 | 58.0% | +0.116 |
| Q4 | 0.40 | 338 | 57.4% | +0.107 |
| **Q5 hi** | **0.68** | **339** | **52.5%** | **+0.017** |

The edge lives in Q3–Q4 (mid-sweet-spot), not "above threshold." Q5 (biggest overnight moves) HURTS. An absolute-threshold filter can't express this shape.

**Era-dependence** (late-vs-early IS half, Welch two-sample t-test): t=+1.96, p=0.0505 — borderline.

### Correction 2 — DSR was NOT computed with Bailey's canonical formula

**Admission.** The DSR=0.9542 at K_upstream=772 I reported in the pre-reg was NOT Bailey-LdP 2014 Eq. 2. I used:
- A simplified `E[max] ≈ √(2·ln(K) − ln(ln(K)) − ln(4π))` approximation
- Divided by a Mertens-style single-strategy SR standard error
- That is NOT the canonical formula

**Canonical formula** (from `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md:42`):

```
DSR ≡ Φ( (ŜR − ŜR_0) · √(T−1) / √(1 − γ̂₃·ŜR + (γ̂₄−1)/4 · ŜR²) )
ŜR_0 = √V[{ŜR_n}] · ((1−γ)·Φ⁻¹[1 − 1/N] + γ·Φ⁻¹[1 − 1/(Ne)])
```

Where `V[{ŜR_n}]` is the variance of SR across the discovery-trial cells (not a single-strategy standard error), `Φ⁻¹` is the inverse-normal quantile function, `γ` ≈ 0.5772.

**Implementation validated** against Bailey's own numerical example (paper page 9-10): SR_ann=2.5, T=1250, V=0.5, skew=−3, kurt=10, N=100. Paper reports DSR ≈ 0.9004; my implementation returns DSR = 0.9004. Match.

**Applied to OVNRNG_50_FAST10 IS** (T=1099, SR_nonann=+0.143, skew=−0.35, kurt_Pearson=+1.126, V[{ŜR_n}]=0.0237 across 772 NYSE_OPEN scan cells):

| N (indep trials) | E[max] z | SR_0 (non-ann) | DSR | Verdict |
|---|---------|---------------|------|---------|
| 46 | 2.244 | 0.346 | 0.0000 | FAIL |
| 100 | 2.531 | 0.390 | 0.0000 | FAIL |
| 772 | 3.181 | 0.490 | 0.0000 | FAIL |
| 9,504 | 3.848 | 0.593 | 0.0000 | FAIL |
| 50,000 | 4.238 | 0.653 | 0.0000 | FAIL |

Canonical DSR ≈ 0 at every N from 46 upward. Observed SR_nonann=0.143 is far below the N=46 deflated threshold of 0.346 (≈ 5.49 annualized).

**Caveat on V[{ŜR_n}].** The 772 scan cells are correlated (same trade pool, overlapping filters). Bailey Appendix A.3 Eq. 9 suggests adjusting via implied-independent-trials: `N̂ = ρ̂ + (1−ρ̂)·M`. A strong positive ρ̂ ≈ 0.8 would give N̂ ≈ 155. Even at N=155, SR_0 ≈ 0.413 non-ann (≈ 6.55 annualized) — still far above observed. The DSR verdict does not flip for any plausible correlation adjustment.

---

## Revised classification (answer to user's prompt)

### VALID / ALIVE / DEAD / MISCLASSIFIED?

**MISCLASSIFIED.** The pre-reg's 12 gates passed by their letter, but:
- The OOS lift over the unfiltered baseline was **+0.001R** (essentially the baseline itself)
- Fire rate grew from 21% → 90% purely from price inflation
- ATR-normalized Q5 (biggest moves) HURTS performance — filter is mis-shaped
- Canonical Bailey DSR ≈ 0 at every N, versus the 0.9542 my approximation returned

The selected strategy's actual edge is ~0 once selection-bias-adjusted. The edge the pre-reg "confirmed" was the unfiltered MNQ NYSE_OPEN 5m RR1.0 CB1 baseline lane, not the OVNRNG_50_FAST10 gate.

### Tunnel-vision framings

| Framing | Status in PR #44 | Honest assessment |
|---------|------------------|-------------------|
| **Standalone** | not tested | Not fairly tested — never isolated the filter's signal from the baseline lane's edge |
| **Filter** (what PR #44 did) | FALSIFIED by the re-audit | 2026 OOS lift +0.001R. Filter adds nothing. |
| **Conditioner / sizing tilt** | NOT TESTED | ATR-normalized Q3-Q4 sweet-spot is a legitimate size-tilt candidate |
| **Allocator input** | NOT TESTED | OVN percentile as regime-router across instruments |
| **Confluence** | NOT TESTED | OVN × other features untested |

**Prematurely ruled out:** the unfiltered MNQ NYSE_OPEN 5m E2 RR1.0 CB1 baseline. It IS a real edge (IS t=3.50, OOS +0.136, 6/7 years positive) that doesn't need a selective filter in the current era.

### Highest-EV honest opportunity (user's prompt)

**Best opportunity:** cross-session ATR-normalized OVNRNG sweet-spot filter. Target bins Q3–Q4 (ovn/atr ∈ ~[0.24, 0.40]) NOT Q5. Run as a fresh Pathway-B pre-reg scoped across all 12 sessions × 3 apertures × 3 RRs (~324 cells) so DSR is honest. Lift in the Q3-Q4 band vs Q5 is ~+0.09R — a real selection signal, not a scale artifact.

**Biggest blocker:** other 5 currently-DEPLOYED lanes probably have similar scale-artifact filters. `COST_LT12` at L4 fires 98.6% (already known pass-through). `ORB_G5`/`ORB_G8` at MNQ sessions were grid-calibrated when MNQ was cheaper. Each deserves a fire-rate-by-year + lift-by-year audit before continuing to treat the 6-lane portfolio as "diversified."

**Biggest miss:** project-level methodology gap — absolute-threshold filters were promoted without scale-stability audits. Should be required BEFORE pre-reg lock for any filter with raw-points/fixed-dollar/fixed-notional thresholds on non-stationary-priced instruments (MNQ, MES, MGC).

**Next best test:** a 7-lane deployed-portfolio scale-stability audit — for each deployed filter compute fire-rate-by-year, lift-by-year, and an ATR/percentile-normalized alternative. Estimated cost: ~1h read-only. Est EV: uncovering which deployed lanes' "edge" is silently the unfiltered baseline + a near-pass-through gate.

---

## Deployment posture

**PR #44 should NOT merge.** Recommended actions:
1. Amend PR #44 body to point to this correction file
2. Do not merge (or close with this file as the close reason)
3. Write a fresh Pathway-B pre-reg for the ATR-normalized sweet-spot variant if the opportunity is worth pursuing
4. Run the 7-lane deployed-portfolio scale-stability audit before any new deployment decision

**PR #46 (q4-band shadow monitor) is separate** and stands on its own — its scope is infrastructure for the q4-band pre-reg (lock SHA aa9999a7, PR #43), not OVNRNG_50_FAST10.

**Parent pre-reg locked, not amended.** Pre-reg discipline does not allow retroactive amendment. This correction file supersedes the parent results MD's verdict without modifying the pre-reg itself.

---

## Artefacts (this PR)

- `docs/audit/results/2026-04-20-nyse-open-ovnrng-fast10-correction.md` (this file)
- `research/audit_ovnrng50_scale_artifact.py` (scale + lift + ATR + era check)
- `research/audit_ovnrng50_canonical_dsr.py` (canonical Bailey Eq. 2, sanity-checked against paper page 9-10)
- Memory feedback rule: `memory/feedback_absolute_threshold_scale_audit.md`
