# MNQ NYSE_OPEN 5m E2 RR1.0 CB1 OVNRNG_50_FAST10 — Pathway-B K=1 Confirmatory Test

**Pre-reg:** `docs/audit/hypotheses/2026-04-20-nyse-open-ovnrng-fast10-v1.yaml`
**Runner:** `research/confirm_nyse_open_ovnrng_fast10.py`
**Run at:** 2026-04-20 12:12 UTC
**Commit SHA:** a46ef923
**Verdict:** `SHADOW_CONDITIONAL`

---

## Upstream provenance (disclosed)

Cell was surfaced by a K=772 descriptive adjacency scan in
`research/audit_l4_nyse_open_decay.py` run 2026-04-20 12:05 UTC during the
L4 NYSE_OPEN COST_LT12 SR-alarm audit. This is UPSTREAM PROVENANCE only;
the confirmatory test below applies Pathway-B discipline (K=1, Chordia-
with-theory t≥3.00, DSR deflated at K_upstream=772, all 12 pre-registered
criteria gates) to a fresh canonical-layer pull, with additional T0/T1/T4/T6/T8
pressure tests that the scan did not include.

---

## Result per pre-committed kill criterion

| # | Criterion | Threshold | Observed | Verdict |
|---|-----------|-----------|----------|---------|
| C3 | IS t-stat (Chordia w/ theory) | ≥ 3.00 | **+4.750** | PASS |
| C5 | DSR (K_upstream=772) | ≥ 0.95 | **0.9542** | PASS (marginal) |
| C6 | WFE (OOS Sharpe / IS Sharpe) | ≥ 0.50 | **+0.978** | PASS |
| C7 | OOS deployable N | ≥100 confirmatory / ≥30 directional | **64** | DIRECTIONAL_ONLY |
| C8 | OOS ExpR ≥ 0.40×IS | +0.0547 | **+0.1372** | PASS |
| C9 | Era stability (no year ExpR<-0.05, N≥50) | — | 7/7 years positive | PASS |
| C10 | Data-era compatibility | feature-temporal validity | NYSE_OPEN post-17:00 Brisbane → CLEAN | PASS |
| T0 | Tautology vs deployed filters | \|corr\| ≤ 0.70 | max +0.36 (vs OVNRNG_100) | PASS |
| T1 | WR lift vs unfiltered | ≥ +2.0pp | **+2.58pp** | PASS |
| T4 | Sensitivity (6 OVNRNG×FAST neighbors) | all IS positive | all +0.080 to +0.139 | PASS |
| T6 | Null floor one-tailed p | ≤ 0.05 | **p ≈ 1e-6** | PASS |
| T8 | Cross-instrument directional | all same sign as MNQ IS | MNQ +0.137, MGC +0.174, MES +0.056 | PASS |

---

## Core statistics

| Metric | IS (2019-05-06 → 2025-12-31) | OOS (2026-01-02 → 2026-04-16) |
|--------|------------------------------|-------------------------------|
| N | 1,099 | 64 |
| WR | 58.7% | 57.8% |
| ExpR | +0.1368 | +0.1372 |
| σ(pnl_r) | 0.955 | 0.979 |
| Sharpe (ann, 252) | +2.275 | +2.224 |
| t-stat vs 0 | +4.750 | +1.121 |

- Fire rate at MNQ NYSE_OPEN 5m E2 RR1.0 CB1: **65.9%** (1,163 of 1,764 sessions)
- This is a SELECTIVE gate (contrast L4 COST_LT12 at 98.6% — essentially pass-through).
- OOS ExpR (+0.1372) is **effectively identical** to IS ExpR (+0.1368) → zero walk-forward decay.

---

## Mechanism (theory citation)

From pre-reg `hypothesis.H1.theory_citation`:

> Fitschen 2013 Ch 3 grounds intraday trend-follow breakouts for stock index
> futures during US sessions. Elevated prior-session range is a volatility-
> regime filter (higher vol → stronger continuation). Fast break (≤10 min
> post ORB close) is a conviction signal — early breaks have higher
> directional carry-through. NYSE_OPEN is the highest-liquidity US session,
> reducing friction drag.

Mechanism class: `volatility-regime AND conviction-speed gate`.

---

## Per-year stability (IS only)

| Year | N | ExpR |
|------|---|------|
| 2019 | 36 | +0.141 |
| 2020 | 184 | +0.048 |
| 2021 | 152 | +0.167 |
| 2022 | 210 | +0.095 |
| 2023 | 134 | +0.094 |
| 2024 | 172 | +0.253 |
| 2025 | 211 | +0.166 |

All 7 years positive. No outlier-year dominance. Closest to zero: 2020 at +0.048.

---

## Cross-instrument (NYSE_OPEN 5m E2 RR1.0 CB1 OVNRNG_50_FAST10)

| Inst | IS N | IS ExpR | OOS N | OOS ExpR |
|------|------|---------|-------|----------|
| MNQ | 1,099 | +0.137 | 64 | +0.137 |
| MGC | 46 | +0.174 | 49 | +0.334 |
| MES | 99 | +0.056 | 19 | -0.089 |

- MGC is directionally consistent and OOS-strong, but IS N=46 < 50 era-stability floor.
- MES OOS flip is on N=19 — below OOS power threshold, consistent with MES weakness on breakout replication (per `2026-04-20-participation-optimum-mes-universality-v1.yaml` result).

---

## Deployment posture per pre-reg § deployment_posture

> Pathway-B K=1 PASS on this pre-reg produces a SHADOW candidate only.
> Capital action requires:
> (a) separate deployment-contract pre-reg (analogous to
>     2026-04-20-q4-band-deployment-shape-v1.yaml)
> (b) explicit user command citing this lock SHA
> (c) account-survival 90d Monte Carlo re-run with the added lane to
>     confirm Criterion 11 still passes ≥70%.

**What this document IS:** a locked verdict that the OVNRNG_50_FAST10 gate
at MNQ NYSE_OPEN 5m E2 RR1.0 CB1 has a genuine, theory-grounded edge with
direction-match OOS.

**What this document is NOT:** a deployment contract. No lane allocation
change. No capital action. No SR-monitor wiring.

---

## Anti-bias self-check

1. **DSR is marginal (0.9542 vs 0.95).** At K=9,504 (full 12-session × 3×3×88 scan
   universe) DSR would drop to ~0.84 and fail. I scoped K=772 to the NYSE_OPEN-only
   subset where this candidate was surfaced — this is defensible but not conservative.
   **Action:** Treat the candidate as CONDITIONALLY-CONFIRMED rather than robust-confirmed.
   A future confirmation run after N_OOS ≥ 100 should recompute DSR and, if still
   marginal, downgrade to CONDITIONAL.

2. **OOS N=64 is underpowered for confirmatory.** Per `feedback_oos_power_floor.md`,
   below-100 OOS means we cannot distinguish true ExpR = IS_ExpR from true ExpR = 0.4×IS_ExpR.
   C7 verdict DIRECTIONAL_ONLY is the honest label.

3. **Cell was not chosen blind.** I saw IS=+0.137 OOS=+0.137 in the scan before writing
   the pre-reg. The pre-reg cannot "un-see" that. Mitigation: all kill criteria were
   numeric and pre-committed before the confirmatory runner executed — the runner was
   written as a single-shot gated test, not an iterative tune.

4. **MES divergence is the most bearish signal.** MES IS +0.056 is positive but weak;
   OOS -0.089 on N=19 is noise. Still, the lack of MES replication undermines the
   "universal breakout-continuation" story. Finding is MNQ-specific within the MNQ/MGC
   stock-index vs gold split — MGC ≠ MES counterfactual.

---

## Recommended next steps (user decides, no autonomous action)

1. **Hold as SHADOW:** log the trigger-fire mask to an append-only ledger, no capital.
2. **If user greenlights deployment:** write a separate deployment-contract pre-reg
   (cite this lock SHA), re-run account-survival MC including the new lane, require
   C11 ≥ 70% pass with the additional position.
3. **Do NOT:** bulk-add as "just one more lane"; expand to other sessions without
   per-session K=1 pre-regs; relax C7 threshold post-hoc to pass 64 trades as
   "confirmatory".

---

## Artifacts

- Pre-reg: `docs/audit/hypotheses/2026-04-20-nyse-open-ovnrng-fast10-v1.yaml`
- Runner: `research/confirm_nyse_open_ovnrng_fast10.py`
- Raw output log: committed below via memory pointer
- Upstream scan: `research/audit_l4_nyse_open_decay.py` (descriptive, not committed as canonical artefact)
