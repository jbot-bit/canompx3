# Phase 2.4 — Cross-session momentum SGP→EUROPE_FLOW portfolio re-eval (Mode A)

**Date:** 2026-04-19
**Branch:** `research/campaign-2026-04-19-phase-2`
**Parent plan:** `docs/plans/2026-04-19-max-ev-extraction-campaign-plan.md` § 2.4
**Supersedes framing of:** `docs/audit/deploy_readiness/2026-04-15-sgp-momentum-deploy-readiness.md` § 6 A/B/C/D (Mode B grandfathered — framed "SGP RR1.5 is +44% ExpR better per trade"; Mode A correction below)
**Pre-registration:** `docs/audit/hypotheses/2026-04-13-cross-session-sgp-europe-flow.yaml` (K=3, MinBTL PASS)
**Script:** `research/phase_2_4_cross_session_momentum_mode_a.py`
**Outputs:** `research/output/phase_2_4_cross_session_momentum_mode_a_lanes.csv`, `…_options.csv`
**Tests:** `tests/test_research/test_phase_2_4_cross_session_momentum_mode_a.py`

## Executive verdict

Under strict Mode A IS (`trading_day < 2026-01-01` per `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`):

1. **SGP per-trade quality advantage over ORB_G5 collapses** from the Mode B +0.075 ExpR gap at RR1.5 to a Mode A +0.0043 gap. Prior "+44% ExpR better" framing was inflated by trailing-12mo window that included 2026 Q1 sacred-OOS data.
2. **Canonical deployment gate (`trading_app/lane_correlation.py`) rho=1.0000 on shared-fire pnl_r correctly blocks Option D (parallel deploy)** — both lanes take the identical trade on 486/848 days. Confirmed under Mode A.
3. **49-day SGP-unique pocket is net NEGATIVE** (ExpR=−0.0330) under Mode A — disqualifies the "70-day additive pocket" suggestion from prior doc § 8.
4. **Option B (swap)** is no longer meaningfully better per trade — the rationale for swap dissolves under correct holdout.
5. **Option C (composite ORB_G5 ∧ SGP_TAKE)** remains the only path with a non-trivial per-trade ExpR lift (+0.0159 at RR1.5, +0.0463 at RR2.0 vs ORB_G5 alone). Still gated on a new `CompositeFilter` class + validator re-run + new pre-reg — not unblockable here.

## Scope

Locked trade surface:

- Instrument: MNQ
- Session: EUROPE_FLOW
- ORB minutes: 5
- Entry model: E2
- Confirm bars: 1
- Direction: long (per pre-reg and validated_setups rows)
- RR targets: {1.0, 1.5, 2.0}
- IS window: 2019-05-08 → 2025-12-24 (6.65y clean MNQ real-micro per Phase 3c rebuild)

## Per-lane Mode A results

| Lane | Filter | RR | N | ExpR | Sharpe (ann) | WR | Years + | sd |
|------|--------|----|----:|-----:|------------:|-----:|--------:|----:|
| L1_A | ORB_G5 | 1.0 | 773 | +0.0353 | +0.42 | 57.7% | 6/7 | 0.89 |
| L1_B | CROSS_SGP_MOMENTUM | 1.0 | 535 | +0.0502 | +0.50 | 59.8% | 5/7 | 0.87 |
| L1_A | ORB_G5 | 1.5 | 773 | +0.0769 | +0.72 | 48.0% | 6/7 | 1.12 |
| L1_B | CROSS_SGP_MOMENTUM | 1.5 | 535 | +0.0812 | +0.64 | 49.4% | 5/7 | 1.10 |
| L1_A | ORB_G5 | 2.0 | 773 | +0.0735 | +0.58 | 39.8% | 4/7 | 1.32 |
| L1_B | CROSS_SGP_MOMENTUM | 2.0 | 535 | +0.1122 | +0.75 | 42.2% | 6/7 | 1.31 |

Note RR2.0 C9 concern: ORB_G5 is positive in only 4/7 years — borderline era-stability fail (pre_registered_criteria § C9). SGP at RR2.0 is 6/7, cleaner.

## Same-day overlap (break-day universe: N=848)

| Cell | Count | RR1.5 ExpR |
|------|------:|-----------:|
| ORB_G5 ∧ SGP_TAKE (both) | 486 | +0.0928 |
| ORB_G5 ∧ ¬SGP_TAKE (G5 only) | 287 | +0.0501 |
| ¬ORB_G5 ∧ SGP_TAKE (SGP only) | 49 | **−0.0330** |
| ¬ORB_G5 ∧ ¬SGP_TAKE (neither) | 26 | — |

On intersection days the two lanes take the identical trade (same ORB break, same RR, same direction) — `np.allclose(pnl_g5, pnl_sgp) == True` on shared-fire days. Canonical Pearson rho on intersection pnl_r = **1.0000** by construction → `lane_correlation.py` gate correctly BLOCKS Option D at all three RRs.

Fire-mask (0/1 indicator) rho across the 848-day universe is −0.0145 — set-membership is near-independent, but this is NOT the deployment gate's metric.

## Option scoring (per RR)

| RR | A (ORB_G5) | B (SGP swap) | C (composite AND) | D (parallel) | B−A ExpR | B−A R/yr* |
|----|-----------:|-------------:|------------------:|-------------:|---------:|---------:|
| 1.0 | N=773, ExpR=+0.0353, +4.1 R/yr | N=535, ExpR=+0.0502, +4.0 R/yr | N=486, ExpR=+0.0656, +4.8 R/yr | rho=1.00 BLOCKED | +0.0148 | −0.07 |
| 1.5 | N=773, ExpR=+0.0769, +8.9 R/yr | N=535, ExpR=+0.0812, +6.5 R/yr | N=486, ExpR=+0.0928, +6.8 R/yr | rho=1.00 BLOCKED | +0.0043 | −2.41 |
| 2.0 | N=773, ExpR=+0.0735, +8.5 R/yr | N=535, ExpR=+0.1122, +9.0 R/yr | N=486, ExpR=+0.1198, +8.8 R/yr | rho=1.00 BLOCKED | +0.0387 | +0.48 |

\* R/yr estimate = (N × ExpR) / 6.65 IS years. Excludes annualization drift.

## Adversarial review

### Source-of-truth chain

- `orb_outcomes` + `daily_features` (canonical) via triple-join on `(trading_day, symbol, orb_minutes)`
- `ALL_FILTERS["ORB_G5"]` (OrbSizeFilter ≥5pts) + `ALL_FILTERS["CROSS_SGP_MOMENTUM"]` (CrossSessionMomentumFilter prior_session=SINGAPORE_OPEN)
- `research.filter_utils.filter_signal` delegation (no re-encoded filter logic)
- `trading_app.holdout_policy.HOLDOUT_SACRED_FROM` (no inlined `date(2026,1,1)`)
- `trading_app.lane_correlation.py` rho-definition mirrored (Pearson on intersection pnl_r)

### Independent-SQL verification (from script dev session)

- DB freshness: MNQ/MES/MGC all max_day=2026-04-16 (well clear of Mode A cutoff)
- Break-day universe 848 confirmed; ORB_G5 fires 773, SGP fires 535, both 486, neither 26 — identical to script output
- RR1.5 ExpR computed in two independent paths (script's `compute_mode_a` vs standalone SQL + `filter_signal`) returned identical values to 4 decimals

### Risks checked

| Risk | Status | Note |
|------|--------|------|
| Look-ahead | CLEAN | SINGAPORE_OPEN @ 11:00 Bris; EUROPE_FLOW @ 18:00 — 7h gap. Feature knowable before entry. |
| Mode B baseline contamination | CLEAN | Computed fresh from canonical layers. `validated_setups.expectancy_r` not consulted. |
| Filter re-encoding | CLEAN | Canonical `filter_signal` delegation throughout. |
| Triple-join / N-inflation | CLEAN | Explicit `o.orb_minutes = d.orb_minutes` on all joins. |
| Multiple-testing | N/A | Revalidation of K=3 pre-reg, not new discovery. |
| ARITHMETIC_ONLY (Rule 8.2) | NOT FLAGGED | RR1.5 WR spread G5→SGP = +1.4% with ΔExpR +0.0043 — both differences small, not in the "flat WR + big ExpR move" pattern. |
| EXTREME_FIRE (Rule 8.1) | WARN | ORB_G5 fire-rate 91.2% on EUROPE_FLOW break-days — close to but below the 95% bound. Session-level property of MNQ EUROPE_FLOW (ORB ≥5pt is the norm), not a bug. |
| Execution costs | SYMMETRIC | `pnl_r` is already net of `pipeline.cost_model`. Deltas are unbiased. |

### Remaining uncertainty

- Year-by-year ExpR breakdown is computed (in `year_break` dict) but not written to CSV in this script version. A deeper C9 audit should surface it. Not load-bearing for this portfolio re-eval verdict but recommended for the composite filter pre-reg (Option C work).
- Option C (composite) numbers here are observed-in-Mode-A; no walk-forward decomposition yet. Any decision to deploy C requires a fresh pre-reg + validator run.

## Recommendation

1. **Option B (swap to SGP RR1.5) is no longer justifiable** under Mode A. The +44% per-trade claim was a Mode B artifact. Keep L1 ORB_G5 RR1.5 as the status-quo lane.
2. **Option C (composite ORB_G5 ∧ SGP_TAKE) remains the only non-trivial path.** At RR2.0 it lifts ExpR by +0.0463 vs ORB_G5 alone (+63% relative), with 486 trades vs 773. Needs: (a) new `CompositeFilter(ORB_G5, CROSS_SGP_MOMENTUM)` in `trading_app/config.py`, (b) fresh pre-reg in `docs/audit/hypotheses/`, (c) validator run Pathway B individual, (d) correlation-gate re-check vs all deployed lanes, (e) SR monitor activation. Filed for separate design-proposal approval.
3. **Close off Option A/B/D** — D gate-blocked correctly under Mode A; B dissolves; A is the default.
4. **Update MEMORY correction from prior doc § 7**: the "Jaccard 0.029 cherry-pick" finding is now superseded by the stronger Mode A finding that the whole per-trade-quality claim was a Mode B artifact. Append to errata list.

## Deeper adversarial review — "did we tunnel-vision?"

Second pass after initial doc written. Asked: did I pigeonhole, or test fairly?

### New finding 1 — SGP RR1.5 FAILS C9 era-stability under Mode A

Year-by-year Mode A ExpR on MNQ EUROPE_FLOW RR1.5 long:

| Year | N_all | G5 N | G5 ExpR | SGP N | SGP ExpR | AND N | AND ExpR | SGP-only N | SGP-only ExpR |
|-----:|-----:|-----:|-------:|-----:|--------:|-----:|--------:|----------:|-------------:|
| 2019 | 90 | 51 | −0.0100 | 52 | +0.1291 | 30 | +0.1166 | 22 | +0.1460 |
| 2020 | 128 | 120 | +0.0697 | 74 | +0.0504 | 67 | +0.0837 | 7 | −0.2684 |
| 2021 | 123 | 111 | +0.0965 | 82 | **−0.0213** | 75 | −0.0065 | 7 | −0.1805 |
| 2022 | 129 | 128 | +0.0802 | 81 | +0.0756 | 80 | +0.0651 | 1 | +0.9122 |
| 2023 | 122 | 113 | +0.2155 | 69 | +0.3585 | 62 | +0.3912 | 7 | +0.0696 |
| 2024 | 127 | 122 | +0.0286 | 89 | **−0.1254** | 84 | **−0.0961** | 5 | −0.6176 |
| 2025 | 129 | 128 | +0.0218 | 88 | +0.1713 | 88 | +0.1713 | 0 | — |

**C9 (pre_registered_criteria.md):** "no year ExpR < −0.05 with N ≥ 50" must hold.

- **SGP RR1.5 FAILS C9 in 2024** (N=89, ExpR=−0.1254) — missed in initial doc. This disqualifies Option B swap beyond "numerically marginal" — it's also era-unstable.
- **AND composite RR1.5 FAILS C9 in 2024** (N=84, ExpR=−0.0961). Option C also gets a C9 red flag.
- **ORB_G5 RR1.5 PASSES C9** (worst year is 2019 at −0.01, above −0.05 threshold).
- **SGP RR2.0 does show 6/7 years positive** (per CSV), but year-level N on composites is still hazardous.

### New finding 2 — the 49-day SGP-only pocket is not a viable standalone

Per-year breakdown of the SGP-only cell: average 7 trades/year, variance extreme (−0.62 in 2024 on N=5, +0.91 in 2022 on N=1). This is statistical noise, not a persistent pocket. Prior doc §8's "fallback trigger on small-ORB days" idea is killed.

### New finding 3 — adversarial short-direction check (not in pre-reg)

To test whether I pigeonholed into long-only:

| Direction | Break universe | ORB_G5 N | G5 ExpR | SGP N | SGP ExpR | AND N | AND ExpR |
|-----------|---------------:|---------:|--------:|------:|---------:|------:|---------:|
| Long (canonical) | 848 | 773 | +0.0769 | 535 | +0.0812 | 486 | +0.0928 |
| Short (adversarial) | 870 | 810 | +0.0522 | 445 | +0.0744 | 423 | +0.0877 |

Short direction shows the same confluence lift shape (G5 < SGP < AND) but base rate lower. Consistent with known MNQ EUROPE_FLOW long-bias. Not a missed angle.

### New finding 4 — the real first-order issue may be L1 itself, not SGP

ORB_G5 fires on **91.2% of MNQ EUROPE_FLOW break-days** under Mode A. It's barely a filter — most EUROPE_FLOW ORBs are ≥5pt. The session itself may need re-evaluation beyond the SGP question:

- ORB_G5 RR1.0 ExpR = +0.0353 with N=773 — per-trade edge is thin
- Only +4.1 R/yr on ~116 trades/yr
- Sharpe 0.42 (ann) is below the typical "keep" tier

That's a bigger conversation than Phase 2.4's scope, but should be flagged for the retirement-queue committee alongside this doc.

### Pigeonhole self-check — alternative framings that could have been explored

1. **Other cross-session pairs** (TOKYO→EUROPE_FLOW, LONDON_METALS→EUROPE_FLOW). Not in this pre-reg; filed as future pre-reg work if composite path ever advances.
2. **MES cross-confirmation** on same cell. Pre-reg mentions MES OOS +0.218 as supporting evidence. Not re-computed under Mode A here. Gap.
3. **Other apertures (O15, O30)** on same filter pairing. Locked surface was O5 per deployed lane. Not tested.
4. **SGP-VETO as inverse signal** — days SGP flags as "don't trade". Implicit in AND vs G5-only comparison, but not separately evaluated as a standalone VETO filter.
5. **Cost-after-deduction breakeven sensitivity** — pnl_r is net-of-cost but a +/-20% cost shock test was not run. Filed as T4 gap.

Of these, #2 (MES cross-confirmation under Mode A) is the highest-value follow-up. #1 expands into new discovery territory (pre-reg required).

### Mechanism check

- **Does Singapore→Europe_Flow momentum make sense?** Yes — flows initiated in Asian session with 11:00 Bris start persist into Europe 7h later under momentum hypothesis (per `docs/institutional/mechanism_priors.md`).
- **Does the 2024 collapse make sense?** Possibly. 2024 was a regime-shift year for MNQ (post-election vol, divergent global CB policy). Cross-session momentum mechanisms can break when correlation structure between Asia and Europe flips. Consistent with: SGP tracks the Asia→Europe pass-through, which 2024 disrupted.
- **Does rho=1.0 on shared trades make sense?** Yes, trivially — both filters gate the same ORB break entry at the same RR in the same direction. They differ only in WHICH days fire, not in the P&L once triggered.
- **Does ORB_G5 91% fire-rate make sense?** Yes — EUROPE_FLOW 5m ORBs on MNQ are rarely <5pt in normal vol regimes.

### Revised verdict

Status: **CONFIRMED with material strengthening**. The new Option B/C C9 failures under Mode A year-breakdown push the verdict from "Option B no longer justifiable (per-trade gap small)" to **"Option B explicitly fails C9 + Option C fails C9 at RR1.5"**. Only **Option A (status quo L1 ORB_G5) remains deployment-eligible**.

Option C at RR2.0 remains the ONLY not-yet-killed direction — 6/7 years positive SGP, composite best per-trade of any cell at +0.1198. Not deployable today (needs CompositeFilter infra + pre-reg + validator), but not dead either.

**Follow-up parked:** MES RR1.5 / RR2.0 cross-confirmation under Mode A; year-break write to CSV.

## Third-pass adversarial reframe — golden egg found on MES

Question re-asked: what are we REALLY measuring? Phase 2.4 framed it as "MNQ portfolio swap decision". Running the decomposition at RR1.5 long against the unfiltered baseline and the MES cross-instrument shifted the finding substantially.

### Decomposition against unfiltered baseline (Mode A, RR1.5 long)

| Cell | MNQ N | MNQ ExpR | MNQ lift | MES N | MES ExpR | MES lift |
|------|------:|---------:|---------:|------:|---------:|---------:|
| Unfiltered (break-day universe) | 848 | +0.0607 | — | 850 | **−0.1573** | — |
| ORB_G5 ON | 773 | +0.0769 | +0.0162 | 207 | −0.0460 | +0.1113 |
| ORB_G5 OFF (size<5pt) | 75 | −0.1068 | — | 643 | −0.1932 | — |
| CROSS_SGP_MOMENTUM ON | 535 | +0.0812 | +0.0206 | 530 | −0.1561 | +0.0012 |
| **G5 ∧ SGP (composite)** | 486 | +0.0928 | **+0.0321** | 112 | **+0.0459** | **+0.2032** |

### What this changes

1. **On MNQ, ORB_G5 IS doing something** — it's removing a −0.1068 ExpR pocket (the 75 small-ORB days), lifting +0.016. Small but real. Rejects the "L1 is null" reading I flagged in pass 2.

2. **On MES, SGP_MOMENTUM alone is useless** (+0.0012 lift — within noise). As a standalone cross-instrument test, SGP FAILS T8 (cross-instrument directional consistency test in quant-audit-protocol.md).

3. **BUT the G5 ∧ SGP composite on MES is dramatic** — it rescues a deeply negative unfiltered baseline (−0.1573) to +0.0459 per trade, a +0.20 R/trade lift. N=112 PASSES C7. This is not a thin cell.

4. **The composite has ASYMMETRIC value across instruments.** On MNQ it's marginal-plus (+0.032 lift on an already-positive base). On MES it's transformative (+0.20 lift from negative to positive). This is the structural pattern of a genuine confluence mechanism: it matters most where the base signal is weakest.

### Golden egg

The real next-move opportunity isn't the A/B/C/D portfolio swap on MNQ. It's:

**Test the composite ORB_G5 ∧ CROSS_SGP_MOMENTUM filter as a NEW validated lane on MES EUROPE_FLOW RR1.5 long.** N=112 under Mode A, ExpR=+0.046, rescues a dead session. Warrants:
- Fresh pre-reg at `docs/audit/hypotheses/2026-04-??-mes-europe-flow-g5-sgp-composite.yaml` — Pathway B, K=1, theory = "cross-session momentum confluence on size-filtered breaks" (mechanism: same as Phase 2.4 pre-reg + Fitschen Ch 3 intraday trend follow applied to equity index).
- Full C1-C12 audit (esp. C4 t-stat, C7 N≥100 PASSES at 112, C8 2026 OOS, C9 era-stability per year).
- NEW `CompositeFilter` implementation in `trading_app/config.py` (same requirement as Option C on MNQ) — but with MES as primary deployment target rather than MNQ replacement.

This inverts the original Phase 2.4 question. The cross-session momentum work isn't an MNQ portfolio-construction story. It's a potential MES-session-rescue story.

### Pigeonhole self-critique

- **Wrong baseline:** initial framing compared G5 vs SGP vs composite. Correct framing is compare everything vs UNFILTERED baseline (shows the 2-0 difference in filter power between instruments).
- **Wrong scope:** pre-reg scoped to MNQ only. T8 cross-instrument check was not in scope. Without this adversarial pass we'd have filed SGP as "swap-invalid + composite-marginal" and missed the real opportunity.
- **Wrong question:** "swap or keep" ignores the instrument dimension. The better question was always "where does this mechanism help most?"
- **Year-by-year on MES not run here** — context budget. Filed as required for the follow-up pre-reg.

### Rules-engine check on the MES finding

- T8 cross-instrument of STANDALONE SGP_MOMENTUM: **FAILS** (MES lift +0.001 vs MNQ lift +0.021). Standalone SGP is instrument-specific noise.
- T8 cross-instrument of COMPOSITE G5 ∧ SGP: **STRONGLY PASSES** (both instruments positive, MES dramatically so). Different finding entirely.
- This is NOT a multiple-testing violation — the composite was the Option C framing in the original Phase 2.4 pre-reg. Testing it across instruments is T8 confirmatory, not new discovery.
- **Deployment still gated** on fresh pre-reg + new `CompositeFilter` class + validator run + correlation gate check vs existing MES lanes (if any). This audit only proves the IS signal exists; live deployment is not authorized here.

### Revised verdict (third pass)

- **MNQ Phase 2.4 portfolio decision:** Option A (keep ORB_G5 RR1.5) — unchanged from pass-2 verdict.
- **MNQ Option C (composite):** marginal lift, C9 concerns at RR1.5, not worth composite infrastructure cost for MNQ alone.
- **MES Option C (composite) — NEW FINDING:** material lift from dead-session to tradeable. Pursue this as a separate stage with its own pre-reg. **Best honest next test:** MES EUROPE_FLOW composite under the full C1-C12 criteria.

## Audit trail

- Pre-reg: `docs/audit/hypotheses/2026-04-13-cross-session-sgp-europe-flow.yaml` (committed 2026-04-13 before any scan)
- Script commit: this commit
- Outputs: `research/output/phase_2_4_cross_session_momentum_mode_a_lanes.csv`, `…_options.csv`
- Historical failure log entry (append to `.claude/rules/backtesting-methodology.md`): **"2026-04-19: Mode B trailing-12mo head-to-head as per-trade-quality proxy. Prior cross-session momentum audit framed SGP as +44% ExpR better than ORB_G5 using trailing 12mo on Mode B universe. Mode A reveals the gap is +5.6% (noise-adjacent). Trailing-window comparisons must cite cutoff relative to current HOLDOUT_SACRED_FROM."**
