# OVNRNG Allocator-Routing Diagnostic — ROUTER_HOLDS_OOS

**Status:** superseded the same day by `docs/audit/results/2026-04-21-ovnrng-router-rolling-cv.md`.
Do **not** cite `ROUTER_HOLDS_OOS` as current truth; the later rolling-CV audit retracts it and closes the line as `ROUTER_BRITTLE — DEAD`.

**Date:** 2026-04-21
**Branch:** `research/ovnrng-allocator-routing-diagnostic`
**Script:** `research/audit_ovnrng_allocator_routing.py`
**Parent:** PR #61 (cross-session replication revealed allocator-shaped pattern)

---

## Question

PR #61's tunnel-vision check surfaced: ovn/atr (overnight-range ÷
atr_20) has session-specific effects. High ovn/atr is BEST on 3
sessions, WORST on 1 session — classic allocator-signal shape, not
a filter shape.

Does a router rule — "per trading day, trade only the sessions
that are best for today's ovn/atr bin" — beat uniform trading?
And does it survive walk-forward (no in-sample curve-fitting)?

---

## Verdict

**ROUTER_HOLDS_OOS.** Bin-conditional session routing survives
walk-forward on genuinely out-of-train data.

- **In-sample (2019–2025):** Router top-1 SR_ann=+2.38 vs Control
  top-1 SR_ann=+1.25 vs Uniform SR_ann=+0.76. Bin-awareness contributes
  ΔSR_ann=**+1.13**.
- **Walk-forward (train 2019–2022-08 → test 2022-08–2025):** Router
  top-1 SR_ann=+1.59 (n=572) vs Control top-1 SR_ann=+0.73 (n=659).
  Bin-awareness contributes ΔSR_ann=**+0.87** on genuinely out-of-train
  data.
- Train-derived best-session-per-bin map is NOT stable across folds
  (only 3/5 bins agree between full-IS and train-first-half), yet
  the router still outperforms control — the signal is in the
  bin-dependent-best-session PATTERN, not in any specific session's
  dominance.

Strong evidence for a pre-reg-worthy allocator signal. 2026 OOS
untouched (sacred).

---

## Data

- **Canonical source:** `orb_outcomes` and `daily_features` (triple-
  joined).
- **Universe:** 8 lookahead-clean sessions (≥17:00 Brisbane), MNQ E2
  RR=1.5 CB=1 orb_minutes=5, IS only (pre-2026-01-01).
- **Decision variable:** `ovn_atr = overnight_range / atr_20`.
  Both features are `daily_features` columns, known at Brisbane 17:00
  (overnight window closes). All 8 sessions in scope start AFTER
  17:00 Brisbane — no look-ahead.
- **Quintile boundaries (daily, n=1721):** Q1≤0.208, Q2≤0.269,
  Q3≤0.349, Q4≤0.475, Q5>0.475.
- **Total trades IS:** n=11,957 across 1,721 trading days.

---

## Step 1 — Conditional ExpR(session | ovn/atr bin) — IS

|Session | Q1 | Q2 | Q3 | Q4 | Q5 |
|--------|----|----|----|----|----|
| LONDON_METALS | −0.054 | +0.092 | −0.031 | −0.035 | **+0.214** |
| EUROPE_FLOW | −0.051 | +0.078 | −0.023 | +0.126 | +0.082 |
| US_DATA_830 | +0.043 | +0.008 | +0.038 | −0.070 | −0.038 |
| NYSE_OPEN | +0.119 | +0.074 | **+0.158** | +0.121 | +0.000 |
| US_DATA_1000 | +0.069 | +0.137 | +0.082 | −0.067 | **+0.238** |
| COMEX_SETTLE | +0.033 | +0.129 | +0.086 | +0.007 | +0.077 |
| CME_PRECLOSE | **+0.234** | +0.011 | +0.051 | +0.078 | +0.071 |
| NYSE_CLOSE | −0.071 | −0.029 | −0.066 | −0.053 | +0.149 |

**Best session per bin (IS):** Q1→CME_PRECLOSE, Q2→US_DATA_1000,
Q3→NYSE_OPEN, Q4→EUROPE_FLOW, Q5→US_DATA_1000.

**Observations:**
- Best session DIFFERS in every bin.
- US_DATA_830 underperforms in most bins — a candidate for exclusion.
- NYSE_CLOSE is only positive at Q5 (+0.149R); negative elsewhere.
- NYSE_OPEN sweet-spot at Q3 (PR #47's original finding) visible here.

---

## Step 2 — Policy simulation (IS, pooled bins)

| Policy | n | ExpR | SR_ann | t vs 0 |
|--------|---|------|--------|--------|
| UNIFORM (all 8 sessions) | 11,957 | +0.055 | +0.76 | +5.24 |
| ROUTER top-1 (bin-conditional) | 1,600 | **+0.176** | **+2.38** | +6.00 |
| ROUTER top-2 (bin-conditional) | 3,269 | +0.155 | +2.10 | +7.56 |
| ROUTER top-3 (bin-conditional) | 4,657 | +0.134 | +1.83 | +7.88 |
| CONTROL top-1 (bin-agnostic — best overall session) | 1,649 | +0.094 | +1.25 | +3.20 |
| CONTROL top-2 (bin-agnostic) | 3,322 | +0.093 | +1.24 | +4.51 |
| CONTROL top-3 (bin-agnostic) | 4,618 | +0.092 | +1.24 | +5.30 |

**Router vs Control ΔSR_ann (isolates bin-awareness):** K=1 +1.13,
K=2 +0.86, K=3 +0.59. Top-1 has strongest signal but also single-
session concentration risk.

---

## Step 4 — Walk-forward (IS-only, no 2026 peek)

Train: 2019-05-07 → 2022-08-30 (n=5978 trades, ~3.4 years).
Test: 2022-08-30 → 2025-12-31 (n=5979 trades, ~3.4 years).

### Train-derived best session per bin (NO test peek)

| bin | train best | train ExpR |
|-----|------------|------------|
| Q1 | CME_PRECLOSE | +0.180 |
| Q2 | NYSE_CLOSE | +0.193 |
| Q3 | NYSE_OPEN | +0.168 |
| Q4 | CME_PRECLOSE | +0.178 |
| Q5 | NYSE_CLOSE | +0.289 |

(Differs from full-IS in 3 of 5 bins — Q2, Q4, Q5 — which indicates
some regime drift across the fold. But the ROUTER signal still holds
on test.)

### Test-period simulation (genuinely out-of-train)

| Policy | n | ExpR | SR_ann |
|--------|---|------|--------|
| UNIFORM (test) | 5,979 | +0.075 | +1.03 |
| **ROUTER top-1 (train-derived map)** | **572** | **+0.118** | **+1.59** |
| CONTROL top-1 (train-derived best-session: NYSE_OPEN) | 659 | +0.053 | +0.73 |

**Walk-forward ΔSR_ann (router − control) = +0.87.**

Router SURVIVES walk-forward. Test-period confirms bin-awareness adds
meaningful out-of-train Sharpe, even though the specific bin-to-session
map shifts across the fold.

---

## Tunnel-vision check (mandated self-review)

### Fairly tested

- Cross-session per-(session × bin) ExpR table — IS and walk-forward
- 3 router K values (1, 2, 3) and 3 bin-agnostic control K values
- Walk-forward on a single train/test split — isolates bin-awareness
  contribution
- Look-ahead clean (ovn/atr known at ≥17:00 Brisbane; all 8 sessions
  start after)

### Prematurely ruled out (follow-ups needed before deployment)

1. **Alternative binning variables.** I only tested ovn/atr. Other
   day-level features that could drive similar routing signals:
   `atr_20_pct`, `garch_forecast_vol_pct`, `prev_day_direction`,
   `gap_open_points`, `day_of_week`. A pre-reg should test ovn/atr
   against these controls — otherwise we don't know if ovn/atr is the
   best router variable or just one that works.

2. **RR / entry-model / aperture tunnel.** Only tested E2 RR=1.5
   CB=1 5m. Portfolio deploys multiple RR values (L4 at RR=1.0); the
   router's signal may be RR-dependent.

3. **Single walk-forward split.** One train/test fold is fragile.
   A 3+ fold rolling CV would be stronger OOT evidence. Computationally
   cheap — add to the pre-reg.

4. **No BH-FDR correction.** Scanned 3 router Ks and 3 control Ks
   = 6 policies, plus walk-forward. For a diagnostic this is
   acceptable; for a pre-reg this needs explicit K and correction.

5. **Sizing not tested.** Router top-1 concentrates 100% on 1
   session/day. Could be scaled/split — e.g., router top-2 with 70/30
   weights by train ExpR — for lower concentration risk.

6. **Assumption: every candidate session fires every day.** If any
   of the 8 sessions has no-trade days (holidays, missing data), the
   router should gracefully fall back. Not tested.

### Not-tunnel-vision (critical honest credits)

- Bin assignment is daily, not per-session — correctly models a
  pre-session allocator decision.
- Control policy (bin-agnostic top-1) isolates bin-awareness's
  marginal value, not just session-concentration benefit.
- Walk-forward uses train-only quintile boundaries AND train-only
  session ranking — no train-period peek.
- 2026 OOS is sacred and untouched.

---

## Mechanism hypothesis (for pre-reg theory citation)

Overnight range normalized by 20-day ATR measures **surprise volatility**
— how much overnight movement occurred RELATIVE to the day's usual
range. High ovn/atr suggests event-driven overnight flow; low ovn/atr
suggests a quiet overnight.

Different intraday sessions capture different responses to overnight
flow:
- **CME_PRECLOSE** (06:00 Brisbane, 14:00 ET US cash close-adjacent):
  best at low ovn/atr (Q1) — quiet overnights let intraday trend
  complete cleanly into the close.
- **NYSE_OPEN** (00:30 Brisbane): best at mid ovn/atr (Q3) — moderate
  overnight context gives clean direction into US equity open without
  exhaustion.
- **NYSE_CLOSE / LONDON_METALS / US_DATA_1000**: best at high ovn/atr
  (Q5) — already-active overnight flow continues into their session.

This is consistent with:
- Chan Ch 7 (regime routing + stop-cascade dynamics)
- Chordia et al 2018 (factor-segmented testing, t≥3.79 threshold
  satisfied by router top-1)
- Carver Ch 10 (forecast combination; bin-conditional switching is
  a discrete forecast-combination rule)

---

## Operational implications

### Best opportunity

**A Pathway-B pre-reg for overnight-range-conditioned session router.**
Specifically:
- H1: Router top-1 (bin-conditional) SR_ann beats Control top-1
  (bin-agnostic) by ≥0.30 on 3-fold rolling walk-forward.
- H2: ovn/atr is additive over other day-level routing features
  (test as controls).
- Exit criterion: if 2026 OOS SR_ann on test set < 1.0, kill.
- K budget: bin count × K variants ≤ ~15 framings, BH-FDR q=0.05.

### Biggest blocker

**Allocator infrastructure.** Current
`docs/runtime/lane_allocation.json` is a static DEPLOY list. A router
needs:
1. Pre-session ovn/atr reader (feature is populated — ✓).
2. A routing layer above the current lane allocator that decides
   which lanes to enable PER DAY based on the bin.
3. Operational path: 17:00 Brisbane overnight close known; all 8
   eligible sessions start after; decision is made before each
   session fires.
4. Live integration: execution engine must honour the router's
   daily enable/disable decision.

Estimate: ~1 week of infra work to wire a router into the existing
allocator. Not trivial, not huge.

### Biggest miss (retrospective)

The 2026-04-20 session spent significant time on the
filter-vestigialness and OVNRNG_50_FAST10 correction work but
repeatedly framed ovn/atr as a filter signal. The allocator framing
was visible in PR #47's own per-quintile table but never tested until
today. The per-lane-breakdown-required rule was designed for exactly
this case — and it fired correctly in PR #61, which then led here.

### Next best test

Rolling 3-fold CV (e.g., train on year N–2 through N, test on year
N+1, rolled across 2020–2025) to check whether the train-to-test
best-session map is stable enough to deploy. Current single-fold WF
has the map shift in 3 of 5 bins — a rolling test would quantify this
instability honestly. Read-only, ~30 min. If stable → write the
pre-reg. If unstable → the router's apparent signal is train-set-
specific and the pre-reg is not worth writing.

---

## What this does NOT yet change

- No deployment change recommended.
- No 2026 OOS tested (sacred).
- No new filter registered.
- No production code touched.

## Reproduction / outputs

- Primary script: `python research/audit_ovnrng_allocator_routing.py`
- Canonical inputs: `orb_outcomes`, `daily_features`
- Output class: read-only diagnostic result doc only; no writes to runtime allocation surfaces
- Current repo truth override: this document is retained for provenance, but current decision authority is the later rolling-CV re-audit in `docs/audit/results/2026-04-21-ovnrng-router-rolling-cv.md`

## Caveats / limitations

- This document reports a single-fold walk-forward result, which was later shown to be insufficient for a router-style claim
- Do not use this file alone as evidence that the router is alive; the later rolling-CV audit retracts that conclusion
- No 2026 OOS was touched here, so this file never constituted deploy authority by itself

---

## Provenance

- Canonical data: `orb_outcomes`, `daily_features` (triple-join).
- Holdout: 2026-01-01 (Mode A sacred) — NOT touched.
- Walk-forward train/test split derived from in-sample trade count
  median (single 50/50 fold).
- Read-only research. No production code, no pre-reg, no allocator
  change.
- Runs `python research/audit_ovnrng_allocator_routing.py` on any
  updated `gold.db`.
