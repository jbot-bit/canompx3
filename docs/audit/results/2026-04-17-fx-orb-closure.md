# FX ORB — Class-Level Closure

**Date:** 2026-04-17
**Owner:** claude (session-recovered from Codex crash earlier same day)
**Class:** CME FX futures ORB (6J, 6B, 6A)
**Verdict:** CLOSED — no further transfer attempts on the locked pilot surface

## Two paths tested, two paths closed

### Path 1 — Raw E2 NO_FILTER ORB

- **Pre-reg:** `docs/audit/hypotheses/2026-04-16-cme-fx-futures-orb-pilot.yaml`
- **Result:** `docs/audit/results/2026-04-17-cme-fx-futures-orb-pilot.md`
- **Verdict:** NO_GO — 7 / 7 asset-session candidates failed the locked gate stack on the $29.10 round-trip friction surface.
- **Worst cell:** 6J TOKYO_OPEN — double-break 79.6%, fakeout 57.8%, continuation E2 41.5%, E2 ExpR -0.424R.
- **Least-bad cell:** 6B US_DATA_1000 — double-break 71.4%, fakeout 42.2%, continuation E2 54.4%, E2 ExpR -0.221R. Still not promotable.

### Path 2 — Transferable live-book filter rescue (K=14)

- **Pre-reg:** `docs/audit/hypotheses/2026-04-17-fx-live-analogue-transfer-test.yaml` (status: `BLOCKED_PRE_EXECUTION`)
- **Design:** K=14, top-2 transferable filters per FX cell, analogues inherited verbatim from Path 1's `exact_benchmark_map`, pre-2026 Mode-A-clean calibration + primary test, BH-FDR hierarchical with K_global=14 as sole promotion gate.
- **Blocker:** pre-flight audit of the pilot's detail CSV showed the pre-registered gate stack could not be met by any cell under Mode A.

#### Pre-flight evidence (pre-2026 subset of pilot data)

Pre-2026 eligible days per cell: 71 - 74. Adequate at raw level, fails under any realistic filter gating.

**Category A — `COST_LT12` directly transferable:**

| Cell | Fire rate | N_on | Median cost/risk |
|---|---|---|---|
| 6J TOKYO_OPEN | 1.4% | 1 | 29.7% |
| 6B LONDON_METALS | 0.0% | 0 | 36.8% |
| 6A LONDON_METALS | 0.0% | 0 | 42.1% |
| 6B US_DATA_830 | 4.2% | 3 | 39.9% |
| 6A US_DATA_830 | 5.7% | 4 | 45.4% |
| 6B US_DATA_1000 | 0.0% | 0 | 34.1% |
| 6A US_DATA_1000 | 0.0% | 0 | 36.8% |

`COST_LT12` requires cost/risk < 12%, i.e. raw ORB risk > $242.50 at the locked $29.10 RT friction. FX raw ORB risk sits $54 - $175. The filter is structurally broken on FX cost economics — not merely underpowered.

**Category B — quantile-ported `ORB_G5` analogue:**

| Quantile cut | N_on per cell | Reaches N >= 50? |
|---|---|---|
| Top-50% (loose) | 35 - 37 | No |
| Top-25% (~MNQ ORB_G5 actual selectivity) | 17 - 18 | No |
| Top-70% (trivially loose) | ~52 | Yes, but filter is no longer selective — invites "you used a placeholder cut" critique |

**Category B — quantile-ported `OVNRNG_100` analogue:** same constraint as ORB_G5. Additionally look-ahead-banned for 6J TOKYO_OPEN (09:00 - 17:00 Brisbane window).

**14 / 14 cells fail the gate stack at pre-flight.** No cell can honestly reach N >= 50 without either relaxing the minimum (pre-reg violation), extending the data window (new pre-reg, not an amendment), or dropping Mode A (sacred-window violation).

## Decision rule followed

14 of 14 pre-registered cells fail the gate stack at pre-flight under Mode A. No cell can reach N ≥ 50 without relaxing the pre-registered minimum, extending the data window (which would require a new pre-reg, not an amendment), or dropping Mode A (sacred-window violation). Structural thinness at this magnitude is doctrine-grounded as a kill under `.claude/rules/backtesting-methodology.md` Rule 5 (comprehensive scope — narrowing allowed only with pre-reg justification) and Rule 8.1 (extreme fire rate — rare-event gating requires pre-registered justification, not post-hoc widening). Neither applies here. Closed.

## Why no Option B (2-year pre-2026 data pull)

Option B — pre-register V2 with net-new DBN extraction covering ~2023-01 to 2025-12 — is arithmetically viable (would give N_eligible ~500 / cell) but was declined. Two independent paths (raw + filter-rescue) have now failed on the locked pilot surface. Further credit and extraction-infrastructure spend on a research tree that has shown two failure modes is not justified.

## Hygiene note on the raw pilot (tracked separately, out of scope here)

The raw pilot's primary analysis ran on 147 - 148 days per cell, spanning 2025-09-18 through 2026-04-06. That window mixes pre-2026 IS and post-2026 OOS data without reporting them separately under Mode A.

- **Impact on raw pilot verdict:** none. Both subsets failed; the NO_GO stands.
- **Impact on deployment-grade citability:** the raw pilot is NOT deployment-grade-citable until re-reported with an explicit pre-2026-only split.
- **Action:** noted; re-report is not a priority since the class is closed. If any future work wants to cite the raw pilot in a deployment-readiness context, the re-report must precede.

## Do not re-open unless

- A brand-new pre-registration is filed, AND
- It uses a data source materially wider than the locked pilot surface (e.g., 2+ years of pre-2026 FX DBNs with full front-contract roll audit, OR a different entry model class, OR a different asset like EUR futures at a validated session), AND
- The reopening rationale is economic / mechanistic, not statistical rescue-fishing.

Patching thresholds, swapping cells, or widening sessions under the current pilot surface is explicitly forbidden by both Path 1 and Path 2 pre-reg rules.

## Artifacts for future session context

- Raw pilot pre-reg: `docs/audit/hypotheses/2026-04-16-cme-fx-futures-orb-pilot.yaml`
- Raw pilot result: `docs/audit/results/2026-04-17-cme-fx-futures-orb-pilot.md`
- Raw pilot script: `research/cme_fx_futures_orb_pilot.py`
- Raw pilot outputs: `research/output/cme_fx_futures_orb_pilot.json`, `research/output/cme_fx_futures_orb_pilot_detail.csv`
- Filter-rescue pre-reg (BLOCKED): `docs/audit/hypotheses/2026-04-17-fx-live-analogue-transfer-test.yaml`
- This closure: `docs/audit/results/2026-04-17-fx-orb-closure.md`
