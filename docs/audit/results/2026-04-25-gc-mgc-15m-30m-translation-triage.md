# GC → MGC 15m / 30m Translation Triage

**Date:** 2026-04-25
**Verdicts (two, distinct):**
- Translation framing: `CLOSE_PATH`.
- MGC-native 30m RR1.0 sub-region: `PARK_NOT_KILLED`.
- MGC-native 15m standalone: `EMPIRICALLY_DEAD_AT_CELL_LEVEL` (standalone role only).

**Closes:** action-queue item `gc_mgc_15m_30m_translation_question` (P2 close-first).
**Closes stage:** `docs/runtime/stages/gc-mgc-15m-30m-translation-question.md`.

## Scope

The stage asked: "after the 5m payoff-compression question is settled, is
there any coherent 15m or 30m GC → MGC translation claim worth testing, or
should the wider-aperture path be closed?" Stage scope-lock: docs-only,
question-framing, no broad gold rediscovery. Apertures in scope: `O15`, `O30`.

Per discovery protocol (canonical-only proof, derived layers orient only),
this triage separates every claim into MEASURED repo truth, GROUNDED prior,
and INFERRED hypothesis. Chordia / Bailey / Harvey-Liu thresholds from
`docs/institutional/literature/` govern the evidence bar.

## Method

All measured rows below come from `orb_outcomes` directly. No derived layer
was used as proof. Query: `symbol`, `orb_minutes`, `orb_label`, `rr_target`
aggregates over `trading_day >= 2022-06-13 AND trading_day < 2026-01-01`
(the MGC-GC overlap era, Mode A IS), `entry_model='E2'`, `confirm_bars=1`,
unfiltered, cell-gate `N >= 80`.

Script: `.discovery_query.py` (ad-hoc, not checked in — queries `pipeline.paths.GOLD_DB_PATH`).

## MEASURED — structural availability check

| symbol | orb_minutes | rows | days | first | last |
|---|---:|---:|---:|---|---|
| GC  | 5  | 9,165 | 1,141 | 2022-01-03 | 2025-12-31 |
| GC  | 15 | **0** | **0** | — | — |
| GC  | 30 | **0** | **0** | — | — |
| MGC | 5  | 8,127 | 1,010 | 2022-06-13 | 2025-12-31 |
| MGC | 15 | 7,963 | 984 | 2022-06-13 | 2025-12-31 |
| MGC | 30 | 7,670 | 958 | 2022-06-13 | 2025-12-31 |

The GC proxy surface does not exist at `orb_minutes IN (15, 30)`. Any
"translation claim" at those apertures has no source data and no
retrievable canonical surface to translate from. Structural.

Independently confirmed by the 2026-04-19 GC → MGC translation audit
disclaimer: "No claim is made about GC 15m/30m proxy transfer because the
canonical GC proxy surface here is 5-minute only."

## MEASURED — MGC-native aperture surface (reframed object)

The honestly-reframed object, if the stage is read charitably, is
"MGC-native viability at wider apertures." MEASURED across 27 session×RR
cells per aperture (9 sessions × 3 RR), unfiltered E2 CB1 overlap IS,
cell-gate N≥80:

| Aperture | Positive cells / 27 | Strongest positive (raw) | Strongest negative (raw) |
|---|---:|---|---|
| 5m | **0** | — (none) | TOKYO_OPEN RR1.0 avg_r=-0.2133 t=-8.77 N=918 |
| 15m | 2 | NYSE_OPEN RR1.0 avg_r=+0.0241 t=+0.81 N=914 | COMEX_SETTLE RR2.0 avg_r=-0.2797 t=-7.55 N=906 |
| 30m | 5 | **NYSE_OPEN RR1.0 avg_r=+0.0628 t=+2.04 N=895**<br>**US_DATA_1000 RR1.0 avg_r=+0.0587 t=+1.93 N=891** | COMEX_SETTLE RR2.0 avg_r=-0.2701 t=-6.95 N=886 |

- MGC 5m: universally negative across all 27 cells. Confirms the
  payoff-compression finding of `docs/audit/results/2026-04-19-mgc-payoff-compression-audit.md`.
- MGC 15m: 2 of 27 positive; strongest |t|<1. No standalone edge.
- MGC 30m: 5 of 27 positive; two cells with raw |t| between 1.9 and 2.1.
  Both at RR1.0 (lowest RR in the enumeration).

## GROUNDED — bar the evidence must clear

- Chordia et al. 2018: t ≥ 3.00 with prior mechanism theory; t ≥ 3.79
  without. Source: `docs/institutional/literature/chordia_et_al_2018_*.md`
  (cited via `docs/institutional/pre_registered_criteria.md` criterion 4).
- Bailey et al. 2013: MinBTL bounds brute-force trials against horizon;
  27 cells × 3 apertures = 81 trials on 3.5 years of overlap data is
  already near the budget; expanding to filter sweeps would violate it.
- Harvey-Liu 2015 + BH-FDR q<0.05 at the declared K framing; raw p≈0.042
  (t=+2.04) at K=81 does not survive.
- Power-floor rule (memory `feedback_oos_power_floor.md`, also §6 of
  `docs/plans/2026-04-25-cross-asset-session-chronology-spec.md`): OOS
  power vs IS effect ≥ 50% required; N_OOS on 2026 Mode A horizon is
  ~80 raw days per cell (UNVERIFIED against current cutoff; upper bound
  from 3 months × ~60 fire rate), well below 50% power for effect sizes
  of +0.06 R.

Neither 30m RR1.0 candidate meets Chordia, BH-FDR, OR the OOS power
floor. Neither meets the standalone-edge bar.

## INFERRED — mechanism consistency (honest, flagged as hypothesis)

The two 30m near-signals both sit at RR1.0, the lowest of {1.0, 1.5, 2.0}.
The `MGC payoff-compression audit` (2026-04-19) concluded
`PAYOFF_COMPRESSION_REAL=YES` and `LOW_RR_RESCUE_PLAUSIBLE=YES`; lower-RR
targets lift expectancy on MGC 5m. Widening the aperture increases the
absolute-R distance between ORB and target, which is a different but
same-direction mechanism: both favour RR1.0 over RR2.0 on compressed MGC.

This makes the 30m RR1.0 positives mechanism-consistent with an already-
published prior, which distinguishes them from a raw post-hoc rescue. It
does NOT lift them above the discovery bar. It only means they are not
noise-by-default either.

## Verdict

Three sub-verdicts, intentionally separated to prevent the common reader-collapse of "translation dead → MGC wider-aperture dead."

### Verdict A — `CLOSE_PATH` on the translation framing

The as-named object ("GC → MGC translation at 15m/30m") is structurally
unanswerable because the GC proxy surface at 15m/30m does not exist in
canonical `orb_outcomes` (0 rows). No reopening path without a new data
architecture that makes GC 15m/30m available, which is out-of-scope for
this stage and not currently a declared project workstream. Translation
framing: closed.

### Verdict B — `PARK_NOT_KILLED` on MGC-native 30m RR1.0

This closure does **not** kill the MGC-native wider-aperture question.
MEASURED canonical evidence shows two 30m RR1.0 cells with raw t between
+1.9 and +2.1 at N ≈ 895. They do not clear the Chordia, BH-FDR, or OOS
power-floor bar, so they are not promotable as discoveries. They are
mechanism-consistent with the 2026-04-19 payoff-compression audit's
`LOW_RR_RESCUE_PLAUSIBLE` finding, so they are also not dismissable as
noise-by-default.

Honest classification: `PARK_NOT_KILLED`, priority lower than the already-
recommended MGC 5m exit-shape prereg path.

### Verdict C — `EMPIRICALLY_DEAD_AT_CELL_LEVEL` on MGC-native 15m standalone

MEASURED: only 2 of 27 15m cells show positive expectancy; strongest t=+0.81.
No standalone edge. Not a KILL for other roles (conditioner / confluence /
filter) — this verdict applies to the standalone-edge framing only.

## Re-open triggers

The MGC-native 30m RR1.0 park clears only under EITHER of the following
(both measurable without rerun):

- **Trigger 1 — K=1 theory-driven pre-registration.** An author writes a
  Pathway-B K=1 pre-reg naming exactly one cell (recommended start:
  `MGC NYSE_OPEN O30 E2 CB1 RR1.0`), citing the payoff-compression
  mechanism from `docs/audit/results/2026-04-19-mgc-payoff-compression-audit.md`
  as prior theory, declaring a kill criterion BEFORE looking at filter
  variants, and accepting the OOS `UNVERIFIED` verdict if OOS N on Mode A
  at execution time does not clear the 50% power floor. If authored,
  this stage's park converts to an active research path.

- **Trigger 2 — the MGC 5m exit-shape prereg resolves as NO_RESCUE.**
  If the 5m exit-shape path (recommended separately by the 2026-04-19
  payoff-compression audit) fails to rescue the 5m MGC expectancy, the
  30m RR1.0 path becomes the next honest MGC discovery question by
  elimination, and a K=1 pre-reg on it is the natural next move under
  the same mechanism theory.

Priority ordering: under the current state, the MGC 5m exit-shape prereg
path is recommended AHEAD of any 30m K=1 test, because the payoff-
compression audit named it explicitly and because 5m has higher fire-rate
→ faster OOS power accumulation.

## Carry-forward state

- Canonical `orb_outcomes` unchanged.
- Validated shelf unchanged — no MGC aperture row is promoted or demoted
  by this doc.
- Live allocator unchanged — verified at commit time: `docs/runtime/
  lane_allocation.json` contains zero MGC lanes
  (`['MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5',
    'MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15',
    'MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5',
    'MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12',
    'MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12',
    'MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15']`).
  The two MGC 30m RR1.0 cells named in this doc are NOT in the allocator.
- 5m MGC exit-shape prereg path remains open elsewhere. This doc does
  not touch it.

## Outputs

- This result doc at
  `docs/audit/results/2026-04-25-gc-mgc-15m-30m-translation-triage.md`.
- `docs/runtime/action-queue.yaml` item `gc_mgc_15m_30m_translation_question`
  transitioned to `status: done` with `notes_ref` pointing here.
- `docs/runtime/stages/gc-mgc-15m-30m-translation-question.md` deleted.

No code changes. Canonical queries run directly against `orb_outcomes` at
`pipeline.paths.GOLD_DB_PATH` with the Mode A IS window
(`HOLDOUT_SACRED_FROM = 2026-01-01`).

## Limitations

- This triage did NOT run a filtered enumeration over the MGC 30m cells.
  Doing so would expand K beyond MinBTL on 3.5 years of overlap data and
  belongs behind a pre-reg, not in a triage.
- This triage did NOT evaluate the MGC 30m RR1.0 cells against Mode A OOS
  (2026-01-01 onward). By design — selecting based on IS t-stat and then
  looking at OOS is data-snooping. A pre-reg must declare the cell and the
  kill criterion BEFORE OOS is observed.
- The mechanism story (wider aperture reduces payoff compression) is
  GROUNDED in the 2026-04-19 audit but is not independently re-verified
  here. A K=1 pre-reg would still have to declare it as prior theory in
  its own right.
- MGC 15m is labelled empirically dead at the cell-level for the
  standalone-edge role only. This doc does NOT kill MGC 15m for use as a
  conditioner, filter input, or confluence surface in some other stage.

## References

- `docs/audit/results/2026-04-19-gc-mgc-translation-audit.md` — the 5m
  translation audit that originally disclaimed 15m/30m.
- `docs/audit/results/2026-04-19-mgc-payoff-compression-audit.md` — source
  of the payoff-compression mechanism and of the recommended 5m exit-shape
  prereg path.
- `docs/plans/2026-04-25-cross-asset-session-chronology-spec.md` — the
  chronology/admissibility/power-floor governance spec from PR #101.
- `docs/institutional/literature/` — Chordia, Bailey, Harvey-Liu sources.
- `docs/institutional/pre_registered_criteria.md` — criterion-4 t thresholds.
- `.claude/rules/backtesting-methodology.md` — RULE 4 (multiple testing),
  RULE 5 (scope discipline), RULE 6 (trade-time knowability).
- `pipeline.paths.GOLD_DB_PATH`, `trading_app/holdout_policy.py::HOLDOUT_SACRED_FROM`.
- Memory: `feedback_oos_power_floor.md`, `feedback_pooled_not_lane_specific.md`.
- `docs/runtime/stages/gc-mgc-15m-30m-translation-question.md` — stage
  closed by this doc.
- `docs/runtime/action-queue.yaml` — queue item set to `status: done`.
