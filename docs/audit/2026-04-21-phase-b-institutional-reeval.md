# 2026-04-21 Phase B Institutional Re-Evaluation

Authority:
- Binding Phase A ledger: [docs/audit/2026-04-21-reset-snapshot.md](/mnt/c/Users/joshd/canompx3/docs/audit/2026-04-21-reset-snapshot.md) at commit `5e768af8`
- Existing adversarial extension: superseded uncommitted P1 draft findings are folded into this document and no longer left loose in the working tree
- Phase B outputs of record: [docs/decisions/2026-04-21-phase-b-rollup.md](/mnt/c/Users/joshd/canompx3/docs/decisions/2026-04-21-phase-b-rollup.md), [research/phase_b_live_lane_verdicts.py](/mnt/c/Users/joshd/canompx3/research/phase_b_live_lane_verdicts.py)

## Integrity Anchors

- Snapshot commit `5e768af8` resolves and matches the Phase A ledger.
- Phase B commits `dfb1bbab`, `f89f0702`, and `f4007c41` are all present on `origin/research/pr48-sizer-rule-oos-backtest`.
- Local literature extracts resolve:
  - [docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md](/mnt/c/Users/joshd/canompx3/docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md)
  - [docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md](/mnt/c/Users/joshd/canompx3/docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md)
  - [docs/institutional/literature/harvey_liu_2015_backtesting.md](/mnt/c/Users/joshd/canompx3/docs/institutional/literature/harvey_liu_2015_backtesting.md)
  - [docs/institutional/literature/chan_2013_ch1_backtesting_lookahead.md](/mnt/c/Users/joshd/canompx3/docs/institutional/literature/chan_2013_ch1_backtesting_lookahead.md)
- Canonical truth for this re-evaluation is restricted to `orb_outcomes`, `daily_features`, and the committed Phase A snapshot. The earlier Phase B / P1 path used `active_validated_setups`; under the refined prompt that is a truth-layer violation for empirical claims. This document quarantines that metadata path to provenance/posture only.

## §1 Truth + Calculation Check

The Phase B combined verdict mixed two different evidence classes:

1. Empirical return quality:
   - recomputed here from canonical `orb_outcomes + daily_features`
   - lane routing spec taken from the live six IDs in the Phase A snapshot plus their active profile definitions
   - `pnl_r` treated as the canonical cost-inclusive return stream, consistent with [docs/RESEARCH_ARCHIVE.md:134](/mnt/c/Users/joshd/canompx3/docs/RESEARCH_ARCHIVE.md:134)
2. Deployment posture:
   - inherited from Phase A snapshot provenance findings
   - not reused from `active_validated_setups` for empirical-edge claims

Canonical re-check of live-lane empirical quality used:
- the exact lane filter via `trading_app.config.ALL_FILTERS[filter_type].matches_row(...)`
- actual `max_orb_size_pts` overlay from the active profile
- `orb_outcomes.pnl_r` on rows with `outcome in ('win','loss')`
- annualized trade-level Sharpe proxy: `mean(pnl_r) / std(pnl_r) * sqrt(252)`
- year-by-year expectancy from the filtered canonical return stream

Result:
- No empirical-edge numbers in this document depend on `validated_setups`, `edge_families`, or `live_config`.
- The inherited Phase B posture metrics that still depend on `active_validated_setups` are treated as provenance signals only, not truth for edge existence.

## §2 Test Quality

The old combined Phase B gate stack answered the wrong question in three places:

| Lane | Holdout under fair framing | DSR under pending-ONC caveat | Chordia band that actually applies |
|---|---|---|---|
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` | `PROVENANCE-BLOCKED`, not empirical fail | `PENDING-ONC` | `3.79` strict |
| `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` | `PROVENANCE-BLOCKED`, not empirical fail | `PENDING-ONC` | `3.79` strict |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` | `PROVENANCE-BLOCKED`, not empirical fail | `PENDING-ONC` | `3.79` strict |
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` | `PROVENANCE-BLOCKED`, not empirical fail | `PENDING-ONC` | `3.79` strict |
| `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12` | `PROVENANCE-BLOCKED`, not empirical fail | `PENDING-ONC` | `3.79` strict |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | `PROVENANCE-BLOCKED`, not empirical fail | `PENDING-ONC` | `3.79` strict |

Audit conclusions:
- Holdout was circular for this cohort. The criteria file itself says the grandfathered book is `Research-provisional` and `NOT OOS-clean` under Mode A: [docs/institutional/pre_registered_criteria.md:476](/mnt/c/Users/joshd/canompx3/docs/institutional/pre_registered_criteria.md:476)-[482](/mnt/c/Users/joshd/canompx3/docs/institutional/pre_registered_criteria.md:482).
- DSR is not a clean deploy/no-deploy gate here because Amendment 2.1 explicitly downgraded it to cross-check while `N_eff` / ONC remains unresolved: [docs/institutional/pre_registered_criteria.md:293](/mnt/c/Users/joshd/canompx3/docs/institutional/pre_registered_criteria.md:293)-[307](/mnt/c/Users/joshd/canompx3/docs/institutional/pre_registered_criteria.md:307).
- The stricter `3.79` Chordia band still applies. `mechanism_priors.md` does not override that for these live lanes; it explicitly says Chordia `t>=3.79` still binds unless a literature extract says otherwise: [docs/institutional/mechanism_priors.md:60](/mnt/c/Users/joshd/canompx3/docs/institutional/mechanism_priors.md:60), [125](/mnt/c/Users/joshd/canompx3/docs/institutional/mechanism_priors.md:125), [175](/mnt/c/Users/joshd/canompx3/docs/institutional/mechanism_priors.md:175).

## §3 Anti-Tunnel Check

Framings and whether they were actually tested:

- Standalone Mode-A institutional deployment gates: tested in Phase B, result `0/6 KEEP`.
- Filter-contribution framing inside a larger book: not tested.
- Portfolio-contribution framing as allocator input: not tested.
- ONC-clustered DSR framing: not computed under freeze; pending upstream ONC work.
- Empirical edge separated from deployment posture: not tested in Phase B; done here.

The tunnel error in Phase B was not a fake result. It was a category error: posture gates were read as edge-death evidence.

## §4 Reframe

The correct question is not "did each lane clear KEEP under the full institutional stack?" The correct split is:

- `Empirical-edge`: does the filtered canonical return stream still show a positive, stable-enough process?
- `Deployment-posture`: is the lane clean enough to call institutionally deployable right now?

### Two-Verdict Table

| Lane | Empirical-edge | Deployment-posture | Phase B combined |
|---|---|---|---|
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` | `CONDITIONAL` | `PROVENANCE-BLOCKED` | `DEGRADE` |
| `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` | `KEEP` | `PROVENANCE-BLOCKED` | `DEGRADE` |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` | `CONDITIONAL` | `PROVENANCE-BLOCKED` | `DEGRADE` |
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` | `CONDITIONAL` | `PROVENANCE-BLOCKED` | `PAUSE-PENDING-REVIEW` |
| `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12` | `KEEP` | `PROVENANCE-BLOCKED` | `DEGRADE` |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | `KEEP` | `PROVENANCE-BLOCKED` | `DEGRADE` |

Empirical-edge classification rule used here:
- `KEEP`: canonical `pnl_r` expectancy positive, annualized Sharpe proxy positive, Phase A `WFE >= 0.5`, and no negative year with `N >= 50`
- `CONDITIONAL`: process positive overall, but one or more negative years with `N >= 50` or live SR alarm
- `DEAD`: not observed in this six-lane book under canonical rows
- `UNVERIFIED`: not needed for these six lanes

Canonical empirical metrics:

| Lane | N | ExpR | Sharpe_ann | Negative years with `N>=50` |
|---|---:|---:|---:|---|
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` | 1592 | `+0.0770` | `+1.0893` | `2019 (-0.0831, N=98)` |
| `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` | 819 | `+0.0938` | `+1.3054` | none |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` | 1564 | `+0.0713` | `+0.9898` | `2019 (-0.2365, N=95)`, `2022 (-0.0252, N=235)` |
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` | 1695 | `+0.0850` | `+1.4089` | `2023 (-0.0012, N=247)` |
| `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12` | 985 | `+0.1296` | `+1.7852` | none |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | 1328 | `+0.1256` | `+1.6618` | none |

Phase A WFE remains positive for all six: [docs/audit/2026-04-21-reset-snapshot.md:586](/mnt/c/Users/joshd/canompx3/docs/audit/2026-04-21-reset-snapshot.md:586)-[603](/mnt/c/Users/joshd/canompx3/docs/audit/2026-04-21-reset-snapshot.md:603).

## §5 Entry / Execution Realism

PR #48 does not materially change the empirical verdict of the six live lanes.

Reason:
- PR #48 / Q5 rel_vol are separate MES/MGC lineages and were already marked timing-invalid as `E2` pre-entry framing in the Phase A snapshot: [docs/audit/2026-04-21-reset-snapshot.md:615](/mnt/c/Users/joshd/canompx3/docs/audit/2026-04-21-reset-snapshot.md:615)-[620](/mnt/c/Users/joshd/canompx3/docs/audit/2026-04-21-reset-snapshot.md:620).
- The active six-lane live book uses `ORB_G5`, `COST_LT12`, and `ATR_P50`, all timing-valid in current framing: [docs/audit/2026-04-21-reset-snapshot.md:609](/mnt/c/Users/joshd/canompx3/docs/audit/2026-04-21-reset-snapshot.md:609)-[613](/mnt/c/Users/joshd/canompx3/docs/audit/2026-04-21-reset-snapshot.md:613).

NYSE deserves one separate realism note:
- Its `PAUSE-PENDING-REVIEW` is an execution/regime control issue first, not clear evidence of empirical edge death, because the live SR monitor is in `ALARM` while the long-horizon canonical return stream remains positive: [docs/audit/2026-04-21-reset-snapshot.md:624](/mnt/c/Users/joshd/canompx3/docs/audit/2026-04-21-reset-snapshot.md:624)-[631](/mnt/c/Users/joshd/canompx3/docs/audit/2026-04-21-reset-snapshot.md:631).

## §6 Edge Location

Lane-level location call:

| Lane | Edge location |
|---|---|
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` | `CONDITIONAL` |
| `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` | `STANDALONE` |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` | `CONDITIONAL` |
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` | `CONDITIONAL` |
| `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12` | `STANDALONE` |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | `STANDALONE` |

Count summary:
- `STANDALONE`: 3
- `CONDITIONAL`: 3
- `PORTFOLIO-ONLY`: 0
- `DEAD`: 0
- `UNVERIFIED`: 0

This is the key hidden answer: the live six do not look empirically dead on canonical rows. They look deployment-blocked and, in half the book, regime-fragile.

## §7 Brutal Filter

Framing kills:
- All 6 are posture-blocked by provenance. None can be called institutionally clean under Mode A.

Real kills:
- None from the canonical empirical return stream.

Documentation-fixable:
- None cleanly. The Chordia `3.00` path is not merely missing a sentence; it lacks the required direct literature grounding for these lanes.

Fragile lanes:
- `EUROPE_FLOW`: negative 2019 with `N=98`
- `COMEX_SETTLE`: negative 2019 and 2022 with `N>=50`
- `NYSE_OPEN`: live SR alarm plus slightly negative 2023 with `N=247`

Non-fragile relative to this cohort:
- `SINGAPORE_OPEN`: one negative year exists but only at `N=47`
- `TOKYO_OPEN`: one negative year exists but only at `N=31`
- `US_DATA_1000`: no negative years in the canonical sample

## §8 Edge Search Direction

Highest-EV next move:
- Resolve baseline-book provenance / ONC posture before touching C1. The current book is not institutionally classifiable as `KEEP`, but the canonical lane returns do not justify calling it dead.

Do NOT touch:
- MNQ biphasic work already shelved
- Q5 rel_vol post-break variants from this terminal
- any new scan, pre-reg, or sizer rerun under the freeze

## Highest-EV Next Move

Separate the baseline book problem from the filter-menu problem. The next useful project action is not ORB_G5 tinkering. It is a baseline-book posture decision that says, in writing, whether the current live cohort should remain explicitly `research-provisional + operationally deployable` while ONC / clean rediscovery remain unresolved.

## Fork Memo

- `A` Proceed to C1 ORB_G5 now:
  - low value because ORB_G5 sits inside a baseline already provenance-blocked across all six lanes
- `B` Pause C-family, address baseline-book provenance first:
  - strong candidate because all six are framing-kills on posture, not clean deployment candidates
- `C` Both, serial:
  - viable, but C1 still optimizes a book whose institutional status is unresolved
- `D` Wait for Terminal 1 ONC + baseline-cleanliness decision before any C-phase movement:
  - best fit for this terminal under the freeze and current split of ownership

Recommendation: `D`

Rationale:
- Empirical edge is still present in the live six.
- Deployment posture is blocked for all six.
- DSR remains explicitly unresolved at the ONC layer.
- C1 on ORB_G5 before that decision would optimize a baseline whose real problem is provenance, not filter degeneracy.
