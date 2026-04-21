# 2026-04-21 Orthogonal POV Map

Purpose:
- Rank orthogonal candidate families for the Fork D window.
- Select the top 3 directions for pre-registration without touching the 6 posture-blocked live lanes.

Authority inputs:
- [docs/audit/2026-04-21-data-landscape.md](docs/audit/2026-04-21-data-landscape.md)
- [docs/institutional/mechanism_priors.md](docs/institutional/mechanism_priors.md)
- [docs/STRATEGY_BLUEPRINT.md](docs/STRATEGY_BLUEPRINT.md)
- `origin/research/pr48-sizer-rule-oos-backtest:docs/audit/2026-04-21-phase-b-institutional-reeval.md` read-only on the Phase B lineage

Scoring rubric:
- `M1` mechanism strength
- `M2` canonical feasibility
- `M3` non-overlap with live six
- `M4` non-overlap with NO-GO registry
- `M5` MinBTL cost efficiency
- `M6` robustness surface
- `M7` deployment-path clarity
- `M8` posture-blocker-proof projection

Score scale:
- `3` = strong
- `2` = workable
- `1` = weak
- `0` = structurally bad fit

## Occupied Surface Fence

The currently occupied active surface is narrow and highly specific:
- instruments: `MES`, `MNQ`
- entry model: `E2`
- confirm bars: `CB1`
- live sessions: `CME_PRECLOSE`, `COMEX_SETTLE`, `EUROPE_FLOW`, `NYSE_OPEN`, `SINGAPORE_OPEN`, `TOKYO_OPEN`, `US_DATA_1000`
- active filters in use: `ATR_P50`, `COST_LT08`, `COST_LT12`, `CROSS_SGP_MOMENTUM`, `ORB_G5`, `ORB_G8`, `OVNRNG_100`, `VWAP_MID_ALIGNED`, `X_MES_ATR60`
- active RR targets: `1.0`, `1.5`, `2.0`

Implication:
- orthogonal hunt candidates should prefer `MGC`, uncovered `MES` sessions, uncovered `MNQ` sessions, unused canonical features, or role changes that do not collapse back into the existing live-six structure.

## Ranked Family Table

| Family | M1 | M2 | M3 | M4 | M5 | M6 | M7 | M8 | Total | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `F3 Regime conditioners orthogonal to current ORB filters` | 3 | 3 | 3 | 2 | 3 | 3 | 3 | 3 | 23 | Best match to unused `daily_features` surface (`atr_vel_ratio`, `atr_vel_regime`, `pit_range_atr`, `prev_week_range`, `prev_month_range`, event-day flags). Distinct from blanket calendar skip no-go because scope can stay session-specific and mechanism-led. |
| `F6 Session-boundary effects` | 3 | 3 | 3 | 2 | 3 | 2 | 3 | 3 | 22 | Strong local mechanism support from Chan intraday momentum and the landscape’s uncovered session-boundary cells (`BRISBANE_1025`, `CME_REOPEN`, `US_DATA_830`, `NYSE_CLOSE`). |
| `F7 Microstructure conditioners inside bars_1m` | 2 | 2 | 3 | 3 | 2 | 2 | 2 | 3 | 19 | Highest orthogonality among non-live surfaces; requires only `bars_1m` derivation, no new pipeline code. Lower than F3/F6 because the local literature grounding is thinner. |
| `F8 Negative-space plays` | 1 | 3 | 3 | 2 | 2 | 2 | 2 | 2 | 17 | Useful search lens, not a mechanism on its own. Best used as a selector over F3/F6/F7 rather than as a standalone family. |
| `F5 Position-sizing edges on canonical entry distributions` | 3 | 2 | 1 | 2 | 2 | 2 | 2 | 1 | 15 | Strong Carver grounding, but it inherits too much from existing entry surfaces and is therefore weak on posture-blocker-proof projection. Better as Stage 2 after a new entry/filter edge exists. |
| `F4 EXIT-side edges on existing canonical outcomes` | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 1 | 15 | Orthogonal in role, but it tends to piggyback on existing entry surfaces and Stage 3 literature grounding in `mechanism_priors.md` is explicitly incomplete. |
| `F2 Cross-instrument relative value` | 1 | 1 | 3 | 3 | 1 | 2 | 1 | 2 | 14 | Vector-space orthogonal, but the repo’s local mechanism support is thin and the deployment path is materially less clear than F3/F6. |
| `F1 Non-ORB entry timing inside established sessions` | 1 | 2 | 3 | 0 | 1 | 1 | 1 | 1 | 10 | Collides with the blueprint’s `non-ORB strategies` no-go family and would consume trial budget rapidly. Dropped unless a genuinely new mechanism emerges. |

## Top 3 Selected

### 1. `POV-REGIME-ORTHO`

Definition:
- session-specific regime conditioners on uncovered `MGC` / uncovered `MES` ORB cells using canonical-but-unused `daily_features` fields.

Why this is first:
- The data landscape shows 289 canonical feature columns and only 7 used by the live six.
- The strongest unused surface is regime information already present in `daily_features`, especially `atr_vel_ratio`, `atr_vel_regime`, `pit_range_atr`, `prev_week_range`, `prev_month_range`, `day_of_week`, `is_nfp_day`, and `is_opex_day`.
- This is the cleanest way to get a new filter family without replaying `ORB_G5`, `COST_LT12`, `ATR_P50`, or the existing MNQ live-six vector.

Target surface:
- `MGC` first, then uncovered `MES`
- prefer sessions outside the current live-six set, especially where the landscape shows dense pre-holdout trade-day budgets and negative-space coverage

Tradeoff:
- strongest posture-blocker-proof projection
- lowest additional engineering burden
- lower narrative risk than a brand-new execution architecture

### 2. `POV-SESSION-BOUNDARY`

Definition:
- session-boundary effects on uncovered cells using `gap_open_points`, pre-session range context, and session-relative location features.

Why this is second:
- Chan’s intraday momentum extract locally grounds overnight-gap / stop-cascade logic for index futures.
- The landscape shows large uncovered stable-vol cells in `BRISBANE_1025`, `CME_REOPEN`, `US_DATA_830`, and `NYSE_CLOSE`, especially on `MNQ` and `MES`, with full canonical support.
- This is orthogonal to the current book because the live six are concentrated in a different session cluster and use none of these session-boundary-specific explanatory variables directly.

Target surface:
- `MES` and `MGC` first for cleaner vector-space separation
- `MNQ` uncovered sessions only if they remain clearly outside the live-six tuples and filter menu

Tradeoff:
- slightly more fragile than F3 because session-boundary narratives can collapse into blanket calendar/gap-no-go if scoped sloppily
- still strong enough to make the top 3 because the canonical inputs and deployment path are clear

### 3. `POV-MICROSTRUCTURE`

Definition:
- bars_1m-derived microstructure conditioners such as early-range expansion, burst intensity, or liquidity-vacuum proxies, derived ad hoc without pipeline edits.

Why this is third:
- It is the most orthogonal remaining surface that still fits the repo’s no-new-dependency and no-pipeline-edit constraints.
- It avoids the dead `break speed / break delay` family by not reusing those exact variables or their already-killed interpretations.
- It is a credible hedge if F3/F6 produce only posture-blocked SILVERs.

Target surface:
- `MGC` and uncovered `MES` sessions before any `MNQ` reuse
- use `bars_1m` only, with derived features materialized into `outputs/` artifacts rather than canonical tables

Tradeoff:
- mechanism support is weaker than F3/F6
- trial-budget discipline has to stay tight because bar-level conditioners can explode combinatorially if not pre-scoped carefully

## Explicit Drops / Second-Tier Candidates

- `F1 Non-ORB entry timing` dropped now:
  - conflicts too directly with the blueprint’s `non-ORB strategies` no-go family
  - high MinBTL burn for weak posture payoff

- `F2 Cross-instrument relative value` dropped to second tier:
  - attractive vector-space distance, but weak local mechanism grounding and unclear deployment path
  - worth reconsidering only if the top 3 all fail or if a tighter spread mechanism is grounded first

- `F4 EXIT-side edges` dropped to second tier:
  - too likely to inherit the same posture blocker as the entry surface it modifies
  - `mechanism_priors.md` explicitly marks Stage 3 stop/target geometry as incompletely grounded

- `F5 Position-sizing edges` dropped to second tier:
  - role is grounded by Carver, but the best use is after a new orthogonal edge exists
  - using sizing first risks re-packaging the blocked live-book problem instead of escaping it

- `F8 Negative-space plays` not selected as a standalone:
  - retained as a search lens
  - every actual candidate still needs to live inside a mechanism family such as F3/F6/F7

## Non-Overlap Proof

The top 3 selected directions stay outside the live occupied surface in one or more of these ways:
- primary instrument tilt toward `MGC` and uncovered `MES`, while the live six are all `MNQ`
- feature menu excludes the current live-six filters `ORB_G5`, `COST_LT12`, and `ATR_P50`
- family focus is on unused canonical descriptors or new bar-derived microstructure, not on the current filter menu or rel-vol / PR #48 lineage
- session preference includes uncovered session-boundary cells rather than defaulting back into the exact live-six tuple set

## Recommendation

Advance to Phase 2 pre-registration with these three concrete hunt directions:
1. `POV-REGIME-ORTHO`
2. `POV-SESSION-BOUNDARY`
3. `POV-MICROSTRUCTURE`

Do not pre-register F1, F2, F4, F5, or F8 in this session unless one of the top 3 is killed before consuming meaningful trial budget.
