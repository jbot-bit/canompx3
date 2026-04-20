# Context Opportunity Map — Grounded In Resources And Canonical Truth

**Date:** 2026-04-20
**Status:** DECISION SURFACE
**Purpose:** close the tunnel on the HTF/LTF / ORB-context thread by ranking the real opportunity set across roles after re-checking against `/resources` and canonical `gold.db` outputs.

## Grounding base

This note is grounded in both repo doctrine and the raw `/resources` PDFs:

- `resources/Algorithmic_Trading_Chan.pdf`
  - bounded, explicit strategy families are acceptable research units when rules are objective and executable
- `resources/Robert Carver - Systematic Trading.pdf`
  - useful signals can live as conditioners / sizers / forecast modifiers rather than forcing every edge into a standalone strategy
- `resources/Lopez_de_Prado_ML_for_Asset_Managers.pdf`
  - theory first, small pre-registered families, do not treat backtests as idea generators in themselves
- `resources/Two_Million_Trading_Strategies_FDR.pdf`
  - honest family-level multiple-testing control is mandatory

Repo-side anchors:

- `docs/institutional/mechanism_priors.md`
- `docs/institutional/edge-finding-playbook.md`
- `docs/institutional/pre_registered_criteria.md`
- `RESEARCH_RULES.md`

## Canonical truth checked

### 1. Bounded ORB retest route

Canonical rerun:

- `research/orb_retest_entry_pilot_v1.py`
- output: `research/output/orb_retest_entry_pilot_v1_cells.csv`

Verdict:

- **DEAD**
- honest `K = 54`
- `1` paired-delta pass, `0` trading-relevant passes
- kills the bounded ORB execution-variant route only

### 2. Pooled prior-day short allocator route

Canonical rerun:

- `research/bull_short_avoidance_pooled_deployed_oos.py`

Verdict:

- **DEAD**
- pooled IS delta only `+0.0061R`
- pooled block-bootstrap `p = 0.8676`
- this is not a hidden broad sizer / allocator edge

### 3. MNQ prior-day positional ORB family

Locked family:

- `docs/audit/hypotheses/2026-04-20-prior-day-direction-split-orb-overlays-v1.yaml`
- `research/prior_day_direction_split_orb_overlays_v1.py`

Verdict:

- **ALIVE** as a context-overlay family
- honest `K = 36`
- `7` primary survivors, but only **4 unique shapes** after collapsing RR duplicates

Unique surviving shapes:

1. `NYSE_OPEN` short `F5_BELOW_PDL` → `TAKE`
   - survives at RR `1.0` and `1.5`
2. `US_DATA_1000` long `F5_BELOW_PDL` → `TAKE`
   - survives at RR `1.0` and `1.5`
3. `US_DATA_1000` long `F6_INSIDE_PDR` → `AVOID`
   - survives at RR `1.0` and `1.5`
4. `COMEX_SETTLE` long `F6_INSIDE_PDR` → `AVOID`
   - survives at RR `1.0` only

Important non-duplication note:

- `US_DATA_1000` long `F5_BELOW_PDL` and `F6_INSIDE_PDR` are not the same signal relabeled
- correlation of the binary fire masks is about `-0.51`
- overlap is `0`
- interpretation is coherent:
  - washed-out below-PDL states help
  - inside-prior-day-range states hurt

### 4. Exact live-lane follow-up

Locked exact-lane verify:

- `docs/audit/hypotheses/2026-04-20-f5-nyo-short-deployed-lane-verify.yaml`
- `research/f5_nyo_short_deployed_lane_verify.py`

Lane:

- `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` short

Verdict:

- **CONDITIONAL_UNVERIFIED**
- canonical IS:
  - `N_on = 117`
  - `N_off = 708`
  - `ExpR_on = +0.3559`
  - `ExpR_off = +0.0519`
  - `delta = +0.3040`
  - Welch `p = 0.0009`
  - block-bootstrap `p = 0.0020`
- IS year consistency:
  - positive delta in every eligible year
- OOS:
  - `N_on = 8`
  - `N_off = 33`
  - `delta = -0.1478`
  - RULE 3.3 power tier = `STATISTICALLY_USELESS`

This is a real exact-lane candidate, but it is not promotable yet.

## Role map

### Standalone

What is proven:

- only narrow PDH/PDL event-study families were tested, and they are dead

What is **not** proven:

- broader standalone open-displacement / retest / reclaim classes are not killed by those null event studies

Current status:

- **UNVERIFIED**

### Filter / conditioner

What is proven:

- prior-day positional features can improve ORB trade quality locally
- both `TAKE` and `AVOID` roles are real in the same family

Current status:

- **ALIVE**

### Allocator / broad sizer

What is proven:

- pooled prior-day short split across all deployed lanes is dead

Current status:

- **DEAD** for the broad pooled prior-day short thesis

### Confluence

What is proven:

- `H04_CMX_SHORT_RELVOL_Q3_AND_F6` is still alive as a narrow exact-lane confluence

Current status:

- **ALIVE but OOS-thin**

## Best opportunity

### Immediate best opportunity

`MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` short conditioned on `F5_BELOW_PDL`

Why this ranks first:

- exact live lane
- strong IS evidence under canonical filter delegation
- already mapped to the correct role: conditioner / deployment-shape
- lower implementation drag than any new standalone family
- better immediate EV than revisiting dead pooled allocator ideas

### Best parallel opportunities

1. `MNQ_US_DATA_1000 O5 long` paired family
   - `F5_BELOW_PDL` TAKE and `F6_INSIDE_PDR` AVOID are now canonically verified in the bounded `K=4` follow-on family
   - see `docs/audit/results/2026-04-20-usdata1000-long-f5-f6-paired-v1.md`
   - role-design synthesis now says the best ORB route is **`NOT_F6_INSIDE_PDR` candidate lane first**, with `F5_BELOW_PDL` acting as the higher-quality sub-state
   - see `docs/plans/2026-04-20-usdata1000-paired-role-design.md`
   - still not an exact live lane today, so this remains a lane-discovery / validation path, not a live overlay path
2. `H04_CMX_SHORT_RELVOL_Q3_AND_F6`
   - already on its own shadow/deployment-shape path
   - do not broaden it

## Biggest blocker

OOS thinness on the exact live-lane candidates.

That is a monitoring / phase-gating problem, not a reason to kill a real IS signal.

## Biggest miss corrected

The branch had been overweighting the wrong prior-day family:

- pooled `bull vs bear previous day` short logic is **dead**
- positional prior-day states (`below PDL`, `inside PDR`) are **alive**

So the mistake was not “prior-day context is useless.”
The mistake was “we were looking at the wrong prior-day context variable.”

## Next best tests

### Rank 1

Write the shadow / deployment-shape pre-reg for:

- `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` short
- `F5_BELOW_PDL`
- role: conditioner / upsize-or-take-shape, not standalone

### Rank 2

Promote the verified `US_DATA_1000` long pair into the next bounded
validation / role-design step:

- `F5_BELOW_PDL` TAKE
- `F6_INSIDE_PDR` AVOID

This should be treated as a paired opportunity family because the states are
adjacent and mutually exclusive, and the bounded `K=4` follow-on verify kept
all four locked cells alive.

That role-design step is now done. The next bounded contract to preserve is the
`NOT_F6_INSIDE_PDR` candidate lane, with `F5_BELOW_PDL` treated as a secondary
priority descriptor rather than the primary gate.

That candidate-lane validation is also now done:

- `docs/audit/results/2026-04-20-usdata1000-long-not-f6-candidate-lane-v1.md`
- verdict: both RR cells are `RESEARCH_SURVIVOR`, not `CANDIDATE_READY`

So the honest next step is a signal-only shadow path, not promotion.

### Rank 3

Only after the above:

- reopen a true standalone family
- likely open-displacement reversal or another non-ORB path
- with fresh pre-reg and resource grounding

Do **not** reopen:

- pooled prior-day short allocator logic
- ORB retest continuation
- simple HTF break-through filters
