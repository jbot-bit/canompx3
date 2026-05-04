# Canonical Coverage and Opportunity Audit

Date: 2026-04-19

## Scope

Re-open the current research/deployment thread from first principles and ask:

- where canonical ORB opportunity is actually concentrated
- which apparent gaps are real versus already explored
- which blockers are valid versus stale or mis-scoped
- what the highest-EV next move is without tunneling into the current local thread

This is a **canonical discovery audit**, not a deployment approval memo.

## Guardrails

- Canonical proof for discovery uses only:
  - `orb_outcomes`
  - `daily_features`
  - `bars_1m` only where needed for future follow-up
- 2026 holdout remains diagnostic only and was not used for selection.
- Derived layers (`experimental_strategies`, `validated_setups`,
  `deployable_validated_setups`, `prop_profiles`) are comparison/context only.
- No claims below are based on prior summaries without direct query support in
  this run.

## Source-of-Truth Chain

Canonical:

1. `gold.db::orb_outcomes`
2. `gold.db::daily_features`
3. `pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS`

Comparison only:

4. `gold.db::experimental_strategies`
5. `gold.db::validated_setups`
6. `gold.db::deployable_validated_setups`
7. `trading_app/prop_profiles.py`

## Audit Design

### Baseline comparator

Use one fixed broad comparator first:

- pre-2026 only
- `entry_model='E2'`
- `confirm_bars=1`
- evaluate `rr_target` surfaces honestly rather than choosing one post-hoc
- report `avg(pnl_r)`, `n`, and t-statistics directly from `orb_outcomes`

This is not enough for promotion, but it is enough to test whether current
concentration is obviously arbitrary or actually present in canonical truth.

### Session-family gap map

For each instrument/session family:

- count strong baseline cells at `RR ∈ {1.0, 1.5}` with `t >= 3` and `avg_r > 0`
- compare against:
  - `experimental_strategies`
  - `validated_setups`
  - `deployable_validated_setups`

This separates:

- untouched gaps
- explored-but-killed families
- promoted/deployable families
- out-of-bounds but structurally strong families

## Executive Verdict

The current repo is **not** mainly missing profit because of random neglect of
the active ORB surface.

What is real:

- active-universe canonical opportunity is genuinely concentrated in `MNQ`
- `MES` sparsity is largely earned by weak or negative canonical baseline
- `MGC` sparsity is also earned at the micro contract level
- the largest structurally important unresolved opportunity is **gold contract
  translation** (`GC` strong, `MGC` weak), not another adjacent `MNQ` overlay
- the one active-universe family that still looks like a serious gap is
  `MNQ NYSE_CLOSE`, but it is **not untouched**; it has already been explored
  in `experimental_strategies` and failed to promote

So the next honest move is **not**:

- more near-neighbor level-event families
- broad inactive-profile activation
- random session fishing

The next honest move is:

1. a `GC -> MGC` translation audit
2. then an `MNQ NYSE_CLOSE` failure-mode audit

## What Is Real

### 1. The active ORB book is canonically MNQ-dominant

Active ORB doctrine still limits the live/tradeable surface to:

- `MES`
- `MGC`
- `MNQ`

Direct pre-2026 baseline results on that active universe show:

- `MNQ` has many strong positive broad cells
- `MES` has **zero** strong positive broad cells at the fixed `E2 / CB1 / RR {1.0, 1.5}` comparator
- `MGC` has **zero** strong positive broad cells at the same comparator

Examples:

- `MNQ US_DATA_1000 E2 RR1.0 CB1`: `n=4683`, `avg_r=+0.0951`, `t=6.84`
- `MNQ NYSE_OPEN E2 RR1.0 CB1`: `n=4508`, `avg_r=+0.0939`, `t=6.54`
- `MNQ CME_PRECLOSE E2 RR1.0 CB1`: `n=1809`, `avg_r=+0.1227`, `t=5.76`
- `MES CME_PRECLOSE E2 RR1.0 CB1`: `n=1669`, `avg_r=+0.0308`, `t=1.47`
- `MGC NYSE_OPEN E2 RR1.0 CB1`: `n=2380`, `avg_r=+0.0105`, `t=0.58`

This means the current deployable concentration on `MNQ` is not just profile
or allocator bias. It is materially present in canonical pre-2026 truth.

### 2. The deployable shelf concentration is aligned with that canonical truth

Current deployable shelf:

- `38` rows total
- `MNQ = 36`
- `MES = 2`
- `MGC = 0`

Every deployable row is:

- `entry_model = E2`
- `confirm_bars = 1`

That sounds narrow, but on the active universe it is not obviously arbitrary.
The broad positive canonical surface for active instruments is already heavily
dominated by `E2 / CB1` on `MNQ`.

### 3. MES scarcity is probably real, not a missed baseline edge

`MES` has broad experimental coverage across many sessions:

- `CME_PRECLOSE`
- `CME_REOPEN`
- `COMEX_SETTLE`
- `EUROPE_FLOW`
- `LONDON_METALS`
- `NYSE_CLOSE`
- `NYSE_OPEN`
- `SINGAPORE_OPEN`
- `TOKYO_OPEN`
- `US_DATA_1000`
- `US_DATA_830`

But under the fixed broad comparator it has no strong positive session-family
cells and many strongly negative ones:

- `MES EUROPE_FLOW E2 RR1.0 CB1`: `avg_r=-0.1121`, `t=-9.50`
- `MES TOKYO_OPEN E2 RR1.0 CB1`: `avg_r=-0.0970`, `t=-8.26`
- `MES US_DATA_830 E2 RR1.0 CB1`: `avg_r=-0.0851`, `t=-6.87`
- `MES COMEX_SETTLE E2 RR1.0 CB1`: `avg_r=-0.0699`, `t=-5.47`

There are some warm direction-specific hints, but not enough to claim that MES
has been broadly neglected.

### 4. MGC scarcity is also real at the micro contract level

`MGC` shows no strong positive broad session-family cells and many large
negative ones:

- `MGC COMEX_SETTLE E2 RR1.0 CB1`: `avg_r=-0.1367`, `t=-8.06`
- `MGC TOKYO_OPEN E2 RR1.0 CB1`: `avg_r=-0.1300`, `t=-8.64`
- `MGC EUROPE_FLOW E2 RR1.0 CB1`: `avg_r=-0.0856`, `t=-5.47`
- `MGC SINGAPORE_OPEN E2 RR1.0 CB1`: `avg_r=-0.1003`, `t=-6.25`

This means the absence of `MGC` from the deployable shelf is not a simple
oversight.

### 5. The real gold opportunity is a translation problem, not a missing raw edge

`GC` is structurally strong in multiple sessions at the same broad comparator:

- `GC NYSE_OPEN E2 RR1.0 CB1`: `avg_r=+0.1225`, `t=8.18`
- `GC US_DATA_1000 E2 RR1.0 CB1`: `avg_r=+0.1072`, `t=6.97`
- `GC LONDON_METALS E2 RR1.0 CB1`: `avg_r=+0.0639`, `t=4.36`
- `GC US_DATA_830 E2 RR1.0 CB1`: `avg_r=+0.0606`, `t=3.91`
- `GC SINGAPORE_OPEN E2 RR1.0 CB1`: `avg_r=+0.0574`, `t=3.89`

But those same sessions are weak or negative in `MGC`.

Examples of direct `GC -> MGC` gaps at `RR=1.0`:

- `SINGAPORE_OPEN`: `GC +0.0574` vs `MGC -0.1003` → gap `+0.1577`
- `LONDON_METALS`: `GC +0.0639` vs `MGC -0.0634` → gap `+0.1273`
- `NYSE_OPEN`: `GC +0.1225` vs `MGC +0.0105` → gap `+0.1120`
- `US_DATA_1000`: `GC +0.1072` vs `MGC +0.0042` → gap `+0.1030`

This is the biggest structurally important unresolved opportunity in the repo:
**gold edge exists canonically, but the micro proxy does not inherit it cleanly.**

## What Is Overstated

### Overstated: "The repo is missing lots of active-universe ORB profit because sessions were not tested equally"

Not supported.

- `MES` has broad experimental coverage and weak canonical baseline
- `MGC` has little promotion and weak canonical baseline
- `MNQ` is the only active instrument with repeated strong broad positives

So "test everything equally again" is not the right conclusion.

### Overstated: "Deployable shelf concentration on E2/CB1 must be research bias"

Also not supported on the active universe.

Current deployable rows are all `E2 / CB1`, but the broad positive canonical
surface on the active universe is also dominated by `E2 / CB1`.

### Overstated: "MGC absence means gold is dead"

Wrong.

Gold is not dead canonically. The problem is **translation from `GC` to `MGC`**.

## Gap Classification

### A. Real unresolved gap: `GC -> MGC` translation

- `GC` has multiple strong broad positive families
- `MGC` broadly does not
- this is a real, high-value unresolved gap

Verdict: `PRIORITY`

### B. Candidate gap, but not untouched: `MNQ NYSE_CLOSE`

Canonical broad baseline:

- `MNQ NYSE_CLOSE E2 RR1.0 CB1`: `n=1329`, `avg_r=+0.1000`, `t=4.04`
- direction split:
  - long: `avg_r=+0.1519`, `t=7.20`
  - short: `avg_r=+0.0992`, `t=4.39`

Derived comparison:

- `experimental_strategies`: `10` rows
- `validated_setups`: `0`
- `deployable_validated_setups`: `0`

So this is **not** an untouched blind spot. It has been explored, but nothing
survived promotion.

Example rejected experiment rows:

- `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_OVNRNG_100`
  - `sample_size=267`
  - `expectancy_r=+0.1510`
  - `p=0.0070`
  - `fdr_adjusted_p_discovery=0.0187`
  - `validation_status=REJECTED`
- `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_ORB_G5_NOFRI`
  - `sample_size=640`
  - `expectancy_r=+0.0734`
  - `fdr_adjusted_p_discovery=0.0829`
  - `validation_status=REJECTED`

This means the right question is not "why haven’t we looked at NYSE_CLOSE?"
The right question is "why does strong broad baseline fail to survive the
filter/promotion path?"

Verdict: `SECONDARY — post-mortem candidate`

### C. Warm but not strong: side-specific `MNQ LONDON_METALS long`

Broad aggregate family is not strong:

- `MNQ LONDON_METALS E2 RR1.0 CB1`: `avg_r=+0.0186`, `t=1.44`

But direction split is interesting:

- long: `avg_r=+0.0337`, `t=3.25`
- short: `avg_r=+0.0003`, `t=0.03`

This is plausible, but second-tier. It has not earned priority over the two
bigger issues above.

Verdict: `WATCHLIST`

### D. Warm but sub-threshold: `MES US_DATA_1000 short`

Broad aggregate is near flat:

- `MES US_DATA_1000 E2 RR1.0 CB1`: `avg_r=+0.0054`, `t=0.41`

Direction split:

- short: `avg_r=+0.0328`, `t=2.94`

Interesting, but not yet strong enough to call a major missed opportunity.

Verdict: `WATCHLIST`

## False Positives / False Negatives

### What we may have called good too fast

- The idea that stale inactive profiles imply a large immediately accessible
  dormant profit pool.
- The idea that more adjacent level-event families are the shortest path to
  more edge.

### What we may have killed too fast

- Not a full kill, but `MNQ NYSE_CLOSE` deserves a proper failure-mode audit.
  Strong broad baseline plus `10` rejected experimental rows means it is not a
  random ghost session.
- `GC` was effectively pushed out by active-instrument doctrine without the
  translation problem being resolved.

## Blocker Audit

### Blocker 1 — Gold translation

- owner layer: active instrument doctrine + contract granularity
- evidence: `GC` strong, `MGC` weak across the same sessions
- still supported?: yes
- cost of keeping it:
  - likely leaves real gold edge unextracted
- risk of weakening/removing it:
  - medium; could lead to over-trusting `GC` history without micro/live
    execution parity
- verdict: `RE-SCOPE`

This blocker should not be removed blindly. It should be turned into a specific
translation audit.

### Blocker 2 — More near-neighbor level-event family work

- owner layer: current local research thread
- evidence:
  - level pass/fail v1: null
  - sweep/reclaim v1: null
- still supported?: no
- cost of keeping it:
  - burns cycles on local mechanism neighbors after two nulls
- risk of weakening/removing it:
  - low
- verdict: `REMOVE`

### Blocker 3 — Inactive-profile activation as the main profit move

- owner layer: deployment translation
- evidence:
  - rewrite is real
  - activation review remains blocked on runtime/readiness
- still supported?: partly
- cost of keeping it as the top priority:
  - distracts from the bigger unresolved edge question
- risk of weakening/removing it:
  - low, if kept as secondary implementation work
- verdict: `WEAKEN`

## Next Actions

Ordered by truth importance, EV potential, and shortest honest path:

1. **Run a `GC -> MGC` translation audit**
   - same sessions
   - same entry model family
   - same era splits
   - gross vs net retention
   - decide whether the edge dies from contract microstructure, friction,
     session timing, or filter mismatch

2. **Run an `MNQ NYSE_CLOSE` failure-mode audit**
   - explain the mismatch between strong broad baseline and zero validated rows
   - use the existing experimental rows first
   - classify the kill mechanism:
     - FDR
     - WFE
     - holdout
     - filter mismatch
     - cost geometry

3. **Keep deployment translation work secondary**
   - inactive-profile rewrite was still worth doing
   - but it is not the main profit unlock until a runtime surface is actually
     activation-ready

4. **Do not open another adjacent level-event family**
   - that path is locally exhausted for now

## Bottom Line

The strongest honest missed opportunity is **not** a random untested active
session. It is the unresolved fact that canonical `GC` ORB opportunity is real
while `MGC` translation is poor.

Inside the active universe, the only session that still looks like a serious
candidate mismatch is `MNQ NYSE_CLOSE`, but it is an **explored-and-rejected**
family, not an untouched blind spot.
