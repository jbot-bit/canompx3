# Prior-Day Context — Blocker Memo (2026-05-02)

**Type:** Read-only proposal-closure memo.
**Purpose:** record the smallest safe conclusion on the `PD_*` branch under current local resources and current runtime.

## Scope

This memo does **not** reopen discovery, does **not** create a new prereg,
does **not** claim deployment readiness, and does **not** change live routing.

It answers only:

1. Can the current local `resources/` + extracted literature honestly theory-grant
   prior-day-context geometry filters (`PD_CLEAR_LONG`, `PD_GO_LONG`,
   `PD_DISPLACE_LONG`) to use the lower `t >= 3.00` hurdle?
2. What role should the surviving `PD_*` rows be treated as right now?
3. What are the explicit stop gates before this branch can be reopened?

## Sources read

Authority / methodology:

- `RESEARCH_RULES.md`
- `docs/institutional/pre_registered_criteria.md`
- `docs/institutional/conditional-edge-framework.md`
- `docs/institutional/README.md`
- `docs/institutional/mechanism_priors.md`

Measured branch state:

- `docs/audit/results/2026-04-25-prior-day-bridge-execution-triage.md`
- `docs/audit/results/2026-04-23-prior-day-geometry-routing-audit.md`
- `docs/audit/results/2026-04-23-prior-day-geometry-execution-translation-audit.md`
- `docs/audit/results/2026-05-02-deployable-pool-edge-survey.md`

Literature grounding checked:

- `docs/institutional/literature/chan_2013_ch7_intraday_momentum.md`

Canonical replay run for this memo:

```bash
./.venv-wsl/bin/python -c "<bounded replay for the 5 PD_* shelf rows; yearly IS/OOS ExpR by strategy>"
```

The replay used existing repo helpers:

- `research.portfolio_additivity_engine.load_candidate_specs`
- `research.portfolio_additivity_engine.load_trade_records`

Truth surface: `orb_outcomes` + `daily_features`.

## Verdict

### 1. Theory-grant status

**BLOCKED under current local resources.**

Why:

- `docs/institutional/README.md` states that Fitschen Ch 3 grounds the **core
  ORB premise** but does **not** ground prior-day H/L or pivot filters; it
  says separate level-based literature is needed and is not present in
  `resources/`.
- `docs/institutional/mechanism_priors.md` says the same: level-based
  mechanism claims remain priors until separate literature exists, and the
  operational consequence is that these tests stay on the stricter Chordia
  path.
- `chan_2013_ch7_intraday_momentum.md` grounds intraday breakout momentum and
  stop cascades. It does **not** directly ground prior-day level geometry as a
  distinct theory class.

**Result:** do **not** lower `PD_*` candidates to the `t >= 3.00` path from the
current local literature set.

### 2. Role classification

The surviving `PD_*` rows should currently be treated as:

- `conditioner` / `confluence` first
- not as fresh standalone live-route candidates

Why:

- `docs/institutional/conditional-edge-framework.md` says prior-day levels,
  pivot proximity, and displacement should default to `conditioner` or
  `confluence`, not `standalone`.
- `docs/audit/results/2026-04-23-prior-day-geometry-routing-audit.md` keeps all
  five survivors on the shelf rather than routing them live.
- `docs/audit/results/2026-04-25-prior-day-bridge-execution-triage.md` says the
  exact bridge path is already consumed and parked on measured blockers.

### 3. Measured branch state

**Do not reopen the exact Pathway-B bridge cells.**

Measured blockers already documented:

- all 6 locked bridge hypotheses were already consumed once
- OOS power floor fails on the exact bridge cells
- all 5 shelf survivors are `KEEP_ON_SHELF`
- the 4 `US_DATA_1000` rows are `ARCHITECTURE_CHANGE_REQUIRED`
  because same-session half-size coexistence is unexpressible on 1-contract
  lanes

### 4. Era-stability note from bounded replay

The branch is **not dead**, but it is also **not clean enough** to promote.

Measured from canonical replay on the 5 shelf rows:

- `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_GO_LONG`
  - positive in every IS year 2019-2025, but weak in 2019/2020/2023/2025
  - OOS 2026: `N=18`, `ExpR=+0.2449`
- `MNQ_US_DATA_1000_E2_RR1.5_CB1_PD_GO_LONG`
  - strong in 2021/2022/2024, flat-to-weak in 2019/2020/2025
  - OOS 2026: `N=18`, `ExpR=+0.2697`
- `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_PD_CLEAR_LONG`
  - negative in 2019 and 2020, positive after that
  - OOS 2026: `N=17`, `ExpR=+0.2225`

Interpretation:

- the branch is alive as a research context family
- the branch is not currently clean enough to support a new live-route claim
- the branch remains vulnerable on role confusion and era-stability grounds

## Explicit stop gates

Do **not** proceed past this memo unless one of these gates is satisfied.

1. **Theory gate**
   A direct local literature extract is added that honestly grounds prior-day
   level/context geometry as its own mechanism class.

   Until then:
   - no `t >= 3.00` downgrade for `PD_*`
   - no claim that Chan Ch 7 is sufficient for prior-day geometry

2. **Bridge-consumption gate**
   Do not rerun or reframe the already-consumed exact Pathway-B bridge cells as
   if they were fresh unexplored hypotheses.

3. **Runtime gate**
   Do not claim live-route readiness for the `US_DATA_1000 O5` survivors while
   same-session reduced-size coexistence remains unexpressible on 1-contract
   lanes under the current runtime.

## Smallest safe next step

**Park the prior-day-context theory-grant path under current local resources.**

If this branch is revisited before new literature arrives, the only honest
follow-through is a bounded role/era memo on the existing `PD_*` shelf rows,
with no theory-grant claim and no deployment claim.

## Reproduction

This memo is a literature-grounding read of `resources/` PDFs against the
project Chordia-strict-unlock workflow; no executable artefact. To
reconstruct the conclusion:

1. Local resources read: `resources/Chan_2013.pdf` Ch 7 (gap mean
   reversion) — confirm passage on 1-2 day post-gap retracement; absent
   passage on prior-day high/low/range/middle as a structural mechanism.
2. Local resources read: `resources/Fitschen_2013.pdf` Ch 3 — confirm
   absence of prior-day-level mechanism distinct from intraday breakout.
3. Local resources read: `resources/Carver_2015.pdf` Ch 9-10 — confirm
   sizing/portfolio scope, not entry-mechanism support for PD_* features.
4. Cross-check: `docs/institutional/literature/` does not contain a
   prior-day-context extract as of 2026-05-02. Therefore filter-grounding-
   status for `PD_*` is UNSUPPORTED, and per
   `docs/institutional/pre_registered_criteria.md` Criterion 4 the strict
   `t >= 3.79` (no theory grant) threshold applies.
5. Memo audit trail: this file + `chordia_audit_log.yaml` PARK row(s) for
   any audited `PD_*` strategies + `docs/runtime/action-queue.yaml` entry
   referencing the deployable-pool survey.

## Caveats / Limitations

- This is a **theory-grant feasibility** memo, not a falsification of any
  specific `PD_*` strategy. Strategies that pass strict `t >= 3.79`
  remain deployable through the standard Chordia gate; the memo only
  forbids invoking the theory-grant `t >= 3.00` downgrade path.
- The memo does not search outside `resources/` or
  `docs/institutional/literature/`. If a literature extract is added
  later that grounds prior-day geometry as a distinct mechanism class
  (e.g., a Fitschen-style intraday-extension chapter not yet extracted),
  the theory gate above can be reopened with the extract path cited.
- Memo is dated 2026-05-02. If `validated_setups` or
  `experimental_strategies` materially shifts (new `PD_*` rows, removal
  of cited rows), revisit before consuming.
- The "smallest safe next step" recommendation is doctrine-level only;
  it does not encode a numeric kill or revive criterion. Per
  `.claude/rules/backtesting-methodology.md` RULE 3.2, any future OOS
  evidence on `PD_*` strategies at `N_OOS < 30` must be treated as
  directional-only, not confirmatory.
