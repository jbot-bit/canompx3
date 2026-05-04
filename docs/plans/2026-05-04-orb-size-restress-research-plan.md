# ORB-size restress research plan (post-Phase-3c, all instruments)

**Status:** PROPOSED (research/audit, no implementation yet)
**Date:** 2026-05-04
**Author:** Claude Code (validator-honesty-fix session)
**Type:** research/audit plan -- not validator code

## Why this plan exists

This document supersedes the original "validator-honesty-fix" plan's Stage 4
("MGC stress-test informational stamp"). That stage was paused mid-session
when an institutional-grade audit found four wrong claims and one unverified
premise:

1. (WRONG) "MGC ALL edges die without size filter" justified MGC-only
   stamping. Actual source `RESEARCH_RULES.md:251` says **"MES/MGC: ALL
   edges die without size filter"** -- the finding covers MES too. MNQ
   CME_REOPEN survived; other MNQ sessions are uncharacterised.
2. (WRONG) Plan said "mirrors DSR pattern (Amendment 2.1)" but routed the
   stamp into ``promotion_provenance`` (existing label field), while DSR
   uses a dedicated ``dsr_score`` column.
3. (WRONG) Test selected (``test_1_no_size_filter``) measures whether the
   strategy's edge survives WITHOUT its size filter -- but the survivor
   strategy is ``ORB_G4`` (defined BY a size filter). The test is
   structurally tautological for size-filtered strategies.
4. (WRONG) Pre-flight hook semantics (refuse-to-start) chosen for an
   informational-only stamp (post-promotion update). Wrong layer.
5. (UNVERIFIED) The Feb 2026 cross-instrument finding has not been
   re-tested on post-Phase-3c canonical bars (Phase 3c rebuild merged
   2026-04-08, commit ``c33805b``, per ``pre_registered_criteria.md``
   Amendment 2.8). The premise the stamp would be based on may be stale.

Per the institutional rule (CLAUDE.md "Audit-First Default for Research
Layers"): **audit -> adversarial audit -> fix -> rerun -> freeze -> move
on**. Do not patch unverified truth-state. This document is the audit step;
it does not propose any implementation until a restress confirms the
premise is still load-bearing.

## What we are NOT doing in this plan

- No schema changes to ``validated_setups``.
- No new columns (``stress_test_status``, ``size_stress_passed``, etc.).
- No validator hooks (pre-flight, post-promotion, or otherwise).
- No new drift checks.
- No promotion stamping.
- No edits to ``trading_app/strategy_validator.py`` or
  ``pipeline/check_drift.py``.
- No changes to ``RESEARCH_RULES.md`` § "Cross-Instrument Stress Test
  Finding (Feb 2026)" until the restress finishes.

This plan produces ONLY a result document with grounded findings.

## Research question (the actual one)

**Does the Feb 2026 cross-instrument finding (MES/MGC: all edges die
without size filter; MNQ CME_REOPEN survives) still hold under
post-Phase-3c canonical bars across all 9 sessions of all 3 active
instruments?**

This is the question that, if answered, would tell us whether ANY
size-stress-test stamp on validated_setups is justified. Answering it
first is required before any implementation discussion resumes.

## Methodology (canonical layers only)

### Inputs
- ``orb_outcomes`` (canonical outcome layer; post-Phase-3c real-micro bars)
- ``daily_features.orb_<session>_size`` (canonical size column per session,
  per ``pipeline.dst.SESSION_CATALOG``)
- ``pipeline.cost_model.COST_SPECS`` for friction floor (live query, no
  inline numerics)
- ``pipeline.dst.SESSION_CATALOG`` for session list (do NOT hardcode)
- ``pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS`` for instrument list

### Cohort
- Instruments: every entry in ``ACTIVE_ORB_INSTRUMENTS`` (currently MGC,
  MES, MNQ).
- Sessions: every dynamic session in ``SESSION_CATALOG`` for each
  instrument (currently 9 each, but query at run time).
- ORB minutes: 5, 15, 30 (canonical apertures).
- Holdout: Mode A -- ``trading_day < 2026-01-01``.
- Discovery scope respects ``trading_app.config.WF_START_OVERRIDE``
  per-instrument (Amendment 3.1 structural data boundary).

### Procedure (per instrument x session x orb_minutes cell)

1. Pull all `orb_outcomes` rows for the cell, IS-only (`trading_day <
   HOLDOUT_SACRED_FROM`).
2. Join `daily_features.orb_<session>_size` to bucket each trade as
   SMALL (`< friction_floor_in_points`) vs LARGE (`>=
   friction_floor_in_points`). The friction floor is computed live as
   `(cost_spec.commission_rt + cost_spec.spread_doubled +
   cost_spec.slippage) * cost_spec.tick_size / cost_spec.point_value` --
   this is the breakeven ORB size where the round-trip friction equals 1
   ATR-point. Do NOT hardcode "5pt" (the value `stress_test.py` uses) --
   it is MGC-friction-calibrated and wrong for MNQ/MES.
3. For each (instrument, session, orb_minutes, bucket) combination,
   compute: N, mean ExpR, win-rate, p-value (one-sample t vs zero on
   bucket-conditioned trade R-multiples), Bailey-LdP DSR informational
   score.
4. Three-way verdict per cell:
   - **LARGE-only edge:** SMALL bucket avg-R <= 0 AND LARGE bucket avg-R > 0
     AND SMALL N >= 30. Matches Feb 2026 "ALL edges die without size
     filter" framing.
   - **Robust:** Both buckets avg-R > 0 (with SMALL N >= 30). Edge does not
     depend on size filter.
   - **Inconclusive:** SMALL N < 30, or LARGE N < 30, or contradictory
     verdicts across orb_minutes for the same (instrument, session).

### Adversarial audit (mandatory)

Per institutional-rigor.md: every research output gets an adversarial pass
BEFORE freeze. The adversarial questions for this restress:

- Is the friction-floor-derived bucket boundary (vs Feb 2026's hardcoded
  5pt) actually informational? Re-run with both boundaries; report whether
  the verdict per cell is bucket-boundary-stable.
- Does the verdict change when filtering by entry model (E1 vs E2)?
- Does the verdict change when filtering by RR target?
- Are LARGE-only edges actually long-only or symmetric? (Direction-mix
  contamination question per `feedback_per_lane_breakdown_required.md`.)
- Is there a Look-ahead concern given the E2 break-bar registry
  (`feedback_e2_lookahead_drift_check_landed.md`)? Restrict E2 cells to
  the cleared subset only.
- Bonferroni correction across the (instr x session x orb_minutes) cell
  count (~9 sessions x 3 minutes x 3 instruments = 81 cells). Honest K.

### Output

A single result document at
`docs/audit/results/2026-05-XX-orb-size-restress-canonical.md` with
front-matter per `docs/audit/results/TEMPLATE-pooled-finding.md`:
- Per-cell breakdown table (mandatory for cross-instrument claims).
- Flip rate vs Feb 2026 framing.
- Heterogeneity verdict.
- Direction-mix breakdown (long vs short).
- Bonferroni-adjusted significance.
- Verdict per cell: LARGE-only, Robust, Inconclusive.

### Acceptance criteria for the research output

- `flip_rate_pct` field populated in front-matter.
- Every cell has a verdict (no missing).
- Every numeric claim cites the SQL or the canonical helper that produced
  it.
- Adversarial-audit section present and addresses the 6 questions above.
- DSR computed informationally per Amendment 2.1.
- Pre-registered before any SQL is run -- yaml at
  `docs/audit/hypotheses/2026-05-XX-orb-size-restress-v1.yaml`.

## What this enables (only after the result lands)

Iff the restress confirms the Feb 2026 framing AND the adversarial pass
holds AND the per-cell verdict pattern is structurally stable, THEN a
follow-up implementation plan would be appropriate covering:

1. Whether to add a schema column on `validated_setups` for the
   per-strategy verdict.
2. Where in the validator to compute it (post-FDR informational block,
   parallel to DSR).
3. Whether the verdict should affect deployment scope (`ACCOUNT_PROFILES`
   gating).
4. Pre_registered_criteria.md amendment to cite the new informational
   gate (informational, not blocking, mirroring Amendment 2.1).

If the restress shows the Feb 2026 framing has decayed (e.g. SMALL bucket
edges are now positive on post-Phase-3c bars), then NO implementation is
warranted and the existing `RESEARCH_RULES.md` text should be updated to
reflect the new finding.

If the restress is INCONCLUSIVE on most cells (low SMALL N for many
sessions), then we cannot ship a stamp; the right output is a data-gap
note, not validator code.

## What lands now (Stages 1-3 of validator-honesty-fix)

Independent of this research plan, Stages 1-3 of the validator-honesty-fix
have already landed:

- 1 MGC survivor identified, full Criterion 1-13 ladder generated, MinBTL
  retro-report computed (~6,625x strict Bailey violation, ~66x operational
  cap). Quarantine doc at
  `docs/audit/results/2026-05-04-mgc-chain-quarantine.md`.
- Validator FDR headline emits per-session K vector + caveat (line ~2272).
- Validator pre-flight refuses NEW promotions without prereg yaml
  (Criterion 1 gate). `--allow-legacy-prereg` flag for migration.
- Drift check #88 (advisory) reports prereg-missing across active
  instruments.
- Drift check #89 (advisory) reports per-session K drift between frozen
  `discovery_k` and live pool.

These do not depend on the size-filter restress. They are the structural
gates; the restress is the empirical answer to a separate question.

## Authority chain

- `RESEARCH_RULES.md:249-256` (Feb 2026 cross-instrument finding under
  re-verification).
- `pre_registered_criteria.md` Amendment 2.8 (Phase 3c canonical-bar
  rebuild, 2026-04-08).
- `pre_registered_criteria.md` Amendment 2.1 (DSR informational pattern
  any future stamp would mirror).
- `CLAUDE.md` "Audit-First Default for Research Layers".
- `feedback_e2_lookahead_drift_check_landed.md` (E2 break-bar registry to
  honor in adversarial audit).
- `feedback_per_lane_breakdown_required.md` (heterogeneity discipline).
- `docs/audit/results/TEMPLATE-pooled-finding.md` (output format).

## Owner / next action

- Author the pre-registration yaml at
  `docs/audit/hypotheses/2026-05-XX-orb-size-restress-v1.yaml` BEFORE any
  SQL.
- Run the cohort-build SQL on the canonical layer.
- Compute per-cell verdicts.
- Run adversarial audit pass.
- Land the result document.
- Decide on implementation only AFTER result lands.

This plan does NOT auto-execute. It is a research/audit charter. A future
session executes it if/when prioritised.
