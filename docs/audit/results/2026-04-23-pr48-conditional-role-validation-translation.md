# PR48 conditional-role validation translation

**Date:** 2026-04-23  
**Stage:** `docs/runtime/stages/pr48-conditional-role-validation-translation.md`
**Scope:** translate the recovered PR48 shortlist into honest current repo objects without pretending conditional-role results are already native validated lanes or allocator inputs.

## Grounding

- Canonical research truth already reproduced on current `main`:
  - `docs/audit/results/2026-04-23-pr48-conditional-edge-recovery-audit.md`
  - `docs/audit/results/2026-04-22-pr48-conditional-role-implementation-v1.md`
  - `docs/audit/results/2026-04-22-pr48-role-followthrough-v1.md`
  - `docs/audit/results/2026-04-22-pr48-promotion-shortlist-v1.md`
  - `docs/audit/results/2026-04-23-pr48-mes-mgc-sizer-rule-backtest-v1.md`
- Runtime / allocator surfaces inspected:
  - `trading_app/strategy_validator.py`
  - `trading_app/validated_shelf.py`
  - `trading_app/lane_allocator.py`
  - `trading_app/prop_profiles.py`
  - `docs/runtime/lane_allocation.json`

## Scope / Question

What is the honest intermediate object for each recovered PR48 shortlist arm in the current repo:

- `MES:q45_exec`
- `MGC:cont_exec`
- `DUO:mes_q45_plus_mgc_cont_exec`
- `MNQ:shadow_addon`

And which of those arms are:

- `SHADOW_ONLY`
- `NEEDS_BRIDGE`
- `READY_FOR_BOUNDED_TRANSLATION`
- `DEAD`

## Repo fit

### What the repo can do natively today

- Promote standalone validated lanes into `validated_setups` via `trading_app/strategy_validator.py`.
- Filter deployable shelf rows by `deployment_scope` via `trading_app/validated_shelf.py`.
- Rank and select deployable standalone lanes via `trading_app/lane_allocator.py`.
- Load concrete `strategy_id` lanes into profiles via `trading_app/prop_profiles.py` and `docs/runtime/lane_allocation.json`.

### What the repo cannot do natively today

- Store a conditional-role object with a native `role` field (`filter` / `allocator` / `confluence` / `shadow`).
- Feed a conditional-role result doc directly into the allocator.
- Carry a cross-session or cross-instrument sizing map as a first-class deployable object.
- Represent a duo / combo object as a shelf row.

## Translation table

| candidate | measured role truth | parent comparator | current live-book interaction | smallest honest bridge | verdict |
|---|---|---|---|---|---|
| `MES:q45_exec` | executable filter arm with strong IS/OOS daily dollar delta, but role metrics remain near-flat/negative rather than clean positive standalone expectancy | all canonical `MES O5 E2 CB1 RR1.5` trades across active sessions and both directions | not consumable by current active book; no active MES allocator route and no native conditional filter object | bounded report-only conditional candidate surface plus a profile-local filter sleeve spec if pursued | `NEEDS_BRIDGE` |
| `MGC:cont_exec` | strongest allocator arm; shortlist survives and frozen-rule replay now stamps `SIZER_DEPLOY_CANDIDATE` | all canonical `MGC O5 E2 CB1 RR1.5` trades across active sessions and both directions | not consumable natively, but there is an existing inactive MGC conditional profile surface (`topstep_50k`) where a bounded translation can attach | one narrow profile-local conditional sizer bridge, frozen and report-backed, without promoting into `validated_setups` | `READY_FOR_BOUNDED_TRANSLATION` |
| `DUO:mes_q45_plus_mgc_cont_exec` | portfolio combination object, not a lane | combined MES parent + MGC parent daily dollar book | no native allocator shape for a duo object; cannot be expressed as a lane or shelf row | report-only portfolio sleeve / shadow composite only | `SHADOW_ONLY` |
| `MNQ:shadow_addon` | explicitly a shadow add-on versus the duo, not a standalone promotion object | duo candidate book plus MNQ continuous executable shadow arm | current live MNQ auto book consumes standalone lanes only; this add-on does not map to the active book’s object model | shadow monitor only | `SHADOW_ONLY` |

## Candidate-by-candidate notes

### `MES:q45_exec`

- Measured shortlist delta is strong:
  - IS mean daily delta `$+17.98`
  - OOS sign `+`
- But role metrics stay too weak to call this a clean deployable filter object:
  - IS policy EV per opportunity `-0.0055R`
  - OOS policy EV per opportunity `-0.0047R`
- There is an inactive MES profile surface in `prop_profiles.py`, but it expects a concrete standalone `strategy_id`, not a cross-session conditional filter arm.

**Conclusion:** alive, but not a native promotion candidate and not the first bridge to build.

### `MGC:cont_exec`

- Measured shortlist delta is positive:
  - IS mean daily delta `$+15.29`
  - OOS sign `+`
- The later frozen-rule replay tightened the shape:
  - `docs/audit/results/2026-04-23-pr48-mes-mgc-sizer-rule-backtest-v1.md`
  - verdict: `SIZER_DEPLOY_CANDIDATE`
- There is already an inactive MGC conditional profile surface in `prop_profiles.py`:
  - `topstep_50k`
  - instrument-limited to `MGC`
  - explicitly conditional / shadow-oriented notes already exist

**Conclusion:** this is the only PR48 arm close enough to justify a bounded translation stage now.

### `DUO:mes_q45_plus_mgc_cont_exec`

- The duo beats its raw parent duo on daily dollars in the shortlist.
- But this is a portfolio object, not a strategy row.
- Current allocator code consumes deployable shelf rows, not duo overlays assembled from conditional-role arms.

**Conclusion:** keep as a report-only composite benchmark, not a promotion object.

### `MNQ:shadow_addon`

- The shortlist shows positive daily dollar delta and positive OOS sign.
- But the object is explicitly named and tested as a `shadow_addon`.
- Its parent comparator is the duo candidate, not the active MNQ auto book.

**Conclusion:** keep as shadow-only. Do not route this into the active MNQ allocator story.

## Verdict / Decision

**Decision:** `NARROW`

- `MES:q45_exec` → `NEEDS_BRIDGE`
- `MGC:cont_exec` → `READY_FOR_BOUNDED_TRANSLATION`
- `DUO:mes_q45_plus_mgc_cont_exec` → `SHADOW_ONLY`
- `MNQ:shadow_addon` → `SHADOW_ONLY`

The repo still cannot consume conditional-role outputs natively. The strongest next move is not a broad schema rebuild and not a fresh confluence scan. It is one narrow translation bridge for `MGC:cont_exec`.

## Recommended next step

Use `docs/runtime/stages/pr48-mgc-cont-exec-bounded-translation.md`.

That stage should answer one question only:

> can the frozen `MGC:cont_exec` size map be carried into a bounded profile-local translation surface without pretending it is a standalone validated lane?

## Caveats / Limitations

- This translation audit does not rerun discovery. It relies on already-reproduced canonical result docs plus current runtime inspection.
- `lane_allocation.json` is operational context, not research proof.
- OOS remains thin and is a monitor, not a tuning surface.
- This audit does not authorize live deployment by itself.

## Reproduction

- Read:
  - `docs/runtime/stages/pr48-conditional-role-validation-translation.md`
  - `docs/audit/results/2026-04-23-pr48-conditional-edge-recovery-audit.md`
  - `docs/audit/results/2026-04-22-pr48-promotion-shortlist-v1.md`
  - `docs/audit/results/2026-04-23-pr48-mes-mgc-sizer-rule-backtest-v1.md`
- Inspect:
  - `trading_app/strategy_validator.py`
  - `trading_app/validated_shelf.py`
  - `trading_app/lane_allocator.py`
  - `trading_app/prop_profiles.py`
  - `docs/runtime/lane_allocation.json`
