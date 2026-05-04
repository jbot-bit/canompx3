---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Asset Universe Pipeline Audit Design

**Date:** 2026-04-20
**Status:** In progress
**Purpose:** Replace fuzzy "what assets should we test?" discussion with a verified asset-universe triage that distinguishes configured assets, canonically rebuildable assets, partial legacy artifacts, and assets worthy of full institutional rigor.

---

## 1. Why this exists

The repo has accumulated several different "asset universes":

- configured assets in `pipeline/asset_configs.py`
- active ORB assets in `ACTIVE_ORB_INSTRUMENTS`
- deployable assets in `DEPLOYABLE_ORB_INSTRUMENTS`
- partially built legacy artifacts in `gold.db`
- diversification candidates mentioned in design docs

Those are **not the same thing**.

This design enforces a strict sequence:

1. verify raw-store integrity
2. verify canonical build depth in `gold.db`
3. verify policy eligibility
4. verify mechanism-specific research plan
5. only then spend discovery / validation budget

This is the only honest way to answer:

- which assets are actually good enough to go through the current system
- which older assets deserve re-checking under the updated framework
- which assets should be left alone until infrastructure or theory changes

---

## 2. Canonical sources

This design is grounded in:

- `docs/plans/PIPELINE_VERIFICATION_MASTER.md`
- `docs/plans/2026-04-07-canonical-data-redownload.md`
- `docs/plans/2026-04-09-phase4-hypothesis-redesign-findings.md`
- `docs/institutional/pre_registered_criteria.md`
- `docs/plans/diversification-candidate-shortlist.md`
- `pipeline/asset_configs.py`
- direct `gold.db` queries on `bars_1m`, `bars_5m`, `daily_features`, `orb_outcomes`, `validated_setups`
- direct filesystem inspection of configured raw DBN stores

Per `PIPELINE_VERIFICATION_MASTER.md`, downstream tables are not allowed to override canonical raw/build truth. A configured asset is not "available" unless its raw store and canonical build path both verify.

---

## 3. Locked audit gates

### Gate 0. Raw-store integrity

An asset is not rebuildable unless:

- its configured `dbn_path` exists
- and the store contains at least one `.dbn.zst` file matching the asset's canonical outright root

Important nuance:

- matching must use the outright root, not the instrument name
- this is required because some micro configs intentionally use parent-data roots:
  - `M2K -> RTY`
  - `MBT -> BTC`
  - `M6E -> 6E`
  - `MCL -> CL`
  - `SIL -> SI`

This gate explicitly rejects false positives where a directory exists but only contains another asset's files.

### Gate 1. Canonical build depth

For each asset, verify actual presence in:

- `bars_1m`
- `bars_5m` when used by the research family
- `daily_features`
- `orb_outcomes`
- `validated_setups`

This separates:

- raw-only research candidates
- full ORB-pipeline assets
- partially built legacy assets

### Gate 2. Policy eligibility

After Gates 0-1 pass, classify under current repo policy:

- `ACTIVE_ORB_INSTRUMENTS`
- `DEPLOYABLE_ORB_INSTRUMENTS`
- `DEAD_ORB_INSTRUMENTS`
- proxy / parent special cases from `pre_registered_criteria.md`

Dead ORB instruments stay dead for the same ORB thesis. Existing tables do not override dead status.

### Gate 3. Mechanism eligibility

Even a rebuildable asset is not eligible for full rigor unless it has:

- a narrow mechanism-first research question
- a pre-registered hypothesis budget appropriate to its horizon
- a valid execution path for the family being tested

Examples:

- `ZT` passes Gates 0-1 but currently fails Gate 3 for the tested simple directional macro-event family
- `2YY` passes Gate 0 but fails practical data-quality for the narrow `08:30 ET` event study
- `MZC`/`MZS` may become eligible only after a locked Stage-1 spec exists

---

## 4. Verified current-state matrix

Verified directly from `gold.db` and the filesystem on 2026-04-20:

| Asset | Raw store verified | `bars_1m` | `daily_features` | `orb_outcomes` | `validated_setups` | Current policy read |
|---|---:|---:|---:|---:|---:|---|
| `MNQ` | yes | yes | yes | yes | 42 total / 36 active | full active + deployable ORB asset |
| `MES` | yes | yes | yes | yes | 2 total / 2 active | full active + deployable ORB asset |
| `MGC` | yes | yes | yes | yes | 0 current active rows | full active ORB research asset, non-deployable by horizon |
| `2YY` | yes | yes | no | no | 0 | research-only raw candidate, current narrow event family no-go |
| `ZT` | yes | yes | no | no | 0 | research-only raw candidate, current simple directional family no-go |
| `GC` | **no** | yes | yes | yes | 17 retired / 0 active | proxy-support asset, but configured raw store is not honest today |
| `ES` | **no** | no | no | no | 0 | parent-support config exists, but raw store is not honest today |
| `NQ` | no | no | no | no | 0 | parent-support config incomplete today |
| `M2K` | no | yes | yes | yes | 0 | dead ORB legacy artifact |
| `M6E` | no | yes | yes | yes | 0 | dead ORB legacy artifact |
| `MBT` | no | yes | yes | yes | 0 | dead ORB legacy artifact |
| `MCL` | no | yes | no | no | 0 | dead ORB legacy artifact |
| `SIL` | no | yes | yes | partial | 0 | dead ORB legacy artifact |

Key verified facts behind the table:

- `MNQ`, `MES`, `MGC` have end-to-end ORB layers present and current raw stores under `data/raw/databento/ohlcv-1m/...`
- `2YY` and `ZT` have direct full-history raw DBN files plus fresh `bars_1m` / `bars_5m`, but no ORB build layers
- `GC` and `ES` looked rebuildable by path existence alone, but direct file inspection showed:
  - `DB/GOLD_DB_FULLSIZE` currently contains only `MGC` backfill files
  - `DB/MES_DB_2019-2024` currently contains only `MES` backfill files
- `NQ` raw path is missing entirely
- dead ORB assets still have legacy tables in `gold.db`, but their configured raw stores are absent

---

## 5. Decision buckets

### Bucket A. Full rigor now

These assets are good enough to spend current institutional ORB rigor on:

- `MNQ`
- `MES`
- `MGC` as research-only, with non-deployable horizon disclosure

Why:

- raw stores verify
- canonical ORB layers exist
- policy admits them into the active pipeline

### Bucket B. Research-only, but not ORB-rigor now

- `2YY`
- `ZT`

Why:

- raw arrival is honest
- current tested mechanism families failed or are structurally invalid
- no locked replacement Stage-1 spec exists yet

Action:

- do not build generic ORB layers
- only proceed if a new narrow mechanism spec is written first

### Bucket C. Support/proxy assets requiring infrastructure cleanup before use

- `GC`
- `ES`
- `NQ`

Why:

- these are still meaningful to parent/proxy policy
- but the configured raw-store truth is not clean enough today to call them canonically rebuildable

Action:

1. decide whether these parent configs remain in-scope
2. if yes, rehydrate the correct raw stores
3. if no, retire or mark them explicitly non-rebuildable so orchestration stops lying

### Bucket D. Dead ORB legacy assets

- `M2K`
- `M6E`
- `MBT`
- `MCL`
- `SIL`

Why:

- policy already says dead ORB
- current raw stores are absent
- existing tables are legacy artifacts, not permission to re-open the thesis

Action:

- do not re-run ORB discovery on these
- only revisit under a materially different mechanism and with raw-store restoration first

---

## 6. Execution order

The queue from here is:

1. **Truth-surface hardening**
   - fail closed when a configured raw store exists but contains the wrong symbol
   - stop rebuild orchestration from treating `GC/ES` as available by directory existence alone
2. **Asset classification audit output**
   - produce a verified matrix artifact
   - mark each asset as `BUILD_NOW`, `HOLD`, `SUPPORT_ONLY`, or `NO_GO`
3. **Parent/proxy cleanup decision**
   - resolve `GC/ES/NQ` as either:
     - correctly rebuilt parent-support assets
     - or explicitly retired / non-rebuildable configs
4. **New-candidate onboarding**
   - only after a locked Stage-1 mechanism spec exists
   - likely agriculture before more rates, per the current shortlist

---

## 7. Immediate implementation implications

This design already implies one engineering rule:

- `require_dbn_available()` and `list_available_instruments()` must not use raw directory existence as a proxy for actual availability

If they do, the system can schedule rebuild work on assets whose configured store contains the wrong symbol, which is a fail-open infrastructure bug.

---

## 8. Success condition

This audit is complete when:

- every configured asset has a verified classification
- no helper reports an asset as rebuildable unless its raw store actually matches the canonical root
- the next research queue is based on verified eligibility, not stale tables or remembered narratives

