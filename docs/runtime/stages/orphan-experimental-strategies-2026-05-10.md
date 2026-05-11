---
task: Triage 223,948 orphan experimental_strategies rows from 2026-05-10 unannounced discovery sweep
mode: DESIGN
slug: orphan-experimental-strategies-2026-05-10
created: 2026-05-12
updated: 2026-05-12
scope_lock:
  - gold.db (experimental_strategies table — read-only inspection only at this stage)
acceptance:
  - "Originating session/operator identified (git log + checkpoint inspection)."
  - "Disposition decided: (a) DELETE as scratch, (b) MOVE to quarantine table, or (c) WRITE retroactive 'exploration log' doc + leave rows in place with quarantine annotation."
  - "If (c): every retained row carries a column or sentinel marking it 'exploratory, unprereg-ed' so future Pathway-A discoveries do not silently inherit it."
  - "validator gate behavior on these rows is documented (currently: hard REJECT via Criterion 1)."
---

## Blast Radius

- **Read-only at this stage.** No DB writes, no code edits.
- 223,948 rows in `experimental_strategies` with `created_at` on 2026-05-10:
  - MNQ: 87,156 rows
  - MES: 76,141 rows
  - MGC: 60,651 rows
- Discovery surface: every (instrument, session, orb_minutes, entry_model, confirm_bars, rr_target, filter, stop_multiplier) combination, all 3 instruments, all entry models including E1/E2/E3, all sessions, all apertures. Brute-force enumeration scan.
- Downstream effect today: `strategy_validator._check_prereg_present` (post-commit `56f69f51`) refuses to promote any of these rows when `--instrument MES` or `--instrument MGC` is run, because no prereg yaml at `docs/audit/hypotheses/2026-05-10-*.yaml` declares MES or MGC in its `scope.instruments`. This is correct gate behavior — Criterion 1 violated.
- The 87,156 MNQ rows from 2026-05-10 may be partially-covered by a 2026-05-10 yaml that exists for MNQ only (not yet inspected). Out of scope for this stage; a separate audit on MNQ-only orphans is filed as P2 follow-up.

## Why this is a stage, not a quick fix

This was **not** my session. The 2026-05-10 sweep was run by another agent/operator before this session began. Per `.claude/rules/parallel-session-isolation.md` and `feedback_parallel_session_awareness.md`, I do not silently delete or modify another session's working data — including its discovery output — without explicit coordination.

The 223K orphan rows are also institutional-rigor debt: per Criterion 1 of `docs/institutional/pre_registered_criteria.md`, every discovery run that writes to `experimental_strategies` must have a pre-registered hypothesis file. This sweep had none. Retroactively writing a prereg is **forbidden** — the entire point of pre-registration per Bailey 2013 and Chordia 2018 is that the hypothesis is committed BEFORE the data is touched. Backdating defeats the bias defense.

## Three disposition options for the originating session

### (A) DELETE as scratch
Treat the 2026-05-10 sweep as exploratory enumeration whose findings cannot be promoted under any institutional protocol. Run:
```sql
DELETE FROM experimental_strategies WHERE CAST(created_at AS DATE) = '2026-05-10';
```
**Pros:** Clean state. Validator gate clears. Future Pathway-A runs uncluttered.
**Cons:** Destructive. Loses ~224K computed-and-cached rows. If the originator had legitimate downstream uses (e.g. correlation analysis across the universe), deletion forecloses them.

### (B) MOVE to quarantine table
Create `experimental_strategies_unprereg` mirror table. Move the 224K rows there. They remain inspectable for cross-strategy analysis but are invisible to the validator.
**Pros:** Non-destructive. Preserves data. Decouples from canonical promotion path.
**Cons:** New table = new schema = new drift checks needed. Maintenance cost.

### (C) WRITE exploration log + tag rows in place
Add a column `unprereg_exploration_id` (nullable) to `experimental_strategies`. Tag the 224K rows with a sentinel pointing at `docs/audit/exploratory/2026-05-10-fullsweep-exploration.md` describing what the originating session was actually trying to learn. Validator's prereg gate continues to reject them (correct), but they remain queryable for downstream exploratory analysis.
**Pros:** Preserves data + intent. Honest about what happened. Future auditors can trace.
**Cons:** Requires schema change to `experimental_strategies` (NEVER_TRIVIAL). Requires the originating session to write the exploration log honestly.

## Recommended path

(C) is most institutionally honest but requires schema work. (A) is the cleanest if the originator confirms the data has no use. (B) is the in-between.

**Decision belongs to the originating session, not this one.** This stage exists to surface the issue and document the options.

## Forbidden in this stage

- Writing retroactive prereg yamls for the 2026-05-10 sweep (defeats Criterion 1 — scientific fraud).
- Using `--allow-legacy-prereg` to push the 224K rows through the validator (legacy-prereg is for one-time migrations, not routine bypass of unannounced sweeps).
- Modifying or deleting `experimental_strategies` rows in this session — those writes belong to the originating session.

## Procedure

1. (Out of scope for this stage) Originating session/operator identifies themselves and picks (A) / (B) / (C).
2. (Out of scope) They execute their chosen disposition.
3. (Out of scope) Verification: re-run validator without `--allow-legacy-prereg` and confirm gate behavior matches the chosen disposition.
4. Update this stage doc with decision and link to the executing commit.

## Cross-references

- Auditor finding: `docs/audit/results/2026-05-12-llm-prereg-trio-audit.md` § "Critical issues" #3.
- Validator gate: `trading_app/strategy_validator.py:_check_prereg_present` (post `57ef3e23`).
- Authority: `docs/institutional/pre_registered_criteria.md` Criterion 1 (pre-registration mandatory).
- Methodology: Bailey et al 2013 MinBTL bound; Chordia et al 2018 t≥3.79 + pre-reg discipline.
