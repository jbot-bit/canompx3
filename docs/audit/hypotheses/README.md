# Pre-Registered Hypothesis Registry

**Purpose:** This directory holds locked, pre-registered hypothesis files for every strategy-discovery run. The existence of a file in this directory BEFORE a discovery run is **Criterion 1** of [`docs/institutional/pre_registered_criteria.md`](../../institutional/pre_registered_criteria.md) (v2, locked 2026-04-07).

**This file exists to close the rule-without-infrastructure gap.** Prior to 2026-04-07, the research-truth-protocol and pre_registered_criteria files referenced this directory but the directory did not exist. Creating it without actual hypothesis content keeps the infrastructure honest while avoiding biased seeding of research direction.

---

## Workflow

1. **Read** [`docs/institutional/pre_registered_criteria.md`](../../institutional/pre_registered_criteria.md) — the 12 locked criteria any strategy must meet. Note the v2 amendments (DSR as cross-check, Chordia banded, Criterion 8 contingent on holdout policy).
2. **Read** [`docs/institutional/finite_data_framework.md`](../../institutional/finite_data_framework.md) — the short-sample methodology.
3. **Declare holdout policy** — **Mode A (holdout-clean) is operative project-wide as of 2026-04-08 per Amendment 2.7.** Sacred window is 2026-01-01 onwards. Every hypothesis file must include `holdout_date: 2026-01-01` (or earlier) in its metadata. Mode B was briefly declared 2026-04-07 and rescinded the next day; see `../../plans/2026-04-07-holdout-policy-decision.md` for the audit trail.
4. **Compute the MinBTL bound** — `MinBTL = 2·Ln[N] / E[max_N]²` with N = total pre-registered trials. If `MinBTL > available_clean_data_years`, reduce N. No exceptions.
5. **Copy the template** — `cp docs/institutional/hypothesis_registry_template.md docs/audit/hypotheses/YYYY-MM-DD-<slug>.yaml` and fill in the specifics.
6. **Commit the hypothesis file BEFORE any backtest code runs.** The pre-commit hook captures the committing SHA as the lock point.
7. **Inspect the route first** with the prereg front door:
   - `scripts/infra/prereg-loop.sh --hypothesis-file docs/audit/hypotheses/YYYY-MM-DD-<slug>.yaml`
   - This tells you whether the prereg is a `grid_discovery` object that writes to `experimental_strategies` or a `bounded_runner` object that stays in a dedicated result-doc flow.
8. **Execute the prereg on the correct branch**:
   - `standalone_edge` / `grid_discovery`:
     - `scripts/infra/prereg-loop.sh --hypothesis-file docs/audit/hypotheses/YYYY-MM-DD-<slug>.yaml --execute`
     - Output lands in `experimental_strategies`, then must go through `trading_app/strategy_validator.py` before anything reaches `validated_setups`.
     - For multi-instrument or multi-aperture preregs, execute once per
       disjoint slice using `--instrument` and/or `--orb-minutes`; the
       single-use gate is scoped by hypothesis SHA + instrument + aperture.
   - `conditional_role` / `bounded_runner`:
     - use the prereg front door with `--runner ...` or add an `execution.entrypoint` block to the prereg
     - output lands in a bounded result artifact, not in `experimental_strategies`
9. **Report results against the pre-registered kill criteria** — no retroactive broadening of the family.

---

## Naming convention

```
YYYY-MM-DD-<slug>.yaml
```

Examples (hypothetical — do NOT treat as real registered hypotheses):
- `2026-05-10-mnq-post-redownload-price-filters.yaml`
- `2026-06-03-mes-overnight-range-revalidation.yaml`

One file per discovery run. Never amend a locked file — supersede it with a new dated file and update the `supersedes` field in the new file's metadata block.

---

## What goes in the file

See [`docs/institutional/hypothesis_registry_template.md`](../../institutional/hypothesis_registry_template.md) for the full YAML schema. Minimum required fields:

- **metadata.name** — short slug
- **metadata.date_locked** — ISO timestamp
- **metadata.holdout_date** — commitment to a clean-end date OR declaration of post-holdout-monitoring mode
- **metadata.total_expected_trials** — sum across all hypotheses; **must be ≤ 300 on clean MNQ data OR ≤ 2000 on proxy-extended data** per Criterion 2 (MinBTL bound)
- **hypotheses[]** — numbered list, each with:
  - `theory_citation` (LdP 2020 theory-first requirement — no hypothesis without economic theory)
  - `economic_basis` (one paragraph explaining WHY you expect this edge)
  - `filter.type` + `filter.column` + `filter.thresholds` (exact)
  - `scope` (instruments, sessions, rr_targets, entry_models, confirm_bars, stop_multipliers)
  - `expected_trial_count` per hypothesis
  - `kill_criteria` (what result would REFUTE the hypothesis — pre-committed, not post-hoc)

---

## Rules

### Forbidden

- **Running discovery code without a committed hypothesis file in this directory.** Criterion 1 of pre_registered_criteria.md.
- **Brute-force enumeration > 300 trials on clean MNQ data** (Criterion 2 MinBTL bound).
- **Amending a locked file to rescue a failing hypothesis.** Supersede only.
- **Citing the template or another hypothesis file as justification.** Each file must cite external literature.
- **Mixing holdout modes.** Amendment 2.3 — declare A or B, commit to one.
- **Treating every prereg as a grid-discovery object.** `conditional_role` and other bounded studies do not belong in `experimental_strategies` unless they are explicitly designed for that path.

### Required

- **Theory before data.** LdP 2020 Lesson 1 (`literature/lopez_de_prado_2020_ml_for_asset_managers.md` § 1.2.1): *"Contrary to popular belief, backtesting is not a research tool. Backtests can never prove that a strategy is a true positive."*
- **Kill criteria pre-committed.** The hypothesis must specify what outcome would refute it, written before results are seen.
- **Provenance.** Every threshold in the file must reference either a literature extract in [`docs/institutional/literature/`](../../institutional/literature/) or a prior validated finding in this repo (with commit SHA).
- **Pipeline branch must be explicit.** A prereg must make it obvious whether it routes to `experimental_strategies` (`standalone_edge`) or to a bounded result-doc runner (`conditional_role` / other non-grid studies).

---

## Status (2026-04-07)

**Directory created:** 2026-04-07.
**Files committed:** 0 (this README only — infrastructure, not a hypothesis).
**Next expected use:** Phase 4 of [`docs/plans/2026-04-07-canonical-data-redownload.md`](../../plans/2026-04-07-canonical-data-redownload.md), after Phase 2 (redownload) and Phase 3 (era schema) complete. Both are currently blocked on the `e2-canonical-window-fix` worktree merging.

---

## Related files

- [`docs/institutional/pre_registered_criteria.md`](../../institutional/pre_registered_criteria.md) — the 12 criteria (v2 with 5 amendments)
- [`docs/institutional/finite_data_framework.md`](../../institutional/finite_data_framework.md) — short-sample methodology
- [`docs/institutional/hypothesis_registry_template.md`](../../institutional/hypothesis_registry_template.md) — YAML schema
- [`docs/institutional/literature/`](../../institutional/literature/) — 7 verbatim paper extracts
- [`.claude/rules/research-truth-protocol.md`](../../../.claude/rules/research-truth-protocol.md) — Phase 0 Literature Grounding section
- [`.claude/rules/institutional-rigor.md`](../../../.claude/rules/institutional-rigor.md) — rule 7: ground in local resources
- [`docs/audits/2026-04-07-finite-data-orb-audit.md`](../../audits/2026-04-07-finite-data-orb-audit.md) — Codex audit that produced the v2 amendments (note the `audits/` plural — pre-existing directory, separate from this `audit/` singular directory which was created per the rule path in `pre_registered_criteria.md`)
