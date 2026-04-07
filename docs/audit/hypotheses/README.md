# Pre-Registered Hypothesis Registry

**Purpose:** This directory holds locked, pre-registered hypothesis files for every strategy-discovery run. The existence of a file in this directory BEFORE a discovery run is **Criterion 1** of [`docs/institutional/pre_registered_criteria.md`](../../institutional/pre_registered_criteria.md) (v2, locked 2026-04-07).

**This file exists to close the rule-without-infrastructure gap.** Prior to 2026-04-07, the research-truth-protocol and pre_registered_criteria files referenced this directory but the directory did not exist. Creating it without actual hypothesis content keeps the infrastructure honest while avoiding biased seeding of research direction.

---

## Workflow

1. **Read** [`docs/institutional/pre_registered_criteria.md`](../../institutional/pre_registered_criteria.md) — the 12 locked criteria any strategy must meet. Note the v2 amendments (DSR as cross-check, Chordia banded, Criterion 8 contingent on holdout policy).
2. **Read** [`docs/institutional/finite_data_framework.md`](../../institutional/finite_data_framework.md) — the short-sample methodology.
3. **Declare holdout policy** — Mode A (holdout-clean, 2026 excluded from discovery) or Mode B (post-holdout-monitoring, 2026 consumed). Amendment 2.3 bans mixing.
4. **Compute the MinBTL bound** — `MinBTL = 2·Ln[N] / E[max_N]²` with N = total pre-registered trials. If `MinBTL > available_clean_data_years`, reduce N. No exceptions.
5. **Copy the template** — `cp docs/institutional/hypothesis_registry_template.md docs/audit/hypotheses/YYYY-MM-DD-<slug>.yaml` and fill in the specifics.
6. **Commit the hypothesis file BEFORE any backtest code runs.** The pre-commit hook captures the committing SHA as the lock point.
7. **Run discovery** with `--hypothesis-file` pointing at the committed file. (The `--hypothesis-file` CLI arg is not yet wired into `strategy_discovery.py`; see `docs/plans/2026-04-07-canonical-data-redownload.md` § Phase 4a for the follow-up task.)
8. **Report results against the pre-registered kill criteria** — no retroactive broadening of the family.

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

### Required

- **Theory before data.** LdP 2020 Lesson 1 (`literature/lopez_de_prado_2020_ml_for_asset_managers.md` § 1.2.1): *"Contrary to popular belief, backtesting is not a research tool. Backtests can never prove that a strategy is a true positive."*
- **Kill criteria pre-committed.** The hypothesis must specify what outcome would refute it, written before results are seen.
- **Provenance.** Every threshold in the file must reference either a literature extract in [`docs/institutional/literature/`](../../institutional/literature/) or a prior validated finding in this repo (with commit SHA).

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
