---
paths:
  - "docs/audit/results/**/*.md"
  - "docs/audit/hypotheses/**/*.md"
  - "docs/institutional/**/*.md"
  - "research/**/*.py"
---

# Pooled-finding rule — per-cell breakdown is mandatory

**Authority:** governs every new audit-result file under `docs/audit/results/` dated 2026-04-20 or later. Enforced by `pipeline/check_drift.py::check_pooled_finding_annotations`.

**Origin:** RULE 14 — retroactive heterogeneity audit committed in aa3399b3 (2026-04-20). Pooled p-values and pooled ExpR averages hid opposite-sign per-cell behaviour in three historical audits (`bull-short-avoidance`, `garch-vol-pct70`, `rel-vol-HIGH-Q3`). Every pooled claim survived BH-global at K_global but ≥25% of its underlying cells flipped sign. Memory files quoted the pooled framing and one of them queued a trading decision on it. The check closes that class of silent artefact.

## Scope

Applies to any new file written to `docs/audit/results/*.md` whose filename starts with a date on or after `2026-04-20`. Historical files modified after the sentinel opt into the schema — if you edit a pre-sentinel file, you adopt the requirement.

Template at `docs/audit/results/TEMPLATE-pooled-finding.md`.

## Required front-matter

Any audit-result file making a pooled-universe claim MUST carry the following YAML front-matter at the top of the file (between `---` delimiters):

```yaml
---
pooled_finding: true
per_cell_breakdown_path: docs/audit/results/<adjacent-or-companion-file>.md
flip_rate_pct: <number between 0 and 100>
heterogeneity_ack: <true if flip_rate_pct >= 25 else omit>
---
```

- `pooled_finding: true` — declares the file makes a pooled-universe claim (p-value or ExpR averaged across lanes, sessions, or instruments).
- `per_cell_breakdown_path` — repo-relative path to the accompanying per-cell breakdown. The breakdown must enumerate each cell with its individual effect size and sign. If the breakdown is an appendix in the same file, point at the file itself with an anchor.
- `flip_rate_pct` — the share of cells whose per-cell sign opposes the pooled sign. Computed as `flips / total_cells`, reported as a percentage. Zero if every cell agrees with the pooled direction.
- `heterogeneity_ack: true` — required only when `flip_rate_pct >= 25`. Acknowledges that the pooled framing is misleading in isolation and that any trading or research decision made on the claim must examine the cell-level distribution, not the pooled number.

A file that does not make a pooled claim simply omits `pooled_finding` — the check does not apply.

## What counts as a pooled claim

Any of:
- A p-value computed on a dataset combining multiple `(instrument, session, orb_minutes, rr_target, entry_model, confirm_bars, filter_type, direction)` tuples.
- An ExpR average across lanes.
- A BH-global survivor list with K derived from a combined grid.
- A universality claim ("X works across all sessions") supported by an aggregate-level test rather than a per-lane sequence of tests.

A single-lane Pathway-B K=1 confirmatory test is NOT a pooled claim.

## The 25% threshold

Pinned to the 2026-04-20 retroactive audit finding. Cells where per-cell sign flipped relative to pooled sign are counted. The threshold is pragmatic — below 25%, a pooled framing is still useful as a headline; above 25%, the pooled framing actively misleads.

Revision requires an amendment commit to this rule and `pre_registered_criteria.md`.

## What this rule forbids

- Writing "universal" or "across all lanes" framings without the per-cell table.
- Queuing a trading decision on a pooled p-value without lane-specific Pathway-B verification first.
- Memory-file bullets that quote a pooled headline without a flip-rate breakdown.
- Passing a pooled finding forward in doctrine (`docs/institutional/*`) without the per-cell evidence inline.
