# G10 — Pre-Reg Commit-SHA Pin Certificate

**Candidate:** ________________________
**Pre-reg:** `docs/audit/hypotheses/________.yaml`

---

## Purpose

Every pre-reg that cites `docs/institutional/pre_registered_criteria.md` must pin a specific commit SHA of that file. Prevents drift where an amendment changes the criteria AFTER the pre-reg is locked but BEFORE it is evaluated.

## Required fields

| Field | Value | Source |
|---|---|---|
| Pre-reg yaml path | | |
| Pre-reg yaml commit SHA at lock | | `git log -1 --format=%H <pre-reg>` |
| `pre_registered_criteria.md` commit SHA pinned IN the pre-reg | | pre-reg yaml's `pre_reg_criteria_commit_sha` field |
| `pre_registered_criteria.md` commit SHA at evaluation time | | `git log -1 --format=%H docs/institutional/pre_registered_criteria.md` |

## Drift check

If pinned commit SHA differs from eval-time commit SHA, verify:

```
$ git diff <pinned-sha> <eval-sha> -- docs/institutional/pre_registered_criteria.md
<paste output>
```

- [ ] Diff is empty (no drift) OR
- [ ] Diff adds new amendments that STRENGTHEN but do not relax the candidate's relevant criteria OR
- [ ] Diff includes amendments that would CHANGE the candidate's evaluation — in which case evaluation is either (a) redone under new criteria, or (b) rolled back to pinned criteria with explicit rationale

## Amendment compliance

Check that the pre-reg explicitly cites the latest applicable amendments:

- [ ] Amendment 2.1 (DSR cross-check) if DSR is a candidate criterion
- [ ] Amendment 2.2 (Chordia t-bands) if t-threshold is a candidate criterion
- [ ] Amendment 2.4 (banded deployability) if deployment path is considered
- [ ] Amendment 2.6 (Mode B holdout decision) if Mode B status is referenced
- [ ] Amendment 2.7 (--holdout-date enforcement) if 2026 OOS is cited
- [ ] Amendment 3.0 (Pathway A/B) if K_family is declared
- [ ] Amendment 3.1 (proxy data policy) if GC/NQ/ES proxy is used

## Verdict

- [ ] CLEAR — commit SHA pinned, drift check passes or diff is harmless
- [ ] FAIL — commit SHA unpinned OR drift diff materially changes the candidate's evaluation

## Failure disposition

FAIL-UNPINNED → pre-reg must be amended to pin the commit SHA. Evaluation results stand pending pin addition.

FAIL-DRIFT → either (a) re-evaluate under new criteria, or (b) roll back to pinned criteria with explicit rationale. Document in decision doc.

## Literature citation

- `docs/institutional/pre_registered_criteria.md` (Amendment 2.6 onwards each includes its own commit-pin guidance)
- Drift check #45 in `pipeline/check_drift.py` (enforces `@research-source` / `@entry-models` / `@revalidated-for` on research-derived config values)

## Authored by / committed

- Author: ____________________________
- Commit SHA of this certificate: ________________
