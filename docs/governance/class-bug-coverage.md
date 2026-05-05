# Class-bug families and per-class defense status

Status: 2026-05-06. Owner: integrity-guardian. Authority: this doc enumerates
all currently-active class-bug families and the specific defense (static
drift-check or workflow-rule) covering each. New families surfaced in
`memory/feedback_*.md` MUST be classified at discovery time per the rule at
the bottom of this file.

## Active families

| Family | Source | Defense | Defense type |
|---|---|---|---|
| `orb_minutes=5` literal in scoring path | recurrence chain PR #189 → #231/#232/#233 → #235 (allocator-class fail-quiet bug) | Drift check `check_aperture_hardcode_in_scoring_paths` (#12) | static |
| `pnl_r IS NOT NULL` silent scratch drop | `feedback_scratch_pnl_null_class_bug.md` | Drift check `check_research_scratch_policy_annotation` — requires `# scratch-policy: drop\|include-as-zero\|realized-eod` annotation grammar | static |
| E2 break-bar look-ahead | E2 LA registry (24 TAINTED + 1 FIXED + 2 CLEARED) at `docs/audit/results/2026-04-28-e2-lookahead-contamination-registry.md` | Drift check `check_e2_lookahead_research_contamination` — requires `# e2-lookahead-policy:` annotation on E2 + break-bar feature combinations | static |
| `iso_utc` silent-None formatter | `feedback_iso_utc_silent_none_class_pattern.md` | Drift check `check_iso_utc_formatter_silent_none` (this stage, 2026-05-06) — AST predicate: isinstance + explicit `return None` tail + no `log.warning\|critical\|error` + no `raise`. Annotation exemption: `# silent-none-policy: <reason>` | static |
| Stale PR merge revert | `feedback_stale_pr_merge_revert_class_2026_05_05.md` | `.claude/rules/branch-discipline.md` + `.claude/rules/adversarial-audit-gate.md` (independent-context audit before merging multi-commit PRs against rapidly-moving main) | workflow |
| CTE-Guard misapplication on aperture-sensitive paths | `feedback_cte_guard_pattern_match_misapplication.md` | `.claude/rules/adversarial-audit-gate.md` — independent-context audit catches pattern-matched retractions of real bugs | workflow |

## Rule for new families

Future class-bug families surfaced in `memory/feedback_*.md` MUST be
classified at discovery time as either:

- **(a) static** — addressable by a pattern-matching drift check (regex, AST,
  or annotation grammar). Author the check in `pipeline/check_drift.py`
  before closing the feedback file as "documented." Cite this file from the
  check's docstring (`@class-memo memory/<feedback_file>.md`) and add a row
  to the matrix above.
- **(b) workflow** — process-level (PR review, audit dispatch, branch
  hygiene). Cite the owning rule in `.claude/rules/`. NEVER silently leave
  uncovered.

## Why per-class, not generalized

Generalized fingerprint scanners — proposed as Stage B
`harden-blast-radius-class-bug-detector` (2026-05-06) — were considered and
rejected. Rationale (audit verdict CONDITIONAL → PASS conditional on this
stage shipping):

1. The high-leverage step is naming the family, not detecting it. Each
   family has a single literal token, AST shape, or annotation grammar;
   targeted checks are < 80 LOC each.
2. Generalized fingerprint scanners optimize for the wrong end of the
   pipeline — pattern recall — when the real failure mode is family
   discovery. The two recent recurrences (`orb_minutes=5` and the iso_utc
   F6 phantom premise) were both caught by humans/auditors reading code,
   not by absent-but-needed fingerprint regex.
3. Per-class checks ship as advisory by default and are promoted to
   blocking only after the first-week shakeout — giving operators a clean
   feedback loop without false-positive fatigue.

This authority statement supersedes any conflicting "we should build a
class-bug fingerprint scanner" suggestion until the matrix above grows past
~12 families OR a class is identified for which static + workflow defenses
both fail.
