---
task: |
  Make the seven-sins / institutional-rigor literature canon ENFORCED, not
  advisory. Today the code-review SKILL.md and institutional-rigor.md §7 say
  "cite Chordia t>=3.79 / DSR / MinBTL for any Sharpe/significance claim" but
  nothing fails a commit that ignores it. Add a staged-only, repo-wide
  literature-anchor gate to scripts/tools/check_claim_hygiene.py: a doc that
  makes a Sharpe / t-stat / significance NUMBER claim must either cite a
  docs/institutional/literature/ extract OR tag the claim UNSUPPORTED / MEASURED
  / INFERRED. Broaden the pre-commit [7/8] staged-file filter from
  docs/audit/results/ only to the research/results/doctrine surfaces so the gate
  is repo-wide, not pigeonholed. Scope is the STAGED set only — historical docs
  are grandfathered (no retroactive landmine across 1044 existing result docs).
mode: IMPLEMENTATION
updated: 2026-06-05T00:00Z
agent: claude (opus 4.8)

scope_lock:
  - scripts/tools/check_claim_hygiene.py
  - tests/test_tools/test_check_claim_hygiene.py
  - .githooks/pre-commit
  - .claude/skills/code-review/SKILL.md

## Blast Radius
- scripts/tools/check_claim_hygiene.py — adds `_STAT_CLAIM_PATTERN`,
  `_STAT_ANCHOR_PATTERN`, `_in_stat_claim_scope`, `check_stat_claim_anchor`, and
  wires `check_stat_claim_anchor` into `main()`'s per-file loop. Pure-additive:
  the existing PR-body + result-doc-section checks are untouched. String/regex
  only, no network/DB. Fail-direction: a doc with a stat NUMBER and zero anchor
  → exit 1 (BLOCK); everything else → no new issue.
- .githooks/pre-commit step [7/8] — broadens the `STAGED_RESULT_DOCS` grep from
  `^docs/audit/results/.*\.md$` to also include docs/audit/, docs/institutional/,
  docs/plans/, research/ .md files, so the checker SEES the repo-wide staged set.
  The checker itself decides per-file which gate applies (result-doc-sections vs
  stat-anchor), so broadening the filter cannot regress the section gate.
- tests/test_tools/test_check_claim_hygiene.py — RED/GREEN: a staged doc with
  "Sharpe = 1.8" and no anchor FAILS; the same doc + a literature citation or an
  UNSUPPORTED tag PASSES; a methodology-prose doc ("compute the Sharpe ratio",
  no number) PASSES (no false positive).
- .claude/skills/code-review/SKILL.md — already edited this session (Sections
  A/C/G now cite the canon); listed in scope_lock because the gate enforces what
  the skill documents and the two must stay consistent.
- Reads: staged file contents. Writes: none. No canonical-source / schema / DB /
  capital change. Enforcement-doctrine only.
---

# Stat-claim literature-anchor gate (enforce the seven-sins canon)

## Why
2026-06-05: audited the system for "built but not wired / not enforced". Finding:
the repo ingested 28 institutional literature extracts (Bailey-LdP DSR, Chordia
t>=3.79, Harvey-Liu-Zhu, MinBTL, FST, Aronson) and the code-review SKILL.md +
institutional-rigor.md §7 map each statistical sin to its extract — but the
mapping is PROSE an agent may or may not follow. A Sharpe/significance claim can
land in a committed doc with zero literature grounding and nothing fails. That is
exactly the "doctrine that isn't enforced" anti-pattern institutional-rigor.md
forbids.

Naive design (scan all docs/audit/results/*.md) was rejected by self-critique:
1044 result docs, 925 make stat claims, the majority predate the anchor
convention → a check that fails 600+ historical files is a landmine, not a gate.

## Fix
Staged-only scope (the pre-commit already passes only changed files) so history
is grandfathered and only NEW claims must be grounded. Repo-wide across
docs/audit/, docs/institutional/, docs/plans/, research/ (not pigeonholed to
results/). The claim trigger requires a NUMBER near the stat word so methodology
prose does not false-fire. Any one anchor satisfies it: a literature path, a
provenance annotation, a named canon author, or a grounding tag
(UNSUPPORTED/MEASURED/INFERRED).

## Acceptance
- tests/test_tools/test_check_claim_hygiene.py PASSES — show output (incl. new
  stat-anchor RED→GREEN + no-false-positive-on-prose test).
- python scripts/tools/check_claim_hygiene.py on a crafted ungrounded doc → exit 1;
  same doc + anchor → exit 0 (show both).
- pre-commit [7/8] broadened filter verified: a staged docs/institutional/*.md
  with an ungrounded Sharpe number is BLOCKED; existing committed docs unaffected
  (staged-only).
- python pipeline/check_drift.py PASSES (176/0 baseline held).
- dead-code sweep: every new helper is referenced.
