---
task: "Operating-layer audit — surgical cleanup (rule dedup + intent-router drift check + 2 agent deletes)"
mode: IMPLEMENTATION
stage_purpose: "Stage A: dedupe 5 rule files, add intent-router parity drift check, delete planner + test-coverage-scout agents, rewire brain + stage-gate skill references to them"
updated: 2026-05-17T00:00:00+10:00
status: COMPLETED
---

## Scope Lock

- .claude/rules/auto-skill-routing.md
- .claude/rules/daily-features-joins.md
- .claude/rules/quant-agent-identity.md
- .claude/rules/integrity-guardian.md
- .claude/rules/institutional-rigor.md
- .claude/skills/brain/SKILL.md
- .claude/skills/stage-gate/SKILL.md
- .claude/agents/planner.md
- .claude/agents/test-coverage-scout.md
- pipeline/check_drift.py
- docs/runtime/stages/operating_layer_audit_2026_05_17.md

## Blast Radius

- 5 rule files: 3 narrow content (integrity-guardian §2 + §7, quant-agent-identity full body, daily-features-joins lookahead list); 1 grows (institutional-rigor adds §10, §11, §12 absorbed content); 1 adds load-policy note (auto-skill-routing). All paths-frontmatter changes affect which rules auto-load on edits.
- 2 SKILL.md files: brain SKILL.md drops 2 agent dispatch entries; stage-gate SKILL.md replaces planner agent dispatch with /design skill. Reads-only changes to skill behavior — agent fallback would have been general-purpose anyway.
- 2 agent files DELETED: planner.md (zero subagent_type references), test-coverage-scout.md (zero subagent_type references). Pre-flight greps confirmed no callers in .claude/, scripts/, docs/governance/.
- pipeline/check_drift.py: adds 1 new check function (~80 lines) + 1 CHECKS list registration. Pure addition; no existing checks modified.
- Reads gold.db: none. Writes gold.db: none. Reads filesystem: yes (rule/hook parsing in new drift check).
- Capital-class impact: zero.

## Acceptance — RESULTS

- ✅ `python pipeline/check_drift.py` — 131 PASS, 20 advisory, 0 violations. New Check 151 PASSED.
- ✅ Mutation probe: renamed `/design` → `/designX` in auto-skill-routing.md, drift check fired with violation `intent-router.py routes to skills not documented in auto-skill-routing.md: ['/design']`. Reverted, post-revert violations = 0.
- ✅ Cleanup grep: only remaining `\bplanner\b` / `test-coverage-scout` refs are (a) the deleted files themselves (now removed), (b) `subagent-budget.md` anti-pattern doctrine (intentional historical reference).
- ✅ `ls .claude/agents/` = 9 files (was 11): planner.md + test-coverage-scout.md gone.
- ✅ `git diff --stat` — 10 files touched, all in scope_lock. -227/+169 = net -58 lines.

## Deferred to next session (out of scope this stage)

1. **evidence-auditor + research-methodologist deletion** — original plan deletion targets, BUT both are doctrine-named:
   - `.claude/rules/adversarial-audit-gate.md:36` ("Dispatch the `evidence-auditor` subagent (independent context)")
   - `.claude/rules/institutional-rigor.md:29` ("The gate requires an independent-context `evidence-auditor` pass")
   - `.claude/skills/capital-review/SKILL.md:37,38` (capital-review dispatches evidence-auditor — circular if plan-as-written executed)
   Decision needed: replace these agents with skills+hooks (markdown + deterministic) or keep them because slash-command-in-same-context cannot provide the "independent context" the adversarial-audit gate mandates. Re-deciding requires user input on the framing question.

2. **MEMORY.md 200-line truncation** — flagged in plan §C as out-of-scope. Private memory, needs index restructuring; user-owned.

3. **Hook consolidation** (stage-awareness ↔ session-start; discovery-loop-guard ↔ pre-edit-discovery-marker) — Explore audit said current separation is intentional. Defer until evidence of pain.

4. **integrity-guardian.md cleanup pass** — file still names "Seven non-negotiable rules" in line 11 but now has 8 sections (the 8th — research-finding-staleness — was always there). Cosmetic only.

5. **brain SKILL.md institutional ceremony line (L69)** — still names `evidence-auditor` at end of pipeline. Keep — see deferred #1.
