---
task: "Add /recall retrospective-decision skill + new-skill-eval-nudge PostToolUse hook"
mode: IMPLEMENTATION
scope_lock:
  - .claude/skills/recall/SKILL.md
  - .claude/skills/recall/eval/eval.json
  - .claude/hooks/new-skill-eval-nudge.py
  - .claude/settings.json
---

## Blast Radius

- `.claude/skills/recall/SKILL.md` — new file, zero callers. Skill is opt-in via `/recall <topic>` invocation; zero impact unless invoked. Delegates kill-verdict slice to existing `/nogo` skill (no re-implementation of the research-catalog verdict filter). Adds non-kill verdicts via `mcp__research-catalog__search_research_catalog` without verdict_tags, current-truth via `mcp__gold-db__get_strategy_fitness`, lessons via `memory/feedback_*.md` grep, and a locked verdict-on-verdict taxonomy (HOLDS/STALE/SUPERSEDED/INSUFFICIENT_EVIDENCE).
- `.claude/skills/recall/eval/eval.json` — new file. 3 tests (rc-01 E2 break-bar look-ahead, rc-02 NYSE_OPEN RR1.5, rc-03 fictional-strategy absence-check). Assertions cover: delegation to /nogo or research-catalog, audit-results reference, locked verdict taxonomy, no-memory-citation, gold-db query on strategy-shaped topic, honest absence-reporting.
- `.claude/hooks/new-skill-eval-nudge.py` — new PostToolUse hook. Fires on every `Write` tool call; filters to paths matching `.claude/skills/<name>/SKILL.md`; checks if `eval/eval.json` exists alongside; emits advisory `additionalContext` JSON when missing. Fail-open on any error (matches `branch-flip-guard` posture). Path-pattern check is ~1ms — no measurable cost on non-skill writes. Tested with stdin payloads: silent when eval present, nudge JSON when absent.
- `.claude/settings.json` — additive PostToolUse entry only. New `{matcher: "Write", hooks: [...]}` block appended after the existing post-edit-schema entry. No edits to any existing hook registration. Timeout 3s.
- Reads: none. Writes: none to DB. No schema, no canonical-source, no production code (`pipeline/`, `trading_app/`) touched.

## Acceptance Criteria

1. Hook smoke test — existing-eval path → silent exit 0 (verified).
2. Hook smoke test — missing-eval path → emits `additionalContext` JSON (verified).
3. `python pipeline/check_drift.py` passes.
4. `git diff --name-only HEAD` after commit shows exactly the 4 scope_lock paths and zero others.
5. Skill end-to-end self-test via `/skill-improve recall` deferred to a follow-up session (heavy subagent operation; not in scope_lock for this stage).

## Non-goals (explicit)

- Capability-claim guard hook — n=1 incident captured in `memory/feedback_capability_confabulation_n1_2026_05_16.md` instead; defer per `feedback_meta_tooling_n1_tunnel_2026_05_01.md` (no hooks on n=1).
- Running `/skill-improve recall` end-to-end this session — heavy subagent spawn, schedule for a fresh session.
- Any changes to `/nogo`, `/pinecone-assistant`, or `research-catalog` MCP (delegated-to, not modified).

## Risk Tier

LOW — operator-tooling, advisory hook, no production code, no schema, no broker/risk surface, no capital impact.
