# Project Mental Model Layer — Design Doc

**Date:** 2026-03-17 (designed), 2026-03-18 (implemented v2)
**Status:** SHIPPED (v2 on main)
**Pattern:** Extract-then-reason (mechanical aggregator + AI narrator)

## Problem

Multiple signal sources exist but are siloed. No single tool synthesizes project state.
Claude auto-loads ~15K tokens of docs with zero synthesis. Codex manually reads 14 steps.
The bat launcher has no project awareness at all. The user rebuilds their mental model
from scratch every session.

## Solution: Pulse Script + AI Narrator

### Architecture

```
project_pulse.py (mechanical, ~3s fast / ~35s full)
  ├── 9 collectors + 4 v2 enhancements
  ├── PulseReport dataclass → JSON / text / markdown
  ├── 2-tier cache: expensive (drift/tests/deep fitness) by HEAD,
  │   cheap collectors always fresh
  ├── Graceful degradation (DB locked, missing files)
  ├── Recommendation engine (one line, one decision)
  └── Skill invocation suggestions on every actionable item

/orient skill (AI layer, ~15s)
  ├── Calls: project_pulse.py --fast --format json
  └── Narrates: prioritized, reasoned assessment using pulse JSON

Renderers:
  ├── ai-workstreams.bat [0] → text output
  ├── session_preflight.py --with-pulse → appended summary
  ├── /orient skill → full AI triage
  └── PROJECT_PULSE.md → Codex startup snapshot
```

### Signal Sources (v2)

| # | Signal | Collector | Source |
|---|--------|-----------|--------|
| 1 | Git state + stashes | collect_git_state() | git CLI |
| 2 | Session baton + next steps + blockers | collect_handoff() | HANDOFF.md single-pass parse |
| 3 | Drift checks | collect_drift() | subprocess check_drift.py |
| 4 | Test suite | collect_tests() | subprocess pytest |
| 5 | Pipeline staleness | collect_staleness() | pipeline_status.staleness_engine() |
| 6 | Strategy fitness | collect_fitness_fast/deep() | SQL validated_setups / strategy_fitness.py |
| 7 | Action queue | collect_action_queue() | MEMORY.md strikethrough parsing |
| 8 | Ralph deferred | collect_ralph_deferred() | deferred-findings.md table parsing |
| 9 | Open worktrees + momentum | collect_worktrees() | .canompx3-worktree.json + git rev-list |
| 10 | Upcoming sessions | collect_upcoming_sessions() | SESSION_CATALOG resolvers + validated_setups |
| 11 | Worktree conflict radar | collect_worktree_conflicts() | git diff --name-only per worktree |
| 12 | Session continuity | collect_session_delta() | .pulse_last_session.json marker |

### v2 Innovations (from research)

| Feature | Source | Impact |
|---------|--------|--------|
| Skill invocation suggestions | SRE handoff + GitHub Copilot | Each item shows → /skill-name to copy-paste |
| Session continuity fingerprint | SRE shift handoff best practices | "Since your last session: 3 commits by Codex" |
| Time since green | Grafana On-Call dashboards | Tracks system health trend over time |
| Worktree conflict radar | Windsurf multi-file reasoning | Warns before merge conflicts happen |
| Trading day layer | Professional quant pre-market checklists | Next sessions with strategy counts |
| Workstream momentum | Obsidian daily notes patterns | Age, commits, STALLED detection |
| Recommendation engine | PagerDuty escalation policies | One line, one decision, one action |

### Output Categories

| Category | Label | Examples |
|----------|-------|---------|
| BROKEN | FIX NOW | Drift failing, test failure |
| DECAYING | ACT SOON | Strategy WATCH→DECAY, stale pipeline, stale handoff |
| READY | ON DECK | Action queue items |
| UNACTIONED | DEBT | Ralph deferred findings |
| PAUSED | PAUSED | Git stashes, open worktrees |

### Cache Design (v2)

Two-tier cache in `.pulse_cache.json`, keyed on git HEAD:

| Tier | Collectors | Behavior |
|------|-----------|----------|
| Expensive | drift, tests, deep fitness | Cached by HEAD. `--fast` serves cached, skips if uncached |
| Cheap | staleness, fitness_fast, handoff, worktrees, conflicts, git, action queue, ralph, sessions | Always fresh (<500ms total) |

- `--fast`: serve cached expensive results when HEAD matches, skip if no cache
- `--deep`: run full fitness computation, cache result for subsequent `--fast` runs
- `--no-cache`: force re-run everything

### Files Touched

| Action | File | Actual Lines |
|--------|------|-------------|
| CREATE | `scripts/tools/project_pulse.py` | ~1300 |
| CREATE | `.claude/skills/orient/SKILL.md` | ~55 |
| CREATE | `tests/test_tools/test_project_pulse.py` | ~640 |
| MODIFY | `scripts/infra/windows-agent-launch.ps1` | +15 ([0] Orient me) |
| MODIFY | `scripts/tools/session_preflight.py` | +15 (--with-pulse) |
| MODIFY | `scripts/infra/claude-worktree.sh` | +2 (pass --with-pulse) |
| MODIFY | `.codex/STARTUP.md` | +1 (step 4.5) |
| MODIFY | `.gitignore` | +3 (cache files + PROJECT_PULSE.md) |

### Token Budget

| Context | Tokens In | Tokens Out |
|---------|----------|------------|
| Raw docs (before pulse) | ~15,000 | 0 (no synthesis) |
| Pulse JSON | ~2,500 | ~500 (AI narration) |
| Savings | 83% reduction | Full synthesis + recommendation |
