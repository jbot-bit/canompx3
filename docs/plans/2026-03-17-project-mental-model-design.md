# Project Mental Model Layer — Design Doc

**Date:** 2026-03-17
**Status:** Approved (4TP auto-proceed)
**Pattern:** Extract-then-reason (mechanical aggregator + AI narrator)

## Problem

8 signal sources exist but are siloed. No single tool synthesizes project state.
Claude auto-loads ~15K tokens of docs with zero synthesis. Codex manually reads 14 steps.
The bat launcher has no project awareness at all. The user rebuilds their mental model
from scratch every session.

## Solution: Option B — Pulse Script + AI Narrator

### Architecture

```
project_pulse.py (mechanical, ~5s, cached <1s)
  ├── 8 collectors (git, handoff, drift, staleness, fitness, ralph, action queue, code debt)
  ├── PulseReport dataclass → JSON / text / markdown
  ├── Cache: git-HEAD + DB-mtime key → .pulse_cache.json
  └── Graceful degradation (DB locked, missing files)

/orient skill (AI layer, ~15s)
  ├── Calls: project_pulse.py --format json
  └── Narrates: prioritized, reasoned assessment

Renderers:
  ├── ai-workstreams.bat [0] → text output
  ├── session_preflight.py --with-pulse → appended summary
  ├── /orient skill → full AI triage
  └── PROJECT_PULSE.md → Codex startup snapshot
```

### Signal Sources

| # | Signal | Collector | Source |
|---|--------|-----------|--------|
| 1 | Git state | collect_git_state() | git CLI |
| 2 | Session baton | collect_handoff() | HANDOFF.md (reuse extract_handoff_snapshot) |
| 3 | Drift checks | collect_drift() | subprocess check_drift.py |
| 4 | Pipeline staleness | collect_staleness() | pipeline_status.staleness_engine() |
| 5 | Strategy fitness | collect_fitness() | SQL on edge_families |
| 6 | Ralph audit debt | collect_ralph() | ralph-ledger.json |
| 7 | Action queue | collect_action_queue() | MEMORY.md strikethrough parsing |
| 8 | Code debt | collect_code_debt() | grep TODO/FIXME in prod code |

### Output Categories

| Category | Meaning | Examples |
|----------|---------|---------|
| BROKEN | Fix before doing anything else | Drift check failing, test failure |
| DECAYING | Act soon or edge degrades | Strategy WATCH→DECAY, stale pipeline |
| READY | Planned work, pick one | Action queue items, ROADMAP TODOs |
| UNACTIONED | Noticed but never fixed | Ralph deferred findings, TODO comments |
| PAUSED | Suspended work | Git stashes, open worktrees |

### Data Model

```python
@dataclass
class PulseItem:
    category: str        # broken/decaying/ready/unactioned/paused
    severity: str        # high/medium/low
    source: str          # which collector found it
    summary: str         # one-line human description
    detail: str | None   # optional extra context

@dataclass
class PulseReport:
    generated_at: str
    cache_hit: bool
    git_head: str
    git_branch: str
    broken: list[PulseItem]
    decaying: list[PulseItem]
    ready: list[PulseItem]
    unactioned: list[PulseItem]
    paused: list[PulseItem]
    handoff_tool: str | None
    handoff_date: str | None
    handoff_summary: str | None
```

### Caching

- Key: `f"{git_HEAD_sha}_{gold_db_stat.st_mtime_ns}"`
- Location: `<project>/.pulse_cache.json`
- Hit: HEAD same AND gold.db unmodified → serve cached (<1s)
- Miss: re-run all collectors (~5s)
- Force refresh: `--no-cache`

### Files Touched

| Action | File | Lines Changed |
|--------|------|--------------|
| CREATE | `scripts/tools/project_pulse.py` | ~250 |
| CREATE | `.claude/skills/orient/SKILL.md` | ~40 |
| CREATE | `tests/test_tools/test_project_pulse.py` | ~150 |
| MODIFY | `scripts/infra/windows-agent-launch.ps1` | ~15 (add [0] menu item) |
| MODIFY | `scripts/tools/session_preflight.py` | ~10 (add --with-pulse) |
| MODIFY | `scripts/infra/claude-worktree.sh` | ~1 (pass --with-pulse) |
| MODIFY | `.codex/STARTUP.md` | ~1 (add step 4.5) |

### Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| gold.db write-locked | Catch IOException, skip DB checks, report "DB locked" |
| check_drift.py slow | Cache + only re-run when HEAD moves |
| MEMORY.md parse fragility | Simple regex + tests, never crash on malformed |
| ralph-ledger.json schema change | Defensive dict.get(), degrade gracefully |
| Pulse crash | Additive to preflight — preflight still works without it |

### Token Budget

| Context | Tokens In | Tokens Out |
|---------|----------|------------|
| Raw docs (today) | ~15,000 | 0 (no synthesis) |
| Pulse JSON | ~2,000 | ~500 (AI narration) |
| Savings | 85% reduction | Full synthesis |
