---
task: Stage 1 — safe stale Claude/MCP process reaper for session-start hygiene
mode: IMPLEMENTATION
scope_lock:
  - scripts/tools/reap_stale_claude_processes.py
  - tests/test_tools/test_reap_stale_claude_processes.py
blast_radius:
  - scripts/tools/reap_stale_claude_processes.py — NEW read-only-by-default tool; zero importers; dry-run default, --apply to kill. No production import path.
  - tests/test_tools/test_reap_stale_claude_processes.py — NEW unit tests over a mocked process table; no real process killed in tests.
  - Reads: .git/.claude.pid (session lock), OS process table (psutil if present, else tasklist/ps). Writes: none to repo, stdout only.
  - Capital safety: hard-excludes webhook_server|bot_dashboard|--demo|--live|launch|broker signatures from kill candidacy.
updated: 2026-05-29T00:00:00+10:00
agent: claude
---

## Purpose
PC + Claude slow down over a session because abandoned prior sessions leave duplicate MCP server
generations + orphaned multiprocessing-fork workers resident, contending for the gold.db read lock.
This tool reaps ONLY provably-stale project processes. Dry-run default; fail-open; capital paths excluded.

## Acceptance
- Dry-run default prints candidates with PID, age, parent-alive, reason; kills nothing.
- --apply kills only: (a) fork workers whose parent PID is dead, (b) project MCP/helper procs older
  than the current .git/.claude.pid iso_started AND not the current session's own children.
- Hard-exclude any process whose command line matches the capital-path signature set.
- Fail-open: any inventory/parse error → exit 0, kill nothing.
- Unit tests (mocked process table) prove: orphan-worker killed, live-bot never killed, ambiguous→skip,
  dry-run kills nothing.
