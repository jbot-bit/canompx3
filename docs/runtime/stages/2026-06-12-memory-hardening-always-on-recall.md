---
task: "Memory hardening + always-on recall (Plan v4) — budget hygiene tool, Pinecone auto-recall hook, budget cue, tests"
mode: IMPLEMENTATION
stage: 2
scope_lock:
  - scripts/tools/memory_hygiene.py
  - .claude/hooks/memory-recall.py
  - .claude/hooks/memory-capture-sessionstart.py
  - .claude/settings.json
  - tests/test_tools/test_memory_hygiene.py
  - tests/test_hooks/test_memory_recall.py
---

# Stage: Memory hardening + always-on recall

## Checkpoint progress (Part A — 3 CPs, /clear between each)

- **CP1 DONE** (commit `63b06aa8`): `scripts/tools/memory_hygiene.py` — read-only
  budget + baton clear-tier report. Verified: budget OVER (26,194 B, byte-cutoff
  binds first, first-dropped line 71, 48 over-long lines); tiers READY=12 /
  LANDED-BUT-OPEN=29 / UNVERIFIED=102; `--print-clear` fully-commented; `--json`
  parses; ruff clean. READY/LBO tiering falsified against `git merge-base`.
- **CP2 NEXT** (`stage: 2`): write `tests/test_tools/test_memory_hygiene.py` —
  load tool via importlib; tmp memory dir + tmp git repo with
  `update-ref refs/remotes/origin/main` for deterministic ancestor checks. Cases:
  byte-cutoff math, 200-line truncation point, over-long-line flag, primary-SHA
  READY selection, open-work-marker→LANDED-BUT-OPEN, unmerged SHA→UNVERIFIED,
  clear-block commented + READY-only. Then `pytest -q` green → commit
  `test(tools): cover memory_hygiene budget + clear tiers` → `stage: 3` → /clear.
- **CP3** (`stage: 3`): +1 budget-cue branch in `memory-capture-sessionstart.py`;
  trim MEMORY.md < 24,400 B; write deferred-Part-B baton. → close stage.

## Task

Implement Plan v4: (A) budget-hygiene reporting so the auto-memory `MEMORY.md`
index stops silently overflowing the 200-line / 25 KB load window; (B) an
always-on `UserPromptSubmit` recall hook that auto-injects top-3 relevant
memories from the EXISTING Pinecone `orb-research` Assistant (no second store);
(C) a +1-line budget cue in the SessionStart capture hook. All read-only or
fail-open; no autonomous baton deletion (operator-cleared via `--print-clear`).

## Scope Lock

- scripts/tools/memory_hygiene.py
- .claude/hooks/memory-recall.py
- .claude/hooks/memory-capture-sessionstart.py
- .claude/settings.json
- tests/test_tools/test_memory_hygiene.py
- tests/test_hooks/test_memory_recall.py

## Blast Radius

- `scripts/tools/memory_hygiene.py` — NEW, read-only report tool; zero callers; imports `.claude/hooks/_memory_capture.py` by path (reuse, no re-encode). Reads gold.db: none. Writes: none.
- `.claude/hooks/memory-recall.py` — NEW UserPromptSubmit hook; fail-open (no key/network/SDK error → exit 0 silent). Adds one Pinecone Assistant `context()` network call per non-trivial prompt; trivial-prompt skip + top-3 cap bound the cost. No write path.
- `.claude/hooks/memory-capture-sessionstart.py` — +1 fail-open budget-cue branch appended to existing `additionalContext`; existing breadcrumb/staleness logic untouched.
- `.claude/settings.json` — register the recall hook under existing UserPromptSubmit array (concatenated additionalContext — documented behavior). Config only.
- `tests/test_tools/`, `tests/test_hooks/` — NEW deterministic tests; mocked Pinecone client (no live network). No production impact.
- Reads: home-based MEMORY.md + memory/*.md (read-only). Writes: none by the tool/hook; only the operator-run `--print-clear` block deletes (and only READY-tier).
