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
