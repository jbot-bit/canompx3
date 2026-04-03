---
stage: IMPLEMENTATION
mode: IMPLEMENTATION
task: Build /next skill + fix brief encoding bug
updated: 2026-04-03T21:00:00Z
scope_lock:
  - scripts/tools/claude_superpower_brief.py
  - .claude/skills/next/SKILL.md
blast_radius:
  - brief.py: encoding fix in main(), no callers affected
  - SKILL.md: new file, zero blast radius
acceptance:
  - python scripts/tools/claude_superpower_brief.py --root . --mode interactive prints without UnicodeEncodeError on Windows
  - .claude/skills/next/SKILL.md exists and is well-formed
  - /next skill can be invoked
---
