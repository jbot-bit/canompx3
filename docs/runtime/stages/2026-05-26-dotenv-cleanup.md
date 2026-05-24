---
task: |
  CLEANUP — `.env` produces 80+ `python-dotenv could not parse statement
  starting at line N` warnings on every script invocation (lines 130-296
  per today's 2026-05-26 smoke log). Pure noise but pollutes triage and
  hides real WARNING lines.

  Per python-dotenv official docs (https://github.com/theskumar/python-dotenv):
  > "The format is not formally specified and still improves over time.
  >  That being said, `.env` files should mostly look like Bash files."
  >
  > "Keys can be unquoted or single-quoted. Values can be unquoted,
  >  single- or double-quoted."
  >
  > "Values can be followed by a comment."

  Likely causes of parse warnings (LINES 130-296 falls in a contiguous
  90-line block — fits "multi-line JSON cert/config block pasted as
  unquoted value" pattern):
    - Multi-line PEM cert pasted unquoted
    - JSON config pasted directly (curly braces aren't valid in
      unquoted KEY=VALUE)
    - Indented continuation lines without `\` escapes
    - Comments in middle of multi-line values

  FIX: operator must inspect `.env` lines 130-296 (sensitive, Claude has
  no read permission). Apply one of:
    (a) Wrap multi-line values in double-quotes
    (b) Use `\n` escapes inside double-quoted single-line value
    (c) Move large blobs to separate files referenced by single-line path
        env vars (e.g., `PROJECTX_JWT_PATH=/path/to/cert.pem`)

mode: TRIVIAL
status: DESIGN_LOCKED_PENDING_POST_SESSION_2026_05_26
priority: P3_COSMETIC
deferred_reason: |
  `.env` is operator-owned + access-controlled. Claude has no read
  permission per `.claude/settings.json` permission policy. Operator
  edits the file manually after the live session.

scope_lock:
  - .env  # OPERATOR EDIT ONLY — Claude cannot touch

agent: operator (joshd)
---

## Blast Radius

- WRITES (operator-only): `.env` lines 130-296.
- READS: `.env` (operator only).
- LIVE-IMPACT: zero on existing env vars. Only fixes parsing noise.
- Rollback: revert single file.

## Acceptance

1. Run `.venv\Scripts\python.exe -c "from dotenv import load_dotenv; load_dotenv()"` — exits with ZERO `could not parse statement` warnings.
2. Existing env vars (PROJECTX_USER, PROJECTX_API_KEY, BROKER, etc.) still load — confirm via `.venv\Scripts\python.exe -c "import os; from dotenv import load_dotenv; load_dotenv(); print('PROJECTX_USER=', os.environ.get('PROJECTX_USER'))"`.
3. Orchestrator startup log no longer shows the 80+ WARNING block.

## Sources

- python-dotenv format docs: https://github.com/theskumar/python-dotenv#file-format
- python-dotenv source `main.py` parsing rules: github.com/theskumar/python-dotenv

## Operator runbook

```powershell
# Backup first
Copy-Item .env .env.backup_20260526

# Inspect lines 130-296
Get-Content .env | Select-Object -Skip 129 -First 167

# Find the malformed block visually; common patterns:
#   - Multi-line PEM: wrap as PEM_CERT="-----BEGIN...END-----"
#   - JSON blob: wrap as CONFIG_JSON='{"key":"value"}'
#   - Comments in middle of value: move to separate line above
```
