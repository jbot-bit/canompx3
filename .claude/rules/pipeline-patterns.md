---
paths:
  - "pipeline/**"
---
# Pipeline Patterns

Core principles (fail-closed, idempotent, one-way dependency) and DST resolution → see CLAUDE.md.

## Database Write Pattern
Uses idempotent DELETE-then-INSERT: delete existing rows for the date range, then insert new ones.
Prevents duplicates without requiring upsert logic.

## Time & Calendar
- Trading day: 09:00 local -> next 09:00 local
- Bars before 09:00 assigned to PREVIOUS trading day
- February (EST): US data 11:30pm Brisbane, NYSE open 12:30am Brisbane
