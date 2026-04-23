# Decision Ledger

Use this file for durable accepted decisions that should survive handoff churn.

## Current

- `runtime-shell-unification` — Startup orientation should consume one derived `system_brief` instead of parallel ad hoc summaries.
- `capsule-is-task-owner` — Active scoped work should carry a single work capsule with route, scope, and verification obligations.
- `history-split` — `HANDOFF.md` is current baton only; durable decisions and debt belong in dedicated ledgers.
- `pr48-conditional-edge-recovered` — The unfinished PR48 conditional-edge / multi-confluence line has been recovered from `origin/wt-codex-conditional-edge-framework`, replayed cleanly onto published `origin/main`, and re-run against canonical layers. Current truth is `CONTINUE as conditional-role / allocator shortlist`, not fresh discovery and not live-ready promotion. The next honest move is the bounded translation stage in `docs/runtime/stages/pr48-conditional-role-validation-translation.md`, not another generic confluence scan.

