---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Copy Trading Orchestrator — Design (2026-04-01)

## Problem

`AccountProfile.copies = 5` is decorative. Nothing spawns multiple accounts.
8 gaps block multi-account execution (see ORIENT analysis below).

## Approach: Take 2.5 — CopyOrderRouter (evolved from Take 2)

**Original design (Take 2):** N independent processes. Rejected after auth research
confirmed ONE token works for ALL accounts — N processes would share credentials
and risk token conflicts. Also causes DuckDB concurrent writer issues on journal.

**Implemented (Take 2.5):** ONE process, ONE auth, ONE DataFeed, ONE SessionOrchestrator.
CopyOrderRouter wraps primary + N shadow OrderRouters. submit() fans out to all.
Primary gets full position tracking. Shadows are best-effort (broker-side DD).

**Why not full Take 3 (refactored signal/order split)?**
SessionOrchestrator is 2070 lines. Take 2.5 requires zero changes to core signal
logic — only the order_router construction changes. Minimal blast radius.

## 5 Blockers to Fix

| # | Blocker | Fix | File |
|---|---------|-----|------|
| 1 | Account discovery returns first only | `resolve_all_account_ids()` | `projectx/contract_resolver.py` |
| 2 | Instance lock per-instrument conflicts | Include account_id in lock key | `instance_lock.py` |
| 3 | Trade journal has no account_id | Add column + migrate | `trade_journal.py` |
| 4 | Bot state is single global file | Key by account_id | `bot_state.py` |
| 5 | No multi-account launcher | `--copies` flag + subprocess loop | `run_live_session.py` |

## Implementation Stages

### Stage 1: Infrastructure (backward compatible)

1. `contract_resolver.py`: add `resolve_all_account_ids() -> list[int]`
2. `instance_lock.py`: `acquire_instance_lock(instrument, account_id=0)` — account in lock key
3. `trade_journal.py`: add `account_id TEXT` column (ALTER TABLE, idempotent)
4. `bot_state.py`: `_state_path(account_id)` → `data/bot_state_{account_id}.json`

### Stage 2: Multi-account runner

5. `run_live_session.py`: `--copies N` discovers N accounts, launches N subprocesses
6. `bot_dashboard.py` + `.html`: show per-account status, launch with copies

### Stage 3: Smoke test

7. Run `--copies 2 --signal-only` — verify isolation

## Firm Rules (verified)

- **TopStep**: 5 Express + 1 Live. Copier on Express only. ProjectX API.
- **Tradeify**: 5 accounts. Same-owner API. Bot must be exclusive.
- **Apex**: PROHIBITED. Manual only. 1 account.

## DD Budget Per Account

Worst-day all-lose at 1ct: $1,384 (TYPE-A) / $1,391 (TYPE-B).
Per-account DD tracked independently by HWMTracker.
If one account blows → that subprocess exits. Others continue.

## Rollback

All changes are additive (new methods, new columns with defaults, new flags).
Existing single-account flow uses default params unchanged.
