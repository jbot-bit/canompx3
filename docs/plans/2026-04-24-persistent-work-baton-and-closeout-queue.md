# Persistent Work Baton And Closeout Queue

## Problem

The repo already shrank `HANDOFF.md`, but the baton still drifts because it is
freehand prose with weak ownership and freshness semantics. Startup cost is
still too high, multi-terminal state is not held cleanly, and meaningful open
work is split across `HANDOFF.md`, stage files, ledgers, and private memory.

## Decision

Use a split model:

- `docs/runtime/action-queue.yaml` is the canonical active-work registry for
  meaningful open work.
- `HANDOFF.md` is a thin generated baton derived from the queue.
- `.session/work_queue_leases.json` is the local ignored session-lease file for
  multi-terminal ownership and heartbeats.
- `docs/runtime/decision-ledger.md` and `docs/runtime/debt-ledger.md` remain
  the durable accepted-decision and debt surfaces.

## Workflow

1. Add or update meaningful open work in `docs/runtime/action-queue.yaml`.
2. Record local claims with `python scripts/tools/work_queue.py claim ...`.
3. Refresh the baton with `python scripts/tools/work_queue.py render-handoff --write`.
4. Close or supersede queue items when the work is truly done.

## Policy

- Soft WIP gate: if `close_before_new_work=true` items remain open, starting a
  different new queue item requires an explicit override note.
- Preflight and pulse should read the queue directly and warn on stale items,
  carry-over work, and fresh lease overlap.
- Do not use Claude private memory as the action queue source.

## Seeded Active Set

The initial queue is seeded from measured open work on 2026-04-24:

1. prior-day Pathway-B bridge execution triage
2. cross-asset chronology spec
3. GC -> MGC 15m/30m translation question
4. PR48 MGC shadow-only observation
5. MES `q45_exec` bridge
6. Track D MNQ COMEX_SETTLE Gate 0 runner design
7. MES event-tail slippage realism debt
