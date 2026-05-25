---
task: Fix UnicodeEncodeError on Windows cp1252 console that crashes run_live_session.py --live before the CONFIRM prompt can be answered
mode: IMPLEMENTATION
scope_lock:
  - scripts/run_live_session.py
blast_radius: |
  scripts/run_live_session.py — entry-point module. Adds a stdout/stderr UTF-8
  reconfigure guard at import time (matches canonical pattern in
  research/allocator_scarcity_surface_audit.py:39-41). Affects only console
  output encoding on non-utf-8 terminals (Windows cp1252). No change to trading
  logic, order placement, preflight probes, portfolio load, or the CONFIRM gate
  semantics. Reads: none. Writes: none (stdout encoding only). Callers: operator
  CLI invocation only; not imported as a library on a hot path.
---

## Context

`run_live_session.py --live` reached the interactive CONFIRM prompt (line 887)
which embeds `⚠` (warning sign) + em-dash. Windows console codepage is
cp1252; `input()` writes the prompt through that codec and raised
`UnicodeEncodeError: 'charmap' codec can't encode character '⚠'`, crashing
the launcher before the operator could type CONFIRM. A real-money trader cannot
launch on a default Windows console.

## Fix

Apply the canonical UTF-8 reconfigure guard at module import (same shape as
`research/allocator_scarcity_surface_audit.py:39-41`): wrap stdout/stderr in a
UTF-8 TextIOWrapper with errors="replace" when the console encoding is not utf-8.

## Acceptance

- `python pipeline/check_drift.py` passes
- Launcher reaches CONFIRM prompt on cp1252 console without crash (operator repro)
- No change to portfolio load / preflight / order logic
