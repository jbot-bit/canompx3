---
task: code-review-hotpatches-2026-05-05
mode: IMPLEMENTATION
slug: code-review-hotpatches-2026-05-05
phase: 1
total_phases: 1
---

# code-review-hotpatches-2026-05-05

Three surgical fixes from a code-review pass on recent main merges (PR #221, #226, #223). Each fix is one commit. Stage covers all three because they share the same review session and verification cycle.

## Findings (subagent code-review, evidence-cited)

1. **Fix 1 — `session_orchestrator.py:1628`** — operator-visible `_feed_status["last_bar_utc"]` calls `bar.ts_utc.isoformat()` directly, bypassing the `bot_state._iso_utc` helper that PR #221 F3 introduced specifically to close this silent-None / type-mismatch class. Same class-bug as F3, missed callsite.

2. **Fix 2 — `chordia.py:229-241`** — `_coerce_audit_date` checks `isinstance(raw, date)` BEFORE `isinstance(raw, datetime)`. Because `datetime` is a subclass of `date`, an unquoted ISO timestamp in YAML returns a `datetime` instance into a field annotated `date | None`. Downstream `(today - audit_date).days` mixes types and produces wrong staleness for the allocator's chordia gate.

3. **Fix 3 — `ask.py:128-129`** — `_resolve_chat_model` accepts any opaque env value with no shape validation. Typo without provider prefix returns 400 from OpenRouter with no operator hint; not capital-risk but cheap to harden.

## Scope Lock

- trading_app/live/session_orchestrator.py
- trading_app/live/bot_state.py
- trading_app/chordia.py
- scripts/tools/ask.py
- tests/test_trading_app/test_session_orchestrator.py
- tests/test_trading_app/test_chordia.py
- tests/test_tools/test_ask_cli.py

## Blast Radius

- `session_orchestrator.py:1628` — single write to `_feed_status["last_bar_utc"]`. Consumed by `_feed_status_payload()` (1227-1232) → `bot_state.build_state_snapshot(..., feed_status=...)` → `data/bot_state.json` → operator dashboard. Fix routes through existing `bot_state._iso_utc` (already log-WARN on unknown types). No public API change.
- `bot_state.py` — `_iso_utc` already exported within module (private-ish but module-local helper). Re-exporting via `__all__` not required; orchestrator imports directly.
- `chordia.py:_coerce_audit_date` — called twice (audit_date, audit_reaffirmed_date) in `load_chordia_audit_log`. Result feeds `ChordiaAuditEntry.audit_date / audit_reaffirmed_date` → `ChordiaAuditLog.audit_age_days` → `lane_allocator.py` chordia gate (`audit_age_days <= audit_freshness_days`). Wrong-type returns produced incorrect staleness.
- `ask.py:_resolve_chat_model` — operator CLI only (`scripts/tools/`). No production trading-path consumer. Tests exist; gap is only the no-slash invalid case.
- Reads: read-only on gold.db. Writes: none — only changes in-memory dataclass coercion, helper delegation, and operator stdout.

## Verification

Per fix:
- `python pipeline/check_drift.py` (full)
- `pytest <targeted test file> -q` (the test file in scope_lock)
- For Fix 1 only (live-trading path): evidence-auditor adversarial pass on the commit before push.

After all 3 commits:
- `pytest tests/test_trading_app/test_session_orchestrator.py tests/test_trading_app/test_chordia.py tests/test_tools/test_ask_cli.py -q`
- `python pipeline/check_drift.py`
- `ruff check` + `ruff format --check` on the 4 prod files.

## Out of Scope

- F6 root cause in `execution_engine.py` (separate stage `f6-execution-engine-pd-timestamp-coercion.md` already filed).
- Any refactor of `_iso_utc` to a `trading_app.live.timefmt` module — defer; the helper works.
- Any change to chordia audit-log YAML schema or freshness threshold.
