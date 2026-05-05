---
task: f6-execution-engine-pd-timestamp-coercion
mode: DESIGN
agent: claude
updated: 2026-05-05
---

# F6 — execution_engine.py raw pd.Timestamp on trade.entry_ts and TradeEvent.timestamp

**Status:** DESIGN (trace complete; awaiting user decision on fix shape before promoting to IMPLEMENTATION)
**Date opened:** 2026-05-04
**Trace completed:** 2026-05-05
**Surfaced during:** F3 grounding pass on `_iso_utc` defensive patch.

## What this is

`trading_app/live/bot_state._iso_utc()` was silently returning None for non-datetime inputs (F3 fixed this with logger warnings). Investigating WHY non-datetime values reach `_iso_utc` in production exposed the upstream class-bug:

`trading_app/execution_engine.py` assigns raw `pd.Timestamp` to both `trade.entry_ts` (3 sites) AND every `TradeEvent.timestamp` it emits (~19 sites). The contract on `TradeEvent.timestamp: datetime` (line 62) is silently violated because `pd.Timestamp` subclasses `datetime` — `isinstance` check passes, `.isoformat()` produces wrong-shaped strings.

## Trace findings (2026-05-05)

### Bar source

`ExecutionEngine.on_bar(self, bar: dict)` (line 480) — bar is a dict with `ts_utc` always a `pd.Timestamp` (DataFrame `.iterrows()`/`.to_dict()` preserves Pandas types). Verified by absence of any coercion at line 488 `ts = bar["ts_utc"]`.

### `trade.entry_ts` assignment sites (3 — original F6 scope)

- `execution_engine.py:978` — `trade.entry_ts = confirm_bar["ts_utc"]`
- `execution_engine.py:1099` → `:1198` — `entry_ts = bar["ts_utc"]` then `trade.entry_ts = entry_ts`
- `execution_engine.py:1374` — `trade.entry_ts = bar["ts_utc"]`

### `TradeEvent(...timestamp=...)` constructions (19 — broader blast than original scope)

`grep -c "TradeEvent(" trading_app/execution_engine.py` = 19. Every one passes `bar["ts_utc"]` or `confirm_bar["ts_utc"]` (raw `pd.Timestamp`) into the `timestamp: datetime` field. Lines: 344, 623, 888, 914, 942, 996, 1016, 1055, 1106, 1132, 1163, 1216, 1234, 1284, 1310, 1339, 1391, 1409, 1561.

### Comparison: paths that DO coerce

`trading_app/entry_rules.py` is the precedent — coerces at lines 142, 207, 288, 360 with `.to_pydatetime()`. So entry_rules is clean; execution_engine and downstream are dirty.

### Downstream impact (operator-visible)

`bot_state.build_state_snapshot._iso_utc(getattr(t, "entry_ts", None))` previously silently None'd `entry_time_utc`, `signal_time_utc`, `exit_time_utc` for every live trade routed through execution_engine.py. F3's `logger.warning` (PR #221) now makes this visible — the warning fires on every live trade until F6 lands.

`paper_trader.py:380` also propagates the wrong type into `JournalEntry.entry_ts: datetime | None` (paper-replay path; not capital-exposure but operator-surface).

## Two fix shapes (decision required from user)

### Shape (a) — Surgical: 3 inline coercions on `trade.entry_ts`, leave `TradeEvent.timestamp` for later

- 3 edits at execution_engine.py:978, 1099, 1374 to add `.to_pydatetime()`
- Closes F3's `_iso_utc` warning for the entry_ts path only
- TradeEvent.timestamp remains pd.Timestamp; downstream consumers (session_orchestrator at line 2107-2108, webhook_server.py:218) continue to receive wrong type
- Lowest blast radius. Highest debt remaining.
- Per institutional-rigor § 3 ("Refactor when you see a pattern of bugs") — this is patch, not fix.

### Shape (b) — Structural: shared `_coerce_to_datetime()` helper applied at all 22 sites

- New helper in `trading_app/entry_rules.py` (or a shared `trading_app/timefmt.py` if we want to avoid the cross-module dep) used at:
  - entry_rules.py:142, 207, 288, 360 (replace inline `.to_pydatetime()`)
  - execution_engine.py:978, 1099, 1374 (3 trade.entry_ts assignments)
  - execution_engine.py:344, 623, 888, 914, 942, 996, 1016, 1055, 1106, 1132, 1163, 1216, 1234, 1284, 1310, 1339, 1391, 1409, 1561 (19 TradeEvent constructions)
- Closes the bug class entirely. New TradeEvent paths cannot regress unless they bypass the helper.
- Per integrity-guardian.md § 2 ("delegate to canonical sources, never re-encode") — this is the rule-aligned shape.
- Higher blast radius: 22 sites + 1 new helper file/symbol.

### Shape (c) — Source coercion: coerce `bar["ts_utc"]` once at `on_bar` ingestion

- Single edit at execution_engine.py:488 (and matching coercion in `on_trading_day_end` and any other bar-ingestion path)
- Mutates the bar dict in place: `bar["ts_utc"] = bar["ts_utc"].to_pydatetime() if isinstance(bar["ts_utc"], pd.Timestamp) else bar["ts_utc"]`
- All 22 downstream sites become correct without per-site edits
- Caveat: mutates caller-owned data; some callers may re-use the bar dict elsewhere with assumptions about Pandas types (need to grep `paper_trader.py` and `session_orchestrator.py` bar-handling paths to verify safe).
- Lowest line count, highest semantic risk.

**Recommendation:** Shape (b). Aligns with integrity-guardian § 2 (delegate to canonical source) and institutional-rigor § 3 (refactor on pattern). Surgical (a) leaves 19 latent bugs; Source (c) has hidden mutation risk. The helper is ~5 lines, the call-site changes are mechanical.

## Severity classification (per adversarial-audit-gate.md)

- **Capital exposure:** NO — the trade still fires; only the timestamp display is wrong.
- **Operator visibility:** YES — `_iso_utc` warning fires on every live trade until fixed; operator dashboard `entry_time_utc` field affected.
- **Truth-layer corruption:** NO — pnl_r and trade outcome math are unaffected (`pd.Timestamp` arithmetic is identical to `datetime` arithmetic).
- **Severity:** **MEDIUM** — operator-visibility class, not capital-exposure class.

Per adversarial-audit-gate.md, MEDIUM does NOT auto-trigger the evidence-auditor pass. But since F3 was a CRIT/HIGH-adjacent fix (silent None on operator surface, near-miss class bug per PR #221 evidence-auditor), it's prudent to dispatch evidence-auditor on the F6 implementation commit anyway. Cheap insurance.

## Required prerequisites — all met (2026-05-05)

1. ✅ Trace each `execution_engine.py:978/1099/1374` to confirm `bar["ts_utc"]` is always a `pd.Timestamp` and never another type. (Verified: bar source is `pd.DataFrame.to_dict()`/`iterrows()` consumer; no other ts_utc producer in the call chain.)
2. ✅ Decide between shape (a), (b), (c). (Recommendation: b. Awaiting user confirmation.)
3. ✅ Audit `paper_trader.py:365/381` (`entry_ts=event.timestamp`) — type derives from `TradeEvent.timestamp` so fixing TradeEvent under shape (b) closes this path automatically.
4. Pending implementation: verify F3's logger.warning fires in test that simulates the broken path, then DOESN'T fire after F6 lands (proves F6 actually fixed it).

## Out of scope for this stage

- F3 _iso_utc helper itself (already shipped PR #221).
- Any change to JournalEntry shape (paper_trader's downstream type stays `datetime | None`; no schema change).
- Refactoring execution_engine.py's bar-handling (Shape (c) considered and recommended-against here).

## Acceptance — when promoted to IMPLEMENTATION

- Shape (b) applied: new helper exported, all 22 sites coerce via the helper.
- F3's `_iso_utc` warning does NOT fire in any test or live run after F6 lands.
- New test asserts `trade.entry_ts` is `datetime` (not `pd.Timestamp`) post-execution.
- New test asserts every `TradeEvent.timestamp` emitted by `on_bar`/`on_trading_day_end` is `datetime` (not `pd.Timestamp`).
- evidence-auditor independent-context pass dispatched on the implementation commit (per recommendation above; not strictly required by severity).
- `python pipeline/check_drift.py` exits 0.
- `pytest tests/test_trading_app/test_execution_engine.py tests/test_trading_app/test_paper_trader.py` green.

## Provenance

- F3 fix that exposed F6: PR #221 (`f871f3f1`), session 2026-05-04.
- Trace session: 2026-05-05, branch `plan/live-trading-rollout-2026-05-05`.
- Companion canonical-source helpers: `entry_rules.py` (lines 142/207/288/360 — already do `.to_pydatetime()`).
- Memory anchor: `feedback_iso_utc_silent_none_class_pattern.md`.

## Implementation gate

Per `.claude/rules/workflow-preferences.md` § Implementation Gating, this stage is DESIGN-resolved but does NOT authorize code edits yet. User must confirm fix shape ((a) / (b) / (c)) before any edits to `trading_app/execution_engine.py` land. The recommendation is (b); change to it requires an explicit "go ship it" / "implement" / "do shape b".
