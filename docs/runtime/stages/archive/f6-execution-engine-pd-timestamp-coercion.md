---
task: f6-execution-engine-pd-timestamp-coercion
mode: FALSIFIED
status: CLOSED-FALSIFIED-2026-05-06
agent: claude
updated: 2026-05-06
scope_lock:
  - docs/runtime/stages/f6-execution-engine-pd-timestamp-coercion.md
  - memory/feedback_phantom_stage_doc_unverified_premise.md
  - memory/MEMORY.md
blast_radius: |
  Doc-only closeout of a phantom stage. Falsifies the premise that
  trading_app/execution_engine.py routes raw pd.Timestamp into _iso_utc's
  warning branch. No production code is touched. Reads:
  trading_app/live/bot_state.py, trading_app/live/bar_aggregator.py,
  trading_app/paper_trader.py, tests/test_trading_app/test_session_orchestrator.py,
  commit 9ba25af4. Writes: this file + 1 memory feedback file + 1 MEMORY.md
  index line. Zero callers, zero importers, zero schema impact.
---

# F6 — execution_engine.py raw pd.Timestamp on trade.entry_ts and TradeEvent.timestamp

**Status:** **FALSIFIED-CLOSED (2026-05-06).** The premise that motivated this stage was wrong. Every load-bearing claim has been disproven by file:line evidence, runtime trace, and an existing canonical regression test that asserts the opposite. **No production-code edits are required.** The 26-site coercion fix recommended by the prior version of this doc would fix nothing.

The original DESIGN body is preserved verbatim in § "Original claims (preserved as quotes)" below for audit-trail integrity.

---

## Why this is the verdict — falsification evidence

### F1. `pd.Timestamp` IS-A `datetime`. The warning branch is unreachable for that input.

`trading_app/live/bot_state.py:91`:

```python
if isinstance(value, datetime):
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()
log.warning("_iso_utc: unsupported type %s — returning None", type(value).__name__)
return None
```

Runtime verification (this session, `.venv/Scripts/python`):

```
>>> from datetime import datetime
>>> import pandas as pd
>>> isinstance(pd.Timestamp('2026-01-01', tz='UTC'), datetime)
True
```

`pandas._libs.tslibs.timestamps.Timestamp` extends `datetime` directly (verified MRO, this session). A raw `pd.Timestamp` flowing into `_iso_utc` takes the **datetime branch** (line 91-94), produces a valid ISO string, and never reaches the `log.warning` at line 95. The original F6 premise — "F3's logger.warning fires on every live trade until F6 lands" (line 43 of the DESIGN doc, preserved below) — is false.

### F2. The bar source is `datetime`, not `pd.Timestamp`. Both live and backtest paths verified.

**Live path:** `trading_app/live/bar_aggregator.py:21` annotates `Bar.ts_utc: datetime`. The aggregator constructs Bars from minute-truncated `datetime` values it generated itself; no pandas object enters the chain. `Bar.as_dict()` (line 29) writes `self.ts_utc` directly into the dict that `ExecutionEngine.on_bar` consumes. Source-side type is `datetime`, not `pd.Timestamp`.

**Backtest/replay path:** `trading_app/paper_trader.py:204-230` (`_get_bars_for_day`). Fetches rows from DuckDB and constructs the bar dict as `{"ts_utc": r[0], ...}`. DuckDB's Python driver returns Python `datetime` for `TIMESTAMPTZ` columns. Verified in this session:

```
>>> con.execute('SELECT ts_utc FROM bars_1m LIMIT 1').fetchone()[0]
datetime.datetime(2021, 7, 12, 10, 33, ..., tzinfo=...)
>>> type(...).__name__ == 'datetime'
True
```

The DESIGN doc's claim (line 25) — *"bar is a dict with `ts_utc` always a `pd.Timestamp` (DataFrame `.iterrows()`/`.to_dict()` preserves Pandas types)"* — assumed a DataFrame-iterating pathway that does not exist on either entry point into `ExecutionEngine.on_bar`. No `iterrows()` or `to_dict()` call is in the chain. (Search for `bar["ts_utc"] =` across the repo found zero production-code assignments of pandas types into the bar dict; `pd.Timestamp(...)` constructions in `outcome_builder.py:81/277` and `entry_rules.py:191` are RHS scalars used in DataFrame mask comparisons, never written into a bar.)

### F3. F3 commit (`9ba25af4`) explicitly says "Defensive-only — does NOT fix upstream class-bug" and tests the *real* trigger.

Commit `9ba25af4` message body, verbatim:

> Defensive-only — does NOT fix upstream class-bug where
> execution_engine.py:978/1099/1374 assigns raw pd.Timestamp to
> trade.entry_ts via 'trade.entry_ts = bar["ts_utc"]' without
> .to_pydatetime(). That's tracked as a separate finding (F6) requiring
> its own IMPLEMENTATION stage with adversarial-audit gate (live-trading
> exposure path).

The author of F3 *also* believed the pd.Timestamp class-bug existed — but their tests prove the actual triggering input was `str` and `int`, not `pd.Timestamp`:

`tests/test_trading_app/test_session_orchestrator.py:411-431`,
`test_iso_utc_unsupported_type_warns_and_returns_none`:

```python
with caplog.at_level(logging.WARNING, logger="trading_app.live.bot_state"):
    assert _iso_utc("2026-03-07T22:15:00+00:00") is None
assert "_iso_utc: unsupported type str" in caplog.text

caplog.clear()
with caplog.at_level(logging.WARNING, logger="trading_app.live.bot_state"):
    assert _iso_utc(1234567890) is None
assert "_iso_utc: unsupported type int" in caplog.text
```

The two types F3 actually fired on are `str` and `int`. `pd.Timestamp` is not in the tested set because `pd.Timestamp` does not trigger the warning branch.

### F4. The smoking gun — a canonical regression test asserts the OPPOSITE of the F6 premise.

`tests/test_trading_app/test_session_orchestrator.py:1158-1175`,
`test_on_bar_feed_status_handles_pandas_timestamp`:

```python
async def test_on_bar_feed_status_handles_pandas_timestamp(self, orch):
    """``pd.Timestamp`` is a ``datetime`` subclass, so it must route
    through the datetime branch of ``_iso_utc`` and produce a valid
    ISO string — NOT trigger the unsupported-type warning. Documents
    the F6 root-cause boundary: pd.Timestamp arriving at this
    callsite is correct-by-default; the upstream coercion gap (F6)
    is at execution_engine.py, not here."""
    import pandas as pd

    pd_ts = pd.Timestamp("2026-03-07T22:15:00", tz="UTC")
    ...
    bar = FakeBar(ts_utc=pd_ts)
    await orch._on_bar(bar)

    assert orch._feed_status["last_bar_utc"] == "2026-03-07T22:15:00+00:00"
```

The test in the same file as F3's regression suite **explicitly asserts that pd.Timestamp passes through silently and produces a valid ISO string**. The same test's docstring even references "F6 root-cause boundary" — a phrase the F6 DESIGN-doc author would have read had they grepped the test file before claiming a class bug.

### F5. The F3 actual contamination class is string-typed timestamps.

`tests/test_trading_app/test_session_orchestrator.py:1118-1142`,
`test_on_bar_feed_status_routes_last_bar_utc_through_iso_utc`:

```python
bar = FakeBar()
bar.ts_utc = "2026-03-07T22:15:00+00:00"  # type-mismatch: str, not datetime

with caplog.at_level(logging.WARNING, logger="trading_app.live.bot_state"):
    await orch._on_bar(bar)

assert orch._feed_status["last_bar_utc"] is None
assert "_iso_utc: unsupported type str" in caplog.text
```

The test fixture explicitly types `bar.ts_utc = "<iso string>"`. This is the actual class of bug F3 was defending against — a string from a hypothetical buggy upstream coercion path.

---

## Step-1 trace — all `ts_utc =` writes across the repo (production / test / research)

Grep performed this session: `ts_utc\s*=` across all `.py` files. Categorized:

**Production code (trading_app/, pipeline/) — `ts_utc =` writes into a bar/dict:**
- `trading_app/live/bar_aggregator.py:169` — `ts_utc=minute,` — `minute` is a `datetime` (live path; see file:148-167).
- `trading_app/paper_trader.py:224` — `"ts_utc": r[0],` — `r[0]` is `datetime` from DuckDB (verified above).
- No other production-code site writes `ts_utc` into a bar dict. Zero `bar["ts_utc"] = pd.Timestamp(...)` assignments anywhere in the codebase.

**Test fixtures — datetime-typed (correct):**
- `tests/test_trading_app/test_bar_persister.py:38, 104` — `ts_utc=datetime(2026, 4, 13, ..., tzinfo=timezone.utc)`.
- `tests/test_trading_app/test_projectx_feed.py:105` — same.
- `tests/test_trading_app/test_bar_aggregator.py:181, 186, 191, 196` — `ts_utc=_ts(0)` (datetime).
- `tests/test_trading_app/test_session_orchestrator.py:854-855, 1153, 1172, 2316, 2369` — `datetime` or `pd.Timestamp` — both correct (pd.Timestamp IS-A datetime).

**Test fixtures — string-typed (synthetic contamination probes for F3):**
- `tests/test_trading_app/test_session_orchestrator.py:1136` — `bar.ts_utc = "2026-03-07T22:15:00+00:00"`. **Synthetic only.** This is a deliberately broken fixture used to verify F3's defensive helper fires its warning. The fixture has no upstream production analog — no real bar source assigns a string into `bar.ts_utc`.

**Research scripts (out of scope for live trading):** several `analyze_*.py` / `research_*.py` files in `research/` rebuild local DataFrames with `ts_utc = ts.dt.tz_convert("UTC")` etc. These are research-side dataframes, not live-engine bar dicts, and never feed `ExecutionEngine.on_bar`. Out of scope.

**Conclusion:** No real production source of string-typed `bar.ts_utc` exists. F3 is correctly classified as defensive — it catches a class of error that does not currently occur in production. Removing the warning would be unsafe (it's a defense-in-depth signal); keeping it is correct as-shipped. **No F6-prime stage opens.**

---

## Why this happened — procedural lesson

The DESIGN doc was 119 lines, cited `institutional-rigor.md` § 3 and § 6 correctly, classified severity correctly under `adversarial-audit-gate.md`, and proposed a structurally clean shape (b) refactor delegating to a canonical helper per `integrity-guardian.md` § 2. Every meta-rule citation was correct. The premise was wrong.

The author wrote the doc by **reading code** (file:line traces of `bar["ts_utc"]` assignments) without **running it** (no isinstance check, no test grep, no DuckDB roundtrip). Specifically, the DESIGN doc never:

1. Ran `isinstance(pd.Timestamp(...), datetime)` to verify the warning branch was reachable for the claimed input. Two-line REPL.
2. Grepped `tests/test_trading_app/test_session_orchestrator.py` for `pd.Timestamp` or `pandas_timestamp` — would have surfaced the smoking-gun test (line 1158) whose docstring contradicts the F6 premise by name.
3. Read F3 commit `9ba25af4`'s test file diff — would have shown the actual fired types are `str` and `int`.
4. Verified the bar source type via DuckDB execution. The DESIGN doc speculated about `iterrows()`/`to_dict()` chains that do not exist on either bar entry point.

This is the `adversarial-audit-gate.md` § Proof case (iter 174 F4 kill-switch race) replayed in the design phase: **the implementer's mental model is the same one that produced the misdiagnosis, and self-review on a non-running premise reproduces the error.** F4 caught a real bug late; F6 would have shipped 26 production-code edits to fix nothing — pure debt institutionally blessed by rule citations.

The cost of this was capped only because the user asked the verification question ("Stop. Prove this is true before using it") before the IMPLEMENTATION promotion. Without that gate, 26 edits to capital-routing code would have landed under "rule-aligned shape (b)" framing.

---

## Original claims (preserved as quotes — DESIGN body verbatim)

For audit-trail integrity, the five load-bearing claims (C1-C5) from the prior DESIGN body are preserved here. Each is annotated with its falsification reference above.

> **C1.** "`trading_app/execution_engine.py` assigns raw `pd.Timestamp` to both `trade.entry_ts` (3 sites) AND every `TradeEvent.timestamp` it emits (~19 sites). The contract on `TradeEvent.timestamp: datetime` (line 62) is silently violated because `pd.Timestamp` subclasses `datetime` — `isinstance` check passes, `.isoformat()` produces wrong-shaped strings."

→ **FALSIFIED by F1 + F4.** `pd.Timestamp.isoformat()` produces a valid `datetime.isoformat()`-shaped string (test line 1175 asserts `"2026-03-07T22:15:00+00:00"`). The "wrong-shaped strings" claim has no execution evidence and the canonical regression test asserts the opposite.

> **C2.** "`ExecutionEngine.on_bar(self, bar: dict)` (line 480) — bar is a dict with `ts_utc` always a `pd.Timestamp` (DataFrame `.iterrows()`/`.to_dict()` preserves Pandas types)."

→ **FALSIFIED by F2.** Both bar entry points (`bar_aggregator.Bar.ts_utc: datetime`, `paper_trader._get_bars_for_day` returning DuckDB `datetime`) source `datetime`, not `pd.Timestamp`. No `iterrows()`/`to_dict()` call exists in the chain.

> **C3.** "`bot_state.build_state_snapshot._iso_utc(getattr(t, "entry_ts", None))` previously silently None'd `entry_time_utc`, `signal_time_utc`, `exit_time_utc` for every live trade routed through execution_engine.py. F3's `logger.warning` (PR #221) now makes this visible — the warning fires on every live trade until F6 lands."

→ **FALSIFIED by F1 + F3 + F4.** Even if `pd.Timestamp` did flow through `entry_ts`, it takes the datetime branch and produces a valid ISO string. The warning does not fire. (No live-trade log evidence was attached to the DESIGN doc; the claim was speculation.)

> **C4.** "Per institutional-rigor § 3 (\"Refactor when you see a pattern of bugs\") — this is patch, not fix. ... Per integrity-guardian § 2 (\"delegate to canonical sources, never re-encode\") — this is the rule-aligned shape."

→ **MISAPPLIED.** The rule citations are correct in the abstract; they were applied to a non-existent bug. Rule citation does not validate premise. Per `integrity-guardian.md` § 5: *"Generation is not validation. No LLM output is trusted until verified with execution evidence. Trace the execution path (file:line → call → file:line) before claiming a bug exists. Confident wrong findings are worse than no findings."* The DESIGN doc's confident wrongness, dressed in correct rule citations, is the exact failure mode that rule warns against.

> **C5.** "**Recommendation:** Shape (b). ... Higher blast radius: 22 sites + 1 new helper file/symbol."

→ **REJECTED.** The proposed 26 edits (3 trade.entry_ts + 19 TradeEvent.timestamp + 4 entry_rules.py inline replacements) would fix nothing. They would consolidate `.to_pydatetime()` calls that are themselves correct as-shipped (the `entry_rules.py` ones operate on `bars_df["ts_utc"]` which IS a pd.Timestamp series — correct site to coerce). Touching capital-routing code with no bug to fix is pure debt.

---

## What survives, what dies, what is conditional

**Dies (rejected):**
- The "22-site class bug" framing.
- The 26-edit Shape (b) coercion fix.
- The Shape (a) and Shape (c) alternatives (predicated on the same false premise).
- Any future stage doc claiming a class bug without executing the isinstance / runtime-trace check first.

**Survives (correct as-shipped, do not touch):**
- F3's `bot_state._iso_utc` defensive warning helper. Catches real `str`/`int` contamination if any upstream regresses; documented and tested.
- All 14 existing `.to_pydatetime()` sites in `outcome_builder.py` / `entry_rules.py` / `build_daily_features.py` / `market_calendar.py`. These coerce pandas-source values where the source IS a pd.Timestamp series — correct sites, do not consolidate.
- `test_on_bar_feed_status_handles_pandas_timestamp` — the canonical regression guard. Do not delete or weaken.
- `test_iso_utc_unsupported_type_warns_and_returns_none` — F3's coverage of the actual `str`/`int` trigger types.

**Conditional (no follow-up needed):**
- F6-prime would have opened only if Step 1 trace found a non-fixture string source for `bar.ts_utc`. None found. F6-prime does NOT open.

---

## Verdict

F6 is **CLOSED-FALSIFIED**. No production code is modified. F3's defensive helper is correct as-shipped. The procedural lesson is captured in `memory/feedback_phantom_stage_doc_unverified_premise.md`.

---

## Provenance

- DESIGN doc opened: 2026-05-04 (prior session, branch `plan/live-trading-rollout-2026-05-05`).
- DESIGN doc trace claimed complete: 2026-05-05.
- User triggered verification ("Stop. Prove this is true before using it"): 2026-05-06.
- Falsification this session: 2026-05-06.
- F3 commit (cited): `9ba25af4` (PR #221).
- Memory anchor: `memory/feedback_phantom_stage_doc_unverified_premise.md`.

---

## evidence-auditor verdict

Per `adversarial-audit-gate.md` § Actor: the falsification record's central claim ("no bug exists on this path; F3's defensive helper is correct as-shipped") is itself a verifiable claim, subject to the same gate the original F6 would have triggered had it been real. Skipping the audit on a falsification reproduces the failure mode being closed.

**Auditor verdict: PASS** (independent-context evidence-auditor pass, 2026-05-06).

Independently verified findings (file:line):

- **C1 (3-site pd.Timestamp claim) — FALSIFIED.** Auditor traced all three `trade.entry_ts =` sites:
  - `trading_app/execution_engine.py:978` — `trade.entry_ts = confirm_bar["ts_utc"]` (E2 path)
  - `trading_app/execution_engine.py:1099` → `:1198` — `entry_ts = bar["ts_utc"]` then `trade.entry_ts = entry_ts` (E1 path)
  - `trading_app/execution_engine.py:1374` — `trade.entry_ts = bar["ts_utc"]` (E3 path)

  All three source `bar["ts_utc"]` from `Bar.as_dict()` (live: `Bar.ts_utc: datetime`) or `paper_trader._get_bars_for_day` (DuckDB TIMESTAMPTZ → Python `datetime`). No pandas type enters this chain. No fourth assignment site exists.

- **C2 (DataFrame iterrows/to_dict source) — FALSIFIED.** Neither bar-feeding path uses `iterrows()` or `to_dict()`. Auditor confirms.

- **C3 (warning fires on every live trade) — FALSIFIED.** `pd.Timestamp` IS-A `datetime`; the `isinstance` predicate at `bot_state.py:91` returns True; line 95 warning is unreachable for that input.

- **C4 (rule citations validate premise) — MISAPPLIED.** Citations correct in the abstract; applied to a non-existent bug. `integrity-guardian.md` § 5 explicitly forbids this.

- **C5 (26 edits required) — REJECTED.** All three `trade.entry_ts` sites source `datetime`. The 14 existing `.to_pydatetime()` sites in `entry_rules.py:207/288/360` and others operate on pandas Series where coercion IS correct and necessary. Consolidating them would change correct behavior for no gain.

- **C6 (test transitivity) — DOWNGRADED TO NON-ISSUE.** Auditor verified `_iso_utc` has a single type-dispatch predicate (`isinstance(value, datetime)`) with no callsite-specific logic. The test at `test_session_orchestrator.py:1158` (`test_on_bar_feed_status_handles_pandas_timestamp`) transitively covers `build_state_snapshot:187, 206` `_iso_utc(getattr(t, "entry_ts", None))` because `_iso_utc` is a pure bottleneck. Type-uniformity argument is structurally valid.

- **E2 entry_rules path independently checked.** `entry_rules.py:407` `entry_ts=touch.touch_bar_ts` feeds `EntrySignal.entry_ts` (typed `datetime | None`), populated at `entry_rules.py:207` via explicit `.to_pydatetime()`. Live-engine E2 path does NOT read `EntrySignal.entry_ts` — it re-derives from `confirm_bar["ts_utc"]` directly. No unaudited path.

**Critical issues:** None. No production bugs exist on this path.

**Silent gaps (low-risk, non-blocking):** No direct test of `build_state_snapshot` with `pd.Timestamp` `trade.entry_ts` (redundant given type-uniformity). `execution_engine.py:1099→1198` assigns `entry_ts` 12 lines before writing to `trade.entry_ts` — future refactor risk, no current evidence of problem.

**Tests missing (optional, not blocking):** Direct round-trip test `confirm_bar["ts_utc"] → trade.entry_ts → build_state_snapshot` would pin the full live-trade → dashboard chain.

**Do-not-touch:**
- `trading_app/live/bot_state.py:70-96` `_iso_utc` — defensive helper correct as-shipped.
- `tests/test_trading_app/test_session_orchestrator.py:1158` `test_on_bar_feed_status_handles_pandas_timestamp` — canonical regression guard for the falsification claim. Do not delete or weaken.
- All 14 `.to_pydatetime()` sites in `entry_rules.py:207/288/360`, `outcome_builder.py`, `build_daily_features.py`, `market_calendar.py` — coerce pandas Series at correct sites.

**Highest-priority fix:** None. F6 is CLOSED-FALSIFIED. No production code is touched. The auditor concurs with the closeout and adds no reopening conditions.
