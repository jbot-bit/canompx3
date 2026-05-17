# Live Trade Throughput Triage — 2026-05-17

**Verdict for all 4 deployed lanes: `UNMEASURED`.**

**Telemetry adjacent finding: `TELEMETRY_BROKEN` on `data/bot_state.json`** (test-fixture contamination of canonical production path).

**Zero filter-loosening recommendations.** Per plan and `RESEARCH_RULES.md`: low N is not a bug unless measured against eligible sessions on a running bot.

---

## Summary

User question: "are enough valid trades firing live?"
Answer: **the bot has not been run live in any operationally meaningful way over the 20-day window covered by signal logs.** No lane can be measured for live throughput because:

1. 16 of 17 `SESSION_START` records across all 9 signal log files are mode `signal_only` (paper). Only one `live`-mode session start exists (`2026-05-15T20:26:47Z`, MNQ), and even that session produced zero entry signals.
2. **Zero `SIGNAL_ENTRY` records and zero `ORDER_ENTRY` records exist in any of the 9 log files** (2026-04-27 through 2026-05-15). The full record-type inventory across all files is: 17 × `SESSION_START`, 1 × `ENTRY_BLOCKED_PAUSED`, 1 × `SIGNAL_EXIT` = 19 records total over ~20 calendar days.
3. `data/bot_state.json` is a test-fixture snapshot — literal `<MagicMock name='...' id='...'>` strings are serialized into the canonical state JSON (6 occurrences). The instrument field reads `MGC`, single lane `TEST_STRAT_001`, trading_day `2026-03-07`. **This is not real bot state.**
4. `live_journal.db` contains 2 rows total ever: one `demo` record (2026-03-25, `E2E_TEST_STRATEGY`) and one `signal` record (2026-04-06, `MNQ_EUROPE_FLOW_E2_RR3.0_CB1_COST_LT10` — not one of the currently-deployed 4 lanes).

The expected stage exit per plan v3 is met: **Stage 1 exit (b) — "paths resolve, files absent or empty → write `UNMEASURED` report, stop."**

---

## Stage 1 — Runtime state located

### Paths resolved (verified, not assumed)

| Telemetry | Resolved path | Source | Status |
|---|---|---|---|
| Signal logs | `<repo_root>/live_signals_YYYY-MM-DD.jsonl` | `session_orchestrator.py:296` `SIGNALS_DIR` (via `Path(__file__).resolve().parents[2]`) | 9 files present, 145–758 bytes each |
| Bot state | `<repo_root>/data/bot_state.json` | `bot_state.py:19` `STATE_FILE = Path(__file__).parent.parent.parent / "data" / "bot_state.json"` | Present (3112 bytes, mtime 2026-05-17 19:36) — **but contaminated** |
| Trade journal | `<repo_root>/live_journal.db` | `trade_journal.py:62` `TradeJournal(db_path)`; instance opened by `session_orchestrator.py` (path passed in) | Present (1.3 MB, mtime 2026-05-04) |
| Lane allocation | `docs/runtime/lane_allocation.json` | direct read | 4 DEPLOY lanes, 831 paused |

### Signal log record inventory (all 9 files combined)

```
Total records: 19
Types: SESSION_START=17, ENTRY_BLOCKED_PAUSED=1, SIGNAL_EXIT=1
Modes: signal_only=16, live=1, (absent)=2
Strategy IDs seen with explicit ID: MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12 (2 records)
```

**Three of four deployed lanes never appear in any signal log file by `strategy_id`** — neither `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_2K`, `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15`, nor `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25`. They might be implicit in unstructured `SESSION_START` events, but no entry-class record carries their ID.

### `live_journal.db` contents (DuckDB direct read)

```
live_trades total rows: 2
  (2026-04-06, MNQ, MNQ_EUROPE_FLOW_E2_RR3.0_CB1_COST_LT10, long, signal,  2026-04-06 18:06 Brisbane)
  (2026-03-25, MNQ, E2E_TEST_STRATEGY,                       long, demo,    2026-03-25 18:47 Brisbane)
```

The Apr 6 row predates the 2026-05-14 rebalance; the strategy_id is not in the current `lanes[]`. The Mar 25 row is an end-to-end-test fixture.

### Bot state contamination

`data/bot_state.json` is structured like real bot state (correct schema: `mode`, `instrument`, `lanes`, `feed_status`, etc.) but is filled with MagicMock objects coerced to string by `bot_state.write_state`'s `json.dumps(data, default=str, ...)` call (`bot_state.py:28`). Excerpts:

```json
"daily_pnl_r": "<MagicMock name='mock.daily_pnl_r.__round__()' id='2436726140816'>",
"orb_break_direction": "<MagicMock name='mock.orbs.get().break_dir' id='2436726137792'>",
...
"strategies_loaded": 1, "lanes": { "TEST_STRAT_001": { "instrument": "MGC", ... } },
"trading_day": "2026-03-07"
```

The trading_day (`2026-03-07`) is 70+ days in the past; the lane (`TEST_STRAT_001`, MGC instrument) is not deployed. A real bot run would overwrite this file via `os.replace` (atomic). The fact that it persists shows **no real bot run has happened since this test snapshot was written**.

This is `TELEMETRY_BROKEN` not on the lanes themselves but on the canonical state surface: a future operator inspecting `bot_state.json` would see a fake MGC TEST_STRAT_001 lane in DEMO mode. **Reported as a separate finding; does not change lane verdict (still UNMEASURED).**

---

## Stages 2–4: Not executed — gated by Stage 1 verdict

Per plan v3:

> Exit conditions (one of, not a fallback):
> - (a) Live logs + uptime found → proceed to Stage 2.
> - (b) Paths resolve, files absent or empty → write `UNMEASURED` report, **stop**.

Running Stage 2 (signal-generation upstream check) or Stage 3 (Expected vs Observed cluster-aware binomial) against `observed=0` over a window where the bot was in `signal_only` mode would falsely classify the lanes as `INVALID_SPARSE` when in fact they were never given the opportunity to fire under live conditions. That is exactly the failure mode the plan calls out.

---

## Historical baseline (read-only, for reference only)

Pulled from `validated_setups` so the user can see what fire-rate the lanes *would* be expected to produce **IF** the bot were running live during MNQ session windows. **Not a benchmark gate** (no `eligible_days` column in current schema; per-lane day-coverage computation requires session-window joins not done here).

| strategy_id | sample_size | first_trade_day | last_trade_day | trade_day_count | trades/mo (span avg) |
|---|---:|---|---|---:|---:|
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_2K | 1418 | 2019-05-07 | 2026-05-15 | 1532 | 16.6 |
| MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15 | 744 | 2019-05-06 | 2026-04-23 | 936 | 8.8 |
| MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12 | 1508 | 2019-05-06 | 2026-05-15 | 1775 | 17.6 |
| MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25 | 1512 | 2019-05-06 | 2026-05-15 | 1741 | 17.7 |

Three of four lanes would meet `CORE_INCOME` tier (≥10/mo) historically. `VWAP_MID_ALIGNED_O15` would land in `RARE_QUALITY` tier (~9/mo). **None of this is grounds for action while the lanes are UNMEASURED.** Per Harris Ch22§22.6 deflation prior (`docs/institutional/literature/harris_2002_trading_exchanges_microstructure.md`), live ExpR will be lower than paper baseline regardless — this is reported here only so the operator does not panic when live fires materialize at less than the paper baseline.

---

## Per-lane verdicts

| Lane | Verdict | One-line justification |
|---|---|---|
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_2K | `UNMEASURED` | Zero `SIGNAL_ENTRY` records; lane never appears in any signal log by id; bot not run in live mode |
| MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15 | `UNMEASURED` | Zero `SIGNAL_ENTRY` records; lane never appears in any signal log by id; bot not run in live mode |
| MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12 | `UNMEASURED` | 2 records in logs (1 `ENTRY_BLOCKED_PAUSED`, 1 `SIGNAL_EXIT`) — both pre-rebalance signal_only mode; no live entries |
| MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25 | `UNMEASURED` | Zero `SIGNAL_ENTRY` records; lane never appears in any signal log by id; bot not run in live mode |

---

## Intended-stop counts (reported separately per plan)

Per plan: **intended stops are policy compliance, NOT throughput loss**. Reporting separately so user sees what the Topstep 50K legacy rule set (`resources/prop-firm-official-rules.md`) would have done if firing.

Across all 9 signal log files:

| Gate | Count | Class |
|---|---:|---|
| `ENTRY_BLOCKED_PAUSED` | 1 | Intended (lane in pause state — lifecycle policy) |
| `KILL_SWITCH` | 0 | Intended (capital protection) |
| `ENTRY_BLOCKED_DD_HALT` (Topstep trailing-DD) | 0 | Intended (Topstep $2,500 trailing DD — see `session_orchestrator.py:2273`) |
| `CME_HOLIDAY` | 0 | Intended (calendar) |
| `POST-MARKET BUFFER` | 0 | Intended (session geometry) |
| `ENTRY_BLOCKED_ORPHAN` | 0 | Intended (lifecycle) |
| `ORB_CAP_SKIP` | 0 | Filter-narrowness (informational) |
| `MAX_RISK_SKIP` | 0 | Filter-narrowness (informational) |
| Infra-failure class (`CIRCUIT_BREAKER`, `SHADOW_DIVERGENCE`, `F-2b`, `REJECT`, `EXIT_FAILED`) | 0 | Would flag `EXECUTION_BLOCKED` |

All zero. The bot has not produced enough live events for any gate (intended, filter, or infra) to actually fire.

---

## Recommendations

### 1. Run a signal-only paper round of ≥30 distinct MNQ trading_days BEFORE re-running this diagnostic

Per plan v3: "Recommendation: signal-only paper run for ≥30 distinct trading_days before re-running."

This satisfies the `n_unique_trading_days ≥ 30` clustered-SE floor (`feedback_n_unique_trading_days_floor_clustered_se.md`). At MNQ's ~21 trading days/month, this is ~6 weeks of continuous bot uptime.

During this run, monitor:
- `SIGNAL_ENTRY` record cadence per lane (expected ~8–18/mo based on historical baseline above).
- `bot_state.json` heartbeat freshness (`heartbeat_utc` field; should refresh every bar).
- `live_journal.db` row growth (one row per entry-class event regardless of broker submission).

Once ≥30 distinct trading_day clusters are accumulated per lane, re-run this triage. Stage 2 (signal-generation upstream check) and Stage 3 (cluster-aware binomial) can then execute meaningfully.

### 2. Clean the `data/bot_state.json` test contamination — separate from any lane decision

The file at `data/bot_state.json` is a test fixture. Two remediation paths, either fine:

**Option A (lowest blast radius — recommended for this triage):** delete the file.
```bash
rm C:/Users/joshd/canompx3/data/bot_state.json
```
The next live or paper bot run will recreate it via `bot_state.write_state` (atomic write). Dashboard `read_state` already handles missing-file gracefully (returns `{}`, `bot_state.py:42`).

**Option B (root-cause fix, deferred — not in this triage's scope):** harden `bot_state.write_state` to validate types before serializing. Currently `json.dumps(data, default=str, ...)` silently coerces any unknown type. Tests using mocks must monkeypatch `STATE_FILE` to a tempdir; if any test calls `write_state(mock_data)` without that patch, the canonical production path gets polluted. This is a separate engineering finding that warrants its own stage (touches `trading_app/live/bot_state.py` — production code, NOT trivial).

A `grep -rn "write_state\|monkeypatch.*bot_state" tests/` pass will identify the offending test. Out of scope for this read-only diagnostic.

### 3. Do NOT modify lane filters, gate thresholds, or `lane_allocation.json` based on this triage

Per `RESEARCH_RULES.md` and explicit user direction: low N is not a bug. The lanes have not been measured under live conditions. Any filter/threshold change here would be filter-loosening to chase a phantom signal-throughput problem that has not been demonstrated to exist.

### 4. Confirm whether the bot is intended to run continuously or on-demand

The pattern of one ~live session ever, surrounded by sparse signal_only test runs, suggests the bot is being manually started occasionally rather than running continuously. If continuous-uptime live trading is the intent, the operator needs to (a) confirm `multi_runner.py` is invoked with `signal_only=False`, (b) confirm broker/account credentials are loaded for `projectx`/Topstep, (c) confirm `data/bot_state.json` is being overwritten by real heartbeats (not test artifacts), and (d) confirm uptime survives across CME session boundaries.

---

## What this triage did NOT do (per plan scope-lock)

- Did not edit any production code in `pipeline/` or `trading_app/`.
- Did not modify `lane_allocation.json`, gate thresholds, filter parameters, or any allocator field.
- Did not tune any validation rule to increase trade count.
- Did not deploy, restart, or interact with a live broker account.
- Did not write under `docs/audit/results/` (research-audit-only directory).
- Did not delete the contaminated `bot_state.json` file (recommended above, but execution requires explicit user OK).
- Did not run Stage 2 (bar-cadence/tick-presence) or Stage 3 (cluster-aware binomial) — gated by Stage 1 exit (b).
- Did not pull MCP `get_strategy_fitness` for Stage 4 — fitness is only meaningful when paired with observed live trades; here observed=0 so the fitness comparison would be uninformative.

---

## Grounding sources cited

- `resources/prop-firm-official-rules.md` — Topstep 50K legacy DD rules; informed the intended-stop classification (zero fires observed; reported for completeness).
- `docs/institutional/literature/harris_2002_trading_exchanges_microstructure.md` Ch22§22.6 — deflation prior cited in "Historical baseline" caveat (so a future live-fire ExpR below paper baseline is not over-interpreted as decay).
- `pipeline.paths.GOLD_DB_PATH` — DuckDB direct-read for `validated_setups` baseline (MCP not used; not gated by MCP availability per `feedback_mcp_venv_drift_cryptography47.md`).
- `docs/runtime/lane_allocation.json` — current 4 DEPLOY lanes, 831 paused.
- `docs/runtime/chordia_audit_log.yaml` — present (34 KB, 2026-05-14), not parsed for this triage (deferred to Stage 4 when applicable).

## Re-run guidance

When ≥30 distinct trading_days of live bot uptime accumulate:
1. `cat C:/Users/joshd/canompx3/live_signals_*.jsonl | python -c "..."` to re-check signal record cadence.
2. Re-read this plan; if `SIGNAL_ENTRY` records exist, proceed to Stage 2 (`bar_aggregator.py` last-bar timestamps, `live_market_state.py` tick presence, session-start coverage vs `pipeline.dst.SESSION_CATALOG`).
3. Stage 3 cluster-aware binomial at trading_day with `n_unique_trading_days ≥ 30` floor.
4. Stage 4 `get_strategy_fitness` MCP call + `chordia_audit_log.yaml` direct read + CUSUM/SR state (`current_sr_stat` not `sr_stat`; `feedback_sr_monitor_peak_vs_current_misread.md`).
5. Stage 5 — re-classify into one of six buckets and write a new diagnostic dated for that day.
