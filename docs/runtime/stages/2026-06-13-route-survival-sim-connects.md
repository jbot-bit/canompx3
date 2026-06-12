task: Route the 3 raw read-only duckdb.connect() sites in trading_app/account_survival.py (the prop-survival-sim / sweep consumers, NOT the EOD nightly chain) through pipeline.db_connect.open_read_only_with_retry, and add a dedicated drift guard (check_no_raw_duckdb_connect_in_survival_sim) so the conversion cannot silently regress. Off-plan-but-valid follow-up to the EOD-chain guard (commit 9fefd9fd); the original dashboard/corpus stage stays open and incomplete.

mode: IMPLEMENTATION

stage: 1 of 1

## Scope Lock

- trading_app/account_survival.py
- pipeline/check_drift.py
- tests/test_pipeline/test_check_drift.py

## Blast Radius

- account_survival.py — 3 read-only connects (:729/:1364/:1514) swapped duckdb.connect(str(db), read_only=True) -> open_read_only_with_retry(str(db)). These are batch survival-sim / sweep paths (_load_profile_daily_scenarios, sweep_survival_cap, evaluate_per_account_survival) reached via CLI + live_readiness_report — NOT the interactive SSE dashboard, so the wrapper's default budget (attempts=6, max_delay=30s) is correct (no sub-7s interactive constraint). read_only is implicit in the wrapper. `import duckdb` stays — still used for duckdb.DuckDBPyConnection type hints (:387/:406/:468/:546); only the .connect( calls change. Read-only, no write-path change, no signature change -> no caller updates.
- check_drift.py — ADD check_no_raw_duckdb_connect_in_survival_sim() delegating to the shared _raw_duckdb_connect_violations() matcher (no new matcher — avoids inline-copy parity-drift class n>=3). Explicit 1-file allowlist (account_survival.py). Register ONE new entry in CHECKS; len(CHECKS) grows by 1 (computed dynamically, never hardcoded). requires_db=False, no DB read. Third sibling of the existing live/eod_chain guards — accurate scope label (account_survival is NOT EOD, so it does NOT belong in the eod_chain allowlist).
- test_check_drift.py — ADD TestNoRawDuckdbConnectInSurvivalSim mirroring TestNoRawDuckdbConnectInEodChain: real-clean PASS + 4 injection cases (direct / aliased / from-import / wrapper-under-from-import) + missing-file no-crash. Proves the guard guards (integrity-guardian §7).
- Reads/Writes to gold.db: none (drift check requires_db=False; the swaps are read-only opens with identical semantics + retry).
- Fail direction: false-BLOCK = documented annoyance (a legitimate raw connect helper lands in account_survival.py -> route it through the wrapper anyway); only false-PASS = a raw form the regex misses — closed by the injection tests.
- Does NOT touch trading_app/live/ write-path -> does NOT trip the adversarial-audit gate (read-only swap + test + drift only). account_survival is a prop-survival-sim consumer per self-funded-sizing-doctrine.md (read-only historical-scenario access).

## Approach

Swap the 3 read opens to open_read_only_with_retry (default budget — batch paths, not interactive). Add `from pipeline.db_connect import open_read_only_with_retry` alongside the existing `import duckdb` (kept for type hints). Add a dedicated survival-sim guard reusing the shared matcher rather than widening the eod_chain allowlist (which would mislabel scope). Mirror the eod_chain test class. Engineering decision — UNSUPPORTED-by-local-lit (no DB-concurrency entry in the quant-research corpus); labelled as such, not dressed in a fabricated cite.

Note: the original docs/runtime/stages/2026-06-13-route-bot-concurrent-connects.md (dashboard.py + corpus.py, 6 opens) remains OPEN and incomplete — those raw connects are still on HEAD. Do not delete that stage.
