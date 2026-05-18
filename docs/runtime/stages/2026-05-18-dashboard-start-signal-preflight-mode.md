---
task: thread mode into dashboard readiness preflight so Start Signal does not run live-mode telemetry gate
mode: IMPLEMENTATION
scope_lock:
  - trading_app/live/bot_dashboard.py
---

## Blast Radius

- `trading_app/live/bot_dashboard.py:653` `_run_preflight_subprocess(profile)` — add `mode` parameter, pass `--signal-only` to the spawned `scripts.run_live_session --preflight` when `mode == "signal"`.
- `trading_app/live/bot_dashboard.py:712` `_prepare_profile_for_start(profile)` — accept `mode` parameter, thread to `_run_preflight_subprocess`.
- `trading_app/live/bot_dashboard.py:2431` call site in `action_start` — already has `mode`; pass it through.
- `trading_app/live/bot_dashboard.py:1779` call site in `action_preflight` — runs ad-hoc preflight from dashboard button. Default to live-mode preflight (current behavior, no change).
- Other call sites: grep confirms only the two above.
- Reads: none new. Writes: same preflight cache entry under same profile key; cache contents differ by mode only via the embedded output field — acceptable per cache contract.

## Why this fix

`_check_telemetry_maturity` in `scripts/run_live_session.py:369-378` auto-passes when `ctx.signal_only=True` and fail-closes otherwise. The dashboard's Start Signal button (`/api/action/start?mode=signal`) currently runs readiness preflight in live mode (line 655 — no `--signal-only` flag), so the telemetry gate fail-closes at 11/30 and the dashboard returns `status=blocked`. This breaks the documented `--signal-only` accumulation path — the very mode meant to clear the gate is blocked by it.

## Done criteria

1. `_run_preflight_subprocess` accepts `mode` param; passes `--signal-only` when mode=signal.
2. `_prepare_profile_for_start` accepts `mode` param; threads it.
3. `action_start` passes its `mode` arg through.
4. `action_preflight` ad-hoc check stays in live mode (default).
5. Drift check: 133/133 still pass.
6. Targeted test: `pytest tests/test_trading_app/test_bot_dashboard*.py -q` green.
7. Dashboard restarted; `POST /api/action/start?mode=signal&profile=topstep_50k_mnq_auto` returns non-blocked status; signal-log file `live_signals_2026-05-18.jsonl` appears within 30s.
