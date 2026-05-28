---
task: Dashboard planned-launch surface — unambiguous SIGNAL/DEMO/LIVE + profile + copies pre-start and while running
mode: IMPLEMENTATION
risk_tier: critical
scope_lock:
  - START_BOT.bat
  - scripts/run_live_session.py
  - trading_app/live/planned_launch.py
  - trading_app/live/bot_dashboard.py
  - trading_app/live/bot_dashboard.html
  - trading_app/live/session_orchestrator.py
  - data/bot_planned_launch.json
  - tests/test_trading_app/test_planned_launch.py
  - tests/test_trading_app/test_planned_launch_integration.py
---

## Why

`bot_dashboard.html:4060` ternary inverts precedence (`bs.demo ? "demo" : bs.signal_only ? "signal" : "demo"`) so:
- `signal_only=True` (which also sets `demo=True` at `run_live_session.py:883`) renders as DEMO.
- `live` branch is unreachable (else-clause is `"demo"`, never `"live"`).
- Pre-start, the dashboard reads stale `_lastStatusData.broker_status` from a previous session — no surface tells it what `START_BOT.bat`'s current `BOT_MODE_FLAGS` is set to.

Two ambiguity classes:
1. **Mode**: SIGNAL vs DEMO vs LIVE not clearly distinguished pre-start; CTA can mislabel; LIVE unreachable in code path.
2. **Account scope**: `profile.copies > 1` means one signal fans out to N broker accounts (real-money multiplier on `--live`), invisible in CTA. Pre-start "active profile" is `p.active` config flag, NOT `ACTIVE_PROFILE` from the .bat — these can disagree.

Both fall through because there's no canonical "planned launch" surface. The dashboard infers from partial signals; inference fails closed only at the orchestrator banner, after the user has clicked.

## What

Add canonical planned-launch artifact `data/bot_planned_launch.json`, written by both `START_BOT.bat` (via tiny Python helper) and the CLI codepath in `run_live_session.py` at session boot, BEFORE orchestrator initialization:

```json
{
  "profile_id": "topstep_50k_mnq_auto",
  "mode": "DEMO",                  // SIGNAL | DEMO | LIVE
  "copies": 1,
  "instruments": ["MNQ"],
  "broker_accounts_count": 1,      // = copies for now; real count after preflight if known
  "source": "START_BOT.bat",       // START_BOT.bat | CLI | dashboard
  "ts": "2026-05-28T03:14:15+00:00"
}
```

Dashboard:
- Reads `data/bot_planned_launch.json` via new endpoint `GET /api/planned-launch`.
- Renders banner above CTA (pre-start): `Next launch: DEMO · topstep_50k_mnq_auto · MNQ · 1 broker account`
- LIVE + copies > 1 banner: `Next launch: LIVE · topstep_50k_mnq_auto · MNQ · 3 broker accounts · REAL MONEY` (red).
- File missing/stale (>24h): banner reads `Next launch: planned launch unknown — edit START_BOT.bat or run CLI` (no guess). Mode CTA disabled.
- Once running, `state.mode` + new `state.profile_id` (added to `_publish_state`) is the authority; planned-launch banner hidden. CTA collapses to "Stop Session".
- Fix `bot_dashboard.html:4060` precedence — but the banner is the real surface; CTA label is secondary.

Orchestrator addition: publish `state.profile_id` so running-mode no longer scrapes `account_name`. Already have `self._profile_id` (session_orchestrator.py:374) — one-line addition to `_publish_state`.

## Blast Radius

- `START_BOT.bat` — add one `.venv\Scripts\python.exe -c "..."` line after Step 2 to write `data/bot_planned_launch.json` with source=START_BOT.bat. Idempotent.
- `scripts/run_live_session.py` — at session start (after mode resolution, before orchestrator init), write planned-launch file with source=CLI. Only when not in `--preflight` mode.
- `trading_app/live/bot_dashboard.py` — new `GET /api/planned-launch` endpoint reading `data/bot_planned_launch.json` with staleness check.
- `trading_app/live/bot_dashboard.html` — new banner above CTA; fix ternary at 4060; banner data poll alongside existing `/api/status` fetch.
- `trading_app/live/session_orchestrator.py` — `_publish_state` includes `profile_id`. One line.
- `data/bot_planned_launch.json` — NEW runtime artifact, gitignored, ephemeral.
- `tests/live/test_planned_launch.py` — NEW tests: writer roundtrip, staleness behavior, missing-file fail-visible, mode precedence.

Reads: `prop_profiles.ACCOUNT_PROFILES` (for copies/instruments lookup). Writes: `data/bot_planned_launch.json` only. No DB writes. No trading-logic changes. No schema changes.

Risk: critical-tier because it touches `trading_app/live/` and the running-mode publish path. Requires adversarial-audit-gate per `.claude/rules/adversarial-audit-gate.md` after commit.

## Done criteria

1. Banner renders SIGNAL/DEMO/LIVE correctly for all three modes with `--signal-only`, `--demo`, `--live` CLI invocations (manual test).
2. With `copies=3` profile, LIVE banner shows "3 broker accounts · REAL MONEY".
3. Stale (>24h) or missing planned-launch file → banner says "planned launch unknown", CTA disabled — no inference.
4. Running session shows `state.profile_id` in dashboard hero (not scraped from account_name).
5. `bot_dashboard.html:4060` ternary corrected — `signal_only` checked first, `live` reachable.
6. `python pipeline/check_drift.py` PASS (no new violations).
7. New tests pass: planned-launch writer roundtrip + staleness + mode precedence.
8. Adversarial-audit-gate via `evidence-auditor` on the commit.
