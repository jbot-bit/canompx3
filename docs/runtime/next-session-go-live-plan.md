# Next Session — Go-Live Plan (MNQ topstep_50k_mnq_auto)

**Written:** 2026-05-16, end of session.
**Updated:** 2026-05-17 (post profile-routing unlock)
**Goal:** Get the live MNQ app trading real capital ASAP on `topstep_50k_mnq_auto`.

## What's already done (this session, on origin/main)

- `a0b3c24b` — `--account-id` sentinel mismatch fixed (Start Live no longer crashes on copies>1).
- `15bde024` — Audit closure on sentinel fix (asymmetric tolerance documented in tests).
- `bb0619d2` — A.6.5 preflight gap closed (new `_check_copy_trading_accounts` dry-runs the live-start account-resolution path).
- `5dd1a822` — Review closure (preflight message wording).

Drift: 133 PASS. Preflight tests: 11 PASS. Session-orchestrator: 222 PASS.

## 2026-05-17 alignment update (routing + doctrine)

- `topstep_50k_mnq_auto.allowed_sessions` now includes `NYSE_CLOSE` and
  `LONDON_METALS` in addition to the prior 7-session set. This removes the
  profile-level routing block for allocator-selected MNQ lanes in those
  sessions.
- This does **not** override doctrine gates. The locked NYSE_CLOSE hypothesis
  loader failure (`trading_app/hypothesis_loader.py:291`, theory_citation ×
  Amendment 3.0 collision) still parks the NYSE_CLOSE cohort until that
  doctrine issue is resolved.
- Operational implication: treat this as a **routing prerequisite complete**,
  not a promotion/deployment approval.

## Pick-up sequence (do in order, do NOT skip)

### 1. Re-orient (5 min)

```bash
git log --oneline -8
git status --short
python pipeline/check_drift.py | tail -5
```

Confirm `origin/main` matches local. Read `HANDOFF.md` top section. Verify `docs/runtime/lane_allocation.json` still shows 4 deployed MNQ lanes (live state may have moved overnight).

### 2. Run real preflight against the active profile (10 min)

```bash
python scripts/run_live_session.py --instrument MNQ --profile topstep_50k_mnq_auto --preflight --signal-only
```

Expected output:
- `[1/7] Auth check (topstep)... OK`
- `[7/7] Copy-trading account resolution (dry run)... OK (copies=2, N accounts discovered)`
- `Preflight: 7/7 passed`
- `All clear — ready to trade.`

If preflight reports anything other than 7/7: STOP. The new check is real-API and may surface broker-side issues that the unit tests with stubbed contracts couldn't see (token expiry, account inactivity, contract resolution failure, etc.). Fix before proceeding.

### 3. Dashboard smoke (5 min)

Launch the dashboard and click "Start Live" on `topstep_50k_mnq_auto`. Confirm:
- `SessionOrchestrator.__init__` reaches steady state without `RuntimeError` from `_select_primary_and_shadow_accounts`.
- `logs/live/live_<ts>.log` shows `Copy trading: primary=..., shadows=[...]`.
- No `--account-id 0 is not in the broker's discovered accounts` line.

This is the regression check for `a0b3c24b`. If it crashes again, the sentinel fix is incomplete and the path forward is to read the actual log line, not re-litigate the design.

### 4. Live-trading readiness gate (30–60 min)

The remaining blockers from action-queue item `lane_allocation_rebalance_2026_05_14_pending_capital_review_blockers` are:
- **(a) CLOSED 2026-05-14** — c8 doctrine.
- **(b) SR-tripwire blind spot** on newly-promoted lanes — paper-trade warmup status undetermined.
- **(c) Live-control trace** — kill/flatten/risk-limit not traced for new lane set.

For go-live on the CURRENT 4 lanes (already deployed per `docs/runtime/lane_allocation.json`), blockers (b) and (c) apply to the proposed rebalance, NOT the existing set. The existing 4 MNQ lanes have been live-routable since their respective add dates.

**Route discipline after the 2026-05-17 allowlist expansion:**
- Keep live trading on the current 4 lanes unless/until a fresh rebalance
  explicitly promotes additional lanes.
- Do not manually inject NYSE_CLOSE/LONDON_METALS lanes into live state solely
  because profile routing now allows them.
- If the next rebalance surfaces NYSE_CLOSE candidates, require doctrine-loader
  fix evidence first.

**Decision needed:** trade the existing 4 lanes today, OR wait for the rebalance verification. The rebalance net delta is +2.80 R/yr (~$84/yr/contract), below noise floor — go-live with existing lanes is the higher-EV path.

### 5. Start a live session (the actual go-live)

```bash
python scripts/run_live_session.py --instrument MNQ --profile topstep_50k_mnq_auto --live
# Type CONFIRM when prompted.
```

Tail `logs/live/live_<ts>.log` in another terminal. Watch for:
- First bar processed without exception.
- `OrderRouter` connection healthy.
- `kill_switch` armed, `risk_manager` reporting state.

### 6. Stop conditions (write these down before starting)

Pre-commit to the kill conditions BEFORE the first real trade fills:
- 1 R loss on day → flat and stop.
- Any `RuntimeError`/`AssertionError` in `logs/live/live_<ts>.log` → flat and stop.
- Broker connection drop > 2 min → flat and stop.
- Any unexpected position vs portfolio expectation → flat and stop.

## What NOT to do next session

- Do NOT re-litigate MGC LONDON_METALS (frozen verdict per HANDOFF #1).
- Do NOT touch Stage 2 NQ-mini wiring (parked, dormant infrastructure, 0 P&L today). See `docs/plans/2026-05-16-stage2-nq-mini-plumbing-gap-finding.md`.
- Do NOT add new strategies before the first live day completes.
- Do NOT mutate `docs/runtime/lane_allocation.json` without re-running the rebalance script and clearing blockers (b) and (c).

## Carry-over capture list (from 2026-05-16 debut — collect during next live run)

Two log-surface gaps from the 2026-05-16 debut are CARRY-OVER per `HANDOFF.md` § Next Steps 5(c). Both need ≥1 more live run to characterize before a fix stage is justified. When you start the next `--live` session, capture the evidence below so the post-session handoff can promote (or close) each item from "carry-over open" → concrete finding.

### (c-i) `/api/bars-recent?instrument=MNQ` returns `"bars":[]`

Working hypothesis: tick → 1m aggregation handoff isn't flushing into whatever the dashboard endpoint reads. Capture during the live session:

1. A wall-clock timestamp where the feed log line shows a tick arriving (e.g., `Subscribed to MNQ quotes` + first `tick` log entry).
2. ≥3 minutes after step 1, hit `curl http://<dashboard-host>:<port>/api/bars-recent?instrument=MNQ` and paste the full response body (not just the empty `bars` field — include any error/meta fields).
3. From the same window, the last 5 lines of any `1m`/`aggregator`/`bar_builder` log entries (grep stdout — there is no log file per c-ii). If zero matches, that itself is evidence: the aggregator never ran.
4. Whether the chart panel is empty in the dashboard UI at the same wall-clock as the curl call.

Do NOT touch `/api/bars-recent` server-side code yet — the trace above narrows root cause (tick layer vs aggregator vs endpoint reader) before any edits.

### (c-ii) No `logs/live/live_<ts>.log` written under `--live`

Working hypothesis: `logging.basicConfig` is stdout-only at the script entry; the live mode doesn't add a `FileHandler` rooted at `logs/live/`. Capture during the live session:

1. `ls logs/live/` immediately after `python scripts/run_live_session.py … --live` starts. Note whether a `live_<ts>.log` is created at session start (handler initialized but not flushed?) OR never appears (no handler).
2. The first 10 stdout lines after `Type CONFIRM` — look for any line resembling `logging` / `handler` / `FileHandler` / `OSError` / `PermissionError`.
3. If `logs/live/` does not exist as a directory: that's the root cause (missing `mkdir -p` on session start). Capture `ls logs/` to confirm.

Do NOT add a `FileHandler` yet — the dashboard log surfacing (separate stage) may want a specific format; gather evidence first.

### Where to put the captures

Drop the curl response body + log excerpts into a fresh `docs/runtime/sessions/<YYYY-MM-DD>-live-debut-followup.md`, then update `HANDOFF.md` § Next Steps 5(c) with "characterized" status + a one-line root-cause sentence per item. After that, each (c-i)/(c-ii) becomes a normal stage candidate.

### One-shot Monday evidence-capture (paste at T+3min after first tick + on any anomaly)

`date -u +%FT%TZ; ls -la logs/live/ 2>/dev/null | tail -5; curl -s http://localhost:8088/api/bars-recent?instrument=MNQ | head -c 500; echo; ls -la data/state/account_hwm_*.json 2>/dev/null; python -c "import sys,json,glob; [print(p,'->',json.load(open(p)).get('hwm_dollars'),'/ last_eq=',json.load(open(p)).get('last_equity')) for p in glob.glob('data/state/account_hwm_*.json')]"` — captures D2/D3/D4 surfaces in one line; baton plan is `C:\Users\joshd\.claude\plans\get-going-on-this-whimsical-rain.md`; ProjectX spec extract is `resources/projectx_api_spec_2026_05_16.md`.

## Files to read first when picking up

1. `HANDOFF.md`
2. This file (`docs/runtime/next-session-go-live-plan.md`)
3. `docs/runtime/lane_allocation.json` (live state truth)
4. `docs/plans/2026-05-16-stage2-nq-mini-plumbing-gap-finding.md` (only if asked about NQ-mini)

## Branch state at end of session

- Branch: `main`
- Last commit: `8c7786cb`
- `origin/main` matches local.
- 1 untracked file: `docs/plans/2026-05-16-stage2-nq-mini-plumbing-gap-finding.md` (parked, do not commit unless reopening Stage 2).
