# Next Session — Go-Live Plan (MNQ topstep_50k_mnq_auto)

**Written:** 2026-05-16, end of session.
**Goal:** Get the live MNQ app trading real capital ASAP on `topstep_50k_mnq_auto`.

## What's already done (this session, on origin/main)

- `a0b3c24b` — `--account-id` sentinel mismatch fixed (Start Live no longer crashes on copies>1).
- `15bde024` — Audit closure on sentinel fix (asymmetric tolerance documented in tests).
- `bb0619d2` — A.6.5 preflight gap closed (new `_check_copy_trading_accounts` dry-runs the live-start account-resolution path).
- `5dd1a822` — Review closure (preflight message wording).

Drift: 133 PASS. Preflight tests: 11 PASS. Session-orchestrator: 222 PASS.

## Pick-up sequence (do in order, do NOT skip)

### 1. Re-orient (5 min)

```bash
git log --oneline -8
git status --short
python pipeline/check_drift.py | tail -5
```

Confirm `origin/main` matches local. Read `HANDOFF.md` top section. Verify `data/lane_allocation.json` still shows 4 deployed MNQ lanes (live state may have moved overnight).

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
- `logs/session.log` shows `Copy trading: primary=..., shadows=[...]`.
- No `--account-id 0 is not in the broker's discovered accounts` line.

This is the regression check for `a0b3c24b`. If it crashes again, the sentinel fix is incomplete and the path forward is to read the actual log line, not re-litigate the design.

### 4. Live-trading readiness gate (30–60 min)

The remaining blockers from action-queue item `lane_allocation_rebalance_2026_05_14_pending_capital_review_blockers` are:
- **(a) CLOSED 2026-05-14** — c8 doctrine.
- **(b) SR-tripwire blind spot** on newly-promoted lanes — paper-trade warmup status undetermined.
- **(c) Live-control trace** — kill/flatten/risk-limit not traced for new lane set.

For go-live on the CURRENT 4 lanes (already deployed per `lane_allocation.json`), blockers (b) and (c) apply to the proposed rebalance, NOT the existing set. The existing 4 MNQ lanes have been live-routable since their respective add dates.

**Decision needed:** trade the existing 4 lanes today, OR wait for the rebalance verification. The rebalance net delta is +2.80 R/yr (~$84/yr/contract), below noise floor — go-live with existing lanes is the higher-EV path.

### 5. Start a live session (the actual go-live)

```bash
python scripts/run_live_session.py --instrument MNQ --profile topstep_50k_mnq_auto --live
# Type CONFIRM when prompted.
```

Tail `logs/session.log` in another terminal. Watch for:
- First bar processed without exception.
- `OrderRouter` connection healthy.
- `kill_switch` armed, `risk_manager` reporting state.

### 6. Stop conditions (write these down before starting)

Pre-commit to the kill conditions BEFORE the first real trade fills:
- 1 R loss on day → flat and stop.
- Any `RuntimeError`/`AssertionError` in `logs/session.log` → flat and stop.
- Broker connection drop > 2 min → flat and stop.
- Any unexpected position vs portfolio expectation → flat and stop.

## What NOT to do next session

- Do NOT re-litigate MGC LONDON_METALS (frozen verdict per HANDOFF #1).
- Do NOT touch Stage 2 NQ-mini wiring (parked, dormant infrastructure, 0 P&L today). See `docs/plans/2026-05-16-stage2-nq-mini-plumbing-gap-finding.md`.
- Do NOT add new strategies before the first live day completes.
- Do NOT mutate `lane_allocation.json` without re-running the rebalance script and clearing blockers (b) and (c).

## Files to read first when picking up

1. `HANDOFF.md`
2. This file (`docs/runtime/next-session-go-live-plan.md`)
3. `data/lane_allocation.json` (live state truth)
4. `docs/plans/2026-05-16-stage2-nq-mini-plumbing-gap-finding.md` (only if asked about NQ-mini)

## Branch state at end of session

- Branch: `main`
- Last commit: `5dd1a822`
- `origin/main` matches local.
- 1 untracked file: `docs/plans/2026-05-16-stage2-nq-mini-plumbing-gap-finding.md` (parked, do not commit unless reopening Stage 2).
