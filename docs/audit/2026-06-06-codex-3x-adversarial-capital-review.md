# 3× Adversarial Capital-Risk Review — Codex (2026-06-06)

**Repo:** `C:/Users/joshd/canompx3` @ HEAD `fb76e8cf` · **Reviewer:** Codex adversarial-review ×3 (independent sessions, unbiased) · **Mandate:** find bugs that wreck a real trading account long-term; smallest-diff fixes only.

All four findings **MEASURED against the live tree** (not INFERRED) before filing — fix-target methods confirmed to exist. Codex sessions: R1 `019e9860`, R2 `019e985e`, R3 `019e9864`.

## Unifying defect

Every finding is the same class: **a safety check that does not bind to the live behavior it claims to protect.** The gate/check and the capital action are decoupled.

| # | Sev | File:line | Capital-loss scenario | Verified |
|---|-----|-----------|------------------------|----------|
| A | critical | `trading_app/live/projectx/contract_resolver.py:51` | `resolve_account_id()` returns `accounts[0]` from broker search — no profile binding. C11 proves safety for `topstep_50k_mnq_auto`; orders can route to a 100K XFA or eval combine if it's listed first. Survival proof is for one account; capital trades on another. | ✅ `accounts[0]` confirmed |
| B | critical | `trading_app/live/session_orchestrator.py:2787-2821` | After a live entry with a missing/failed bracket leg, both branches log `critical` + `_notify` only — never `_fire_kill_switch()`/`_emergency_flatten()`. Worse: `brackets_submitted += 1` runs unconditionally after, so telemetry records success on a naked position. One failed SL attach + adverse gap = unbounded loss. | ✅ both branches alert-only; counter increments |
| C | high | `trading_app/account_survival.py:407` | `contracts_per_trade = 1` hardcoded; DD math models 1 micro. Live sizing (`execution_engine._compute_contracts`) computes from equity, clamps to `max_contracts`. The moment `max_contracts` is lifted, C11 still proves DD on 1 contract → 2-contract lane ≈ doubles the $1,535 90d DD past the $1,800 express belt while gate reads PASS. | ✅ hardcode + separate live path confirmed |
| D | high | `trading_app/account_survival.py:220-225` | `_criterion11_code_paths()` fingerprints only `account_survival.py` + `derived_state.py`. Drift in `session_orchestrator.py` / `execution_engine.py` / `prop_profiles.py` / `portfolio.py` leaves a stale C11 PASS valid — same class as the previously-fixed DSR stale-PASS bug. | ✅ only 2 files in fingerprint |

## Confirmed clean (R3 negative results — unbiased)

- Lane registry / gate loading reads `risk_cap_pts` before p90.
- Explicitly missing profiles **raise** (no silent fallback profile).
- Topstep payout policy is **not** a flat-only $5000 cap.
- Current 1-contract sizing does **not** exceed the 5/10/15 scaling ladder on this path.

## Smallest-diff fixes

- **A — account binding (fail-closed):** add `account_id` / `account_name_pattern` to `topstep_50k_mnq_auto`; validate `Account/search` metadata against it before constructing `ProjectXOrderRouter`; fail closed if `--live` finds no match. Immediate guard: `scripts/run_live_session.py` rejects `--live --profile` without a validated `--account-id`. + tests for START_BOT / dashboard / single-copy so none falls back to `accounts[0]`.
- **B — act, don't just alert:** in **both** the missing-leg branch and the no-fallback exception branch: persist known leg IDs, `self._fire_kill_switch()`, `await self._emergency_flatten()`, `return` (do not increment `brackets_submitted`). + test: `verify_bracket_legs` → `(None, tp_id)` asserts kill switch fires and flatten invoked.
- **C — D-3 parity guard (fail-closed):** in C11, fail closed unless every live-built profile strategy has `max_contracts == 1`; OR pass the real live contract count into `_load_lane_trade_paths()` and scale pnl/MAE/MFE/open-lot. + test: raise `max_contracts > 1` ⇒ C11 scales DD or blocks.
- **D — full live-risk fingerprint:** extend `_criterion11_code_paths()` to include `prop_profiles.py`, `portfolio.py`, `execution_engine.py`, `session_orchestrator.py` — or a separate live-risk fingerprint checked by `check_survival_report_gate`.

## Roadmap impact

`docs/plans/2026-06-06-clamp-lift-d3-seam-income-scope.md` plans to **lift `max_contracts` to earn more**. Findings **C + D prove that is unsafe in current code**: lifting the clamp makes C11 understate drawdown (C) and the stale fingerprint won't notice the live sizing changed (D). **Fix C + D before any clamp lift.** A and B are pre-existing live-arming blockers regardless.

**Recommended fix order (smallest/highest-leverage first):** B → A → C+D.

**Status:** review-only. No edits made, nothing armed.
