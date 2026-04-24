# Stage: live-b6-f1-signal-only-seed

mode: IMPLEMENTATION
date: 2026-04-25
scope_lock:
  - trading_app/live/session_orchestrator.py
  - tests/test_trading_app/test_session_orchestrator.py
  - docs/runtime/stages/live-b6-f1-signal-only-seed.md
  - HANDOFF.md

## Why

Discovered 2026-04-24 NYSE_OPEN smoke test. `live_signals.jsonl` shows every entry since 2026-04-15 REJECT'd with:

> `risk_rejected: topstep_scaling_plan: EOD XFA balance unknown — refusing entry. Orchestrator must call set_topstep_xfa_eod_balance() at session start. (F-1 fail-closed)`

Root cause traced through three files:

1. `trading_app/live/session_orchestrator.py:337-343` — `RiskLimits` is constructed with `topstep_xfa_account_size` from the XFA profile **regardless of `signal_only`**. F-1 enforcement is therefore enabled.
2. `trading_app/live/session_orchestrator.py:527` — the HWM init block (which contains the only call to the F-1 seeder `_apply_broker_reality_check` at :588-595) is gated `if not signal_only and ...`. In signal-only the seeder never runs.
3. `trading_app/risk_manager.py:214-229` — `can_enter()` fail-closes when `topstep_xfa_account_size is not None AND _topstep_xfa_eod_balance is None`. With F-1 enabled but never seeded, every entry rejects.

The HANDOFF item B6 framed two candidate fixes:
- (a) seed from profile `account_size` when broker equity is None + `topstep_xfa_account_size` is set — preserves F-1 semantics.
- (b) `disable_f1` automatically in signal-only — bypass.

## Design choice — refinement of HANDOFF option (a)

Option (a) literally reads "seed with the dollar value of `prof.account_size`". The canonical source (`trading_app/topstep_scaling_plan.py:51-53` + `docs/research-input/topstep/topstep_mll_article.md`) is decisive:

> "Day-1 of any new XFA: balance = $0 (XFA starts at $0 ...), so max position is the bottom-row tier (2 lots for 50K, 3 lots for 100K/150K)."

Seeding with `account_size` (e.g. $50,000 for 50K XFA) would resolve to the **highest** tier (5 lots for 50K) per `SCALING_PLAN_LADDER`. That is unsafe and wrong: it would make signal-only OBSERVE entries that a fresh-XFA day-1 trader could not actually take.

Chosen seed: **$0.0**. This is the canonical day-1 XFA balance, matches the existing test pattern (`tests/test_trading_app/test_risk_manager.py:736 set_topstep_xfa_eod_balance(0.0)` for the day-1 50K case), and gives the most-restrictive cap. Signal-only then observes the entries that WOULD fire under day-1 caps — the conservative interpretation of "what would happen live".

Option (b) (`disable_f1` in signal-only) is rejected: it would mute F-1 entirely and prevent signal-only from being a faithful preview of live behaviour. Per `.claude/rules/integrity-guardian.md` rule 3, never silently disable an audit/risk path.

## Blast Radius

- `trading_app/live/session_orchestrator.py`:
  - New module-level helper `_apply_signal_only_f1_seed(risk_mgr, logger=None) -> bool` (mirrors `_apply_broker_reality_check` extraction pattern at :94-137).
  - One new call site inside `__init__`, immediately after `self.risk_mgr` is constructed (line ~343), gated `if signal_only:`.
  - Net: ~25 lines insertion.
- `tests/test_trading_app/test_session_orchestrator.py`:
  - 3 new tests in a new class `TestF1SignalOnlySeed`:
    - F-1 active in signal-only → helper seeds with 0.0.
    - F-1 inactive (XFA not in profile) → helper does NOT seed (returns False).
    - Helper integration: signal-only orchestrator path actually invokes the helper.
- `HANDOFF.md`: drop B6 from "Next Steps — Active"; add closeout pointer to this stage.

No production behaviour change for live/demo paths (the new code is gated `if signal_only`). No new public API. Helper is extracted exactly to be testable without standing up a full SessionOrchestrator (matches the institutional rigor pattern already used for `_apply_broker_reality_check`).

## Verification

1. `pytest tests/test_trading_app/test_session_orchestrator.py -k "F1" -q` — 3 new tests + existing 14 F-1 tests all pass.
2. `pytest tests/test_trading_app/test_session_orchestrator.py tests/test_trading_app/test_risk_manager.py -q` — full sweep no regression.
3. `python pipeline/check_drift.py` — no NEW failures (pre-existing #4 work_queue and #59 MNQ daily_features known untouched).
4. Mutation probe (manual): change the seed to `account_size` and confirm test catches `assert_called_once_with(0.0)` failure.

## Out of scope

- Reconnecting the periodic rollover refresh (`session_orchestrator.py:1257-1276`) for signal-only. That path also requires broker equity and is correctly skipped in signal-only — the day-start seed is enough for a session-bounded smoke test.
- Pinning `.gitattributes` for CRLF (HANDOFF item O-CRLF — separate stage).
- `_topstep_xfa_eod_balance` profile-override mechanism (env var or config field). The $0.0 seed is the canonical safe default; if operators later need to test against a higher tier they can extend `prop_profiles.py`.

## Commit

`fix(live): seed F-1 XFA EOD balance to $0 in signal-only (closes B6)`
