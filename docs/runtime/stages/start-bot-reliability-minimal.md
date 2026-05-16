---
task: "START_BOT preflight reliability — wire real bracket/fill-poller probes + narrow profile-account safeguard excepts"
mode: IMPLEMENTATION
scope_lock:
  - scripts/run_live_session.py
  - trading_app/live/session_orchestrator.py
  - tests/test_scripts/test_run_live_session_preflight.py
  - tests/test_trading_app/test_session_orchestrator.py
  - tests/test_trading_app/test_bot_dashboard.py
---

## Blast Radius

- `scripts/run_live_session.py` — `_run_lightweight_component_self_tests()` currently hardcodes `results["brackets"] = True` and `results["fill_poller"] = True` (lines 74-75). Replace with real probes that exercise the actual `router_class.supports_native_brackets()`/`build_bracket_spec()` and `query_order_status()` paths against the auth context produced by `create_broker_components`. Update `_check_notifications` to surface per-component PASS/FAIL in the inline message so the operator sees `5/7 PASS · brackets:FAIL · fill_poller:FAIL` instead of a misleading `7/7 PASS`. Callers: invoked once from `_check_notifications` at line 220; preflight runs from `_run_preflight` (single entrypoint).
- `trading_app/live/session_orchestrator.py` — three safeguard blocks at construction time:
  - Lines 361-378 (`_orb_caps` load via `get_lane_registry`): currently `except Exception` — narrow to `ImportError, KeyError, ValueError, TypeError` (the actual failure modes from a `get_lane_registry` call); preserve the `raise` on profile accounts.
  - Lines 383-392 (`_max_risk_per_trade` from `ACCOUNT_PROFILES`): currently `except Exception: raise` — narrow to `ImportError, KeyError, AttributeError`; profile accounts always raise on miss.
  - Lines 399-417 (`_regime_paused` from `lane_allocation.json`): currently `except Exception` — narrow to `FileNotFoundError, json.JSONDecodeError, KeyError, ValueError, OSError`; preserve the `raise` on profile accounts.
  These narrowings prevent `KeyboardInterrupt`/`SystemExit`/`AttributeError` (typos, refactor breakage) from being silently absorbed by the warn-and-continue branch on paper/signal-only sessions. The hard `raise` on profile accounts is already in place, so this is a quality-of-error-handling tightening rather than a capital-safety bug fix — but it closes the "silent fall-through on unrelated bug" hole the plan author was concerned about.
- `tests/test_scripts/test_run_live_session_preflight.py` — extend (do not rewrite) with three new tests: (a) bracket probe FAIL surfaces in summary, (b) fill_poller `NotImplementedError` surfaces in summary, (c) preflight overall summary string contains explicit `brackets:FAIL`/`fill_poller:FAIL` tokens when those probes return False. Existing 14 tests must continue to pass.
- `tests/test_trading_app/test_session_orchestrator.py` — extend with load-block-replay tests (matching the existing pattern at line 5227): malformed `lane_allocation.json` on a `profile_*` portfolio raises `RuntimeError` (currently bubbled by `raise`); same malformed JSON on a non-profile portfolio logs warning and yields empty `_regime_paused`. Mirror for the narrowed-class case (e.g., a missing `lane_registry` import → `ImportError` no longer caught by overly-broad `except Exception`).
- `tests/test_trading_app/test_bot_dashboard.py` — single-test update: `test_preflight_helper_opens_no_duckdb_connection` asserted `results["brackets"] is True` and `results["fill_poller"] is True` because the helper previously hardcoded those. Under the new contract, broker probes require `components` and return False when `components is None`. The test's actual intent ("helper must not open a DuckDB connection") is preserved; the assertion is updated to thread a stub `components` dict through and verify probes execute against it (still no DB opened).
- Reads: none. Writes: none to DB or runtime. No schema, no canonical-source changes.

## Audit gate

Per `.claude/rules/adversarial-audit-gate.md`: this commit touches `trading_app/live/` with a judgment classification. After the implementation lands and tests pass, dispatch `evidence-auditor` for independent review before closing the stage.

## Non-goals (explicit)

- HWM seeding fix (CARRY-OVER per HANDOFF).
- Log surface fix (CARRY-OVER).
- Watchdog hard-kill, kill-switch in reconnect loop (out of minimal scope).
- Any dashboard work (Phase 2, sequenced after this stage's audit verdict).
