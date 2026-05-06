---
task: Refactor scripts/run_live_session.py preflight from manual checks_total constant to list-of-callables (closes debt-ledger LOW-1)
mode: IMPLEMENTATION
scope_lock:
  - scripts/run_live_session.py
  - tests/test_scripts/test_run_live_session_preflight.py
  - docs/runtime/debt-ledger.md
---

## Blast Radius

- `scripts/run_live_session.py` — modifies `_run_preflight()` (~150 lines, lines 75-227). Pure refactor; behavior identical for every (instrument, broker, demo, portfolio) input. No change to existing call sites: `main()` calls `_run_preflight(instrument, broker, demo, portfolio)` and reads its bool return — that contract is preserved.
- `tests/test_scripts/test_run_live_session_preflight.py` — new file. Companion test injects a known-failing check into the `checks` list and confirms it is counted toward `checks_total`. This is the verification mandated by the debt-ledger close-out spec ("companion test should inject a known-failing check and confirm it is counted toward the total").
- `docs/runtime/debt-ledger.md` — close `preflight-checks-total-hardcode` line 18; mark RESOLVED with commit reference.
- Reads: nothing new (preflight reads what it already reads — broker auth, daily_features, contracts, trade_journal).
- Writes: nothing (preflight is read-only).
- Live order-route impact: NONE. `_run_preflight` is gating logic only — it returns True/False. The 6 checks already exist and run today; this restructures HOW they're enumerated, not WHAT they do.

## Why this is not TRIVIAL

`scripts/run_live_session.py` is the live-trading entry point — operator runs `START_BOT.bat` → `run_live_session.py`. A behavioral regression in preflight (e.g., a check silently skipped, or a False return when all pass) would either (a) block legitimate live sessions or (b) admit a session that should have been blocked. Both are capital-class consequences. Hence DESIGN stage with a companion test, not TRIVIAL inline edit.

## Approach (state-aware list-of-callables)

The 6 checks are NOT uniform — there's state coupling:
- Check 1 (auth) produces `components` — consumed by checks 4 (contracts) and 5 (notifications).
- Check 2 (portfolio) reads the injected `portfolio` arg — pure.
- Check 3 (daily_features) reads `portfolio.strategies` to decide if `atr_20=None` is fatal.
- Check 5 (notifications) sets `all_pass` flag read by the final summary print.
- Check 6 (trade journal) — pure.

A naive `for check in checks: passed += check()` loses this state. The clean canonical pattern (mirrors `pipeline/check_drift.py`):

```python
@dataclass
class PreflightContext:
    instrument: str
    broker_name: str
    demo: bool
    portfolio: Portfolio
    components: dict | None = None       # set by check_auth, read by check_contracts/check_notifications
    components_all_pass: bool = True     # set by check_notifications, read by final summary

@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str  # printed inline after "[i/N] <name>... "

CheckFn = Callable[[PreflightContext], CheckResult]

def _run_preflight(...) -> bool:
    ctx = PreflightContext(...)
    checks: list[CheckFn] = [
        _check_auth,            # populates ctx.components
        _check_portfolio,
        _check_daily_features,
        _check_contracts,       # reads ctx.components
        _check_notifications,   # reads ctx.components, sets ctx.components_all_pass
        _check_trade_journal,
    ]
    checks_total = len(checks)
    checks_passed = 0
    for i, check in enumerate(checks, 1):
        print(f"[{i}/{checks_total}] {check.__doc__ or check.__name__}...", end=" ", flush=True)
        result = check(ctx)
        print(result.message)
        if result.passed:
            checks_passed += 1
    print(f"\nPreflight: {checks_passed}/{checks_total} passed")
    if checks_passed == checks_total:
        if not ctx.components_all_pass:
            print("All checks passed, but component warnings present. Review above.\n")
        else:
            print("All clear — ready to trade.\n")
    else:
        print("FIX FAILURES before starting a live session.\n")
    return checks_passed == checks_total
```

Each check function takes `ctx`, returns `CheckResult`, and is responsible for its own try/except (preserving the existing fail-quiet-with-message behavior — checks 1, 4, 5 currently print FAILED-or-SKIPPED messages but don't raise). The `_check_*` functions are module-private; signatures are stable and easy to test in isolation.

## Behavioral parity guarantees

For each of the 6 checks, the refactored version must produce **byte-identical stdout** for every (success, failure, skip) branch. The companion test compares stdout via `capsys` against a frozen golden output for two scenarios: (a) all-pass with injected portfolio, (b) auth-fail cascading to checks 4 & 5 SKIPPED.

Specifically preserved:
- Check 1 sets `components = None` on failure → check 4 prints `SKIPPED (auth failed)`, check 5 prints `FAILED: auth failed`. → modeled via `ctx.components is None` guard in `_check_contracts` and `_check_notifications`.
- Check 3's split between FAILED (ATR-dependent filters present) vs WARN (no dependent filters) — preserved verbatim.
- Check 5's WARNINGS-but-still-counts-as-passed semantics — preserved by setting `passed=True, message="WARNINGS: ..."`.
- Final summary's three-way branch (all_pass + components_all_pass / all_pass but warnings / fail) — preserved.

## Companion test (new file)

`tests/test_scripts/test_run_live_session_preflight.py`:

1. **`test_checks_total_equals_len_checks`** — monkeypatch the `checks` list to inject a 7th check; confirm the print line shows `[7/7]` not `[6/7]` or `[7/6]`.
2. **`test_known_failing_check_counted`** — inject a check that always returns `passed=False`; confirm summary reports `6/7 passed` and final return is False.
3. **`test_all_pass_smoke`** — inject a minimal in-memory portfolio + monkeypatch broker_factory; run real preflight; confirm bool=True and stdout contains `All clear — ready to trade`.
4. **`test_auth_fail_cascades`** — monkeypatch auth to raise; confirm checks 4 and 5 print SKIPPED/FAILED with the original messages; final return False.

Test 1 + 2 are the load-bearing ones for the LOW-1 close-out — they prove the count is dynamic. Tests 3 + 4 are regression coverage so the refactor doesn't silently change observable behavior.

## Acceptance criteria — all four required before deleting this stage file

1. `pytest tests/test_scripts/test_run_live_session_preflight.py -v` → 4/4 pass (show output).
2. `pytest tests/test_scripts/ -k run_live_session -q` → existing `test_run_live_session_account_selection.py` still passes (regression).
3. `python pipeline/check_drift.py` → 119+ checks PASS, 0 new violations.
4. `grep -n "checks_total = 6" scripts/run_live_session.py` → no match (the literal is gone). `grep -n "checks_total = len(checks)" scripts/run_live_session.py` → exactly one match.
5. `docs/runtime/debt-ledger.md` `preflight-checks-total-hardcode` line struck through with `~~...~~` and **CLOSED <date>** + commit SHA.

## Out of scope

- Adding new preflight checks. The diff is structural-only.
- Changing what each check tests. Same broker auth, same daily_features lookup, same contract resolution, same notifications probe, same trade_journal health.
- Touching the `_run_lightweight_component_self_tests` helper (line 51) — already a separate function; no refactor needed.

## Risk register

- **Risk:** Stdout drift from format-string consolidation (e.g., dropping a trailing space). **Mitigation:** golden-output assertion in companion test #3 + #4.
- **Risk:** State-coupling between checks creates a subtle ordering bug (e.g., check 5 reads `ctx.components` set by check 1 — if list reorders, check 5 sees None). **Mitigation:** the `checks` list is defined once at module level in execution order; companion test #3 exercises the success path end-to-end, test #4 exercises the auth-fail cascade — both prove the ordering contract.
- **Risk:** Any new exception class introduced by the refactor's structure. **Mitigation:** each `_check_*` keeps the existing try/except wrapping verbatim — no exception type changes.
