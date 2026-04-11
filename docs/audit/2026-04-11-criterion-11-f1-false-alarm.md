# Criterion 11 / F-1 BLOCKER — Audit Report: FALSE ALARM

**Date:** 2026-04-11
**Auditor:** Claude (session-triggered read-only audit per user directive "audit and improve, no gaps no bias")
**Trigger:** Pulse BROKEN signal "BLOCKED: Criterion 11 report fails scaling-feasibility check"
**Status:** **IMPLEMENTED 2026-04-11** — fix shipped with 117/117 tests passing and `account_survival` verifying scaling breach 68.1% → 0.0%, gate FAIL → PASS
**Scope:** Read-only code audit + numerical verification. Fix implemented in separate commit.

## Resolution summary (2026-04-11)

- **Fixed:** `trading_app/topstep_scaling_plan.py` (`total_open_lots` rewritten to aggregate per-instrument contracts before ceiling; new `project_total_open_lots` helper)
- **Fixed:** `trading_app/account_survival.py` (`TradePath` gains `contracts` and `instrument` fields; `_scenario_from_trade_paths` event loop tracks per-instrument contracts and computes `max_open_lots` via canonical aggregate-then-ceiling)
- **Fixed:** `trading_app/risk_manager.py` F-1 projection (uses `project_total_open_lots` instead of `total_open_lots + lots_for_position(inst, 1)`)
- **Fixed:** `tests/test_trading_app/test_topstep_scaling_plan.py` (contradictory test rewritten; 5 new canonical aggregation tests added)
- **Fixed:** `tests/test_trading_app/test_risk_manager.py` (4 F-1 tests rewritten with canonical cap values; `_make_rm` fixture relaxed to isolate F-1)
- **Fixed:** `tests/test_trading_app/test_account_survival.py` (scenario test fixture uses realistic contract counts)
- **Verified:**
  - `pytest test_topstep_scaling_plan test_risk_manager test_account_survival` → 117/117 pass
  - `python -m trading_app.account_survival --profile topstep_50k_mnq_auto`:
    - Before: gate=FAIL, scaling breach 68.1%, scaling feasible False, DD survival 26.2%
    - After: gate=PASS, scaling breach 0.0%, scaling feasible True, DD survival 86.2%
- **Deferred:** `pipeline/check_drift.py` run — parallel Stage 4 ML DELETE process was actively editing that file at the time of this fix. Drift will be re-run after Stage 4 DELETE completes.
- **F-1 status in `docs/audit/2026-04-08-topstep-canonical-audit.md`:** changed from BLOCKER/OPEN to RESOLVED/FALSE ALARM.

---

## 1. TL;DR

The "F-1 Scaling Plan Day-1 violation" recorded as a BLOCKER in memory (`topstep_canonical_audit_apr8.md`: "5-lane bot is 2.5× over 2-lot Day-1 cap on 50K XFA") is a **simulation artifact**, not a real infeasibility.

- **Real exposure:** 5 MNQ micro lanes × 1 contract each = 5 micro contracts concurrent = **0.5 mini-equivalent = 1 lot** (ceiling) ≤ 2-lot Day-1 cap. **Compliant with 50% headroom.**
- **Simulated exposure:** 5 lots (due to per-trade ceiling aggregation bug) > 2-lot cap. **Reports BREACH.**
- **Memory-recorded "2.5×" number** = 5/2, the exact ratio of buggy count to cap. The entire F-1 BLOCKER claim traces to this single misinterpretation.
- **Test file locks the bug in CI.** `test_5_simultaneous_lanes_breach_50k_day1` explicitly asserts `total == 5` for the exact scenario. This test MUST be rewritten as part of any fix.
- **Impact if fixed:** `account_survival` for `topstep_50k_mnq_auto` flips from `scaling_feasible=False` (68.1% breach) to `scaling_feasible=True` with significant Day-1 headroom. Criterion 11 report unblocks.

**Recommendation:** Fix the per-trade ceiling aggregation in `trading_app/topstep_scaling_plan.py:total_open_lots` AND `trading_app/account_survival.py` path simulation AND the contradictory test. Update `topstep_canonical_audit_apr8.md` status tracker to mark F-1 as "RESOLVED — false alarm, fixed 2026-04-11". **Still requires user authorization before implementation.**

---

## 2. Canonical Source Review

Quoted verbatim from `trading_app/topstep_scaling_plan.py` (which itself cites `docs/research-input/topstep/topstep_scaling_plan_article.md` and the scraped Scaling Plan chart PNG):

### Quote 1 — the contract ratio
> "On TopstepX, Micros and Minis are calculated using a 10:1 ratio: 1 Mini contract = 10 Micro contracts"
> — `topstep_scaling_plan.py:49`, @verbatim from canonical article

### Quote 2 — the cap composition rule
> "A 50K XFA at Level 1 (2 lots) can hold 2 minis OR 20 micros OR any combination summing to 2 mini-equivalents."
> — `topstep_scaling_plan.py:136-137` (docstring, not @verbatim but derivable from Quotes 1 and the ladder)

### Quote 3 — the scaling ladder
> ```
> $50K XFA
>   Below $1,500        → 2 lots
>   $1,500 to $2,000    → 3 lots
>   Above $2,000        → 5 lots
> ```
> — `topstep_scaling_plan.py:29-33`, @verbatim from xfa_scaling_chart.png

### Derived canonical math
- 1 MNQ micro = 1/10 mini-equivalent = 0.1 lot
- 20 MNQ micros = 2 lots (matches Quote 2)
- **5 MNQ micros = 0.5 lots, ceiling → 1 lot** ≤ 2-lot Day-1 cap

There is nothing in the canonical source that supports "each position rounds up independently." The ceiling rounding applies to the AGGREGATE mini-equivalent, not to each trade.

---

## 3. Code Trace — The Bug

### Layer 1: the aggregation function

`trading_app/topstep_scaling_plan.py:161-204`

```python
def total_open_lots(active_trades: list, instrument: str | None = None) -> int:
    total = 0
    for t in active_trades:
        ...
        contracts = getattr(t, "contracts", 0)
        if contracts <= 0:
            continue
        total += lots_for_position(t_inst, contracts)  # ← BUG: per-trade ceiling
    return total
```

**Problem:** `lots_for_position(t_inst, contracts)` is called **per trade** with the trade's own contract count. For 5 separate 1-contract MNQ trades, this calls `lots_for_position("MNQ", 1)` five times, each returning `ceil(1/10) = 1`, summed to **5**.

**Correct aggregation:** sum contracts per instrument first, THEN apply `lots_for_position` once per instrument group:

```python
contracts_by_inst: dict[str, int] = {}
for t in active_trades:
    ...
    contracts_by_inst[t_inst] = contracts_by_inst.get(t_inst, 0) + contracts

total = 0
for inst, n in contracts_by_inst.items():
    total += lots_for_position(inst, n)
return total
```

With this fix: 5 trades of 1 MNQ each → `contracts_by_inst = {"MNQ": 5}` → `lots_for_position("MNQ", 5)` → `micros_to_mini_equivalent(5)` → `ceil(5/10) = 1`. Correct.

### Layer 2: the path simulation

`trading_app/account_survival.py:221, 245, 289, 485-490`

```python
# Line 221 — compute per-trade lots once, store
lots = lots_for_position(params["instrument"], 1)
# Line 245 — store in TradePath
lots=lots
# Line 289 — aggregate across concurrent open trades
open_lots += trade.lots
# Line 290 — track max ever reached
max_open_lots = max(max_open_lots, open_lots)
# Lines 485-490 — compare against ladder cap
if scenario.max_open_lots > allowed_lots:
    scaling_breaches += 1
    scaling_feasible = False
```

**Same bug pattern, different code path.** Each `TradePath` stores `lots=1` (the ceiling of 1 contract), and `open_lots += trade.lots` sums them. 5 concurrent MNQ lanes → `max_open_lots = 5` → compared against 2-lot cap → breach.

This is the code path that produces the **68.1% scaling breach** rate in `account_survival --profile topstep_50k_mnq_auto`. It is the bug that generates the pulse BROKEN signal.

### Layer 3: the test that locks in the bug

`tests/test_trading_app/test_topstep_scaling_plan.py:237-249`

```python
class TestDayOneViolationScenario:
    """The exact scenario the audit (F-1) flagged as a BLOCKER."""

    def test_5_simultaneous_lanes_breach_50k_day1(self):
        """topstep_50k_mnq_auto has 5 lanes × 1 micro each.
        Day 1: balance=$0 → max 2 lots → 5 simultaneous = 5 lots > 2 = violation.

        Each lane is 1 micro = 1 mini-equivalent (ceiling). 5 lanes = 5 mini-equivalents.
        """
        active = [_Trade(_Strategy("MNQ"), 1) for _ in range(5)]
        total = total_open_lots(active)
        day_max = max_lots_for_xfa(50_000, 0)
        assert total == 5
```

**This test explicitly codifies the bug as the expected behavior** and cites F-1 in the class docstring. The docstring says "Each lane is 1 micro = 1 mini-equivalent (ceiling)" — which is incorrect per the canonical rule. The correct interpretation is "Each lane is 1 micro = 0.1 mini-equivalent; 5 lanes = 0.5 mini-equivalent = 1 lot (ceiling)."

### Layer 4: the contradiction with a sibling test

`tests/test_trading_app/test_topstep_scaling_plan.py:197-199`

```python
def test_one_micro_trade(self):
    active = [_Trade(_Strategy("MNQ"), 5)]
    assert total_open_lots(active) == 1  # 5 micros = 1 mini-equiv (ceiling)
```

**Same real-world exposure (5 MNQ micros open concurrently), different assertion.** If "5 MNQ micros = 1 lot" is correct in this test, then 5 MNQ micros as 5 trades × 1 contract must ALSO be 1 lot. The test suite is internally inconsistent — the author structured tests around the buggy code's mechanics, not around the canonical semantic.

### Layer 5: memory cascade

`C:\Users\joshd\.claude\projects\C--Users-joshd-canompx3\memory\MEMORY.md` references `topstep_canonical_audit_apr8.md` with the F-1 claim:

> "**F-1 Scaling Plan Day-1 violation** (5-lane bot is **2.5× over** 2-lot Day-1 cap on 50K XFA)"

The "2.5×" number is `5 / 2 = 2.5` — the exact ratio of the buggy per-trade ceiling sum (5) to the Day-1 cap (2). **The entire F-1 claim is derived from the buggy calculation.** There is no independent canonical evidence that the 5-lane profile violates the cap.

---

## 4. Numerical Verification

Ran from Python interpreter 2026-04-11:

```python
from trading_app.topstep_scaling_plan import lots_for_position, micros_to_mini_equivalent, max_lots_for_xfa

# Buggy per-trade ceiling aggregation (current sim behavior)
per_trade = sum(lots_for_position('MNQ', 1) for _ in range(5))
# → 5 lots

# Correct aggregate ceiling (canonical interpretation)
aggregate = lots_for_position('MNQ', 5)
# → 1 lot

# Day-1 cap from canonical ladder
max_lots_for_xfa(50_000, 0.0)
# → 2 lots

# Ceiling behavior for micros
[micros_to_mini_equivalent(n) for n in [1, 5, 10, 11, 15, 20, 21, 30]]
# → [1, 1, 1, 2, 2, 2, 3, 3]
```

**Observations:**

- `micros_to_mini_equivalent` itself is correct. 5 micros → 1 lot, 20 micros → 2 lots, 21 micros → 3 lots.
- `lots_for_position('MNQ', 5)` = 1 (correct aggregate). The CALLER that passes contracts=1 per trade breaks the calculation.
- Day-1 cap of 2 lots = 20 MNQ micros max. 5 MNQ micros uses 25% of the cap.

---

## 5. Historical Trade Concurrency (optional deeper verification)

*Not run in this audit — would require querying `live_journal.db` or `paper_trades` for actual lane concurrency patterns. Deferred because the bug finding is already strong enough on canonical-math grounds. Could be run as a secondary verification if the fix is questioned.*

The session-staggering of the 5 lanes (EUROPE_FLOW, TOKYO_OPEN, NYSE_OPEN) means real concurrency is probably **lower than 5**, making the real utilization even smaller than the 25% headroom number above.

---

## 6. Bias Check (per user directive "no bias")

### Counter-argument 1: "Maybe TopStep interprets the rule strictly per-position"

**Rejected.** Canonical Quote 2 says "any combination summing to 2 mini-equivalents." "Summing" is additive, not per-position. If TopStep used per-position ceiling, the quote would say "each position counts as at least 1 lot." It does not.

### Counter-argument 2: "Maybe the per-trade ceiling is intentional conservatism"

**Rejected.** The docstring for `total_open_lots` cites "conservative" only in the context of GROSS vs NET exposure (long + short vs long - short). There is no documented justification for per-trade ceiling. It is an implementation error, not a documented conservative choice.

### Counter-argument 3: "Maybe the audit that flagged F-1 had independent evidence"

**Rejected.** The "2.5× over" number in memory is mathematically `5/2`, which is the exact buggy sum divided by the cap. There is no way to derive 2.5× from the canonical rule independently. If the audit had independent evidence, it would cite a different number.

### Counter-argument 4: "Maybe TopStep charges per-order margin not per-aggregate"

**Unverified but unlikely.** The Scaling Plan article is about MAXIMUM CONCURRENT POSITION SIZE, not margin. Margin is a separate concern handled by the broker's initial-margin engine. The Scaling Plan rule is about net exposure at any point in time.

### Counter-argument 5: "Maybe 5 separate positions trigger some per-order risk check"

**Unverified.** The canonical Scaling Plan article does not mention per-order caps. A separate `max_concurrent_trades` limit exists in `risk_manager.py` but that's distinct from the Scaling Plan cap. If there is a per-order limit, it should be enforced as a separate constraint, not masked as a Scaling Plan violation.

### Honest conclusion

The audit finding is strong. The per-trade ceiling aggregation is a clean bug with no documented justification. The memory-recorded 2.5× number is derived from the bug and has no independent canonical grounding. The fix is well-bounded.

---

## 7. Proposed Fix — Design Proposal Gate

### Files to touch (blast radius)

1. **`trading_app/topstep_scaling_plan.py`**
   - Function: `total_open_lots(active_trades, instrument=None)`
   - Change: aggregate contracts by instrument BEFORE applying `lots_for_position`
   - LOC: ~15 lines changed

2. **`trading_app/account_survival.py`**
   - Functions: `_build_trade_paths` (line 221), `_scenario_from_trade_paths` (line 251)
   - Change: either store raw contracts per trade and aggregate at scenario build time, OR rewrite `_scenario_from_trade_paths` to group by instrument
   - LOC: ~20 lines changed

3. **`tests/test_trading_app/test_topstep_scaling_plan.py`**
   - Rewrite `test_5_simultaneous_lanes_breach_50k_day1` to assert `total == 1` (correct canonical value)
   - Rename class docstring to reflect that F-1 was a FALSE ALARM
   - Verify `test_one_micro_trade` and `test_filter_by_instrument` still pass (they test the single-trade case which is already correct)
   - Add new test `test_5_simultaneous_lanes_at_day1_cap_PASS` that asserts the correct behavior
   - LOC: ~30 lines changed

4. **Memory update:** `C:\Users\joshd\.claude\...\memory\topstep_canonical_audit_apr8.md` (or its equivalent memory entry)
   - Mark F-1 as RESOLVED — false alarm, fixed 2026-04-11
   - Reference this audit report
   - LOC: memory entry update

5. **`docs/audit/2026-04-08-topstep-canonical-audit.md`**
   - Update status tracker: F-1 marked "RESOLVED (false alarm) — see `docs/audit/2026-04-11-criterion-11-f1-false-alarm.md`"
   - LOC: 1-2 lines

### Blast radius — what must NOT break

- `topstep_scaling_plan.py`'s other functions (`max_lots_for_xfa`, `micros_to_mini_equivalent`, `lots_for_position`) are UNCHANGED — they're correct.
- All other callers of `total_open_lots` in the codebase (if any) need to be audited to make sure the new behavior doesn't regress anything. Grep needed.
- Drift check #92 (canonical-source annotations) is UNAFFECTED.
- The 52 existing tests in `test_topstep_scaling_plan.py` must still pass (except `test_5_simultaneous_lanes_breach_50k_day1`, which must be rewritten).
- `account_survival.py` regression tests (if any) must still pass.
- `risk_manager.py` tests — may or may not be affected depending on whether `total_open_lots` is called from `risk_manager`. Need to verify.

### What the fix does NOT change

- Canonical Scaling Plan ladder (correct)
- Day-1 cap values (correct)
- Micro-to-mini ratio (correct)
- Other F-x findings in the audit (F-2, F-2b, F-5, F-6, F-9, F-10, F-11 unaffected)
- `max_concurrent_trades` in `risk_manager.py` (separate constraint)
- The profile architecture of `topstep_50k_mnq_auto` (5 lanes is FINE under correct math)

### What the fix does NOT attempt

- Does not touch `account_survival.py`'s other scenario models (`trade_path_conservative` vs other modes) beyond the lot aggregation
- Does not add new canonical Scaling Plan rules (e.g., the deferred net-position article `intercom.help/topstep-llc/en/articles/8284209`)
- Does not re-enable Stage 4 ML DELETE or touch any file in that scope_lock
- Does not modify the wave5 filter class registration stage (still design-blocked)

### Acceptance criteria for the fix

1. `pytest tests/test_trading_app/test_topstep_scaling_plan.py` all pass with rewritten test
2. `pytest tests/test_trading_app/test_account_survival.py` (if exists) pass
3. `pytest tests/test_trading_app/test_risk_manager.py` pass (to verify `total_open_lots` consumers)
4. `python -m trading_app.account_survival --profile topstep_50k_mnq_auto` returns `scaling_feasible=True` with scaling breach rate at or near 0%
5. `python -m pipeline.check_drift` passes (no regression in drift check count)
6. Self-review: confirm no other caller of `total_open_lots` is silently broken by the fix
7. Memory + audit status tracker updated
8. Single clean commit (or 2 if fix vs docs split is cleaner)

### Kill criteria for the fix

- If any other caller depends on per-trade ceiling semantics (unlikely — would be a cascading bug) → stop, present to user
- If rewriting `account_survival.py` path simulation breaks a regression test for a different scenario → stop, present to user
- If drift check count changes unexpectedly → stop, present to user

### Rollback plan

`git reset --hard HEAD~1` (single commit) or `git revert <commit>` (if pushed).

---

## 8. Gaps in This Audit (per user directive "no gaps")

These are honestly-declared gaps the user should consider before authorizing the fix:

1. **Historical trade concurrency not queried.** The 25% headroom number assumes 5 lanes never go above 5 concurrent micros. If lanes can accidentally pile up through retries or unexpected fills, real concurrency could be higher. Not a showstopper (the Day-1 cap is 20 micros, so even 10 concurrent is fine), but worth noting.

2. **Live broker behavior not verified.** The audit is based on canonical documentation. TopStep's actual risk engine might apply the rule differently. A good test would be to paper-trade 3+ concurrent MNQ positions on a real practice XFA account and verify no auto-flattening. This is a real-world verification gap.

3. **Net vs gross position calculation deferred.** The canonical `intercom.help/topstep-llc/en/articles/8284209` about net position (long minus short) vs gross (long plus short) has not been fetched. If TopStep uses GROSS (which is the current implementation), same-instrument opposite-direction positions would count double — but the topstep_50k_mnq_auto profile is LONG-ONLY so this is not a concern for the current audit target.

4. **Only `topstep_50k_mnq_auto` analyzed.** The same bug pattern affects all profiles with multiple micro lanes. `topstep_50k_type_a` (5 copies × 16 max_slots) would also be affected. Fix should cover the aggregation logic once; all profiles benefit.

5. **Other F-x findings not re-audited.** The 2026-04-08 TopStep canonical audit flagged 15 findings. I audited only F-1. If F-1 was mis-interpreted, other findings MAY have similar issues. Not audited in this pass — recommend separate audit only if F-2 through F-11 show symptomatic problems.

6. **The test suite's INTERNAL inconsistency** (between `test_one_micro_trade` line 199 and `test_5_simultaneous_lanes_breach_50k_day1` line 249) should have been caught at review time. The code-review process missed it. Worth a separate audit of code-review discipline, not in scope here.

---

## 9. Next Step — Authorization Required

**The user directive was "audit and improve ... then implement."** Audit is complete. Improve = propose fix (this document). **Implement requires user confirmation before any code is touched** per the Design Proposal Gate in `.claude/rules/institutional-rigor.md` and `.claude/rules/workflow-preferences.md`.

**Waiting for user authorization to execute § 7 fix plan.**

If authorized, the execution sequence will be:

1. Write a minimal stage file at `docs/runtime/stages/criterion-11-f1-false-alarm-fix.md` with scope_lock
2. Fix `total_open_lots` aggregation in `topstep_scaling_plan.py`
3. Fix the path simulation in `account_survival.py`
4. Rewrite the two contradictory tests in `test_topstep_scaling_plan.py`
5. Add the new passing test
6. Grep for other `total_open_lots` callers and verify
7. Run affected test suites — show output
8. Run `python -m trading_app.account_survival --profile topstep_50k_mnq_auto` — show output
9. Run `python -m pipeline.check_drift` — show output (but only if Stage 4 DELETE parallel process is not touching check_drift.py)
10. Update memory + audit status tracker
11. Commit
12. Close stage file

**Estimated LOC touched:** ~65-80 across 5 files + memory update.
**Estimated time:** 30-45 minutes.
**Risk:** LOW — canonical-math fix with clear test coverage.

---

**End of audit report.**
