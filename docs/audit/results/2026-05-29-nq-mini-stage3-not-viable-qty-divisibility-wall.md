# NQ-mini Stage 3 — NOT VIABLE AS-DESIGNED: the qty-divisibility wall

**Date:** 2026-05-29
**Author:** Claude Code (Opus 4.8), explanatory mode
**Verdict:** **CLOSED — activation blocked by a qty-divisibility wall.** The MNQ→NQ
order substitution wired (dormant) in Stage 2 (`ea0d4fec`) cannot be activated on
the live `topstep_50k_mnq_auto` profile without dead-blocking every order.
**No code, config, or profile mutation accompanies this finding.**

---

## Scope

This audit tests whether NQ-mini Stage 3 (activating the dormant MNQ→NQ order
substitution wired in Stage 2) is viable on the live `topstep_50k_mnq_auto`
profile. The question: can a 1-contract-per-order micro profile route as
whole-number full-size NQ contracts?

## 1. What Stage 3 was supposed to do

Stage 2 wired a dormant MNQ→NQ execution-substitution path: a live profile may
declare `execution_symbol_map = {"MNQ": "NQ"}` plus `execution_qty_divisor =
{"MNQ": 10}`, and the order-build path translates strategy MNQ signals into
broker NQ orders at `strategy_qty // 10`. **Driver:** NQ (full-size E-mini
Nasdaq) carries far lower commission per unit of notional than MNQ (micro);
the Stage 2 closeout cites ~77% commission reduction (≈$26K→$52K/yr per
contract-equivalent) on activation. Stage 3 was the explicit decision to
populate that map on a real profile.

## 2. The blocker (proven, not asserted)

The live `topstep_50k_mnq_auto` profile routes **exactly 1 contract per order**.
A MNQ→NQ divisor of 10 requires `strategy_qty % 10 == 0`; `1 % 10 == 1`, so
`resolve_execution_order` **fails closed on every order** by design, and the
orchestrator converts that to `ENTRY BLOCKED`. Activation would therefore halt
the entire profile — zero orders would ever reach the broker.

### Execution-path trace (PREMISE → TRACE → EVIDENCE → CONCLUSION)

| Step | File:line | Evidence |
|------|-----------|----------|
| Per-order size is clamped to `max_contracts` | `trading_app/execution_engine.py:922` | `trade.contracts = self._compute_contracts(risk_points, cost, trade.strategy.max_contracts)` |
| `_compute_contracts` clamps to `max_contracts` | `trading_app/execution_engine.py:294-301` | `if contracts > max_contracts: ... contracts = max_contracts` |
| Live profile-portfolio hardcodes `max_contracts=1` | `trading_app/portfolio.py:877` | `max_contracts=1,` |
| Dataclass default is also 1 | `trading_app/portfolio.py:86` | `max_contracts: int = 1` |
| ⇒ `event.contracts == 1` flows into substitution | `trading_app/live/session_orchestrator.py:2173` | `execution_contract, execution_qty = self._resolve_execution_order(event.contracts)` |
| `_resolve_execution_order` delegates to canonical resolver | `trading_app/live/session_orchestrator.py:2132-2138` | `resolve_execution_order(account_profile, self.instrument, strategy_qty)` |
| Resolver fails closed on non-integer division | `trading_app/prop_profiles.py:199-203` | `if strategy_qty % qty_divisor != 0: raise ValueError(...)` |
| Orchestrator converts ValueError → ENTRY BLOCKED | `trading_app/live/session_orchestrator.py:2140-2141` | `msg = f"ENTRY BLOCKED - {strategy_id}: invalid execution quantity ..."` |

**Worked case:** `resolve_execution_order(topstep_50k_mnq_auto, "MNQ", 1)` with
`execution_qty_divisor={"MNQ": 10}` → `1 % 10 == 1` → `ValueError` → every entry
on every lane of the live MNQ auto profile is rejected.

### Why the divisor is 10 (not negotiable downward without changing instruments)

`pipeline/cost_model.py:127` — *"NQ = E-mini Nasdaq 100 (full-size). 10x MNQ by
contract multiplier."* MNQ `point_value=10.0`-class micro vs NQ full-size: the
contract-multiplier ratio is 10, so 10 MNQ-equivalents == 1 NQ. The divisor is
declared as explicit profile data (`prop_profiles.py:126-128`) precisely so a
cost-spec edit cannot silently change executed exposure — it is correctly 10 and
cannot be lowered without changing what NQ *is*.

## 3. Why this is a wall, not a bug

The fail-closed behavior is **correct and load-bearing**, not a defect:
- Truncating `1 // 10 = 0` would silently drop the order (a silent failure —
  forbidden by institutional-rigor § 6).
- Rounding `1 → 1 NQ` would route **10× the intended notional** to the broker —
  a catastrophic over-size on a $50K trailing-MLL account.

The resolver raising is the only safe outcome. The incompatibility is structural:
**a 1-contract-per-order micro profile cannot be expressed as whole full-size
contracts.** Nothing in the wiring is wrong; the *economic premise* (route micros
as fulls) is unreachable at this sizing.

## Verdict

**CLOSED — NOT VIABLE AS-DESIGNED.** Stage 2 wiring (`ea0d4fec`) remains
permanently DORMANT under the current single-contract sizing model. No
`ACCOUNT_PROFILES` row is populated; `execution_symbol_map` stays `None`
(identity path) on every live and demo profile.

The parked action-queue item `nq_mini_stage2_wiring_2026_05_15` is closed on
this finding (no further `/next`-eligible work).

## 5. Unblock criteria (what would make activation viable)

Either of the following — both are new design work, NOT resumptions of this stage:

1. **Multiples-of-10 sizing.** Raise the live MNQ profile's effective per-order
   size so `strategy_qty` is reliably a clean multiple of the divisor (e.g.
   `max_contracts ≥ 10` *and* vol-sizing that actually produces ≥10). This is a
   capital/risk-limit decision — 10 MNQ ≈ 1 NQ ≈ 10× the current per-trade
   notional and tail risk; gated by Carver Table-20 ceiling re-derivation and the
   $2K MLL / $450 daily-loss breaker recalibration. Requires `/design` +
   `/capital-review`.
2. **Separate NQ-native profile (no substitution).** Stand up a dedicated Topstep
   profile that trades NQ directly where the strategy edge justifies full-size
   sizing. Substitution machinery stays dormant; this is a from-scratch profile
   design (account, sizing model, lane re-validation at full-size cost). Requires
   `/design`.

## Reproduction

Static execution-path trace, no runtime needed (the blocker is structural, not
data-dependent):

```bash
# 1. Confirm live per-order size ceiling is 1:
grep -n "max_contracts" trading_app/portfolio.py            # :86 default=1, :877 build=1
grep -n "contracts > max_contracts" trading_app/execution_engine.py  # :294 clamp

# 2. Confirm the divisor is 10 (NQ = 10x MNQ):
grep -n "10x MNQ" pipeline/cost_model.py                    # :127

# 3. Confirm fail-closed on indivisible qty:
grep -n "is not divisible" trading_app/prop_profiles.py     # :199-203 raise ValueError

# 4. Confirm ValueError -> ENTRY BLOCKED:
grep -n "ENTRY BLOCKED" trading_app/live/session_orchestrator.py  # :2141
```

Drift at commit time: `python pipeline/check_drift.py` → 167 passed, 0 violations,
21 advisory. No production `.py` staged → no test run required.

## Limitations

Per operator directive, this finding touches NOTHING outside the doc layer. The
concurrent sibling-terminal work (NYSE_PREOPEN O30 MNQ overlay; the uncommitted
`docs/institutional/pre_registered_criteria.md` Amendment 3.5 DSR universe lock)
was explicitly NOT read for mutation and NOT altered.

## References

- Stage 2 closeout (from git): `git show HEAD~1:docs/runtime/stages/2026-05-29-nq-mini-stage2-wiring-closeout.md` (deleted in working tree; content in baton)
- Driver economics: `memory/mini_vs_micro_commission_fix.md`
- Canonical resolver: `trading_app/prop_profiles.py:172-204`
- Live profile: `trading_app/prop_profiles.py:512-572`
