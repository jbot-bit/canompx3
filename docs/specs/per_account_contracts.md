# Per-Account Contracts — Canonical Spec

**Status:** Stage 3a-i (2026-06-11). Plumbing-only; live order path stays clamped
at `DEPLOYED_MAX_CONTRACTS_CLAMP=1` (`trading_app/portfolio.py:94`) until per-account
C11 survival proves >1 safe AND the operator GOes.
**Canonical home for:** `AccountProfile.account_contracts`
(`trading_app/prop_profiles.py`).

---

## Purpose

Supply a **real per-account contract count** to replace the inert uniform `{aid: 1}`
hardcode at `trading_app/live/session_orchestrator.py:694`. That hardcode feeds
`RiskManager.configure_accounts()` (Stage 2), which already arms a MODELED
daily-loss belt **per account** scaled by `contracts / primary_contracts`. With a
uniform map every account charges identical modeled dollars and halts together
(correct for `copies=1` / the 1:1 CopyOrderRouter mirror). A non-uniform map is
the first thing that makes the Stage 2 belts **diverge** — e.g. Express @ 1
contract halts later than a Combine @ 3 contracts on the same primary exit.

This is the **first independent multi-account seam** (Stage 3a). It is a
*size-only* divergence: every account still trades the same lanes; only the
contract count differs. True per-account *lane assignment* (strategy divergence)
is a separate later increment.

## The field

```python
# trading_app/prop_profiles.py :: AccountProfile
account_contracts: tuple[tuple[int, int], ...] = ()
```

- **Shape:** a tuple of `(account_id, contracts)` pairs. Frozen-dataclass-safe
  (tuples are hashable/immutable), version-controlled, drift-checkable.
- **Keyed by the broker `int` account_id — NOT by selection position.** This is
  load-bearing (see § Account-ID-keyed invariant below).
- **Default `()`** = every live account trades **1** contract. This is
  byte-identical to the pre-Stage-3a `{aid: 1}` hardcode. An unset field changes
  nothing.
- **`contracts >= 1`** for every pair, and **account_ids are unique** within the
  tuple. Enforced in `AccountProfile.__post_init__` (fail-closed: a malformed map
  is a construction error, never a silently-tolerated state).

## Resolution

```python
profile.resolve_account_contracts(live_account_ids) -> dict[int, int]
```

For each `aid` in the live set (primary + shadows, from
`session_orchestrator.order_router.all_account_ids`), the resolver returns the
declared count, **falling back to 1 for any account not listed**. The result is
exactly the `dict[int, int]` shape `RiskManager.configure_accounts()` already
consumes.

**Fail-safe, never silent-misassign:** because real broker account_ids are only
known at session start (they are discovered, not static config), the field is
OPTIONAL and the resolver defaults unlisted accounts to 1. A typo'd or stale
account_id in the map simply never matches a live account and contributes
nothing — it can never attach a contract count to the *wrong* physical account.

## Account-ID-keyed invariant (C1 — load-bearing)

The field MUST be keyed by broker account_id, never by position in the selection
order. `trading_app/live/preflight.py::_select_primary_and_shadow_accounts`
(`:957-967`) **reorders** the discovered account list — it `remove()`s the
requested `--account-id` and `insert(0, ...)`s it to the front so the
primary is always inside the `n_copies` slice. Broker discovery order is itself
not guaranteed stable.

A positional map (`account_contracts[0]` = primary, etc.) would therefore bind a
contract count to *whichever account happened to land at that index* — so
changing `--account-id` would silently move a 3-contract sizing onto a different
physical account. That is a capital misassignment. Keying by account_id makes the
map position-independent: account 202 gets its declared count no matter where the
selection logic places it.

This matches how `RiskManager._account_contracts` is already keyed
(`{account_id: contracts}`, `risk_manager.py:111`).

## Per-account survival (C2 — SizingContext, not the vestigial field)

Before a non-uniform map may arm a live cap >1, each account's divergent belt
must be proven safe by re-running Criterion 11 survival **at that account's
contract scale**. The contract count enters the sim through **`SizingContext`**
(`trading_app/account_survival.py:197`) — `account_equity` / `account_size` /
`max_contracts_by_strategy` — consumed by `_load_lane_trade_paths` (`:545`),
which pre-scales every contract-derived `TradePath` field.

`SurvivalRules.contracts_per_trade_micro` (`:139`) is **VESTIGIAL** — explicitly
"set but never read by `simulate_survival`". It is NOT the sizing knob. Any
per-account survival re-run varies `SizingContext`, never that dead field.

`evaluate_per_account_survival(profile_id, account_contracts)` reuses the
existing `_scenarios_for_context` → `simulate_survival` → `_evaluate_gate` chain
(the same machinery `sweep_survival_cap` uses) once per **distinct** contract
count in the map — re-encoding ZERO sim or gate math (institutional-rigor § 4).

## Relationship to neighbouring fields

| Field | Relationship |
|---|---|
| `copies` | Count of identical accounts. `account_contracts` sizes *within* that set per account_id. Independent. |
| `daily_lanes` | Per-lane execution spec (which strategies). `account_contracts` is size-only; identical lanes are still shared across accounts in 3a. |
| `daily_loss_dollars` | The per-account modeled belt cap. For `contracts > 1`, the belt's reachability is feasibility-checked (drift): a fixed dollar belt against ×contracts-scaled per-trade risk must still be bounded by `daily_loss_dollars` and the tier MLL — see drift check. |
| `is_express_funded` | Funded-wrapper flag. Orthogonal — `account_contracts` is a capability map, not a prop ladder. |

## Self-funded doctrine (capability map, NOT an earnings ceiling)

Per `.claude/rules/self-funded-sizing-doctrine.md`: `account_contracts` is a
**capability / margin / sanity** map — what each account *can* trade — never a
prop-firm contract cap leaked onto a self-funded earnings ceiling. Self-funded
sizing remains risk-first (drawdown → vol-targeting → margin → liquidity).

## Drift enforcement

`pipeline/check_drift.py::check_account_contracts_feasibility`:
1. Every `account_contracts` key/value is shape-valid: `int` account_id, `int`
   `contracts >= 1`, unique keys, positive.
2. **Feasibility (the new invariant, M1):** for any account with `contracts > 1`,
   a fixed `daily_loss_dollars` belt (if set) and the tier MLL (`tier.max_dd`)
   must still bound `contracts × single-contract worst-day-dollars`. A belt that
   cannot be reached before per-trade risk ×contracts blows the MLL is a config
   error — the divergence isn't survivable by the belt that's supposed to guard
   it.

(Broker account_ids are not statically knowable, so the check asserts *shape +
positivity + feasibility*, not membership in a specific account set.)

## Out of scope (Stage 3b / 3c — separately gated)

- Per-account broker-fill routing (the reserved `on_trade_exit(account_id=...)`
  real-fill path). 3a keeps the no-`execution_engine.py`-PnL-change constraint.
- Per-account lane assignment (true strategy divergence).
- Concentration cap across simultaneous accounts.
- Lifting `DEPLOYED_MAX_CONTRACTS_CLAMP` above 1.
- Operator-settable contracts via the dashboard (3a is canonical-source = the
  profile; dashboard read-only display is 3a-ii).
