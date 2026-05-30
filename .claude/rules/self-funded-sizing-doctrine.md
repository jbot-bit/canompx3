# Self-Funded Sizing — Risk-First, Never Prop-Capped (HARD DOCTRINE)

**Load-policy:** auto-injected when editing `trading_app/prop_profiles.py`,
`trading_app/prop_portfolio.py`, `trading_app/account_survival.py`, or
`pipeline/check_drift.py`. Read on demand when sizing or building books for any
personal-capital (`self_funded`) account.

**Authority:** operator-emphatic, 2026-05-31 — *"I don't want to cap my live
personal capital earnings just because we started this on prop firms."* This is
a permanent project invariant, not a preference. It supersedes any convenience
that would route a prop-firm contract cap into a personal-capital sizing path.

---

## The distinction (load-bearing)

The system was **born on prop firms**, so prop-firm contract caps
(`PropFirmAccount.max_contracts_micro` / `_mini`, via `ACCOUNT_TIERS`) got wired
into the book-builder: `trading_app/prop_portfolio.py` →
`contract_budget = tier.max_contracts_micro` (currently line 558). `build_book`
has **no firm branch**, so that cap applies to `self_funded` books exactly as it
applies to a prop book. Left framed as an earnings ceiling, it silently throttles
real personal-capital earnings to a prop-firm-shaped limit. That is the trap.

Two different decision classes must never be conflated:

- **Sim-survival / rule-compliance** — modelled in
  `trading_app/account_survival.py`. Prop contract caps legitimately bind here:
  the sim models *the prop vehicle's own rules* (trailing DD, MLL, contract
  ceiling). This is correct and stays. (The sim's own per-trade contract count
  is hardcoded to 1 micro — `SurvivalRules.contracts_per_trade_micro` — it does
  NOT read the tier ceiling for sizing, so the sim is not a leak surface.)
- **Earnings capacity / book-building / position sizing** — for personal
  capital this is bounded by **risk, not by any prop firm's contract cap**.

## The rules

1. **Prop caps apply ONLY to prop-firm survival / rule-compliance sims.**
2. **Prop caps must NEVER bound self-funded book-building or personal-capital
   earning capacity.**
3. **Self-funded sizing is bounded, in order, by:**
   1. drawdown tolerance,
   2. volatility-targeting / Kelly-style risk sizing
      (`docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md`),
   3. broker margin,
   4. liquidity / slippage sanity limits.
4. **Any `max_contracts_*` on a `self_funded` tier is a broker / margin / sanity
   guard — NOT a prop-style earnings ceiling.** It exists so a fat-finger or a
   runaway allocator can't exceed account margin; it is not the binding bound on
   how much the strategy is allowed to earn.
5. **A drift check enforces this** so prop caps cannot silently leak into
   `self_funded` allocation paths:
   `check_prop_caps_do_not_leak_into_self_funded` in `pipeline/check_drift.py`.

## What this forbids

- Treating `tier.max_contracts_micro` as the *earnings* ceiling for any
  `self_funded` (or future personal-capital) profile.
- Copy-pasting a prop firm's contract ladder onto a self-funded tier as if it
  were a rule that must be obeyed for earnings purposes.
- Adding a personal-capital firm whose book-building binds on a prop-shaped
  contract cap instead of on risk.
- Silencing the leak guard to make a book "fit" a prop number.

## Current enforcement scope (honest floor, not ceiling)

The drift check is a **marker guard** today: it asserts (a) every `self_funded`
`ACCOUNT_TIERS` entry carries the `@margin-guard-not-earnings-cap` marker, and
(b) this doctrine file exists. It pins intent and fails loud if a new
`self_funded` tier is added without the marker, or if the doctrine is deleted.

It does NOT yet *structurally* prove the book-builder refuses to bind a
`self_funded` book on the prop cap — that firm-aware branch in
`prop_portfolio.py` is a separate, adversarial-audit-gated follow-up
(capital allocation path). The marker guard is the cheap immediate layer; the
structural branch is the eventual fix.

## Related

- `memory/doctrine_self_funded_sizing_risk_first_not_prop_capped.md` — the
  durable doctrine note (operator wording).
- `.claude/rules/institutional-rigor.md` § 5 (no dead/lying fields), § 6 (no
  silent failures) — a prop cap silently bounding personal earnings is exactly a
  silent failure.
- `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md`
  — the risk-first sizing the doctrine points to.
