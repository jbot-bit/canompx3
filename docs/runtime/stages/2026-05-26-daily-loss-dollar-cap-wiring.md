---
task: |
  IMPLEMENTATION — Wire an explicit dollar-denominated daily-loss circuit
  breaker for live profiles, replacing the silent 5.0R magic default.

  GROUNDING (Topstep canonical resources):
  - topstep_50k_mnq_auto is an Express Funded Account (XFA).
  - Per docs/research-input/topstep/topstep_dll_article.md, XFA accounts have
    NO mandatory broker Daily Loss Limit — it is an opt-in safety net Topstep
    deliberately removed from auto-application. So PropFirmAccount
    daily_loss_limit=None for the topstep 50K tier is CORRECT, not a bug.
  - The binding broker guard is the $2,000 Maximum Loss Limit (trailing DD),
    already wired fail-closed via AccountHWMTracker
    (session_orchestrator.py:805-826) and the cumulative equity DD breaker
    (session_orchestrator.py:543, max_equity_dd_r = -$2000/avg_risk).
  - The per-DAY software circuit breaker (Portfolio.max_daily_loss_r, default
    5.0R) is a SELF-IMPOSED discipline belt, not a broker rule.

  MEASURED (live data, 4 deployed MNQ lanes 2026-05-26): avg median_risk =
  $76.12/trade. So the current 5.0R stop already halts at ~$381/day — already
  below the $2K MLL. Operator decision: make the daily belt an EXPLICIT $500
  dollar cap (intentional round-number discipline), dollar-denominated so it
  does not drift as lane composition changes (R would).

  DECISION (operator, 2026-05-26): TRUE dollar-denominated breaker, halting at
  exactly -$500 realized loss/day, summing real per-trade pnl_dollars — NOT an
  R-approximation. Rationale: R drifts per-trade (each trade's R uses its own
  risk, not the portfolio average), so an R-derived cap lands ~$400-650 not
  exactly $500. The orchestrator already computes per-trade pnl_dollars
  (session_orchestrator.py:2057) and the engine has cost_spec + contracts +
  risk_points, so a dollar breaker is wireable end-to-end with crash recovery.

  GROUNDING for the $450 figure (official + empirical, not ad-hoc):
  - Carver 2015 Table 20 (docs/institutional/literature/
    carver_2015_volatility_targeting_position_sizing.md:72-83): for a prop
    account with ~$2K trailing DD, "worst daily loss each month" must be ≪ the
    DD → ≤25% vol target. $450 = 22.5% of the $2,000 per-account MLL, under
    Carver's ceiling.
  - EMPIRICAL: 100k-day Monte Carlo on the REAL 2026 per-trade risk_dollars
    distribution from orb_outcomes (NOT the stored validated_setups median,
    which understates 2026 risk ~2x because MNQ price ~doubled over the 6yr
    backtest). 2026 per-account daily P&L: avg losing day -$156, 5%-tile -$288,
    1%-tile -$431, worst -$1018. $450 fires ~1% of days/account — a genuine
    tail, not ordinary variance. ($400 fires 1.5%/acct ≈ 11% household with
    copies=2 = strangles edge; $500 = exactly at Carver ceiling.)
  - PER-ACCOUNT semantics: engine tracks ONE account's contracts; CopyOrderRouter
    mirrors to shadows. Each Topstep XFA has its own independent $2K MLL, so the
    breaker is correctly per-account. copies=2 → both halt together when primary
    hits -$450. A $431 per-account day = 22% of THAT account's MLL (compliant).
  - METHODOLOGY NOTE (operator catch): risk limits MUST be calibrated on the
    live (current-year) risk distribution actually traded, never the historical
    backtest median. Stored median_risk_dollars is a 6yr average across a period
    where price was ~half today's.

  FIX (true $ breaker, end-to-end):
    1. AccountProfile gains `daily_loss_dollars: float | None = None`; set 500.0
       on topstep_50k_mnq_auto only (None = unchanged for all others).
    2. RiskLimits gains `max_daily_loss_dollars: float | None = None`.
    3. RiskManager tracks `daily_pnl_dollars`; can_enter Check 1 ALSO halts on
       dollar breach; on_trade_exit accepts pnl_dollars.
    4. ExecutionEngine tracks daily_pnl_dollars, computes per-trade pnl_dollars
       (mirrors orchestrator:2057), passes to on_trade_exit.
    5. SessionOrchestrator wires RiskLimits.max_daily_loss_dollars from profile;
       persists daily_pnl_dollars to safety_state symmetric to daily_pnl_r.
    6. SessionSafetyState persists daily_pnl_dollars (crash recovery).
    7. Drift check: declared daily_loss_dollars must be < tier.max_dd (MLL).

  EDGE CASES to harden (capital work):
    - daily_loss_dollars=None → entire dollar path inert; R path unchanged.
    - pnl_dollars uncomputable (risk_points None) → fail-closed: do NOT silently
      skip the dollar accrual; log + use R-equivalent fallback OR refuse entry.
    - Crash mid-day → safety_state restores daily_pnl_dollars AND daily_pnl_r;
      both must match the same trading_day stamp or reset.
    - EOD rollover → daily_pnl_dollars resets with daily_pnl_r (same daily_reset).
    - Partial fills / multi-contract → pnl_dollars uses event.contracts.
    - Signal-only → breaker still accrues (preview), never routes orders.

mode: IMPLEMENTATION
scope_lock:
  - trading_app/prop_profiles.py
  - trading_app/risk_manager.py
  - trading_app/execution_engine.py
  - trading_app/live/session_orchestrator.py
  - trading_app/live/session_safety_state.py
  - trading_app/portfolio.py
  - pipeline/check_drift.py
  - tests/test_trading_app/test_prop_profiles.py
  - tests/test_trading_app/test_risk_manager.py
  - tests/test_trading_app/test_execution_engine.py
  - tests/test_trading_app/test_session_orchestrator.py
  - tests/test_trading_app/test_session_safety_state.py
agent: claude (opus 4.7)
note: |
  SCOPE > 5 files (7 production). Justified: a daily-loss circuit breaker is ONE
  atomic safety mechanism; splitting it leaves a half-wired breaker (RiskLimits
  field with no engine accrual = silent no-op = unsafe). Per task-splitter
  doctrine, upstream-before-downstream splitting would create an unsafe
  intermediate state, so this ships as one verified stage. Operator acked wider
  scope ("wired in full proper").
---

## Blast Radius

- `trading_app/prop_profiles.py` — ADD `daily_loss_dollars: float | None = None` field to `AccountProfile` (new optional field, defaults None = no change for all existing profiles). SET `daily_loss_dollars=500.0` on `topstep_50k_mnq_auto` only. No other profile touched. `is_express_funded` drift-check pattern is the precedent for adding a declared field.
- `trading_app/portfolio.py` — `build_profile_portfolio()`: after strategies loaded + avg median-risk computed (the value already computed for DD check), if `profile.daily_loss_dollars is not None`, convert to R: `daily_loss_r = profile.daily_loss_dollars / avg_median_risk_dollars`, clamp/log, pass to `Portfolio(max_daily_loss_r=...)` instead of the 5.0 default. If None → unchanged 5.0 default. Fail-closed: if avg risk is 0/unavailable but a dollar cap is set, raise (cannot honor the cap silently).
- Downstream consumers UNCHANGED in behavior: `session_orchestrator.py:571` still negates to `-abs(...)`; `bot_state.py:284` dashboard display still reads the R value; `portfolio.py:1421` worst-case calc still multiplies. They just receive a calibrated R now.
- `pipeline/check_drift.py` — NEW check: every profile that declares `daily_loss_dollars` must have it strictly less than its tier `max_dd` (the MLL). Asserts the belt halts before the broker MLL. Self-reported count bumps by 1.
- Tests: `test_prop_profiles.py` (field exists, topstep_50k_mnq_auto=500.0, others None), `test_portfolio.py` (dollar→R conversion correct; None path unchanged; fail-closed on zero avg risk with a cap set).

## Reads / Writes
- Reads: `gold.db` read-only (validated_setups median_risk_dollars, existing path). Canonical tier via `get_account_tier`.
- Writes: none (config + conversion only). No DB, no allocation, no broker, no live-runtime file.

## Done criteria
1. New tests pass (show output) — conversion, None-path, fail-closed.
2. `python pipeline/check_drift.py` passes (show count + 0 violations); new check enumerated.
3. `grep -r` confirms no dead code; no other profile silently changed.
4. Live verify: `build_profile_portfolio('topstep_50k_mnq_auto').max_daily_loss_r` × avg risk ≈ $500.
5. Self-review against institutional-rigor §§ 4 (canonical median-risk reuse), 6 (no silent failure — fail-closed on zero risk), 8 (verify before claim).
6. Adversarial-audit gate (HIGH — live risk path) AFTER fix, BEFORE marking CLOSED.

## Doctrine references
- `docs/research-input/topstep/topstep_dll_article.md` — XFA has no mandatory DLL (verbatim grounding).
- `institutional-rigor.md` § 4 — reuse canonical median-risk conversion, don't re-encode.
- prop_profiles.py `@lfa-only` / F-3 deferred finding — LFA DLL wiring stays deferred (no LFA today).
