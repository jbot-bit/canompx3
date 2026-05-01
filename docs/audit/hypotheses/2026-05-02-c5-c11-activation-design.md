# C5 (DSR) + C11 (MC account-death) — Activation Gate Design

**Locked:** 2026-05-01 evening (design only; implementation lands Stage 4 of `handoff-2026-05-02.md`)
**Cohort:** 5 PR #51 CANDIDATE_READY cells, **re-verified live 2026-05-01** against canonical `orb_outcomes`
**Author:** Claude (Opus 4.7) ahead of next session

---

## Truth-state verification — HONESTY CORRECTION 2026-05-01 evening

PR #51 body is metadata. I tried to re-derive but discovered:

```
# scratch-policy: WHERE pnl_r IS NOT NULL — see feedback_scratch_pnl_null_class_bug.md

Session        Apt RR  N_IS_NOW  ExpR_IS_NOW  N_OOS_NOW  ExpR_OOS_NOW  PR#51_N_IS  PR#51_ExpR  N_DRIFT
US_DATA_1000   15  1.5     1717      +0.1203         72       +0.1922       1495      +0.1063     +222
NYSE_OPEN      15  1.0     1715      +0.0895         73       +0.1828       1545      +0.0974     +170
US_DATA_1000   15  1.0     1717      +0.0921         72       +0.2052       1594      +0.0966     +123
NYSE_OPEN       5  1.5     1719      +0.0988         75       +0.0953       1650      +0.0953     + 69
NYSE_OPEN       5  1.0     1719      +0.0777         75       +0.1278       1693      +0.0807     + 26
```

**THE PROBLEM:** PR #51 was locked 2026-04-21. IS window is `trading_day < 2026-01-01` (fixed —
no time has passed in IS). But N grew by 26-222 trades per cell. This means **`orb_outcomes`
for these lane specs has been restamped/rebuilt between 2026-04-21 and 2026-05-01**, adding
trades to the IS window.

**This is a data-state drift.** The PR-locked DSR/MinBTL/BH-FDR was computed against a
data state that no longer exists. Comparing PR's `t_IS=+3.42` to my recompute of `t=+4.39`
is meaningless — different N, different sample, different denominator.

**What I cannot conclude:** "candidates still pass" — that requires re-running the K=105
scan against current data state, not just observing the cells still have positive ExpR.

**What I CAN conclude:**
- 5 cells exist in `orb_outcomes` now with N_IS ≥ 1715 and ExpR > 0
- 5 cells have OOS N ≥ 72 with positive sign (no FLIP)
- LA-registry grep returned 0 matches (necessary but not sufficient — would need to
  trace `orb_outcomes` build path to confirm true LA-cleanliness)

**What's required before C5/C11 can run:** re-run `research/mnq_unfiltered_baseline_cross_family_v1.py`
against current data state to get a fresh K=105 family with current `family_sharpes` and a
current BH-FDR cut. That is a new scan, not a re-validation of an old one. It MUST live
under its own pre-reg per `research-truth-protocol.md` Phase 0 rules.

If the re-run produces fewer than 5 survivors → cohort changes → this design doc is for the
WRONG cohort and must be updated.

---

## Why C5 + C11 are the right gates

Per `pre_registered_criteria.md` Amendment 3.2 + binding criteria:

- **C5 (Deflated Sharpe Ratio)** — Bailey-LdP 2014. Corrects ŜR for selection bias under multiple testing AND non-Normal returns. The **primary** strategy-selection gate. Quote (p.3 verbatim): *"a backtest where the researcher has not controlled for the extent of the search involved is worthless."*
- **C11 (MC account-death)** — Carver 2015 Ch12 framing. Even a positive-EV strategy can blow an account at the wrong allocation. Validates that the proposed weight × candidate distribution does not exceed the account's drawdown capacity at >1% probability over a 250-trading-day horizon.

**Currently implemented:** C1, C2, C8, C9 (in `strategy_validator.py:_check_criterion_*`).
**Currently MISSING:** C5, C11 as runnable validator methods.

This doc designs the implementation. No code is written here — Stage 4b implements per this design.

---

## C5 — Deflated Sharpe Ratio

### Formula (Bailey-LdP 2014 Equation 2, page 8 verbatim)

```
DSR ≡ PŜR(ŜR_0) = Z[ ((ŜR - ŜR_0) · √(T-1)) / √(1 - γ̂₃·ŜR + (γ̂₄-1)/4 · ŜR²) ]

where:
  ŜR_0 = √V[{ŜR_n}] · ((1-γ)·Z⁻¹[1 - 1/N] + γ·Z⁻¹[1 - 1/(Ne)])
  γ    = 0.5772156... (Euler-Mascheroni)
  Z    = standard normal CDF
  Z⁻¹  = standard normal inverse CDF
  e    = Euler's number
```

### Variables we need per candidate

1. **ŜR** — annualized Sharpe ratio of the candidate (we have this; recompute from `orb_outcomes`)
2. **T** — sample length in trading observations (= N_IS for our cells; per-trade Sharpe convention)
3. **V[{ŜR_n}]** — variance of Sharpe across the **discovery family** (= K=105 cells in PR #51 scan per the pre-reg `b4089f9c`)
4. **N** — implied independent trials. Per Bailey-LdP Appendix A.3 Eq.9: `N̂ = ρ̂ + (1 - ρ̂) · M`. With M=105, need ρ̂ from cross-cell return correlation.
5. **γ̂₃, γ̂₄** — skewness, kurtosis of candidate returns

### Implementation plan (`trading_app/strategy_validator.py`)

Add `_check_criterion_5_dsr(meta: dict, returns: pd.Series, family_sharpes: list[float], avg_correlation: float) -> tuple[str | None, str | None]`:

```python
def _check_criterion_5_dsr(meta, returns, family_sharpes, avg_correlation):
    """C5 — Deflated Sharpe Ratio per Bailey-LdP 2014 Eq. 2.

    Returns (failure_msg | None, advisory_msg | None).
    Pass condition: DSR >= 0.95 (matches pre_registered_criteria.md C5).
    """
    import numpy as np
    from scipy.stats import norm

    if len(returns) < 30:
        return ("C5: returns sample too small for DSR (N<30)", None)

    sr = returns.mean() / returns.std(ddof=1)
    sr_annualized = sr * np.sqrt(252)  # convention: per-trade->annualized
    skew = returns.skew()
    kurt = returns.kurtosis() + 3  # pandas returns excess kurt; Bailey uses raw kurt

    M = len(family_sharpes)
    if M < 2:
        return ("C5: family size <2 — cannot compute V[{ŜR_n}]", None)
    rho_hat = max(0.0, min(1.0, avg_correlation))  # clamp [0,1]
    N_hat = rho_hat + (1.0 - rho_hat) * M
    V_sr = np.var(family_sharpes, ddof=1)

    gamma = 0.5772156649
    sr_0 = np.sqrt(V_sr) * (
        (1 - gamma) * norm.ppf(1 - 1/N_hat)
        + gamma * norm.ppf(1 - 1/(N_hat * np.e))
    )

    T = len(returns)
    denom = np.sqrt(1 - skew * sr + ((kurt - 1) / 4) * sr**2)
    if denom <= 0 or not np.isfinite(denom):
        return ("C5: DSR denominator non-finite — non-normality breaks formula", None)

    z = ((sr - sr_0) * np.sqrt(T - 1)) / denom
    dsr = norm.cdf(z)

    meta["dsr_score"] = float(dsr)
    meta["sr0_at_discovery"] = float(sr_0)
    meta["n_implied_independent_trials"] = float(N_hat)

    if dsr < 0.95:
        return (f"C5: DSR={dsr:.4f} < 0.95 (selection-corrected SR not significant)", None)
    return (None, f"C5: DSR={dsr:.4f} (PASS, N̂={N_hat:.0f}, T={T})")
```

### How to obtain `family_sharpes` and `avg_correlation`

The PR #51 scan locked `K_family = 105` cells. Their per-cell Sharpe is in the scan output (or recomputable from `orb_outcomes`). Average correlation between candidate returns is computed from the per-day pnl_r series of each cell.

**Critical scope note:** `V[{ŜR_n}]` MUST be computed across the **discovery family the candidate was selected from** (the K=105 PR #51 scan), NOT across all of `validated_setups`. Mixing discovery families inflates V and underestimates DSR.

### Failure mode awareness

- **DSR < 0.95** → KILL. The candidate's reported Sharpe is consistent with selection bias from a 105-cell search.
- **DSR formula denominator goes negative** → indicates extreme non-Normality. Handle with the explicit error message — do NOT silently fall back to PSR or to bare Sharpe (the whole point of DSR is the non-Normality correction).
- **Returns N < 30** → cannot compute reliable skew/kurt. Refuse to run rather than emit a junk number.

---

## C11 — Monte Carlo account-death

### Framing (Carver 2015 Ch12 doctrine + LdP 2020 path-dependent risk)

Even a positive-EV strategy with DSR > 0.95 can blow an account if:
- The proposed weight × distribution puts >1% probability mass on losing >X% of equity within 250 trading days.
- For TopStep $50k XFA: max trailing drawdown $2,000 = 4% of account. C11 should reject any candidate where `P(max_drawdown > $2000 | proposed_weight) > 0.01` over 250 trading days.

### Implementation plan

Add `_check_criterion_11_mc_account_death(meta: dict, returns: pd.Series, account_size_dollars: float, max_dd_dollars: float, weight: float, n_paths: int = 10000, horizon_days: int = 250) -> tuple[str | None, str | None]`:

```python
def _check_criterion_11_mc_account_death(meta, returns, account_size_dollars, max_dd_dollars, weight, n_paths=10000, horizon_days=250):
    """C11 — Monte Carlo account-death simulation.

    Bootstrap n_paths trade sequences from returns distribution; on each path,
    track cumulative P&L and the max drawdown reached over horizon_days.
    Fail if P(max_dd_observed > max_dd_dollars * account_size) > 0.01.
    """
    import numpy as np

    if len(returns) < 100:
        return ("C11: returns sample <100 — bootstrap unreliable", None)

    # Convert per-trade pnl_r to per-trade dollar P&L at proposed weight
    # weight = fraction of account per trade; risk_per_trade = weight * max_dd_dollars
    risk_per_trade = weight * max_dd_dollars
    trade_dollars = returns.values * risk_per_trade  # pnl_r is in R units; R = risk_per_trade
    trades_per_year = meta.get("trades_per_year", 50)
    n_trades_per_path = int(round(horizon_days / 252 * trades_per_year))

    if n_trades_per_path < 10:
        return ("C11: <10 trades expected over 250-day horizon — too sparse", None)

    rng = np.random.default_rng(42)  # deterministic seed for reproducibility
    n_blow = 0
    max_dds = []
    for _ in range(n_paths):
        sample = rng.choice(trade_dollars, size=n_trades_per_path, replace=True)
        equity = np.cumsum(sample)
        peak = np.maximum.accumulate(equity)
        dd = peak - equity
        max_dd = dd.max()
        max_dds.append(max_dd)
        if max_dd > max_dd_dollars:
            n_blow += 1
    p_blow = n_blow / n_paths
    meta["mc_p_account_death"] = float(p_blow)
    meta["mc_max_dd_p99"] = float(np.percentile(max_dds, 99))

    if p_blow > 0.01:
        return (f"C11: P(max_dd > ${max_dd_dollars:.0f}) = {p_blow:.4f} > 0.01 at weight={weight}", None)
    return (None, f"C11: P(account_death)={p_blow:.4f} (PASS)")
```

### Inputs from `lane_allocation.json` / profile

- `account_size_dollars` = $50,000 (TopStep XFA profile constant)
- `max_dd_dollars` = $2,000 (TopStep trailing drawdown limit)
- `weight` = comes from allocator's per-lane weight at proposed allocation

### Failure mode awareness

- **P(blow) > 1%** at proposed weight → C11 KILL. Allocator must rebalance to lower weight OR lane is rejected.
- **<100 trade returns** → refuse. A 50-trade bootstrap on a 250-day horizon underestimates tail risk.
- **Random seed = 42 fixed** for reproducibility. Drift check should re-run with a different seed and confirm `p_blow` differs by <0.005 (Monte Carlo stability test).

---

## Wiring into existing `validate_strategy()`

`strategy_validator.py:410 def validate_strategy(...)` is the entrypoint. Add C5 and C11 calls inside the criteria loop, in this order:

1. C1 (hypothesis file) — already implemented
2. C2 (MinBTL) — already implemented
3. **C5 (DSR)** — NEW
4. C8 (OOS sign + ratio, tiered Amendment 3.2) — already implemented
5. C9 (era stability) — already implemented
6. **C11 (MC account-death)** — NEW
7. C12 (Shiryaev-Roberts monitor) — DEFER. SR is a post-deployment monitor per Pepelyshev-Polunchenko 2015. Not an IS validation gate. Implement when first lane is monitored, not now.

C5 must run BEFORE C8 because if DSR fails, OOS dir-match is moot.
C11 must run LAST because it depends on the proposed allocator weight, which only makes sense after C1-C9 have passed.

---

## Pre-registered acceptance criteria for the 5 candidates

Run-order:
1. Implement C5 method, unit-test on a known-PASS strategy (an existing deployed lane should clear DSR > 0.95) AND a known-FAIL synthetic (e.g., random-walk returns with M=105 trials).
2. Run C5 on each of 5 candidates. Record DSR, sr_0, N̂.
3. Implement C11 method, unit-test on a known-FAIL synthetic (3-sigma loss distribution with weight=1.0 must blow account ≥50%).
4. Run C11 on each of 5 candidates at proposed allocator weight (from a dry-run rebalance after Stage 1+2 land).

**Promotion conditions** (all must be true):
- DSR ≥ 0.95
- C8 PASS at the cell's OOS tier
- P(account_death | weight) ≤ 0.01
- Direction sign IS == OOS (already cleared via re-verification 2026-05-01)
- N_IS ≥ 1000 (all 5 cells have ~1700 — comfortably clear)

**Kill conditions** (any fails):
- DSR < 0.95 → strategy is selection bias not edge
- C11 fails → allocator lowers weight OR lane rejected
- C8 fails → OOS does not confirm IS
- Any 2 of the 5 fail any gate → STOP and audit cohort, do not partially activate

---

## Risk awareness — what this design might miss

1. **`family_sharpes` from K=105 scan output**: PR #51's K=105 scan was run against a **prior data state**. The current data state has 26-222 more IS trades per cell. Re-running `research/mnq_unfiltered_baseline_cross_family_v1.py` against current data is REQUIRED before C5 runs. This means the activation path is now:
   a. New pre-reg for "PR #51 cohort re-validation under current data state" (per `research-truth-protocol.md` Phase 0)
   b. Re-run K=105 scan
   c. Confirm 5 cells (or which subset) still survive BH-FDR q<0.05 + Chordia t≥3 + WFE
   d. THEN run C5 against the surviving subset's family_sharpes
   e. THEN run C11 at proposed allocator weight
2. **Average correlation `ρ̂`**: not currently computed. Need a one-off script that pulls per-day pnl_r from each of 105 cells and computes pairwise correlation matrix → average. Costs ~5 min compute on `orb_outcomes`.
3. **C11 weight assumption**: if the allocator's rebalance puts a candidate at weight 0.10 (1/10 of risk budget), C11 might pass; at weight 0.25, C11 might fail. Must run C11 against the allocator's actual proposed weight, not a default.
4. **Trades-per-year for C11**: from `validated_setups.trades_per_year` — but candidates aren't in `validated_setups` yet. Use `len(IS_returns) / years_in_IS` as the substitute, but document the substitution.
5. **Scratch-row exclusion bias** (`feedback_scratch_pnl_null_class_bug.md`): the candidate stats above used `WHERE pnl_r IS NOT NULL` which silently drops scratches. ExpR is upper-bound. **C5 + C11 should be re-run after the scratch-EOD-MTM fix lands** (separate stage).

---

## What this doc does NOT do

- Implement code. That's Stage 4b in the handoff.
- Replace pre-registered criteria. C5 and C11 are already in `pre_registered_criteria.md`; this is the implementation-level design only.
- Pre-register a new hypothesis. The hypothesis ("these 5 candidates can deploy") is PR #51's own pre-reg.
- Decide allocation weights. That's the allocator's job after Stages 1+2 land.

---

## Cross-references

- `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md` — DSR formula, Eq.2 + Eq.9
- `docs/institutional/literature/carver_2015_ch12_speed_and_size.md` — position sizing prior for C11
- `docs/institutional/literature/lopez_de_prado_bailey_2018_false_strategy.md` — burden of proof
- `docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md` — C12 (deferred)
- `docs/institutional/pre_registered_criteria.md` — binding criterion definitions
- `trading_app/strategy_validator.py:410` — entrypoint to extend
- `docs/runtime/handoff-2026-05-02.md` — Stage 4 of tomorrow's plan
- PR #51 body — `gh pr view 51` (locked 2026-04-21, sha b4089f9c)
- `memory/feedback_scratch_pnl_null_class_bug.md` — known bias in scratch-NULL handling
- `docs/audit/results/2026-04-28-e2-lookahead-contamination-registry.md` — confirmed candidates clear of E2 LA
