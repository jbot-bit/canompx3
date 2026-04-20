# Carver 2015 — Portfolios of Trading Subsystems (Ch 11, Systematic Trading)

**Source:** `resources/Robert Carver - Systematic Trading.pdf`
**Author:** Robert Carver (ex-AHL portfolio manager)
**Publication:** Harriman House, 2015 (ISBN 9780857194459)
**Extracted:** 2026-04-20
**Pages cited:** 165-176 (Chapter 11 Portfolios, body + summary)

**Criticality for our project:** 🟢 **HIGH — MANDATORY for multi-lane deployment decisions.** This is the literature grounding for our 6-lane portfolio construction, instrument weights, diversification multiplier, and correlation-aware sizing. Used by `docs/audit/results/2026-04-20-6lane-correlation-concentration-audit.md`.

Companion to `carver_2015_volatility_targeting_position_sizing.md` (Ch 9-10).

---

## Portfolios and instrument weights (p 165-167)

Each trading subsystem gets a positive **instrument weight**; weights sum to 100%. Portfolio-weighted position = subsystem position × instrument weight × instrument diversification multiplier.

- **Asset allocating investors** run a fixed portfolio of subsystems (one per instrument, constant forecast).
- **Staunch systems traders** have a group of trading rules whose forecasts combine to drive each subsystem.
- **Semi-automatic traders** (our project) don't hold fixed subsystems — they enter opportunistic bets on varying instruments over time.

## Instrument weights — systems traders / asset allocators (p 167-168)

Subsystems have been **volatility standardised** upstream (Ch 9-10), so all have equal expected return standard deviation. Handcrafting uses correlations:

1. Group assets by class (bonds / equities / FX / commodities) first.
2. Within each group, divide weight equally; within groups, use pairwise handcraft table (Ch 4, Table 8 p 79).
3. **Subsystem return correlation ≈ 0.7 × instrument return correlation** (p 167-168) for DYNAMIC trading systems. Static asset allocators: use instrument correlation without the 0.7 factor.

Rationale (p 167): dynamic systems spend time flat or reversed, which dilutes the return correlation vs pure long-only buy-and-hold.

Carver recommends handcrafting over bootstrapping unless a full rolling-OOS back-test is available (footnote 115 p 167).

> "**I wouldn't recommend adjusting instrument weights for Sharpe ratios**, since there's rarely enough evidence of different performance between subsystems for different instruments, even once we account for different levels of costs." (p 168)

**Implication for us:** equal subsystem weights are the default. Adjusting by Sharpe is discouraged unless cost structure materially differs — which it doesn't across our 6 MNQ lanes.

## Instrument weights — semi-automatic traders (p 169)

Semi-automatic traders (our project, before Carver full adoption) should:

- Allocate trading capital **equally between a notional maximum number of concurrent bets**.
- Weight per bet = 100% / max_concurrent_bets.
- Recommendation: max ≤ 2.5 × average number of concurrent bets — "for reasons that will become clear" (= diversification-multiplier clip).

## Instrument diversification multiplier (p 169-171)

Volatility-standardised subsystems have expected std dev below that of underlying assets. Uncorrelated assets → portfolio std << individual subsystem std → portfolio under-realizes target risk. The **instrument diversification multiplier** scales positions up to restore target risk.

### Formulation (p 170)

```
D = 1 / sqrt(w' · C · w)
```

Where `w` is the weight vector (sums to 1) and `C` is the correlation matrix of subsystem returns. For equal weights and a book of N subsystems, effective independent bets N_eff = D². (Carver expresses this via table 18 p 131 approximations.)

### Hard cap 2.5 (p 170)

> "It's possible to get very high values for the diversification multiplier, if you have enough assets, and they are relatively uncorrelated. However in a crisis such as the 2008 crash, correlations tend to jump higher exposing us to serious losses — an example of **unpredictable risk**. To avoid the serious dangers this poses I strongly recommend **limiting the value of the multiplier to an absolute maximum of 2.5**."

This is NOT a statistical cap — it is a robustness cap against regime change. Even if back-tested D computes to 3.5, you deploy 2.5.

### Semi-automatic-trader approximation (p 171)

D = max_bets / avg_bets. So if you average 4 bets and cap at 10, D = 2.5 (at the ceiling). Increasing max beyond 2.5× average is discouraged.

### Example (p 169-170, Table 29)

Trading-subsystem correlations for the Bond/S&P/NASDAQ example after 0.7× dynamic adjustment:

| | Bond | S&P | NASDAQ |
|---|---|---|---|
| Bond | 1.00 | 0.07 | 0.07 |
| S&P | | 1.00 | 0.53 |
| NASDAQ | | | 1.00 |

Avg off-diagonal = 0.22, D ≈ 1.41 (N_eff ≈ 2.0 from 3 nominal subsystems).

## A portfolio of positions and trades (p 171-174)

Once subsystem position (column K in Carver worksheet) is known, compute portfolio-weighted position:

```
portfolio_position_i = subsystem_position_i × instrument_weight_i × diversification_multiplier
```

Then **round** to integer blocks = **rounded target position**.

### Position inertia (p 174)

> "Position inertia is a way of avoiding small frequent trades that increase costs without earning additional returns. [...] I recommend that if the current position is within 10% of the rounded target position, then you shouldn't bother trading."

Rationale (p 174): two-sided cost of entering + exiting for a small incremental change rarely justifies the position-move. Only applies to holding-period > a few days; faster systems need tighter or no inertia.

## Summary table of variables (Carver p 175)

| Term | Definition |
|---|---|
| Subsystem position | Contracts per subsystem given forecast × cash vol target |
| Instrument weights | Must sum 100%; handcrafted or semi-auto 100%/max_bets |
| Instrument diversification multiplier | Accounts for subsystem correlation; HARD CAP 2.5 |
| Portfolio instrument position | subsystem × weight × multiplier |
| Rounded target position | Above, rounded to integer blocks |
| Desired trade | Round-target − current; apply 10% inertia rule |

---

## Application to our project

### How our 6-lane MNQ book maps to Carver's framework

| Carver concept | Our state (2026-04-20) |
|---|---|
| Max concurrent bets | 6 (one per deployed session-lane) |
| Avg concurrent bets | Unknown until measured; estimate 1.8-2.5 (most days only 2-3 sessions produce signals) |
| Semi-auto D approximation | max/avg ≈ 6/2.5 = 2.4 (near cap) |
| Empirical D (backtest, 2019-2026) | **2.412** (measured, audit 2026-04-20) |
| Carver cap | 2.5 |
| Effective independent bets | 5.82 of 6 nominal |
| Subsystem correlation | mean +0.006 (all pairs < 0.30) |
| Filter correlations adjusted 0.7× | N/A — already using measured subsystem returns, not instrument returns |

### Weighting decisions

**Current book:** 6 lanes, equal-weight per prop profile (TopStep 50k × 2 copies). Carver-grounded: OK, this matches semi-automatic trader default and handcraft-default both.

**Sharpe-weighted alternative:** Discouraged by Carver (p 168). We should NOT move to Sharpe-weighted subsystems without materially different cost structures across lanes (which we don't have — all MNQ, same session-agnostic cost model).

### Diversification multiplier decision

Current D = 2.412 (empirical). Applied at portfolio-position layer, this implies each per-lane subsystem position is multiplied by 2.412 to hit target portfolio vol. Combined with Carver's 2.5 cap: current book is at 96.5% of Carver's maximum. Adding a 7th lane could push to D ≈ 2.50; at that point further lanes give zero Carver-compliant value (you'd cap at 2.5 and re-size existing 6 down).

### The crisis-jump warning: backtest-refuted for this book

Carver's 2.5 cap exists explicitly because correlations jump in crises. Our audit measured correlations in COVID 2020-Mar (mean +0.003), 2022 bear (mean -0.001), and 2024-Aug vol spike (mean -0.070). **No jump observed in backtest.** Reason: lanes trade non-overlapping sessions — news shocks in one session cannot price-infect another until the news cycle propagates.

**Actionable:** we can reason as if D can safely approach 2.5 (not just be capped there) for THIS book's structure. The cap remains defensive; it becomes most binding if session-spread is lost (e.g., if we added a second US-hours lane overlapping NYSE_OPEN).

### Pipeline placement

| Carver layer | Our code location |
|---|---|
| Instrument weight | `trading_app/prop_profiles.py` (ACCOUNT_PROFILES — lane allocation %) |
| Diversification multiplier | NEW: `trading_app/portfolio_sizing.py` (to be built) |
| Position inertia | NEW: same file (10%-band filter on order routing) |
| Rolling correlation monitor | NEW: `trading_app/lane_correlation.py` (existing skeleton — extend with Carver-grounded 30-day rolling mean + pair-max > 0.30 alarm) |

---

## Usage rules (Extract-Before-Cite per CLAUDE.md)

1. Cite this extract for any portfolio-weighting decision (equal vs Sharpe-weighted → default equal).
2. Cite for any diversification-multiplier computation (always cap at 2.5).
3. Cite for any rolling-correlation-monitor design (crisis-jump tripwire comes from p 170).
4. Cite for position inertia default (10% band, p 174).

---

## Related literature

- `carver_2015_volatility_targeting_position_sizing.md` — Ch 9-10 precedes this, provides cash vol target and subsystem position inputs feeding the portfolio layer.
- `fitschen_2013_path_of_least_resistance.md` — grounds the ORB breakout subsystem whose portfolio is being constructed here.
- `bailey_lopez_de_prado_2014_deflated_sharpe.md` — DSR haircut applies BEFORE half-Kelly sizing (Ch 9) which feeds into this chapter's subsystem positions.
