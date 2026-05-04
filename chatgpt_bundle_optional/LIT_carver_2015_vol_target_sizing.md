# Carver 2015 — Volatility Targeting & Position Sizing (Ch 9-10, Systematic Trading)

**Source:** `resources/Robert Carver - Systematic Trading.pdf`
**Author:** Robert Carver (ex-AHL portfolio manager)
**Publication:** Harriman House, 2015 (ISBN 9780857194459)
**Extracted:** 2026-04-15
**Pages cited:** 135-155 (Chapter 9 Volatility Targeting, Chapter 10 Position Sizing overview)

**Criticality for our project:** 🟡 **HIGH** — this is the literature grounding for Stage 2+ sophisticated position sizing of our prior-day level signals. Currently we size binary (1 contract per lane). Carver's framework lets signal strength modulate size continuously.

---

## The Three-Layer Volatility Target Framework (Ch 9, p 135-151)

Carver's core claim is that every systematic trader should define risk via ONE number — a **percentage volatility target**: the expected annualized standard deviation of portfolio returns (p 137). All position sizing flows from this.

### Three cascading definitions (p 137-138)

```
Percentage volatility target   = desired annualized portfolio return σ (a %)
Trading capital                = cash at risk in the account (a $ value)
Annualised cash vol target     = Trading capital × Percentage vol target
Daily cash vol target          = Annualised cash vol target / 16
                                 (16 = √256, approximating 256 trading days/year)
```

Applied to our TopStep 50K MNQ setup: if we choose 25% percentage vol target, on $50K capital → $12,500 annualized cash vol → $781 daily cash vol target.

### Four tests for any proposed target (p 138)

1. **How much can you lose?** — what's the trading capital base
2. **How much risk can you cope with?** — psychological + account-level DD tolerance
3. **Can you realise that risk?** — leverage available + instrument natural volatility
4. **Is this level right for your system?** — Kelly-linked to expected Sharpe

---

## Kelly → Volatility Target Link (p 143-147)

**Key result (Carver p 143-144, citing Kelly 1956 via Poundstone's *Fortune's Formula*):**

> Optimal percentage volatility target = expected Sharpe ratio

So SR 0.5 → 50% vol target (full Kelly). SR 1.0 → 100% (blow-up risk).

**But full Kelly is too aggressive** (p 146). Carver's explicit recommendation:

1. Take back-tested SR, multiply by 0.75 for OOS degradation → realistic SR
2. Optimal full-Kelly % vol target = realistic SR (as a %)
3. **Use HALF-KELLY: halve that.** For negative-skew systems, HALVE AGAIN.

### Recommended volatility targets (Carver Table 25, p 147)

| Realistic SR | Positive/zero skew | Negative skew |
|---|---|---|
| 0.25 | 12% | 6% |
| 0.40 | 20% | 10% |
| 0.50 | 25% | 12% |
| 0.75 | 37% | 19% |
| 1.0+ | 50% | 25% |

**Applied to our lanes:** our live lanes report Sharpe ~0.8-1.2 backtested. Carver-realistic = 0.6-0.9. Half-Kelly = 30-45% vol target. Our skew is positive (ORB breakouts = positive skew, lottery-ticket style). **A 30-40% annualized vol target is Carver-grounded for our setup.**

---

## Loss Expectation Tables (Carver Tables 20-23, p 139-142)

At Sharpe 0.5, zero skew, $100K capital:

| Loss event | 25% vol | 50% vol | 100% vol | 200% vol |
|---|---|---|---|---|
| Worst daily loss/month | $2,500 | $5,000 | $10,000 | $20,000 |
| Worst weekly loss/year | $6,900 | $14,000 | $28,000 | $55,000 |
| Worst monthly/10yrs | $16,000 | $32,000 | $63,000 | $80,000 |
| 10%-tile cumulative loss | $9,300 | $15,000 | $30,500 | $62,000 |

**Chance of losing ≥half capital over 10 years (Carver Table 23):**
- 25% vol target: <1%
- 50% vol target: 10%
- 100% vol target: 40%
- 200% vol target: 93%

**Implication for prop firms:** TopStep-style accounts with $2.5K trailing DD force LOW effective vol targets. At $50K capital and $2.5K DD, we need "worst daily loss each month" ≪ $2.5K → maps to **≤25% vol target** per Carver Table 20.

---

## Rolling Up Profits/Losses (Carver p 149, Kelly criterion implication)

**Kelly says: adjust absolute risk to current capital, not initial.**

- Down $2K from $100K → capital now $98K → new cash vol target = $98K × pct_target (not $100K)
- Up $3K → capital now $103K → new cash vol target = $103K × pct_target

Carver's automated system checks account value and adjusts hourly (p 149). For non-automated: daily recalculation if vol target > 15%, weekly if ≤ 15%.

**Applied to our prop firm setup:** TopStep account with trailing DD requires CONSTANT recalculation of vol target relative to current equity minus lockable buffer, not static notional. Current code may not do this.

---

## Position Sizing Framework (Carver Ch 10, p 153+)

Three inputs combine:
1. **Combined forecast** (−20 to +20 scale; 0 = flat, ±10 = normal conviction, ±20 = max)
2. **Cash volatility target** (daily or annualised)
3. **Instrument price volatility** (recent std dev of price changes)

Produces **subsystem position** (number of contracts) — the actual trade size.

### Informal formula (Carver p 159, paraphrased)

```
Position = (Combined_forecast / 10) × (Cash_vol_target) / (Price × Instrument_vol × FX × Contract_multiplier)
```

Division by (Price × Vol × Contract_mult) converts cash risk budget into contract count using instrument's inherent volatility.

Multiplication by (Forecast/10) scales position by conviction: forecast +20 → 2× normal size; forecast +5 → 0.5× normal size; forecast 0 → flat.

**This is exactly the framework our prior-day level signals need.**

---

## Application to our project

### Current state (pre-Carver framework)

- Fixed 1-contract-per-lane sizing (no volatility scaling)
- No combined forecast — each filter is binary take/skip
- No Kelly or vol-target discipline — sizing is "whatever account allows"
- Rolling DD management is operational, not systematic

### What Carver enables for our prior-day level signals

**Stage 2 roadmap (per `mechanism_priors.md`):**

Instead of binary `SKIP_NEAR_PIVOT_LONG`:
```
forecast_NEAR_PIVOT = -10 when |orb_mid - pivot| / atr < 0.15
forecast_NEAR_PIVOT =  -5 when |orb_mid - pivot| / atr < 0.30
forecast_NEAR_PIVOT =   0 when |orb_mid - pivot| / atr < 0.50
forecast_NEAR_PIVOT =  +5 when |orb_mid - pivot| / atr >= 0.50
```

Combined with F5 BELOW_PDL signal (bullish when price below PDL):
```
combined_forecast = 0.5 × forecast_NEAR_PIVOT + 0.5 × forecast_BELOW_PDL
                  (weights per Carver Ch 8, handcrafted based on per-signal correlations)
```

Position = (combined_forecast/10) × cash_vol_target / (MNQ_price × MNQ_vol × $2)

### Pipeline placement for this framework

| Layer | Carver concept | Our location |
|---|---|---|
| Percentage vol target | Set once per account | New constant in `trading_app/prop_profiles.py` |
| Trading capital | Per-account cash | Existing: broker equity snapshot |
| Combined forecast | Weighted signal sum | NEW: `trading_app/forecast_combiner.py` |
| Instrument vol | Recent price std | EXISTING: `daily_features.atr_20` (proxy) |
| Position sizing | Formula | NEW: `trading_app/risk_manager.py` extension |

### Institutional constraints Carver teaches us

1. **Never use full Kelly.** Half-Kelly minimum. Quarter-Kelly for negative-skew (not our case, but ORB during news = tail risk).
2. **Back-tested SR has a 0.75× OOS discount built in.** Any size-calibration from backtest must apply this.
3. **Max 50% annualized vol target ever.** Any proposed sizing that implies > 50% requires formal justification per Table 25.
4. **Vol target stays fixed; cash target rolls up/down with capital.** Kelly auto-compounding, not target-changing.

---

## What this extract does NOT cover

Chapter 10 (Position Sizing) was only partially extracted (start, p 153). Full formulas with FX adjustments, contract multipliers, and instrument blocks are on pages 153-163. If precise deployment requires those, re-extract those pages.

Chapter 11 (Portfolios, p 165-175): instrument weights, diversification multiplier, portfolio-level risk budget. Not extracted. Relevant for multi-lane correlation-aware sizing — extract before Stage 4 portfolio-level deployment.

Chapter 12 (Speed and Size, p 177-203): cost-aware sizing, when capital is too small for leverage-dependent targets. Not extracted. Relevant for prop-firm small-account constraints.

---

## Usage rules (Extract-Before-Cite per `CLAUDE.md:79-81`)

1. Cite this extract for any size-modification logic in the position-sizing pipeline
2. Any Kelly-related claim must cite Carver Ch 9 p 143-147, not training memory
3. For signal-to-forecast conversion (continuous scaling of prior-day level signals), this extract is the canonical support
4. For portfolio-level weighting (instrument weights, diversification), ADDITIONAL extract of Carver Ch 11 required before claims

---

## Related literature

- `bailey_et_al_2013_pseudo_mathematics.md` — MinBTL constrains HOW MANY trial signals we can feed into Carver's forecast combiner
- `bailey_lopez_de_prado_2014_deflated_sharpe.md` — DSR correction must be applied to the back-tested SR before the 0.75 discount → halve → Kelly pipeline
- `fitschen_2013_path_of_least_resistance.md` — grounds the CORE ORB strategy that Carver's framework would size

## Related literature (not yet in `resources/`)

- Kelly 1956 original paper — Carver cites via Poundstone *Fortune's Formula*
- Thorp commentary on Kelly — Carver quotes footnote 107 p 146
