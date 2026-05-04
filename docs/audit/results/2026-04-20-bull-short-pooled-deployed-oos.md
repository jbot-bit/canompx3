# Bull-day short avoidance — pooled deployed-lane OOS test

**Date:** 2026-04-20 (same day as lane-specific verify; follow-up to power-floor correction)
**Scan:** `research/bull_short_avoidance_pooled_deployed_oos.py`
**Data cutoff:** orb_outcomes + daily_features through 2026-04-19
**Verdict:**
- **Pooled-portfolio scope:** **DEAD** (no universal effect)
- **NYSE_OPEN-specific scope:** **CONDITIONAL** (IS valid, OOS still underpowered)

---

## Why this test

Following the 2026-04-20 RULE 3.3 correction on the lane-specific verify
(`2026-04-20-bull-short-avoidance-deployed-lane-verify.md`), the highest-EV
next action was stated as: pool 2026 Q1 shorts across all 6 DEPLOY lanes under
the same bull/bear prior-day partition, so OOS per-group N climbs from ~20 to
~100+ and the dir_match gate becomes powered.

The pooled test answered a question that turned out to be more important:
**is the bull-short effect a universal prior-day-direction mechanism, or is it
localized to specific sessions?**

## Method

- Canonical sources only: `orb_outcomes`, `daily_features`, `bars_1m`
- Lane roster read from `docs/runtime/lane_allocation.json` (6 DEPLOY lanes,
  rebalance 2026-04-18), `parse_strategy_id()` decomposes each strategy_id
  into its canonical lane spec
- Canonical filter delegation via `research.filter_utils.filter_signal` for
  each lane's individual filter
- Triple-join (`trading_day`, `symbol`, `orb_minutes`) for every lane; shorts
  concatenated across lanes with `lane_id` preserved
- Mode A holdout at `HOLDOUT_SACRED_FROM` (2026-01-01)
- Moving-block bootstrap (block_len=5, n_boot=5,000, seed=20260420) with
  labels FIXED (RULE 3.3 correction lineage)
- RULE 3.3 power-floor check on OOS vs IS Cohen's d

## Results

### Pooled across all 6 DEPLOY lanes

N total: 4,141 shorts. IS=3,949 / OOS=192 (3-month holdout).

| Split | bear N | bear mean | bear WR | bull N | bull mean | bull WR | delta | p_welch |
|---|---|---|---|---|---|---|---|---|
| **IS** | 1,792 | +0.0962 | 49.3% | 2,157 | +0.0900 | 49.2% | **+0.006** | **0.864** |
| **OOS** | 92 | +0.047 | 45.7% | 100 | +0.280 | 56.0% | -0.233 | 0.163 |

- Pooled IS block-bootstrap p (labels fixed, 5k draws): **0.868**
- Pooled IS WR spread: **+0.001** (indistinguishable from zero)
- bear>bull IS years: 5/7
- RULE 3.3 power: Cohen's d=0.005, OOS power=5.0%, N per group for 80% power=520,396
- Tier: **STATISTICALLY_USELESS** (because IS effect ~ 0, not because N is small)

### Per-lane IS breakdown — the decisive detail

| Lane | bear N | bear mean | bull N | bull mean | delta | p |
|---|---|---|---|---|---|---|
| **MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12** | 360 | +0.184 | 465 | +0.026 | **+0.157** | **0.019** |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5 | 339 | +0.113 | 387 | +0.071 | +0.042 | 0.627 |
| MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5 | 358 | +0.059 | 451 | +0.044 | +0.014 | 0.858 |
| MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15 | 187 | -0.012 | 230 | +0.017 | -0.030 | 0.794 |
| MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12 | 224 | +0.081 | 257 | +0.192 | -0.111 | 0.294 |
| MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15 | 324 | +0.096 | 367 | +0.221 | -0.125 | 0.174 |

- NYSE_OPEN is the ONLY lane with a statistically significant effect.
- Three lanes flip sign (SINGAPORE_OPEN, TOKYO_OPEN, US_DATA_1000) — bull-day
  shorts actually OUTPERFORM bear-day shorts on those sessions.
- The 2026-04-04 pooled-universe p=0.0007 was riding on NYSE_OPEN alone;
  pooling across the deployed portfolio reveals the signal is NOT universal.

### Per-lane OOS (each cell N~15-20 per group — power useless)

All 6 lanes except SINGAPORE_OPEN show bull > bear in 2026 Q1 (directional
pattern consistent with a possible regime shift OR noise from N<20 per group).
Per-lane OOS delta is not individually interpretable at this N.

## Framework analysis (applied post-run)

### TRUTH CHECK
| Claim | Status |
|---|---|
| "Universal bull-exhaustion across deployed shorts" | **DEAD** |
| NYSE_OPEN-specific effect | **STILL VALID** (unchanged by pooling) |
| Effect is lane-specific, not market-structural | **CONFIRMED** — opposite sign on 3 of 6 lanes |

### DE-TUNNEL
NYSE_OPEN is US cash open. The hypothesized mechanism (bull-day overnight
retracement pressure → US short dip-buying stops) plausibly exists at NYSE
but would NOT exist at TOKYO_OPEN (Asia session, different participants) or
EUROPE_FLOW (London participants, different flow drivers). The opposite-sign
lanes are consistent with a different intraday-momentum continuation at
those sessions. This was NOT considered in the original 2026-04-04 pooled
audit — it framed the effect as universal.

### EDGE EXTRACTION
- Real edge: isolated to NYSE_OPEN. +0.157R/trade, 7/7 years, WR spread +7.8%
  over N=825 shorts 2019-2025 (Mode A).
- No universal deployable rule.
- Opposite-sign lanes are a research side-question — not immediate edge.

### BRUTAL FILTER
- The 2026-04-04 queue entry "implement on all NYSE_OPEN lanes when they deploy"
  was imprecise — there is ONE NYSE_OPEN lane deployed. The rule is
  lane-specific, not session-general.
- 3 lanes with opposite-sign effect is a yellow flag that cross-session
  prior-day translation is invalid. Any future feature claim should be tested
  per-lane, not pooled-universe.

### FINAL DECISION

| Scope | Verdict |
|---|---|
| Deployed-portfolio pooled | **DEAD** (p=0.87, delta≈0, 3 lanes flip sign) |
| NYSE_OPEN-specific (lane COST_LT12) | **CONDITIONAL** (IS valid, lane-specific OOS underpowered per RULE 3.3) |

## Implications

### For live trading
**Zero immediate action.** No filter is live. The deployed portfolio continues
running its existing 6 lanes unchanged. The question of whether to half-size
bull-day shorts on the NYSE_OPEN lane specifically is DEFERRED — two options:

1. **Harvey-Liu haircut path (prop-desk standard):** deploy the half-size rule
   on NYSE_OPEN only, apply a 50% Sharpe haircut to the expected uplift, run
   a 3-6 month live shadow monitor alongside, kill on Shiryaev-Roberts CUSUM
   breach. EV under haircut ≈ half of naive IS projection.
2. **Wait-for-powered-OOS path:** don't deploy, accumulate 2026+ lane-specific
   OOS until per-group N ≥ 100 (≈ 5 more quarters ≈ 2027 Q1 at current trade
   rate). Cleaner but costs the IS-period opportunity.

**Recommend (1) with shadow monitor** only if portfolio can withstand
additional research-phase complexity. Otherwise default to (2) for hygiene.

### For the memory / research queue
- Supersede both 2026-04-04 (`bull_short_avoidance_signal.md` original) and the
  2026-04-20 intermediate "PARKED" classification.
- New classification: **session-specific CONDITIONAL on NYSE_OPEN only,
  REJECTED as universal rule.**
- The "opposite-sign on 3 of 6 lanes" finding is its own research question —
  is there a session-specific regime where bull-day shorts OUTPERFORM?
  Potential parking-lot item.

### For research methodology
- RULE 3.3 power-floor enforcement worked as intended — caught my own
  RULE 3.2 violation from the morning session.
- The pooled test was higher-EV than waiting for lane-specific OOS to grow.
- New "per-lane before pooled" rule reinforced: pooling MUST come with a
  per-lane breakdown that sanity-checks homogeneity of effect.

## Files

- Script: `research/bull_short_avoidance_pooled_deployed_oos.py`
- Reusable helper: `research/oos_power.py` (canonical RULE 3.3 enforcement)
- Previous verdict doc: `docs/audit/results/2026-04-20-bull-short-avoidance-deployed-lane-verify.md`
- Methodology addition: `.claude/rules/backtesting-methodology.md` § 2026-04-20 entries
- Feedback memory: `memory/feedback_oos_power_floor.md`, `memory/feedback_pooled_not_lane_specific.md`
