# Project Reference — Canompx3 (Consolidated)

Assembled 2026-03-22, updated 2026-03-23. Merges architecture, data flow, validation details, research methodology routing, and strategic direction into one document.

---

## Architecture

### Data Flow
```
Databento .dbn.zst files
  -> pipeline/ingest_dbn.py (validate, select front contract)
  -> gold.db:bars_1m (1-minute OHLCV, UTC timestamps)
  -> pipeline/build_bars_5m.py (deterministic 5m aggregation)
  -> gold.db:bars_5m
  -> pipeline/build_daily_features.py (ORBs, sessions, RSI, ATR)
  -> gold.db:daily_features
  -> trading_app/outcome_builder.py (pre-compute trade outcomes)
  -> gold.db:orb_outcomes
  -> trading_app/strategy_discovery.py (grid search ~105K combos)
  -> gold.db:experimental_strategies
  -> trading_app/strategy_validator.py (walk-forward + FDR)
  -> gold.db:validated_setups
  -> scripts/tools/build_edge_families.py (cluster by trade-day hash)
  -> gold.db:edge_families
```

### Database
- DuckDB (`gold.db`) — single local file, no cloud sync
- All timestamps UTC. Trading day = 09:00 Brisbane -> next 09:00 Brisbane
- Brisbane timezone: Australia/Brisbane (UTC+10, no DST)

### Design Principles
- **Fail-closed:** Any validation failure aborts immediately
- **Idempotent:** All operations safe to re-run (DELETE+INSERT pattern)
- **Pre-computed outcomes:** 5/15/30m ORB apertures, reused for all discovery
- **One-way dependency:** pipeline/ -> trading_app/ (never reversed)

### Price Data Sources

Active instruments (Mar 2026): MGC, MNQ, MES.

| Symbol | Source | Cost Model | Status |
|--------|--------|------------|--------|
| MGC | GC (full gold) | $10/pt, $5.74 RT | ACTIVE |
| MES | ES (pre-Feb 2024), then native MES | $5/pt, $3.74 RT | ACTIVE |
| MNQ | MNQ (native micro) | $2/pt, $2.74 RT | ACTIVE |
| M2K | RTY (E-mini Russell) | $5/pt, $3.24 RT | DEAD (0/18 null test) |
| M6E | 6E (full EUR/USD) | $12,500/pt, $3.74 RT | DEAD (0/2064 validated) |
| SIL | SI (full silver) | $1,000/pt, $20 RT | DEAD (0/432 validated) |
| MCL | CL (full crude oil) | $100/pt | DEAD (0 validated) |

---

## Entry Models (Current)

| Code | Mechanism | Risk | Bias Status |
|------|-----------|------|-------------|
| E1 | Market order at NEXT bar open after N confirm bars close outside ORB | ~1.18x ORB width | **None.** Industry standard. |
| E2 | Stop-market at ORB level + 1 tick slippage. Triggers on ANY bar crossing ORB (fakeouts included). | ~1.0x ORB width + slippage | **Minimal.** Conservative crossing logic. |
| E3 | Limit order at ORB edge, wait for retrace after confirm. Stop-breach guard. | 1.0x ORB width | Fill-on-touch bias. **Soft-retired.** |

**E0: PURGED Feb 2026.** 3 compounding optimistic biases:
1. Fill-on-touch (filled on exact boundary touch vs industry standard of fill-through)
2. Fakeout fill exclusion (only counted fills where bar also confirmed — excluded fakeout losers)
3. Fill-bar wins (entry and target on same 1m bar — intra-bar sequence unknown)

E0 won 33/33 instrument/session combos in backtests. This was artifact, not evidence. Replaced by E2.

### Confirm Bars (CB)
CB1-CB5 = 1-5 consecutive 1m bars must close outside ORB. If ANY bar closes back inside, count resets.

### Risk-Reward (RR)
Target = RR x risk. Risk = |entry_price - stop_price|. Stop = other side of ORB.
- E2: entry = ORB boundary + slippage, risk ~ ORB size
- E1: entry = next bar open (past boundary), risk = ORB size + overshoot (~1.18x)

### ORB Size Filters
| Code | Meaning | Edge |
|------|---------|------|
| NO_FILTER | Take every trade | LOSES money. No edge. |
| G4+ | ORB >= 4 points | Breakeven starts here. |
| G5+ | ORB >= 5 points | Solid edge. |
| G6+ | ORB >= 6 points | Strongest per-trade edge, fewer trades. |
| G8+ | ORB >= 8 points | Very selective. |

---

## Sessions (12 total, all dynamic/event-based)

All resolve times dynamically per-day from SESSION_CATALOG, eliminating DST contamination.

| Code | Event | Reference TZ |
|------|-------|-------------|
| CME_REOPEN | CME daily reopen | 5:00 PM CT |
| TOKYO_OPEN | Tokyo open | 10:00 AM AEST |
| SINGAPORE_OPEN | Singapore/HK open | 11:00 AM AEST |
| LONDON_METALS | London metals open | 8:00 AM UK |
| US_DATA_830 | US economic data release | 8:30 AM ET |
| NYSE_OPEN | NYSE equity open | 9:30 AM ET |
| US_DATA_1000 | US 10 AM data/post-open | 10:00 AM ET |
| CME_PRECLOSE | CME equity pre-close | 2:45 PM CT |
| COMEX_SETTLE | COMEX gold settlement | 1:30 PM ET |
| NYSE_CLOSE | NYSE equity close | 4:00 PM ET |

---

## Strategy Classification (per RESEARCH_RULES.md)

| Class | Min Trades | Usage |
|-------|-----------|-------|
| HIGH-CONFIDENCE | 500+ | Institutional-grade if multi-regime |
| CORE | 200+ | Reliable for capital allocation if mechanism exists |
| PRELIMINARY | 100-199 | Actionable with caveats |
| REGIME | 30-99 | Conditional signal only. Never standalone. |
| INVALID | < 30 | Not tradeable |

**Trade Day Invariant:** A valid trade day requires BOTH: (1) a break occurred, AND (2) the strategy's filter makes the day eligible. Low trade counts under strict filters (G6/G8) are EXPECTED, not bugs.

**Edge Families:** Strategies with identical trade days are clustered into families. One head per family represents the cluster.

---

## Validation Pipeline Details

### Grid Search Space
Grid size varies by instrument and session (E2/E3 are CB1-only, filters are session-specific). Query `experimental_strategies` for current count. 3 active instruments, 12 sessions, 3 apertures.

### Validation Gates (in order)
1. **Walk-forward:** 3-year train / 1-year test, rolling windows. WFE > 50% required.
2. **BH FDR:** Benjamini-Hochberg correction. Honest test count per instrument.
3. **Classification:** CORE/REGIME/INVALID by trade count.
4. **Edge Family Clustering:** Dedup strategies with identical trade days.

### Current State (volatile — query for current)
Strategy counts, family counts, and live portfolio change after every rebuild. Query `validated_setups`, `edge_families`, and `live_config` for current numbers. 3 active instruments (MGC, MNQ, MES). M2K dead Mar 2026. min_sample=30 locked. 2026 holdout SACRED.

---

## Research Test Sequence (from Strategy Blueprint)

When evaluating ANY new idea, follow these gates IN ORDER:

### Gate 1: Mechanism Check
Why should this work? What structural market reason exists? "The numbers show it works" is NOT a mechanism. Must have plausible structural basis (friction, liquidity, institutional flow).

### Gate 2: Baseline Viability
Is there positive ExpR at any point in the variable space? Before declaring any dimension dead, test >= 3 values (e.g., RR: 1.0, 1.5, 2.0; Aperture: O5, O15, O30; Entry: E1, E2; Sessions: ALL).

### Gate 3: Statistical Significance
Does the positive baseline survive multiple testing? BH FDR at honest test count. p < 0.05 to note, p < 0.01 to recommend, p < 0.005 for discovery claims.

### Gate 4: Walk-Forward Validation
Does the finding hold out-of-sample? WFE > 50% = likely real. < 50% = likely overfit. Expected Sharpe decay 50-63% from IS to OOS.

### Gate 5: Sensitivity Analysis
Change parameter +/-20%. If small changes kill the finding, it's curve-fit, not real.

---

## ML Status (Research Only — NOT Production)

3 open methodology FAILs from adversarial audit:
1. **Negative baselines violate de Prado's meta-labeling assumptions** — need positive ExpR underlying
2. **Underpowered samples** — EPV=2.4 (need 10+). Reduce features to <=5 or pool sessions.
3. **Selection bias** — quality gates then bootstrap on same data (White 2000 violation)

Bootstrap (5K perms, Phipson & Smyth): 3/7 PASS (p=0.0016, 0.0176, 0.0376), 3 MARGINAL, 1 FAIL.
ML on negative baselines: DEAD (threshold artifact, p=0.350).
Positive-baseline ML: UNTESTED post-fixes. Layer 2 is paper-only.

---

## Confirmed NO-GOs (Complete List)

- No filter / L-filters: ALL negative ExpR
- MGC/MES unfiltered: negative at all RR
- M2K, MCL, SIL, M6E, MBT: 0 validated strategies each
- E0: PURGED (3 backtest biases)
- ML on negative baselines: DEAD (p=0.350)
- Non-ORB strategies: 30 archetypes tested, ALL DEAD
- Calendar cascade, cross-asset lead-lag, prior-day context: NOISE
- Breakeven trail stops, VWAP overlay, RSI reversion, gap fade: ALL negative
- Quintile conviction: DEAD (vol filter on best decade)
- Pyramiding: structural NO-GO (correlated intraday positions = tail risk)
- Inside Day / Ease Day filters: ALL worse vs G4 baseline
- V-Box CLEAR filter: filters nothing (98% pass)
- Volume confirmation: does NOT predict follow-through
- Session cascade: prior session range does NOT predict next session quality
- MES LONDON_METALS E3: 7/8 years negative, 81% double-break rate
- E3 retrace timing optimization: 0 survivors after FDR
- Wider ORB apertures (10-60min): 5min has highest per-trade avgR
- Transition ORBs (dead zones): lack volume for follow-through
- Cross-instrument TOKYO_OPEN LONG portfolio: MNQ/MES correlation +0.83

---

## Strategic Direction (Mar 23, 2026)

- **Raw baseline = the tradeable path.** Layer 1 (O5 RR1.0) statistically clean (p<1e-9, survives Bonferroni).
- **8-strategy live portfolio (holdout-clean):** MNQ 7 (CME_PRECLOSE x5, COMEX_SETTLE, CME_REOPEN) + MES 1 (CME_PRECLOSE ATR70_VOL via rolling).
- **MNQ RR1.0 at NYSE_OPEN + COMEX_SETTLE:** effect is real. 2026 forward test is binding.
- **3 pre-registered strategies** for 2026 holdout (NYSE_OPEN, COMEX_SETTLE, CME_PRECLOSE). Holdout is SACRED.
- **Rolling portfolio now functional.** Rebuilt with event-based session names. HOT-path stability scoring operational.
- **Threshold audit COMPLETE.** All 9 thresholds sensitivity-tested ±20%. 8/9 robust. DOUBLE_BREAK=0.67 is the one load-bearing heuristic (CME_PRECLOSE margins: MNQ 7.6pp, MES 5.0pp, MGC 1.7pp). Proximity warning added.
- **MGC 0 live = structural scarcity, not a bug.** Only validated edge is TOKYO_OPEN (all noise_risk=True, 100% double-break degraded in rolling). Live specs are MNQ-shaped. Open question: instrument-specific spec policy.
- **ML = research-only.** Fix 3 methodology FAILs before expanding.
- **Prop firm path:** Apex manual -> Tradeify+TopStep auto -> IBKR self-funded.
- **0.75x stops for prop** — survival over income.

## Methodology Grounding (Mar 23, 2026 audit)

Pipeline backbone is academically grounded:
- **BH FDR:** Benjamini & Hochberg 1995 (in resources)
- **Walk-forward:** Aronson Ch.12, Carver Ch.3, de Prado AFML
- **Multiple testing awareness:** Bailey & de Prado (Deflated Sharpe), Carver (p.76)
- **Rolling windows (12/18m):** de Prado AFML Ch.7, Carver (p.73-75)
- **Cost filtering:** Chan, Carver, every book
- **Mechanism requirement:** Aronson Evidence Based TA framework

Threshold grounding (all ±20% sensitivity-tested 2026-03-23):
- `STABLE_THRESHOLD=0.6` — ROBUST, no live cliff (heuristic, annotated)
- `TRANSITIONING_THRESHOLD=0.3` — ROBUST, label only (heuristic)
- `FULL_WEIGHT_SAMPLE=50` — ROBUST, 91% MNQ at full weight (heuristic)
- `stress_multiplier=1.5` — ROBUST, wide margin all live strategies (heuristic)
- `MIN_EXPECTANCY_R=0.10` — ROBUST, smooth gradient (heuristic)
- `FIT/WATCH/DECAY (15/10/-0.1)` — ROBUST, monitoring only (heuristic)
- **`DOUBLE_BREAK_THRESHOLD=0.67` — SENSITIVE/HEURISTIC.** One-directional cliff.
  Per-instrument margins: MNQ 7.6pp, MES 5.0pp, MGC 1.7pp (trending up).
  Proximity warning fires at 5pp. No academic source. Mechanism sound, value arbitrary.
  BUG FIXED: was blending across 7 instruments (incl 4 dead), now per-instrument.

---

## What Works (Confirmed Edges)

1. ORB size IS the edge — G4+ minimum, without it ALL edges die
2. MNQ E2 = ONLY positive unfiltered baseline
3. MGC/MES need G5+ on select sessions
4. TOKYO_OPEN is LONG-ONLY (short breakouts negative)
5. Negative session correlation provides real diversification (TOKYO_OPEN vs LONDON_METALS: -0.39)
6. E1 = honest conservative, E2 = honest aggressive
7. Kill losers early at CME_REOPEN and TOKYO_OPEN (T80 research)
8. Winter edges stronger across instruments (DST resolution benefit)
