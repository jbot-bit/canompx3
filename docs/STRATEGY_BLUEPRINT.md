# Strategy Research Blueprint

Single reference doc for strategy research, planning, and implementation. Open this BEFORE starting any session.

**Authority:** This doc ROUTES to governing docs. It does not override them.
- Trading logic → `TRADING_RULES.md`
- Research methodology → `RESEARCH_RULES.md`
- Code structure → `CLAUDE.md`
- Feature specs → `docs/specs/`

**Freshness:** Variable space and active state sections are VOLATILE. Query canonical sources, don't cite from memory. Last structural audit: 2026-03-21.

---

## 1. Quick Reference — "What Am I Doing?"

| If you're... | Go to section | Also read |
|-------------|--------------|-----------|
| Researching a new idea | §3 Research Test Sequence | RESEARCH_RULES.md |
| Training/evaluating ML | §6 ML Sub-Pipeline | `trading_app/ml/config.py` |
| Building/changing a portfolio | §4 Variable Space | TRADING_RULES.md Session Playbook |
| Setting up paper trading | §7 Paper Trading Checklist | Pre-registration doc |
| Running a pipeline rebuild | §8 Pipeline Order | `docs/ARCHITECTURE.md` |
| Checking if something is dead | §5 NO-GO Registry | TRADING_RULES.md "What Doesn't Work" |

---

## 2. The One Thing That Matters

**ORB size IS the edge.** Cross-instrument stress test (Feb 2026) proved: strip the size filter and ALL edges die. CB, RR, entry model are refinements. Without G4+, only MNQ E2 has positive baseline. Source: TRADING_RULES.md "ORB Size = The Edge".

**Current reality (from gold.db, verified 2026-03-21):**
- **MNQ E2:** ONLY instrument × entry model with positive unfiltered baseline
- **MGC E2 unfiltered:** Negative at all RR (−0.06 to −0.33R). Needs G5+ size filter.
- **MES E2 unfiltered:** Negative at all RR (−0.04 to −0.25R). Needs G4+ size filter.
- **E1 unfiltered:** Negative everywhere including MNQ (−0.03 to −0.20R)
- **Only 11 validated setups exist.** All MNQ. All E2. CME_PRECLOSE + COMEX_SETTLE + BRISBANE_1025.

→ `from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS`  # ['MES', 'MGC', 'MNQ']
→ `from trading_app.config import ENTRY_MODELS`  # ['E1', 'E2', 'E3']

---

## 3. Research Test Sequence

When evaluating ANY new idea, follow these gates IN ORDER. Each gate has a kill criterion.

### Gate 1: Mechanism Check
**Question:** Why should this work? What structural market reason exists?
**Reference:** RESEARCH_RULES.md "The Mechanism Test"
**Pass:** Plausible structural mechanism (friction, liquidity, institutional flow)
**Kill:** "The numbers show it works" is not a mechanism. → STOP or add extra scrutiny.
**Warning:** Mechanism check does NOT prove causation. It screens obvious artifacts.

### Gate 2: Baseline Viability
**Question:** Is there positive ExpR at any point in the variable space?
**Test:** Query `orb_outcomes` grouped by relevant variables.
**Kill:** Negative ExpR at ALL tested combinations → DEAD for this approach.
**CRITICAL RULE:** Before declaring a variable dimension dead, test at LEAST 3 values across that dimension.
- RR: test 1.0, 1.5, 2.0 minimum
- Aperture: test O5, O15, O30
- Entry model: test E1, E2
- Sessions: test ALL sessions, not just 2-3

### Gate 3: Statistical Significance
**Question:** Does the positive baseline survive multiple testing?
**Test:** BH FDR at honest test count (not cherry-picked count)
**Reference:** RESEARCH_RULES.md "Significance Testing"
**Thresholds:** p < 0.05 minimum to note. p < 0.01 to recommend. p < 0.005 for discovery claims.
**Kill:** 0 survivors after FDR → noise, not edge.
**Always report:** N trades, time span, number of variations tested, exact p-values.

### Gate 4: Out-of-Sample Validation
**Question:** Does it hold on unseen data?
**Test:** Walk-forward, holdout split, or pre-registered forward test.
**Reference:** RESEARCH_RULES.md "In-Sample vs Out-of-Sample"
**Kill:** WFE < 50% (Sharpe decay > 50% from IS to OOS) → likely overfit.
**2026 holdout is SACRED:** 3 pre-registered MNQ strategies only. See `docs/pre-registrations/`.

### Gate 5: Adversarial Audit
**Question:** What could make this result fake?
**Tests (pick applicable):**
- Bootstrap permutation (mandatory for ML, recommended for novel findings)
- Sensitivity analysis: change param ±20%, does result survive?
- Fresh agent audit: new context, no prior bias
**Kill:** Bootstrap p > 0.05, or ±20% param change kills the finding → artifact.

### Gate 6: Replay Validation
**Question:** Does the ExecutionEngine match the statistical expectations?
**Test:** `paper_trader.py` replay on historical data, compare to orb_outcomes stats.
**Kill:** Material discrepancy (>10% trade count difference, opposite sign on PnL) → execution bug.

### Gate 7: Paper Trade with Kill Criteria
**Question:** Does it work forward?
**Pre-register:** Kill criteria BEFORE starting. Lock in a doc with git commit.
**Reference:** `docs/pre-registrations/` for existing kill criteria.
**Kill criteria template:**
- After N trades per session: ExpR < threshold → STOP that session
- After N trades: slippage > threshold → STOP (cost model wrong)
- After N total: portfolio ExpR < threshold → STOP everything

---

## 4. Variable Space

Every tunable parameter, its values, and canonical source. QUERY these, never cite from memory.

### Instruments
| Instrument | Status | Unfiltered E2 O5 RR1.0 | Source |
|-----------|--------|------------------------|--------|
| MNQ | **ACTIVE** | +0.085R (POSITIVE) | `ACTIVE_ORB_INSTRUMENTS` |
| MGC | ACTIVE (filter-dependent) | −0.157R unfiltered, positive with G5+ on select sessions | `ACTIVE_ORB_INSTRUMENTS` |
| MES | ACTIVE (filter-dependent) | −0.061R unfiltered, positive with G4+ on select sessions | `ACTIVE_ORB_INSTRUMENTS` |
| M2K | DEAD | 0/18 families survive noise | Removed 2026-03-18. **Trap:** `orb_active=True` in ASSET_CONFIGS but excluded by `DEAD_ORB_INSTRUMENTS`. Always use `ACTIVE_ORB_INSTRUMENTS`, never `orb_active` directly. |
| MCL, SIL, M6E, MBT | DEAD | 0 validated | `ASSET_CONFIGS` (present, not active) |

→ `from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS`

### Entry Models
| Model | Status | Mechanism | Notes |
|-------|--------|-----------|-------|
| E2 | **PRIMARY** | Stop-market at ORB + slippage | Honest, includes fakeouts |
| E1 | Active (conservative) | Market on next bar, ~1.18x risk | No backtest bias |
| E3 | RETIRED | Limit retrace at ORB edge | Adverse selection. Per-combo only. In `SKIP_ENTRY_MODELS` — skipped at runtime. |
| E0 | **PURGED** | 3 optimistic biases | Never use. |

→ `from trading_app.config import SKIP_ENTRY_MODELS`  # frozenset({"E3"}) — excluded from outcome_builder and grid search

→ `from trading_app.config import ENTRY_MODELS`

### RR Targets (MNQ E2 O5 NO_FILTER, verified 2026-03-21)
| RR | ExpR | WR | Status |
|----|------|-----|--------|
| 1.0 | +0.085 | 54.8% | Strongest unfiltered baseline |
| 1.5 | +0.083 | 42.5% | Strong |
| 2.0 | +0.077 | 34.4% | ML signal at O30 (AUC 0.658, bootstrap p=0.005) |
| 2.5 | +0.053 | 28.3% | Weakening |
| 3.0 | +0.033 | 23.9% | Marginal |
| 4.0 | −0.010 | 17.9% | DEAD |

**MGC/MES:** All RR targets negative without size filters. With G5+ filter, MGC CME_REOPEN becomes positive (see TRADING_RULES.md Session Playbook). Size filter is mandatory for MGC/MES.

### Confirm Bars
| Value | Notes |
|-------|-------|
| CB1 | Default for E2 (no confirmation needed — stop-market triggers on touch) |
| CB2 | Optimal for E1 on G5+ days per TRADING_RULES.md |
| CB3-CB5 | Identical to CB1-CB2 for E3 (same limit order) |

### Direction
| Value | When |
|-------|------|
| BOTH | Default. Most sessions. |
| LONG-ONLY | TOKYO_OPEN (shorts negative across all instruments — TRADING_RULES.md) |
| DIR_LONG / DIR_SHORT | Available as discovery grid filters via `config.ALL_FILTERS` |

### Apertures (MNQ E2 RR1.0 unfiltered)
| Aperture | ExpR | Notes |
|----------|------|-------|
| O5 | +0.085 | Primary. Highest per-trade edge. |
| O15 | +0.075 | Weaker. TOKYO_OPEN is exception (O15 > O5). |
| O30 | +0.065 | Weakest baseline. But ML discriminates here at RR2.0. |

### Sessions (MNQ E2 O5 RR1.0 NO_FILTER, verified 2026-03-21)
| Session | ExpR | N | Notes |
|---------|------|---|-------|
| CME_PRECLOSE | +0.202 | 1,254 | Strongest. Pre-registered for 2026. |
| US_DATA_1000 | +0.140 | 1,311 | Strong |
| COMEX_SETTLE | +0.121 | 1,266 | Pre-registered for 2026 |
| NYSE_OPEN | +0.117 | 1,311 | Pre-registered for 2026 |
| EUROPE_FLOW | +0.097 | 1,313 | |
| NYSE_CLOSE | +0.073 | 1,102 | Low WR (30.8%). Excluded from raw baseline. |
| TOKYO_OPEN | +0.064 | 1,314 | LONG-ONLY per TRADING_RULES.md |
| CME_REOPEN | +0.059 | 1,169 | Friday skip recommended |
| US_DATA_830 | +0.055 | 1,310 | |
| LONDON_METALS | +0.053 | 1,313 | |
| SINGAPORE_OPEN | +0.029 | 1,314 | MGC excluded (74% double-break) |
| BRISBANE_1025 | +0.011 | 1,314 | Weakest |

→ `from pipeline.dst import SESSION_CATALOG`

### Filters
| Filter | What it does | Source |
|--------|-------------|--------|
| NO_FILTER | Take all trades (pass-through) | `config.ALL_FILTERS["NO_FILTER"]` |
| ORB_G4 through ORB_G8 | ORB size >= N points | `config.ALL_FILTERS` |
| ATR70_VOL | ATR percentile >= 70 | `config.ALL_FILTERS["ATR70_VOL"]` |
| DIR_LONG / DIR_SHORT | Direction filter | `config.ALL_FILTERS` |
| VOL_RV12_N20 | Relative volume >= 1.2× median | `config.ALL_FILTERS` |
| X_MES_ATR60 / X_MES_ATR70 | Cross-asset MES ATR filter | `config.ALL_FILTERS` |
| X_MGC_ATR60 / X_MGC_ATR70 | Cross-asset MGC ATR filter | `config.ALL_FILTERS` |
| Composites (ORB_G4_NODBL etc) | Size + no-double-break | `config.ALL_FILTERS` |

**CRITICAL:** `filter_type` must EXACTLY match a key in `ALL_FILTERS`. Unknown strings cause silent trade drops (fail-closed). Use `"NO_FILTER"`, never `"NONE"` or `"BASE"`.

→ `from trading_app.config import ALL_FILTERS`

### Cost Models (verified 2026-03-21)
| Instrument | Point Value | Total Friction | Tick Size |
|-----------|-------------|---------------|-----------|
| MNQ | $2.00 | $2.74 | 0.25 |
| MGC | $10.00 | $5.74 | 0.10 |
| MES | $5.00 | $3.74 | 0.25 |

→ `from pipeline.cost_model import COST_SPECS`

---

## 5. NO-GO Registry

Everything confirmed dead. Do NOT re-test without a fundamentally new approach.

| Path | Verdict | Evidence | What Would Reopen |
|------|---------|----------|-------------------|
| MGC/MES unfiltered ORB | DEAD | All negative. 0 survivors. | New instrument economics (lower friction) |
| E0 entry model | PURGED | 3 compounding biases | Nothing — structurally flawed |
| E3 entry model | RETIRED | Adverse selection. 19/20 both negative. | At-break architecture (not pre-break) |
| RR4.0 (any instrument) | DEAD | Negative even on MNQ E2 | Nothing at current cost structure |
| ML on PORTFOLIO-LEVEL negative baselines | DEAD | Threshold artifact, bootstrap p=0.35 | Nothing — mathematical trap |
| ML on PER-SESSION negative baselines | **ALIVE** | Bootstrap p=0.005 on NYSE_OPEN O30 RR2.0 | Requires bootstrap verification (mandatory) |
| Calendar blanket skip | DEAD | Mixed results (some days BETTER) | Per-combo only, never blanket |
| Non-ORB strategies | DEAD | 6 archetypes, 540 tests, 0 survivors | Fundamentally different market model |
| MCL, SIL, M6E, MBT, M2K ORB | DEAD | 0 validated per instrument | New data source or contract change |
| Quintile conviction | DEAD | Vol filter artifact on best decade | — |
| RSI, MACD, Bollinger, MA cross | GUILTY | RESEARCH_RULES.md: guilty until proven | Sensitivity + OOS + mechanism required |
| Pyramiding | DEAD | Correlated intraday = tail risk | — |
| Breakeven trail stops | DEAD | −0.12 to −0.17 Sharpe | — |

Full NO-GO table with details: TRADING_RULES.md "What Doesn't Work"

---

## 6. ML Sub-Pipeline

Separate decision tree for ML meta-labeling. ML is OPTIONAL — raw baselines are tradeable without it.

### When to use ML
- Baseline exists (positive or negative) and you want per-trade selection
- Feature signal exists (univariate quartile test shows discrimination)
- **ML CAN work on negative-baseline sessions** — verified 2026-03-21 (NYSE_OPEN O30 RR2.0: raw baseline −0.136R, ML delta +30.5R, bootstrap p=0.005). The key: bootstrap MUST verify it's not the threshold artifact.
- ML is NOT a replacement for having SOME positive population in the variable space. If the entire instrument × entry model space is negative at every point, ML can't help.

### ML Test Sequence
```
1. UNIVARIATE SIGNAL → quartile test per feature × session × RR
   Kill: no feature shows meaningful spread → ML can't help
2. TRAIN → 3-way split (60/20/20), per-session RF, E6 noise filter
   Gates: CPCV AUC ≥ 0.50, Test AUC ≥ 0.52, delta ≥ 0, skip ≤ 85%
3. BOOTSTRAP → 200 permutations, shuffle labels, rerun pipeline
   Kill: p > 0.05 → threshold artifact, not skill
4. REPLAY → paper_trader with --use-ml vs without
   Kill: ML-filtered worse than raw baseline → configuration error
5. PAPER TRADE → forward test with kill criteria
```

### ML Variable Coverage Checklist
Before declaring ML dead for an instrument, verify ALL cells tested:
- [ ] RR1.0 flat
- [ ] RR1.5 flat
- [ ] RR2.0 flat
- [ ] RR2.0 per-aperture (O5, O15, O30 separately)
- [ ] Bootstrap on all survivors

### ML Features
→ `from trading_app.ml.config import GLOBAL_FEATURES, SESSION_FEATURE_SUFFIXES, LOOKAHEAD_BLACKLIST`

5 global + 4 session + 4 categorical + 3 cross-session + 6 level proximity = ~22 base features. One-hot encoding expands categoricals. E6 noise filter drops ~20-25 columns (4 prefix patterns × multiple one-hots + 3 exact).

Lookahead blacklist: 35 items. Enforced in `ml/config.py`. Session guard in `pipeline/session_guard.py`.

### Current ML State (VOLATILE — query, don't cite)
Model on disk: `models/ml/meta_label_MNQ_hybrid.joblib`
To inspect: `python -c "import joblib; b=joblib.load('models/ml/...'); print(b.keys())"`

---

## 7. Paper Trading Checklist

Before deploying paper trading:

- [ ] Pre-registration doc exists in `docs/pre-registrations/` (git-committed BEFORE testing)
- [ ] Kill criteria defined (per-session ExpR threshold, slippage threshold, portfolio threshold)
- [ ] Portfolio built from canonical source (not hardcoded)
- [ ] `filter_type = "NO_FILTER"` verified (ALL_FILTERS.get() fail-closes on unknown keys)
- [ ] Replay validation matches expectations (trade count, WR, PnL within 10%)
- [ ] Cost model verified: `from pipeline.cost_model import get_cost_spec`
- [ ] 2026 holdout is SACRED — no "quick checks" on 2026 data

### Raw Baseline Paper Trading
```bash
python -m trading_app.paper_trader --instrument MNQ --raw-baseline --rr-target 1.0 \
  --start 2025-01-01 --end 2025-12-31 --quiet
```

### ML-Filtered Paper Trading
```bash
python -m trading_app.paper_trader --instrument MNQ --raw-baseline --rr-target 2.0 \
  --orb-minutes 30 --use-ml --start 2025-01-01 --end 2025-12-31 --quiet
```

### Live Signal-Only
```bash
PYTHONPATH=. python scripts/run_live_session.py --instrument MNQ --signal-only --raw-baseline
```

---

## 8. Pipeline Order

```
Databento .dbn.zst
  → ingest_dbn.py       → bars_1m (14.9M rows)
  → build_bars_5m.py    → bars_5m (3.1M rows)
  → build_daily_features.py → daily_features (34K rows)
  → outcome_builder.py  → orb_outcomes (8.4M rows)
  → strategy_discovery.py → experimental_strategies
  → strategy_validator.py → validated_setups (11 rows, all MNQ)
  → build_edge_families.py → edge_families (0 rows — needs rebuild)
  → [optional] ML meta_label → models/ml/*.joblib
  → portfolio builder    → Portfolio object
  → paper_trader.py      → trade journal
```

Row counts verified 2026-03-21. Commands: `docs/ARCHITECTURE.md`.

**Rebuild dependencies:** Outcomes depend on daily_features. Discovery depends on outcomes. Validation depends on discovery. Edge families depend on validation. ML depends on outcomes + daily_features. Pipeline is ONE-WAY (pipeline/ → trading_app/, never reversed).

**When adding a session:** init_db.py → rebuild daily_features → rebuild outcomes → rebuild discovery → rebuild validation → rebuild edge families. See hard_lessons.md #15.

---

## 9. Active Research Threads

**⚠ VOLATILE — update every session. Query for current state.**

| Thread | Stage | Next Step | Blocking? |
|--------|-------|-----------|-----------|
| MNQ RR1.0 raw baseline | Gate 7 (paper trade) | Deploy signal-only. Kill criteria in pre-reg doc. | No |
| MNQ RR2.0 O30 ML | Gate 6 (replay done) | Multi-RR portfolio design, then paper trade | No |
| 2026 holdout test | Gate 4 (waiting) | April 2026, N≥100 per session | Time-gated |
| Simple regime filter (ATR>50pct) | Gate 2 (untested) | Run quartile comparison vs ML | Deferred |
| Edge families rebuild | Infrastructure | Run build_edge_families.py | Needed for fitness tracking |

---

## 10. Failure Patterns — What Catches Us

| Pattern | Example | Prevention |
|---------|---------|------------|
| **Incomplete variable search** | ML tested at RR1.0 only → missed RR2.0 signal | Gate 2: test ≥3 values per dimension |
| **Missing adversarial gate** | ML +2,366R → p=0.35 artifact | Gate 5: bootstrap mandatory for ML |
| **Stale information** | VWAP "57%" → actually 99% | Always query canonical sources |
| **Blanket rules** | SINGAPORE_OPEN off → only MGC should be off | Per instrument × session, never blanket |
| **No plan-first** | 30+ sessions jumping to code | Hard lesson #14: plan is first output |
| **Metadata trusted** | Strategy_fitness table doesn't exist | Hard lesson #7: verify schema before SQL |
