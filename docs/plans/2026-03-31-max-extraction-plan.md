# Max Extraction Plan — Full Automation Scaling (2026-03-31)

## Grounding: What the Data Says

**Strategy universe:** 788 active validated (747 CORE + 41 REGIME) across 12 sessions, 3 instruments.
**Best-per-session×instrument:** 22 non-overlapping lanes at N≥100, total $146.5/day EV at 1 contract.
**Forward data:** STALE (outcomes end Mar 23). All 2025 forward results = 0. Must rebuild before trusting forward metrics.

### The 22-Lane Universe (sorted by ExpR)

| # | Session | Inst | Filter | RR | SM | N | ExpR | ShANN | Risk/ct |
|---|---------|------|--------|---:|----:|---:|-----:|------:|--------:|
| 1 | CME_PRECLOSE | MNQ | VOL_RV20_N20 | 1.0 | S1.0 | 293 | 0.3397 | 2.41 | $22 |
| 2 | NYSE_CLOSE | MNQ | VOL_RV25_N20 | 1.0 | S1.0 | 227 | 0.3254 | 1.98 | $26 |
| 3 | EUROPE_FLOW | MGC | ORB_G4 | 2.0 | S1.0 | 111 | 0.2971 | 0.73 | $40 |
| 4 | TOKYO_OPEN | MGC | ORB_G4_CONT | 2.0 | S0.75 | 125 | 0.2832 | 1.03 | $42 |
| 5 | SINGAPORE_OPEN | MNQ | ORB_VOL_2K | 1.5 | S1.0 | 194 | 0.2764 | 1.19 | $36 |
| 6 | COMEX_SETTLE | MNQ | ORB_VOL_8K | 1.5 | S1.0 | 310 | 0.2640 | 1.34 | $46 |
| 7 | CME_PRECLOSE | MES | VOL_RV20_N20 | 1.0 | S0.75 | 224 | 0.2444 | 1.94 | $23 |
| 8 | TOKYO_OPEN | MNQ | VOL_RV30_N20 | 3.0 | S0.75 | 558 | 0.2359 | 1.22 | $16 |
| 9 | EUROPE_FLOW | MNQ | OVNRNG_100 | 2.0 | S1.0 | 552 | 0.2165 | 1.19 | $30 |
| 10 | LONDON_METALS | MES | VOL_RV25_N20 | 3.0 | S0.75 | 318 | 0.2146 | 1.01 | $18 |
| 11 | LONDON_METALS | MNQ | ATR70_VOL | 1.5 | S0.75 | 558 | 0.1778 | 1.35 | $25 |
| 12 | NYSE_CLOSE | MES | COST_LT10 | 1.0 | S1.0 | 134 | 0.1773 | 0.90 | $34 |
| 13 | NYSE_OPEN | MES | OVNRNG_25 | 2.0 | S0.75 | 465 | 0.1687 | 1.15 | $47 |
| 14 | NYSE_OPEN | MNQ | ATR70_VOL | 1.0 | S1.0 | 486 | 0.1675 | 1.28 | $69 |
| 15 | US_DATA_1000 | MNQ | X_MES_ATR70 | 1.0 | S0.75 | 602 | 0.1500 | 1.73 | $59 |
| 16 | US_DATA_1000 | MGC | ORB_G6 | 1.0 | S1.0 | 274 | 0.1317 | 0.76 | $59 |
| 17 | CME_REOPEN | MNQ | VOL_RV30_N20 | 1.0 | S1.0 | 407 | 0.1299 | 0.96 | $23 |
| 18 | COMEX_SETTLE | MES | ORB_G8 | 1.0 | S0.75 | 215 | 0.1284 | 0.90 | $39 |
| 19 | BRISBANE_1025 | MNQ | COST_LT10 | 1.5 | S0.75 | 378 | 0.1276 | 0.80 | $26 |
| 20 | US_DATA_1000 | MES | VOL_RV15_N20 | 1.0 | S0.75 | 731 | 0.0760 | 1.00 | $33 |
| 21 | US_DATA_830 | MNQ | COST_LT12 | 1.0 | S1.0 | 1346 | 0.0757 | 0.95 | $34 |
| 22 | US_DATA_830 | MES | VOL_RV20_N20 | 1.0 | S0.75 | 576 | 0.0727 | 0.89 | $26 |

### Conflict-Free Profiles

**TYPE-A** (CME_PRECLOSE fork): 8 sessions, 16 lanes (with MES/MGC stacking)
- US_DATA_1000 → COMEX_SETTLE → CME_PRECLOSE → CME_REOPEN → TOKYO_OPEN → LONDON_METALS → US_DATA_830 → NYSE_OPEN
- EV: $100.3/day at 1ct | Worst-day risk: $581/ct

**TYPE-B** (NYSE_CLOSE fork): 8 sessions, 15 lanes (with MES/MGC stacking)
- US_DATA_1000 → COMEX_SETTLE → NYSE_CLOSE → CME_REOPEN → SINGAPORE_OPEN → EUROPE_FLOW → US_DATA_830 → NYSE_OPEN
- EV: $106.0/day at 1ct | Worst-day risk: $601/ct

**Shared core** (both profiles): US_DATA_1000, COMEX_SETTLE, CME_REOPEN, US_DATA_830, NYSE_OPEN
**Fork difference:** TYPE-A gets CME_PRECLOSE + TOKYO_OPEN + LONDON_METALS. TYPE-B gets NYSE_CLOSE + SINGAPORE_OPEN + EUROPE_FLOW.

---

## Current State vs Target

### What's Deployed (NOW)

| Profile | Firm | Lanes | EV/day | DD Used | Status |
|---------|------|------:|-------:|--------:|--------|
| apex_100k_manual | Apex | 5 MNQ | ~$30 | 25% | ACTIVE, manual |
| topstep_50k | TopStep | 1 MGC | ~$12 | 10% | ACTIVE, 5 copies |
| topstep_50k_mnq_auto | TopStep | 1 MNQ | ~$12 | 6% | ACTIVE, 1 account |
| **TOTAL** | | **7** | **~$54** | | |

### What's Validated (TARGET)

| Config | Firm | Lanes | Cts | EV/day | Annual (50% shrink) |
|--------|------|------:|----:|-------:|--------------------:|
| 5x TopStep 150K TYPE-A | TopStep | 16 | 7 | $702 | $325,993 |
| 5x Tradeify 150K TYPE-B | Tradeify | 15 | 7 | $742 | $345,169 |
| **COMBINED** | | **31** | **7** | **$1,444** | **$671,162/yr** |

Conservative/moderate alternatives:

| Scenario | Annual (50% shrinkage) | DD Risk |
|----------|----------------------:|---------|
| TS+TF 10x 50K, 3ct | $287,641 | YOLO (89%) |
| TS+TF 10x 100K, 5ct | $479,401 | YOLO (99%) |
| **TS+TF 10x 150K, 7ct** | **$671,162** | **AGGRO-YOLO (78-92%)** |
| TS+TF 10x 150K, 10ct | $958,802 | YOLO (131%) |

Plus Apex manual on top: ~$8,700/yr at 1ct (daytime sessions).

---

## Firm Constraints (Hard Rules)

| Firm | Auto | Max Accounts | DD Tiers | Split | Blocked |
|------|------|-------------|----------|-------|---------|
| **TopStep** | FULL (ProjectX) | 5 Express + 1 Live | $2K/$3K/$4.5K | 50%→90% | DLL exists |
| **Tradeify** | FULL (Tradovate) | 5 | $2K/$4K/$6K | 90% flat | No DLL, 10s min hold |
| **Apex** | NONE (manual) | Unlimited | $2K/$3K/$4K | 100% | Metals banned, no auto |

**Key:** Tradeify has NO daily loss limit and HIGHER DD at 150K ($6K vs $4.5K). But Tradovate auth is currently broken.

### DD Budget Reality (worst-case all-lose day)

| Account | 3ct | 5ct | 7ct | 10ct |
|---------|-----|-----|-----|------|
| 50K ($2K DD) | 89% YOLO | 148% YOLO | 207% YOLO | 296% YOLO |
| 100K ($3K DD) | 59% AGGRO | 99% YOLO | 138% YOLO | 197% YOLO |
| 150K TS ($4.5K DD) | 39% SAFE | 66% AGGRO | 92% YOLO | 131% YOLO |
| 150K TF ($6K DD) | 30% SAFE | 50% SAFE | 70% AGGRO | 100% YOLO |

**Tradeify 150K at 7ct is 70% AGGRO** — the sweet spot for riskier-but-not-suicidal.
**TopStep 150K at 5ct is 66% AGGRO** — safer per-account, scales with copies.

Reset costs: 50K=$50, 100K=$100, 150K=$150. At 5 accounts = $250/$500/$750 per full wipe.
Against $671K annual upside, even monthly full wipes ($9K/yr) are noise.

---

## The Plan

### Phase 0: Unblock Data (NOW — before anything else)

**Blocker:** Outcomes stale at Mar 23. Forward data = 0. Can't trust any forward metrics.

1. Refresh MES bars: `python pipeline/ingest_dbn.py --instrument MES --resume`
2. Rebuild daily features (all 3): `python pipeline/build_daily_features.py --instrument {MGC,MNQ,MES}`
3. Rebuild outcomes (all 3): `python -m trading_app.outcome_builder --instrument {MGC,MNQ,MES}`
4. Fix drift: `python scripts/tools/select_family_rr.py` (18 missing family_rr_locks)
5. Verify: `python pipeline/check_drift.py`

**Gate:** Outcomes through Mar 28+ for all 3 instruments. Drift = 0 failures.

### Phase 1: Scale TopStep via ProjectX (Week 1)

**What:** Expand from 1 MNQ auto lane to full TYPE-A profile across 5 Express accounts.
**Why TopStep first:** ProjectX API is WORKING. Already proven with COMEX_SETTLE.

1. Create `topstep_150k_auto` profile with full TYPE-A lane set (16 lanes)
2. Upgrade 5 Express accounts to 150K tier ($4.5K DD, 150 micro max)
3. Run signal-only for 3 sessions to verify all 16 lanes fire correctly
4. Run demo for 1 full trading day
5. Go live at 2ct (39% DD = SAFE)
6. Scale to 5ct after first profitable week (66% DD = AGGRO)
7. Scale to 7ct after first profitable month (92% DD = YOLO)

**Infrastructure needed:**
- Multi-session orchestrator (currently 1 session per run)
- Copy-to-5-Express pipeline (ProjectX copy trading)
- DD circuit breaker per-account (exists, needs profile wiring)

**EV at target (5x 150K, 7ct):** $325,993/yr realistic

### Phase 2: Fix Tradovate + Activate Tradeify (Week 2-3)

**Blocker:** Tradovate auth broken (password rejected). Must fix before Tradeify goes live.

1. Debug Tradovate auth — likely password/API key rotation needed
2. Create `tradeify_150k_auto` profile with full TYPE-B lane set (15 lanes)
3. Open 5x Tradeify 150K accounts ($6K DD each — best DD/cost ratio)
4. Run signal-only → demo → live ramp (same as Phase 1)
5. Start at 3ct (30% DD = SAFE on Tradeify's $6K)
6. Scale to 5ct (50% = still SAFE on Tradeify)
7. Scale to 7ct (70% = AGGRO — the sweet spot)

**Why TYPE-B for Tradeify:** Filter diversity. TYPE-A and TYPE-B share 5 core sessions but diverge on the other 3. Running different profiles on different firms = more independent bets. If one profile's unique sessions have a bad week, the other may not.

**EV at target (5x 150K, 7ct):** $345,169/yr realistic

### Phase 3: Multi-Instrument Stacking (Week 3-4)

**What:** Add MES and MGC lanes on top of MNQ in the same session windows.
**Why now:** Same session time = no extra monitoring. Just more contracts in parallel.

Both TYPE-A and TYPE-B profiles already include MES/MGC stacking in the lane tables above. This is about actually wiring the multi-instrument runner to handle 2-3 instruments per session.

**Infrastructure needed:**
- `MultiInstrumentRunner` already exists — verify it handles per-instrument lane specs
- Per-instrument ORB cap enforcement
- Per-instrument DD tracking (MNQ $2/pt vs MGC $10/pt vs MES $5/pt)

### Phase 4: Contract Ramp to Max (Month 2-3)

Once both firms are live and profitable:

| Milestone | TopStep 5x | Tradeify 5x | Combined |
|-----------|-----------|-------------|----------|
| Week 1 (proof) | 2ct, $93K/yr | — | $93K |
| Week 3 (AGGRO) | 5ct, $233K/yr | 3ct, $148K/yr | $381K |
| Month 2 (target) | 7ct, $326K/yr | 7ct, $345K/yr | **$671K** |
| Month 3 (max) | 10ct, $466K/yr | 10ct, $493K/yr | **$959K** |

**Reset budget:** At 7ct AGGRO, expect ~1 account blow per month across 10 accounts. Reset cost: $150/account = $1,800/yr. Against $671K annual = 0.3% drag.

### Phase 5: Apex Manual + MFFU Expansion (Month 3+)

- Apex stays MANUAL — 100% split, 5 daytime sessions. ~$8,700/yr at 1ct. Worth keeping for zero-split profit.
- MFFU as third auto firm if TopStep/Tradeify scaling proves out. News restriction adds complexity.
- Self-funded IBKR is the endgame ($100K+ capital, zero split, zero DD games). Build when proven.

---

## Blockers & Risks

| Blocker | Severity | Fix |
|---------|----------|-----|
| Outcomes 8 days stale | CRITICAL | Phase 0 data rebuild |
| MES bars 7 days stale | CRITICAL | Databento resume ingest |
| Tradovate auth broken | HIGH | Debug password/key (blocks Tradeify) |
| family_rr_locks drift | MEDIUM | Run select_family_rr.py |
| Forward results = 0 | HIGH | Outcome rebuild populates 2025 data |
| Multi-session orchestrator | MEDIUM | Need to wire 8-session sequential run |
| Copy-to-5-Express | LOW | ProjectX copy trading exists |

| Risk | Impact | Mitigation |
|------|--------|------------|
| Worst-day all-lose at 7ct | $4,067 (91% of TopStep DD) | Accept resets. $150 reset vs $671K annual. |
| Correlated blow across 10 accounts | $1,500 in resets | TYPE-A/TYPE-B diversification. Different sessions = different market conditions. |
| Edge decay (strategies stop working) | Revenue drops | CUSUM drift monitoring + quarterly revalidation. 788 strategies = deep bench. |
| Prop firm rule changes | Lose a firm | 3-firm diversification (TS + TF + Apex). Self-funded as escape hatch. |
| Data staleness (forget to refresh) | Stale filters → missed/bad trades | Dashboard data-status endpoint + cron refresh |

---

## Success Metrics

| Metric | Month 1 | Month 3 | Month 6 |
|--------|---------|---------|---------|
| Active auto accounts | 6 (5 TS + 1 TF proof) | 10 (5+5) | 10+ |
| Contracts per lane | 2-3 | 5-7 | 7-10 |
| Gross monthly (realistic) | $8-15K | $30-55K | $55-80K |
| Reset cost/month | $0-300 | $0-600 | $0-600 |
| Firms active | 2 (TS + Apex) | 3 (+ Tradeify) | 3+ |

---

## What Needs Building

| Item | Effort | Blocks |
|------|--------|--------|
| Full TYPE-A/TYPE-B profile definitions | Small | Phase 1 |
| Multi-session sequential orchestrator | Medium | Phase 1 (run all 8 sessions in one night) |
| 150K account tier upgrade (TopStep) | Admin task | Phase 1 |
| Tradovate auth fix | Unknown | Phase 2 |
| Tradeify 150K account opening | Admin task | Phase 2 |
| Multi-instrument lane wiring | Small | Phase 3 |
| CUSUM drift alerting | Medium | Ongoing monitoring |
| Cron data refresh | Small | Operational |
| Reset tracking dashboard | Small | Operational |

---

## The Numbers (Honest Accounting)

All annual figures use **50% shrinkage** (half of backtest EV realized in live trading).
Worst-case DD uses **all lanes losing simultaneously** (conservative — correlation < 0.1 between sessions).

| What | Number | Source |
|------|-------:|--------|
| Backtest daily EV (1ct, all 22 lanes) | $146.50 | validated_setups, best per session×instrument |
| Realistic daily EV (50% shrinkage) | $73.25 | Standard shrinkage assumption |
| Annual realistic at 1ct, 1 account | $18,313 | $73.25 × 250 days |
| Annual realistic at 7ct, 10 accounts | **$671,162** | Weighted TYPE-A/B × firm splits |
| Worst-day risk per account (7ct) | $4,067 | ~$581 × 7ct |
| Annual reset budget (1 blow/month) | $1,800 | $150 × 12 |
| Net after resets | **$669,362** | Resets are noise |

**Caveat:** These numbers assume the edge persists. Forward validation is the binding test. Phase 0 data rebuild is non-negotiable before any scaling.
