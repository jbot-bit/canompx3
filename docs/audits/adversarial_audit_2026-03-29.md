# Adversarial Audit — 2026-03-29

Ground-up audit of canompx3. Every number from executed code against gold.db.
No assertions without evidence. Firm rules from playbook OFFICIAL RULE tags
(websites Cloudflare-blocked — flagged as UNVERIFIED_WEB for TopStep/Tradeify).

## DELIVERABLE 1 — CANONICAL FIRM RULES TABLE

| Firm | DD Type | Max DD | DLL | Max Micro | Max Mini | Close ET | Auto | Banned | Consistency | Min Hold |
|------|---------|--------|-----|-----------|----------|----------|------|--------|-------------|----------|
| Apex 50K | eod_trailing | $2,000 | None | 40 | 4 | 16:59 | none (PROHIBITED) | MGC,GC,SI,SIL,HG,PL,PA | 30% | None |
| TopStep 50K | eod_trailing | $2,000 | None | 50 | 5 | 16:00 | full | none | 40% | None |
| Tradeify 50K | eod_trailing | $2,000 | None | 40 | 4 | 16:00 | full | none | None | 10s |

### Apex EOD Mechanics (from playbook OFFICIAL RULE)
- Liquidation threshold starts at $48,000 ($2K below $50K start)
- Threshold trails up based on **highest EOD close** (4:59:59 PM ET), NOT intraday peaks
- Threshold **freezes at $50,100** when highest close reaches $52,100
- After freeze: $50,100 is a permanent floor (no more trailing)
- Source: playbook lines 444-479 citing official Apex support articles

### Codebase vs Firm Discrepancies

| Finding | Severity | Detail |
|---------|----------|--------|
| DD freeze not implemented | CRITICAL | `account_hwm_tracker.py` trails forever — no freeze at breakeven. Code will halt trading on a profitable account that the firm considers safe. |
| DD basis is intraday, not EOD | CRITICAL | `update_equity()` ratchets HWM on every poll. Firm only ratchets on EOD close. Code over-penalizes by counting intraday unrealized peaks. |
| Combined DD exceeds limit | CRITICAL | Profile self-documents: hist combined DD = -$3,409 (170% of $2K). SINGAPORE_OPEN alone: -$3,540. |
| TopStep/Tradeify rules unverified | WARNING | Cloudflare blocked all web access. DD/contract limits from playbook only. |

## DELIVERABLE 2 — ORB SIZE DISTRIBUTIONS (last 6 months, trade-eligible)

### Points Table

| Inst | Session | Om | Filter | N | Min | P5 | P25 | Med | P75 | P90 | P95 | Max | Mean | Std | Floor | Med/Fl |
|------|---------|---:|--------|--:|----:|---:|----:|----:|----:|----:|----:|----:|-----:|----:|------:|-------:|
| MNQ | NYSE_CLOSE | 15 | VOL_RV12_N20 | 2454 | 14.5 | 16.8 | 23.0 | 35.2 | 50.2 | 66.2 | 95.8 | 203.2 | 42.6 | 32.8 | 0 | inf |
| MNQ | SINGAPORE_OPEN | 15 | ORB_G8 | 4290 | 6.2 | 11.0 | 20.0 | 28.2 | 39.0 | 58.5 | 74.5 | 160.2 | 33.9 | 21.6 | 8 | 3.5x |
| MNQ | COMEX_SETTLE | 5 | ATR70_VOL | 3954 | 7.2 | 9.7 | 15.8 | 25.8 | 37.2 | 51.5 | 73.5 | 136.2 | 30.3 | 21.0 | 0 | inf |
| MNQ | NYSE_OPEN | 15 | X_MES_ATR60 | 4164 | 38.2 | 57.5 | 93.2 | 119.0 | 156.5 | 211.8 | 230.8 | 284.5 | 128.6 | 53.0 | 0 | inf |
| MNQ | US_DATA_1000 | 5 | X_MES_ATR60 | 4140 | 5.5 | 22.2 | 38.0 | 53.8 | 80.0 | 100.5 | 115.2 | 199.0 | 60.3 | 30.7 | 0 | inf |
| MGC | TOKYO_OPEN | 5 | ORB_G4_CONT | 4260 | 1.9 | 2.8 | 4.3 | 6.7 | 10.2 | 20.4 | 27.0 | 59.3 | 9.4 | 8.7 | 4 | 1.7x |

### Dollar Table (raw ORB * point value, per 1 contract)

| Inst | Session | PtVal | Med$ | P90$ | P95$ | Max$ |
|------|---------|------:|-----:|-----:|-----:|-----:|
| MNQ | NYSE_CLOSE | $2 | $70 | $132 | $192 | $406 |
| MNQ | SINGAPORE_OPEN | $2 | $56 | $117 | $149 | $320 |
| MNQ | COMEX_SETTLE | $2 | $52 | $103 | $147 | $272 |
| MNQ | NYSE_OPEN | $2 | $238 | $424 | $462 | $569 |
| MNQ | US_DATA_1000 | $2 | $108 | $201 | $230 | $398 |
| MGC | TOKYO_OPEN | $10 | $67 | $204 | $270 | $593 |

**Key finding:** NYSE_OPEN median ORB is $238 raw, $178 at 0.75x stop. This is the largest risk lane by far (8.9% of DD per trade at median, 15.9% at P90). It already has a 150pt cap in prop_profiles — the only lane with one.

## DELIVERABLE 3 — RISK RESTATEMENT (0.75x stop, $2K DD limit)

| Session | Inst | RR | MedStop$ | P90Stop$ | MedTgt$ | P90Tgt$ | Med%DD | P90%DD | Flag |
|---------|------|---:|--------:|---------:|--------:|--------:|-------:|-------:|------|
| NYSE_CLOSE | MNQ | 1.0 | $53 | $99 | $53 | $99 | 2.6% | 5.0% | |
| SINGAPORE_OPEN | MNQ | 4.0 | $42 | $88 | $170 | $351 | 2.1% | 4.4% | |
| COMEX_SETTLE | MNQ | 1.0 | $39 | $77 | $39 | $77 | 1.9% | 3.9% | |
| NYSE_OPEN | MNQ | 1.0 | $178 | $318 | $178 | $318 | 8.9% | 15.9% | |
| US_DATA_1000 | MNQ | 1.0 | $81 | $151 | $81 | $151 | 4.0% | 7.5% | |
| TOKYO_OPEN | MGC | 2.0 | $50 | $153 | $100 | $306 | 2.5% | 7.6% | |

### Max-Loss Day (all 5 Apex MNQ lanes stop at P90)

| Lane | P90 Stop$ |
|------|----------:|
| NYSE_CLOSE | $99 |
| SINGAPORE_OPEN | $88 |
| COMEX_SETTLE | $77 |
| NYSE_OPEN | $318 |
| US_DATA_1000 | $151 |
| **TOTAL** | **$733** |
| DD limit | $2,000 |
| Headroom | $1,267 (36.6% committed) |

Single-day all-stop is survivable at P90. But this does NOT account for multi-day losing streaks.

## DELIVERABLE 4 — APERTURE VERDICT TABLE

32 aperture groups with FDR-significant strategies. Key deployed + conditional:

| Inst | Session | Om | Ns | TotTrd | WR | ExpR | Sharpe | Verdict |
|------|---------|---:|---:|-------:|---:|-----:|-------:|---------|
| MNQ | NYSE_CLOSE | 15 | 5 | 1,880 | 63.2% | 0.1614 | 0.182 | DEPLOYED |
| MNQ | SINGAPORE_OPEN | 15 | 43 | 70,669 | 35.6% | 0.1132 | 0.084 | DEPLOYED |
| MNQ | COMEX_SETTLE | 5 | 52 | 65,059 | 44.4% | 0.1308 | 0.118 | DEPLOYED |
| MNQ | NYSE_OPEN | 15 | 9 | 16,497 | 51.1% | 0.0711 | 0.080 | DEPLOYED |
| MNQ | US_DATA_1000 | 5 | 15 | 24,945 | 46.4% | 0.0970 | 0.099 | DEPLOYED |
| MGC | TOKYO_OPEN | 5 | 3 | 375 | 51.2% | 0.2582 | 0.262 | DEPLOYED |
| MNQ | CME_PRECLOSE | 5 | 34 | 36,293 | 53.8% | 0.1798 | 0.185 | CONDITIONAL |
| MNQ | CME_PRECLOSE | 15 | 14 | 5,316 | 60.2% | 0.1836 | 0.214 | CONDITIONAL |
| MNQ | SINGAPORE_OPEN | 30 | 55 | 105,770 | 33.0% | 0.1011 | 0.073 | CONDITIONAL |

**Note:** MGC TOKYO_OPEN has only 375 total trades across 3 strategies. Per-strategy N is ~125 — above CORE threshold but ERA_DEPENDENT (77.6% from 2025 gold vol regime). Already flagged in profile notes.

## DELIVERABLE 5 — PROFILE TEST RESULTS

| Test | Profile | Result | Finding |
|------|---------|--------|---------|
| TEST 1: DD Arithmetic | apex_50k_manual | PASS | $733 max-day-loss < $2K (36.6%) |
| TEST 1: DD Arithmetic | topstep_50k | PASS | $153 < $2K (7.6%) |
| TEST 2: DD Budget Bypass | ALL | WARNING | No pre-trade DD budget check in prop_profiles.py. daily_lanes are static config — no validation that sum(worst_case_stops) < dd_limit. Enforcement only at runtime via account_hwm_tracker. |
| TEST 3: Concurrent Positions | apex_50k_manual | PASS | Sessions non-overlapping. Max concurrent = 1. |
| TEST 4: Contract Cap | ALL | PASS | 1 micro per trade << 40-50 firm cap |
| TEST 5: HWM Simulation | apex_50k_manual | PASS | Synthetic: 187 days, $10,668 P&L, max DD $1,204 (60% of $2K). NOT filtered by regime/filter gates — overstates trade count. |

### Test 5 Detail (per-strategy, 2025-07 to present)

| Lane | N | Sum(R) | Sum($) |
|------|--:|-------:|-------:|
| NYSE_CLOSE O15 | 124 | -1.2 | +$489 |
| SINGAPORE_OPEN O15 | 186 | +26.1 | +$241 |
| COMEX_SETTLE O5 | 179 | +12.4 | +$1,240 |
| NYSE_OPEN O15 | 186 | +30.1 | +$6,982 |
| US_DATA_1000 O5 | 187 | +16.0 | +$1,716 |
| **PORTFOLIO** | **187 days** | | **+$10,668** |

NYSE_OPEN dominates dollar P&L ($6,982 of $10,668 = 65%) because its ORBs are 3-4x larger than other lanes.

## DELIVERABLE 6 — PRIORITISED FIX LIST

### CRITICAL

**[CRITICAL-1] HWM tracker has no DD freeze logic**
- File: `trading_app/account_hwm_tracker.py`
- What: Add `freeze_threshold` parameter. When HWM reaches safety net balance ($52,100 for Apex 50K EOD), stop trailing. Threshold locks at $50,100 permanently.
- Without this: code over-penalizes profitable accounts, could cause user to abandon a live account the firm considers safe.

**[CRITICAL-2] HWM tracker uses intraday peaks, not EOD close**
- File: `trading_app/account_hwm_tracker.py::update_equity()`
- What: For `eod_trailing` firms, only update HWM at session close (`record_session_end()`), not on every equity poll. Currently `update_equity()` ratchets HWM on every call.
- Impact: Intraday unrealized P&L inflates code HWM above firm HWM. A day that closes flat still ratchets code DD.

**[CRITICAL-3] No pre-trade DD budget validation at config level**
- File: `trading_app/prop_profiles.py`
- What: Add import-time validation: sum P90-stop-losses across all `daily_lanes` and assert < `max_dd`. Fail-closed.
- Currently: profile notes self-document the breach ($3,409 > $2K) but nothing enforces it.

### WARNING

**[WARNING-1] TopStep/Tradeify firm rules not independently verified**
- File: `trading_app/prop_profiles.py` (`ACCOUNT_TIERS`)
- What: Manually verify DD limits, contract caps, DLL on firm websites. Cloudflare blocked automated access.

**[WARNING-2] No max_orb_size_pts cap on most lanes**
- File: `trading_app/prop_profiles.py` (`daily_lanes`)
- What: Only NYSE_OPEN has `max_orb_size_pts=150`. Other lanes uncapped.
- A 200pt MNQ ORB at 0.75x = $300 stop = 15% of DD in one trade.
- Recommend caps: NYSE_CLOSE 100pts, SINGAPORE_OPEN 80pts, COMEX_SETTLE 80pts, US_DATA_1000 120pts.

### INFO

**[INFO-1] SINGAPORE_OPEN RR4.0 documented risk**
- Already noted in `execution_notes` (0.5x sizing, hist DD -$3,540).
- Consider removing from Apex profile until DD budget validation passes.

**[INFO-2] Apex metals ban correctly encoded**
- MGC routed to TopStep only. Apex `banned_instruments` includes MGC, GC, SI, SIL, HG, PL, PA.

**[INFO-3] NYSE_OPEN dominates dollar P&L (65%)**
- Concentration risk. If this lane decays, portfolio economics change materially.

---

## Methodology Notes

- All ORB size data from `daily_features` joined to `orb_outcomes` on `(trading_day, symbol, orb_minutes)`.
- Date range: 2025-10-01 to present (last ~6 months).
- Point values: MNQ $2/pt, MGC $10/pt (from `pipeline.cost_model.COST_SPECS`).
- DD limits from `trading_app.prop_profiles.ACCOUNT_TIERS`.
- Validated strategies from `validated_setups` where `fdr_significant = TRUE`.
- Equity simulation uses all matching `orb_outcomes` rows for the specific strategy parameters — does NOT apply filter gating (VOL_RV12_N20, ORB_G8, etc.), so overstates trade count.
- Firm rules sourced from `docs/plans/manual-trading-playbook.md` OFFICIAL RULE tags. Web verification blocked by Cloudflare.
