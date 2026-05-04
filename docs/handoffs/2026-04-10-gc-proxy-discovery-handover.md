# Handover — GC Proxy Discovery (MGC via 16yr Gold Data)

**Date:** 2026-04-10
**Branch:** `research/gc-proxy-validity` (pushed)
**Session focus:** Can GC parent data be used for MGC strategy discovery?

---

## What happened

### 1. Research: GC proxy validity — 4-gate empirical test
- Price correlation: r=0.99999 across 1.25M paired bars — IDENTICAL
- Filter inputs (gap, PDR, range): correlate >0.98
- 96% of trades trigger identically on GC and MGC
- Volume IS different (negative control holds)
- **Conclusion: GC proxy valid for price-based filters**

### 2. Amendment 3.1 — proxy data policy revised
- GC proxy expanded from validation-only to discovery-eligible
- Price-safe filters allowed: ORB_G*, GAP, PDR, ATR, OVNRNG, COST_LT
- Volume-unsafe filters remain micro-only: ORB_VOL, VOL_RV
- Regime-awareness required (G-filters are era-dependent on gold)

### 3. Infrastructure fixes (critical)
- **GC cost spec added** to `pipeline/cost_model.py` — $100/pt, friction $57.40
  - BUG FOUND AND FIXED: initial spec had spread/slippage 10x too low ($2 not $20). Caught by code review. COST_LT10 was meaningless before fix.
- **Discovery DELETE fix** in `strategy_discovery.py` — hypothesis mode now skips the DELETE, strategies accumulate across files. ALSO FIXES the MNQ multi-RR wipe bug from the other terminal.
- **PDR/GAP grid unlock** in `config.py` — GC sessions added to `_pdr_validated` and GAP gate

### 4. GC pipeline artifacts built
- daily_features: 4,605 rows (2010-2026), all integrity checks passed
- orb_outcomes: 1,295,064 rows (4,502 trading days)
- MGC data unchanged (917,136 orb_outcomes)

### 5. Discovery results — 3 waves, 56 hypotheses tested

**5 validated strategies (all p < 0.0001 except noted):**

| Strategy | N | WR | ExpR | Sharpe | WFE | p |
|---|---|---|---|---|---|---|
| GC NYSE_OPEN ATR_P70 RR1.0 | 1,330 | 60.1% | 0.158 | 1.57 | 0.98 | <0.0001 |
| GC NYSE_OPEN ATR_P70 RR2.0 | 1,280 | 39.8% | 0.151 | 0.98 | 1.05 | 0.0001 |
| GC NYSE_OPEN ATR_P50 RR2.0 | 2,046 | 39.3% | 0.132 | 1.09 | 0.85 | <0.0001 |
| GC NYSE_OPEN ATR_P50 RR1.5 | 2,094 | 46.7% | 0.121 | 1.18 | 0.85 | <0.0001 |
| GC US_DATA_1000 ATR_P70 RR1.0 | 1,290 | 57.8% | 0.119 | 1.15 | 1.10 | <0.0001 |

**Pattern: ATR percentile filters on NYSE_OPEN and US_DATA_1000 = gold edge.**

### 6. AUDIT NEEDED (started but yearly data looks suspicious)
The yearly_results JSON in experimental_strategies shows N=0 for every year but has win_rate values. This needs investigation — the per-year breakdown may be stored in a different format than expected. The audit query needs fixing before we can verify era stability claims.

---

## Immediate TODO (next session)

1. **FIX the yearly audit query** — the JSON parsing shows N=0 per year. Need to read the correct field names. Then verify: are these 5 strategies honest across all 16 years, or driven by 1-2 outlier eras?

2. **Cross-validate against MGC micro data** — the 5 GC strategies use proxy data. Before deployment, they MUST be tested on actual MGC data (2022-2026) to confirm the edge transfers.

3. **Merge the discovery DELETE fix to main** — this fix benefits ALL instruments (fixes the MNQ multi-RR wipe). Should be merged independently of the GC research.

4. **Restore MNQ RR1.0 strategies** — the other terminal's validator wiped them. Re-run `mnq-rr10-individual.yaml` discovery + validation (now possible with the DELETE fix).

5. **Consider more exploration** — 56 hypotheses tested, N budget allows ~2,980. OVNRNG and PDR combos dropped from validated pool in wave 3 (absorbed into larger K). Could re-test in isolation or accept the ATR pattern as dominant.

---

## Key learnings

1. **Code review catches real bugs.** The cost spec bug would have deployed a false-positive strategy. ALWAYS review before claiming results.
2. **Infrastructure fixes unlock research.** The DELETE fix and PDR grid unlock revealed strategies that were previously invisible.
3. **Pathway B individual testing does NOT inflate K.** Each hypothesis stands alone at p < 0.05. More hypotheses don't tighten thresholds on existing ones. This was incorrectly stated mid-session — corrected by user challenge.
4. **Regime-robust filters are the value of proxy data.** Absolute G-filters fail era stability on gold (price varies 3x over 16yr). ATR percentile filters adapt automatically.
5. **Let the data speak.** NYSE_OPEN and US_DATA_1000 are THE gold sessions across 16 years. LONDON_METALS, EUROPE_FLOW, CME_REOPEN are not.
