# Phase 2.7 reanalysis + institutional-grade future-proofing

**Date:** 2026-04-19
**Prior docs (corrected/augmented here):**
- `docs/audit/results/2026-04-19-regime-break-2024-audit.md` (original Phase 2.7)
- `docs/audit/results/2026-04-19-phase-2-7-caveats-verification.md` (caveat audit)

**Origin:** User challenge "VERIFICATION IN FUCKING /RESOURCES PDFS LITERATURE OTHERWISE WE MAKING SHIT UP" — correct institutional-rigor push. Triggered re-extraction of `resources/` PDFs directly via pypdf and reframing of Phase 2.7 verdict.

## TL;DR reframing

The original Phase 2.7 result doc said "5 GOLD + 1 WATCH + 2 PURE_DRAG retire." After PDF-grounded re-verification + correlation-gate testing + year-by-year vol-regime analysis:

- **2 truly deploy-safe GOLD lanes** (down from 5): `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30`, `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15`
- **3 redundant-GOLD** (Chordia-PASS + regime-neutral BUT correlation-redundant with existing `COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5`): swap-candidates only, not parallel-deploy
- **1 WATCH** that per Chan-p120 MUST be deployed via **Carver vol-standardisation sizing**, not binary regime-gating
- **"2024 regime break" reframed as "high-vol-year regime effect"** — recurring (2020, 2022, 2024 all registered elevated ATR)
- **2 PURE_DRAG double-confirmed retires** — unchanged

## PDF-verified citations (this session, via pypdf direct extraction)

### Chan 2008 "Quantitative Trading" Ch 7 § Regime Switching — pp 119-121 (PDF pp 141-143)

Verbatim verified:

> "Among these, volatility regime switching seems to be most amenable to classical econometric tools such as the generalized autoregressive conditional heteroskedasticity (GARCH) model... **While such predictions of volatility regime switches can be of great value to options traders, they are unfortunately of no help to stock traders.**" — Chan book p120

**Omitted from the literature/ extract** until now: the bolded sentence. User's "are we stock or options?" challenge surfaced this omission. We are futures = directional = "stock traders" per Chan's framing.

> "Despite the elegant theoretical framework, such Markov regime-switching models are generally useless for actual trading purposes. The reason for this weakness is that they assume constant transition probabilities among regimes at all times." — Chan book p121

### Carver "Systematic Trading" Ch 2 (p40) § CONCEPT: VOLATILITY STANDARDISATION — PDF p57

Verbatim verified:

> "One of the most powerful techniques I use in my trading system framework is volatility standardisation. This is adjusting the returns of different assets so that they have the same expected risk... As you shall see in chapter nine, 'Volatility targeting', the different characteristics of positive and negative skew trading determine how much risk you should take."

**Implication for deploy doctrine:** the WATCH lane (`MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` — Phase 2.7 flagged 2024_CRITICAL) should be deployed via Carver-style vol-scaled sizing (scale position size inverse to current ATR_20_pct), NOT via binary "only trade when vol is X" gate. The gate approach is Chan's warning zone; vol-standardisation is literature-blessed.

## Reframing of Phase 2.7 verdict (post-verification)

### The 5 "GOLD" lanes re-ranked with correlation gate

| Lane | Chordia t | 2024 flag | Correlation gate | Deploy status |
|------|----------:|-----------|:----------------:|---------------|
| `MNQ_COMEX_SETTLE E2 RR1.0 OVNRNG_100` | 3.42 | NEUTRAL | **FAIL** (rho=0.80 vs COMEX ORB_G5 RR1.5 on 7/10 profiles) | SWAP-CANDIDATE only |
| `MNQ_COMEX_SETTLE E2 RR1.0 X_MES_ATR60` | 4.29 | NEUTRAL | **FAIL** (rho=0.79) | SWAP-CANDIDATE only |
| `MNQ_COMEX_SETTLE E2 RR1.5 X_MES_ATR60` | 3.32 | NEUTRAL | **FAIL** (rho=1.00) | SWAP-CANDIDATE only |
| `MNQ_SINGAPORE_OPEN E2 RR1.5 ATR_P50_O30` | 4.14 | NEUTRAL | **PASS** | **GOLD — DEPLOY-ELIGIBLE** |
| `MNQ_US_DATA_1000 E2 RR1.5 VWAP_MID_ALIGNED_O15` | 3.20 | NEUTRAL | **PASS** | **GOLD — DEPLOY-ELIGIBLE** |

Plus: `MES_CME_PRECLOSE E2 RR1.0 ORB_G8` (Chordia t=3.66, full_N=130) — not in Phase 2.5 PASS set due to correlation filter tie-breaking; genuinely strong per-trade but thin-N on 2024 alone (Phase 2.7 UNEVALUABLE). Third deploy-eligible candidate on a different instrument.

### Phase 2.7 regime-break framing corrected

Per-year ATR_20_pct medians (MNQ):

- 2020 COVID: 73.0 (HIGH)
- 2022 rate-hike: 59.5 (HIGH)
- **2024: 77.4 (HIGHEST)**
- 2025: 62.7 (elevated)
- 2021/2023 LOW

**"2024 regime break" is too narrow.** This is a RECURRING vol-regime pattern. Our SGP-momentum failure and X_MES_ATR60 CME_PRECLOSE failure in 2024 may also be present in 2020 and 2022 — NOT YET TESTED. Next-audit scope.

## Institutional-grade future-proofing (process + infrastructure)

User's push triggered a quality check on our own process. Here's what we're formalizing:

### 1. PDF-verification discipline (tighten `institutional-rigor.md` rule 7)

**Current:** literature/ extracts claim verbatim status via metadata. I've been trusting them this session without re-extracting PDFs.

**Proposed addition to `.claude/rules/institutional-rigor.md` rule 7:**

> When citing a literature claim in a user-facing verdict, deploy decision, or committee action doc, the SPECIFIC PASSAGE must be re-extracted from `resources/*.pdf` in the current session via `pypdf.PdfReader` (or equivalent) and quoted verbatim in the verdict doc. Trusting the extract in `docs/institutional/literature/` alone is insufficient for load-bearing decisions. Reading the extract is acceptable for CONTEXT; the verdict-doc QUOTE must be PDF-sourced this session.

### 2. Correlation-gate sweep as standard verification layer

Phase 2.5 + 2.7 both produced GOLD candidates. Neither automatically ran `trading_app.lane_correlation.check_candidate_correlation` against existing deployed profiles. Phase 2.7 caveat verification ran it ad-hoc and revealed 3 of 5 "GOLD" lanes are correlation-redundant.

**Proposed addition to `.claude/rules/backtesting-methodology.md`:** new `RULE 14 — correlation gate applies to audit GOLD candidates`. Any lane proposed for deploy from an audit must pass `check_candidate_correlation` against each active profile before being labeled GOLD.

### 3. Multi-year regime stratification as standard

Phase 2.7 stratified only 2024 vs rest. Should have stratified ALL full-year windows to test whether regime-sensitivity is a recurring pattern.

**Proposed:** next Phase 2.7-class audit stratifies per-year for all years with >=100 trade-days. Flag recurrence: if a lane fails C9 in 2+ years, it's regime-recurring not year-specific.

### 4. Vol-regime deploy decision tree (for WATCH lanes)

Based on Chan p120 + Carver p40:

```
Lane passes Chordia + is regime-DEPENDENT (WATCH):
  ├── Option A (Chan p120 warned FAILS for futures): binary regime-gate
  │     "only trade when ATR > X"           ← REJECTED, futures = directional
  ├── Option B (Carver p40 grounded): vol-standardised sizing
  │     "always trade; size ∝ 1/ATR"        ← PREFERRED path
  └── Option C (Bailey-LdP 2014 DSR): signal-only shadow
        "monitor 6-12mo, no capital, Shiryaev-Roberts tracking"
                                            ← SAFE DEFAULT before any deploy
```

**Doctrine:** default is Option C (signal-only shadow) until Option B infrastructure exists (`trading_app/vol_scaled_sizing.py` module not yet built).

### 5. Remaining PDF verification queue

The 6 PDFs I cited-without-verifying this session:

| PDF | Literature extract | What I cited it for | Priority |
|-----|-------------------|---------------------|:--------:|
| `Two_Million_Trading_Strategies_FDR.pdf` | chordia_et_al_2018 | t≥3.00 threshold | HIGH |
| `backtesting_dukepeople_liu.pdf` | harvey_liu_2015 | N≥100 Exhibit 4 | HIGH |
| `deflated-sharpe.pdf` | bailey_lopez_de_prado_2014 | DSR methodology | MED |
| `Algorithmic_Trading_Chan.pdf` | chan_2013_ch7 | intraday momentum / FSTX | MED |
| `Evidence_Based_Technical_Analysis_Aronson.pdf` | (Ch 6 data-mining via quant-audit-protocol) | lift-vs-noise framing | MED |
| `Pseudo-mathematics-and-financial-charlatanism.pdf` | bailey_et_al_2013 | MinBTL bound | LOW (trivially satisfied in our K=6 work) |

Filed as follow-up stage `phase-2-8-pdf-re-verification`.

## Immediate next steps (user's call)

1. **Re-verify remaining 6 PDFs** via direct pypdf extraction — ~30 min per PDF, produces PDF-grounded citation blocks
2. **Run multi-year regime stratification** (ex-2020 / ex-2022 / ex-2023 / ex-2024) on all 38 lanes — 1h
3. **Deploy decision on the 2 truly-GOLD lanes** (SGP ATR_P50_O30, US_DATA_1000 VWAP_MID_ALIGNED_O15) — governance, not research
4. **Build `trading_app/vol_scaled_sizing.py`** module per Carver p40 to unblock Option B path for WATCH lanes — eng work
5. **Test 2020/2022 vol-regime break** on SGP and X_MES_ATR60 lanes — confirms or refutes the "recurring high-vol regime" reframing

## Self-audit

- Chan 2008 passages: PDF-verified this session ✓
- Carver p40 VOLATILITY STANDARDISATION: PDF-verified this session ✓
- Other 6 PDFs: NOT re-verified, flagged, filed as follow-up
- Correlation gate: run via canonical `check_candidate_correlation`, not narrative
- Vol-by-year: run via canonical `daily_features` SQL, not narrative
- Reframing is grounded in EXECUTION + PDF, not inference
