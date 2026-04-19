# Session handoff — 2026-04-19 regime-classify session

**Branch:** `research/campaign-2026-04-19-phase-2`
**Worktree:** `C:/Users/joshd/canompx3/.worktrees/campaign-2026-04-19-phase-2`
**Last commit (pushed):** `dbb0fa53` audit(phase-2-8): multi-year regime stratification
**Background job running at handoff:** `phase_2_8_multi_year_regime_stratification.py --comprehensive` (7-year sweep, ~10 min wall time)

## What was accomplished this session

| Commit | Work |
|--------|------|
| `01ec8ecd` | Phase 2.4 cross-session momentum Mode A re-eval |
| `56fb46e4` | MES composite C3/C4 KILL + RULE 8.3 ARITHMETIC_LIFT addendum |
| `cb756a45` | Code review items 1-5 (canonical delegation, C8 tier, tests) |
| `051c2851` | Phase 2.5 portfolio subset-t + lift-vs-unfiltered sweep |
| `df0ace80` | Wire Phase 2.5 into consolidated retirement verdict |
| `7ea23b0d` | Pre-reg stubs for X_MES_ATR60 + ATR_P50 cross-session extensions |
| `391b417f` | Phase 2.6 X_MES_ATR60 K=6 audit (0 clean pass, 2 BH-pass C9-2024-fail) |
| `1192f184` | Phase 2.7 2024 regime-break systemic audit (original) |
| `f7b1897e` | Phase 2.7 caveat verification + PDF-grounded reanalysis (reframed GOLD 5→2, Chan p120 stock/options finding) |
| `dbb0fa53` | Phase 2.8 multi-year stratification refutes "recurring vol regime" reframe — 2024 IS year-specific |

## Current state of the book (post all audits)

**Truly deploy-safe GOLD (2 lanes):**
1. `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30` — t=4.14, VOL_NEUTRAL, correlation-gate PASS
2. `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` — t=3.20, VOL_NEUTRAL, correlation-gate PASS

**Swap-candidate GOLD (Chordia + regime-neutral BUT correlation-redundant on most profiles):**
- `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100` — rho=0.80 vs deployed ORB_G5 RR1.5
- `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60` — rho=0.79
- `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60` — rho=1.00 on shared days

**DOUBLE-CONFIRMED retire (Phase 2.4 + Phase 2.7 + Phase 2.8):**
- `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_CROSS_SGP_MOMENTUM` — SINGLE_YEAR_DRAG 2024 (year-specific)
- `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_CROSS_SGP_MOMENTUM` — SINGLE_YEAR_DRAG 2024

**Rule 8.3 ARITHMETIC_LIFT retire/reframe (Phase 2.5):**
- `MNQ_US_DATA_1000_E2_RR2.0_CB1_VWAP_MID_ALIGNED_O15`
- `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`

## PDF-verified literature grounding

- **Chan 2008 Ch 7 § Regime Switching** (resources/Quantitative_Trading_Chan_2008.pdf pp 142-143) — verified direct pypdf extraction this session. Key passage: "volatility regime switches can be of great value to options traders, they are unfortunately of no help to stock traders." We trade futures = directional = Chan's warning applies. Binary vol-regime-gate REJECTED; Carver vol-standardised sizing PREFERRED.
- **Carver Ch 2 p40 § VOLATILITY STANDARDISATION** (resources/Robert Carver - Systematic Trading.pdf p57) — verified. "One of the most powerful techniques I use... adjusting the returns of different assets so that they have the same expected risk."

## PDF verification DEBT (flag for fresh terminal)

NOT re-verified this session (still trusting `docs/institutional/literature/` extracts):
- `Two_Million_Trading_Strategies_FDR.pdf` (Chordia 2018 t≥3.00)
- `backtesting_dukepeople_liu.pdf` (Harvey-Liu 2015 N≥100)
- `deflated-sharpe.pdf` (Bailey-LdP 2014 DSR)
- `Algorithmic_Trading_Chan.pdf` (Chan 2013 Ch 7 FSTX example)
- `Evidence_Based_Technical_Analysis_Aronson.pdf` (Ch 6 data-mining)
- `Pseudo-mathematics-and-financial-charlatanism.pdf` (Bailey 2013)

Each takes ~5-15 min via pypdf. Priority: Chordia (most-cited) + Carver Ch 9-10 (continue from Ch 2).

## Open work / follow-ups (for fresh terminal)

### IMMEDIATE (0-30 min)
1. **Read comprehensive 7-year sweep output** at `research/output/phase_2_8_multi_year_regime_stratification_comprehensive.csv` (should be complete by fresh-terminal start). Interpret findings:
   - Any RECURRING_VOL_DRAG patterns now?
   - Does 2023 (low-vol) reveal any regime-dependent lanes?
   - Does 2025 partial-year show 2024-pattern continuing or reverting?
2. **Commit the comprehensive sweep output + update result doc** with new findings
3. **Re-read** `docs/audit/results/2026-04-19-phase-2-7-reanalysis-and-future-proofing.md` for the consolidated doctrine proposals

### SHORT HORIZON (30min - 2h)
4. **PDF re-verification queue** — at minimum Chordia + Carver Ch 9-10 (continues from Ch 2 already done)
5. **Per-instrument year-pattern audit** (MES/MGC vs MNQ) — currently aggregated
6. **Filter-class aggregation** (all SGP, all X_MES_ATR60 pooled) — may reveal filter-level regime patterns the per-lane view hides
7. **Per-year Sharpe (not just ExpR)** — volatility-adjusted view

### MED HORIZON (2-4h each)
8. **Build `trading_app/vol_scaled_sizing.py`** per Carver p40 — unblocks Option B deploy path for WATCH lanes
9. **Lock + execute ATR_P50 cross-session K=4 pre-reg** (stub exists at `docs/audit/hypotheses/2026-04-19-atr-p50-cross-session-extension-stub.md`) — **update stub first to use O30 preferentially** (per Phase 2.7 finding O30 is regime-robust, O15 is regime-dependent)
10. **Governance decisions (user's call):**
   - Retire the 2 SGP PURE_DRAG lanes
   - Reclassify 2 ARITHMETIC_LIFT lanes to research-provisional
   - Deploy the 2 truly-GOLD lanes
   - Amendment v3.2 lock (from prior-terminal work at commit `5f7463aa` / `fc920871`)

## Known blockers

- **`CompositeFilter` infrastructure not yet built** — blocks any composite lane deploy
- **6 PDFs still trust-metadata** — flagged above
- **Allocator-level portfolio simulation** not yet run (caveat b from Phase 2.7)
- **Phase 0 branch hygiene on main** still pending (14 unpushed commits + PR #7 pull per prior session)

## Institutional-rigor reminders (from this session)

1. **PDF verification discipline** — verdict-doc QUOTES must be pypdf-extracted this session, not trusted from literature/ metadata (proposed rule 7 tightening)
2. **Correlation gate as standard** — GOLD candidates MUST pass check_candidate_correlation before deploy-label (proposed RULE 14)
3. **Multi-year regime stratification standard** for year-specific-fail findings
4. **Chan p120 stock/options discrimination** — we are stock/directional, vol-regime-binary-gate = WARNED; prefer vol-standardised sizing per Carver p40

## Fresh-terminal start prompt

```
Resuming campaign 2026-04-19 phase 2 work. Last commit dbb0fa53 on
origin/research/campaign-2026-04-19-phase-2.

Worktree: C:/Users/joshd/canompx3/.worktrees/campaign-2026-04-19-phase-2

Read docs/handoffs/2026-04-19-regime-classify-session-handoff.md first.

First task: check if background Phase 2.8 comprehensive 7-year sweep completed
(expected output: research/output/phase_2_8_multi_year_regime_stratification_comprehensive.csv).
If yes, interpret findings and commit. If not, wait/run.

Then: pick from the open-work queue in the handoff doc based on user direction.
Default priority: (1) read comprehensive sweep, (2) PDF re-verification of
Chordia, (3) filter-class aggregation audit.

Institutional-rigor reminders:
- PDF verification discipline: pypdf-extract passages you cite; don't trust
  literature/ extract metadata alone
- Canonical delegations: load_active_setups, compute_mode_a, filter_signal,
  HOLDOUT_SACRED_FROM, GOLD_DB_PATH, SESSION_CATALOG
- Chan p120: we are directional/stock — vol-regime-binary-gate is warned;
  prefer Carver vol-standardised sizing
- Pre-reg BEFORE scan runs (Phase 0 doctrine); commit pre-reg first then
  script execution

DB override for worktree: DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db uv run python ...
```

## Notes for user

- Session built 10 commits (01ec8ecd → dbb0fa53) on campaign-phase-2 branch
- All tests pass (Phase 2.4 18, Phase 2.5 27, Phase 2.6 17, Phase 2.7 21, Phase 2.8 14 = 97 new tests green this session)
- Drift check: pre-existing Check 37 only (worktree lacks local gold.db, expected)
- No production code mutations; all work is in `research/` + `docs/` + new tests
