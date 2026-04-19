# X_MES_ATR60 cross-session extension — K=6 audit results

**Date:** 2026-04-19
**Pre-reg:** `docs/audit/hypotheses/2026-04-19-x-mes-atr60-cross-session-extension-v1.yaml` (locked 22:30 Bris before script ran)
**Script:** `research/phase_2_6_x_mes_atr60_cross_session_audit.py`
**Output CSV:** `research/output/phase_2_6_x_mes_atr60_cross_session_audit.csv`
**Origin:** Phase 2.5 Tier-1 unlock candidate (commit 051c2851)

## Executive verdict

**Per-cell PASS (all C3-C10 gates): 0 / 6.**
**Family BH-FDR K=6 q=0.05 PASS: 2 / 6** (CME_PRECLOSE RR1.5 + RR2.0).

The 2 cells that clear BH-FDR at family level (strong discovery evidence,
t≥3.00, p<0.002) BOTH fail C9 era-stability on 2024 — same 2024 regime
break we saw in SGP RR1.5 Phase 2.4 work. No clean new Tier-1 additions
from this audit.

**Honest-null outcome:** US_DATA_830 is a CLEAN NULL across all 3 RRs
(t=0.88, -0.43, 0.16). X_MES_ATR60 mechanism does NOT extend to
pre-US-cash-open macro release window.

## Per-cell results

| # | Cell | N | ExpR | t | p | WFE | C3 | C4 | C6 | C7 | C9 | C10 | BH-FDR K=6 | Verdict |
|---|------|---:|------:|------:|-------:|-------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|--------|
| 1 | MNQ CME_PRECLOSE RR1.5 | 271 | +0.244 | +3.45 | 0.0006 | +1.36 | P | P | P | P | **F** | P | **PASS** | FAIL_CELL (C9 2024) |
| 2 | MNQ CME_PRECLOSE RR2.0 | 244 | +0.277 | +3.11 | 0.0021 | +1.20 | P | P | P | P | **F** | P | **PASS** | FAIL_CELL (C9 2024) |
| 3 | MNQ US_DATA_830 RR1.0 | 345 | +0.044 | +0.88 | 0.38 | +14.35 | F | F | P | P | P | P | FAIL | FAIL_CELL |
| 4 | MNQ US_DATA_830 RR1.5 | 341 | −0.026 | −0.43 | 0.67 | — | F | F | F | P | **F** | P | FAIL | FAIL_CELL |
| 5 | MNQ US_DATA_830 RR2.0 | 336 | +0.012 | +0.16 | 0.87 | — | F | F | F | P | **F** | P | FAIL | FAIL_CELL |
| 6 | MNQ US_DATA_1000 RR2.0 | 353 | −0.009 | −0.12 | 0.91 | +0.60 | F | F | P | P | **F** | P | FAIL | FAIL_CELL |

C-gate legend: **C3** (p<0.05), **C4** (Chordia t≥3.00), **C6** (WFE≥0.50), **C7** (N≥100), **C9** (no year N≥50 with ExpR<−0.05), **C10** (MICRO-only era).

## Interpretation

### CME_PRECLOSE (cells 1-2): conditional signal, not deployable

- **Pre-reg prior:** X_MES_ATR60 passes at RR1.0 on CME_PRECLOSE (t=4.19 from Phase 2.5) — expected RR-axis extension.
- **Observed:** Both RR1.5 (t=3.45) and RR2.0 (t=3.11) clear Chordia discovery-strict AND BH-FDR at K=6. **Stronger individual evidence than most of the existing 9-lane Tier-1 book.**
- **BUT:** both fail C9 2024 era-stability. This is the same 2024 regime break pattern we saw in SGP RR1.5 (Phase 2.4). Suggests: the 2024 cross-asset-vol regime change affected BOTH cross-session momentum signals.
- **Not a deploy candidate without regime-conditioning.** Could be revived as:
  - Composite with a 2024-regime detector (separate filter to build)
  - Signal-only shadow for 6-12 months under SR monitor
  - De-sized contribution to a Carver combiner (not binary deploy)

### US_DATA_830 (cells 3-5): CLEAN NULL

- All 3 RRs produce t in [−0.43, +0.88]. No directional hint.
- WFE=14.35 at RR1.0 is suspicious but from near-zero mean sharpe ratio (garbage in garbage out).
- RR1.5 and RR2.0 have 2024-negative + 2025-negative era fails.
- **Mechanism DOES NOT extend to pre-US-cash-open macro release window.** Honest refutation.

### US_DATA_1000 RR2.0 (cell 6): confirms RR-expansion null

- Phase 2.5 showed RR1.0 fails (t=1.56). RR2.0 also fails (t=−0.12).
- **RR-axis adversarial test confirms US_DATA_1000 is not a session where X_MES_ATR60 works** at any tested RR.

## What this audit DID deliver

1. **Literature-grounded refutation on 3 sessions** (US_DATA_830 all RRs + US_DATA_1000 RR2.0). That's solid negative evidence that constrains the mechanism — X_MES_ATR60 works on post-US-cash-close sessions (COMEX_SETTLE, CME_PRECLOSE), not pre-cash-open.
2. **Confirmation of CME_PRECLOSE RR-axis strength** — RR1.5 and RR2.0 both pass discovery-strict.
3. **2024 regime break is cross-cellular, not SGP-specific.** CME_PRECLOSE RR1.5 and RR2.0 also C9-fail on 2024. Portfolio-wide signal.
4. **Honest 0/6 cell-pass rate** — pre-reg predicted "2-4 pass" as baseline; got 0. Mechanism is tighter than theorized, useful calibration.

## What this audit did NOT deliver

- No new clean Tier-1 lanes (0 full-pass cells).
- No rescue path for the existing 9 Tier-1 lanes.
- No new deploy candidates.

## Honest framing per pre-reg expected-result table

The pre-reg stated:
> "Prior: some cells pass, some fail, NOT all-pass. If all 6 pass → suspect.
> If 2-4 pass → honest extension. If 0 pass → mechanism is tighter than
> theorized (only certain-session alignment matters) — useful null."

Actual result: **0 cells fully pass. 2 cells pass family BH-FDR but fail C9 on 2024.** This is between "0 pass" and "2-4 pass" framings — leaning toward "mechanism is tighter than theorized" with a partial-signal observation on CME_PRECLOSE.

## Self-audit / bias check

- Pre-reg locked BEFORE script ran (commit 7ea23b0d) ✓
- Canonical delegations: compute_mode_a, C4/C7/C9 constants, MICRO launch, filter_signal, HOLDOUT_SACRED_FROM ✓
- SESSION_CATALOG whitelist enforced ✓
- Triple-join on (trading_day, symbol, orb_minutes) ✓
- X_MES_ATR60 cross-asset injection mirrored from mode_a_revalidation canonical pattern ✓
- BH-FDR at K=6 computed correctly (sorted p, BHY step-down) ✓
- 6 WFE folds per cell (expanding-window) ✓
- No look-ahead: MES ATR_20 is known at prior close ✓

## Recommendations

1. **Do NOT add any of the 6 cells to the deploy book.** None meet the full C1-C10 gate.
2. **Consider CME_PRECLOSE RR1.5/2.0 for SIGNAL-ONLY SHADOW monitoring** (6-12 months live) — the t-stats are strong enough that a 2024-excluded or 2024-regime-conditional version could be promotable later.
3. **Retire US_DATA_830 × X_MES_ATR60 from the search space.** 3 RRs tested, all null. Not worth re-testing without new theory.
4. **2024 regime break is now a portfolio-wide pattern worth formalizing.** SGP RR1.5 failed C9 2024. CME_PRECLOSE X_MES_ATR60 RR1.5 + RR2.0 fail C9 2024. Existing validated lanes with 2024 exposure may also fail strict Mode A C9 — consider a dedicated "2024 regime break" audit that identifies which currently-deployed lanes survive excluding 2024.

## Next actions

- ATR_P50 cross-session pre-reg (K=4) remains as the next Tier-1 extension to test. Stub already drafted at `docs/audit/hypotheses/2026-04-19-atr-p50-cross-session-extension-stub.md`.
- 2024 regime-break systemic audit — new stage, separate pre-reg.

## Audit trail

- Pre-reg committed 2026-04-19 (commit `7ea23b0d`) BEFORE this script ran
- Stage file: `docs/runtime/stages/phase-2-6-x-mes-atr60-cross-session.md`
- CSV: `research/output/phase_2_6_x_mes_atr60_cross_session_audit.csv`
- Tests: `tests/test_research/test_phase_2_6_x_mes_atr60_cross_session_audit.py`
- Cross-refs:
  - `docs/audit/results/2026-04-19-portfolio-subset-t-sweep.md` (Phase 2.5 origin)
  - `docs/audit/results/2026-04-19-phase-2-4-cross-session-momentum-mode-a.md` (2024 regime-break precedent)
  - `docs/audit/results/2026-04-19-mes-europe-flow-g5-sgp-composite-audit.md` (Rule 8.3 precedent)
