# Pre-reg STUB — X_MES_ATR60 cross-session extension

**Status:** STUB — awaits user approval before locking as yaml + executing.
**Origin:** Phase 2.5 portfolio subset-t sweep (commit 051c2851) identified
`X_MES_ATR60` as a Tier-1 filter class: 3 of 9 Chordia-PASS lanes use it.
Partial session coverage suggests volatility/regime mechanism worth testing
on currently-untested sessions.
**Proposed slug:** `2026-04-19-x-mes-atr60-cross-session-extension-v1.yaml`

## Evidence from Phase 2.5

| Session | RR | N | ExpR | Subset t | Tier |
|---------|-----|---:|------:|---------:|------|
| COMEX_SETTLE | 1.0 | 385 | +0.195 | **4.29** | 1 PASS |
| COMEX_SETTLE | 1.5 | 379 | +0.198 | **3.32** | 1 PASS |
| CME_PRECLOSE | 1.0 | 306 | +0.214 | **4.19** | 1 PASS |
| NYSE_OPEN | 1.0 | 345 | +0.078 | 1.51 | 4 FAIL |
| NYSE_OPEN | 1.5 | 334 | +0.066 | 1.00 | 4 FAIL |
| US_DATA_1000 | 1.0 | 371 | +0.077 | 1.56 | 4 FAIL |

**Pattern:** passes on COMEX_SETTLE (evening US close) and CME_PRECLOSE
(overnight-to-Asia handoff) but FAILS on NYSE_OPEN (intraday US equity open)
and US_DATA_1000 (mid-US-session macro releases). Sessions where
MES vol reading is a "resting" proxy seem to pass; sessions where MES is
active-moving seem to fail.

## Proposed K=6 cells

Scope narrowed by theory (MES is canonical equity-index vol signal — must be
tested where MES is actively trading; Asian/European sessions outside US
hours skipped):

1. MNQ CME_PRECLOSE E2 O5 RR1.5 long
2. MNQ CME_PRECLOSE E2 O5 RR2.0 long
3. MNQ US_DATA_830 E2 O5 RR1.0 long  — pre-open macro release, MES active
4. MNQ US_DATA_830 E2 O5 RR1.5 long
5. MNQ US_DATA_830 E2 O5 RR2.0 long
6. MNQ US_DATA_1000 E2 O5 RR2.0 long — RR axis completion on known-fail RR1.0

## Literature grounding

- Chan 2013 Ch 7 `chan_2013_ch7_intraday_momentum.md` — intraday vol regime
  predicts breakout edge on equity indices
- Carver 2015 Ch 9-10 `carver_2015_volatility_targeting_position_sizing.md` —
  volatility-targeted sizing; if MES ATR is a regime proxy for MNQ, this
  extends to continuous sizing once validated
- LdP 2020 `lopez_de_prado_2020_ml_for_asset_managers.md` — theory-first /
  CPCV methodology applies: K=6 is tight enough to pre-commit hypotheses

## Bailey MinBTL check

`MinBTL = 2·ln(6) / E[max_N]²` ≈ trivially satisfied at K=6 on 6.65yr clean
MNQ. Pathway B individual per cell.

## Kill criteria (per cell)

- C3 p < 0.05 at K=1 per-cell (or BH-FDR at K=6 family)
- C4 Chordia t ≥ 3.00 (with-theory per citations above)
- C6 WFE ≥ 0.50
- C7 N ≥ 100
- C9 no year with N≥50 and ExpR < −0.05
- T0 tautology |corr(new_fire, X_MES_ATR60 fire on other session)| > 0.90
- T5 direction symmetry: short must not exceed long on same cell
- T8 cross-instrument: MES's OWN performance on same cell must show consistent sign

## Expected results

Prior: some cells pass, some fail, NOT all-pass. If all 6 pass → suspect.
If 2-4 pass → honest extension. If 0 pass → mechanism is tighter than
theorized (only certain-session alignment matters) — useful null.

## Action required from user

Review this stub. If approved: rename to `.yaml`, lock with ISO timestamp,
commit as pre-reg, then execute with a K=6 scan script modeled on
`research/phase_2_4_mes_composite_c1_c12_audit.py` (K=1 audit) extended to
loop over the 6 cells.

## Deferred — NOT in scope of this stub

- Cross-RR expansion on NYSE_OPEN (already failed at RR1.0/1.5; RR2.0 may
  or may not pass; low-theory-motivation, separate audit if needed)
- Cross-instrument MES's own X_MNQ_ATR60 (reverse polarity) — new pre-reg
- Compositing X_MES_ATR60 with other Tier-1 filters (ATR_P50, OVNRNG_100) —
  separate composite pre-reg; requires `CompositeFilter` infrastructure
