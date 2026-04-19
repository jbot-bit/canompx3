# Pre-reg STUB — ATR_P50 cross-session extension

**Status:** STUB — awaits user approval before locking as yaml + executing.
**Origin:** Phase 2.5 portfolio subset-t sweep (commit 051c2851) identified
`ATR_P50` as a Tier-1 filter class: twin Chordia-PASS on SINGAPORE_OPEN O15
and O30 at RR1.5 long. Only tested on SGP; high theory-motivation for
cross-session extension.
**Proposed slug:** `2026-04-19-atr-p50-cross-session-extension-v1.yaml`

## Evidence from Phase 2.5

| Session | Aperture | RR | N | ExpR | Subset t | Tier |
|---------|----------|-----|---:|------:|---------:|------|
| SINGAPORE_OPEN | 15 | 1.5 | 496 | +0.205 | **3.96** | 1 PASS |
| SINGAPORE_OPEN | 30 | 1.5 | 485 | +0.221 | **4.14** | 1 PASS |

**Pattern:** `ATR_P50_O{15,30}` is an instrument-level volatility-percentile
regime gate (ATR ≥ 50th percentile of rolling lookback). Mechanism is
session-agnostic — works wherever there's a tradeable ORB in a non-dormant
vol regime. Partial coverage (SGP only) is an artifact of prior discovery
scope, not a theoretical limitation.

## Proposed K=4 cells

Scope: extend to 4 core sessions where ATR_P50_O15 was not previously
tested, all at RR1.5 long (matching PASS surface):

1. MNQ TOKYO_OPEN E2 O5 RR1.5 long ATR_P50_O15
2. MNQ COMEX_SETTLE E2 O5 RR1.5 long ATR_P50_O15
3. MNQ US_DATA_1000 E2 O5 RR1.5 long ATR_P50_O15
4. MNQ CME_PRECLOSE E2 O5 RR1.5 long ATR_P50_O15

Aperture collapsed to O15 only (O30 is a twin; per lane_correlation.py
`subset_coverage > 0.80` rule, testing both apertures on 4 new sessions
would inflate K to 8 with expected Jaccard ~0.65 per session — SGP
result shows same twin pattern. If O15 passes and aperture-doubling is
desired later, O30 can be tested separately.)

## Literature grounding

- Aronson `Evidence_Based_Technical_Analysis.pdf` Ch 6 data-mining — ATR
  percentile is a classic regime-conditioning filter; not a data-mined feature
- Fitschen 2013 Ch 3 `fitschen_2013_path_of_least_resistance.md` — intraday
  trend-follow on equity indices requires vol regime support; low-ATR days
  fade, high-ATR days follow
- LdP 2020 — Pathway B theory-first K=1 per cell (K=4 family)

## Bailey MinBTL

K=4 trivially satisfied on 6.65yr clean MNQ. Pathway B individual per cell.

## Kill criteria (per cell, matches X_MES_ATR60 stub for consistency)

- C3 p < 0.05 at K=1 per-cell (or BH-FDR at K=4 family)
- C4 Chordia t ≥ 3.00 (with-theory)
- C6 WFE ≥ 0.50
- C7 N ≥ 100
- C9 no year with N≥50 and ExpR < −0.05
- T0 tautology |corr(new_fire, ATR_P50_O15 fire on SGP)| — if >0.90
  on cross-session universe, mechanism is session-coincidence not regime
- T5 direction symmetry
- T8 cross-instrument MES consistency

## Expected results

Prior: if mechanism is truly vol-regime (not SGP-specific), 2-4 should pass.
If 0 pass → mechanism is SGP-specific and the PASS finding was
session-interaction, not pure regime.

## Action required from user

Review stub. If approved → lock as yaml, commit, execute via a K=4 scan
script modeled on Phase 2.5.

## Deferred — NOT in scope

- O30 aperture extension (aperture-twin; test separately if O15 passes)
- RR1.0 / RR2.0 axis completion — separate pre-reg if O15 RR1.5 validates
- Composite ATR_P50 × OVNRNG_100 (another Tier-1 filter) — requires
  `CompositeFilter` infrastructure

## Relationship to X_MES_ATR60 stub

Both stubs test filter classes from Phase 2.5 Tier-1. Different mechanisms:
- X_MES_ATR60: cross-asset (MES vol → MNQ trade)
- ATR_P50: same-asset (MNQ's own ATR percentile)

If BOTH mechanisms pass, composite (X_MES_ATR60 ∧ ATR_P50) would be
the highest-theoretical-confidence filter pair in the book — but would
need CompositeFilter infra + third pre-reg. Noted as potential future step
only if both individual extensions pass.
