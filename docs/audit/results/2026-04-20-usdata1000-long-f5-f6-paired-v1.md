# US_DATA_1000 Long F5/F6 Paired Verify V1

**Pre-reg:** `docs/audit/hypotheses/2026-04-20-usdata1000-long-f5-f6-paired-v1.yaml`
**Family verdict:** **ALIVE**

## Scope

- Instrument: MNQ
- Session: US_DATA_1000
- Direction: long
- Aperture: O5
- Entry model: E2
- Confirm bars: CB1
- RR targets: 1.0, 1.5
- Hypotheses: F5_BELOW_PDL TAKE; F6_INSIDE_PDR AVOID

## Resource grounding

- `resources/Algorithmic_Trading_Chan.pdf`: bounded executable strategy families are valid research units when rules are explicit and objective.
- `resources/Robert Carver - Systematic Trading.pdf`: useful signals can live as TAKE/AVOID conditioners rather than being forced into standalone systems.
- `resources/Lopez_de_Prado_ML_for_Asset_Managers.pdf`: theory-first, small-family testing over broad post-hoc fishing.
- `resources/Two_Million_Trading_Strategies_FDR.pdf`: honest family-level FDR remains mandatory even on small bounded families.

## Family accounting

- Locked K: 4
- Primary survivors: 4

## Cell results

- RR1.0 F5_BELOW_PDL TAKE: IS on/off ExpR 0.3258/-0.0112 | delta 0.3370 | t=4.02 | q=0.0003 | N_on/N_off=136/745 | OOS delta=0.1155 | dir_match=None | verdict=PASS
- RR1.5 F5_BELOW_PDL TAKE: IS on/off ExpR 0.4022/-0.0024 | delta 0.4046 | t=3.65 | q=0.0006 | N_on/N_off=135/731 | OOS delta=0.0335 | dir_match=None | verdict=PASS
- RR1.0 F6_INSIDE_PDR AVOID: IS on/off ExpR -0.0434/0.1583 | delta -0.2017 | t=-3.15 | q=0.0017 | N_on/N_off=513/368 | OOS delta=-0.0223 | dir_match=None | verdict=PASS
- RR1.5 F6_INSIDE_PDR AVOID: IS on/off ExpR -0.0588/0.2261 | delta -0.2849 | t=-3.52 | q=0.0006 | N_on/N_off=503/363 | OOS delta=0.2272 | dir_match=None | verdict=PASS

## Interpretation

- F5_BELOW_PDL remains a TAKE state at RR1.0: washed-out below-PDL opens improve long trade quality.
- F5_BELOW_PDL remains a TAKE state at RR1.5: washed-out below-PDL opens improve long trade quality.
- F6_INSIDE_PDR remains an AVOID state at RR1.0: inside-range opens continue to degrade long trade quality.
- F6_INSIDE_PDR remains an AVOID state at RR1.5: inside-range opens continue to degrade long trade quality.

## Caveats

- This is a context-family verify, not a standalone strategy claim.
- 2026 OOS remains descriptive when a cell has fewer than 30 on-signal or 30 off-signal observations.
- No capital, sizing, allocator, or live-filter action is authorized by this result alone.

## Artefacts

- CSV: `research/output/usdata1000_long_f5_f6_paired_v1.csv`
- Script: `research/usdata1000_long_f5_f6_paired_v1.py`
