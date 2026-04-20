# F5 NYSE_OPEN Short Deployed-Lane Verify

**Pre-reg:** `docs/audit/hypotheses/2026-04-20-f5-nyo-short-deployed-lane-verify.yaml`
**Lane:** `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` short
**Verdict:** **CONDITIONAL_UNVERIFIED**

## Resource grounding

- `resources/Algorithmic_Trading_Chan.pdf`: bounded executable strategy families are acceptable research units when the rules are explicit.
- `resources/Robert Carver - Systematic Trading.pdf`: a useful signal can live as a conditioner or sizing input without needing to become a standalone system.
- `resources/Lopez_de_Prado_ML_for_Asset_Managers.pdf`: theory-first and small-family verification over broad fishing.

## Canonical results

- IS: N_on=117, N_off=708, ExpR_on=+0.3559, ExpR_off=+0.0519, delta=+0.3040, t=3.38, p=0.0009
- IS block-bootstrap p=0.0020
- OOS: N_on=8, N_off=33, ExpR_on=-0.0167, ExpR_off=+0.1311, delta=-0.1478, p=0.7255
- OOS dir_match=False

## RULE 3.3 power floor

```text
  OOS power:
    Cohen's d (IS effect): 0.319
    Expected OOS SE:       0.3752
    Expected 95% CI half-width: 0.7354
    Power at alpha=0.05 two-sided: 12.4%
    N per group for 80% power: 155
    RULE 3.3 tier: STATISTICALLY_USELESS
OOS tier: STATISTICALLY_USELESS
```

## IS year-by-year

- 2019: N_on=17, N_off=61, delta=+0.3238
- 2020: N_on=9, N_off=116, delta=NA
- 2021: N_on=13, N_off=104, delta=+0.3073
- 2022: N_on=28, N_off=111, delta=+0.1944
- 2023: N_on=13, N_off=92, delta=+0.2193
- 2024: N_on=20, N_off=110, delta=+0.0953
- 2025: N_on=17, N_off=114, delta=+0.6854

## Interpretation

- IS evidence is strong enough to keep this as a real exact-lane candidate.
- OOS is still too thin to refute or promote; any sign flip here is descriptive only, so the correct label remains CONDITIONAL_UNVERIFIED rather than CONFIRMED.
- Correct role if pursued further: conditioner / deployment-shape, not standalone strategy.
