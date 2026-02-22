# Shinies Shortlist (saved for later hardening)

Purpose: keep only candidates worth revisiting for implementation.

Last updated: 2026-02-22

## A) Keep / Promote candidates

### A1) M6E_US_EQUITY_OPEN -> M2K_US_POST_EQUITY
- Strategy slice: `E1 / CB5 / RR1.5`
- Baseline avgR: `-0.0601`
- Filter ON avgR: `+0.0814`
- Uplift (ON-OFF): `+0.2879`
- Yearly uplift positive: `5/6`
- 2025 OOS uplift: `+0.4401` (`N_on=104`)
- Status: **KEEP (top shiny)**

### A2) MES_US_DATA_OPEN -> M2K_US_DATA_OPEN
- Strategy slice: `E0 / CB1 / RR1.5`
- Baseline avgR: `-0.0557`
- Filter ON avgR: `+0.0409`
- Uplift (ON-OFF): `+0.2100`
- Yearly uplift positive: `5/6`
- 2025 OOS uplift: `+0.1851` (`N_on=107`)
- Status: **KEEP (strong)**

### A3) MES_1000 -> M2K_US_POST_EQUITY
- Strategy slice: `E1 / CB5 / RR1.5`
- Baseline avgR: `-0.0624`
- Filter ON avgR: `+0.0345`
- Uplift (ON-OFF): `+0.1997`
- Yearly uplift positive: `5/6`
- 2025 OOS uplift: `+0.2801` (`N_on=91`)
- Status: **KEEP (backup / borderline promote)**

## B) Explicitly not kept

### Relay-chain hypothesis (M6E + MES -> M2K_US_POST_EQUITY)
- Result: did not beat single-leader filter enough for promotion.
- Status: **KILL (for now)**

## Notes
- These are saved as current best candidates for future hardening.
- Not production-promoted yet.
- Promotion later requires strict DD + stability + implementation checks.
