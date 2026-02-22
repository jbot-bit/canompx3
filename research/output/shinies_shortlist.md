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

## B) High-frequency candidate (>=150 signals/year target)

### B1) M2K_1000 -> MES_1000
- Strategy slice: `E0 / CB1 / RR2.5`
- Estimated usable frequency: `~166 signals/year` (`N_on=830 over 5 years`)
- avg_on: `+0.0283`
- uplift (ON-OFF): `+0.1464`
- 2025 OOS uplift: `+0.1644`
- Status: **KEEP (frequency-qualified common-ground candidate)**

### B2) MES 1000 (single-asset filter)
- Strategy slice: `E0 / CB1 / RR2.5`
- Filter: `fast_le_15` (break_delay <= 15m)
- Estimated usable frequency: `~164 signals/year` (`N_on=1311`)
- avg_on: `+0.0111`
- uplift (ON-OFF): `+0.1245`
- 2025 OOS uplift: `+0.3414` (`N_test_on=210`)
- Status: **KEEP (high-frequency common-ground filter)**

## C) Explicitly not kept

### Relay-chain hypothesis (M6E + MES -> M2K_US_POST_EQUITY)
- Result: did not beat single-leader filter enough for promotion.
- Status: **KILL (for now)**

## D) Rejected overlays (tested)

### BQS false-breakout overlays on shinies
- Tested variants: `veto_D`, `BQS>=3`, `strict_all4` across A1/A2/A3/B1/B2.
- Outcome: no meaningful practical improvement for core high-frequency deployment; gains were either tiny or came with major frequency loss.
- Decision: **NO-GO for default deployment** (keep as optional research path only).

## Notes
- These are saved as current best candidates for future hardening.
- Not production-promoted yet.
- Promotion later requires strict DD + stability + implementation checks.
