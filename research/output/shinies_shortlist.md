# Shinies Shortlist (saved for later hardening)

Source runs:
- `research/research_lead_lag_best_wide_grid.py`
- commit: `fcfebec`

## 1) PROMOTE CANDIDATE
**Leader -> Follower:** `M6E_US_EQUITY_OPEN -> M2K_US_POST_EQUITY`
- Strategy slice: `E1 / CB5 / RR1.5`
- avg_on: `+0.0814`
- uplift (on-off): `+0.2879`
- yearly positives: `4/5`
- test uplift: `+0.3214`

## 2) PROMOTE CANDIDATE
**Leader -> Follower:** `MES_US_DATA_OPEN -> M2K_US_DATA_OPEN`
- Strategy slice: `E0 / CB1 / RR1.5`
- avg_on: `+0.0409`
- uplift (on-off): `+0.2100`
- yearly positives: `5/5`
- test uplift: `+0.1851`

## 3) HOLD / PROMOTE BORDERLINE
**Leader -> Follower:** `MES_1000 -> M2K_US_POST_EQUITY`
- Strategy slice: `E1 / CB5 / RR1.5`
- avg_on: `+0.0345`
- uplift (on-off): `+0.1997`
- yearly positives: `5/5`
- test uplift: `+0.1618`

## Notes
- These are saved as current best candidates.
- Not production-promoted yet; hardening/verification pass pending.
