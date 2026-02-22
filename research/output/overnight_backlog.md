# Overnight Backlog

## Lead-lag full session sweep (queued, not now)

Run an overnight exhaustive test for lead-lag filters across **all enabled sessions** and symbols,
including these condition families:

1. Same-direction (current baseline)
2. Opposite-direction
3. Asymmetric side-specific (leader long-only / short-only effects)
4. Strength-gated (leader break delay/size thresholds)
5. Divergence conditions

Hard gates:
- No-lookahead (`leader_break_ts <= follower_entry_ts`)
- Frequency target (~150 usable signals/year)
- Positive avgR and meaningful uplift
- OOS stability

Status: queued by user request on 2026-02-22.
