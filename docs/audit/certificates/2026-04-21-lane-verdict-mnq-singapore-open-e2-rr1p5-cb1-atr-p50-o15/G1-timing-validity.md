# G1 — Timing-Validity Certificate

**Candidate:** `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15`
**Date authored:** `2026-04-21`
**Pre-reg:** `N/A — deployed lane retrospective verdict`
**Entry model:** `E2`
**Decision-time bar:** `STARTUP`

---

## Purpose

This retrospective lane verdict reuses the binding timing-validity result from [the Phase A snapshot](/mnt/c/Users/joshd/canompx3/docs/audit/2026-04-21-reset-snapshot.md:615). The lane uses `ATR_P50` only; no rel_vol or break-bar variables are present.

## Ban-list cross-check (`.claude/rules/daily-features-joins.md` § Look-Ahead Columns)

- [x] No `break_ts`, `break_delay_min`, `break_bar_volume`, or `rel_vol_*` inputs in the deployed lane definition
- [x] No outcome-derived columns (`pnl_r`, `mae_r`, `mfe_r`) used as filters
- [x] No look-ahead ban-list hits in the current live framing

## Variable inventory

| Variable | Source table | Source column(s) | Source bar / window | Knowable at decision-time bar? | Evidence |
|---|---|---|---|---|---|
| `ATR_P50` gate | `daily_features` / canonical filter registry | `daily_features.atr_20_pct (prior-only rolling percentile)` | `STARTUP` | CLEAR | `Phase A A4 marked ATR_P50 timing-valid for the live six-lane book.` |
| max_orb_size_pts overlay | prop_profiles + daily_features | `orb_SINGAPORE_OPEN_size` with cap `37.8` | ORB_FORMATION | CLEAR | Phase A snapshot 5e768af8 separated the execution overlay and measured the cap pass-rate for this lane. |

## Canonical-window verification

- [x] The Phase A snapshot already cleared the active six-lane book as timing-valid in current framing.
- [x] This lane does not use post-break or outcome-derived filter inputs.
- [x] No post-break-role reframing is required.

## Verdict

- [x] CLEAR — timing-valid under the binding Phase A snapshot.

## Literature citation

- `docs/institutional/literature/chan_2013_ch1_backtesting_lookahead.md` p.4
- `docs/audit/2026-04-21-reset-snapshot.md` A4

## Authored by / committed

- Author: `Codex`
- Commit SHA of candidate script: `dfb1bbab`
- Commit SHA of this certificate: `dfb1bbab`
