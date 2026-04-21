# G1 — Timing-Validity Certificate

**Candidate:** ________________________
**Date authored:** ____________
**Pre-reg:** `docs/audit/hypotheses/________.yaml`
**Entry model:** ____ (E1 / E2 / E2-CB2 / ...)
**Decision-time bar:** ____________________________________________

---

## Purpose

Prove every variable the candidate's edge uses is knowable BEFORE the entry-model decision-time bar. Prevents the Chan 2013 Ch 1 p.4 look-ahead class of error (`docs/institutional/literature/chan_2013_ch1_backtesting_lookahead.md`).

## Ban-list cross-check (`.claude/rules/daily-features-joins.md` § Look-Ahead Columns)

Any variable below is AUTOMATIC FAIL unless the entry model is demonstrably post-bar:

- [ ] `double_break` — uses full-session forward scan
- [ ] `took_pdh_before_1000` / `took_pdl_before_1000` — 1000 window end
- [ ] `overnight_range_pct` — ratio requires overnight close
- [ ] `break_dir`, `break_ts`, `break_delay_min` — break-bar derived
- [ ] `outcome`, `mae_r`, `mfe_r`, `pnl_r` — trade-outcome
- [ ] any `*_volume`, `*_break_bar_volume`, `rel_vol_{session}` — **post-break-bar**, E2-incompatible as pre-entry filter

Mark each with CLEAR / PRESENT. Any PRESENT → must be justified by entry-model timing OR reframed as post-break role.

## Variable inventory

| Variable | Source table | Source column(s) | Source bar / window | Knowable at decision-time bar? | Evidence (query + output) |
|---|---|---|---|---|---|
| (list every variable the edge uses) | | | | CLEAR / POST-BREAK / VIOLATION | |
| | | | | | |

## Canonical-window verification

- [ ] All window-boundary computations use `pipeline.dst.orb_utc_window(trading_day, orb_label, orb_minutes)` per 2026-04-07 postmortem (`docs/postmortems/2026-04-07-e2-canonical-window-fix.md`).
- [ ] No derivation from `break_delay_min` as a proxy for window timing.
- [ ] No fallback to `break_ts` as a timestamp primary key.

Evidence (grep output showing canonical usage in the candidate's script):

```
$ grep -n "orb_utc_window\|break_delay_min\|break_ts" <script>
<paste output>
```

## Post-break-role reframing (if any variable is POST-BREAK)

If the variable is post-break, the candidate is NOT a pre-entry filter. Reframe to one of the mechanism_priors.md §4 post-break roles:

- [ ] R3 POSITION SIZE (at entry, post-break bar close is knowable)
- [ ] R4 EXIT CRITERION (post-entry, fine — outside G1 scope)
- [ ] R5 ALLOCATOR (session/day-level weight, untouched by per-trade timing)
- [ ] R6 ENTRY-MODEL SWITCH (e.g., different entry model where timing works)
- [ ] R7 CONFLUENCE WEIGHT (composite signal, scored at entry)
- [ ] R8 POST-BREAK CONDITIONER (new role class — evaluates at entry after break bar)

Selected reframing: ______________________

## Verdict

- [ ] CLEAR — all variables knowable at decision-time bar, no ban-list hits, canonical window API used.
- [ ] POST-BREAK REFRAMED — at least one variable is post-break; candidate moves to the selected role above, NOT as a pre-entry filter.
- [ ] VIOLATION — variable is look-ahead with no valid reframing. Candidate is KILLED pending pre-reg rewrite.

## Failure disposition

VIOLATION → candidate cannot advance. Pre-reg must be rewritten specifying a different variable set or a different entry model. The failure is logged in `.claude/rules/backtesting-methodology-failure-log.md`.

POST-BREAK REFRAMED → candidate advances but as the selected post-break role, not as pre-entry filter. A fresh pre-reg is required for that role (not the original pre-reg).

## Literature citation

- `docs/institutional/literature/chan_2013_ch1_backtesting_lookahead.md` p.4: "look-ahead bias means that your backtest program is using tomorrow's prices to determine today's trading signals."

## Authored by / committed

- Author: ____________________________
- Commit SHA of candidate's script: ________________
- Commit SHA of this certificate: ________________
