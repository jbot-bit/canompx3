# Strict C11 remediation pass - topstep_50k_mnq_auto

Date: 2026-06-03
Profile: `topstep_50k_mnq_auto`
Verdict: `NO-GO_CURRENT_PROFILE`
Promotion ranking: not performed, because canonical Criterion 11 did not clear.

## Evidence sources

- Current Plan v2 artifact: `docs/audit/results/2026-06-03-plan-v2-current-execution.md`.
- Fresh command: `python -m trading_app.account_survival --profile topstep_50k_mnq_auto --no-write-state`.
- Canonical DB: `C:\Users\joshd\canompx3\gold.db`.
- Criterion 11 authority: `docs/institutional/pre_registered_criteria.md` requires 10,000 Monte Carlo paths and 90-day account survival >= 70% before funded deployment; strict repo gate additionally requires zero historical daily-loss breach days and 90-day observed drawdown <= 80% of account max loss.
- Literature/doctrine grounding: `docs/institutional/literature/lopez_de_prado_2020_ml_for_asset_managers.md` supports Monte Carlo/resampling for generalization under finite data; `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md` grounds the small-account sizing problem and warns that prop trailing DD forces low effective risk.
- Topstep current docs checked 2026-06-03:
  - [Maximum Loss Limit](https://help.topstep.com/en/articles/8284204-what-is-the-maximum-loss-limit): 50K MLL is $2,000; XFA starts at $0 and trails upward.
  - [Daily Loss Limit](https://help.topstep.com/en/articles/10490293-daily-loss-limit-in-the-trading-combine-and-express-funded-account): DLL is optional in Trading Combine/XFA, fixed checkout DLL for 50K is $1,000, and triggering it is not a rule violation.

## Current live repo truth

Git state is dirty and behind/ahead; this is not clean-main proof.

Current fresh C11 command exited non-zero because the gate failed:

| Metric | Value |
|---|---:|
| source days | 2,048, 2019-05-31 to 2026-06-03 |
| Monte Carlo paths | 10,000 |
| operational pass | 73.26% |
| daily-loss breach probability | 26.45% |
| trailing-DD breach probability | 0.29% |
| scaling breach probability | 0.00% |
| strict DD budget | $1,600 |
| observed max 90d DD | $2,788.33 |
| historical DLL breach days | 7 |

Historical DLL breach days under current canonical replay: `2022-05-12`, `2025-04-04`, `2025-04-07`, `2025-04-09`, `2025-04-23`, `2026-02-20`, `2026-05-19`.

DB freshness check:

| Table | Count | Min day | Max day |
|---|---:|---|---|
| `orb_outcomes` | 8,949,570 | 2010-06-07 | 2026-05-31 |
| `daily_features` | 35,424 | 2010-06-07 | 2026-06-01 |
| `validated_setups` | 871 | 2025-11-21 | 2025-12-31 |

## Blocker taxonomy

| Blocker | Label | Evidence | Decision |
|---|---|---|---|
| Current profile C11 | MEASURED | Fresh canonical C11 fails: strict gate false, 7 DLL breach days, max 90d DD $2,788.33 > $1,600. | `NO-GO_CURRENT_PROFILE` |
| Account MLL threshold | MEASURED | Repo Topstep 50K max loss is $2,000; official Topstep docs also show 50K MLL $2,000. | Not the cause. |
| Daily-loss threshold | MEASURED | Profile uses internal `daily_loss_dollars=450`; official Topstep 50K checkout DLL is $1,000 and DLL is optional for XFA. Raising/removing DLL improves operational survival, but strict DD still fails. | Contributing blocker, not sufficient alone. |
| Lane composition | MEASURED | Current 3-lane book has 7 DLL breach days and max90 $2,788.33. Diagnostic one/two-lane subsets were measured only as blocker decomposition, not promotion candidates. | Current composition is unsafe under current replay. |
| Runtime ORB cap missing from C11 replay | MEASURED | `SessionOrchestrator` enforces `ORB_CAP_SKIP`; `account_survival.py` replay applies stop multiplier but does not filter by `max_orb_size_pts`. | Stale/misaligned survival-input blocker. |
| Current cap-emulated replay | MEASURED | Emulating ORB caps removes historical DLL breach days and raises operational pass to 99.72%, but max90 remains $2,038.84 > $1,600. | Still no canonical clearance. |
| Stop sizing remediation | MEASURED, not implemented | Cap-emulated `all_stop=0.5` diagnostic clears strict C11: op 99.97%, 0 DLL breach days, max90 $1,141.88 <= $1,600. | Candidate remediation path, not a promotion decision. |
| Profile thresholds stale | INFERRED | `$450` DLL is stricter than current Topstep XFA docs, but it may be intentional internal belt. | Needs owner decision before changing. |
| Data stale as primary cause | UNSUPPORTED | DB is present and current enough for this pass; current state envelope is valid. | Not the primary cause found. |
| Real capital no-go | MEASURED | Current canonical profile and C11 command fail. | Block live/funded start for this profile. |

## Disconfirming checks

1. **Maybe it is only the daily-loss belt?**
   Disconfirmed. Removing DLL entirely gives operational pass 95.21% and 0 historical DLL breach days, but observed max90 remains $2,788.33 > $1,600.

2. **Maybe Topstep 50K account thresholds are wrong?**
   Mostly disconfirmed. Official docs match repo MLL at $2,000. The `$450` DLL is internal/profile-specific, not current Topstep 50K MLL.

3. **Maybe C11 fails only because survival ignores runtime ORB caps?**
   Partly disconfirmed. Cap-emulated replay materially improves risk, but current 0.75 stop still has max90 $2,038.84 > strict $1,600.

4. **Maybe lane composition alone fixes it?**
   Not cleared. Diagnostic one/two-lane subsets are not promotion candidates and were not ranked; every measured current subset still fails strict C11 under the `$450` DLL because at least one historical DLL breach remains.

5. **Maybe smaller account sizing can clear?**
   Supported as a remediation hypothesis only. Cap-emulated `all_stop=0.5` clears strict diagnostics, while uncapped `all_stop=0.5` still fails. This means any remediation must align canonical survival replay with live ORB caps and then retest profile sizing.

6. **Maybe ASX/allocation novelty should be scanned next?**
   Rejected. This pass did not touch ASX, allocation ranking, or new-candidate novelty. C11 remediation remains the binding problem.

## Sensitivity summary

### Daily-loss and account-threshold sensitivity

| Diagnostic | Operational pass | Historical DLL days | Max90 DD | Strict pass |
|---|---:|---:|---:|---|
| current DLL $450, MLL $2,000 | 73.26% | 7 | $2,788.33 | no |
| DLL $600 | 91.27% | 2 | $2,788.33 | no |
| DLL $750 | 95.11% | 1 | $2,788.33 | no |
| DLL $1,000 | 95.11% | 1 | $2,788.33 | no |
| DLL $2,000 | 95.11% | 1 | $2,788.33 | no |
| no DLL, MLL $2,000 | 95.21% | 0 | $2,788.33 | no |
| no DLL, hypothetical MLL $4,500 | 99.99% | 0 | $2,788.33 | yes |

The $4,500 row is a disconfirming threshold test only; it does not apply to a Topstep 50K account.

### Runtime-cap emulation

| Replay | Kept trades | ORB-cap skipped | Operational pass | Historical DLL days | Max90 DD | Strict pass |
|---|---:|---:|---:|---:|---:|---|
| canonical uncapped | 1,958 | 0 | 73.26% | 7 | $2,788.33 | no |
| cap-emulated, stop 0.75 | 1,827 | 131 | 99.72% | 0 | $2,038.84 | no |
| cap-emulated, stop 0.60 | 1,848 | 132 | 99.95% | 0 | $1,664.67 | no |
| cap-emulated, stop 0.50 | 1,861 | 133 | 99.97% | 0 | $1,141.88 | yes |
| cap-emulated, stop 0.40 | 1,876 | 136 | 100.00% | 0 | $1,528.68 | yes |
| cap-emulated, stop 0.30 | 1,890 | 140 | 100.00% | 0 | $1,299.82 | yes |
| cap-emulated, stop 0.25 | 1,894 | 141 | 100.00% | 0 | $1,409.48 | yes |

Interpretation: the current canonical failure is not pure lane edge failure and not pure Topstep threshold error. It is a combined account-risk problem: the canonical survival replay is less restrictive than live runtime because it omits ORB caps, and the current 0.75 stop is still too large for the strict 80% DD budget even after cap emulation.

## Wider remediation space

This section is deliberately broader than the measured cap-plus-stop path. None of these are promotion rankings. They are problem-solving tracks that must be evaluated against C11 before capital.

| Track | Label | Evidence | Risk / caveat | Next check |
|---|---|---|---|---|
| Align C11 with runtime ORB caps | MEASURED | Runtime has `ORB_CAP_SKIP`; cap-emulated replay removes DLL breach days but still fails current 0.75 stop. | If implemented incorrectly, C11 can become a runtime-fiction rescue. | Add canonical cap filtering to survival replay with tests proving parity with `SessionOrchestrator`. |
| Reduce stop multiplier | MEASURED | Cap-emulated `0.5` stop clears strict C11 in diagnostic replay. | Not canonical until profile and survival replay are changed; may reduce edge/alter trade distribution. | Pre-register sizing remediation; rerun C1-C12 deployability for changed profile. |
| Tighten ORB caps rather than stops | INFERRED | Oversized-risk skips materially reduce drawdown; cap thresholds are already runtime-controlled. | Lower caps can silently starve sessions or select a different distribution; needs trade-count and expectancy recheck. | Sweep cap percentiles under prereg, with minimum trade-count and era-stability floors. |
| Per-day realized dollar breaker | MEASURED capability, UNSUPPORTED C11 impact | `RiskManager` has `max_daily_loss_dollars`; HWM tracker has daily-loss kill-switch state. | Current C11 models daily loss as a breach, not a continue-after-flat operational policy; historical path impact not measured. | Add survival mode for "flat remainder of day at breaker" and compare to current breach semantics. |
| One-loss-per-day or one-trade-per-day throttle | UNSUPPORTED | Current failure days often involve multi-lane same-day losses, but no canonical throttle replay was run. | Could discard profitable recovery trades; may be operator-unfriendly across time zones. | Build a scenario replay variant that stops after first realized loss or first trade and measures C11 plus expected PnL. |
| Session scheduling / time-window sequencing | INFERRED | Breach composition includes US data, Tokyo, and COMEX legs on same historical stress days. | Could become post-hoc cherry-picking if sessions are selected from failure days. | Preregister scheduling rules based on operational availability or ex-ante session risk, not 2025-04 stress rescue. |
| Disable or shadow specific high-stress lanes | MEASURED decomposition, not ranked | Diagnostic subsets identify where breaches appear, but every measured current subset still fails strict C11 under $450 DLL. | Ranking/removal would violate the user constraint unless C11 clears and additivity is rechecked. | Treat as blocker taxonomy only; no lane promotion/removal until a prereg remediation run exists. |
| Account class change | MEASURED threshold effect, UNSUPPORTED availability | Hypothetical no-DLL, $4,500 DD clears strict diagnostics, matching a larger Topstep MLL class. | Not a 50K solution; account availability, buying power, rules, costs, and profile identity would change. | If considered, create a separate profile/account-fit pass, not a mutation of `topstep_50k_mnq_auto`. |
| Build account buffer before live automation | INFERRED | Topstep XFA MLL locks only after buffer; current strict DD budget is tight at startup. | Paper/signal does not prove funded survivability; waiting for buffer is not executable without initial risk. | Model staged activation: signal-only until live/manual cushion exists, then C11 with starting balance/buffer. |
| Daily Topstep/PDLL settings | INFERRED | Official docs say XFA DLL can be optional/manual, with fixed checkout DLL at $1,000 for 50K. | Removing internal $450 belt improves operational survival but does not clear strict DD alone. | Decide whether `$450` is internal doctrine or stale config; document before changing. |
| Circuit-breaker after drawdown streak | UNSUPPORTED | Weekly review references DD circuit breaker reporting, but C11 does not model a multi-day pause/cooldown remediation here. | Easy to overfit to April 2025 drawdown cluster. | Pre-register a cooldown rule using only ex-ante drawdown/R loss triggers and replay full history. |
| Data/model correction only | PARTLY DISCONFIRMED | DB exists and state is valid; however C11 omits live ORB cap semantics. | Treating DB freshness as the whole cause is unsupported. | Fix model-live parity first; do not claim stale DB caused the failure. |
| Accept no-go / do not remediate | MEASURED | Current canonical profile fails strict C11. | Lowest implementation risk; opportunity cost. | Keep profile blocked and move to a different bounded, preregistered account-fit problem only if requested. |

Important: these tracks are alternatives, not a grab bag. Each one needs a clear hypothesis, a replay that matches runtime semantics, and the same strict C11 gate. Combining several levers without preregistration would be post-hoc rescue.

## LIVE RISK AUDIT

Verdict: `NO-GO_CURRENT_PROFILE`.

Hard blockers:

- `BLOCKED_CRITERION_11_CURRENT_CANONICAL`: current C11 gate fails.
- `BLOCKED_SURVIVAL_REPLAY_ALIGNMENT`: live runtime ORB-cap skip is not represented in canonical C11 replay.
- `BLOCKED_PROFILE_SIZING`: current stop multiplier 0.75 does not clear strict DD budget under cap-emulated replay.

Not blockers found in this pass:

- `SCALING`: measured scaling breach probability is 0.00%.
- `DB_ABSENT`: false; DB is present.
- `ASX_OR_ALLOCATION_NOVELTY`: not touched.

Required next remediation before any promotion ranking:

1. Decide whether C11 must emulate runtime ORB caps. If yes, change `account_survival.py` to apply the same `max_orb_size_pts` skip semantics as `SessionOrchestrator`.
2. Re-run current profile C11 after that replay-alignment change.
3. In parallel with that decision, choose one bounded remediation family from the wider space above. Do not combine cap, stop, session, lane, account, and circuit-breaker changes in one rescue run.
4. Only if replay-aligned C11 still fails, pre-register the chosen remediation family for `topstep_50k_mnq_auto`. Do not silently change the profile.
5. Re-run dashboard/live start preflight and strict live-readiness after canonical C11 passes.

## Bottom line

Current C11 failure is caused by a combination of lane-book risk, profile sizing, and stale/misaligned survival replay inputs. It is not explained by stale DB data or a wrong Topstep 50K MLL threshold. The current profile remains a real capital no-go until canonical C11 is replay-aligned and the profile is retested. The strongest measured diagnostic path is live ORB-cap-aligned C11 plus reduced stop sizing, but other remediation families remain viable hypotheses and must not be collapsed into that one answer.
