---
task: Register code-backed SR-alarm WATCH review for the deployed Tokyo lane (MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08) so topstep_50k_mnq_auto can launch live with all 3 lanes per operator decision 2026-06-10.
mode: IMPLEMENTATION
scope_lock:
  - trading_app/sr_review_registry.py
blast_radius: |
  trading_app/sr_review_registry.py — adds ONE SrAlarmReview entry (outcome=watch)
  to the SR_ALARM_REVIEWS dict. Pure data addition, no logic change.
  Consumers: lifecycle_state.py (get_sr_alarm_review at line 232) and
  deployability.py read this dict to downgrade an SR alarm from hard-block
  (sr_alarm_unreviewed) to warning (sr_alarm_watch_reviewed). Effect: the Tokyo
  lane's SR alarm becomes a watch-reviewed warning instead of a live-launch
  blocker. Reads: validated_setups (read-only, already queried for grounding).
  Writes: none to DB. Capital effect: ALLOWS the Tokyo lane to arm live —
  reviewed against WFE 0.99 / OOS-IS 110.8% / FDR-sig / N=427 deploy floors.
---

## Blast Radius

- `trading_app/sr_review_registry.py` — one new `SrAlarmReview` dict entry, data-only.
- Downstream: `lifecycle_state.py:232`, `deployability.py:155/765-767` consume it to
  flip `sr_alarm_unreviewed` (hard block) → `sr_alarm_watch_reviewed` (warning).
- Capital impact: enables the deployed Tokyo lane to arm in the live session.
  Grounded in validated_setups deploy-floor pass (queried live 2026-06-10).
- No DB writes, no schema change, no logic change.

## Grounding (queried live 2026-06-10, not memory)

Deployed lane MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08 validated_setups:
WFE=0.9891, oos_exp_r=0.2257, expectancy_r=0.2037 (OOS/IS=110.8%),
fdr_significant=True (adj_p=0.007606), p=0.000362, N=427, Sharpe=0.1726.
SR alarm path-real: stream N=24, SR=45.22 > thr=31.96, recent_10_mean_r=-0.376.
Clears L3/L6 WATCH precedent floors; tighter recheck trigger for the negative recent stream.
