# Stage: C11 ORB-cap remediation research — cap ALONE clears C11 at the $1,800 belt — HOLD live

task: Pre-registered ORB-cap remediation for topstep_50k_mnq_auto C11. A tighter
  ORB-size cap (cap_x0.80 or cap_x0.75) clears the C11 drawdown-magnitude gate at
  the current $1,800 strict-DD budget WITHOUT a stop reduction. C11 stays a measured
  NO-GO until the cap value is wired, the bracket-parity audit (9b3fc530) closes,
  and operator GO is given. Do not arm live.
mode: PARKED (cap-value selection + wiring + audit required, not yet implementation)
created: 2026-06-04 ~10:00 local
session_origin: session "committing" (edf38a72)
corrected: 2026-06-05 — the prior "cap-emulated 0.75 stop → $2,038, cap insufficient,
  needs stop≤0.50" claim was FALSE. $2,038.84 is the UNCAPPED baseline DD, not a
  capped result; it was mislabeled. A fresh canonical-loader run (commit 9429c540)
  proves cap ALONE clears C11 at the $1,800 belt. Numbers below are the truth.

## CANONICAL ANSWER (fresh canonical-loader run, commit 9429c540, 2026-06-05)

Harness `c11_unified_levers.py` (read-only, parity-anchored to the canonical
$2,038.84 baseline exactly). Profile topstep_50k_mnq_auto, n_days 2049, 90d horizon,
stop 0.75 canonical, lanes 49.8 / 143.2 / 44.2 pts.

| Lever | max-90d DD | vs $1,800 belt | edge% | WF |
|---|---|---|---|---|
| **L0 baseline (UNCAPPED, stop 0.75)** | **$2,038.84** | FAIL by $238.84 | 100% | 4/5 |
| **cap_x0.80 (BEST current 50k fix)** | **$1,594.03** | PASS ($206 room) | **97.5%** | 5/5 |
| cap_x0.75 (works, conservative) | $1,535.22 | PASS ($265 room) | 91.7% | 5/5 |
| cap_x0.60 | $1,548.19 | PASS | 84.0% | 5/5 |

- **Budget = $1,800** = 0.90 × $2,000 Topstep MLL (`STRICT_DD_BUDGET_FRACTION_EXPRESS`
  = 0.90, account_survival.py). The earlier **$1,600** (0.80 fraction) is RETIRED
  (operator raised it 2026-06-04).
- The earlier "$2,038 capped at 0.75 / cap insufficient / needs stop≤0.50" was a
  **labeling error**: $2,038.84 is the baseline with NO cap applied. Applying any of
  cap_x0.60 / 0.75 / 0.78 / 0.80 brings the strict 90d DD ≤ $1,800. A stop reduction
  is NOT required to clear C11.

## Context / why

Capping ORB size removes the fat-loser tail trades, keeps ~90% of trades, and
RAISES ExpR (the wide-stop tail is negative-EV). At the current $1,800 belt the cap
is both edge-positive AND sufficient to clear the C11 drawdown-magnitude gate on its
own.

C11 has TWO gates: (a) MC 90-day operational survival ≥70% AND (b) strict observed
90d DD ≤ budget with zero historical DLL breach days. cap_x0.80 clears BOTH:
op-survival passes, breach days = 0, strict 90d DD $1,594 ≤ $1,800.

GROUND TRUTH (read live, NOT from memory):
- The cap is NOT yet in the deployed lane registry. Live values are:
  - COMEX_SETTLE  max_orb_size_pts = 49.8
  - TOKYO_OPEN    max_orb_size_pts = 44.2
  - US_DATA_1000  max_orb_size_pts = 143.2   ← the lane a tighter cap targets
  (source: trading_app.prop_profiles.get_lane_registry(profile_id='topstep_50k_mnq_auto'))
- So the cap decision lives in analysis/memory only — neither the live orchestrator
  nor the survival model reflects a tightened cap yet (registry still holds 143.2).

## What was DONE (committed + pushed)

- `9b3fc530` "fix(c11): align replay and live risk parity" — pushed to origin/main.
  Threads max_orb_size_pts into the offline account_survival replay so it skips
  trades whose risk >= the lane ORB cap, matching the live ORB_CAP_SKIP the
  session_orchestrator already enforces. Correctness plumbing only.
- Effect: the survival model now HONORS whatever cap the registry holds — but the
  registry still holds 143.2 (the cap factor above is applied in-harness, not yet
  in the live registry).

## NEXT STEPS (cap-value selection + wiring + audit — NOT "go live ASAP")

1. LOCK the cap value. cap_x0.80 is the recommended pick for the $50k path
   (best edge retention at 97.5%, $206 headroom). Translate the factor into the
   per-lane registry value (US_DATA_1000 143.2 × 0.80 ≈ 114.6 pts; verify per lane).
2. PRE-REGISTER the cap remediation (RULE 10 / overfit warning): define the cap and
   the acceptance gates BEFORE wiring — no post-hoc fitting.
3. DSR / era-split robustness checks on the chosen cap.
4. Adversarial-audit gate (capital-path change) AND the independent audit of the
   bracket-risk-parity fix (9b3fc530) — required by adversarial-audit-gate.md; that
   closeout (2026-06-03-bracket-risk-parity-closeout.md line 47) had no independent
   reviewer yet. Run BEFORE any live arming.
5. ONLY after 1-4 pass does the live path open. Live arming is a separate
   operator-GO gate.

NOTE: the cap value above is a research result, not yet a deployed registry value.
The registry currently holds per-session p90 (US_DATA_1000 = 143.2). Step 1 must
establish and lock the concrete deployable cap value.

## Account-size alternative (separate fork)

The $100k path (topstep_100k_mnq_auto, MLL $3,000 → budget 0.90×$3,000 = $2,700)
clears the FULL UNCAPPED book ($2,038.84) with ~$661 room and ZERO edge loss — but
that profile does not exist yet and must be created (capital path, separate stage).
See memory project_c11_deploy_decision_100k_now_bridge_later_2026_06_05.

## scope_lock
- docs/runtime/lane_allocation/topstep_50k_mnq_auto.json  (or prop_profiles lane source — verify)
- (re-run only, no edit) trading_app/account_survival.py
- (re-run only) trading_app/sr_monitor.py

## Blast Radius
- Lane allocation config is the value the LIVE session_orchestrator reads at runtime
  (get_lane_registry → max_orb_size_pts → ORB_CAP_SKIP). Changing it alters which
  trades the live bot takes — CAPITAL PATH. Re-run account_survival + sr_monitor
  after. The parity commit 9b3fc530 (already on main) ensures the survival model will
  correctly reflect the new cap.
- Reads: gold.db (read-only, via account_survival replay). Writes: lane config +
  regenerated survival/SR state JSONs.

## Recovery anchors (nothing lost)
- branch c11-orb-cap-preserve-2026-06-04 (48071955) — 17 files: the full in-flight
  C11 + bracket work INCLUDING the c11-remediation/attribution audit docs.
- Parity already landed: origin/main 9b3fc530 (max_orb_size_pts wired into survival model).
- Canonical answer: memory project_c11_CANONICAL_answer_2026_06_05.

## HARD CONSTRAINTS
- C11 = measured NO-GO until the cap value is locked + wired, the bracket-parity
  audit (9b3fc530) closes with an independent reviewer, and operator GO is given.
  Cap ALONE clears the C11 DD gate at $1,800 — but wiring + audit + GO are still owed.
  Do not arm live before that.
- The cap is the EARNINGS-relevant, survival-relevant knob (clears op-survival +
  breach-days AND the DD-magnitude gate). The $1,800 budget = 0.90×$2,000 MLL; the
  old $1,600 (0.80) is retired/advisory.
- Adversarial-audit gate applies (capital-path remediation + the bracket-parity fix
  9b3fc530 still owes an independent reviewer).
