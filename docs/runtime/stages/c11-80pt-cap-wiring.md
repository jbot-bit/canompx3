# Stage: C11 ORB-cap remediation research (cap is necessary, NOT sufficient) — HOLD live

task: Pre-registered cap+stop remediation research for topstep_50k_mnq_auto C11.
  The ~80pt ORB-size cap alone does NOT clear C11 — it clears the operational-survival
  and breach-day gates but NOT the drawdown-magnitude gate. This is research (pre-reg
  cap+stop), not config wiring. C11 stays a measured NO-GO. Do not arm live.
mode: PARKED (research-required, not implementation)
created: 2026-06-04 ~10:00 local
session_origin: session "committing" (edf38a72)
corrected: 2026-06-04 — original "wire 80pt → C11 passes → live" premise is FALSE
  (see CORRECTED NEXT STEPS); cap-emulated 0.75 strict 90d DD = $2,038 > $1,600
  budget AND > full $2,000 Topstep MLL.

## Context / why

Operator established last night that capping ORB size at ~80pts removes the
fat-loser tail trades, keeps ~90% of trades, and RAISES ExpR +0.109 → +0.130
(the wide-stop tail is negative-EV). The cap is genuinely useful — but it does
NOT by itself make the lane SURVIVE C11.

CORRECTION (verified against docs/audit/results/2026-06-03-c11-remediation-*.md
lines 28-37, 100-110): C11 has TWO gates — (a) MC 90-day operational survival
≥70% AND (b) strict observed 90d DD ≤ 80% of MLL ($1,600) with zero historical
DLL breach days. The cap fixes (a) and the breach-day count, but NOT (b):
cap-emulated at the current 0.75 stop, strict observed max 90d DD = $2,038 —
which fails the $1,600 budget AND exceeds even the full $2,000 Topstep MLL.
Only a TIGHTER stop (≤0.50) closes the DD gap (cap+0.50 → $1,142 ✓). The cap is
necessary but not sufficient; cap+stop is a pre-registered remediation (RULE 10),
not a config tweak. (1.0 stop is the WRONG direction — wider stop → larger
per-trade loss → DD goes ABOVE $2,038; 1.0 is also the self-funded default per
prop_profiles.py:97, a different account class, not this Topstep 50K path.)

GROUND TRUTH (read live this session, NOT from memory):
- The 80pt cap is NOT in the deployed lane registry. Live values are:
  - COMEX_SETTLE  max_orb_size_pts = 49.8
  - TOKYO_OPEN    max_orb_size_pts = 44.2
  - US_DATA_1000  max_orb_size_pts = 143.2   ← this is the one the 80pt cap targets
  (source: trading_app.prop_profiles.get_lane_registry(profile_id='topstep_50k_mnq_auto'))
- So the 80pt decision lives in analysis/memory only — neither the live
  orchestrator nor the survival model reflects it yet.

## What was DONE this session (committed + pushed)

- `9b3fc530` "fix(c11): align replay and live risk parity" — pushed to origin/main.
  Threads max_orb_size_pts into the offline account_survival replay so it skips
  trades whose risk >= the lane ORB cap, matching the live ORB_CAP_SKIP the
  session_orchestrator already enforces. Correctness plumbing only.
- Effect: the survival model now HONORS whatever cap the registry holds — but
  the registry still holds 143.2, not 80. So C11 math still uses 143.2.

## CORRECTED NEXT STEPS (research-required — NOT config wiring)

The original "wire the cap → re-run → live path opens" plan is FALSE: wiring the
cap at the current 0.75 stop leaves strict 90d DD = $2,038, still failing gate (b).
Wiring alone does not open the live path. The real path to GO is a pre-registered
cap+stop remediation, then an independent live-order-path audit:

1. PRE-REGISTER the cap+stop remediation (RULE 10 / overfit warning): the cap is
   one lever, the stop reduction (≤0.50) is the second. Define both, plus the
   acceptance gates, BEFORE running — no post-hoc fitting.
2. Re-run python -m trading_app.account_survival --profile topstep_50k_mnq_auto
   under the pre-registered cap+stop≤0.50; confirm BOTH gates: MC op-survival
   ≥70% AND strict observed 90d DD ≤ $1,600 with zero DLL breach days.
3. DSR / era-split robustness checks on the remediated configuration.
4. Adversarial-audit gate (capital-path change) AND independent audit of the
   bracket-risk-parity fix (9b3fc530) — required by adversarial-audit-gate.md;
   that closeout (2026-06-03-bracket-risk-parity-closeout.md line 47) had no
   independent reviewer yet. Run BEFORE any live arming.
5. ONLY after 1-4 pass does the live path open. Live arming is a separate
   operator-GO gate. None of this is "go live ASAP" — it is a research session.

NOTE: there is no settled deployable "80pt" value. The registry holds per-session
p90 (US_DATA_1000 = 143.2). The audits emulated caps generically; no specific cap
number was locked. Step 1 of any remediation must establish the cap value itself.

## scope_lock
- docs/runtime/lane_allocation/topstep_50k_mnq_auto.json  (or prop_profiles lane source — verify)
- (re-run only, no edit) trading_app/account_survival.py
- (re-run only) trading_app/sr_monitor.py

## Blast Radius
- Lane allocation config is the value the LIVE session_orchestrator reads at
  runtime (get_lane_registry → max_orb_size_pts → ORB_CAP_SKIP). Changing it
  alters which trades the live bot takes — CAPITAL PATH. Re-run account_survival
  + sr_monitor after. The parity commit 9b3fc530 (already on main) ensures the
  survival model will correctly reflect the new cap.
- Reads: gold.db (read-only, via account_survival replay). Writes: lane config +
  regenerated survival/SR state JSONs.

## Recovery anchors (nothing lost)
- branch c11-orb-cap-preserve-2026-06-04 (48071955) — 17 files: the full
  in-flight C11 + bracket work INCLUDING the c11-remediation/attribution audit
  docs that have the 80pt analysis. NOT on main; pull the cap value from here.
- Parity already landed: origin/main 9b3fc530 (max_orb_size_pts wired into survival model).

## HARD CONSTRAINTS
- C11 = measured NO-GO. The cap ALONE does not clear it (strict 90d DD $2,038 >
  $1,600 and > $2,000 MLL at 0.75 stop). GO requires a pre-registered cap+stop≤0.50
  remediation that passes BOTH C11 gates, plus an independent live-path audit. Do
  not arm live before that.
- The cap is the EARNINGS-relevant, survival-relevant knob (clears op-survival +
  breach-days); it is NOT the arbitrary $1,600 strict-budget knob (operator
  disowned that — advisory now). But honest math: at 0.75 the DD fails even the
  full $2,000 MLL, so the $1,600-vs-higher debate is a red herring — the stop is
  the binding lever, not the budget fraction.
- Adversarial-audit gate applies (capital-path remediation + the bracket-parity
  fix 9b3fc530 still owes an independent reviewer).
