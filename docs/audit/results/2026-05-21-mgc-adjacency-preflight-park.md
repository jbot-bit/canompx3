# MGC Allocation-Intel Adjacency — Pre-Flight PARK

**Date:** 2026-05-21
**Author:** Claude Code session (allocation-intel adjacency follow-up)
**Verdict:** **PARK — both pre-regs withdrawn before lock**
**Upstream trigger:** `scripts/tools/allocation_intel.py` § 4 HOT-undermapped scan flagged MGC NYSE_OPEN + MGC SINGAPORE_OPEN as `validated_n=0` cells with 6-month seeds at `avg_r≈+0.19, WR≈61%`.

## What this audit actually is

A **pre-lock sanity replay** of the proposed Pathway B K=1 pre-regs on the canonical IS-only universe (`trading_day >= WF_START_OVERRIDE['MGC']` = 2022-01-01, `< HOLDOUT_SACRED_FROM` = 2026-01-01). The audit-revised plan accepted that the seed was partially contaminated by post-2026-01-01 holdout data and asserted the strict-unlock runner would replay clean. This pre-flight asks whether the strict-IS replay produces a t-stat anywhere near the Chordia hurdle before committing K=1 of MinBTL budget.

## Stage 1 — NO-GO scan (executed)

| Cell | Verdict | Source |
|---|---|---|
| MGC SINGAPORE_OPEN | **HARD KILL** | `pipeline/init_db.py:17` ("MGC excluded (74% double-break)"), `trading_app/config.py:4296-4302` `EXCLUDED_FROM_FITNESS = {"MGC": {"SINGAPORE_OPEN"}}` revalidated 2026-03-10 for E2 |
| MGC SINGAPORE_OPEN historical | All 6 prior NODBL-validated strategies were artifacts | `trading_app/config.py:3996-3999` "double_break column is LOOK-AHEAD — All 6 validated strategies using NODBL were artifacts of hindsight bias" |
| MGC NYSE_OPEN | No explicit NO-GO in BLUEPRINT or audit results | Cleared NO-GO scan, proceeded to IS pre-flight |

SINGAPORE_OPEN pre-reg dropped at Stage 1. NYSE_OPEN proceeded to Stage 1.5 pre-flight IS replay.

## Stage 1.5 — Strict IS replay (executed, definitive)

**Cell sweep:** MGC NYSE_OPEN E2 O5 CB1, RR ∈ {1.0, 1.5, 2.0, 2.5, 3.0} × ORB_size_threshold ∈ {0, 3, 4, 5, 6, 8}.
**Window:** 2022-01-01 ≤ trading_day < 2026-01-01 (HOLDOUT_SACRED_FROM).
**Universe:** `orb_outcomes` JOIN `daily_features` on canonical keys. No filter beyond `orb_NYSE_OPEN_size >= threshold`. `pnl_r IS NOT NULL`. Naive (non-clustered) t.

```
RR   ORB≥   N_IS    ExpR     WR%    t_naive   OOS_N  OOS_ExpR
1.0  0      918    -0.0400   55.9   -1.420    81     +0.2406
1.0  3      585    +0.0083   56.4   +0.226    80     +0.2353  ← MAX t across grid
1.0  4      373    +0.0013   54.7   +0.027    79     +0.2509  ← plan's target cell
1.0  5      240    +0.0023   53.8   +0.039    79     +0.2509
1.0  6      159    -0.0127   52.2   -0.171    78     +0.2670
1.0  8       88    +0.0031   52.3   +0.030    76     +0.2516
1.5+        ...    all negative IS at every ORB threshold ...
```

**Read:**

1. Max IS naive-t across the entire (RR, ORB) plane = **+0.226** (RR1.0, ORB≥3, N=585). Theory-grant Chordia hurdle = **3.00**. Cell is ~**13× below hurdle** at its strict-IS best.
2. Plan-targeted ORB_G4 RR1.0 cell: **IS naive-t = +0.027, IS ExpR = +0.001**. Not edge — coin flip.
3. **OOS slice (N≈76-81)** is uniformly +0.19 to +0.31 at every RR × ORB threshold. The entire positive signal that motivated this adjacency lives in the 5-month sacred-holdout window.
4. RR ≥ 1.5 is **negative IS at every ORB threshold**. Higher-RR seed cells flagged in the plan (RR2.5/3.0/4.0 with avg_r>+0.18 on 6-month) are ARITHMETIC_ONLY-on-holdout-only.

**Mechanism for the IS-OOS asymmetry:** likely 2026 H1 regime shift on MGC NYSE_OPEN. Distinct from any tradeable edge. The 6-month rolling window in `allocation_intel.py` produces positive seeds for cells like this because ~5 of the 6 months are post-2026-01-01.

## Stage 1.5 verdict — PARK

- **MGC NYSE_OPEN E2 RR1.0 CB1 ORB_G4 → PARK** (do not pre-reg, do not run).
  - Justification: pre-flight IS naive-t = +0.027 vs theory-grant hurdle 3.00. Running the strict-unlock would burn one K_effective MinBTL trial on a verdict already determinable from canonical layers.
  - Per `feedback_oos_does_not_accrue_holdout_is_frozen.md`: holdout is frozen; structurally-underpowered/IS-empty cells should be rejected at design time, not by waiting.
  - Per `feedback_handoff_pointer_overridden_by_audit_results.md`: this PARK record overrides any future "MGC NYSE_OPEN looks promising in allocation_intel" pointer that re-surfaces.

- **MGC SINGAPORE_OPEN → existing NO-GO upheld** (74% double-break, canonical exclusion since 2026-03-10).

## Reopen criteria

Re-test only when ANY of:
1. Three or more additional MGC NYSE_OPEN E2 trading years accumulate (≈ 2029), allowing a fresh IS window that includes the 2026 H1 regime.
2. A literature-grounded mechanism for **why MGC NYSE_OPEN should differ from MGC LONDON_METALS/COMEX_SETTLE** at the breakout level is articulated (Chan Ch 7 NYSE-cash-open momentum is the citation pattern, but it's been negative-IS on MGC for 4 years — the mechanism does not transfer cleanly from MNQ to MGC).
3. A directional asymmetry test (long-only or short-only) on MGC NYSE_OPEN E2 finds IS t ≥ 2.5 with N ≥ 100 on either side **before** any pre-reg lock. The pooled-direction sweep above hides any such asymmetry; do not pre-reg pooled until per-direction IS clears 2.5.

## What this saves

- 1 × MinBTL trial that would have FAIL_CHORDIA at t≈0 vs 3.00.
- 1 × chordia_audit_log.yaml entry that would have read `verdict: KILL_LANE_FAIL_CHORDIA` for a 4-year-known-flat cell.
- Slot in the `lhp` open-hypothesis queue.
- One iteration of the project's "trying a different filter on the same dead seed" anti-pattern (Sin #2 data-snooping).

## Files NOT created (explicit scope_lock close-out)

- `docs/audit/hypotheses/2026-05-21-mgc-nyseopen-e2-rr10-o5-orb-g4-v1.yaml` — withdrawn
- `docs/audit/hypotheses/2026-05-21-mgc-singaporeopen-e2-rr10-o5-orb-g4-v1.yaml` — withdrawn (NO-GO)
- `docs/runtime/chordia_audit_log.yaml` — no append
- No `validated_setups` mutation, no allocator mutation

## Allocation-intel feedback loop

`scripts/tools/allocation_intel.py` § 4 will continue to flag MGC NYSE_OPEN / MGC SINGAPORE_OPEN as adjacency candidates as long as its 6-month seed window straddles 2026-01-01. Two paths forward:

- **Suppression list:** add `(MGC, NYSE_OPEN)` and `(MGC, SINGAPORE_OPEN)` to an exclusion set the script reads, citing this PARK record + the SINGAPORE_OPEN canonical exclusion.
- **Window strictness:** narrow `allocation_intel.py`'s seed window to IS-only (`trading_day < HOLDOUT_SACRED_FROM`). This is the safer global fix — the script currently surfaces any cell whose holdout is positive, which is exactly the wrong attractor.

Either is a separate change. Out of scope for this PARK record.

## Cross-refs

- Upstream plan (revised): inline in conversation 2026-05-21
- Independent audit that pre-empted Pathway A → Pathway B revision: inline 2026-05-21
- `docs/STRATEGY_BLUEPRINT.md` § 5 NO-GO Registry (consider adding MGC NYSE_OPEN E2 entry pointing here)
- `feedback_oos_does_not_accrue_holdout_is_frozen.md` — same anti-pattern
- `feedback_default_cross_session_scope.md` — same anti-pattern
- `feedback_handoff_pointer_overridden_by_audit_results.md` — invocation rule
