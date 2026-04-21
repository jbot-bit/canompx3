<!-- BEGIN CLAUDE_TERMINAL_FOLLOW_ON_2026-04-21 -->
## Update (2026-04-21 Claude terminal follow-on — ovnrng closure + post-hoc-rejection sweep)

Authored by Claude session in `canompx3-6lane-baseline` worktree, scope-locked to
non-rel_vol territory. MES/MGC/MNQ rel_vol lineage is Codex's (above fence and
Codex's subsequent commits `336aa73c` filter-form pre-reg, `3df2acb1` MGC sizer
institutional clearance REJECT).

### Ovnrng router line — CLOSED

- Branch: `research/ovnrng-router-rolling-cv` pushed to origin. Upstream restored
  (prior `[gone]` was a local broken-ref artifact; origin always had the commit).
- Final HEAD: `631bda30`. Commits in recovery pass:
  - `4dfd3000` KILL: ovnrng allocator router — rolling 4-fold CV retracts PR #62
  - `39315b52` sweep(post-hoc-rejection)
  - `631bda30` methodology: RULE 3.4 + RULE 3.5
- Verdict: **ROUTER_BRITTLE — DEAD.** PR #62's ROUTER_HOLDS_OOS retracted. Rolling
  4-fold CV: 1/4 folds win, mean ΔSR_ann = −0.525, test-2022 ΔSR = −3.23 while
  test-2024 +1.42. Single-fold WF was a regime-selection artifact. Ablation across
  3 vol-regime binning variables: all brittle.
- No production code touched. No deployment change. 2026 OOS UNTOUCHED.

### Post-hoc-rejection sweep — material finding

Full doc: `docs/audit/results/2026-04-21-post-hoc-rejection-sweep.md` (commit
`39315b52`). Sweep audited today's non-rel_vol research branches for the same
pathology that caused PR #59 to mis-classify MGC sizer as MISCLASSIFIED when the
pre-reg had actually PASSED honestly.

**Two branches need remediation:**

1. **PR #51 + PR #50 — DSR kill is POST_HOC_REJECTION.** The PR #51 pre-reg
   (`2026-04-20-mnq-unfiltered-baseline-cross-family-v1.yaml`) locked gates
   H1/C6/C8/C9. DSR (C5) is NOT enumerated. The DSR audit (`305336f3` v1,
   `4e545950` v2) applied DSR as a hard kill, violating
   `pre_registered_criteria.md` Amendment 2.1 which explicitly downgraded DSR to
   cross-check "until N_eff is formally solved in-repo." v2's N_eff resolution
   used Bailey Appendix A.3 pairwise correlation, not the ONC clustering that
   Amendment 2.1's adversarial-review prerequisite requires.
   - Canonical data verified: MNQ 15m RR=1.0 NYSE_OPEN IS N=1715 ExpR=+0.0974
     t=+4.17 — H1 gate PASSES (stronger in canonical rerun than PR #51's N=1545).
   - **Remediation:** restore 5 CANDIDATE_READYs (+ implicit 2 from PR #50) with
     DSR-PENDING status. Open N_eff-resolution workstream (ONC clustering per LdP
     2020). Do NOT auto-promote — Phase 0 promotion still requires C5 (binding
     post-N_eff-resolution), C11 account-death MC, C12 SR-monitor.

2. **RULE 14 retroactive heterogeneity audit — BORDERLINE_POST_HOC.** The
   ≥25%-cell-flip-is-heterogeneity-artefact rule is mechanistically sound and now
   a permanent institutional rule going forward. But retroactive application to
   prior findings introduced a criterion not in their pre-regs.
   - **Remediation:** findings labelled HETEROGENEOUS by RULE 14 should be
     downgraded to CONDITIONAL on surviving homogeneous sub-lanes, not marked
     DEAD. Preserves legitimate signal at the honest unit of deployment.

### Other branches — held up under classification

- `research/ovnrng-router-rolling-cv` — HONEST_KILL (my own, rolling CV is LdP
  2020 canonical, not a new gate)
- `research/correction-aperture-audit-rerun` — HONEST_CORRECTION (canonical
  parser delegation bug; L2 at correct `orb_minutes=15` is +0.050R t=+1.86, NOT
  net-negative as PR #54 premised — verified against `orb_outcomes`)
- `research/mnq-30m-rr10-confirmatory` (PR #49) — PRE_REG_HONORED (RESEARCH_SURVIVOR
  parked on C8 N_OOS<50, legitimate gate failure)
- `research/nyse-open-ovnrng-fast10-correction` — HONEST_CORRECTION (original
  PR #44 used non-canonical DSR formula; scale-artifact is real mechanism flaw)
- `research/mgc-participation-monotonic-broad` — PRE_REG_HONORED (3/3
  MONOTONIC_CONFIRMED, t=+9.59/+11.80/+7.54, no post-hoc shifts)
- `research/l2-atr-p50-stability` — NO_PRE_REG (diagnostic audit; now moot,
  superseded by correction-aperture-rerun)

### Methodology updates (committed `631bda30`)

- **`.claude/rules/backtesting-methodology.md` RULE 3.4** — single-fold WF
  insufficient for router/allocator hypotheses. Require ≥3 rolling folds with
  ≥3 wins AND mean ΔSR ≥ +0.30, OR CPCV per LdP 2020 Ch 8.
- **RULE 3.5** — post-hoc criterion creep is post-hoc REJECTION (mirror of
  post-hoc rescue). Pre-reg verdict STANDS on locked gate; new criterion becomes
  its own follow-on pre-reg on fresh data. NEVER retroactively invalidate a
  legitimate pre-reg pass.
- Failure-log appended (`backtesting-methodology-failure-log.md`) with both
  2026-04-21 incidents fully cited (commits `4dfd3000` and post-hoc-rejection
  lineage).

### MNQ biphasic participation-exhaustion hypothesis — DRAFT ONLY, NOT COMMITTED

The PR #59 re-audit raw quintile table reveals MNQ is inverted-U (Q1..Q4 rising,
Q5 crash: +0.016/+0.035/+0.064/+0.142/+0.010), DISTINCT from MGC monotonic-up and
MES binary-Q5 patterns. Three pre-committed forms drafted:
- H1 exhaustion filter (skip Q5)
- H2 Q4-peak-only (trade only Q4)
- H3 curvature feature (continuous inverted-U multiplier)

Draft at `docs/audit/hypotheses/draft-2026-04-21-mnq-participation-exhaustion-v1.yaml`
(untracked per prompt instructions, status: DRAFT_PENDING_REVIEW). Fresh-OOS
accrual required post-2026-04-22 before lock — current 2026-01-01..2026-04-21
window is contaminated by the re-audit peek.

### Per-lane gold nuggets inventory (informational only)

9 lane-direction cells with N≥28 AND sizer delta ≥+0.10R (MNQ CME_PRECLOSE_long,
MGC US_DATA_1000_short, etc.). Listed in sweep doc under "future confluence-gate
candidates" — N too small for standalone pre-reg; do NOT pre-reg individually
on this (contaminated) OOS.

### Open items for next session

- N_eff resolution workstream (ONC clustering) before DSR can be binding.
- Remediation of PR #51 + PR #50 verdicts (restore to CANDIDATE_READY with
  DSR-PENDING label).
- Remediation of RULE 14 KILL verdicts (downgrade to CONDITIONAL on sub-lanes).
- MNQ biphasic pre-reg lock — pending user approval of the 3 draft forms and
  fresh-OOS start.
- User review of Codex's MGC sizer REJECT decision and filter-form pre-reg
  (`336aa73c`) — both outside this terminal's scope.
<!-- END CLAUDE_TERMINAL_FOLLOW_ON_2026-04-21 -->
