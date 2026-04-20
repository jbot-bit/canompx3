# Post-Hoc Rejection Sweep — 2026-04-21

**Date:** 2026-04-21
**Auditor:** Claude terminal (6lane-baseline worktree)
**Scope:** Non-rel_vol research branches touched 2026-04-20/21. Classify each branch's verdict against its pre-reg locked gates to detect post-hoc goalpost-shifts — the same pathology that caused PR #59 to mis-classify MGC sizer as "MISCLASSIFIED" when it had actually PASSED its pre-reg honestly.

**Method:** For each branch, read pre-reg yaml + result md (if any) + subsequent audit docs. Spot-verify contested numerical claims against canonical `orb_outcomes` data (not metadata). Classify as:

- **PRE_REG_HONORED** — verdict follows the locked gates; no new criteria introduced
- **POST_HOC_REJECTION** — verdict applied criteria not in the pre-reg or added gates after seeing outcome
- **HONEST_CORRECTION** — bug/leakage fix using pre-committed institutional rules (not a new gate)
- **NO_PRE_REG** — diagnostic/exploratory audit; no gate-violation possible
- **HONEST_KILL** — pre-reg honored and kill is methodologically sound

**Canonical data sanity check (per user directive "don't trust metadata, run the numbers"):**

| Claim under test | Source | Canonical query result | Verdict |
|---|---|---|---|
| L2 MNQ SGO RR=1.5 unfilt IS is net-negative at orb_minutes=5 | PR #54 | n=1722 ExpR=-0.010 t=-0.38 | CONFIRMED |
| L2 at correct orb_minutes=15 is positive | correction-aperture-rerun | n=1720 ExpR=+0.050 t=+1.86 | CONFIRMED |
| L6 MNQ US_DATA_1000 RR=1.0 15m stronger than 5m | correction-aperture-rerun | 5m t=+3.83 vs 15m t=+4.19 | CONFIRMED |
| PR #51 MNQ 15m RR=1.0 NYSE_OPEN passes Chordia t≥3.0 | PR #51 result table | canonical rerun ExpR=+0.0974 t=+4.17 | CONFIRMED (stronger than claimed; PR #51 had N=1545 via extra script filter, canonical N=1715) |

---

## Classification table

| Branch | Verdict (as written) | Locked pre-reg gates | Post-hoc gate introduced? | Institutional status |
|---|---|---|---|---|
| research/mnq-unfiltered-baseline-cross-family-v1 (PR #51) | 5 CANDIDATE_READY, later "MISCLASSIFIED" by DSR audit | H1 (Chordia t≥3.0 + BH-FDR q<0.05) + C6 (WFE≥0.50) + C8 (OOS/IS≥0.40, N_OOS≥50) + C9 (era stability) | **YES — DSR kill applied as hard gate, violating Amendment 2.1 of `pre_registered_criteria.md`** which downgraded DSR to cross-check "until N_eff is formally solved in-repo" | **POST_HOC_REJECTION** |
| research/mnq-15m-rr10-unfiltered-lane-discovery-v1 (PR #50) | 2 CANDIDATE_READY (NYSE_OPEN, US_DATA_1000) — implicitly killed via PR #51 DSR audit rollup | H1 + C6 + C8 + C9 (same as above) | Same DSR rollup | **POST_HOC_REJECTION** (same pathology as PR #51) |
| research/mnq-30m-rr10-confirmatory (PR #49) | 2 RESEARCH_SURVIVORs — not CANDIDATE_READY (C8 N_OOS<50 fails) | H1 + C6 + C8 + C9 | None; C8 fail is pre-reg-locked | **PRE_REG_HONORED** — legitimately parked as RESEARCH_SURVIVOR awaiting N_OOS accrual |
| research/ovnrng-router-rolling-cv (this terminal) | ROUTER_BRITTLE — DEAD | None formally pre-registered; re-audit of PR #62 single-fold WF finding | Rolling 4-fold CV is canonical institutional method (LdP 2020 Ch 8), not a new gate | **HONEST_KILL** — correctly retracts PR #62's single-fold artifact |
| research/correction-aperture-audit-rerun | Supersedes PR #52 L2/L6 rows, PR #54 entirely | Canonical parser delegation rule (`research-truth-protocol.md`) | None; canonical-source-of-truth chain rule was pre-existing | **HONEST_CORRECTION** — canonical data verified L2 at correct orb_minutes=15 is positive, overturning prior "fully filter-dependent" verdict |
| research/l2-atr-p50-stability (PR #54 original, later superseded) | "HOLDING but marginal" | No pre-reg — diagnostic audit of deployed filter | N/A | **NO_PRE_REG** — now moot (superseded by correction-aperture-rerun) |
| research/nyse-open-ovnrng-fast10 correction (PR #44 → PR #47) | MISCLASSIFIED — DSR~0, scale-artifact confirmed | Original pre-reg DSR was computed with non-canonical formula (admission in correction doc) | DSR recomputed with canonical Bailey-LdP Eq.2 (validated against paper's worked example SR_0=0.1132, DSR=0.9004) | **HONEST_CORRECTION** — original was buggy formula, corrected to canonical; scale-artifact is an independent mechanism flaw |
| research/retroactive-heterogeneity-audit (RULE 14) | A2 garch_vol_pct HETEROGENEOUS, A3 volume-confirm SEMI-HETEROGENEOUS, B3 break_quality NO-GO holds | By name "retroactive" — applied new "≥25% cell flip = heterogeneity artefact" rule to prior findings | **YES, by definition** — but rule is now codified as permanent institutional rule (memory: "every pooled claim needs a per-lane breakdown") | **BORDERLINE_POST_HOC** — rule is mechanistically sound but retroactive application overshoots; recommended remediation below |
| research/mgc-participation-monotonic-broad | 3/3 MONOTONIC_CONFIRMED (MNQ t=+9.59, MES +11.80, MGC +7.54) | β₁ rank-regression pre-committed; per-cell agreement 91-100% | None | **PRE_REG_HONORED** — clean universal finding upstream of PR #48/#51 lineage |

---

## Detailed findings

### [POST_HOC_REJECTION] PR #51 DSR audit

**What the pre-reg locked (`docs/audit/hypotheses/2026-04-20-mnq-unfiltered-baseline-cross-family-v1.yaml`):**

```yaml
testing_discipline:
  K_family: 108
  BH_FDR_alpha: 0.05
  Chordia_t_threshold: 3.00
  mandatory_downstream_gates_non_waivable:
    - "C6 (WFE ≥ 0.50)"
    - "C8 (OOS ≥ 0.40 × IS AND N_OOS ≥ 50)"
    - "C9 (era stability, no year N≥50 with Net ExpR < -0.05)"
```

DSR (C5) is **not enumerated** as a non-waivable gate. The result doc's "Not done by this result" section explicitly defers C5 DSR to a separate promotion pre-reg.

**What institutional doctrine says about DSR (`docs/institutional/pre_registered_criteria.md` Amendment 2.1, 2026-04-07):**

> DSR is a **CROSS-CHECK, not a hard gate**, until `N_eff` is formally solved in-repo. DSR does NOT override BH FDR or WFE as deploy/don't-deploy switches until N_eff is resolved.

**What the DSR audit did (commits 305336f3 and 4e545950):**

- v1 computed DSR with Bailey-LdP Eq.2 (self-sanity-checked against paper's worked example — implementation correct).
- v1 found DSR range 0.0003–0.0070 for all 5 PR #51 CANDIDATE_READYs, labelled them "all FAIL Phase 0 C5 (DSR < 0.95)", marked PR #51 "MISCLASSIFICATION".
- v2 attempted to resolve the Amendment 2.1 N_eff gating condition via Bailey Appendix A.3 + Eq. 9 pairwise-correlation approach: M=105 raw, rho_hat=+0.0578, N_eff≈99. This is ONE way to resolve N_eff, but Amendment 2.1 references the adversarial-review requirement for **ONC (Optimal Number of Clusters)** per the Lopez de Prado 2020 approach — not covered by pairwise correlation averaging.

**The pathology:** a criterion that institutional doctrine explicitly classifies as cross-check-only was used as a hard kill. This is the mirror of post-hoc rescue. Pre-registration discipline (Bailey-LdP 2014 §3, Harvey-Liu 2015 §2) requires the gate to be locked BEFORE the scan. PR #51's pre-reg locked H1/C6/C8/C9. All 5 CANDIDATE_READYs PASSED those gates.

**Canonical data verification:** my rerun on canonical `orb_outcomes` confirms the ExpR and t-stat claims for the 5 CANDIDATE_READY cells (differences in N due to PR #51 script's additional filters, but direction and significance hold). The pre-reg verdict was honestly computed.

**Remediation recommendation:**

1. **Restore PR #51's 5 CANDIDATE_READY status** under the locked pre-reg. Mark DSR audit v1/v2 as "informational cross-check — DSR=0.0003–0.0070, requires formal N_eff resolution via ONC before using as deploy-blocker."
2. **Do NOT auto-promote to shadow-deploy.** CANDIDATE_READY is not deploy-eligible by itself — Phase 0 still requires C5 (DSR with resolved N_eff), C11 (account-death Monte Carlo), C12 (SR-monitor) per the pre-reg's own "Follow-on actions" section.
3. **Open a separate workstream to formally resolve N_eff** per Amendment 2.1's gating condition. Pairwise correlation (Bailey A.3) and ONC clustering (LdP 2020) should both be computed; whichever gives the more conservative N_eff is the institutional answer. Then DSR becomes a binding gate.
4. **Meanwhile: these 5 cells are RESEARCH_SURVIVOR with DSR-pending status**, not deploy-candidate and not dead.

### [BORDERLINE_POST_HOC] RULE 14 retroactive heterogeneity audit

The "pooled averages can hide opposite-sign per-lane cells" insight is mechanistically correct and is now a codified standing rule (`research-truth-protocol`, memory `feedback_per_lane_breakdown_required.md`). Going forward, every pre-reg must include a per-lane breakdown. Good.

The problem: applying a NEW rule retroactively to kill prior findings is structurally the same pathology as PR #59. When the rule was not in the original pre-reg, retroactive application is post-hoc goalpost-shift — even if the rule is sound.

**Remediation recommendation:**

- For findings the RULE 14 audit classified HETEROGENEOUS (A2 garch_vol_pct≥70, 31.5% cell flip) or SEMI-HETEROGENEOUS (A3 volume-confirm, 0 survivors at 5/12 sessions): **do NOT mark DEAD**. Mark as **CONDITIONAL** on the surviving homogeneous sub-lanes, downgrade pooled verdict, and require a pre-reg that targets only the surviving sub-lanes for any deployment step. This preserves the legitimate finding at the honest unit of deployment.
- For findings that already carried per-lane heterogeneity evidence in their original pre-reg (e.g., agreement-rate ≥ 90%): PRE_REG_HONORED, no change.
- Going forward: every new discovery pre-reg must include a per-lane heterogeneity gate (cell-flip % ≤ 25%, or pre-committed otherwise).

### [HONEST_CORRECTION] Correction-aperture-audit-rerun

**Canonical data verified the core claim.** Prior scripts (`research/audit_6lane_scale_stability.py::load_lane_universe`) re-encoded `parse_strategy_id` and hardcoded `orb_minutes=5`. Lanes L2 and L6 carry `_O15` suffix → canonical `orb_minutes=15`. The canonical parser at `trading_app/eligibility/builder.py:122-125` extracts this correctly; the research script discarded it.

| Lane | Wrong aperture (5m) | Correct aperture (15m) |
|---|---|---|
| L2 MNQ SGO RR=1.5 unfilt IS | n=1722 ExpR=-0.010 t=-0.38 | n=1720 ExpR=+0.050 t=+1.86 |
| L6 MNQ US_DATA_1000 RR=1.0 unfilt IS | n=1718 ExpR=+0.087 t=+3.83 | n=1717 ExpR=+0.097 t=+4.19 |

This is not a new gate — it's canonical-source-of-truth chain rule enforcement (pre-existing in `research-truth-protocol.md` § "Canonical filter delegation"). **Correction stands; L2 is NOT net-negative unfiltered at correct aperture.**

### [HONEST_CORRECTION] NYSE_OPEN OVNRNG_50_FAST10 (PR #44 → PR #47)

Two independent corrections, both legitimate:

1. **DSR formula bug** — PR #44 admitted to using a Mertens-approximation, not Bailey-LdP Eq.2. Corrected DSR (validated against paper worked example) collapses from claimed 0.9542 to ~0. Not a new gate; a formula bug fix.
2. **Scale-artifact** — OVNRNG_50 is an absolute-points threshold on a feature (overnight range) that grew 4.4× as MNQ tripled 2019→2026. Fire rate went 21% (selective filter) to 90% (pass-through). Mechanistically the filter in 2026 is not the same filter as in 2019. This is a real mechanism flaw, not a new statistical gate.

Both corrections are justified. Filter is dead. Scale-stability now a standing institutional rule (memory `feedback_absolute_threshold_scale_audit.md`).

---

## Per-lane gold nuggets inventory (informational — N too small for standalone pre-reg)

From the PR #59 re-audit raw per-lane heterogeneity tables (not in any deployment path, flagged for future confluence-gate candidate list):

| Lane | N_OOS | Sizer delta | Sign |
|---|---:|---:|---|
| MNQ CME_PRECLOSE_long | 28 | +0.195 | +++ |
| MNQ BRISBANE_1025_short | 32 | +0.148 | +++ |
| MNQ LONDON_METALS_long | 41 | +0.143 | +++ |
| MNQ SINGAPORE_OPEN_short | 30 | +0.126 | ++ |
| MGC US_DATA_1000_short | 31 | +0.145 | +++ |
| MGC NYSE_OPEN_short | 32 | +0.114 | ++ |
| MES SINGAPORE_OPEN_short | 32 | +0.125 | ++ |
| MES LONDON_METALS_long | 39 | +0.106 | ++ |
| MGC US_DATA_830_long | 36 | +0.106 | ++ |

**Interpretation:** these cells show the sizer effect is not uniform — it's concentrated in specific session-direction combinations. Promising as **confluence-gate candidates** (rel_vol × session × direction) in a future pre-reg, but N per cell is below the 100-trade deployable floor.

**Do NOT:** pre-reg any of these individually on this OOS (same-OOS peek via the heterogeneity table contaminates them).

---

## Summary verdict

**Two branches need remediation:**

1. **PR #51 (+ PR #50 via rollup)** — restore CANDIDATE_READY status, mark DSR as pending formal N_eff resolution, open N_eff workstream. 5 legitimate CANDIDATE_READYs are currently mis-shelved as dead.
2. **RULE 14 retroactive audit** — downgrade KILL verdicts on HETEROGENEOUS findings to CONDITIONAL on surviving sub-lanes. Do not retire the underlying signals.

**Everything else:** classification held up under canonical verification.

**The pathology we're watching for** (recurrent this week): introducing new gates between pre-reg lock and verdict call. Two distinct mechanisms cause it — (a) user-in-loop skeptical re-audit (PR #59 sizer), (b) sibling institutional audit finding new criteria (PR #51 DSR). Both deserve the same institutional response: the original pre-reg's verdict stands; the new finding becomes its own pre-registered follow-on, not a retroactive invalidation.

---

## Provenance

- Canonical data verified via direct query to `gold.db::orb_outcomes` (`pipeline.paths.GOLD_DB_PATH`).
- Institutional doctrine cited from `docs/institutional/pre_registered_criteria.md` (v3.1, Amendments 2.1–3.1).
- Read-only research. No production code touched. No deployment action recommended beyond "stop treating the 5 PR #51 cells as dead".
- 2026 OOS (Mode A sacred) UNTOUCHED.
- Peer cross-terminal: Codex is handling MES/MGC rel_vol sizer + filter-form lineage on the main worktree; this sweep is strictly scoped to non-rel_vol branches.
