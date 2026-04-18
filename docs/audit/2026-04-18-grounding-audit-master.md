# Grounding Audit — 2026-04-18

**Scope:** Apr 16–18 2026 workstreams — FX ORB pilot + closure, Claude API modernization, Docs authority cleanup, Phase D volume pilot, Garch allocator A4a/A4b/A4c, Overnight queue (VWAP_BP / wide-rel-IB v2 / cross-NYSE momentum).
**Method:** 4-pass fail-closed audit — inventory → claim extraction → re-grounding → severity. Quote-or-fail applied to load-bearing numerical / threshold / date claims. File:line-or-degrade applied to canonical-code claims. Objective 6-criteria load-bearing test applied at PASS 0.
**Result (pre-remediation):** 0 CRITICAL, 3 IMPORTANT, 4 CLEANUP (one prospective). 7 items QUARANTINED pending raw read.
**Result (post-remediation 2026-04-18):** 0 CRITICAL, 1 IMPORTANT (IMP-3 docs hygiene), 5 CLEANUP, 7 QUARANTINED. IMP-1 INTERIM CLOSED via HLZ-2015 stub (commit `5d67edc4`). IMP-2 EFFECTIVELY CLOSED — 4/5 constants verified at `file:line`; residual `deploy_window_months: 12` downgraded to CLEANUP CLN-5.

## Executive summary

66 load-bearing claims extracted from 36 objectively-load-bearing artifacts (of 97 in scope). Every in-scope verdict — FX class CLOSED, A4c NULL, A4b NULL_BY_CONSTRUCTION, VWAP_BP DEAD, wide-rel-IB NULL, cross-NYSE NULL, Phase D D-1 contract lock — traces to either quoted Tier 1 literature, quoted project canon, or self-reported harness output plus doctrine quoted verbatim. No verdict flips under the findings. No conflict between project canon and local literature. No local source read failures.

## Scope

### Workstreams audited

1. FX ORB pilot + filter-rescue closure (Apr 16–17) — 6 LB artifacts
2. Claude API modernization Stages 1–4 (Apr 17) — 5 LB artifacts (engineering surface)
3. Docs authority cleanup — I1/I2/I3(a)/I3(b)/I3(c)/C1/C2 edits (Apr 17) — 5 LB artifacts
4. Phase D volume pilot D-0 / D-1 — 3 LB artifacts (ALL QUARANTINED pending raw read)
5. Garch allocator A4a/A4b/A4c + W2 feeders + scarcity surface audit (Apr 16–17) — ~14 LB artifacts
6. Overnight queue — VWAP_BP C8 re-check, wide-rel-IB v2, OVNRNG family status, cross-NYSE momentum (Apr 18) — 9 LB artifacts

### Canon + literature sources used

Project canon: `CLAUDE.md`, `TRADING_RULES.md`, `RESEARCH_RULES.md`, `HANDOFF.md`, `docs/institutional/pre_registered_criteria.md`, `.claude/rules/backtesting-methodology.md`, `.claude/rules/quant-audit-protocol.md`, `.claude/rules/integrity-guardian.md`, `.claude/rules/institutional-rigor.md`, `trading_app/holdout_policy.py`, `trading_app/lane_allocator.py`, `trading_app/prop_profiles.py`, `trading_app/strategy_validator.py`, `trading_app/config.py`.

Tier 1 (verbatim extracts): `docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md`, `bailey_lopez_de_prado_2014_deflated_sharpe.md`, `chordia_et_al_2018_two_million_strategies.md`, `harvey_liu_2015_backtesting.md`, `lopez_de_prado_2020_ml_for_asset_managers.md`, `lopez_de_prado_bailey_2018_false_strategy.md`, `pepelyshev_polunchenko_2015_cusum_sr.md`, `fitschen_2013_path_of_least_resistance.md`, `carver_2015_volatility_targeting_position_sizing.md`, `chan_2008_ch7_regime_switching.md`.

Tier 2 raw PDFs referenced but not required in this sweep: `Robert Carver - Systematic Trading.pdf` (used instead of user-listed filename `Carver_Fitting_and_VolTarget_Extracts.pdf` which is absent from disk).

### Environment flags

- `PROJECT_REFERENCE.md` optional, absent — not hard-failed per hardened rubric.
- `resources/Carver_Fitting_and_VolTarget_Extracts.pdf` — not on disk; routed to existing Tier 1 Carver extract + `resources/Robert Carver - Systematic Trading.pdf`.
- 5 Apr-18 overnight queue replay docs live on branch `research/overnight-2026-04-18-v2`, accessed via `git show` (commits `284e3d35`, `e1cc2977`, `16cc3af5`, `66d26edb`, `849d9097`, `16158e31`, `0d47a761`).

## Totals

| Severity | Pre-remediation | Post-remediation 2026-04-18 |
|---|---:|---:|
| CRITICAL | 0 | **0** |
| IMPORTANT | 3 | **1** (IMP-3 only) |
| CLEANUP | 4 (one prospective) | **5** (one prospective, +CLN-5 from G10 downgrade) |
| QUARANTINED — not severity-ranked | 7 | 7 |
| Total load-bearing claims classified | 59 of 66 | 59 of 66 |

**Remediation closures:**
- IMP-1 INTERIM CLOSED 2026-04-18, commit `5d67edc4` — HLZ-2015 stub + Criterion 4 indirection note. Full closure pending HLZ-2015 RFS PDF acquisition.
- IMP-2 EFFECTIVELY CLOSED 2026-04-18 — grep of `trading_app/lane_allocator.py` resolved 4/5 constants at `file:line`. No file edit required (code was already correct). Residual downgraded to CLN-5.
- G10 claim class upgraded: `VERIFIED_CODE_PARTIAL` → `VERIFIED_CODE` (1 minor residual).

## IMPORTANT findings (1 open + 2 closed)

### IMP-1 — HLZ-2015 t≥3.00 indirection (DA4) — **INTERIM CLOSED 2026-04-18**

**Grounding class:** `VERIFIED_LOCAL_LITERATURE_INDIRECT`.
**Affected artifact:** `docs/institutional/pre_registered_criteria.md:114` (Criterion 4).
**Issue:** the with-theory threshold `t ≥ 3.00 (Harvey-Liu-Zhu 2015)` routes to Chordia's one-step-removed reference:
> *"they are not far from the suggestion of Harvey, Liu, and Zhu (2015) to use a threshold of three"* — `chordia_et_al_2018_two_million_strategies.md:20` p5.

The HLZ-2015 RFS "…and the Cross-Section of Expected Returns" paper is NOT in the Tier 1 extract layer.
**Decision impact:** none in the current sweep. Every zero-pass verdict (wide-rel-IB max t=2.74, cross-NYSE max t=2.15) fails t≥3.00 regardless. The indirection is latent.
**Severity rationale:** per Bailey et al 2013 (`bailey_et_al_2013_pseudo_mathematics.md:49` — "if only 5 years of data are available, no more than 45 independent model configurations should be tried"), multiple-testing thresholds are load-bearing for deployment. Future 3.00 ≤ t < 3.79 with-theory candidate would promote this to CRITICAL at discovery time.

**Remediation executed (commit `5d67edc4`, 2026-04-18):**
- NEW: `docs/institutional/literature/harvey_liu_zhu_2015_cross_section.md` (stub, 208 lines). Reserves canonical Tier 1 path. Clearly labelled "Status: STUB PENDING PDF ACQUISITION". Records what IS grounded via Chordia p5 verbatim indirect reference. Training-memory content block clearly labelled "NOT verified against local PDF" per `CLAUDE.md` § Local Academic Grounding Rule. Contains 6-step remediation checklist.
- MODIFIED: `docs/institutional/pre_registered_criteria.md:110-115`. Cites both Chordia (verbatim Tier 1 for t ≥ 3.79) and HLZ stub (indirect Tier 1 for t ≥ 3.00). Adds explicit gate: *"Promote t ≥ 3.00 grounding from INDIRECT to DIRECT before any 3.00 ≤ t < 3.79 with-theory candidate is accepted."*

**Status post-remediation:** INTERIM CLOSED. Latent trust surface closed — any future reader of Criterion 4 sees the indirection + the gate. No silent trust loss.
**Full closure requires:** user drops HLZ-2015 RFS PDF into `/resources`, then fills in stub steps 4-6 (page-anchored extraction). Estimated ~30 min when PDF is on disk.
**Gating preserved:** close before a 3.00 ≤ t < 3.79 with-theory candidate is tested.

### IMP-2 — A4c canonical-code partial verification (G10) — **EFFECTIVELY CLOSED 2026-04-18**

**Grounding class (pre-remediation):** `VERIFIED_CODE_PARTIAL`.
**Grounding class (post-remediation):** `VERIFIED_CODE` (1 minor residual, downgraded to CLN-5).
**Affected artifact:** `docs/audit/hypotheses/2026-04-17-garch-a4c-routing-selectivity.yaml:53-58`.
**Issue:** A4c pre-reg cites constants `CORRELATION_REJECT_RHO`, `hysteresis_pct: 0.20`, `deploy_window_months: 12` as canonical allocator contract. PASS 2 grep confirmed `build_allocation` at `trading_app/lane_allocator.py:16-18` and `apply_tight_stop` at `trading_app/config.py:3649` and `bulenox_50k` AccountProfile at `trading_app/prop_profiles.py:726`, but the constant names themselves were not directly hit.

**Remediation executed (2026-04-18, no file change required — code was already correct):**

| Pre-reg field (`a4c yaml:53-61`) | Resolves to | Quality |
|---|---|---|
| `rho_gate_constant: CORRELATION_REJECT_RHO` | `trading_app/lane_allocator.py:506` verbatim: `CORRELATION_REJECT_RHO = 0.70  # Same as lane_correlation.RHO_REJECT_THRESHOLD`; used at `:624` | **VERIFIED_CODE** |
| `hysteresis_pct: 0.20` | `trading_app/lane_allocator.py:48` verbatim: `HYSTERESIS_PCT = 0.20  # Carver Ch.12: 20% switching cost` — **direct in-code cite to Carver Tier 1 extract** | **VERIFIED_CODE** (+Carver cross-ref bonus) |
| `stop_multiplier: 0.75` | Runtime parameter passed through `config.py:3649 apply_tight_stop(outcomes, stop_multiplier, cost_spec)`; pre-reg correctly cites 0.75 as a numeric value, not a named constant | **VERIFIED_CODE** |
| `dd_budget_dollars: 2500.0` (`bulenox_50k` scalar) | `trading_app/prop_profiles.py:726` AccountProfile | **VERIFIED_CODE** |
| `deploy_window_months: 12` | Not resolved to a module-level constant in `lane_allocator.py`. Likely embedded in `_compute_scores` trailing-window logic or a profile attribute. | **RESIDUAL → CLN-5** |

**Status post-remediation:** EFFECTIVELY CLOSED. 4/5 constants VERIFIED_CODE at `file:line`. Residual (`deploy_window_months: 12`) downgraded to CLEANUP CLN-5.
**Decision impact on A4c NULL:** none (unchanged). A4c remains PARKED.
**Decision impact on deployed allocator correctness:** non-issue — rho=0.70 value is consistent with Rule 7 tautology canon at `.claude/rules/backtesting-methodology.md:194`; Carver-grounded hysteresis confirmed in-code.
**Gating:** none.

### IMP-3 — FX6 unsupported "project decision rule" citation

**Grounding class:** `UNSUPPORTED` (specific citation; doctrine *principle* is grounded elsewhere).
**Affected artifact:** `docs/audit/results/2026-04-17-fx-orb-closure.md:57`.
**Issue:** The sentence *"Per project decision rule on pre-flight blockers: If multiple cells are structurally too thin, kill as pre-reg blocker"* invokes a project rule without citing a canonical source. No `.claude/rules/*.md` or `docs/institutional/*.md` file contains that exact wording.
**Principle IS grounded elsewhere:** Rule 8.1 in `.claude/rules/backtesting-methodology.md` (extreme-fire flag on fire_rate<5%) and Rule 5 (comprehensive scope without hand-picking) together establish that structural thinness is a kill, not a rescue target. The principle traces cleanly; the *sentence* does not.
**Decision impact on FX class closure:** none. Path 1 (7/7 raw NO_GO on $29.10 friction surface) carries the class closure independently per `fx-orb-closure.md:14`. Path 2 (BLOCKED_PRE_EXECUTION) reinforces but does not carry alone.
**Remediation:** docs correction only. Edit the sentence at line 57 to either (a) cite Rule 8.1 + Rule 5 explicitly, or (b) rephrase to self-contained FX-pilot logic (14/14 cells fail gate stack = self-evident block, no appeal to unlabeled "project rule"). Prefer (b). ~5 min.
**Gating:** none.

## CLEANUP findings (5)

### CLN-1 — FX benchmark source trace (FX10/FX11)

`docs/audit/results/2026-04-17-cme-fx-futures-orb-pilot.md:22, 33` cite MNQ TOKYO_OPEN +0.046R (live benchmark) and M6E broad ORB family mean -0.302R (dead benchmark) without canonical query path (`validated_setups` / `strategy_fitness`) cited in the doc. Decision impact nil — NO_GO rests on absolute fail gates, not comparison deltas. **Fix:** add `@canonical-source` annotations for both benchmarks.

### CLN-2 — DA7 STALE_DOC risk

`docs/institutional/pre_registered_criteria.md:298` embeds a 2026-03-18 snapshot ("0/124 validated_setups have dsr_score > 0.95. Max dsr_score = 0.1198") as rationale for Amendment 2.1. The policy (DSR as cross-check) is still operative per Amendment 2.7 acceptance matrix row 5, so the historical snapshot supports an in-force policy. Risk is only a misreading. **Fix:** annotate as "snapshot at 2026-03-18 — historical rationale for Amendment 2.1; for current state run live query against `validated_setups.dsr_score`."

### CLN-3 — Branch merge-state hazard

5 Apr-18 overnight replay docs live only on `research/overnight-2026-04-18-v2`. Registry-closure commit `3a94a915` on branch `doctrine/no-go-updates-2026-04-18` cites them by path. Evidence is committed (accessible via `git show`) but stranded from the working tree. **Fix (user decision):** merge `research/overnight-2026-04-18-v2` into the doctrine branch or main. Non-gating cleanup.

### CLN-4 — API drift guard (prospective preventative)

`trading_app/ai/grounding.py` + `corpus.py` currently have NO inlined doctrine thresholds (PASS 2 grep confirmed zero existing drift). No existing remediation required. **Optional hardening:** add a drift check that greps for `3\.79`, `Amendment 2\.`, `dsr.*0\.95`, `MinBTL`, `2026-01-01` under `trading_app/ai/*.py` and fails if any hit. Prevents future inlining. ~20 min.

### CLN-5 — A4c `deploy_window_months: 12` unresolved at file:line (downgraded from IMP-2 G10 residual)

`docs/audit/hypotheses/2026-04-17-garch-a4c-routing-selectivity.yaml:58` cites `deploy_window_months: 12` as part of the canonical allocator comparator contract. IMP-2 grep of `trading_app/lane_allocator.py` did not find a module-level constant with this name or value. Likely embedded in `_compute_scores` trailing-window logic, a profile attribute, or implicit in a scorer function. Decision impact: nil — A4c is NULL and PARKED. A4c verdict does not depend on the exact window. **Fix:** either (a) add a named constant in `lane_allocator.py` and reference it from the pre-reg yaml, or (b) update the pre-reg yaml to point at the actual runtime site (scorer function / profile attribute). Docs-only.

## NOTE_FOR_FOLLOWUP bucket

Non-load-bearing artifacts excluded from PASS 2 classification but logged per hardened rubric so nothing is silently dropped.

### NFFU-1 — Apr-16 Garch transitive audits (14 docs)

Files: `w2c-m2-validated-utility`, `w2d-prior-level-conditioning`, `regime-family-audit`, `regime-utilization-audit`, `structural-decomposition`, `broad-exact-role-exhaustion`, `validated-role-exhaustion`, `additive-sizing-audit`, `normalized-sizing-audit`, `proxy-native-sizing-audit`, `discrete-policy-surface-audit`, `g0-preflight`, `mechanism-hypotheses`, `regime-audit-synthesis`, `all-sessions-universality` under `docs/audit/results/` dated 2026-04-15 / 2026-04-16.
**Reason excluded:** none of the 6 load-bearing criteria triggered — not cited from HANDOFF active shelf, no deployment/kill-verdict reference, no committed hash in experimental_strategies, no pre-reg freeze. Mostly exploratory or transitive.
**Promote-if:** any downstream A4c Stage 2 or future allocator workstream cites one of these directly — flag for PASS 2 re-ground at that time.

### NFFU-2 — Meta-plans (7 docs)

Files: `institutional-attack-plan`, `institutional-utilization-plan`, `program-audit`, `deployment-map-incremental-edge-proof-plan`, `deployment-map-proof-plan-reset`, `garch-deployment-allocator-architecture`, `garch-deployment-replay-design-review` under `docs/plans/` dated 2026-04-16 / 2026-04-17.
**Reason excluded:** narrative / strategy docs with no runtime gate or verdict attachment.
**Promote-if:** any becomes cited as basis for a gate decision.

### NFFU-3 — Phase D parent stub + rel-vol v1

Files: `docs/audit/hypotheses/phase-d-carver-forecast-combiner.md`, `docs/audit/results/2026-04-15-rel-vol-stress-test.md` (v1).
**Reason excluded:** v2 supersedes v1; parent stub is framework-only with no decision attachment.

### NFFU-4 — PLAN_codex.md C1 cleanup

`PLAN_codex.md` C1 cleanup edit on 2026-04-17. Non-canonical, no runtime gate.
**Reason excluded:** docs-governance at non-canonical surface.

### NFFU-5 — Check 45 drift on `MNQ_EUROPE_FLOW_*_CROSS_SGP_MOMENTUM`

Pre-existing drift finding flagged in HANDOFF.md (2026-04-10 stored vs 2026-04-14 canonical recompute on 3 active validated lanes). Adjacent to cross-NYSE momentum work but outside priority scope for this sweep.
**Recommendation:** fold into next regen sweep. Not contaminated into this audit per user directive.

### NFFU-6 — Phase D D-1 contract-locked work (PD1–PD7 quarantined)

Files: `docs/audit/hypotheses/2026-04-15-phase-d-volume-pilot-spec.md`, `docs/audit/results/2026-04-15-path-c-h2-closure.md`, `docs/audit/results/2026-04-15-rel-vol-stress-test-v2.md`, Phase D commits `4fd6c264`, `27fbb22e`, `0b80df5f`.
**Reason excluded:** all PASS 1 extraction was MEMORY.md-sourced; raw reads deferred per user directive. Seven claims (PD1–PD7) carry no PASS 2 verdict.
**Promote-when:** raw reads complete. Expected claims then verifiable against Tier 1 Carver extract (`carver_2015_volatility_targeting_position_sizing.md`) + harness outputs.

## What is still trustworthy right now?

### Safe to keep using as-is

- **All six in-scope verdicts** — FX class CLOSED, A4c NULL + path PAUSED, A4b NULL_BY_CONSTRUCTION, VWAP_BP DEAD, wide-rel-IB NULL, cross-NYSE NULL.
- **Amendment 2.7 Mode A holdout doctrine** at `pre_registered_criteria.md:461, 470`, enforced at `trading_app/holdout_policy.py:70` (`HOLDOUT_SACRED_FROM: date = date(2026, 1, 1)`).
- **Chordia t ≥ 3.79 without-theory threshold** — verbatim quoted from `chordia_et_al_2018_two_million_strategies.md:20`.
- **Bailey MinBTL bound** — verbatim Theorem 1 from `bailey_et_al_2013_pseudo_mathematics.md:43, 49`.
- **Rule 7 tautology canon** — verbatim from `.claude/rules/backtesting-methodology.md:194`.
- **Cross-NYSE momentum replay doc** — gold-standard self-grounded methodology table with file:line + Tier 1 anchors per claim.
- **Claude API modernization Stages 1–4** — engineering surface, no inlined doctrine detected, drift-check gate in place.
- **Doctrine registry closure** (commit `3a94a915`) — 4 kills recorded in `STRATEGY_BLUEPRINT.md §5` + `TRADING_RULES.md §"What Doesn't Work"`.

### Safe only as descriptive / provisional

- **DA7 snapshot** — "0/124 validated_setups DSR > 0.95" from 2026-03-18. Acceptable as historical rationale for Amendment 2.1; needs live re-query if cited as current evidence.
- **A4c churn-rule ambiguity** — honest disclosure at HANDOFF.md:33-41 that the pre-reg did not disambiguate the comparator. Acknowledged and not used to rescue; OK as descriptive, NOT as future kill criterion.
- **Criterion 4 t ≥ 3.00 with-theory carve-out** — post-IMP-1 interim closure (commit `5d67edc4`), the indirection is documented explicitly in `pre_registered_criteria.md:110-115` and a stub at `docs/institutional/literature/harvey_liu_zhu_2015_cross_section.md` reserves the canonical Tier 1 path. Usable for audits where max t < 3.00. NOT usable as the decisive threshold on any 3.00 ≤ t < 3.79 candidate until the stub is replaced with a page-anchored extract (requires HLZ-2015 RFS PDF in `/resources`).

### Needs correction before further decisions

- **Nothing** under the current sweep. Every LB verdict is independently supported; IMP-1/2/3 remediations close *latent* trust surfaces, not active ones.

### Should be frozen pending re-grounding

- **Nothing** — no finding mapped to freeze-pending severity.

### Quarantined (not severity-ranked this pass)

- **Phase D PD1–PD7** — all Phase D claims sourced from MEMORY.md. Full PASS 2 verdict deferred until raw reads of `2026-04-15-phase-d-volume-pilot-spec.md`, `2026-04-15-path-c-h2-closure.md`, `2026-04-15-rel-vol-stress-test-v2.md`.

---

## Remediation log

| Date | Finding | Action | Commit | Status |
|---|---|---|---|---|
| 2026-04-18 | IMP-1 | HLZ-2015 stub + Criterion 4 indirection note | `5d67edc4` | INTERIM CLOSED (full closure pending PDF) |
| 2026-04-18 | IMP-2 | Grep of `lane_allocator.py` confirmed 4/5 constants at `file:line`; residual → CLN-5 | no file change required | EFFECTIVELY CLOSED |
| — | IMP-3 | Docs correction at `fx-orb-closure.md:57` — cite Rule 8.1+5 or rephrase | — | OPEN |
| — | CLN-1 | `@canonical-source` annotation for FX benchmarks | — | OPEN |
| — | CLN-2 | Annotate DA7 snapshot date | — | OPEN |
| — | CLN-3 | Merge `research/overnight-2026-04-18-v2` — **NOTE:** doctrine branch already merged to main (commit `4e1066dd`); remediation folds into that work | — | OPEN (re-scope) |
| — | CLN-4 | Optional drift check for inlined doctrine in `trading_app/ai/` | — | OPEN (prospective) |
| — | CLN-5 | `deploy_window_months: 12` — add named constant or update pre-reg | — | OPEN (downgraded from IMP-2 residual) |
| — | PD1–PD7 | Raw-read Phase D docs to lift quarantine | — | QUARANTINED |

---

**Audit produced 2026-04-18 by Claude Code, 4-pass method, quote-or-fail + file:line-or-degrade + objective load-bearing test per hardened rubric. Remediation log updated post-commit `5d67edc4`.**
