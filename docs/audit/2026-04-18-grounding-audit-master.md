# Grounding Audit — 2026-04-18

**Scope:** Apr 16–18 2026 workstreams — FX ORB pilot + closure, Claude API modernization, Docs authority cleanup, Phase D volume pilot, Garch allocator A4a/A4b/A4c, Overnight queue (VWAP_BP / wide-rel-IB v2 / cross-NYSE momentum).
**Method:** 4-pass fail-closed audit — inventory → claim extraction → re-grounding → severity. Quote-or-fail applied to load-bearing numerical / threshold / date claims. File:line-or-degrade applied to canonical-code claims. Objective 6-criteria load-bearing test applied at PASS 0.
**Result (pre-remediation):** 0 CRITICAL, 3 IMPORTANT, 4 CLEANUP (one prospective). 7 items QUARANTINED pending raw read.
**Result (post-remediation 2026-04-18 sweep 1):** 0 CRITICAL, 1 IMPORTANT (IMP-3 docs hygiene), 5 CLEANUP, 7 QUARANTINED. IMP-1 INTERIM CLOSED via HLZ-2015 stub (commit `5d67edc4`). IMP-2 EFFECTIVELY CLOSED — 4/5 constants verified at `file:line`; residual `deploy_window_months: 12` downgraded to CLEANUP CLN-5.
**Result (post-remediation 2026-04-18 sweep 2):** 0 CRITICAL, 0 IMPORTANT, 3 CLEANUP open (CLN-3 restored per correction below, CLN-4 prospective, CLN-6 new), 0 QUARANTINED. IMP-3 + CLN-1 + CLN-2 closed via docs corrections (commit `d24d8e1c`). CLN-5 CLOSED — `DEPLOY_WINDOW_MONTHS = 12` resolves at `trading_app/lane_allocator.py:40` with in-code Carver Ch.11 cite; G10 upgraded to full VERIFIED_CODE (5/5 constants at `file:line`). **CLN-3 correction:** sweep-1 erroneously annotated CLN-3 as "RESOLVED via branch merge" — verification (`git ls-files docs/audit/results/ | grep 2026-04-18`) returns empty, so `research/overnight-2026-04-18-v2` did NOT merge with the doctrine branch into main. Apr-18 replay docs still stranded from working tree. CLN-3 restored to OPEN. **CLN-6 new:** Phase D D-1 contract-lock yaml (`2026-04-17-phase-d-d1-signal-only-shadow.yaml`) lives on `phase-d-volume-pilot-d0` branch, not main — same branch-merge-state hazard class as CLN-3. PD1–PD7 quarantine LIFTED via raw-read verification.

## Executive summary

66 load-bearing claims extracted from 36 objectively-load-bearing artifacts (of 97 in scope). Every in-scope verdict — FX class CLOSED, A4c NULL, A4b NULL_BY_CONSTRUCTION, VWAP_BP DEAD, wide-rel-IB NULL, cross-NYSE NULL, Phase D D-1 contract lock — traces to either quoted Tier 1 literature, quoted project canon, or self-reported harness output plus doctrine quoted verbatim. No verdict flips under the findings. No conflict between project canon and local literature. No local source read failures.

## Scope

### Workstreams audited

1. FX ORB pilot + filter-rescue closure (Apr 16–17) — 6 LB artifacts
2. Claude API modernization Stages 1–4 (Apr 17) — 5 LB artifacts (engineering surface)
3. Docs authority cleanup — I1/I2/I3(a)/I3(b)/I3(c)/C1/C2 edits (Apr 17) — 5 LB artifacts
4. Phase D volume pilot D-0 / D-1 — 3 LB artifacts (QUARANTINE LIFTED sweep 2; see NFFU-6 for per-claim PASS 2 classifications)
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

| Severity | Pre-remediation | Sweep 1 (2026-04-18 AM) | Sweep 2 (2026-04-18 PM) |
|---|---:|---:|---:|
| CRITICAL | 0 | **0** | **0** |
| IMPORTANT | 3 | **1** (IMP-3) | **0** |
| CLEANUP | 4 (one prospective) | **5** (one prospective, +CLN-5) | **3** open (CLN-3 corrected-OPEN, CLN-4 prospective, CLN-6 NEW) + 3 closed (CLN-1, CLN-2, CLN-5) |
| QUARANTINED — not severity-ranked | 7 | 7 | **0** — PD1–PD7 lifted |
| Total load-bearing claims classified | 59 of 66 | 59 of 66 | **66 of 66** |

**Remediation closures:**
- IMP-1 INTERIM CLOSED 2026-04-18 sweep 1, commit `5d67edc4` — HLZ-2015 stub + Criterion 4 indirection note. Full closure pending HLZ-2015 RFS PDF acquisition.
- IMP-2 EFFECTIVELY CLOSED 2026-04-18 sweep 1, then FULLY CLOSED sweep 2 — grep in sweep 2 located the 5th constant: `DEPLOY_WINDOW_MONTHS = 12` at `trading_app/lane_allocator.py:40` with inline `# Carver Ch.11: forecast weighting window` cite. All 5/5 constants now VERIFIED_CODE at `file:line`.
- G10 claim class upgraded: `VERIFIED_CODE_PARTIAL` (sweep 1) → `VERIFIED_CODE` (full, sweep 2).
- IMP-3 CLOSED 2026-04-18 sweep 2, commit `d24d8e1c` — `fx-orb-closure.md:54-60` rephrased to cite `.claude/rules/backtesting-methodology.md` Rule 5 + Rule 8.1 directly instead of invoking unlabeled "project decision rule".
- CLN-1 CLOSED 2026-04-18 sweep 2, commit `d24d8e1c` — `cme-fx-futures-orb-pilot.md` gained "Benchmark provenance" subsection with `@canonical-source` annotations for live (orb_outcomes same-window recompute) and dead (prior M6E family scan) benchmarks.
- CLN-2 CLOSED 2026-04-18 sweep 2, commit `d24d8e1c` — `pre_registered_criteria.md:300` annotated as "snapshot at 2026-03-18 — historical rationale for Amendment 2.1".
- CLN-5 CLOSED 2026-04-18 sweep 2 — `DEPLOY_WINDOW_MONTHS = 12` found at `trading_app/lane_allocator.py:40`; no file edit needed.
- **CLN-3 CORRECTION (sweep 2):** sweep 1 claimed CLN-3 resolved by doctrine-branch merge. Adversarial audit verification (`git ls-files docs/audit/results/ | grep 2026-04-18` returns empty) proved that only the doctrine branch (`doctrine/no-go-updates-2026-04-18`) merged to main — the `research/overnight-2026-04-18-v2` branch holding the actual replay docs did NOT merge. CLN-3 restored to OPEN.
- **CLN-6 NEW (sweep 2):** Phase D D-1 contract-lock yaml at `docs/audit/hypotheses/2026-04-17-phase-d-d1-signal-only-shadow.yaml` lives on `phase-d-volume-pilot-d0` branch, not main. Same branch-merge-state hazard class as CLN-3. Cleanup-tier.
- PD1–PD7 QUARANTINE LIFTED 2026-04-18 sweep 2 — raw-read verification against source docs produced 6/7 VERIFIED classifications + 1 PARTIALLY_VERIFIED (PD1 "top-3 universality" specific cells not in `path-c-h2-closure.md`; live in raw CSV `h2_universality_garch_vol_pct_GT70_raw.csv` referenced from MEMORY).

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
| `deploy_window_months: 12` | `trading_app/lane_allocator.py:40` verbatim: `DEPLOY_WINDOW_MONTHS = 12  # Carver Ch.11: forecast weighting window` — **second direct in-code cite to Carver Tier 1 extract** (first was HYSTERESIS_PCT Ch.12). Located via sweep-2 adversarial audit. | **VERIFIED_CODE** (+Carver cross-ref bonus) |

**Status post-remediation:** FULLY CLOSED (sweep 2). All 5/5 constants VERIFIED_CODE at `file:line`. Sweep 1 classified this as EFFECTIVELY CLOSED with 1 residual (CLN-5); sweep 2 adversarial audit located the 5th constant, closing G10 completely.
**Decision impact on A4c NULL:** none (unchanged). A4c remains PARKED.
**Decision impact on deployed allocator correctness:** non-issue — rho=0.70 value is consistent with Rule 7 tautology canon at `.claude/rules/backtesting-methodology.md:194`; Carver-grounded hysteresis AND deploy window both confirmed in-code.
**Gating:** none.

### IMP-3 — FX6 unsupported "project decision rule" citation — **CLOSED 2026-04-18 sweep 2**

**Grounding class:** `UNSUPPORTED` (specific citation; doctrine *principle* is grounded elsewhere).
**Affected artifact:** `docs/audit/results/2026-04-17-fx-orb-closure.md:57`.
**Issue:** The sentence *"Per project decision rule on pre-flight blockers: If multiple cells are structurally too thin, kill as pre-reg blocker"* invokes a project rule without citing a canonical source. No `.claude/rules/*.md` or `docs/institutional/*.md` file contains that exact wording.
**Principle IS grounded elsewhere:** Rule 8.1 in `.claude/rules/backtesting-methodology.md` (extreme-fire flag on fire_rate<5%) and Rule 5 (comprehensive scope without hand-picking) together establish that structural thinness is a kill, not a rescue target. The principle traces cleanly; the *sentence* does not.
**Decision impact on FX class closure:** none. Path 1 (7/7 raw NO_GO on $29.10 friction surface) carries the class closure independently per `fx-orb-closure.md:14`. Path 2 (BLOCKED_PRE_EXECUTION) reinforces but does not carry alone.

**Remediation executed (commit `d24d8e1c`, sweep 2):** `docs/audit/results/2026-04-17-fx-orb-closure.md:54-60` rewritten to remove the uncited "project decision rule" invocation. Replaced with self-contained logic grounded in `.claude/rules/backtesting-methodology.md` Rule 5 (comprehensive scope — narrowing allowed only with pre-reg justification) and Rule 8.1 (extreme fire rate — rare-event gating requires pre-registered justification, not post-hoc widening). FX class closure verdict unchanged; Path 1 still carries it independently.
**Status post-remediation:** CLOSED. Doctrine basis now explicit.
**Gating:** none.

## CLEANUP findings (6 — 3 closed sweep 2, 3 open)

### CLN-1 — FX benchmark source trace (FX10/FX11) — **CLOSED 2026-04-18 sweep 2**

`docs/audit/results/2026-04-17-cme-fx-futures-orb-pilot.md:22, 33` cited MNQ TOKYO_OPEN +0.046R (live benchmark) and M6E broad ORB family mean -0.302R (dead benchmark) without canonical query path cited in the doc. Decision impact nil — NO_GO rests on absolute fail gates, not comparison deltas.

**Remediation executed (commit `d24d8e1c`):** added "Benchmark provenance" subsection with `@canonical-source` annotations. Live benchmark traces to `gold.db` `orb_outcomes` same-window recompute. Dead benchmark traces to prior M6E family scan before M6E retirement. Both flagged "not deployment-grade-citable" to prevent misuse. **Status:** CLOSED.

### CLN-2 — DA7 STALE_DOC risk — **CLOSED 2026-04-18 sweep 2**

`docs/institutional/pre_registered_criteria.md:300` (was line 298 pre-IMP-1 edit; shifted by sweep-1 commit `5d67edc4` adding 2 lines at Criterion 4) embedded a 2026-03-18 snapshot ("0/124 validated_setups have dsr_score > 0.95. Max dsr_score = 0.1198") as rationale for Amendment 2.1.

**Remediation executed (commit `d24d8e1c`):** annotated inline as "snapshot at 2026-03-18 — historical rationale for Amendment 2.1; for current state run live query against `validated_setups.dsr_score`." **Status:** CLOSED.

### CLN-3 — Branch merge-state hazard — **OPEN (sweep 1 resolution claim was wrong; corrected sweep 2)**

**Sweep 1 claim (incorrect):** "effectively resolved by doctrine-branch merge into main (commit `4e1066dd`)."
**Sweep 2 verification (corrective):** `git ls-files docs/audit/results/ | grep 2026-04-18` returned EMPTY on main. The doctrine branch (`doctrine/no-go-updates-2026-04-18`) merged to main, but that brought in only the registry closure (`3a94a915`) — NOT the replay docs on `research/overnight-2026-04-18-v2`. Those 5 replay docs remain stranded. Commits `284e3d35`, `e1cc2977`, `16cc3af5`, `66d26edb`, `849d9097`, `16158e31`, `0d47a761` still hold the evidence but the working tree doesn't see it. **Fix:** merge `research/overnight-2026-04-18-v2` into main. Non-gating, but CLN-3 is legitimately OPEN, not resolved.

### CLN-4 — API drift guard (prospective preventative)

`trading_app/ai/grounding.py` + `corpus.py` currently have NO inlined doctrine thresholds (PASS 2 grep confirmed zero existing drift). No existing remediation required. **Optional hardening:** add a drift check that greps for `3\.79`, `Amendment 2\.`, `dsr.*0\.95`, `MinBTL`, `2026-01-01` under `trading_app/ai/*.py` and fails if any hit. Prevents future inlining. ~20 min.

### CLN-5 — A4c `deploy_window_months: 12` — **CLOSED 2026-04-18 sweep 2**

**Sweep 1 status:** downgraded from IMP-2 G10 residual; classified as docs-hygiene pending grep.
**Sweep 2 adversarial audit:** located the constant at `trading_app/lane_allocator.py:40` — `DEPLOY_WINDOW_MONTHS = 12  # Carver Ch.11: forecast weighting window`. This is the second direct in-code Carver Tier 1 cite (after `HYSTERESIS_PCT` at line 48 cross-referencing Carver Ch.12). G10 therefore upgrades from `VERIFIED_CODE_PARTIAL` to full `VERIFIED_CODE` (5/5 constants at `file:line`). **No file edit required.** **Status:** CLOSED.

### CLN-6 — Phase D branch merge-state hazard — **NEW (sweep 2)**

Phase D D-1 signal-only shadow contract file `docs/audit/hypotheses/2026-04-17-phase-d-d1-signal-only-shadow.yaml` (committed at `4fd6c264`) lives on `phase-d-volume-pilot-d0` branch, not on main. PD5 claim ("Phase D D-1 signal-only shadow: contract locked 2026-04-17") VERIFIED via `git show 4fd6c264` but the yaml itself is stranded from working tree — same branch-merge-state hazard class as CLN-3. Decision impact: nil while Phase D is in pre-build spec phase; **blocks reading the pre-reg yaml content from main without `git show`**. **Fix:** merge `phase-d-volume-pilot-d0` into main when Phase D moves past D-0 / D-1 signal-only shadow phase. Non-gating.

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

### NFFU-5 — Check 45 drift on `MNQ_EUROPE_FLOW_*_CROSS_SGP_MOMENTUM` — **RESOLVED 2026-04-18 via commit `1a0a4a24`**

**Original finding:** pre-existing drift flagged in HANDOFF.md (2026-04-10 stored vs 2026-04-14 canonical recompute on 3 active validated lanes). Adjacent to cross-NYSE momentum work but outside priority scope for the original sweep.

**Investigation (2026-04-18 sweep 3):** classified as stale annotation — the drift was real as of 2026-04-17 runs (A4b, A4c, FX-pilot all flagged it) but resolved on 2026-04-18 at 12:13 via commit `1a0a4a24 fix(check-45): canonical refresh tool for stale validated_setups trade windows`. That commit shipped `scripts/migrations/backfill_validated_trade_windows.py` using the same `StrategyTradeWindowResolver` the check uses. Migration was executed: 3 SGP rows updated from `(2019-05-08, 2026-04-10, N=1020)` → `(2019-05-08, 2026-04-14, N=1021)`. Post-fix drift state per commit message: `103 passed, 0 failed, 6 advisory`.

**Current state (verified 2026-04-18 PM):**
- All 3 rows have `last_trade_day = 2026-04-14, trade_day_count = 1021`.
- `check_active_native_trade_windows_match_provenance()` returns 0 violations.

**Root-cause analysis:** not corruption, not false positive — architectural snapshot-vs-live pattern. `validated_setups.{first_trade_day, last_trade_day, trade_day_count}` are snapshots written at validation time; they drift when new trading days are ingested without an explicit refresh. Canonical resolver always computes from `orb_outcomes`. `1a0a4a24` shipped the refresh tool (7 TDD tests covering dry-run, live, idempotent re-run, retired/legacy exclusion, strategy-id scoping, main exit code).

**Status:** INSTANCE RESOLVED; class-level mitigation PARTIAL. The manual refresh tool is idempotent and covers the current instance, but it is NOT auto-triggered on daily ingest. Each new trading day that advances `orb_outcomes` without an accompanying `backfill_validated_trade_windows.py` run will re-surface the same drift class. A future daily-pipeline hook invoking the refresh tool (or equivalent auto-sync in the ingest chain) would promote this to CLASS RESOLVED. Not in current audit scope; flagged here as a known gap. No further action on NFFU-5's current instance.

### NFFU-6 — Phase D D-1 contract-locked work (PD1–PD7) — **QUARANTINE LIFTED 2026-04-18 sweep 2**

Sources: `docs/audit/hypotheses/2026-04-15-phase-d-volume-pilot-spec.md`, `docs/audit/results/2026-04-15-path-c-h2-closure.md`, `docs/audit/results/2026-04-15-rel-vol-stress-test-v2.md`, Phase D commits `4fd6c264`, `27fbb22e`, `0b80df5f`.

Sweep 2 raw-read PASS 2 classifications:

| ID | Claim | Source + exact span | Verdict |
|---|---|---|---|
| PD1 | Path C H2 verdict: NO CAPITAL | `path-c-h2-closure.md:117` verbatim: *"Deployment posture unchanged from prior handover: nothing to live capital until the composite and DSR resolve."* Lines 111-115 "H2 T5 family: PASS based on 68.5% generalization across 527 combos. Feature is genuinely cross-asset cross-session, not a single-cell find." | **VERIFIED_DB_OR_CANONICAL_QUERY** on main verdict. **PARTIALLY_VERIFIED** on "top-3 universality" sub-claim — specific cells (MGC COMEX_SETTLE O5 RR2.0 long Δ=+0.311, MGC EUROPE_FLOW O15 RR2.0 long +0.295, MES TOKYO_OPEN O15 RR2.0 long +0.286) are in raw CSV `h2_universality_garch_vol_pct_GT70_raw.csv`, not in the closure doc itself. MEMORY.md-only at the specific-cell level. |
| PD2 | rel_vol v2 DSR recalibrated to empirical var_sr≈0.012 | `rel-vol-stress-test-v2.md:8` verbatim: *"var_sr: 0.012190 (empirical cross-lane per-trade SR variance across 64 lanes)"*. DSR default var_sr=0.047 at `trading_app/dsr.py:13` verbatim: `compute_sr0(n_eff=253, var_sr=0.047)`. Ratio: 0.047/0.012 ≈ 3.92 (MEMORY says 3.8 — minor rounding, not material). | **VERIFIED_CODE** (default) + **VERIFIED_DB_OR_CANONICAL_QUERY** (calibration). Note: MEMORY.md references "dsr.py line 35" but actual location is line 13 — minor trace correction. |
| PD3 | H2 garch_vol_pct≥70: 68.5% positive across 527 combos; ExpR +0.263 garch-alone | `path-c-h2-closure.md:36` verbatim: *"Positive delta: 361 (68.5%)"* of `N=527 testable combos`. `path-c-h2-closure.md:91` garch-only row: ExpR=+0.263. Line 93 both-fire row: ExpR=+0.220. | **VERIFIED_DB_OR_CANONICAL_QUERY** |
| PD4 | Composite rel_vol × garch corr=0.069 orthogonal; no synergy | `path-c-h2-closure.md:83` verbatim: *"T7 orthogonality — corr(fire_rel, fire_garch) on full data: 0.069"*. Line 104-105 verbatim: *"Synergy: ExpR(both) - max_marginal = -0.043 -> NO SYNERGY / SUBSUMED — composite is not additive."* | **VERIFIED_DB_OR_CANONICAL_QUERY** |
| PD5 | Phase D D-1 signal-only shadow: contract locked 2026-04-17 | Commit `4fd6c264` on 2026-04-17: *"pre-reg(phase-d): D-1 signal-only shadow — contract locked"*. Adds `docs/audit/hypotheses/2026-04-17-phase-d-d1-signal-only-shadow.yaml` (254 lines). **Branch-state note:** yaml lives on `phase-d-volume-pilot-d0`, NOT on main — see CLN-6. | **VERIFIED_DB_OR_CANONICAL_QUERY** via `git show` |
| PD6 | Phase D pilot spec: MNQ COMEX_SETTLE as D-0, 5 stages D-0 to D-4, pre-reg criteria, kill criteria, 15-week timeline | `phase-d-volume-pilot-spec.md:35` verbatim: *"MNQ COMEX_SETTLE O5 RR1.5"* (pilot lane). Lines 60-68 stage table D-0 through D-4. Lines 72-93 pre-reg criteria §4. Lines 97-102 kill criteria §5. Line 133 verbatim: *"~15 weeks for full Phase D deployment"*. | **VERIFIED_PROJECT_CANON** |
| PD7 | Phase D parent framework cites Carver Ch 9-10 size-scaling + forecast combiner | `phase-d-volume-pilot-spec.md:22` verbatim: *"Mechanism (Carver Ch 9-10 grounded)"*. Tier 1 extract confirmed present at `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md`. | **VERIFIED_BOTH** (canon + Tier 1 literature) |

**Net quarantine lift:** 6/7 VERIFIED, 1/7 PARTIALLY_VERIFIED (PD1 top-3 detail lives in raw CSV rather than the closure doc). No conflicts with existing canon. **Decision impact:** none — all PD claims are project-state descriptors, not deployment decisions. Phase D spec remains pre-build.

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

- **Empty as of sweep 2.** PD1–PD7 lifted — see NFFU-6 for per-claim classifications. 6/7 VERIFIED, 1/7 PARTIALLY_VERIFIED (PD1 top-3 universality detail traces to raw CSV, not closure doc).

---

## Remediation log

| Date | Finding | Action | Commit | Status |
|---|---|---|---|---|
| 2026-04-18 AM | IMP-1 | HLZ-2015 stub + Criterion 4 indirection note | `5d67edc4` | INTERIM CLOSED (full closure pending PDF) |
| 2026-04-18 AM | IMP-2 | Grep of `lane_allocator.py` confirmed 4/5 constants; residual → CLN-5 | no file change required | EFFECTIVELY CLOSED (sweep 1), FULLY CLOSED sweep 2 |
| 2026-04-18 PM | IMP-3 | Docs correction at `fx-orb-closure.md:54-60`: removed uncited "project decision rule"; replaced with explicit Rule 5 + Rule 8.1 citation | `d24d8e1c` | CLOSED |
| 2026-04-18 PM | CLN-1 | Added "Benchmark provenance" subsection with `@canonical-source` annotations to `cme-fx-futures-orb-pilot.md` | `d24d8e1c` | CLOSED |
| 2026-04-18 PM | CLN-2 | Annotated `pre_registered_criteria.md:300` snapshot date (2026-03-18) | `d24d8e1c` | CLOSED |
| 2026-04-18 PM | CLN-3 | **Corrective update:** sweep 1 claimed resolution via doctrine-branch merge; sweep 2 verified `research/overnight-2026-04-18-v2` did NOT merge. Restored to OPEN. | — | **OPEN** (correction from sweep 1 claim) |
| 2026-04-18 PM | CLN-5 | Located `DEPLOY_WINDOW_MONTHS = 12` at `lane_allocator.py:40` via sweep-2 adversarial audit (Carver Ch.11 in-code cite). G10 upgraded to full VERIFIED_CODE (5/5 constants). | no file change required | CLOSED |
| 2026-04-18 PM | CLN-6 | Identified Phase D branch-merge-state hazard (phase-d-d1 yaml on `phase-d-volume-pilot-d0` branch, not main). New cleanup item. | — | OPEN (NEW sweep 2) |
| 2026-04-18 PM | PD1–PD7 | Raw-read verification against source docs; quarantine lifted. 6/7 VERIFIED, 1/7 PARTIALLY_VERIFIED (PD1 top-3 universality live in raw CSV). | no file change required | QUARANTINE LIFTED |
| — | CLN-4 | Optional drift check for inlined doctrine in `trading_app/ai/` | — | OPEN (prospective) |
| — | CLN-5 | `deploy_window_months: 12` — add named constant or update pre-reg | — | OPEN (downgraded from IMP-2 residual) |
| — | PD1–PD7 | Raw-read Phase D docs to lift quarantine | — | QUARANTINED |

---

**Audit produced 2026-04-18 by Claude Code, 4-pass method, quote-or-fail + file:line-or-degrade + objective load-bearing test per hardened rubric. Remediation log updated post-commit `5d67edc4`.**
