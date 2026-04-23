# A2b-2 — DSR ranking patch — scope (Stage-1)

- phase: A2b-2 / Phase 3 of `docs/plans/2026-04-18-multi-phase-audit-roadmap.md`
- status: **SCOPE — Shape E PICKED (see §11 update); Stage-2 awaits final user OK**
- author: audit/a2b-1-regime-gate-phase2 (branch retained for thread continuity; A2b-1 paused)
- created: 2026-04-18 (post Phase 3a empirical, post A2b-1 PAUSED pivot)
- shape picked: 2026-04-18 (post Phase 3b + final code-review pass; see §11)
- upstream empirical: `docs/audit/results/2026-04-18-dsr-ranking-empirical-verification.md`
- parent scope: `docs/audit/hypotheses/2026-04-18-a2b-portfolio-optimization-audit-scope.md`
- pivot rationale: `docs/plans/2026-04-18-multi-phase-audit-roadmap.md` § Addendum 2026-04-18

## 1. Problem statement

`trading_app/lane_allocator.py::_effective_annual_r` ranks lanes by raw `annual_r_estimate` (with multiplicative SR-alarm and recent-decay discounts). Per Bailey-Lopez de Prado (2014) Eq 2 and the False Strategy Theorem (LdP-Bailey 2018), raw point estimates from selected strategies are upward-biased — the maximum of N strategy Sharpes is a biased estimator of the true Sharpe of the best strategy. The deflated Sharpe ratio (DSR) corrects this. The current ranking does not apply that correction; the allocator selects on biased point estimates.

## 2. Empirical grounding (Phase 3a)

From `docs/audit/results/2026-04-18-dsr-ranking-empirical-verification.md`, 30 profile-eligible lanes on the 2026-04-18 `topstep_50k_mnq_auto` rebalance:

| ranking | live selection | non-tied delta vs raw |
|---|---|---:|
| R_raw (current) | 6 lanes (4 ORB_G5, 1 COST_LT12, 1 ATR_P50) | 0 (baseline) |
| R_dsr (DSR alone) | 7 lanes (2 OVNRNG_100 + 1 VWAP_MID_ALIGNED + ...) | 7 |
| R_combo (annual_r × DSR) | 6 lanes (2 OVNRNG_100 + 1 VWAP_MID_ALIGNED + ...) | 6 |

**Top finding:** only 2 of 30 lanes hold DSR > 0.10 even at N_eff=253:
- `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100` — DSR_canonical (N=21) = 0.756, DSR_n253 = 0.173
- `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` — DSR_canonical = 0.718, DSR_n253 = 0.132

All four currently-deployed `ORB_G5` lanes have DSR_canonical < 0.013 — they are below the noise floor under DSR. The current ranking is selecting them because their raw `annual_r_estimate` is large; DSR says that magnitude is consistent with the "best of N noise strategies" outcome, not a true edge.

Verdict: **RANKING_MATERIAL** — the patch changes 4-5 of 6 deployed lanes.

## 3. Patch specification — three coherent shapes (user picks)

### Shape A — `R_combo`: annual_r × DSR (RECOMMENDED)

- Replace `_effective_annual_r(s)` body with a multiplicative DSR discount:
  ```python
  def _effective_annual_r(s: LaneScore) -> float:
      adj = s.annual_r_estimate
      if s.sr_status == "ALARM": adj *= SR_ALARM_DISCOUNT
      if s.recent_3mo_expr is not None and s.recent_3mo_expr < 0 and s.trailing_expr > 0:
          adj *= RECENT_DECAY_DISCOUNT
      adj *= max(getattr(s, "dsr_score", 0.0) or 0.0, MIN_DSR_FLOOR)
      return adj
  ```
- Adds `dsr_score: float | None = None` to `LaneScore`, populated in `compute_lane_scores` via canonical `trading_app.dsr.compute_sr0` + `compute_dsr`.
- `MIN_DSR_FLOOR = 0.05` prevents the ranker from collapsing all lanes to zero when DSR is universally tiny (sanity guard, not a magical constant — calibrated against Phase 3a min-DSR observation).
- **Why recommended:** preserves `annual_r` as the primary economic objective, treats DSR as the confidence discount it was designed to be (per `dsr.py` docstring line 35 "DSR is INFORMATIONAL"). Phase 3a Shape A flips 4 lanes; magnitude-aware so lanes with both high DSR and high annual_r dominate.

### Shape B — `R_dsr`: DSR-only ranking

- Replace `_effective_annual_r(s)` with `s.dsr_score` directly.
- **Why offered:** strictest Bailey-LdP interpretation. Selects only lanes with strong evidence-of-edge regardless of magnitude.
- **Why NOT recommended:** elevates DSR from "informational" to primary objective without doctrine support. dsr.py docstring explicitly says DSR is informational because N_eff is uncertain; making it the primary ranker reverses that.

### Shape C — `R_raw + DSR_FLOOR_GATE`: keep raw ranking, gate by DSR threshold

- Keep `_effective_annual_r` unchanged.
- Add a binary gate: lane is ineligible for selection if `dsr_score < DSR_FLOOR` (e.g., 0.05).
- **Why offered:** minimum-disruption variant. Lanes ranked by raw annual_r as today; DSR only filters out the worst-noise lanes.
- **Why NOT recommended for Stage-2 default:** a binary gate at single N_eff inherits the N_eff-sensitivity problem (lanes flicker in/out as edge_families count shifts).

### Common to all shapes

- `LaneScore.dsr_score` field added, populated by canonical `compute_sr0` + `compute_dsr` calls inside `compute_lane_scores`. Inputs follow `strategy_validator.py:2180-2229` verbatim:
  - `var_sr_by_em` from `experimental_strategies` per entry_model
  - `n_eff` = `COUNT(DISTINCT family_hash) FROM edge_families`
  - lane's `sharpe_ratio`, `sample_size`, `skewness`, `kurtosis_excess` from `validated_setups`
- DSR computation is delegated to canonical `trading_app.dsr` — no re-encoding (Rule 4).
- New fields surface in `lane_allocation.json` for transparency: `dsr_score`, `sr0`, `n_eff`, `var_sr_em` per lane.

## 4. N_eff sensitivity — explicit policy

Per `dsr.py` docstring lines 26-35 + the 2026-04-15 rel_vol v2 stress-test lesson, DSR is highly sensitive to `N_eff` choice. Phase 3a measured DSR at `N_eff ∈ {5, 12, 21, 36, 72, 253}`; OVNRNG_100 lanes vary from 0.98 to 0.13.

Policy for Stage-2:
- **Single canonical `N_eff` = `COUNT(DISTINCT family_hash) FROM edge_families`** (same as `strategy_validator.py:2199`). This keeps the allocator and validator on the same number.
- **New drift check** (numbered to match `pipeline/check_drift.py` series) that flags if `N_eff` changes by >25% between rebalances. Sudden N_eff shifts mean the edge-family inventory moved; ranking would too.
- **Sensitivity diagnostic** (NOT a gate): `lane_allocation.json` records DSR at the N_eff bands `{5, 21, 72, 253}` for each lane so `regime-check` / future audits can see when a lane is N_eff-fragile.
- **NOT in scope:** ONC-algorithm N_eff estimation (LdP 2020). Future enhancement.

## 5. Mode-B grandfather caveat — Phase 3b prerequisite

Per the 2026-04-19 Mode-A revalidation audit (`docs/audit/results/2026-04-19-mode-a-revalidation-of-active-setups.md`), all 38 active `validated_setups` rows drift materially from strict Mode-A IS. Phase 3a consumed those stored values (the same data the validator's DSR pipeline uses), so the Phase 3a result is apples-to-apples with current allocator behavior — but it is NOT the Mode-A-true ranking.

**Stage-2 prerequisite (binding):** before any allocator code change ships, run **Phase 3b** — a read-only audit that recomputes per-lane Sharpe / sample_size / skewness / kurtosis from canonical `orb_outcomes` joined with `daily_features` (filter applied via canonical `research.filter_utils.filter_signal`) restricted to `trading_day < HOLDOUT_SACRED_FROM`. Compare the resulting DSR ranking to Phase 3a. If the direction holds (RANKING_MATERIAL preserved), Stage-2 proceeds. If the direction flips or weakens to RANKING_COSMETIC, Stage-2 halts and scope is revised.

Phase 3b is its own pre-registered audit; this scope just gates Stage-2 on its completion.

## 6. Files in scope (`scope_lock`, Stage-2)

- `trading_app/lane_allocator.py` — `LaneScore.dsr_score` field, populate in `compute_lane_scores`, modify `_effective_annual_r` per chosen shape, write extra fields to `save_allocation`.
- `pipeline/check_drift.py` — new check for N_eff stability between rebalances.
- `tests/test_trading_app/test_lane_allocator.py` — 6 new tests listed in §8.
- `tests/test_pipeline/test_check_drift.py` — 1 new test for the N_eff drift check.

Out of scope explicitly:
- `trading_app/dsr.py` — pure delegation, never modified
- `trading_app/strategy_validator.py` — its DSR write to `validated_setups.dsr_score` continues unchanged
- `trading_app/prop_profiles.py`, `validated_shelf.py`, `lane_correlation.py` — untouched
- Any research script outside the new tests
- Any change to `_compute_session_regime` (that's A2b-1, currently PAUSED)
- Any sizing change (that's A2b-3, downstream)

## 7. Literature grounding

- **Bailey, D.H. & López de Prado, M. (2014)** "The Deflated Sharpe Ratio." J. Portfolio Mgmt 40(5):94-107. Local extract: `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md`. Source of the DSR formula in `trading_app/dsr.py`. Cited in `pre_registered_criteria.md` Criterion 5.
- **López de Prado, M. & Bailey, D.H. (2018)** "The False Strategy Theorem." Math. Methods of Operations Research 86. False Strategy Theorem motivates the SR0 calibration in `compute_sr0`.
- **Harvey & Liu (2015)** "Backtesting." J. Portfolio Mgmt — alternate haircut formulation. NOT used here (DSR is the project's canonical correction); cited as cross-reference for the bias-correction family.
- **`.claude/rules/institutional-rigor.md` Rule 4** — delegate to canonical `trading_app.dsr`, no re-encoding.
- **Project-local rationale (NOT a literature claim):** Phase 3a empirical is the direct evidence; literature supports the framework, the empirical supports the local applicability.

## 8. Tests required (all pre-registered)

T1. `test_dsr_score_populated_in_lane_score` — `compute_lane_scores` produces `LaneScore.dsr_score` in [0, 1] for every scored lane; raises ValueError if outside.

T2. `test_dsr_score_matches_canonical_compute_dsr` — for a fixture lane with known Sharpe / N / skew / kurt / EM, `LaneScore.dsr_score` equals `compute_dsr(...)` called directly with the same inputs (canonical equivalence; guards against parallel-encoding regression).

T3. `test_effective_annual_r_shape_combo_uses_dsr_floor` — Shape A picks the DSR floor when `dsr_score` is missing or below `MIN_DSR_FLOOR`; verify floor is exactly the constant value, not a magic number.

T4. `test_effective_annual_r_shape_combo_changes_ranking` — fixture with two lanes A (high annual_r, DSR=0.01) and B (lower annual_r, DSR=0.5); under Shape A, B ranks above A.

T5. `test_compute_lane_scores_passes_canonical_var_sr_n_eff` — monkeypatch `compute_sr0` to record kwargs; assert each call receives canonical `n_eff` and per-EM `var_sr` matching `strategy_validator.py:2199`/`2186-2195` patterns.

T6. `test_save_allocation_writes_dsr_diagnostic_fields` — `lane_allocation.json` post-patch contains `dsr_score`, `sr0`, `n_eff` per lane.

T7. (drift) `test_check_drift_n_eff_stability` — drift check passes when N_eff changes by ≤25% between rebalances; fails (advisory) when >25%.

All tests use synthetic fixtures except T7 which uses a small seeded DB. No live `gold.db` dependency in unit tests; no OOS consumption.

## 9. Kill criteria (pre-registered — if ANY fires, Stage-2 HALTS)

K1. **Phase 3b not complete** — no Stage-2 code touches `lane_allocator.py` until Phase 3b result MD exists and shows RANKING_MATERIAL preserved under Mode-A-true Sharpe.

K2. **Existing test regression** — `pytest tests/test_trading_app/test_lane_allocator.py` must remain green. Any pre-existing test failing on the patched tree → HALT and revert.

K3. **Drift regression** — `python pipeline/check_drift.py` must not add new failures attributable to this commit.

K4. **Phase 1 reproduction break** — `python research/audit_allocator_rho_excluded.py` reproduction must still match `lane_allocation.json` selection (modulo the existing tie-break tolerance) with Shape A active. Allocator non-determinism on ties is acceptable; substantive selection mismatch is not.

K5. **N_eff sanity** — at deployment time, `n_eff ≥ 2` and `var_sr_e2 > 0`. Otherwise `dsr_score = 0` and the ranking falls back to raw annual_r per Shape A floor logic; log a warning. Hard fail if `n_eff < 2` AND no `var_sr_e2` AND every lane has `dsr_score = 0` simultaneously — that's a pipeline failure, not a ranking question.

K6. **Live-lane verdict flip without user approval** — the patched `compute_lane_scores` output for the next post-patch rebalance MUST present a diff vs current `lane_allocation.json` to the user before any push to that JSON. The patch ships (code merged) but the rebalance script `scripts/tools/rebalance_lanes.py` is gated by an interactive confirmation in the same commit (or a `--confirm-flips` CLI flag) the first time the new ranking changes selection. After user confirms once, gate is removed by a follow-up commit.

## 10. Rollback plan

- Single-file revert on `trading_app/lane_allocator.py` (and the matching test/drift-check files) via `git revert <commit-sha>`.
- `lane_allocation.json` snapshot pre-patch is preserved automatically by git history; if a bad rebalance shipped, restore via `git checkout <pre-patch-sha> -- docs/runtime/lane_allocation.json` then re-run the rebalance with the reverted code.
- No data migration, no schema change. `validated_setups.dsr_score` is already a column; its content is untouched.

## 10a. Adversarial self-review (preempting the §10a-style audit)

Findings from re-auditing this scope's first draft. Addressed inline; preserved here as honesty trail.

| # | Severity | Finding | Resolution |
|---|---|---|---|
| 1 | HIGH | Shape A introduces `MIN_DSR_FLOOR = 0.05` constant — what's the basis? | Explicitly labeled as a sanity guard; calibrated against Phase 3a min-DSR; requires unit test (T3) verifying the floor is the literal constant, not a magic number. |
| 2 | HIGH | Phase 3a uses Mode-B-stale Sharpe; making A2b-2 binding without re-verifying under Mode A risks shipping a patch built on stale numbers. | §5 makes Phase 3b a binding Stage-2 prerequisite (K1). |
| 3 | HIGH | DSR is INFORMATIONAL per `dsr.py` line 35; promoting it to primary ranker (Shape B) violates project doctrine. | Shape B explicitly NOT recommended; Shape A keeps DSR as discount, honors line 35 framing. |
| 4 | MEDIUM | N_eff sensitivity is large; locking to a single N_eff is fragile. | §4 adds drift check + diagnostic record at multiple N_eff bands. Future-proofs the choice. |
| 5 | MEDIUM | Test T4 hard-codes "A above B" — brittle if MIN_DSR_FLOOR changes. | T4 spec uses fixture inputs that produce >2× ratio; resistant to floor adjustment within reasonable range. |
| 6 | MEDIUM | Patch adds field `LaneScore.dsr_score` which `audit_allocator_rho_excluded.py` (Phase 1) does NOT read — but Phase 1 reproduction must still pass per K4. | Field default = `None` keeps Phase 1 unaffected; canonical `compute_lane_scores` populates it on every call. |
| 7 | LOW | Bailey-LdP literature extract is in `docs/institutional/literature/`; cited correctly. No overclaim found. | — |
| 8 | LOW | Shape C (DSR_FLOOR_GATE) added to give user a true minimum-disruption option. Not just a strawman. | Documented honestly as offered, with clear "not recommended" rationale. |

## 10b. Should A2b-2 be the next phase at all?

Honest re-check after Phase 3a evidence:

| argument for A2b-2 next | weight |
|---|---|
| Phase 3a RANKING_MATERIAL: 4-5 of 6 deployed lanes flip — direct deployment impact | high |
| Bailey-LdP literature-grounded; canonical `dsr.py` already exists | high |
| Phase 1 evidence: 20 of 24 excluded lanes BLOCKED_BY_RANKING — ranking IS the active gate | high |
| Aligns ranker with validator's DSR pipeline (one calibration story across both surfaces) | medium |

| argument against A2b-2 first | weight |
|---|---|
| Mode-B grandfather caveat — Phase 3a built on stale Sharpe; Phase 3b prerequisite adds 1-2 days | medium (mitigated by §5) |
| DSR as ranker is doctrine drift from "informational" stance | medium (mitigated by Shape A choice) |
| A2b-3 (Half-Kelly sizing) is also literature-grounded and addresses dollar-R uplift; could go first | low (Carver 2015 says size scales after selection — selection patch is logically prior) |

Conclusion: A2b-2 is the right next phase given Phase 3a's RANKING_MATERIAL verdict, with Shape A and Phase 3b prerequisite as the disciplined patch shape.

## 11. User approval gate — UPDATED 2026-04-18 post Phase 3b + final code-review

### History of options

The original gate offered four shapes:

A. **annual_r × DSR + floor** (multiplicative discount; recommended at scope time)
B. **DSR-only ranking** (strictest Bailey-LdP)
C. **raw + DSR floor gate** (binary filter)
D. **defer A2b-2 entirely** (skip to A2b-3 Half-Kelly)

Phase 3b satisfied the K1 binding prerequisite (Mode-A direction confirmed; selection still flips). The final code-review pass surfaced an honesty issue with the verdict label ("RANKING_MATERIAL_PRESERVED" → "PRESERVED-WITH-PARTIAL-AGREEMENT", 86%/67% overlap with Phase 3a). On reflection, the four-shape menu had a hidden assumption: that the empirical evidence forces a behavioral patch. It does not.

### Re-analysis: what the empirical state actually proves

The Phase 3a/3b finding (4-of-6 deployed lanes have DSR_canonical < 0.013) is **consistent with two opposing interpretations**:

1. **"DSR is right"** — current ranker selects noise lanes; expect mean reversion; ship Shape A or B to redirect.
2. **"DSR is wrong here"** — N_eff=21 (edge_families) overstates true selection bias OR var_sr is computed across noisy strategies that don't reflect the actual trial space; current ranker is fine; ship nothing.

Neither interpretation is fast-disambiguatable from the current data. Forward observation is the only adjudicator. Shapes A/B commit to interpretation #1 prematurely; Shape D wastes the audit work; Shape C is incoherent across its calibration range (default floor=0.05 catastrophically filters 4-5 of 6 deployed lanes since most have DSR < 0.05; low floor ≈ no-op).

### Shape E (NEW) — diagnostic-only DSR exposure — PICKED (literature-grounded)

Per-lane fields added to `LaneScore`, populated in `compute_lane_scores`:
- `dsr_score: float | None` — canonical `trading_app.dsr.compute_dsr` output at the validator's N_eff
- `sr0_at_rebalance: float | None` — canonical `compute_sr0` output (the noise-floor Sharpe)

Per-rebalance globals computed in `compute_lane_scores` and recorded in JSON:
- `n_eff_raw` — `COUNT(DISTINCT family_hash) FROM edge_families` (validator's current choice, raw M)
- `n_hat_eq9` — Bailey-LdP 2014 Equation 9 correlation-adjusted: `N̂ = ρ̂ + (1 - ρ̂)·M`
- `avg_rho_hat` — mean of off-diagonal entries in `compute_pairwise_correlation` over eligible lanes
- `var_sr_em` — `{E1: float, E2: float}` from canonical `experimental_strategies` per validator pattern

Per-lane DSR sensitivity also recorded in JSON: `dsr_at_n_eff_raw` + `dsr_at_n_hat_eq9` (so operators see both calibrations side-by-side).

`_effective_annual_r` UNCHANGED — selection and live deployment unchanged.
`regime-check` skill gains DSR + N̂ columns next to `trailing_expr` (skill content edit only).

**Literature grounding (added during plan-improve cycle):**

- `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md` Equation 9 (page 14-15): `N̂ = ρ̂ + (1-ρ̂)·M` is the correlation-adjusted independent-trial count. The validator's `COUNT(DISTINCT family_hash)` returns M (cluster count), not N̂. Recording both is the lit-correct disclosure.
- `docs/institutional/literature/lopez_de_prado_bailey_2018_false_strategy.md` Theorem 1 (page 3): expected max Sharpe under noise = `(1-γ)·Z⁻¹[1 - 1/K] + γ·Z⁻¹[1 - 1/(Ke)]` (where K = independent trials). Section "Application to our project" line 80-82 explicitly calculates that the project's best deployed lane (annualized Sharpe ~1.23) is **below the noise floor of ~3.87** under K=35,616 trials — Phase 3a/3b's RANKING_MATERIAL finding is a literature-confirmed restatement of this 2026-04-07 institutional grounding finding.
- `trading_app/dsr.py` line 35: "Until N_eff is properly estimated, DSR is INFORMATIONAL, not a hard gate." Shape E aligns with this doctrine literally — DSR becomes visible without becoming an objective.

**Open question Shape E does NOT resolve (deferred, logged for future audit):**

`var_sr_by_em` is computed across `experimental_strategies WHERE is_canonical=TRUE`, i.e., variance across SURVIVOR strategies. Bailey-LdP Equation 2 calls for variance across ALL trials in the search. The project's choice (post-survivor variance) is smaller, which makes DSR look more favorable than the strict Bailey-LdP intent. Shape E records the value used + this caveat in the JSON output; resolving the variance-population question is a separate audit (potential A2b-4 or doctrine update to `pre_registered_criteria.md` Criterion 5).

**Why Shape E is the institutionally correct call:**

1. Honors `dsr.py` line 35 ("DSR is INFORMATIONAL, not a hard gate") literally — DSR becomes visible without becoming an objective.
2. Zero deployment risk — current 6 deployed lanes unchanged; no first-rebalance gate (K6) required because no flips can happen.
3. Makes the Phase 3a/3b signal LIVE — every future rebalance records DSR alongside selection, building longitudinal evidence to disambiguate interpretation #1 vs #2.
4. Preserves optionality — if 6+ months of forward data shows low-DSR deployed lanes underperforming high-DSR alternatives, Shape A becomes empirically justified at that point (not before).
5. Smallest patch that uses the audit work — ~30 lines of production code + 2 tests vs Shape A's ~80 lines + 6 tests + binding K1/K6 gates. MIN_DSR_FLOOR magic constant is removed from the hot path entirely (not needed in Shape E).
6. Addresses the binary "DSR-right vs DSR-wrong" epistemic state honestly: instrument it; let forward data decide.

**Honest tradeoff for Shape E:**
- Doesn't act on Phase 3a/3b's directional finding NOW. Defers the Shape A/B strategic decision until forward evidence accumulates.
- Could be characterized as kicking the can. Counter: Shape E IS a decision — to act on instrumentation rather than theory. The wrong move would be ignoring the audit (Shape D) or committing to one theory without evidence (Shapes A/B).

### Shape E Stage-2 specification — IMPROVED PLAN

Files in scope (`scope_lock` for Stage-2):
- `trading_app/lane_allocator.py`
  - Add `dsr_score`, `sr0_at_rebalance`, `dsr_at_n_eff_raw`, `dsr_at_n_hat_eq9` to `LaneScore` dataclass (all `float | None = None` for backward compat).
  - Add new function `enrich_scores_with_dsr_diagnostics(scores, pairs, db_path=None)` mirroring the existing `enrich_scores_with_liveness` post-scoring pattern. Returns the per-rebalance globals dict (`n_eff_raw`, `n_hat_eq9`, `avg_rho_hat`, `var_sr_em`) for downstream consumers.
  - `save_allocation` accepts optional `dsr_globals` kwarg and writes both per-lane DSR fields (when populated on `LaneScore`) and the per-rebalance globals block. When omitted, behavior unchanged.
  - NO change to `_effective_annual_r`, `build_allocation`, `_compute_session_regime`, `_classify_status`, or any selection/ranking logic.
- `tests/test_trading_app/test_lane_allocator.py` — 4 new tests (T1-T4 below).
- `.claude/skills/regime-check/SKILL.md` — add DSR + N̂ columns to output template (content edit, no Python).
- `scripts/tools/rebalance_lanes.py` — **EXPANDED SCOPE 2026-04-18**: 3-line wiring change to call `enrich_scores_with_dsr_diagnostics(scores, corr_matrix)` between `compute_pairwise_correlation` and `save_allocation`, then pass returned `dsr_globals` to `save_allocation(..., dsr_globals=...)`. Reason: without this wiring the DSR fields stay `None` on real rebalances and the diagnostic is dead code. Blast radius: single CLI script, no library change, no caller change. Without expansion, the field-add infrastructure ships but isn't populated until someone manually calls the enrichment helper — defeats Shape E's "make DSR live" rationale.

Out of scope (deferred until forward data justifies promotion):
- Modifying `_effective_annual_r` — future Shape A patch
- DSR-as-gate / behavioral consumption — future Shape B/C patch
- `var_sr` recomputation against full-trial population (Bailey-LdP intent) — future audit (A2b-4 candidate)
- ONC-algorithm proper N_eff estimation (LdP 2020) — future enhancement
- First-rebalance flip-confirmation gate — moot under Shape E

Tests (pre-registered):
- **T1** `test_dsr_fields_populated_in_lane_score` — `compute_lane_scores` returns `LaneScore` with `dsr_score in [0, 1]`, `sr0_at_rebalance >= 0`, both `dsr_at_n_eff_raw` and `dsr_at_n_hat_eq9` populated for every scored lane that has the prerequisite stored stats. Synthetic-fixture DB.
- **T2** `test_dsr_canonical_equivalence` — for a fixture lane with known `(sharpe_ratio, sample_size, skewness, kurtosis_excess)`, `LaneScore.dsr_score` equals the result of calling `trading_app.dsr.compute_dsr(...)` directly with the same inputs. Guards against parallel-encoding regression.
- **T3** `test_dsr_fields_default_none_for_backward_compat` — constructing `LaneScore(...)` with the pre-patch keyword set (no DSR kwargs) succeeds and the new fields default to `None`. Guards external constructors.
- **T4** `test_save_allocation_writes_dsr_block` — `save_allocation` post-patch writes per-lane `dsr_score`, `sr0`, `dsr_at_n_eff_raw`, `dsr_at_n_hat_eq9` AND per-rebalance `n_eff_raw`, `n_hat_eq9`, `avg_rho_hat`, `var_sr_em`. Round-trips via `json.loads`.

Kill criteria for Shape E Stage-2 (4 K):
- **K1** Existing tests regression — `pytest tests/test_trading_app/test_lane_allocator.py` plus `pytest tests/test_research/` must remain green. Drift check `python -m pipeline.check_drift` must not add violations attributable to this commit.
- **K2** Phase 1 reproduction preservation — `python research/audit_allocator_rho_excluded.py` (when re-run with the lock cleared) must reproduce `lane_allocation.json` with the same PASS_TIED tolerance as today.
- **K3** Backward-compat on `LaneScore` constructor — T3 must pass first try; if any caller breaks, HALT.
- **K4** Consumer compat — `trading_app.prop_profiles.load_allocation_lanes(profile_id)` must continue returning the same `DailyLaneSpec` tuple post-patch (verified by re-reading lane_allocation.json after the patch and asserting set equality of returned spec keys). Already verified by code inspection: `load_allocation_lanes` uses `entry.get(...)` with defaults and only consumes `status`, `strategy_id`, `instrument`, `orb_label`, `p90_orb_pts` — new diagnostic fields are inert to it.

### Final user gate

**Stage-2 implementation begins ONLY after explicit user OK on this Shape E spec.**

If user wants Shape A/B/C/D instead: this scope is rewritten and the original options return to play. If user accepts Shape E: Stage-2 work begins with the 30-line patch + 2 tests + skill edit.

## 12. Provenance

- Phase 3a empirical → `docs/audit/results/2026-04-18-dsr-ranking-empirical-verification.md` (this scope's direct input)
- Phase 2a + A2b-1 PAUSE → `docs/audit/hypotheses/2026-04-18-a2b-1-regime-gate-filtered-patch-preregistered.md` (PAUSED status)
- Multi-phase pivot → `docs/plans/2026-04-18-multi-phase-audit-roadmap.md` § Addendum 2026-04-18
- Mode-A revalidation evidence → `docs/audit/results/2026-04-19-mode-a-revalidation-of-active-setups.md` (motivates Phase 3b prerequisite)
- DSR canonical implementation → `trading_app/dsr.py`
- Validator DSR pattern → `trading_app/strategy_validator.py:2180-2229`
- Pre-registered criteria reference → `docs/institutional/pre_registered_criteria.md` Criterion 5
