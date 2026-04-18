# A2b-2 — DSR ranking patch — scope (Stage-1)

- phase: A2b-2 / Phase 3 of `docs/plans/2026-04-18-multi-phase-audit-roadmap.md`
- status: **SCOPE — awaiting user approval before Stage-2 implementation**
- author: audit/a2b-1-regime-gate-phase2 (branch retained for thread continuity; A2b-1 paused)
- created: 2026-04-18 (post Phase 3a empirical, post A2b-1 PAUSED pivot)
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

## 11. User approval gate

**Stage-2 (implementation) holds until user picks one (or proposes a fourth shape):**

A. **Proceed with Shape A (annual_r × DSR + floor)**, run Phase 3b first per K1, then Stage-2.
B. **Defer A2b-2** — skip ahead to A2b-3 (Half-Kelly sizing); revisit later. (Honest tradeoff: A2b-3 sizing on the wrong-selected lanes is lower EV than fixing selection first.)
C. **Promote Shape B (DSR-only)** — strictest Bailey-LdP interpretation; requires explicit doctrine update declaring DSR primary objective.
D. **Promote Shape C (raw + DSR floor gate)** — minimum-disruption variant; ranks unchanged, lanes only filtered out by DSR floor.

If the user picks A and Phase 3b confirms, Stage-2 begins. If Phase 3b reverses the verdict, this scope is rewritten before any code change.

## 12. Provenance

- Phase 3a empirical → `docs/audit/results/2026-04-18-dsr-ranking-empirical-verification.md` (this scope's direct input)
- Phase 2a + A2b-1 PAUSE → `docs/audit/hypotheses/2026-04-18-a2b-1-regime-gate-filtered-patch-preregistered.md` (PAUSED status)
- Multi-phase pivot → `docs/plans/2026-04-18-multi-phase-audit-roadmap.md` § Addendum 2026-04-18
- Mode-A revalidation evidence → `docs/audit/results/2026-04-19-mode-a-revalidation-of-active-setups.md` (motivates Phase 3b prerequisite)
- DSR canonical implementation → `trading_app/dsr.py`
- Validator DSR pattern → `trading_app/strategy_validator.py:2180-2229`
- Pre-registered criteria reference → `docs/institutional/pre_registered_criteria.md` Criterion 5
