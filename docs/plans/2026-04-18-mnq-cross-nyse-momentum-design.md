# MNQ CROSS_NYSE_MOMENTUM — Design

**Date:** 2026-04-18
**Status:** DESIGN LOCKED — hypothesis pre-registered; replay NOT authored
**Hypothesis file:** `docs/audit/hypotheses/2026-04-18-mnq-cross-nyse-momentum.yaml`
**Distinctness audit:** `docs/audit/results/2026-04-18-mnq-cross-nyse-momentum-distinctness.md`

---

## Authority chain

- `CLAUDE.md`
- `RESEARCH_RULES.md`
- `TRADING_RULES.md`
- `docs/institutional/pre_registered_criteria.md` (12 locked criteria v2.7)
- `docs/institutional/literature/fitschen_2013_path_of_least_resistance.md` (mechanism baseline)
- `.claude/rules/backtesting-methodology.md` (Rule 7 fire correlation, Rule 8 fire-rate bounds)
- `docs/plans/2026-04-11-cross-session-state-round3-memo.md` (prior-work disclosure, not scope-shaping)

---

## 1. What this test answers

> Does `CROSS_NYSE_MOMENTUM` — an existing filter class using NYSE_OPEN as prior session with 4-state TAKE/VETO logic — produce a positive expectancy delta over same-day valid-state baseline on MNQ US_DATA_1000 and MNQ NYSE_CLOSE at O5, under Mode A sacred-holdout discipline and t ≥ 3.79 (without-theory) significance threshold?

Not answered by this test:
- Whether CROSS_NYSE_MOMENTUM works on any other session (COMEX_SETTLE, NYSE_OPEN-as-current, etc.)
- Whether the VETO side of the state machine works (Stage 2 reserved)
- Whether the filter works at O15 or other apertures (Stage 2 reserved)
- Whether MES cross-confirmation holds (Stage 2 reserved)
- Whether the memo's Apr 11 lifts re-validate under Mode A (memo pre-dates Amendment 2.7)

---

## 2. Prior-work relationship

The `NYSE_OPEN → US_DATA_1000` pairing was scouted at Phase A on 2026-04-13 with 3/3 BH-FDR at K=18, but:
- Memo predates Amendment 2.7 Mode A (2026-01-01 sacred holdout)
- `experimental_strategies` table has zero rows — never went through formal validator pipeline
- Round-3 Pack A was DESIGNED (memo §Round-3 Pack A) but never executed
- OOS lifts in memo (1.8-3.4× IS lifts) trigger LEAKAGE_SUSPECT per quant-audit-protocol T3 — not a clean pass

The NYSE_CLOSE pairing was NOT scouted in the memo.

**This pre-reg does NOT recycle memo OOS numbers.** IS/OOS re-evaluation uses Mode A 2026-01-01 boundary. Memo is informational only in the audit trail.

---

## 3. Scope (locked per user Option C directive)

| Dimension | Value | Basis |
|---|---|---|
| Filter | `CROSS_NYSE_MOMENTUM` (existing class, prior_session=NYSE_OPEN) | No new code |
| Instruments | MNQ | MES/MGC out of user scope |
| Sessions | US_DATA_1000, NYSE_CLOSE | User directive + timing-verified look-ahead-free |
| Aperture | O5 | Matches CROSS_SGP_MOMENTUM canonical deployment + NYSE_OPEN's native aperture |
| Entry | E2 CB=1 stop_mult=1.0 | Canonical baseline |
| RR | 1.0 / 1.5 / 2.0 | Standard triplet |
| **Cells** | **K=6** (2 lanes × 3 RR) | Tight, within MinBTL budget |

Excluded and locked (all justified):
- MNQ CME_PRECLOSE (already REJECTED for CROSS_COMEX_MOMENTUM; CME_PRECLOSE OOS-flip pattern across VWAP_BP, wide-rel-IB)
- O15 aperture (Stage 2 reserved)
- Other sessions (out of user scope)
- MES, MGC (out of user scope; MGC also NO-GO)

---

## 4. Distinctness pre-check results (confirmed before pre-reg)

From the companion distinctness audit:

| Check | MNQ US_DATA_1000 | MNQ NYSE_CLOSE | Verdict |
|---|---|---|:---:|
| Rule 7 fire rho vs X_MES_ATR60 | −0.025 | no deployed filters | PASS |
| Rule 8.1 fire rate | 67.1% | 55.9% | PASS both |
| Look-ahead (prior-before-current) | 30min | 6.5h | PASS both |

---

## 5. Baseline comparison per cell

For each of the 6 cells, baseline = all days where the 4-state classification is valid (prior NYSE_OPEN data present AND current session data present). Candidate = TAKE-state subset.

- `delta_IS = ExpR(TAKE_state days, IS) - ExpR(all_valid_state days, IS)`
- `delta_OOS = ExpR(TAKE_state days, OOS) - ExpR(all_valid_state days, OOS)`

IS boundary: `trading_day < 2026-01-01`. OOS: `trading_day >= 2026-01-01`.

---

## 6. Pass rule (per-cell primary)

ALL of the following must hold for a cell to pass:

1. **BH-FDR q<0.05 at K=6** (primary family framing)
2. **Chordia t ≥ 3.79** (without-theory threshold per Criterion 4)
3. **Walk-forward efficiency WFE ≥ 0.50**
4. **N_OOS (TAKE-state days) ≥ 30** per cell
5. **OOS direction match:** sign(delta_IS) == sign(delta_OOS)
6. **OOS effect ratio:** 0.40 ≤ delta_OOS / delta_IS ≤ **3.00**

**Notable addition vs wide-rel-IB v2:** upper bound on effect ratio (3.00) added as LEAKAGE_SUSPECT guard. Rationale: memo's OOS/IS eff_ratio was 1.8-3.4× — if Mode A OOS reproduces that pattern, it's flagged and killed, not celebrated. This prevents treating the small-sample OOS spike (which sank both wide-rel-IB v2 TOKYO_OPEN and the Apr 11 memo story) as an edge.

Any single failure = cell DEAD. No rescue by dropping criteria.

---

## 7. Tautology guards (hard kill, run BEFORE primary eval)

Per Rule 7 canonical metric (BOOLEAN fire correlation):
- For each cell, compute `|corr(CROSS_NYSE_MOMENTUM fire, alternative_filter fire)|`
- Pre-computed in distinctness audit: −0.025 on US_DATA_1000 vs X_MES_ATR60; N/A on NYSE_CLOSE
- Replay re-runs this on canonical universe to confirm

Rule 7's escape hatch does not exist.

---

## 8. Replay script requirements

Script: `research/cross_nyse_momentum_pre_reg_replay.py` (to be authored only after user authorization)

Mandatory structure:
1. **Universe loading:** canonical `orb_outcomes` joined with `daily_features` for MNQ × {US_DATA_1000, NYSE_CLOSE} × O5 × E2 CB=1 stop_mult=1.0 × 3 RR. IS ≤ 2026-01-01, OOS > 2026-01-01. Use `pipeline.paths.GOLD_DB_PATH` (import, not hardcoded).
2. **State classification:** compute 4-state label per day per session, matching `CrossSessionMomentumFilter._compute_state()` (config.py:2591-2628).
3. **Tautology pre-gate (Rule 7):** fire correlation of CROSS_NYSE_MOMENTUM vs X_MES_ATR60 on US_DATA_1000. NYSE_CLOSE has no alternatives.
4. **Primary eval per cell:** delta_IS, delta_OOS, Welch t-stat on delta (two-sample with proper df), WFE, N_OOS, dir_match, eff_ratio (with upper bound 3.00).
5. **BH-FDR:** K=6 and K_lane=3 frames; report both.
6. **Verdict matrix:** per-cell PASS/FAIL with specific failing criterion.
7. **Output:** `docs/audit/results/2026-04-18-mnq-cross-nyse-momentum-replay.md` + JSON in `research/output/`.

Script MUST:
- Use `pipeline.paths.GOLD_DB_PATH`
- Pass `py_compile` and `ruff check` before commit
- NOT modify `pipeline/` or `trading_app/`
- NOT add a new filter class (existing `CROSS_NYSE_MOMENTUM` is already registered)

---

## 9. Dual verdict matrix (family-level)

| Cells passing (of 6) | Verdict | Action |
|---|---|---|
| 4-6 | STRONG PASS | Proceed to Stage 2 (VETO filter, O15, MES); propose promotion of passing cells through `strategy_validator` |
| 2-3 | STANDARD PASS | Promote passing cells only; Stage 2 not triggered |
| 1 | MARGINAL | Reevaluate under allocator correlation gate |
| 0 | NULL | Close family; do NOT rescue |

---

## 10. Controls carried from canonical research protocol

- BH-FDR with multi-framing per `.claude/rules/backtesting-methodology.md` Rule 4
- OOS sacred per Mode A `pre_registered_criteria.md` v2.7
- Rule 7 tautology check (BOOLEAN fire correlation)
- Rule 8.1 extreme-fire-rate flag — pre-checked (67.1% / 55.9%)
- **Leakage-suspect upper bound on eff_ratio (3.00)** — new vs wide-rel-IB v2, derived from T3 + memo OOS pattern

---

## 11. Scope hard boundaries

- No new state variables beyond existing `CROSS_NYSE_MOMENTUM` filter
- No aperture expansion beyond O5
- No session expansion beyond declared list
- No new sibling-state filters created from this pre-reg (VETO-only variant is Stage 2)
- No instrument expansion to MES (Stage 2 trigger only)
- No MGC (NO-GO)
- No ML wrapper
- No composite with other new filters
- No deployment claim without full 6-criterion primary pass
- No t-threshold relaxation to 3.00 without new literature citation
- No eff_ratio upper-bound relaxation — LEAKAGE_SUSPECT guard is binding

---

## 12. Next order

1. User approves design + pre-reg (this commit) OR withdraws.
2. If approved, write `research/cross_nyse_momentum_pre_reg_replay.py` on the same branch.
3. Run replay with all 6 cells + Rule 7 tautology guard.
4. Emit result MD.
5. If 2+ cells pass: promote individually via `strategy_validator`.
6. If 0-1 pass: close family, record kill.

---

## 13. What this design explicitly does NOT claim

- Does NOT claim CROSS_NYSE_MOMENTUM universal — 2 sessions only
- Does NOT claim memo Apr 11 numbers re-validate — informational only
- Does NOT claim the mechanism is theory-supported per Criterion 4 — extension of a deployed class, conservative t ≥ 3.79
- Does NOT claim VETO-state logic works — only TAKE side in this pre-reg
- Does NOT claim deployment — requires full pass + validator + allocator correlation gate

It claims ONE thing: the 4-state mechanism proven empirically via CROSS_SGP on EUROPE_FLOW, applied with NYSE_OPEN as prior to US_DATA_1000 and NYSE_CLOSE as current sessions on MNQ at O5, is testable under institutional criteria. Stage 1 replay will settle whether any cell clears the t ≥ 3.79 + Amendment 2.7 + LEAKAGE_SUSPECT-guarded bar.

Everything else is NULL until tested.
