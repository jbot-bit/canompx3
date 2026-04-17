# MES + MNQ Wide-Relative-IB Filter — Design

**Date:** 2026-04-18
**Status:** DESIGN LOCKED — hypothesis pre-registered, replay not yet written
**Hypothesis file:** `docs/audit/hypotheses/2026-04-18-mes-mnq-wide-rel-ib.yaml`
**Distinctness audit:** `docs/audit/results/2026-04-18-mes-wide-ib-distinctness-audit.md`

## Authority chain

- `CLAUDE.md`
- `RESEARCH_RULES.md`
- `TRADING_RULES.md`
- `docs/STRATEGY_BLUEPRINT.md` (NO-GO registry line 287 reopen flag)
- `docs/institutional/pre_registered_criteria.md` (12 locked criteria)
- `.claude/rules/backtesting-methodology.md` (Rule 7 tautology, Rule 8 extreme fire rate)

---

## 1. Why this design exists

The NO-GO registry flagged `Narrow relative IB width (compression-take) — DEAD (reversed, MES signal persists)` and explicitly offered a reopen path: "MES-specific relative-width filter could be tested as hypothesis-scoped." Apr 2026 audit (script no longer in repo) found 16/425 cross-family BH-FDR survivors on MES; MNQ claimed "not independently significant" at p=0.074.

The 2026-04-18 distinctness audit tested the reopen claim with reproducible methodology and found:

- WIDE_REL is correlated at rho=0.71-0.75 with absolute size (close to the tautology threshold but NOT below it)
- Conditional-lift test within G5 fires shows WIDE_REL adds +0.10 to +0.14R incremental signal on 4 of 8 tested (instrument, session) combos
- NO-GO's "MNQ not significant" claim does NOT reproduce — MNQ CME_PRECLOSE and MNQ TOKYO_OPEN both show conditional lift comparable to MES
- OOS direction pre-check kills CME_PRECLOSE on both instruments (same archetype as VWAP_BP death this session), but preserves TOKYO_OPEN on both and MES COMEX_SETTLE as a probe
- WIDE_REL is uncorrelated with atr_20_pct (rho=-0.001) — it is NOT a disguised vol-state filter

This design proposes the minimum viable pre-reg test to confirm or kill the surviving scope.

---

## 2. What the test answers

> Within ORB_G5 fires on TOKYO_OPEN and MES COMEX_SETTLE, does requiring `ORB_size / 20d_trailing_same_session_mean >= 1.0` (WIDE_REL) add incremental signal that meets the 12 locked criteria of `pre_registered_criteria.md`?

Not answered by this test:
- Whether WIDE_REL works on O15 (deferred to Stage 2)
- Whether the filter works on CME_PRECLOSE (killed in pre-check OOS)
- Whether other threshold values (1.2x, 1.5x) or trailing windows (10d, 30d) work
- Whether the filter would survive at K=425 cross-family framing (this is a K=9 family hypothesis)

---

## 3. Scope (locked)

| Dimension | Value | Rationale |
|---|---|---|
| Filter | `WIDE_REL_1.0X_20D AND ORB_G5` (conjunction) | Incumbent G5 plus session-local regime add |
| Feature formula | `orb_size / 20d_trailing_mean_same_session >= 1.0` | Zero look-ahead (strict lag-1 to lag-20 window) |
| Instruments | MES, MNQ | MGC architecturally dead; MGC wide-IB would be new mechanism requiring separate pre-reg |
| Sessions | TOKYO_OPEN (MES+MNQ), COMEX_SETTLE (MES only) | Only sessions surviving OOS direction pre-check |
| Aperture | O5 | Distinctness audit was O5; O15 reserved for Stage 2 |
| Entry | E2 CB=1 stop_mult=1.0 | Canonical baseline |
| RR | 1.0 / 1.5 / 2.0 | Standard triplet |
| **Cells** | **9** (3 lanes × 3 RR) | Tight, well within MinBTL budget |

Excluded and locked:
- CME_PRECLOSE on any instrument (OOS direction flip in pre-check)
- EUROPE_FLOW (no baseline signal on MES)
- Any aperture other than O5 (Stage 2 reserved)
- Any threshold other than 1.0x (no sweep allowed)
- Any trailing window other than 20d (no sweep allowed)

---

## 4. Baseline comparison

For each of the 9 cells, baseline = the same lane with ORB_G5 alone (no WIDE_REL). Delta = ExpR(WIDE_REL × G5) − ExpR(G5 alone), evaluated IS, then OOS.

| Cell | Instrument | Session | RR | Current deployed lane (if any) |
|---|---|---|---|---|
| 1-3 | MES | TOKYO_OPEN | 1.0/1.5/2.0 | None (MES has no TOKYO_OPEN lanes) |
| 4-6 | MES | COMEX_SETTLE | 1.0/1.5/2.0 | None (MES has no COMEX_SETTLE lanes) |
| 7-9 | MNQ | TOKYO_OPEN | 1.0/1.5/2.0 | COST_LT12 (RR1.0/1.5 deployed); ORB_G5 RR1.5/2.0 also deployed |

---

## 5. Pass rule (per-cell primary)

A cell passes only if ALL of the following hold:
1. BH-FDR q<0.05 at K=9 (family framing)
2. Chordia t ≥ 3.00 on delta vs G5-only baseline (with-theory threshold)
3. Walk-forward efficiency ≥ 0.50 across 5 expanding folds
4. N_OOS ≥ 30 per cell
5. OOS direction match: sign(delta_IS) == sign(delta_OOS)
6. OOS effect ratio: delta_OOS / delta_IS ≥ 0.40

Any single failure = cell DEAD. No rescue by dropping criteria.

---

## 6. Tautology guards (hard kill — run BEFORE primary eval)

For each cell, compute `|rho(cell_firing, alternative_filter_firing)|` against:
- `X_MES_ATR60` (if this filter exists on the lane — MNQ TOKYO_OPEN RR1.0/1.5 have it deployed)
- `COST_LT12` (MNQ TOKYO_OPEN has this)
- Any OTHER deployed filter on the same (instrument, session) lane at the moment of testing

If any |rho| > 0.70 → cell is DUPLICATE_FILTER, kill before primary eval.

**Do NOT** compute rho against `ORB_G5` alone — that's the conditioning baseline, not an alternative. The correct distinctness test is conditional-lift (already done in the audit result), not firing correlation with G5.

---

## 7. Replay script requirements (when user authorizes writing it)

Script: `research/wide_rel_ib_pre_reg_replay.py`

Mandatory structure:
1. **Universe loading:** pull canonical `orb_outcomes` joined with `daily_features` for 2 instruments × 2 sessions × O5 × E2 CB=1 stop_mult=1.0 × 3 RR, IS ≤ 2026-01-01, OOS > 2026-01-01.
2. **Feature derivation:** compute `WIDE_REL_1.0X_20D` in-script from `daily_features.orb_{session}_size` with strict LAG-1 trailing window. Fail-closed on NULL trailing mean.
3. **Tautology pre-gate:** for each cell, compute rho against every deployed filter on the same (inst, sess) lane; kill if any > 0.70.
4. **Primary eval:** compute delta_IS, delta_OOS, t, WFE (5-fold expanding), N_OOS, dir_match, eff_ratio per cell.
5. **BH-FDR:** apply at K=9 and K_lane (3 per lane) frames; report both.
6. **Verdict matrix:** per cell PASS/FAIL with specific criterion that failed if FAIL.
7. **Output:** `docs/audit/results/2026-04-18-mes-mnq-wide-rel-ib-replay.md` + companion JSON in `research/output/`.

Script MUST:
- Use `pipeline.paths.GOLD_DB_PATH` (no hardcoded DB path)
- Pass `py_compile` and `ruff check` before commit
- Not modify pipeline/ or trading_app/ code
- Not add a new filter class to `trading_app/config.py` (filter is research-time-computed only; promotion to canonical filter is Stage 3 if applicable)

---

## 8. Dual verdict matrix (family-level)

After per-cell evaluation:

| Cells passing (of 9) | Verdict | Action |
|---|---|---|
| 5-9 | STRONG PASS | Proceed to Stage 2 (O15 extension), propose promotion of passing cells through strategy_validator |
| 3-4 | STANDARD PASS | Promote passing cells; do NOT proceed to Stage 2 |
| 1-2 | MARGINAL | Reevaluate if any pass is a MES cell with N_OOS > 50. Otherwise no deployment. |
| 0 | NULL | Close family; do NOT rescue via threshold sweep or aperture expansion |

---

## 9. Controls carried from canonical research protocol

- BH-FDR with honest K framing per `.claude/rules/backtesting-methodology.md` Rule 4
- OOS sacred per Mode A `pre_registered_criteria.md` v2.7 (2026-01-01 boundary)
- Rule 7 tautology check (T0) against every deployed filter on same lane
- Rule 8.1 extreme-fire-rate flag (fire rate < 5% or > 95% → flag)
- Rule 8.2 ARITHMETIC_ONLY flag (WR flat + ExpR moves only via payoff → cost-gate class)
- Rule 12 red-flag stops (|t| > 7, delta_IS > 0.6 → investigate temporal alignment)

---

## 10. Scope hard boundaries

- No new state variables beyond `WIDE_REL_1.0X_20D`
- No aperture expansion beyond O5 (Stage 2 reserved)
- No session expansion beyond the 3 declared lanes
- No threshold sweep on 1.0x
- No trailing-window sweep on 20d
- No ML wrapper
- No composite with other new filters
- No deployment claim without full 6-criterion primary pass
- No rescuing a failed cell by aperture/threshold swap

---

## 11. Outputs (when Stage 1 replay runs)

- `docs/audit/results/2026-04-18-mes-mnq-wide-rel-ib-replay.md`
- `research/output/wide_rel_ib_pre_reg_replay.json` (gitignored)
- `research/wide_rel_ib_pre_reg_replay.py` (committed before execution)

---

## 12. Next order

1. User approves this design + pre-reg. (THIS COMMIT does NOT authorize replay.)
2. If approved, write `research/wide_rel_ib_pre_reg_replay.py`.
3. Run replay with all 9 cells + tautology guards.
4. Emit result MD.
5. If 3+ cells pass primary + controls: promote individually via strategy_validator.
6. If 0-2 pass: close family, record kill in NO-GO registry addendum.

---

## 13. What this design explicitly does NOT claim

- Does NOT claim wide-relative-IB is a universal mechanism (MNQ COMEX_SETTLE and EUROPE_FLOW failed distinctness)
- Does NOT claim the +0.15 to +0.20 IS ExpR on CME_PRECLOSE reflects a real edge (OOS killed it)
- Does NOT claim the filter is independent of ORB_G5 on rho alone (rho=0.71-0.75 is too high for that claim)
- Does NOT claim this is a deployment recommendation (that requires full pass + validator + correlation gate at allocator)

It claims ONE thing: within G5 fires on the 3 declared lanes, WIDE_REL appears to add incremental signal that survived OOS direction pre-check, and this pre-reg tests whether that claim meets the project's locked validation criteria.

Everything else is NULL until tested.
