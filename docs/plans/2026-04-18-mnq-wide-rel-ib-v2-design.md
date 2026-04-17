# MNQ Wide-Relative-IB Filter — Design (v2)

**Date:** 2026-04-18
**Status:** DESIGN LOCKED — hypothesis pre-registered; replay NOT authored
**Hypothesis file:** `docs/audit/hypotheses/2026-04-18-mnq-wide-rel-ib-v2.yaml`
**Distinctness audit:** `docs/audit/results/2026-04-18-mes-wide-ib-distinctness-audit.md` (v2)
**Supersedes:** `docs/audit/hypotheses/2026-04-18-mes-mnq-wide-rel-ib.yaml` (v1, **WITHDRAWN** — OOS-shaped scope + wrong Rule 7 metric)

---

## Authority chain

- `CLAUDE.md`
- `RESEARCH_RULES.md`
- `TRADING_RULES.md`
- `docs/STRATEGY_BLUEPRINT.md` (NO-GO registry line 287 reopen flag)
- `docs/institutional/pre_registered_criteria.md` (12 locked criteria v2.7)
- `docs/institutional/literature/fitschen_2013_path_of_least_resistance.md` (mechanism baseline)
- `.claude/rules/backtesting-methodology.md` Rule 7 (canonical fire-correlation tautology metric)

---

## 1. What changed from v1

Code-review audit of v1 identified two structural issues:

**C1 — OOS-shaped pre-registration.** v1 peeked at 2026 OOS on 5 cells, then excluded CME_PRECLOSE from the pre-reg based on observed OOS direction flips. That is data-snooping — the pre-reg is not a pre-reg when its scope is set from the holdout window. v2 fixes by (a) not querying OOS in this audit and (b) INCLUDING MNQ CME_PRECLOSE in the pre-reg despite prior OOS knowledge, so Stage 1 is a real test.

**C2 — Wrong Rule 7 metric.** v1 reported `rho(abs_size, rel_width) = 0.71-0.75` (continuous-variable correlation between two derived numerical columns) and treated it as a Rule 7 tautology flag. `.claude/rules/backtesting-methodology.md` Rule 7 requires BOOLEAN fire correlation, not variable correlation. On the correct metric, fire rho = 0.13-0.55 across all 8 tested combos — well below the 0.70 threshold. No tautology; no "conditional-lift rescue" needed.

**Additional corrections:**
- v1 reported conditional lift by eyeballing deltas ("+0.135", "+0.139") without t-statistic. v2 uses Welch two-sample t-test with BH-FDR at K=8. Only 2 of 8 combos pass at q=0.05.
- v1 used t ≥ 3.00 (with-theory) on the basis of an unsupported "session-local size regime" narrative. v2 cites Fitschen 2013 Ch 3 as partial mechanism support (plausible one-step inference) and applies t ≥ 3.79 (without-theory) threshold per Criterion 4.
- v1 claimed "MNQ p=0.074 not independently significant" from NO-GO was "wrong." v2 says: the NO-GO's finding was at O5+O15 pooled under a different methodology; at O5-only with proper BH-FDR, the signal is MNQ-specific (not MES-specific), which is the OPPOSITE of what v1 claimed after reading the NO-GO.

---

## 2. What the test answers

> Within ORB_G5 fires on MNQ CME_PRECLOSE O5 and MNQ TOKYO_OPEN O5, does requiring `ORB size / 20d-trailing-same-session-mean >= 1.0` (WIDE_REL) add incremental signal that meets the 12 locked criteria of `pre_registered_criteria.md` under the t ≥ 3.79 (without-theory) pass threshold?

Not answered by this test:
- Whether WIDE_REL works on O15 (deferred to Stage 2)
- Whether the filter works on MES (distinctness did not reach p<0.05 on O5 — Stage 2 or separate pre-reg required)
- Whether other threshold values (1.2x, 1.5x) or trailing windows (10d, 30d) work
- Whether the filter would survive at K=425 cross-family framing (this is a K=6 family hypothesis)
- Whether CME_PRECLOSE will pass OOS (prior-branch OOS peek observed negative direction; included anyway for pre-reg process integrity)

---

## 3. Scope (locked, not shaped by OOS)

| Dimension | Value | Basis |
|---|---|---|
| Filter | `WIDE_REL_1.0X_20D AND ORB_G5` (conjunction) | Incumbent G5 plus session-local size-regime add |
| Formula | `orb_size / 20d_trailing_mean_same_session >= 1.0` | Zero look-ahead (strict lag-1 to lag-20 window) |
| Instruments | MNQ only | Only instrument with IS conditional-lift passing BH-FDR K=8 at O5 |
| Sessions | CME_PRECLOSE, TOKYO_OPEN | Both passed (or marginally passed) BH-FDR K=8 conditional-lift test |
| Aperture | O5 | Distinctness tested at O5; O15 reserved for Stage 2 |
| Entry | E2 CB=1 stop_mult=1.0 | Canonical baseline |
| RR | 1.0 / 1.5 / 2.0 | Standard triplet |
| **Cells** | **K=6** (2 lanes × 3 RR) | Tight, within MinBTL budget |

Excluded and locked:
- MES (did not pass BH-FDR K=8 conditional-lift at O5)
- MGC (architecturally dead per CLAUDE.md)
- MNQ COMEX_SETTLE, EUROPE_FLOW (conditional delta not significant)
- O15 aperture (Stage 2 reserved)
- Alternate thresholds or trailing windows (no sweep allowed)

---

## 4. Baseline comparison per cell

For each of the 6 cells, baseline = same lane with ORB_G5 filter only (no WIDE_REL). Delta = ExpR(WIDE_REL × G5) − ExpR(G5 alone), evaluated IS, then OOS.

| Cell | Instrument | Session | RR | Current deployed lane (if any) |
|---|---|---|---|---|
| 1-3 | MNQ | CME_PRECLOSE | 1.0/1.5/2.0 | X_MES_ATR60 at RR=1.0 deployed |
| 4-6 | MNQ | TOKYO_OPEN | 1.0/1.5/2.0 | COST_LT12 (RR1.0/1.5); ORB_G5 (RR1.5/2.0); deployed |

---

## 5. Pass rule (per-cell primary)

A cell passes only if ALL of the following hold:

1. BH-FDR q<0.05 at K=6 (family framing)
2. **Chordia t ≥ 3.79** on delta vs G5-only baseline (WITHOUT-theory threshold — Fitschen extension is plausible inference, not direct theory)
3. Walk-forward efficiency ≥ 0.50 across 5 expanding folds
4. N_OOS ≥ 30 per cell
5. OOS direction match: sign(delta_IS) == sign(delta_OOS)
6. OOS effect ratio: delta_OOS / delta_IS ≥ 0.40

Any single failure = cell DEAD. No rescue by dropping criteria.

---

## 6. Tautology guards (hard kill — run BEFORE primary eval)

Per Rule 7 canonical metric:
- For each cell, compute `|corr(cell_fire, alternative_filter_fire)|` on BOOLEAN fires.
- Alternatives checked on each lane (existing deployed filters only; not G5 which is a conjunction component):
  - MNQ CME_PRECLOSE: `X_MES_ATR60`
  - MNQ TOKYO_OPEN: `COST_LT12`, `ORB_G5` (standalone)
- If any |rho| > 0.70 → cell is DUPLICATE_FILTER, kill before primary eval.

Rule 7's escape hatch does not exist. v1's "conditional-lift rescue" narrative is withdrawn.

---

## 7. Replay script requirements

Script: `research/wide_rel_ib_pre_reg_replay.py` (to be authored only after user authorization)

Mandatory structure:
1. **Universe loading:** pull canonical `orb_outcomes` joined with `daily_features` for MNQ × {CME_PRECLOSE, TOKYO_OPEN} × O5 × E2 CB=1 stop_mult=1.0 × 3 RR; IS ≤ 2026-01-01, OOS > 2026-01-01. Use `pipeline.paths.GOLD_DB_PATH` (no hardcoded DB path).
2. **Feature derivation:** compute `WIDE_REL_1.0X_20D` in-script from `daily_features.orb_{session}_size` with strict LAG-1 trailing window. Fail-closed on NULL trailing mean.
3. **Tautology pre-gate (Rule 7 canonical metric):** for each cell, compute BOOLEAN fire correlation against every deployed filter on the same lane (not including G5 which is a conjunction component); kill if any |rho| > 0.70.
4. **Primary eval per cell:** compute delta_IS, delta_OOS, Welch t-statistic on the delta (with proper df), WFE (5-fold expanding), N_OOS, dir_match, eff_ratio.
5. **BH-FDR:** apply at K=6 and K_lane=3 frames; report both.
6. **Verdict matrix:** per-cell PASS/FAIL with the specific criterion that failed.
7. **Output:** `docs/audit/results/2026-04-18-mnq-wide-rel-ib-v2-replay.md` + companion JSON in `research/output/`.

Script MUST:
- Use `pipeline.paths.GOLD_DB_PATH` (import, don't hardcode)
- Pass `py_compile` and `ruff check` before commit
- NOT modify `pipeline/` or `trading_app/` code
- NOT add a new filter class to `trading_app/config.py` (filter is research-time-computed only; promotion to canonical filter is Stage 3 if applicable)

---

## 8. Dual verdict matrix (family-level)

After per-cell evaluation:

| Cells passing (of 6) | Verdict | Action |
|---|---|---|
| 4-6 | STRONG PASS | Proceed to Stage 2 (O15 or MES re-audit), propose promotion of passing cells through `strategy_validator` |
| 2-3 | STANDARD PASS | Promote passing cells only; do NOT proceed to Stage 2 |
| 1 | MARGINAL | Reevaluate under correlation gate at allocator; usually not deployed |
| 0 | NULL | Close family; do NOT rescue via aperture / threshold / window swap |

---

## 9. Scope hard boundaries

- No new state variables beyond `WIDE_REL_1.0X_20D`
- No aperture expansion beyond O5 (Stage 2 reserved)
- No session expansion beyond the 2 declared lanes
- No threshold sweep on 1.0x
- No trailing-window sweep on 20d
- No instrument expansion to MES without Stage 2 trigger
- No MGC (NO-GO)
- No ML wrapper
- No composite with other new filters
- No deployment claim without full 6-criterion primary pass
- No t-threshold relaxation to 3.00 without a new literature citation (Fitschen-as-extension does not qualify)

---

## 10. OOS-peek audit-trail note

Prior branch `research/overnight-2026-04-18` (commit `849d9097`, superseded) queried 2026+ OOS data on 5 cells and used the results to shape the v1 pre-reg scope. Observed OOS (for the record):
- MNQ CME_PRECLOSE WIDE+G5 RR1.0: OOS ExpR = −0.009 (N=27)
- MNQ TOKYO_OPEN WIDE+G5 RR1.0: OOS ExpR = +0.088 (N=37)
- MES cells: omitted from this v2 pre-reg scope for independent distinctness reasons

v2 pre-reg includes MNQ CME_PRECLOSE despite the prior peek showing negative OOS, because:
1. Excluding it based on OOS-peek observation would replicate v1's data-snooping sin.
2. Stage 1 replay will apply Amendment 2.7 criteria uniformly; if CME_PRECLOSE fails on OOS (expected), the kill is a real pre-reg outcome, not a scope-filtering decision.
3. MNQ TOKYO_OPEN is the primary candidate that didn't rely on OOS-peek knowledge for inclusion.

This is the cleanest available path given the prior peek.

---

## 11. Controls carried from canonical research protocol

- BH-FDR with multi-framing per `.claude/rules/backtesting-methodology.md` Rule 4
- OOS sacred per Mode A `pre_registered_criteria.md` v2.7 (2026-01-01 boundary)
- Rule 7 tautology check (BOOLEAN fire correlation) against every deployed filter on same lane
- Rule 8.1 extreme-fire-rate flag (fire rate < 5% or > 95% → flag)
- Rule 8.2 ARITHMETIC_ONLY flag (WR flat + ExpR moves only via payoff → cost-gate class)

---

## 12. Next order

1. User approves v2 design + pre-reg (this commit) OR withdraws.
2. If approved, write `research/wide_rel_ib_pre_reg_replay.py` on the same branch.
3. Run replay with all 6 cells + tautology guards.
4. Emit result MD.
5. If 2+ cells pass: promote individually via `strategy_validator`.
6. If 0-1 pass: close family, record kill.

---

## 13. What this design explicitly does NOT claim

- Does NOT claim WIDE_REL is a universal mechanism (MES + 2 MNQ sessions failed distinctness at p<0.05 O5)
- Does NOT claim this is a theory-supported strategy per Criterion 4 (plausible inference from Fitschen, not direct test)
- Does NOT claim the filter is independent of ORB_G5 on continuous-variable rho — the test is fire correlation per Rule 7
- Does NOT claim this is a deployment recommendation (requires full pass + validator + correlation gate at allocator)

It claims ONE thing: within G5 fires on the 2 declared MNQ lanes, WIDE_REL appears to add incremental IS signal that survived proper BH-FDR K=8 conditional-lift testing, and this pre-reg tests whether that claim meets the project's locked validation criteria under the conservative t ≥ 3.79 threshold.

Everything else is NULL until tested.
