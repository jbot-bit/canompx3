# MGC CPCV Audit — PASS 1 (audit-only, methodology-correct)

**Status:** PASS 1 COMPLETE. **PASS 2 COMPLETE (2026-05-29).**
Runner `research/mgc_cpcv_audit.py`; result `docs/audit/results/2026-05-29-mgc-cpcv-audit.md`;
tests `tests/test_research/test_mgc_cpcv_audit.py` (15/15). Read-only, no DB writes,
drift 150/0. **Verdict: 0 VALID, 6 CONDITIONAL, 0 UNVERIFIED, 0 WRONG** — none reach
deployment-grade (max full-path t=2.64 < 3.79; pooled power 0.56-0.74, all
DIRECTIONAL_ONLY). CPCV did NOT rescue the cells, exactly as pre-registered.

**PASS 2 METHODOLOGICAL FINDING (load-bearing):** the phi=5 full-coverage CPCV paths
are *mathematically identical* for a FIXED-OUTCOME backtest (no model refit) — each
path is a union of all N groups = the full series, and `pnl_r` does not change with
which split tested it. The genuine multi-path object is the per-split TEST-FOLD
distribution (15 folds), whose means DO vary. Full-coverage-path t (== pooled t) drives
the K4 t-gate; the fold distribution drives K1/K2. See result doc § METHODOLOGICAL FINDING.
**Mandate:** CPCV on MGC as a *methodology-correct* audit, NOT a threshold rescue.
No post-hoc threshold changes. No deployment claim. Same gates; better method.

---

## Why CPCV (not a gate change)

RULE 3.3 names CPCV as the sanctioned remedy for **underpowered short-horizon OOS**:
*"binary IS/OOS split is misspecified when OOS < ~20% of total sample."* MGC has
~1001 trading days (3.0yr real-micro). The binary trade-fraction split this session
ran is the tool RULE 3.3 flags as wrong for this regime — every MGC cell came back
STATISTICALLY_USELESS (max OOS power 0.41) precisely because the single OOS path is
too thin. CPCV converts the single path into φ multi-path distribution **without
loosening any threshold** — it is a better estimator of the same quantity.

## Literature grounding (PASS 1 read — verbatim, not memory)

- **Primary CPCV source:** AFML 2018 § 12.4 / § 12.4.1 — `docs/institutional/literature/lopez_de_prado_2018_afml_ch_3_7_8.md` lines 177-191. Formula: for T obs in N groups, testing-set size k groups → `φ[N,k] = k/N · C(N, N−k)` backtest paths. N=6, k=2 → C(6,4)=15 splits → **φ=5 paths**.
- **Purging + embargo:** AFML Ch 7 § 7.4 (same extract). Embargo `h ≈ 0.01·T` (LdP default). Purge train observations whose label window overlaps any test observation.
- **PBO (Probability of Backtest Overfitting):** Bailey et al 2013 `bailey_et_al_2013_pseudo_mathematics.md` (MinBTL) + the CPCV path distribution gives PBO via the rank-logit of in-sample-best vs OOS performance.
- **Threshold authority (UNCHANGED):** `pre_registered_criteria.md` 12 criteria — power floor, t≥3.79 (no theory), N≥100. Not relaxed for MGC.

## Exact MGC candidate set (from this session's powered wide scan)

Source: `research/powered_mgc_wide_scan.py` over canonical `orb_outcomes JOIN
daily_features` (triple-join, orb_minutes PINNED). The top promising-but-underpowered
cells (positive full ExpR + dir_match) — these are the CPCV inputs:

| # | session | om | rr | dir | filter | N_full | t_full | ExpR |
|---|---|---|---|---|---|---|---|---|
| 1 | US_DATA_830 | 30 | 2.0 | long | day_of_week==1 (Tue) | 100 | 2.64 | +0.328 |
| 2 | NYSE_OPEN | 30 | 2.0 | long | day_of_week==3 (Thu) | 103 | 2.56 | +0.237 |
| 3 | NYSE_OPEN | 30 | 1.0 | long | day_of_week==3 | 103 | 2.50 | +0.181 |
| 4 | SINGAPORE_OPEN | 30 | 2.0 | long | day_of_week==4 (Fri) | 98 | 2.40 | +0.312 |
| 5 | EUROPE_FLOW | 30 | 2.0 | long | atr_20_pct>=60 | 296 | 2.27 | +0.174 |
| 6 | LONDON_METALS | 30 | 1.5 | long | overnight_range_pct>=80 | 124 | 2.12 | +0.216 |

**Note on K:** the wide scan tested 1,992 MGC cells. CPCV on the top-6 is a
*confirmatory* re-test of survivors, NOT new discovery — but the PBO / DSR accounting
MUST carry K=1992 as the honest selection budget (these 6 were selected from 1992).
Per RULE 4 / MinBTL: K=1992 with 3.0yr horizon is well over the Bailey bound, so the
EXPECTED PASS 2 outcome is UNVERIFIED even under CPCV. This audit tests *whether the
multi-path estimator changes that* — it likely will not, and that is a valid finding.

## Eligibility / costs / canonical query (PASS 1 confirmed)

- **Costs:** MGC `total_friction = 5.74` (canonical `pipeline.cost_model.COST_SPECS['MGC']`).
  `orb_outcomes.pnl_r` is already net of canonical friction — CPCV scores on `pnl_r`
  directly; no re-costing.
- **Canonical query:** `orb_outcomes o JOIN daily_features d ON (trading_day, symbol,
  orb_minutes)` — the triple-join (RULE 9). MGC E2 CB1 only.
- **No 2026 holdout tuning — CONFIRMED:** CPCV partitions the WHOLE sample into N
  groups; it does NOT carve a sacred 2026 window and does NOT tune against it. The
  sacred-holdout rule (`HOLDOUT_SACRED_FROM`) is a *discovery* constraint; CPCV here is
  a confirmatory estimator over the full history. No parameter is optimized against any
  fold. Thresholds are read-only constants from `pre_registered_criteria.md`.

## CPCV design (locked for PASS 2 — do NOT redesign)

- **Groups:** N=6, contiguous, no shuffle (temporal). Group sizes per AFML floor formula.
- **Test size:** k=2 → C(6,4)=15 train/test splits → **φ=5 backtest paths**.
- **Purge:** drop train trades whose trading_day falls within the test groups' span.
- **Embargo:** `h = 0.01·T` trading days after each test group (LdP default). For ORB
  the label window is intraday (entry→same-day exit), so overlap is minimal, but apply
  the embargo for serial-correlation safety per § 7.4.
- **Per-path score:** ExpR, per-trade Sharpe, one-sample t on path pnl_r.
- **Aggregate:** median ExpR/Sharpe across 5 paths, dispersion (IQR), WORST path,
  fraction of paths with ExpR>0.
- **PBO:** rank in-sample-selected-best vs its OOS path performance across the φ paths;
  report PBO probability (logit of OOS underperformance).

## Kill criteria (LOCKED — no post-hoc change)

- K1: median path ExpR <= 0 → UNVERIFIED (no edge survives multi-path).
- K2: worst path ExpR < −0.05 AND >40% of paths negative → WRONG (overfit single-path artifact).
- K3: PBO > 0.50 → WRONG (more likely overfit than real).
- K4: median path t < 3.79 (strict Chordia, no-theory — UNCHANGED threshold) AND
  power across pooled paths < 0.50 → UNVERIFIED.
- VALID only if: median ExpR>0 AND median t≥3.79 AND PBO<0.50 AND worst path not
  catastrophic AND pooled-path power≥0.50. (Expected: NOT met, given K=1992/3.0yr.)

## PASS 2 execution plan (fresh session)

1. New `research/mgc_cpcv_audit.py` — import `pipeline.cost_model`, canonical join,
   implement CPCV per the LOCKED design above. Delegate purging/embargo to a Ch-7-faithful
   splitter; do NOT re-encode filter logic (`research.filter_utils.filter_signal`).
2. Run on canonical gold.db (read-only). NO DB writes.
3. Emit per-path table + aggregate + PBO to `docs/audit/results/2026-05-29-mgc-cpcv-audit.md`.
4. Classify each of the 6 candidates VALID/CONDITIONAL/UNVERIFIED/WRONG.
5. Code-review the script (canonical delegation, no look-ahead, PBO math).

## Uncommitted artifacts from PASS 1 / prior turns (resume context)

- `research/powered_oos_graveyard_resweep.py` (new, reviewed A-)
- `research/powered_mgc_wide_scan.py` (new, reviewed A-, 2 fixes applied)
- `docs/audit/results/2026-05-29-powered-oos-graveyard-resweep-and-mgc-wide-scan.md`
- `docs/audit/results/2026-05-29-powered-mgc-wide-scan.csv`
- memory: `feedback_powered_oos_holdout_at_discovery_no_calendar_wait_2026_05_29.md` (+ MEMORY.md index)
- All UNCOMMITTED on main. Read-only snapshot used: `%TEMP%/gold_ro_snapshot.db` (gold.db
  writer-locked by live backfill PID 56072 — do NOT kill).
