"""MNQ endogenous-RR via IS-grid optimization on realized-EOD-MTM data — v1.

# scratch-policy: realized-eod

Pre-reg: docs/audit/hypotheses/2026-04-27-mnq-mfe-distribution-endogenous-rr-v1.yaml

For each (session, aperture) cell among the 4 v1-sign-flipped sessions
(NYSE_OPEN, US_DATA_1000, CME_PRECLOSE, COMEX_SETTLE) x (5m, 15m), test
whether IS-grid-optimal RR in {1.0, 1.5, 2.0, 2.5, 3.0, 4.0} produces
a measurably higher IS ExpR than the deployed default RR=1.5.

Pathway A family. K=8 cells. BH-FDR at q=0.05. Chordia t >= +3.00 with theory
(Carver Ch 9-10 + Bailey-LdP 2014). OOS reported descriptively (Mode A).

No capital action — observation only. Survivors require Pathway B K=1
verification under separate pre-reg per Amendment 3.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from pipeline.paths import GOLD_DB_PATH
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

PREREG_PATH = "docs/audit/hypotheses/2026-04-27-mnq-mfe-distribution-endogenous-rr-v1.yaml"
RESULT_DOC = Path("docs/audit/results/2026-04-27-mnq-mfe-distribution-endogenous-rr-v1.md")
INSTRUMENT = "MNQ"
SESSIONS = ["NYSE_OPEN", "US_DATA_1000", "CME_PRECLOSE", "COMEX_SETTLE"]
APERTURES = [5, 15]
RR_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
BASELINE_RR = 1.5
ALPHA_BH = 0.05
T_THRESHOLD = 3.0  # Chordia with theory


@dataclass
class CellResult:
    session: str
    aperture: int
    n_is: int
    n_oos: int
    is_expr_per_rr: dict[float, float]
    oos_expr_per_rr: dict[float, float]
    optimal_rr: float
    is_expr_optimal: float
    is_expr_baseline: float
    is_t_stat: float
    is_p_one_tailed: float
    bh_q: float
    oos_expr_optimal: float
    oos_expr_baseline: float
    h1_pass: bool


def _load_cell_paired(con, session: str, aperture: int) -> pd.DataFrame:
    """Pivot to one row per trading_day with pnl_r columns for each RR."""
    sql = """
    SELECT trading_day, rr_target, pnl_r
    FROM orb_outcomes
    WHERE symbol = ? AND orb_label = ? AND orb_minutes = ?
      AND entry_model = 'E2' AND confirm_bars = 1
      AND rr_target IN (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
      AND pnl_r IS NOT NULL
      AND outcome IN ('win', 'loss', 'scratch')
    """
    df = con.execute(sql, [INSTRUMENT, session, aperture]).df()
    if len(df) == 0:
        return df
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    pivoted = df.pivot_table(
        index="trading_day", columns="rr_target", values="pnl_r", aggfunc="first"
    ).reset_index()
    # Keep only rows where ALL 6 RR values are present (paired comparison requirement)
    pivoted = pivoted.dropna(subset=RR_GRID)
    return pivoted


def _bh_fdr(pvals: list[float], alpha: float) -> tuple[list[float], list[bool]]:
    """Benjamini-Hochberg FDR. Returns (q_values, pass_flags)."""
    n = len(pvals)
    if n == 0:
        return [], []
    order = np.argsort(pvals)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    p_arr = np.array(pvals)
    q_arr = p_arr * n / ranks
    # Monotonize
    sorted_q = q_arr[order]
    for i in range(len(sorted_q) - 2, -1, -1):
        sorted_q[i] = min(sorted_q[i], sorted_q[i + 1])
    q_arr_final = np.empty(n)
    q_arr_final[order] = sorted_q
    return q_arr_final.tolist(), [q < alpha for q in q_arr_final]


def run_cell(con, session: str, aperture: int) -> CellResult | None:
    df = _load_cell_paired(con, session, aperture)
    if len(df) == 0:
        return None
    is_mask = df["trading_day"] < pd.Timestamp(HOLDOUT_SACRED_FROM)
    is_df = df[is_mask].copy()
    oos_df = df[~is_mask].copy()
    if len(is_df) < 100:
        return None

    is_expr = {rr: float(is_df[rr].mean()) for rr in RR_GRID}
    oos_expr = {rr: float(oos_df[rr].mean()) if len(oos_df) > 0 else float("nan") for rr in RR_GRID}

    optimal_rr = max(RR_GRID, key=lambda r: is_expr[r])
    is_expr_opt = is_expr[optimal_rr]
    is_expr_base = is_expr[BASELINE_RR]

    # Paired t-test on (pnl_r at optimal RR) - (pnl_r at baseline RR=1.5)
    diff = is_df[optimal_rr].to_numpy() - is_df[BASELINE_RR].to_numpy()
    if len(diff) < 2 or np.std(diff, ddof=1) == 0:
        return None
    t_stat = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))
    # one-tailed (we hypothesize optimal > baseline)
    p_one = float(1.0 - stats.t.cdf(t_stat, len(diff) - 1))

    return CellResult(
        session=session,
        aperture=aperture,
        n_is=int(len(is_df)),
        n_oos=int(len(oos_df)),
        is_expr_per_rr=is_expr,
        oos_expr_per_rr=oos_expr,
        optimal_rr=optimal_rr,
        is_expr_optimal=is_expr_opt,
        is_expr_baseline=is_expr_base,
        is_t_stat=float(t_stat),
        is_p_one_tailed=p_one,
        bh_q=float("nan"),  # filled later
        oos_expr_optimal=oos_expr[optimal_rr] if not np.isnan(oos_expr[optimal_rr]) else float("nan"),
        oos_expr_baseline=oos_expr[BASELINE_RR] if not np.isnan(oos_expr[BASELINE_RR]) else float("nan"),
        h1_pass=False,
    )


def main() -> None:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    results: list[CellResult] = []
    for ses in SESSIONS:
        for apt in APERTURES:
            cell = run_cell(con, ses, apt)
            if cell is None:
                print(f"SKIP: {ses} {apt}m — insufficient data")
                continue
            results.append(cell)

    if not results:
        print("No cells produced results.")
        return

    pvals = [r.is_p_one_tailed for r in results]
    qs, passes = _bh_fdr(pvals, ALPHA_BH)
    for r, q, p in zip(results, qs, passes, strict=False):
        r.bh_q = q
        r.h1_pass = bool(p) and r.is_t_stat >= T_THRESHOLD and r.optimal_rr != BASELINE_RR

    # Print summary
    print()
    print(f"{'session':18s} {'apt':>4s} {'N_IS':>6s} {'opt_RR':>7s} "
          f"{'ExpR_opt':>9s} {'ExpR_1.5':>9s} {'t_IS':>6s} {'p_one':>8s} {'q_BH':>7s} "
          f"{'OOS_opt':>9s} {'OOS_1.5':>9s} {'H1':>4s}")
    print("-" * 130)
    for r in sorted(results, key=lambda x: (x.aperture, x.session)):
        verdict = "PASS" if r.h1_pass else "fail"
        print(
            f"{r.session:18s} {r.aperture:>4d} {r.n_is:>6d} {r.optimal_rr:>7.1f} "
            f"{r.is_expr_optimal:>+9.4f} {r.is_expr_baseline:>+9.4f} "
            f"{r.is_t_stat:>+6.2f} {r.is_p_one_tailed:>8.4f} {r.bh_q:>7.4f} "
            f"{r.oos_expr_optimal:>+9.4f} {r.oos_expr_baseline:>+9.4f} {verdict:>4s}"
        )

    # Write result MD
    n_pass = sum(1 for r in results if r.h1_pass)
    n_optimal_above_baseline = sum(1 for r in results if r.optimal_rr > BASELINE_RR)
    n_optimal_equal_baseline = sum(1 for r in results if r.optimal_rr == BASELINE_RR)

    md = f"""# MNQ endogenous-RR via IS-grid optimization — v1

**Pre-reg:** `{PREREG_PATH}`
**Runner:** `research/mnq_mfe_distribution_endogenous_rr_v1.py`
**Scratch policy:** realized-eod (Stage 5 fix landed; rebuild verified at >=99.5% population per `pipeline/check_drift.py::check_orb_outcomes_scratch_pnl`).

**Scope:** does the IS-grid-optimal RR target produce a measurably higher IS ExpR than fixed RR=1.5 on the 4 sessions whose v1-high-RR cells sign-flipped under realized-EOD MTM (NYSE_OPEN, US_DATA_1000, CME_PRECLOSE, COMEX_SETTLE) x (5m, 15m)?

**Outcome (verdict):** {n_pass} of {len(results)} cells pass H1 (BH-FDR q < {ALPHA_BH}, t >= +{T_THRESHOLD}, optimal_RR != baseline). H2 (descriptive): {n_optimal_above_baseline}/{len(results)} cells have optimal RR > baseline 1.5; {n_optimal_equal_baseline}/{len(results)} equal baseline.

## Per-cell table

| Session | Apt | N_IS | N_OOS | Optimal RR | ExpR opt | ExpR @ RR=1.5 | t_IS | p_one | q_BH | OOS opt | OOS @ 1.5 | H1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
"""
    for r in sorted(results, key=lambda x: (x.aperture, x.session)):
        verdict = "**PASS**" if r.h1_pass else "fail"
        md += (
            f"| {r.session} | {r.aperture} | {r.n_is} | {r.n_oos} | "
            f"{r.optimal_rr:.1f} | {r.is_expr_optimal:+.4f} | {r.is_expr_baseline:+.4f} | "
            f"{r.is_t_stat:+.2f} | {r.is_p_one_tailed:.4f} | {r.bh_q:.4f} | "
            f"{r.oos_expr_optimal:+.4f} | {r.oos_expr_baseline:+.4f} | {verdict} |\n"
        )

    md += "\n## ExpR by RR per cell (IS only)\n\n"
    md += "| Session | Apt |"
    for rr in RR_GRID:
        md += f" RR={rr} |"
    md += "\n|---|---|" + "---:|" * len(RR_GRID) + "\n"
    for r in sorted(results, key=lambda x: (x.aperture, x.session)):
        md += f"| {r.session} | {r.aperture} |"
        for rr in RR_GRID:
            cell = r.is_expr_per_rr[rr]
            mark = " **opt**" if rr == r.optimal_rr else ""
            md += f" {cell:+.4f}{mark} |"
        md += "\n"

    md += f"""

## Verdict and follow-on

- **{n_pass} of {len(results)} cells pass H1.** Decision rule:
  - >= 3 cells pass: continue (write Pathway B K=1 confirmatory pre-reg per Amendment 3.0)
  - 1-2 cells pass: park for follow-up
  - 0 cells pass: KILL — endogenous-RR hypothesis refuted on these sessions

  Resolved verdict: {'CONTINUE' if n_pass >= 3 else ('PARK' if n_pass >= 1 else 'KILL')}.

- No deployment from this scan. Survivors require Pathway B K=1 verification per
  pre_registered_criteria.md Amendment 3.0 before any allocator change.

## Mechanism note

Stage 6 found scratch_mean_R = +0.9955 on MNQ NYSE_OPEN 15m RR=4.0 — empirical evidence that
post-confirmed-break drift is positive on these sessions. Carver Ch 9-10 grounds the
hypothesis that optimal RR is a function of signal strength; the result above shows whether
that varies enough across sessions to warrant per-lane RR tuning.

## Reproduction

```bash
python research/mnq_mfe_distribution_endogenous_rr_v1.py
```

## Limitations

- Discretized RR grid: only 6 candidate RR values. True optimum could lie between grid points.
- Cell-level RR optimization: this is in-sample selection of a single parameter per cell. Bailey
  et al 2013 MinBTL bound 0.46 years (well within available 7-year IS window) — not at risk of overfit.
- Direction-pooled: longs and shorts averaged. Per-direction analysis deferred to follow-up.
- Vol-adaptive (per-trade RR conditional on regime): out of scope; deferred to Stage-2 pre-reg.
- OOS is descriptive only. Mode A holdout is sacred; do not tune against OOS.

## Cross-references

- Pre-reg: `{PREREG_PATH}`
- Stage 5 fix: `trading_app/outcome_builder.py` commit 68ee35f8
- Stage 6 sign-flip table: `docs/audit/results/2026-04-27-canonical-scratch-fix-downstream-impact.md`
- Criterion 13 scratch policy: `docs/institutional/pre_registered_criteria.md`
- Mechanism prior: `docs/institutional/mechanism_priors.md` § 11.5
"""

    RESULT_DOC.write_text(md, encoding="utf-8")
    print()
    print(f"Wrote {RESULT_DOC}")


if __name__ == "__main__":
    main()
