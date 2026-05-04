"""PR #51 DSR audit — Bailey Exhibit 4 effective-N correction (v2).

V1 (`mnq_pr51_dsr_audit_v1.py`) computed DSR assuming 105 INDEPENDENT trials
and flagged all 5 CANDIDATE_READYs as FAIL. That is the upper-bound-conservative
DSR. Bailey-López de Prado 2014 Appendix A.3 + Exhibit 4 give the correction
for CORRELATED trials:

    N̂ = ρ̂ + (1 − ρ̂)·M   (Bailey-LdP 2014 Eq. 9)

where M is the number of raw trials (105) and ρ̂ is the average pairwise
correlation between their return series. Since PR #51's 105 cells share many
trade-days (different apertures/RRs on the same session-day), ρ̂ is expected
to be materially > 0, which makes N̂ << 105, which lowers SR_0, which raises DSR.

This script computes ρ̂ empirically, derives N̂, re-runs DSR, and reports the
honest Phase 0 C5 verdict.

Grounding:
- `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md`
  §Exhibit 4 + §Appendix A.3 (pp 14-15, 20).

No capital action.
"""

from __future__ import annotations

import sys
from math import e as EULER_E
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from pipeline.paths import GOLD_DB_PATH
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

RESULT_DOC = Path("docs/audit/results/2026-04-21-mnq-pr51-dsr-audit-v2-effective-n.md")
INSTRUMENT = "MNQ"
APERTURES = [5, 15, 30]
RRS = [1.0, 1.5, 2.0]
MIN_N_IS = 100
DSR_THRESHOLD = 0.95

PR51_CANDIDATES = [
    (5, 1.0, "NYSE_OPEN"),
    (5, 1.5, "NYSE_OPEN"),
    (15, 1.0, "NYSE_OPEN"),
    (15, 1.0, "US_DATA_1000"),
    (15, 1.5, "US_DATA_1000"),
]

EULER_GAMMA = 0.5772156649


def _list_sessions(con: duckdb.DuckDBPyConnection, orb_minutes: int) -> list[str]:
    rows = con.execute(
        """
        SELECT DISTINCT orb_label FROM orb_outcomes
        WHERE symbol = ? AND orb_minutes = ? AND entry_model = 'E2'
          AND confirm_bars = 1 AND pnl_r IS NOT NULL
        ORDER BY orb_label
        """,
        [INSTRUMENT, orb_minutes],
    ).fetchall()
    return [r[0] for r in rows]


def _load_cell_series(con: duckdb.DuckDBPyConnection, orb_minutes: int, rr: float, session: str) -> pd.DataFrame:
    """Return DataFrame indexed by trading_day with pnl_r for this cell."""
    sql = """
    SELECT trading_day, pnl_r
    FROM orb_outcomes
    WHERE symbol = ? AND orb_label = ? AND orb_minutes = ?
      AND entry_model = 'E2' AND confirm_bars = 1 AND rr_target = ?
      AND pnl_r IS NOT NULL
      AND trading_day < ?
    """
    df = con.execute(sql, [INSTRUMENT, session, orb_minutes, rr, HOLDOUT_SACRED_FROM]).df()
    if len(df) == 0:
        return df
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    return df.set_index("trading_day").sort_index()


def _bailey_sr0(v_sr: float, n_trials: float) -> float:
    z_inv_1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z_inv_2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * EULER_E))
    return float(np.sqrt(v_sr) * ((1.0 - EULER_GAMMA) * z_inv_1 + EULER_GAMMA * z_inv_2))


def _dsr(sr: float, sr_0: float, T: int, skew: float, kurt: float) -> float:
    denom_sq = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * (sr**2)
    if denom_sq <= 0:
        return float("nan")
    denom = np.sqrt(denom_sq)
    z_stat = ((sr - sr_0) * np.sqrt(T - 1)) / denom
    return float(stats.norm.cdf(z_stat))


def main() -> int:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    # 1) Load all 105 cell series and compute per-cell stats
    cell_keys: list[tuple[int, float, str]] = []
    cell_series: dict[tuple[int, float, str], pd.DataFrame] = {}
    cell_sr: dict[tuple[int, float, str], float] = {}
    cell_skew: dict[tuple[int, float, str], float] = {}
    cell_kurt: dict[tuple[int, float, str], float] = {}
    cell_n: dict[tuple[int, float, str], int] = {}
    try:
        for apt in APERTURES:
            sessions = _list_sessions(con, apt)
            for rr in RRS:
                for s in sessions:
                    df = _load_cell_series(con, apt, rr, s)
                    if len(df) < MIN_N_IS:
                        continue
                    vals = df["pnl_r"].astype(float).to_numpy()
                    std = float(vals.std(ddof=1))
                    if std == 0:
                        continue
                    key = (apt, rr, s)
                    cell_keys.append(key)
                    cell_series[key] = df
                    cell_sr[key] = float(vals.mean()) / std
                    cell_skew[key] = float(stats.skew(vals, bias=False))
                    cell_kurt[key] = float(stats.kurtosis(vals, fisher=False, bias=False))
                    cell_n[key] = len(vals)
    finally:
        con.close()

    M = len(cell_keys)
    if M != 105:
        print(f"[warn] reproduced family K = {M}; PR #51 said K=105. Continuing.")

    # 2) Build a (trading_day × cell) matrix of daily mean pnl_r.
    #    Per-day aggregation: if a cell has multiple trades on the same day,
    #    take the day's mean pnl_r (representative per-day return for correlation).
    daily_rows: dict[pd.Timestamp, dict[int, float]] = {}
    all_days: set[pd.Timestamp] = set()
    for i, key in enumerate(cell_keys):
        ser = cell_series[key]["pnl_r"].groupby(level=0).mean()
        for day, val in ser.items():
            ts = pd.Timestamp(day)
            all_days.add(ts)
            daily_rows.setdefault(ts, {})[i] = float(val)

    all_days_sorted = sorted(all_days)
    mat = np.full((len(all_days_sorted), M), np.nan, dtype=float)
    for r, day in enumerate(all_days_sorted):
        row = daily_rows.get(day, {})
        for col, v in row.items():
            mat[r, col] = v

    # 3) Pairwise correlation on cell series (columns).
    #    Use pandas which handles NaN via pairwise-complete observations.
    df_mat = pd.DataFrame(mat, index=all_days_sorted)
    corr = df_mat.corr(method="pearson", min_periods=30)  # 30-day overlap min
    upper = np.triu(np.ones((M, M), dtype=bool), k=1)
    corr_values = corr.to_numpy()[upper]
    finite = corr_values[np.isfinite(corr_values)]
    if len(finite) == 0:
        raise SystemExit("No finite pairwise correlations — cannot compute rho_hat.")
    rho_hat = float(finite.mean())

    # 4) Effective N per Bailey Eq. 9
    n_eff = rho_hat + (1.0 - rho_hat) * M

    # 5) SR_0 with N_eff
    sr_values = np.array([cell_sr[k] for k in cell_keys], dtype=float)
    v_sr = float(sr_values.var(ddof=1))
    sr_0_m = _bailey_sr0(v_sr, M)
    sr_0_eff = _bailey_sr0(v_sr, n_eff)

    # 6) Per-cell DSR — raw M and corrected N_eff
    results: list[tuple[tuple[int, float, str], float, float, float, float, bool, bool]] = []
    for key in PR51_CANDIDATES:
        if key not in cell_series:
            continue
        sr = cell_sr[key]
        skew = cell_skew[key]
        kurt = cell_kurt[key]
        n_is = cell_n[key]
        dsr_raw = _dsr(sr, sr_0_m, n_is, skew, kurt)
        dsr_eff = _dsr(sr, sr_0_eff, n_is, skew, kurt)
        results.append((key, sr, dsr_raw, dsr_eff, n_eff, dsr_raw >= DSR_THRESHOLD, dsr_eff >= DSR_THRESHOLD))

    # 7) Render result
    RESULT_DOC.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    parts.append("# PR #51 DSR audit — v2 Bailey Exhibit 4 effective-N correction\n")
    parts.append(
        "**Authority:** Bailey-López de Prado 2014 Appendix A.3 + Exhibit 4 "
        "(Eq. 9: `N̂ = ρ̂ + (1 − ρ̂)·M`). See "
        "`docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md`.\n"
    )
    parts.append("**Phase 0 C5 threshold:** DSR >= 0.95.\n")
    parts.append(
        "**Follow-on to v1 (`2026-04-21-mnq-pr51-dsr-audit-v1.md`).** V1 computed DSR "
        "assuming 105 independent trials — upper-bound-conservative. This v2 applies the "
        "Bailey Eq. 9 correction for correlated trials to produce the honest effective-N DSR.\n"
    )
    parts.append("## Family + correlation")
    parts.append("")
    parts.append(f"- Raw trials M: **{M}**")
    parts.append(f"- Pairwise Pearson correlation (off-diagonal mean, min 30d overlap): **ρ̂ = {rho_hat:+.4f}**")
    parts.append(f"- Effective independent trials: **N̂ = ρ̂ + (1 − ρ̂)·M = {n_eff:.2f}**")
    parts.append(f"- Family V[SR] (ddof=1): `{v_sr:.5f}`")
    parts.append(f"- SR_0 at M=105 (raw): `{sr_0_m:+.5f}` (v1 value)")
    parts.append(f"- **SR_0 at N̂ (corrected): `{sr_0_eff:+.5f}`**")
    parts.append("")
    parts.append("## Per-cell DSR — raw M vs corrected N̂")
    parts.append("")
    parts.append("| Apt | RR | Session | Trade SR | DSR (M=105) | DSR (N̂) | Phase 0 C5 @ N̂ |")
    parts.append("|---:|---:|---|---:|---:|---:|---|")
    for key, sr, dsr_raw, dsr_eff, _neff, _pass_raw, pass_eff in results:
        verdict = "PASS" if pass_eff else "FAIL"
        parts.append(f"| {key[0]} | {key[1]} | {key[2]} | {sr:+.5f} | {dsr_raw:.4f} | {dsr_eff:.4f} | **{verdict}** |")
    parts.append("")
    n_pass = sum(1 for r in results if r[6])
    parts.append("## Summary")
    parts.append("")
    parts.append(f"- CANDIDATEs tested: {len(results)}")
    parts.append(f"- DSR PASS at N̂ (>= {DSR_THRESHOLD}): **{n_pass}**")
    parts.append(f"- DSR FAIL at N̂: {len(results) - n_pass}")
    parts.append("")
    parts.append("## Interpretation")
    parts.append("")
    if n_pass == len(results):
        parts.append(
            f"All {len(results)} PR #51 cells pass Phase 0 C5 under the honest effective-N "
            f"correction (N̂ = {n_eff:.2f}, ρ̂ = {rho_hat:+.4f}). The v1 FAIL was driven by "
            "the conservative M=105 assumption; cell-series are heavily correlated on "
            "shared trade-days so effective N is much smaller and SR_0 drops accordingly."
        )
    elif n_pass == 0:
        parts.append(
            f"None of the {len(results)} PR #51 cells pass Phase 0 C5 even after the "
            f"Exhibit 4 effective-N correction (ρ̂ = {rho_hat:+.4f}, N̂ = {n_eff:.2f}). "
            "The v1 finding hardens: these cells fail DSR under both raw-M and corrected-N̂ "
            "framings. Shadow-deployment remains institutionally blocked under Pathway A. "
            "A Pathway B K=1 pre-reg (theory-driven, single-cell) per "
            "pre_registered_criteria.md Amendment 3.0 is the remaining legitimate path."
        )
    else:
        parts.append(
            f"{n_pass} of {len(results)} PR #51 cells pass Phase 0 C5 under the Exhibit 4 "
            f"corrected N̂ = {n_eff:.2f} (ρ̂ = {rho_hat:+.4f}). Those passing cells become "
            "the first genuinely Phase-0-complete survivors from the PR #51 family; the "
            "failing cells must remain RESEARCH_SURVIVOR."
        )
    parts.append("")
    parts.append("## Methodology notes")
    parts.append("")
    parts.append(
        "- Pairwise correlation computed on per-day MEAN pnl_r per cell (aggregates multiple "
        "trades per day into one representative value). This maps each cell to a daily "
        "series so correlations are well-defined across cells that fire at different "
        "intra-day counts."
    )
    parts.append(
        "- `min_periods=30` on pairwise correlation to avoid spurious ρ̂ contributions from "
        "sparsely-overlapping cell pairs."
    )
    parts.append(
        "- ρ̂ is the MEAN of the off-diagonal upper-triangle finite entries of the correlation "
        "matrix, matching Bailey's 'average correlation between the trials'."
    )
    parts.append(
        "- Bailey Eq. 9 is an interpolation between ρ̂ → 0 (full independence, N̂ = M) and "
        "ρ̂ → 1 (full dependence, N̂ = 1). Linear in ρ̂."
    )
    parts.append("")
    parts.append("## Not done by this result")
    parts.append("")
    parts.append("- No writes to validated_setups / edge_families / lane_allocation / live_config.")
    parts.append("- No capital action.")
    parts.append("- Does NOT compute C11 (Monte Carlo account death) or C12 (live Shiryaev-Roberts).")

    RESULT_DOC.write_text("\n".join(parts), encoding="utf-8")

    print(f"M = {M}, rho_hat = {rho_hat:+.4f}, N_eff = {n_eff:.2f}")
    print(f"SR_0 (M=105) = {sr_0_m:+.5f}")
    print(f"SR_0 (N_eff) = {sr_0_eff:+.5f}")
    print("\nPer-cell DSR (raw M vs corrected N_eff):")
    for key, sr, dsr_raw, dsr_eff, _n, _pr, pass_eff in results:
        tag = "PASS" if pass_eff else "FAIL"
        print(
            f"  MNQ {key[0]}m RR={key[1]} {key[2]}: SR={sr:+.5f} DSR_raw={dsr_raw:.4f} DSR_Neff={dsr_eff:.4f} [{tag}]"
        )
    print(f"\nRESULT_DOC: {RESULT_DOC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
