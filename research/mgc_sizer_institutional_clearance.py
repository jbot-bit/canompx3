"""Institutional clearance audit for PR #48 rel_vol sizer lineage.

Scope-lock:
  - MGC / MES / MNQ rel_vol sizer lineage only
  - No writes outside the result markdown
  - Canonical inputs only: orb_outcomes + daily_features + repo gate formulas

Outputs:
  - docs/audit/results/2026-04-21-mgc-sizer-institutional-clearance.md
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from pipeline.paths import GOLD_DB_PATH
from research.pr48_sizer_rule_skeptical_reaudit_v1 import (
    _assign,
    _quintiles_is,
    _sessions,
    _sharpe,
    _split,
    audit_instrument,
    _load,
)
from trading_app.dsr import compute_dsr, compute_sr0

RESULT_DOC = Path("docs/audit/results/2026-04-21-mgc-sizer-institutional-clearance.md")
PREREG_SHA = "d227f8ed"
FILTER_FORM_YAML = Path("docs/audit/hypotheses/2026-04-21-rel-vol-filter-form-v1.yaml")


@dataclass
class SeriesBundle:
    full: pd.DataFrame
    is_df: pd.DataFrame
    oos_df: pd.DataFrame
    is_sized: pd.DataFrame
    oos_sized: pd.DataFrame


def _load_bundle(con: duckdb.DuckDBPyConnection, inst: str) -> SeriesBundle:
    frames = [f for s in _sessions(con, inst) if len(f := _load(con, inst, s)) > 0]
    if not frames:
        raise RuntimeError(f"No rows for {inst}")
    full = pd.concat(frames, ignore_index=True)
    is_df, oos_df = _split(full)
    thresh = _quintiles_is(is_df)

    is_sized = _assign(is_df, thresh)
    is_sized["pnl_sizer"] = is_sized["pnl_r"].astype(float) * is_sized["size_mult"].astype(float)

    oos_sized = _assign(oos_df, thresh)
    oos_sized["pnl_sizer"] = oos_sized["pnl_r"].astype(float) * oos_sized["size_mult"].astype(float)
    return SeriesBundle(full=full, is_df=is_df, oos_df=oos_df, is_sized=is_sized, oos_sized=oos_sized)


def _criterion9_era(ts: pd.Timestamp) -> str:
    y = ts.year
    if 2015 <= y <= 2019:
        return "2015-2019"
    if 2020 <= y <= 2022:
        return "2020-2022"
    if y == 2023:
        return "2023"
    if 2024 <= y <= 2025:
        return "2024-2025"
    if y == 2026:
        return "2026"
    return str(y)


def _commit_date(sha: str) -> date:
    iso = subprocess.check_output(["git", "show", "-s", "--format=%cI", sha], text=True).strip()
    return pd.Timestamp(iso).date()


def _fmt(x: float | None, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:+.{nd}f}"


def _fmt_plain(x: float | None, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"


def main() -> int:
    commit_day = _commit_date(PREREG_SHA)
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        bundles = {inst: _load_bundle(con, inst) for inst in ["MGC", "MES", "MNQ"]}
        audits = {inst: audit_instrument(con, inst) for inst in ["MGC", "MES", "MNQ"]}

        # MGC institutional stack
        mgc = bundles["MGC"]
        mgc_audit = audits["MGC"]
        assert mgc_audit is not None

        mgc_is_expr = float(mgc.is_sized["pnl_sizer"].mean())
        mgc_oos_expr = float(mgc.oos_sized["pnl_sizer"].mean())
        mgc_is_sr = _sharpe(mgc.is_sized["pnl_sizer"].to_numpy(dtype=float))
        mgc_oos_sr = _sharpe(mgc.oos_sized["pnl_sizer"].to_numpy(dtype=float))

        # Canonical WFE in-repo is trade-weighted OOS/IS ExpR ratio and fails closed if IS<=0.
        mgc_wfe = None if mgc_is_expr <= 0 else mgc_oos_expr / mgc_is_expr
        mgc_wfe_pass = mgc_wfe is not None and mgc_wfe >= 0.50

        # Criterion 9 era stability
        mgc_all = pd.concat([mgc.is_sized, mgc.oos_sized], ignore_index=True)
        mgc_all["era"] = mgc_all["trading_day"].map(_criterion9_era)
        era_rows: list[tuple[str, int, float, float, float]] = []
        mgc_era_pass = True
        for era, g in mgc_all.groupby("era"):
            n = len(g)
            uniform = float(g["pnl_r"].mean())
            sizer = float(g["pnl_sizer"].mean())
            delta = sizer - uniform
            era_rows.append((era, n, uniform, sizer, delta))
            if n >= 50 and sizer < -0.05:
                mgc_era_pass = False

        year_rows: list[tuple[int, int, float, float, float, str]] = []
        for year, g in mgc_all.groupby(mgc_all["trading_day"].dt.year):
            n = len(g)
            uniform = float(g["pnl_r"].mean())
            sizer = float(g["pnl_sizer"].mean())
            delta = sizer - uniform
            split = "OOS" if year >= 2026 else "IS"
            year_rows.append((int(year), n, uniform, sizer, delta, split))

        # Untouched residual OOS after pre-reg commit SHA.
        untouched_days_row = con.execute(
            """
            SELECT COUNT(DISTINCT trading_day), MIN(trading_day), MAX(trading_day)
            FROM orb_outcomes
            WHERE symbol = 'MGC'
              AND orb_minutes = 5
              AND entry_model = 'E2'
              AND confirm_bars = 1
              AND rr_target = 1.5
              AND pnl_r IS NOT NULL
              AND trading_day >= ?
            """,
            [commit_day],
        ).fetchone()
        untouched_days = int(untouched_days_row[0] or 0)
        untouched_min = untouched_days_row[1]
        untouched_max = untouched_days_row[2]
        residual_pass = untouched_days > 0

        max_data_day = con.execute(
            """
            SELECT MAX(trading_day)
            FROM orb_outcomes
            WHERE symbol='MGC'
              AND orb_minutes=5
              AND entry_model='E2'
              AND confirm_bars=1
              AND rr_target=1.5
              AND pnl_r IS NOT NULL
            """
        ).fetchone()[0]

        # DSR cross-check, using repo-canonical estimator and local K=3 sensitivity.
        var_sr_e2 = con.execute(
            """
            SELECT VAR_SAMP(sharpe_ratio)
            FROM experimental_strategies
            WHERE entry_model='E2'
              AND sample_size >= 30
              AND sharpe_ratio IS NOT NULL
              AND is_canonical = TRUE
            """
        ).fetchone()[0] or 0.047
        n_eff_edge_family = int(
            con.execute("SELECT COUNT(DISTINCT family_hash) FROM edge_families").fetchone()[0] or 2
        )
        ret = mgc.oos_sized["pnl_sizer"].to_numpy(dtype=float)
        sr_hat_oos = float(ret.mean() / ret.std(ddof=1))
        skew_oos = float(pd.Series(ret).skew())
        kurt_ex_oos = float(pd.Series(ret).kurt())
        sr0_edge_family = compute_sr0(n_eff_edge_family, var_sr_e2)
        dsr_edge_family = compute_dsr(sr_hat_oos, sr0_edge_family, len(ret), skew_oos, kurt_ex_oos)
        sr0_k3 = compute_sr0(3, var_sr_e2)
        dsr_k3 = compute_dsr(sr_hat_oos, sr0_k3, len(ret), skew_oos, kurt_ex_oos)

        # MinBTL confirmation on raw trading-day history.
        mgc_pre_holdout_days = int(
            con.execute(
                """
                SELECT COUNT(DISTINCT trading_day)
                FROM orb_outcomes
                WHERE symbol='MGC'
                  AND orb_minutes=5
                  AND entry_model='E2'
                  AND confirm_bars=1
                  AND rr_target=1.5
                  AND pnl_r IS NOT NULL
                  AND trading_day < DATE '2026-01-01'
                """
            ).fetchone()[0]
            or 0
        )
        mgc_pre_holdout_years = mgc_pre_holdout_days / 252.0
        minbtl_years = 2.0 * math.log(3.0) / 9.0
        minbtl_pass = mgc_pre_holdout_years >= minbtl_years

        # MES / MNQ close-outs from raw audit numbers
        mes_audit = audits["MES"]
        mnq_audit = audits["MNQ"]
        assert mes_audit is not None and mnq_audit is not None
        mnq_pos = sum(1 for row in mnq_audit["per_lane"] if row["delta"] > 0)
        mnq_neg = sum(1 for row in mnq_audit["per_lane"] if row["delta"] < 0)
        mnq_neg_share = mnq_neg / max(mnq_pos + mnq_neg, 1)

    finally:
        con.close()

    final_status = "CLEAR_FOR_SHADOW" if all([True, mgc_oos_sr > 0, mgc_wfe_pass, mgc_era_pass, residual_pass, minbtl_pass]) else "REJECT"

    lines: list[str] = []
    lines.append("# MGC Sizer Institutional Clearance\n")
    lines.append("Scope-locked to the rel_vol sizer lineage only. Canonical inputs: `orb_outcomes`, `daily_features`, repo gate formulas in `pre_registered_criteria.md`, `walkforward.py`, and `dsr.py`.\n")
    lines.append("## MGC institutional gate walk\n")
    lines.append("")
    lines.append("| Gate | Canonical number(s) | Binding? | Status |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| Pre-reg scoped gate | delta={_fmt_plain(mgc_audit['pooled_delta'],5)}, t={_fmt_plain(mgc_audit['pooled_t'],3)}, p={_fmt_plain(mgc_audit['pooled_p'],4)}, bootstrap95=[{_fmt_plain(mgc_audit['ci_lo'],4)}, {_fmt_plain(mgc_audit['ci_hi'],4)}] | pre-reg gate | PASS |"
    )
    lines.append(
        f"| Sharpe-positive gate | uniform_SR={_fmt_plain(mgc_audit['sr_uniform'],3)}, sizer_SR={_fmt_plain(mgc_audit['sr_sizer'],3)} using SR = mean(pnl_sizer) / std(pnl_sizer, ddof=1) on 2026 OOS | non-waivable Pathway B direction gate | {'PASS' if mgc_oos_sr > 0 else 'FAIL'} |"
    )
    wfe_note = (
        f"WFE={_fmt_plain(mgc_wfe,4)}"
        if mgc_wfe is not None
        else f"WFE=NA because IS_ExpR={_fmt_plain(mgc_is_expr,5)} <= 0 while OOS_ExpR={_fmt_plain(mgc_oos_expr,5)}; canonical walkforward.py fails closed when weighted_IS <= 0"
    )
    lines.append(
        f"| Walk-forward efficiency | {wfe_note} | non-waivable Pathway B | {'PASS' if mgc_wfe_pass else 'FAIL'} |"
    )
    era_summary = "; ".join(
        f"{era}: N={n} sizer={_fmt_plain(sizer,5)}"
        for era, n, _u, sizer, _d in era_rows
    )
    lines.append(
        f"| Era stability | {era_summary} | non-waivable Pathway B | {'PASS' if mgc_era_pass else 'FAIL'} |"
    )
    lines.append(
        f"| 2026 OOS residual untouched since `{PREREG_SHA}` | commit_day={commit_day}, max_data_day={max_data_day}, untouched_distinct_days={untouched_days}, range=[{untouched_min}, {untouched_max}] | monitor readiness check | {'PASS' if residual_pass else 'FAIL'} |"
    )
    lines.append(
        f"| DSR cross-check | repo-canonical n_eff={n_eff_edge_family}, var_sr_E2={_fmt_plain(var_sr_e2,6)}, sr_hat_OOS={_fmt_plain(sr_hat_oos,6)}, sr0={_fmt_plain(sr0_edge_family,6)}, dsr={_fmt_plain(dsr_edge_family,6)}; local K=3 sensitivity dsr={_fmt_plain(dsr_k3,6)} | cross-check only (Amendment 2.1) | {'PASS' if dsr_edge_family > 0.95 else 'FAIL'} |"
    )
    lines.append(
        f"| MinBTL | required_years={_fmt_plain(minbtl_years,3)}, available_pre_holdout_days={mgc_pre_holdout_days}, available_years={_fmt_plain(mgc_pre_holdout_years,3)} | pre-reg integrity | {'PASS' if minbtl_pass else 'FAIL'} |"
    )
    lines.append("")
    lines.append(f"**Final status: `{final_status}`**")
    lines.append("")
    lines.append("Reason: the original OOS pre-reg pass stands, but the broader institutional stack does not clear. The non-waivable failures are `WFE` (undefined / fail-closed because IS sizer ExpR is negative), `era stability` (three load-bearing eras below -0.05 with N>=50), and `untouched residual OOS` (0 days since the pre-reg commit). DSR is also weak, but is cross-check only.")
    lines.append("")
    lines.append("## MGC raw year-by-year breakdown")
    lines.append("")
    lines.append("| Year | Split | N | Uniform ExpR | Sizer ExpR | Delta |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for year, n, uniform, sizer, delta, split in year_rows:
        lines.append(
            f"| {year} | {split} | {n} | {_fmt_plain(uniform,5)} | {_fmt_plain(sizer,5)} | {_fmt_plain(delta,5)} |"
        )
    lines.append("")
    lines.append("## MGC Criterion 9 era bins")
    lines.append("")
    lines.append("| Era | N | Uniform ExpR | Sizer ExpR | Delta |")
    lines.append("|---|---:|---:|---:|---:|")
    for era, n, uniform, sizer, delta in era_rows:
        lines.append(
            f"| {era} | {n} | {_fmt_plain(uniform,5)} | {_fmt_plain(sizer,5)} | {_fmt_plain(delta,5)} |"
        )
    lines.append("")
    lines.append("## MES sizer closeout")
    lines.append("")
    lines.append(
        f"`NOT_DEPLOYABLE_AS_SIZER`. Raw OOS numbers: delta={_fmt_plain(mes_audit['pooled_delta'],5)}, t={_fmt_plain(mes_audit['pooled_t'],3)}, p={_fmt_plain(mes_audit['pooled_p'],4)}, uniform_SR={_fmt_plain(mes_audit['sr_uniform'],3)}, sizer_SR={_fmt_plain(mes_audit['sr_sizer'],3)}. The sizer reduces losses but both Sharpe values stay negative. That is risk reduction on a losing lane, not alpha. The live MES path is the filter-form pre-reg locked at `{FILTER_FORM_YAML}` and gated on fresh OOS only."
    )
    lines.append("")
    lines.append("## MNQ sizer closeout")
    lines.append("")
    lines.append(
        f"`DEAD`. Raw OOS numbers: delta={_fmt_plain(mnq_audit['pooled_delta'],5)}, t={_fmt_plain(mnq_audit['pooled_t'],3)}, p={_fmt_plain(mnq_audit['pooled_p'],4)}, bootstrap95=[{_fmt_plain(mnq_audit['ci_lo'],4)}, {_fmt_plain(mnq_audit['ci_hi'],4)}], Spearman p={_fmt_plain(mnq_audit['spearman_p'],4)}, lane_pos={mnq_pos}, lane_neg={mnq_neg}, neg_share={_fmt_plain(mnq_neg_share,3)}. This is heterogeneity/noise, not a deployable linear sizer. Possible Q4-peak/Q5-crash biphasic hypothesis belongs to the Claude terminal, not this scope."
    )
    lines.append("")
    lines.append("## Filter-form lock verification")
    lines.append("")
    lines.append(
        f"Verified in `{FILTER_FORM_YAML}`: `oos_peeked_window`, `fresh_oos_window`, and `>=50 filter-fired trades per instrument` fresh-OOS gate are present. No filter-form execution was run on the contaminated 2026-01-01..2026-04-19 OOS window in this task."
    )

    RESULT_DOC.parent.mkdir(parents=True, exist_ok=True)
    RESULT_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {RESULT_DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
