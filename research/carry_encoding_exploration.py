"""Carry Encoding Exploration — continuous carry features on the validated shelf.

Pre-registered: docs/audit/hypotheses/2026-04-16-carry-encoding-exploration.yaml
Design: docs/plans/2026-04-16-carry-encoding-exploration-design.md
Predecessor: W2e (binary gate DEAD, corr=+0.016 orthogonal to garch).

Tests 3 continuous carry encodings (E1, E2, E3) via quintile monotonicity
on the validated shelf. K=18 (3 encodings × 2 roles × 3 session groups).
BH-FDR at encoding level K=3.

Output: docs/audit/results/2026-04-16-carry-encoding-exploration.md
"""

from __future__ import annotations

import io
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import duckdb
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from pipeline.dst import orb_utc_window
from pipeline.paths import GOLD_DB_PATH
from research import garch_broad_exact_role_exhaustion as broad
from research import garch_partner_state_provenance_audit as prov

OUTPUT_MD = Path("docs/audit/results/2026-04-16-carry-encoding-exploration.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

GARCH_HIGH = 70.0
IS_END = "2026-01-01"
N_QUINTILES = 5
LAMBDA_DEFAULT = math.log(2) / 2  # half-life 2 hours
LAMBDA_SENS = [math.log(2) / 1, math.log(2) / 4]  # 1h and 4h for sensitivity

SESSION_GROUPS = {
    "late_day": ["COMEX_SETTLE"],
    "mid_day": ["EUROPE_FLOW", "SINGAPORE_OPEN"],
    "early_day": ["TOKYO_OPEN"],
}


@dataclass
class EncodingResult:
    encoding: str
    session_group: str
    n_total: int = 0
    n_with_feature: int = 0
    coverage: float = 0.0
    quintile_exp: list[float] = field(default_factory=list)
    quintile_wr: list[float] = field(default_factory=list)
    quintile_n: list[int] = field(default_factory=list)
    spearman_rho: float = float("nan")
    spearman_p: float = float("nan")
    wr_spread: float = float("nan")
    expr_spread: float = float("nan")
    is_arithmetic_only: bool = False
    garch_high_rho: float = float("nan")
    garch_low_rho: float = float("nan")
    oos_sign_matches: bool = False
    oos_rho: float = float("nan")
    oos_n: int = 0
    verdict: str = ""


def load_target_population(con: duckdb.DuckDBPyConnection, rows: pd.DataFrame) -> pd.DataFrame:
    """Pull all target trades from validated shelf with garch + target break_dir."""
    dfs: list[pd.DataFrame] = []
    for _, row in rows.iterrows():
        filter_sql, join_sql = broad.exact_filter_sql(row["filter_type"], row["orb_label"], row["instrument"])
        if filter_sql is None:
            continue
        ts = row["orb_label"]
        q = f"""
        SELECT o.trading_day, o.symbol, o.pnl_r, o.outcome,
               d.garch_forecast_vol_pct AS gp,
               d.orb_{ts}_break_dir AS target_dir,
               '{ts}' AS target_session,
               {row["orb_minutes"]} AS orb_minutes
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.trading_day = d.trading_day
         AND o.symbol = d.symbol
         AND o.orb_minutes = d.orb_minutes
        {join_sql}
        WHERE o.symbol = '{row["instrument"]}'
          AND o.orb_label = '{ts}'
          AND o.orb_minutes = {row["orb_minutes"]}
          AND o.entry_model = '{row["entry_model"]}'
          AND o.rr_target = {row["rr_target"]}
          AND o.pnl_r IS NOT NULL
          AND d.garch_forecast_vol_pct IS NOT NULL
          AND d.orb_{ts}_break_dir IS NOT NULL
          AND {filter_sql}
        ORDER BY o.trading_day
        """
        df = con.execute(q).df()
        if len(df) == 0:
            continue
        df["strategy_id"] = row["strategy_id"]
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    out["trading_day"] = pd.to_datetime(out["trading_day"]).dt.date
    out["pnl_r"] = pd.to_numeric(out["pnl_r"], errors="coerce")
    out["gp"] = pd.to_numeric(out["gp"], errors="coerce")
    return out


def attach_target_start(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df
    cache: dict[tuple, object] = {}
    starts = []
    for _, r in df.iterrows():
        key = (r["trading_day"], r["target_session"], int(r["orb_minutes"]))
        if key not in cache:
            try:
                cache[key] = pd.Timestamp(orb_utc_window(*key)[0])
            except Exception:
                cache[key] = pd.NaT
        starts.append(cache[key])
    out = df.copy()
    out["target_start_ts"] = pd.to_datetime(starts, utc=True)
    return out.dropna(subset=["target_start_ts"])


def load_all_priors(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """All resolved prior trades (E2/CB1/RR1.0/O5) with pnl_r, exit_ts, break_dir."""
    df = con.execute("""
    SELECT o.trading_day, o.symbol, o.orb_label, o.pnl_r, o.outcome, o.exit_ts,
           d.orb_CME_REOPEN_break_dir, d.orb_TOKYO_OPEN_break_dir,
           d.orb_SINGAPORE_OPEN_break_dir, d.orb_LONDON_METALS_break_dir,
           d.orb_EUROPE_FLOW_break_dir, d.orb_US_DATA_830_break_dir,
           d.orb_NYSE_OPEN_break_dir, d.orb_US_DATA_1000_break_dir,
           d.orb_COMEX_SETTLE_break_dir, d.orb_CME_PRECLOSE_break_dir,
           d.orb_NYSE_CLOSE_break_dir, d.orb_BRISBANE_1025_break_dir
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day AND o.symbol = d.symbol AND o.orb_minutes = d.orb_minutes
    WHERE o.entry_model = 'E2' AND o.confirm_bars = 1 AND o.rr_target = 1.0
      AND o.orb_minutes = 5 AND o.outcome IS NOT NULL AND o.exit_ts IS NOT NULL
      AND o.pnl_r IS NOT NULL
    ORDER BY o.trading_day, o.symbol, o.exit_ts
    """).df()
    df["trading_day"] = pd.to_datetime(df["trading_day"]).dt.date
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True)
    df["pnl_r"] = pd.to_numeric(df["pnl_r"], errors="coerce")
    return df


def get_prior_break_dir(prior_row: pd.Series) -> str | None:
    col = f"orb_{prior_row['orb_label']}_break_dir"
    val = prior_row.get(col)
    if pd.isna(val) or val is None:
        return None
    return str(val)


def compute_encodings(target: pd.DataFrame, priors: pd.DataFrame, lam: float = LAMBDA_DEFAULT) -> pd.DataFrame:
    """For each target row, compute E1, E2, E3 from resolved priors."""
    results = []
    grouped_priors = priors.groupby(["trading_day", "symbol"])

    for idx, row in target.iterrows():
        day = row["trading_day"]
        sym = row["symbol"]
        tstart = row["target_start_ts"]
        tdir = row["target_dir"]
        key = (day, sym)

        e1 = float("nan")
        e2 = float("nan")
        e3 = float("nan")

        if key not in grouped_priors.groups:
            results.append({"_idx": idx, "E1": e1, "E2": e2, "E3": e3})
            continue

        day_priors = grouped_priors.get_group(key).copy()
        resolved = day_priors[
            (day_priors["orb_label"] != row["target_session"]) & (day_priors["exit_ts"] < tstart)
        ].copy()

        if len(resolved) == 0:
            results.append({"_idx": idx, "E1": e1, "E2": e2, "E3": e3})
            continue

        resolved = resolved.sort_values("exit_ts")
        most_recent = resolved.iloc[-1]

        # E1: most recent prior pnl_r
        e1 = float(most_recent["pnl_r"])

        # E2: recency-weighted carry score
        hours_gaps = (tstart - resolved["exit_ts"]).dt.total_seconds() / 3600.0
        weights = np.exp(-lam * hours_gaps.to_numpy())
        wsum = weights.sum()
        if wsum > 0:
            e2 = float((resolved["pnl_r"].to_numpy() * weights).sum() / wsum)

        # E3: direction-aware carry intensity
        prior_dir = get_prior_break_dir(most_recent)
        if prior_dir is not None and tdir is not None:
            sign = 1.0 if prior_dir == tdir else -1.0
            e3 = e1 * sign

        results.append({"_idx": idx, "E1": e1, "E2": e2, "E3": e3})

    enc_df = pd.DataFrame(results).set_index("_idx")
    return enc_df


def quintile_analysis(pnl: np.ndarray, feature: np.ndarray, outcome: np.ndarray) -> dict[str, object]:
    """Equal-count quintile split. Returns per-quintile ExpR, WR, N, and monotonicity."""
    valid = np.isfinite(feature) & np.isfinite(pnl)
    if valid.sum() < N_QUINTILES * 10:
        return {"valid": False, "n": int(valid.sum())}

    f = feature[valid]
    p = pnl[valid]
    o = outcome[valid]
    n = len(f)

    try:
        qbins = pd.qcut(f, N_QUINTILES, labels=False, duplicates="drop")
    except ValueError:
        return {"valid": False, "n": n}

    actual_q = int(qbins.max()) + 1
    if actual_q < 3:
        return {"valid": False, "n": n}

    q_exp = []
    q_wr = []
    q_n = []
    for qi in range(actual_q):
        mask = qbins == qi
        qi_pnl = p[mask]
        qi_out = o[mask]
        q_n.append(int(mask.sum()))
        q_exp.append(float(qi_pnl.mean()) if mask.sum() > 0 else float("nan"))
        q_wr.append(float((qi_out == "win").mean()) if mask.sum() > 0 else float("nan"))

    rho, rho_p = scipy_stats.spearmanr(range(actual_q), q_exp)
    wr_spread = max(q_wr) - min(q_wr) if len(q_wr) >= 2 else 0.0
    expr_spread = max(q_exp) - min(q_exp) if len(q_exp) >= 2 else 0.0
    arithmetic_only = bool(wr_spread < 0.05 and expr_spread > 0.05)

    return {
        "valid": True,
        "n": n,
        "actual_quintiles": actual_q,
        "q_exp": q_exp,
        "q_wr": q_wr,
        "q_n": q_n,
        "rho": float(rho),
        "rho_p": float(rho_p),
        "wr_spread": float(wr_spread),
        "expr_spread": float(expr_spread),
        "arithmetic_only": arithmetic_only,
    }


def run_encoding(target: pd.DataFrame, encoding_col: str, session_group: str, sessions: list[str]) -> EncodingResult:
    sub = target[target["target_session"].isin(sessions)].copy()
    res = EncodingResult(encoding=encoding_col, session_group=session_group)
    res.n_total = len(sub)
    if res.n_total == 0:
        res.verdict = "no_data"
        return res

    feat = sub[encoding_col].to_numpy(dtype=float)
    res.n_with_feature = int(np.isfinite(feat).sum())
    res.coverage = res.n_with_feature / res.n_total if res.n_total else 0.0

    if res.coverage < 0.30:
        res.verdict = "encoding_sparse"
        return res

    pnl = sub["pnl_r"].to_numpy(dtype=float)
    outcome = sub["outcome"].to_numpy()
    gp = sub["gp"].to_numpy(dtype=float)

    # IS only for quintile analysis
    is_mask = sub["trading_day"].apply(lambda d: d < pd.Timestamp(IS_END).date()).to_numpy()
    oos_mask = ~is_mask

    # Full IS quintile analysis
    qa = quintile_analysis(pnl[is_mask], feat[is_mask], outcome[is_mask])
    if not qa.get("valid"):
        res.verdict = "encoding_sparse"
        return res

    res.quintile_exp = qa["q_exp"]
    res.quintile_wr = qa["q_wr"]
    res.quintile_n = qa["q_n"]
    res.spearman_rho = qa["rho"]
    res.spearman_p = qa["rho_p"]
    res.wr_spread = qa["wr_spread"]
    res.expr_spread = qa["expr_spread"]
    res.is_arithmetic_only = qa["arithmetic_only"]

    # Garch interaction
    gh = is_mask & (gp >= GARCH_HIGH)
    gl = is_mask & (gp < GARCH_HIGH)
    qa_gh = quintile_analysis(pnl[gh], feat[gh], outcome[gh])
    qa_gl = quintile_analysis(pnl[gl], feat[gl], outcome[gl])
    if qa_gh.get("valid"):
        res.garch_high_rho = qa_gh["rho"]
    if qa_gl.get("valid"):
        res.garch_low_rho = qa_gl["rho"]

    # OOS direction check (descriptive only)
    if oos_mask.sum() >= 20:
        qa_oos = quintile_analysis(pnl[oos_mask], feat[oos_mask], outcome[oos_mask])
        if qa_oos.get("valid"):
            res.oos_rho = qa_oos["rho"]
            res.oos_n = qa_oos["n"]
            is_sign = 1 if res.spearman_rho > 0 else -1
            oos_sign = 1 if res.oos_rho > 0 else -1
            res.oos_sign_matches = bool(is_sign == oos_sign)

    # Verdict per pre-registered gates
    mono_pass = abs(res.spearman_rho) >= 0.80
    wr_or_expr = res.wr_spread >= 0.05 or res.expr_spread >= 0.10
    not_arith = not res.is_arithmetic_only

    if mono_pass and wr_or_expr and not_arith:
        res.verdict = "encoding_monotonic"
    elif mono_pass and res.is_arithmetic_only:
        res.verdict = "encoding_arithmetic_only"
    elif not mono_pass:
        res.verdict = "encoding_flat"
    else:
        res.verdict = "encoding_flat"

    return res


def build() -> tuple[list[EncodingResult], list[EncodingResult], dict[str, object]]:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    rows = prov.load_rows(con)
    print(f"Validated shelf rows: {len(rows)}")

    target = load_target_population(con, rows)
    print(f"Target population rows: {len(target)}")
    target = attach_target_start(target)
    print(f"After target_start_ts: {len(target)}")

    priors = load_all_priors(con)
    print(f"Prior trade rows: {len(priors)}")

    # Compute encodings at default lambda
    enc = compute_encodings(target, priors, LAMBDA_DEFAULT)
    target = target.join(enc)
    print(f"E1 coverage: {target['E1'].notna().mean():.3f}")
    print(f"E2 coverage: {target['E2'].notna().mean():.3f}")
    print(f"E3 coverage: {target['E3'].notna().mean():.3f}")

    # Main results per (encoding, session_group)
    main_results: list[EncodingResult] = []
    for sg_name, sg_sessions in SESSION_GROUPS.items():
        for enc_col in ["E1", "E2", "E3"]:
            res = run_encoding(target, enc_col, sg_name, sg_sessions)
            main_results.append(res)
            print(f"  {enc_col} x {sg_name}: {res.verdict} (rho={res.spearman_rho:+.3f}, coverage={res.coverage:.2f})")

    # E2 sensitivity at alternate lambdas
    sensitivity_results: list[EncodingResult] = []
    for lam in LAMBDA_SENS:
        hl = math.log(2) / lam
        enc_sens = compute_encodings(target, priors, lam)
        target_sens = target.copy()
        target_sens["E2_sens"] = enc_sens["E2"]
        for sg_name, sg_sessions in SESSION_GROUPS.items():
            res = run_encoding(target_sens, "E2_sens", sg_name, sg_sessions)
            res.encoding = f"E2_hl{hl:.0f}h"
            sensitivity_results.append(res)
            print(f"  E2(hl={hl:.0f}h) x {sg_name}: {res.verdict} (rho={res.spearman_rho:+.3f})")

    con.close()

    meta = {
        "validated_rows": int(len(rows)),
        "target_rows": int(len(target)),
        "prior_rows": int(len(priors)),
        "main_cells": int(len(main_results)),
        "sensitivity_cells": int(len(sensitivity_results)),
    }
    return main_results, sensitivity_results, meta


def emit(
    main: list[EncodingResult],
    sens: list[EncodingResult],
    meta: dict[str, object],
) -> None:
    lines: list[str] = [
        "# Carry Encoding Exploration Results",
        "",
        "**Date:** 2026-04-16",
        f"**Pre-registered:** `docs/audit/hypotheses/2026-04-16-carry-encoding-exploration.yaml`",
        "**Boundary:** validated shelf only, IS < 2026-01-01 for quintile gates, OOS descriptive only",
        "",
        "## Scope",
        "",
        f"- Validated shelf rows: **{meta['validated_rows']}**",
        f"- Target population (after filter + start_ts): **{meta['target_rows']}**",
        f"- Prior trade rows: **{meta['prior_rows']}**",
        f"- Main cells (3 encodings × 3 session groups): **{meta['main_cells']}**",
        f"- E2 sensitivity cells: **{meta['sensitivity_cells']}**",
        "",
        "## Main results",
        "",
        "| Encoding | Session group | N total | N with feature | Coverage | Spearman rho | rho p | WR spread | ExpR spread | ARITH_ONLY | Garch-high rho | Garch-low rho | OOS rho | OOS match | Verdict |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---|",
    ]
    for r in main:
        gh_s = f"{r.garch_high_rho:+.3f}" if np.isfinite(r.garch_high_rho) else "—"
        gl_s = f"{r.garch_low_rho:+.3f}" if np.isfinite(r.garch_low_rho) else "—"
        oos_s = f"{r.oos_rho:+.3f}" if np.isfinite(r.oos_rho) else "—"
        lines.append(
            f"| {r.encoding} | {r.session_group} | {r.n_total} | {r.n_with_feature} | "
            f"{r.coverage:.2f} | {r.spearman_rho:+.3f} | {r.spearman_p:.4f} | "
            f"{r.wr_spread:.3f} | {r.expr_spread:.3f} | "
            f"{'YES' if r.is_arithmetic_only else 'no'} | "
            f"{gh_s} | {gl_s} | {oos_s} | "
            f"{'yes' if r.oos_sign_matches else 'no'} | "
            f"**{r.verdict}** |"
        )

    # Quintile detail for non-sparse results
    lines.extend(["", "## Quintile detail (IS only)", ""])
    for r in main:
        if not r.quintile_exp:
            continue
        lines.append(f"### {r.encoding} × {r.session_group}")
        lines.append("")
        lines.append("| Quintile | N | ExpR | WR |")
        lines.append("|---:|---:|---:|---:|")
        for qi, (n, e, w) in enumerate(zip(r.quintile_n, r.quintile_exp, r.quintile_wr)):
            lines.append(f"| Q{qi + 1} | {n} | {e:+.3f} | {w:.3f} |")
        lines.append("")

    # E2 sensitivity
    lines.extend(["## E2 sensitivity (half-life variants)", ""])
    lines.append("| Variant | Session group | Coverage | Spearman rho | WR spread | ExpR spread | Verdict |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for r in sens:
        lines.append(
            f"| {r.encoding} | {r.session_group} | {r.coverage:.2f} | "
            f"{r.spearman_rho:+.3f} | {r.wr_spread:.3f} | {r.expr_spread:.3f} | "
            f"**{r.verdict}** |"
        )

    # BH-FDR at encoding level
    lines.extend(["", "## BH-FDR at encoding level (K=3)", ""])
    enc_best: dict[str, float] = {}
    for r in main:
        if r.verdict in ("no_data", "encoding_sparse"):
            continue
        key = r.encoding
        if key not in enc_best or abs(r.spearman_rho) > abs(enc_best.get(key, 0)):
            enc_best[key] = r.spearman_p
    if enc_best:
        sorted_p = sorted(enc_best.items(), key=lambda x: x[1])
        lines.append("| Rank | Encoding | Best p | BH threshold (q=0.05) | Pass |")
        lines.append("|---:|---|---:|---:|---|")
        for rank, (enc_name, p) in enumerate(sorted_p, 1):
            bh_thresh = 0.05 * rank / len(sorted_p)
            passes = "YES" if p <= bh_thresh else "no"
            lines.append(f"| {rank} | {enc_name} | {p:.4f} | {bh_thresh:.4f} | {passes} |")
    lines.append("")

    # Guardrails
    lines.extend(
        [
            "## Guardrails",
            "",
            "- **Chronology:** all prior trades must have `exit_ts < target_start_ts` (dynamic per `pipeline.dst.orb_utc_window`).",
            "- **Canonical sources:** `orb_outcomes` + `daily_features` only. `validated_setups` used to enumerate targets, each verified by raw query.",
            "- **IS/OOS split:** quintile gates use IS (pre-2026-01-01) only. OOS (2026+) is descriptive and not used for promotion.",
            "- **Binary gates banned:** no threshold on any encoding. Quintile analysis only.",
            "- **K budget:** 3 encodings × 3 session groups = 9 main cells. BH-FDR applied at encoding level K=3.",
            "- **No deployment claims.**",
            "",
        ]
    )

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    main_results, sens_results, meta = build()
    emit(main_results, sens_results, meta)
    print(f"\nWrote {OUTPUT_MD}")
    print(f"meta: {meta}")

    # Summary verdict
    passed = [r for r in main_results if r.verdict == "encoding_monotonic"]
    arith = [r for r in main_results if r.verdict == "encoding_arithmetic_only"]
    flat = [r for r in main_results if r.verdict == "encoding_flat"]
    sparse = [r for r in main_results if r.verdict in ("encoding_sparse", "no_data")]
    print(f"\nSummary: {len(passed)} monotonic, {len(arith)} arithmetic_only, {len(flat)} flat, {len(sparse)} sparse")
    if passed:
        print("GATE: at least one encoding passes → carry-as-soft-feature worth investigating")
    elif arith and not passed:
        print("GATE: all passing are ARITHMETIC_ONLY → carry is a cost screen, not a signal")
    else:
        print("GATE: all encodings fail → PARK carry family entirely")


if __name__ == "__main__":
    main()
