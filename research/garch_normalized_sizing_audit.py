"""Normalized sizing audit for garch as an R3 regime variable.

Purpose:
  Evaluate whether garch improves portfolio utility as a normalized size
  modifier rather than as a binary gate. The audit is intentionally simple:

    - 5 pre-committed weight maps
    - 2 locked scopes:
        1. broad exact-filter rows
        2. validated-only rows
    - normalization fixed on IS only, then applied unchanged to OOS

The goal is portfolio utility:
  - total R / total $
  - daily Sharpe
  - max drawdown
  - worst day / worst 5-day run
  - daily risk-dollar concentration

Pre-registration:
  docs/audit/hypotheses/2026-04-16-garch-normalized-sizing-audit.yaml

Output:
  docs/audit/results/2026-04-16-garch-normalized-sizing-audit.md
"""

from __future__ import annotations

import io
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import duckdb
import numpy as np
import pandas as pd

from pipeline.paths import GOLD_DB_PATH
from research import garch_broad_exact_role_exhaustion as broad
from research import garch_regime_family_audit as fam

OUTPUT_MD = Path("docs/audit/results/2026-04-16-garch-normalized-sizing-audit.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

MAPS = [
    "LOW_CUT_ONLY",
    "HIGH_BOOST_ONLY",
    "SESSION_CLIPPED",
    "SESSION_LINEAR",
    "GLOBAL_LINEAR",
]


def session_profiles(cells: list[fam.CellRecord]) -> dict[str, dict[str, bool]]:
    directional = fam.family_directional(cells)
    monotone = fam.family_monotonicity(cells)

    profiles: dict[str, dict[str, bool]] = {}
    for sess in sorted({c.orb_label for c in cells}):
        d = directional[directional["session"] == sess]
        m = monotone[monotone["session"] == sess]
        profiles[sess] = {
            "high_dir": bool(((d["side"] == "high") & (d["bh_dir"] == True) & (d["mean_sr_lift"] > 0)).any()),
            "low_dir": bool(((d["side"] == "low") & (d["bh_dir"] == True) & (d["mean_sr_lift"] < 0)).any()),
            "high_mono": bool(((m["side"] == "high") & (m["bh_tail"] == True) & (m["mean_tail_bias"] > 0)).any()),
            "low_mono": bool(((m["side"] == "low") & (m["bh_tail"] == True) & (m["mean_tail_bias"] < 0)).any()),
        }
    return profiles


def load_scope_rows(con: duckdb.DuckDBPyConnection, scope: str) -> pd.DataFrame:
    rows = broad.load_rows(con)
    rows = rows[rows["filter_type"].map(broad.in_scope)].copy()
    if scope == "validated":
        rows = rows[rows["src"] == "validated"].copy()
    elif scope == "broad":
        pass
    else:
        raise ValueError(f"Unknown scope: {scope}")
    return rows.reset_index(drop=True)


def load_scope_trades(con: duckdb.DuckDBPyConnection, rows: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, row in rows.iterrows():
        filter_sql, join_sql = broad.exact_filter_sql(row["filter_type"], row["orb_label"], row["instrument"])
        if filter_sql is None:
            continue
        for direction in ["long", "short"]:
            q = f"""
            SELECT
              o.trading_day,
              o.symbol AS instrument,
              o.orb_label,
              o.rr_target,
              o.entry_model,
              o.pnl_r,
              o.pnl_dollars,
              o.risk_dollars,
              d.garch_forecast_vol_pct AS gp
            FROM orb_outcomes o
            JOIN daily_features d
              ON o.trading_day = d.trading_day
             AND o.symbol = d.symbol
             AND o.orb_minutes = d.orb_minutes
            {join_sql}
            WHERE o.symbol = '{row["instrument"]}'
              AND o.orb_minutes = {row["orb_minutes"]}
              AND o.orb_label = '{row["orb_label"]}'
              AND o.entry_model = '{row["entry_model"]}'
              AND o.rr_target = {row["rr_target"]}
              AND o.pnl_r IS NOT NULL
              AND o.pnl_dollars IS NOT NULL
              AND o.risk_dollars IS NOT NULL
              AND d.garch_forecast_vol_pct IS NOT NULL
              AND d.orb_{row["orb_label"]}_break_dir = '{direction}'
              AND {filter_sql}
            ORDER BY o.trading_day
            """
            df = con.execute(q).df()
            if len(df) == 0:
                continue
            df["trading_day"] = pd.to_datetime(df["trading_day"])
            df["strategy_id"] = row["strategy_id"]
            df["src"] = row["src"]
            df["direction"] = direction
            df["filter_type"] = row["filter_type"]
            df["orb_minutes"] = row["orb_minutes"]
            df["is_oos"] = df["trading_day"] >= pd.Timestamp(broad.IS_END)
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["gp"] = out["gp"].astype(float)
    out["pnl_r"] = out["pnl_r"].astype(float)
    out["pnl_dollars"] = out["pnl_dollars"].astype(float)
    out["risk_dollars"] = out["risk_dollars"].astype(float)
    return out


def raw_weight(map_name: str, gp: float, session: str, profiles: dict[str, dict[str, bool]]) -> float:
    p = profiles.get(session, {"high_dir": False, "low_dir": False, "high_mono": False, "low_mono": False})

    if map_name == "LOW_CUT_ONLY":
        return 0.5 if p["low_dir"] and gp <= 30 else 1.0
    if map_name == "HIGH_BOOST_ONLY":
        return 1.5 if p["high_dir"] and gp >= 70 else 1.0
    if map_name == "SESSION_CLIPPED":
        if p["high_dir"] and gp >= 70:
            return 1.5
        if p["low_dir"] and gp <= 30:
            return 0.5
        return 1.0
    if map_name == "SESSION_LINEAR":
        if p["high_dir"] or p["low_dir"]:
            return max(0.5, min(1.5, 0.5 + gp / 100.0))
        return 1.0
    if map_name == "GLOBAL_LINEAR":
        return max(0.5, min(1.5, 0.5 + gp / 100.0))
    raise ValueError(f"Unknown map: {map_name}")


def max_drawdown(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    equity = series.cumsum()
    peak = equity.cummax()
    dd = equity - peak
    return float(dd.min())


def ann_sharpe(daily: pd.Series) -> float:
    daily = daily.astype(float)
    sd = daily.std(ddof=1)
    if len(daily) < 2 or sd <= 0:
        return 0.0
    return float((daily.mean() / sd) * math.sqrt(252.0))


def summarize(df: pd.DataFrame, weight_col: str) -> dict[str, float]:
    if len(df) == 0:
        return {
            "n_trades": 0,
            "exp_r": 0.0,
            "total_r": 0.0,
            "total_dollars": 0.0,
            "sharpe_ann_r": 0.0,
            "sharpe_ann_dollars": 0.0,
            "max_dd_r": 0.0,
            "max_dd_dollars": 0.0,
            "worst_day_r": 0.0,
            "worst_day_dollars": 0.0,
            "worst_5day_dollars": 0.0,
            "max_daily_risk_dollars": 0.0,
        }
    work = df.copy()
    work["weighted_r"] = work["pnl_r"] * work[weight_col]
    work["weighted_dollars"] = work["pnl_dollars"] * work[weight_col]
    work["weighted_risk_dollars"] = work["risk_dollars"] * work[weight_col]

    daily = (
        work.groupby("trading_day", as_index=True)[["weighted_r", "weighted_dollars", "weighted_risk_dollars"]]
        .sum()
        .sort_index()
    )
    roll5 = daily["weighted_dollars"].rolling(5).sum()

    return {
        "n_trades": int(len(work)),
        "exp_r": float(work["weighted_r"].mean()),
        "total_r": float(work["weighted_r"].sum()),
        "total_dollars": float(work["weighted_dollars"].sum()),
        "sharpe_ann_r": ann_sharpe(daily["weighted_r"]),
        "sharpe_ann_dollars": ann_sharpe(daily["weighted_dollars"]),
        "max_dd_r": max_drawdown(daily["weighted_r"]),
        "max_dd_dollars": max_drawdown(daily["weighted_dollars"]),
        "worst_day_r": float(daily["weighted_r"].min()),
        "worst_day_dollars": float(daily["weighted_dollars"].min()),
        "worst_5day_dollars": float(roll5.min()) if roll5.notna().any() else 0.0,
        "max_daily_risk_dollars": float(daily["weighted_risk_dollars"].max()),
    }


def evaluate_scope(df: pd.DataFrame, scope: str, profiles: dict[str, dict[str, bool]]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    out_rows = []
    contribution_tables: dict[str, pd.DataFrame] = {}

    base = df.copy()
    base["w_base"] = 1.0
    base_is = summarize(base[~base["is_oos"]], "w_base")
    base_oos = summarize(base[base["is_oos"]], "w_base")
    base_full = summarize(base, "w_base")

    for map_name in MAPS:
        work = df.copy()
        work["raw_weight"] = [
            raw_weight(map_name, gp, sess, profiles)
            for gp, sess in zip(work["gp"].astype(float), work["orb_label"].astype(str))
        ]
        is_mean = float(work.loc[~work["is_oos"], "raw_weight"].mean())
        norm = 1.0 / is_mean if is_mean > 0 else 1.0
        work["weight"] = work["raw_weight"] * norm

        is_metrics = summarize(work[~work["is_oos"]], "weight")
        oos_metrics = summarize(work[work["is_oos"]], "weight")
        full_metrics = summarize(work, "weight")

        expr_delta_is = is_metrics["exp_r"] - base_is["exp_r"]
        expr_delta_oos = oos_metrics["exp_r"] - base_oos["exp_r"] if oos_metrics["n_trades"] else 0.0
        retention = (
            expr_delta_oos / expr_delta_is
            if abs(expr_delta_is) > 1e-9
            else float("nan")
        )

        out_rows.append(
            {
                "scope": scope,
                "map": map_name,
                "norm_factor": norm,
                "mean_weight_full": float(work["weight"].mean()),
                "min_weight": float(work["weight"].min()),
                "max_weight": float(work["weight"].max()),
                "full_total_r": full_metrics["total_r"],
                "full_delta_r": full_metrics["total_r"] - base_full["total_r"],
                "full_total_dollars": full_metrics["total_dollars"],
                "full_delta_dollars": full_metrics["total_dollars"] - base_full["total_dollars"],
                "full_sharpe_r": full_metrics["sharpe_ann_r"],
                "full_sharpe_r_delta": full_metrics["sharpe_ann_r"] - base_full["sharpe_ann_r"],
                "full_max_dd_r": full_metrics["max_dd_r"],
                "full_max_dd_r_delta": full_metrics["max_dd_r"] - base_full["max_dd_r"],
                "worst_day_dollars": full_metrics["worst_day_dollars"],
                "worst_day_dollars_delta": full_metrics["worst_day_dollars"] - base_full["worst_day_dollars"],
                "worst_5day_dollars": full_metrics["worst_5day_dollars"],
                "worst_5day_dollars_delta": full_metrics["worst_5day_dollars"] - base_full["worst_5day_dollars"],
                "max_daily_risk_dollars": full_metrics["max_daily_risk_dollars"],
                "max_daily_risk_dollars_delta": full_metrics["max_daily_risk_dollars"] - base_full["max_daily_risk_dollars"],
                "is_exp_r": is_metrics["exp_r"],
                "is_exp_r_delta": expr_delta_is,
                "oos_exp_r": oos_metrics["exp_r"],
                "oos_exp_r_delta": expr_delta_oos,
                "oos_retention": retention,
            }
        )

        contrib = work.copy()
        contrib["base_dollars"] = contrib["pnl_dollars"]
        contrib["alt_dollars"] = contrib["pnl_dollars"] * contrib["weight"]
        contrib["delta_dollars"] = contrib["alt_dollars"] - contrib["base_dollars"]
        contribution_tables[map_name] = (
            contrib.groupby(["instrument", "orb_label"], as_index=False)[["base_dollars", "alt_dollars", "delta_dollars"]]
            .sum()
            .sort_values("delta_dollars", ascending=False)
            .reset_index(drop=True)
        )

    return pd.DataFrame(out_rows), contribution_tables


def emit(
    directional_profiles: dict[str, dict[str, bool]],
    scope_results: pd.DataFrame,
    contrib_tables: dict[str, dict[str, pd.DataFrame]],
) -> None:
    lines = [
        "# Garch Normalized Sizing Audit",
        "",
        "**Date:** 2026-04-16",
        "**Pre-registration:** `docs/audit/hypotheses/2026-04-16-garch-normalized-sizing-audit.yaml`",
        "**Purpose:** test garch as an R3 size modifier with normalized weights rather than as a binary filter.",
        "",
        "**Grounding:**",
        "- `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md`",
        "- `docs/institutional/regime-and-rr-handling-framework.md`",
        "- `docs/institutional/mechanism_priors.md`",
        "- `docs/audit/results/2026-04-16-garch-regime-family-audit.md`",
        "- `docs/audit/results/2026-04-16-garch-g0-preflight.md`",
        "",
        "## Session profiles used for sizing",
        "",
        "| Session | High directional support | Low directional support | High monotonicity support | Low monotonicity support |",
        "|---|---|---|---|---|",
    ]
    for sess, p in sorted(directional_profiles.items()):
        lines.append(
            f"| {sess} | {'Y' if p['high_dir'] else '.'} | {'Y' if p['low_dir'] else '.'} | "
            f"{'Y' if p['high_mono'] else '.'} | {'Y' if p['low_mono'] else '.'} |"
        )

    lines += [
        "",
        "## Map definitions",
        "",
        "- `LOW_CUT_ONLY`: sessions with low-directional support get `0.5x` at `gp<=30`; otherwise `1.0x`.",
        "- `HIGH_BOOST_ONLY`: sessions with high-directional support get `1.5x` at `gp>=70`; otherwise `1.0x`.",
        "- `SESSION_CLIPPED`: combines the two clipped rules above.",
        "- `SESSION_LINEAR`: supported sessions get `clip(0.5 + gp/100, 0.5, 1.5)`; unsupported sessions stay `1.0x`.",
        "- `GLOBAL_LINEAR`: every trade gets `clip(0.5 + gp/100, 0.5, 1.5)`.",
        "",
        "All maps are normalized on IS only so mean raw weight becomes `1.0x`. The same normalization factor is then applied unchanged to OOS.",
        "",
    ]

    for scope in ["broad", "validated"]:
        sub = scope_results[scope_results["scope"] == scope].sort_values(
            ["full_delta_dollars", "full_sharpe_r_delta"], ascending=False
        )
        lines += [
            f"## {scope.title()} scope results",
            "",
            "| Map | Norm | Weight range | Full Δ$ | Full ΔR | Sharpe Δ | MaxDD ΔR | Worst day Δ$ | Worst 5d Δ$ | Max daily risk Δ$ | IS ExpR Δ | OOS ExpR Δ | OOS retention |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for _, r in sub.iterrows():
            retain = "n/a" if pd.isna(r["oos_retention"]) else f"{r['oos_retention']:+.2f}"
            lines.append(
                f"| {r['map']} | {r['norm_factor']:.3f} | {r['min_weight']:.2f}-{r['max_weight']:.2f} | "
                f"{r['full_delta_dollars']:+.1f} | {r['full_delta_r']:+.1f} | {r['full_sharpe_r_delta']:+.3f} | "
                f"{r['full_max_dd_r_delta']:+.1f} | {r['worst_day_dollars_delta']:+.1f} | {r['worst_5day_dollars_delta']:+.1f} | "
                f"{r['max_daily_risk_dollars_delta']:+.1f} | {r['is_exp_r_delta']:+.4f} | {r['oos_exp_r_delta']:+.4f} | {retain} |"
            )

        best = sub.iloc[0]["map"] if len(sub) else None
        if best is not None:
            lines += ["", f"### {scope.title()} best-map contributions: `{best}`", "", "| Instrument | Session | Base $ | Alt $ | Δ$ |", "|---|---|---|---|---|"]
            for _, r in contrib_tables[scope][best].head(15).iterrows():
                lines.append(
                    f"| {r['instrument']} | {r['orb_label']} | {r['base_dollars']:+.1f} | {r['alt_dollars']:+.1f} | {r['delta_dollars']:+.1f} |"
                )

    lines += [
        "",
        "## Reading the audit",
        "",
        "- `Full Δ$` and `Full ΔR` answer the total take-home question after normalized sizing.",
        "- `Sharpe Δ` and `MaxDD ΔR` answer the risk-adjusted portfolio question. Positive `MaxDD ΔR` means the drawdown became less severe.",
        "- `OOS retention` compares OOS ExpR uplift to IS ExpR uplift. Positive is directionally good; high ratios are better.",
        "- `Worst day Δ$`, `Worst 5d Δ$`, and `Max daily risk Δ$` are directional risk diagnostics. More negative is worse; more positive is safer.",
        "- `Max daily risk Δ$` is a concentration proxy, not an account-breach simulation.",
        "",
        "## Caveats",
        "",
        "- This is still a backtest-side utilization audit, not production proof.",
        "- Fractional weights are a research abstraction for normalized sizing. Live implementation would need contract rounding and account budgeting.",
        "- No map was tuned after results; only the pre-committed five maps were run.",
        "",
    ]

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {OUTPUT_MD}")


def main() -> None:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    cells, _ = fam.build_cells()
    profiles = session_profiles(cells)

    scope_frames = {}
    for scope in ["broad", "validated"]:
        rows = load_scope_rows(con, scope)
        scope_frames[scope] = load_scope_trades(con, rows)
    con.close()

    all_results = []
    all_contribs: dict[str, dict[str, pd.DataFrame]] = {}
    for scope, df in scope_frames.items():
        res, contrib = evaluate_scope(df, scope, profiles)
        all_results.append(res)
        all_contribs[scope] = contrib

    results_df = pd.concat(all_results, ignore_index=True)
    emit(profiles, results_df, all_contribs)


if __name__ == "__main__":
    main()
