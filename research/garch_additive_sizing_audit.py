"""Additive sizing audit for garch vs ATR / overnight proxies.

Purpose:
  Compare garch to related regime proxies inside the same clipped
  session-sizing scaffold so the comparison is operationally fair.

Pre-registration:
  docs/audit/hypotheses/2026-04-16-garch-additive-sizing-audit.yaml

Output:
  docs/audit/results/2026-04-16-garch-additive-sizing-audit.md
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
import pandas as pd

from pipeline.paths import GOLD_DB_PATH
from research import garch_broad_exact_role_exhaustion as broad
from research import garch_regime_family_audit as fam

OUTPUT_MD = Path("docs/audit/results/2026-04-16-garch-additive-sizing-audit.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

MAPS = [
    "GARCH_SESSION_CLIPPED",
    "ATR_SESSION_CLIPPED",
    "OVN_SESSION_CLIPPED",
    "GARCH_ATR_MEAN_CLIPPED",
    "GARCH_OVN_MEAN_CLIPPED",
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
        }
    return profiles


def load_scope_rows(con: duckdb.DuckDBPyConnection, scope: str) -> pd.DataFrame:
    rows = broad.load_rows(con)
    rows = rows[rows["filter_type"].map(broad.in_scope)].copy()
    if scope == "validated":
        rows = rows[rows["src"] == "validated"].copy()
    elif scope != "broad":
        raise ValueError(scope)
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
              o.pnl_r,
              o.pnl_dollars,
              o.risk_dollars,
              d.garch_forecast_vol_pct AS gp,
              d.atr_20_pct AS atr_pct,
              d.overnight_range_pct AS ovn_pct
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
              AND d.atr_20_pct IS NOT NULL
              AND d.overnight_range_pct IS NOT NULL
              AND d.orb_{row["orb_label"]}_break_dir = '{direction}'
              AND {filter_sql}
            ORDER BY o.trading_day
            """
            df = con.execute(q).df()
            if len(df):
                df["trading_day"] = pd.to_datetime(df["trading_day"])
                df["src"] = row["src"]
                df["is_oos"] = df["trading_day"] >= pd.Timestamp(broad.IS_END)
                parts.append(df)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    for col in ["pnl_r", "pnl_dollars", "risk_dollars", "gp", "atr_pct", "ovn_pct"]:
        out[col] = out[col].astype(float)
    return out


def max_drawdown(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    eq = series.cumsum()
    return float((eq - eq.cummax()).min())


def ann_sharpe(daily: pd.Series) -> float:
    sd = daily.std(ddof=1)
    if len(daily) < 2 or sd <= 0:
        return 0.0
    return float((daily.mean() / sd) * math.sqrt(252.0))


def score(map_name: str, row: pd.Series) -> float:
    if map_name == "GARCH_SESSION_CLIPPED":
        return float(row["gp"])
    if map_name == "ATR_SESSION_CLIPPED":
        return float(row["atr_pct"])
    if map_name == "OVN_SESSION_CLIPPED":
        return float(row["ovn_pct"])
    if map_name == "GARCH_ATR_MEAN_CLIPPED":
        return float((row["gp"] + row["atr_pct"]) / 2.0)
    if map_name == "GARCH_OVN_MEAN_CLIPPED":
        return float((row["gp"] + row["ovn_pct"]) / 2.0)
    raise ValueError(map_name)


def raw_weight(map_name: str, row: pd.Series, profiles: dict[str, dict[str, bool]]) -> float:
    p = profiles.get(str(row["orb_label"]), {"high_dir": False, "low_dir": False, "high_mono": False})
    s = score(map_name, row)
    if p["high_dir"] and s >= 70:
        return 1.5
    if p["low_dir"] and s <= 30:
        return 0.5
    return 1.0


def summarize(df: pd.DataFrame, weight_col: str) -> dict[str, float]:
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
        "exp_r": float(work["weighted_r"].mean()) if len(work) else 0.0,
        "total_r": float(work["weighted_r"].sum()) if len(work) else 0.0,
        "total_dollars": float(work["weighted_dollars"].sum()) if len(work) else 0.0,
        "sharpe_ann_r": ann_sharpe(daily["weighted_r"]) if len(daily) else 0.0,
        "max_dd_r": max_drawdown(daily["weighted_r"]) if len(daily) else 0.0,
        "worst_day_dollars": float(daily["weighted_dollars"].min()) if len(daily) else 0.0,
        "worst_5day_dollars": float(roll5.min()) if roll5.notna().any() else 0.0,
        "max_daily_risk_dollars": float(daily["weighted_risk_dollars"].max()) if len(daily) else 0.0,
    }


def evaluate_scope(
    df: pd.DataFrame, scope: str, profiles: dict[str, dict[str, bool]]
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    base = df.copy()
    base["w"] = 1.0
    base_is = summarize(base[~base["is_oos"]], "w")
    base_oos = summarize(base[base["is_oos"]], "w")
    base_full = summarize(base, "w")

    rows = []
    contribs: dict[str, pd.DataFrame] = {}
    for map_name in MAPS:
        work = df.copy()
        work["raw_weight"] = [raw_weight(map_name, row, profiles) for _, row in work.iterrows()]
        norm = 1.0 / float(work.loc[~work["is_oos"], "raw_weight"].mean())
        work["w"] = work["raw_weight"] * norm

        is_m = summarize(work[~work["is_oos"]], "w")
        oos_m = summarize(work[work["is_oos"]], "w")
        full_m = summarize(work, "w")

        expr_delta_is = is_m["exp_r"] - base_is["exp_r"]
        expr_delta_oos = oos_m["exp_r"] - base_oos["exp_r"]
        retention = expr_delta_oos / expr_delta_is if abs(expr_delta_is) > 1e-9 else float("nan")

        rows.append(
            {
                "scope": scope,
                "map": map_name,
                "full_delta_dollars": full_m["total_dollars"] - base_full["total_dollars"],
                "full_delta_r": full_m["total_r"] - base_full["total_r"],
                "full_sharpe_delta": full_m["sharpe_ann_r"] - base_full["sharpe_ann_r"],
                "full_maxdd_delta": full_m["max_dd_r"] - base_full["max_dd_r"],
                "worst_day_delta": full_m["worst_day_dollars"] - base_full["worst_day_dollars"],
                "worst_5day_delta": full_m["worst_5day_dollars"] - base_full["worst_5day_dollars"],
                "max_daily_risk_delta": full_m["max_daily_risk_dollars"] - base_full["max_daily_risk_dollars"],
                "is_exp_r_delta": expr_delta_is,
                "oos_exp_r_delta": expr_delta_oos,
                "oos_retention": retention,
            }
        )

        c = work.copy()
        c["base_dollars"] = c["pnl_dollars"]
        c["alt_dollars"] = c["pnl_dollars"] * c["w"]
        c["delta_dollars"] = c["alt_dollars"] - c["base_dollars"]
        contribs[map_name] = (
            c.groupby(["instrument", "orb_label"], as_index=False)[["delta_dollars"]]
            .sum()
            .sort_values("delta_dollars", ascending=False)
            .reset_index(drop=True)
        )

    return pd.DataFrame(rows), contribs


def emit(profiles: dict[str, dict[str, bool]], res: pd.DataFrame, contribs: dict[str, dict[str, pd.DataFrame]]) -> None:
    lines = [
        "# Garch Additive Sizing Audit",
        "",
        "**Date:** 2026-04-16",
        "**Pre-registration:** `docs/audit/hypotheses/2026-04-16-garch-additive-sizing-audit.yaml`",
        "**Purpose:** compare garch to ATR / overnight proxies inside the same clipped session-sizing scaffold.",
        "",
        "## Session scaffold",
        "",
        "| Session | High directional support | Low directional support | High monotonicity support |",
        "|---|---|---|---|",
    ]
    for sess, p in sorted(profiles.items()):
        lines.append(
            f"| {sess} | {'Y' if p['high_dir'] else '.'} | {'Y' if p['low_dir'] else '.'} | {'Y' if p['high_mono'] else '.'} |"
        )

    lines += [
        "",
        "## Map definitions",
        "",
        "- `GARCH_SESSION_CLIPPED`: session-clipped map using `garch_forecast_vol_pct`.",
        "- `ATR_SESSION_CLIPPED`: same session-clipped map using `atr_20_pct`.",
        "- `OVN_SESSION_CLIPPED`: same session-clipped map using `overnight_range_pct`.",
        "- `GARCH_ATR_MEAN_CLIPPED`: same scaffold using mean(`garch_pct`, `atr_20_pct`).",
        "- `GARCH_OVN_MEAN_CLIPPED`: same scaffold using mean(`garch_pct`, `overnight_range_pct`).",
        "",
    ]

    for scope in ["broad", "validated"]:
        sub = res[res["scope"] == scope].sort_values(["full_delta_dollars", "full_sharpe_delta"], ascending=False)
        lines += [
            f"## {scope.title()} scope",
            "",
            "| Map | Full Δ$ | Full ΔR | Sharpe Δ | MaxDD ΔR | Worst day Δ$ | Worst 5d Δ$ | Max daily risk Δ$ | IS ExpR Δ | OOS ExpR Δ | OOS retention |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for _, r in sub.iterrows():
            retain = "n/a" if pd.isna(r["oos_retention"]) else f"{r['oos_retention']:+.2f}"
            lines.append(
                f"| {r['map']} | {r['full_delta_dollars']:+.1f} | {r['full_delta_r']:+.1f} | {r['full_sharpe_delta']:+.3f} | "
                f"{r['full_maxdd_delta']:+.1f} | {r['worst_day_delta']:+.1f} | {r['worst_5day_delta']:+.1f} | "
                f"{r['max_daily_risk_delta']:+.1f} | {r['is_exp_r_delta']:+.4f} | {r['oos_exp_r_delta']:+.4f} | {retain} |"
            )
        best = sub.iloc[0]["map"] if len(sub) else None
        if best is not None:
            lines += [
                "",
                f"### {scope.title()} best-map contributions: `{best}`",
                "",
                "| Instrument | Session | Δ$ |",
                "|---|---|---|",
            ]
            for _, r in contribs[scope][best].head(15).iterrows():
                lines.append(f"| {r['instrument']} | {r['orb_label']} | {r['delta_dollars']:+.1f} |")

    lines += [
        "",
        "## Reading the audit",
        "",
        "- This is an additive-value comparison, not a new discovery sweep.",
        "- Positive `MaxDD ΔR` means drawdown became less severe.",
        "- Positive `Worst day/5d Δ$` means the loss became smaller in magnitude.",
        "- `Max daily risk Δ$` is a concentration proxy, not a breach simulation.",
        "",
    ]
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {OUTPUT_MD}")


def main() -> None:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    cells, _ = fam.build_cells()
    profiles = session_profiles(cells)
    all_res = []
    all_contribs: dict[str, dict[str, pd.DataFrame]] = {}
    for scope in ["broad", "validated"]:
        rows = load_scope_rows(con, scope)
        df = load_scope_trades(con, rows)
        res, contrib = evaluate_scope(df, scope, profiles)
        all_res.append(res)
        all_contribs[scope] = contrib
    con.close()
    emit(profiles, pd.concat(all_res, ignore_index=True), all_contribs)


if __name__ == "__main__":
    main()
