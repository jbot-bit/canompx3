"""Fair proxy-native sizing audit for garch vs related vol proxies.

Purpose:
  Re-run the additive comparison without forcing ATR / overnight proxies through
  garch's discovered session map. Each score earns its own session-family
  scaffold via the same fixed-threshold family audit, then gets evaluated inside
  the same normalized clipped sizing logic.

Pre-registration:
  docs/audit/hypotheses/2026-04-16-garch-proxy-native-sizing-audit.yaml

Output:
  docs/audit/results/2026-04-16-garch-proxy-native-sizing-audit.md
"""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import duckdb
import pandas as pd

from pipeline.paths import GOLD_DB_PATH
from research import garch_additive_sizing_audit as add
from research import garch_broad_exact_role_exhaustion as broad
from research import garch_regime_family_audit as fam

OUTPUT_MD = Path("docs/audit/results/2026-04-16-garch-proxy-native-sizing-audit.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class ProxyCell:
    orb_label: str
    direction: str
    high_sr_lift: float
    high_lift: float
    high_p_sharpe: float
    high_oos_lift: float | None
    low_sr_lift: float
    low_lift: float
    low_p_sharpe: float
    low_oos_lift: float | None
    shape_skip: bool
    tail_bias: float | None
    best_bucket: int | None


def score_sql(map_name: str) -> str:
    if map_name == "GARCH_SESSION_CLIPPED":
        return "d.garch_forecast_vol_pct"
    if map_name == "ATR_SESSION_CLIPPED":
        return "d.atr_20_pct"
    if map_name == "OVN_SESSION_CLIPPED":
        return "d.overnight_range_pct"
    if map_name == "GARCH_ATR_MEAN_CLIPPED":
        return "((d.garch_forecast_vol_pct + d.atr_20_pct) / 2.0)"
    if map_name == "GARCH_OVN_MEAN_CLIPPED":
        return "((d.garch_forecast_vol_pct + d.overnight_range_pct) / 2.0)"
    raise ValueError(map_name)


def load_score_trades(
    con: duckdb.DuckDBPyConnection,
    row: pd.Series,
    direction: str,
    map_name: str,
    *,
    is_oos: bool,
) -> pd.DataFrame:
    filter_sql, join_sql = broad.exact_filter_sql(row["filter_type"], row["orb_label"], row["instrument"])
    if filter_sql is None:
        return pd.DataFrame()
    date_clause = ">=" if is_oos else "<"
    score_expr = score_sql(map_name)
    q = f"""
    SELECT
      o.trading_day,
      o.pnl_r,
      {score_expr} AS gp
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
      AND {score_expr} IS NOT NULL
      AND d.orb_{row["orb_label"]}_break_dir = '{direction}'
      AND {filter_sql}
      AND o.trading_day {date_clause} DATE '{broad.IS_END}'
    ORDER BY o.trading_day
    """
    df = con.execute(q).df()
    if len(df) == 0:
        return df
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["year"] = df["trading_day"].dt.year
    df["pnl_r"] = df["pnl_r"].astype(float)
    df["gp"] = df["gp"].astype(float)
    return df


def build_cells_for_map(con: duckdb.DuckDBPyConnection, map_name: str) -> list[ProxyCell]:
    rows = broad.load_rows(con)
    rows = rows[rows["filter_type"].map(broad.in_scope)].copy()
    cells: list[ProxyCell] = []
    for _, row in rows.iterrows():
        for direction in ["long", "short"]:
            df = load_score_trades(con, row, direction, map_name, is_oos=False)
            if len(df) < broad.MIN_TOTAL:
                continue
            df_oos = load_score_trades(con, row, direction, map_name, is_oos=True)
            high = broad.test_spec(df, df_oos, broad.ThresholdSpec("high", 70))
            low = broad.test_spec(df, df_oos, broad.ThresholdSpec("low", 30))
            if high.get("skip") or low.get("skip"):
                continue
            shape = broad.ntile_shape(df)
            cells.append(
                ProxyCell(
                    orb_label=str(row["orb_label"]),
                    direction=direction,
                    high_sr_lift=float(high["sr_lift"]),
                    high_lift=float(high["lift"]),
                    high_p_sharpe=float(high["p_sharpe"]),
                    high_oos_lift=None if pd.isna(high["oos_lift"]) else float(high["oos_lift"]),
                    low_sr_lift=float(low["sr_lift"]),
                    low_lift=float(low["lift"]),
                    low_p_sharpe=float(low["p_sharpe"]),
                    low_oos_lift=None if pd.isna(low["oos_lift"]) else float(low["oos_lift"]),
                    shape_skip=bool(shape.get("skip", False)),
                    tail_bias=None if shape.get("skip") else float(shape["tail_bias"]),
                    best_bucket=None if shape.get("skip") else int(shape["best_bucket"]),
                )
            )
    return cells


def native_profiles(con: duckdb.DuckDBPyConnection) -> dict[str, dict[str, dict[str, bool]]]:
    out: dict[str, dict[str, dict[str, bool]]] = {}
    for map_name in add.MAPS:
        cells = build_cells_for_map(con, map_name)
        out[map_name] = add.session_profiles(cells)
    return out


def evaluate_scope_native(
    df: pd.DataFrame,
    scope: str,
    profiles_by_map: dict[str, dict[str, dict[str, bool]]],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    base = df.copy()
    base["w"] = 1.0
    base_is = add.summarize(base[~base["is_oos"]], "w")
    base_oos = add.summarize(base[base["is_oos"]], "w")
    base_full = add.summarize(base, "w")

    rows = []
    contribs: dict[str, pd.DataFrame] = {}
    for map_name in add.MAPS:
        profiles = profiles_by_map[map_name]
        work = df.copy()
        work["raw_weight"] = [add.raw_weight(map_name, row, profiles) for _, row in work.iterrows()]
        norm = 1.0 / float(work.loc[~work["is_oos"], "raw_weight"].mean())
        work["w"] = work["raw_weight"] * norm

        is_m = add.summarize(work[~work["is_oos"]], "w")
        oos_m = add.summarize(work[work["is_oos"]], "w")
        full_m = add.summarize(work, "w")

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


def scaffold_label(profiles: dict[str, dict[str, bool]]) -> str:
    sessions = []
    for sess, p in sorted(profiles.items()):
        bits = []
        if p.get("high_dir"):
            bits.append("H")
        if p.get("low_dir"):
            bits.append("L")
        if p.get("high_mono"):
            bits.append("M")
        if bits:
            sessions.append(f"{sess}({''.join(bits)})")
    return ", ".join(sessions)


def emit(
    profiles_by_map: dict[str, dict[str, dict[str, bool]]],
    res: pd.DataFrame,
    contribs: dict[str, dict[str, pd.DataFrame]],
) -> None:
    lines = [
        "# Garch Proxy-Native Sizing Audit",
        "",
        "**Date:** 2026-04-16",
        "**Pre-registration:** `docs/audit/hypotheses/2026-04-16-garch-proxy-native-sizing-audit.yaml`",
        "**Purpose:** compare garch to related vol proxies after letting each score earn its own session-family scaffold under the same locked family method.",
        "",
        "## Native scaffolds",
        "",
        "| Map | Native session scaffold |",
        "|---|---|",
    ]
    for map_name in add.MAPS:
        lines.append(f"| {map_name} | {scaffold_label(profiles_by_map[map_name]) or 'none'} |")

    for scope in ["broad", "validated"]:
        sub = res[res["scope"] == scope].sort_values(["full_delta_dollars", "full_sharpe_delta"], ascending=False)
        lines += [
            "",
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
            lines += ["", f"### {scope.title()} best-map contributions: `{best}`", "", "| Instrument | Session | Δ$ |", "|---|---|---|"]
            for _, r in contribs[scope][best].head(15).iterrows():
                lines.append(f"| {r['instrument']} | {r['orb_label']} | {r['delta_dollars']:+.1f} |")

    lines += [
        "",
        "## Reading the audit",
        "",
        "- This is the fair proxy-native counterpart to the common-scaffold additive audit.",
        "- Positive `MaxDD ΔR` means drawdown became less severe.",
        "- Positive `Worst day/5d Δ$` means the loss became smaller in magnitude.",
        "- `Max daily risk Δ$` is a concentration proxy, not a breach simulation.",
        "",
    ]
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {OUTPUT_MD}")


def main() -> None:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    profiles_by_map = native_profiles(con)
    all_res = []
    all_contribs: dict[str, dict[str, pd.DataFrame]] = {}
    for scope in ["broad", "validated"]:
        rows = add.load_scope_rows(con, scope)
        df = add.load_scope_trades(con, rows)
        res, contrib = evaluate_scope_native(df, scope, profiles_by_map)
        all_res.append(res)
        all_contribs[scope] = contrib
    con.close()
    emit(profiles_by_map, pd.concat(all_res, ignore_index=True), all_contribs)


if __name__ == "__main__":
    main()
