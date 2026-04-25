"""Validated-shelf mechanism pairing audit for garch W2.

Purpose:
  Test two locked garch-anchored mechanism pairings on the validated shelf only:
    - M1 latent expansion: high garch + lower/moderate overnight
    - M2 active transition: high garch + expanding atr_vel

This stage is not a deployment test. It asks whether the pairing improves setup
quality locally on exact validated populations and whether the correct read is:
  - garch_distinct
  - complementary_pair
  - partner_dominant
  - unclear

Output:
  docs/audit/results/2026-04-16-garch-w2-mechanism-pairing-audit.md
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
import numpy as np
import pandas as pd

from pipeline.paths import GOLD_DB_PATH
from research import garch_broad_exact_role_exhaustion as broad

OUTPUT_MD = Path("docs/audit/results/2026-04-16-garch-w2-mechanism-pairing-audit.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

MIN_TOTAL = 50
MIN_SIDE = 10
MIN_CONJ = 30
GARCH_HIGH = 70.0
OVN_HIGH = 80.0


@dataclass(frozen=True)
class Family:
    session: str
    label: str


FAMILIES = [
    Family("COMEX_SETTLE", "COMEX_SETTLE_high"),
    Family("EUROPE_FLOW", "EUROPE_FLOW_high"),
    Family("TOKYO_OPEN", "TOKYO_OPEN_high"),
    Family("SINGAPORE_OPEN", "SINGAPORE_OPEN_high"),
    Family("LONDON_METALS", "LONDON_METALS_high"),
]


@dataclass(frozen=True)
class Mechanism:
    name: str
    label: str
    description: str


MECHANISMS = [
    Mechanism("M1", "latent_expansion", "high garch + overnight not high"),
    Mechanism("M2", "active_transition", "high garch + atr_vel expanding"),
]


def sharpe_like(arr: pd.Series) -> float:
    arr = pd.Series(arr).astype(float)
    if len(arr) < 2:
        return 0.0
    sd = arr.std(ddof=1)
    return float(arr.mean() / sd) if sd and not pd.isna(sd) else 0.0


def exp_r(arr: pd.Series) -> float:
    return float(pd.Series(arr).astype(float).mean())


def compare_split(df: pd.DataFrame, mask: pd.Series) -> dict[str, object]:
    on = df.loc[mask, "pnl_r"].astype(float)
    off = df.loc[~mask, "pnl_r"].astype(float)
    if len(on) < MIN_SIDE or len(off) < MIN_SIDE:
        return {"valid": False, "n_on": int(len(on)), "n_off": int(len(off))}
    lift = exp_r(on) - exp_r(off)
    return {
        "valid": True,
        "n_on": int(len(on)),
        "n_off": int(len(off)),
        "exp_on": exp_r(on),
        "exp_off": exp_r(off),
        "lift": lift,
        "sr_lift": sharpe_like(on) - sharpe_like(off),
        "support": lift > 0,
    }


def load_rows(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    rows = broad.load_rows(con)
    rows = rows[rows["src"] == "validated"].copy()
    rows = rows[rows["filter_type"].map(broad.in_scope)].copy()
    rows = rows[rows["orb_label"].isin({f.session for f in FAMILIES})].copy()
    return rows.reset_index(drop=True)


def load_trades(
    con: duckdb.DuckDBPyConnection,
    row: pd.Series,
    direction: str,
    *,
    is_oos: bool,
) -> pd.DataFrame:
    filter_sql, join_sql = broad.exact_filter_sql(row["filter_type"], row["orb_label"], row["instrument"])
    if filter_sql is None:
        return pd.DataFrame()
    date_clause = ">=" if is_oos else "<"
    q = f"""
    SELECT
      o.trading_day,
      o.pnl_r,
      d.garch_forecast_vol_pct AS gp,
      d.overnight_range_pct,
      d.atr_vel_ratio,
      d.atr_vel_regime
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
      AND d.garch_forecast_vol_pct IS NOT NULL
      AND d.overnight_range_pct IS NOT NULL
      AND d.orb_{row["orb_label"]}_break_dir = '{direction}'
      AND {filter_sql}
      AND o.trading_day {date_clause} DATE '{broad.IS_END}'
    ORDER BY o.trading_day
    """
    df = con.execute(q).df()
    if len(df) == 0:
        return df
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    for col in ["pnl_r", "gp", "overnight_range_pct", "atr_vel_ratio"]:
        df[col] = df[col].astype(float)
    df["atr_vel_regime"] = df["atr_vel_regime"].astype("string")
    return df


def garch_high(df: pd.DataFrame) -> pd.Series:
    return df["gp"] >= GARCH_HIGH


def partner_masks(df: pd.DataFrame, mech: Mechanism) -> tuple[pd.Series, pd.Series]:
    if mech.name == "M1":
        fav = df["overnight_range_pct"] < OVN_HIGH
        unfav = df["overnight_range_pct"] >= OVN_HIGH
        return fav, unfav
    if mech.name == "M2":
        fav = df["atr_vel_regime"] == "Expanding"
        unfav = df["atr_vel_regime"].notna() & (df["atr_vel_regime"] != "Expanding")
        return fav, unfav
    raise ValueError(mech.name)


def descriptive_partner_bucket(df: pd.DataFrame, mech: Mechanism) -> pd.DataFrame:
    if mech.name == "M1":
        labels = pd.Series(index=df.index, dtype="object")
        labels.loc[df["overnight_range_pct"] <= 20] = "overnight_low"
        labels.loc[(df["overnight_range_pct"] > 20) & (df["overnight_range_pct"] < 80)] = "overnight_mid"
        labels.loc[df["overnight_range_pct"] >= 80] = "overnight_high"
    else:
        labels = df["atr_vel_regime"].astype("object")
    work = df.copy()
    work["bucket"] = labels
    work = work.dropna(subset=["bucket"])
    if len(work) == 0:
        return pd.DataFrame()
    return (
        work.groupby("bucket", as_index=False)
        .agg(n=("pnl_r", "size"), exp_r=("pnl_r", "mean"), total_r=("pnl_r", "sum"))
        .sort_values("bucket")
        .reset_index(drop=True)
    )


def analyze_cell(df: pd.DataFrame, df_oos: pd.DataFrame, mech: Mechanism) -> dict[str, object]:
    if len(df) < MIN_TOTAL:
        return {"valid": False}
    g = garch_high(df)
    p_fav, p_unfav = partner_masks(df, mech)
    p_any = p_fav | p_unfav
    conj = g & p_fav

    base_exp = exp_r(df["pnl_r"])
    base_sr = sharpe_like(df["pnl_r"])
    g_marg = compare_split(df, g)
    p_marg = compare_split(df.loc[p_any], p_fav.loc[p_any]) if p_any.sum() >= (2 * MIN_SIDE) else {"valid": False}

    g_sub = df.loc[g & (p_fav | p_unfav)].copy()
    g_cond = compare_split(g_sub, p_fav.loc[g_sub.index]) if len(g_sub) >= (2 * MIN_SIDE) else {"valid": False}

    p_sub = df.loc[p_any].copy()
    p_cond = compare_split(p_sub, g.loc[p_sub.index]) if len(p_sub) >= (2 * MIN_SIDE) else {"valid": False}

    conj_n = int(conj.sum())
    conj_exp = exp_r(df.loc[conj, "pnl_r"]) if conj_n else float("nan")
    conj_sr = sharpe_like(df.loc[conj, "pnl_r"]) if conj_n else float("nan")

    oos = {}
    if len(df_oos) > 0:
        g_oos = garch_high(df_oos)
        p_fav_oos, p_unfav_oos = partner_masks(df_oos, mech)
        conj_oos = g_oos & p_fav_oos
        oos["n_conj_oos"] = int(conj_oos.sum())
        oos["exp_conj_oos"] = exp_r(df_oos.loc[conj_oos, "pnl_r"]) if conj_oos.sum() >= 3 else None
    else:
        oos["n_conj_oos"] = 0
        oos["exp_conj_oos"] = None

    return {
        "valid": True,
        "n_total": int(len(df)),
        "base_exp_r": base_exp,
        "base_sr": base_sr,
        "n_garch_high": int(g.sum()),
        "n_partner_fav": int(p_fav.sum()),
        "n_conj": conj_n,
        "exp_garch_high": exp_r(df.loc[g, "pnl_r"]) if g.sum() else float("nan"),
        "exp_partner_fav": exp_r(df.loc[p_fav, "pnl_r"]) if p_fav.sum() else float("nan"),
        "exp_conj": conj_exp,
        "sr_conj": conj_sr,
        "conj_fire_pct": float(conj.mean()),
        "g_marg": g_marg,
        "p_marg": p_marg,
        "partner_inside_garch": g_cond,
        "garch_inside_partner": p_cond,
        **oos,
    }


def family_verdict(df: pd.DataFrame, mech: Mechanism) -> tuple[str, str]:
    g_support = df["g_marg_support"].mean() if df["g_marg_valid"].any() else np.nan
    p_support = df["p_marg_support"].mean() if df["p_marg_valid"].any() else np.nan
    cond_partner_support = (
        df["partner_inside_garch_support"].mean() if df["partner_inside_garch_valid"].any() else np.nan
    )
    cond_garch_support = df["garch_inside_partner_support"].mean() if df["garch_inside_partner_valid"].any() else np.nan
    conj_n = float(df["n_conj"].sum())
    conj_exp = float((df["exp_conj"] * df["n_conj"]).sum() / conj_n) if conj_n > 0 else np.nan
    g_exp = (
        float((df["exp_garch_high"] * df["n_garch_high"]).sum() / df["n_garch_high"].sum())
        if df["n_garch_high"].sum() > 0
        else np.nan
    )
    p_exp = (
        float((df["exp_partner_fav"] * df["n_partner_fav"]).sum() / df["n_partner_fav"].sum())
        if df["n_partner_fav"].sum() > 0
        else np.nan
    )
    base_exp = (
        float((df["base_exp_r"] * df["n_total"]).sum() / df["n_total"].sum()) if df["n_total"].sum() > 0 else np.nan
    )

    if conj_n < MIN_CONJ:
        return "unclear", "unclear"

    cond_partner_pos = bool(cond_partner_support >= 0.5) if not pd.isna(cond_partner_support) else False
    cond_garch_pos = bool(cond_garch_support >= 0.5) if not pd.isna(cond_garch_support) else False
    g_pos = bool(g_support >= 0.5) if not pd.isna(g_support) else False
    p_pos = bool(p_support >= 0.5) if not pd.isna(p_support) else False

    if cond_partner_pos and cond_garch_pos and conj_exp > max(g_exp, p_exp, base_exp):
        return "complementary_pair", "R7_candidate_only"
    if g_pos and (not cond_partner_pos) and (pd.isna(conj_exp) or conj_exp <= g_exp):
        return "garch_distinct", "R3/R7_candidate_only"
    if p_pos and (not cond_garch_pos) and (pd.isna(conj_exp) or conj_exp <= p_exp):
        return "partner_dominant", "demote_garch_for_family"
    return "unclear", "unclear"


def pooled_summary(df: pd.DataFrame) -> dict[str, object]:
    if len(df) == 0:
        return {"n_total": 0}
    out = {
        "n_total": int(df["n_total"].sum()),
        "n_conj": int(df["n_conj"].sum()),
        "mean_base_exp": float((df["base_exp_r"] * df["n_total"]).sum() / df["n_total"].sum()),
        "mean_conj_exp": float((df["exp_conj"] * df["n_conj"]).sum() / df["n_conj"].sum())
        if df["n_conj"].sum() > 0
        else float("nan"),
        "g_marg_support_cells": int(df.loc[df["g_marg_valid"], "g_marg_support"].sum()),
        "g_marg_valid_cells": int(df["g_marg_valid"].sum()),
        "p_marg_support_cells": int(df.loc[df["p_marg_valid"], "p_marg_support"].sum()),
        "p_marg_valid_cells": int(df["p_marg_valid"].sum()),
        "partner_inside_garch_support_cells": int(
            df.loc[df["partner_inside_garch_valid"], "partner_inside_garch_support"].sum()
        ),
        "partner_inside_garch_valid_cells": int(df["partner_inside_garch_valid"].sum()),
        "garch_inside_partner_support_cells": int(
            df.loc[df["garch_inside_partner_valid"], "garch_inside_partner_support"].sum()
        ),
        "garch_inside_partner_valid_cells": int(df["garch_inside_partner_valid"].sum()),
    }
    return out


def to_row(
    family: Family, mech: Mechanism, row: pd.Series, direction: str, res: dict[str, object]
) -> dict[str, object]:
    return {
        "family": family.label,
        "session": family.session,
        "mechanism": mech.name,
        "mechanism_label": mech.label,
        "instrument": row["instrument"],
        "direction": direction,
        "filter_type": row["filter_type"],
        "n_total": res["n_total"],
        "base_exp_r": res["base_exp_r"],
        "n_garch_high": res["n_garch_high"],
        "exp_garch_high": res["exp_garch_high"],
        "n_partner_fav": res["n_partner_fav"],
        "exp_partner_fav": res["exp_partner_fav"],
        "n_conj": res["n_conj"],
        "exp_conj": res["exp_conj"],
        "sr_conj": res["sr_conj"],
        "conj_fire_pct": res["conj_fire_pct"],
        "g_marg_valid": bool(res["g_marg"]["valid"]),
        "g_marg_lift": res["g_marg"].get("lift"),
        "g_marg_support": bool(res["g_marg"].get("support", False)),
        "p_marg_valid": bool(res["p_marg"]["valid"]),
        "p_marg_lift": res["p_marg"].get("lift"),
        "p_marg_support": bool(res["p_marg"].get("support", False)),
        "partner_inside_garch_valid": bool(res["partner_inside_garch"]["valid"]),
        "partner_inside_garch_lift": res["partner_inside_garch"].get("lift"),
        "partner_inside_garch_support": bool(res["partner_inside_garch"].get("support", False)),
        "garch_inside_partner_valid": bool(res["garch_inside_partner"]["valid"]),
        "garch_inside_partner_lift": res["garch_inside_partner"].get("lift"),
        "garch_inside_partner_support": bool(res["garch_inside_partner"].get("support", False)),
        "n_conj_oos": res["n_conj_oos"],
        "exp_conj_oos": res["exp_conj_oos"],
    }


def build() -> tuple[pd.DataFrame, dict[tuple[str, str, str], pd.DataFrame]]:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    rows = load_rows(con)
    result_rows = []
    bucket_tables: dict[tuple[str, str, str], pd.DataFrame] = {}
    for family in FAMILIES:
        fam_rows = rows[rows["orb_label"] == family.session].copy()
        for mech in MECHANISMS:
            for _, row in fam_rows.iterrows():
                for direction in ["long", "short"]:
                    df = load_trades(con, row, direction, is_oos=False)
                    if len(df) < MIN_TOTAL:
                        continue
                    df_oos = load_trades(con, row, direction, is_oos=True)
                    res = analyze_cell(df, df_oos, mech)
                    if not res["valid"]:
                        continue
                    result_rows.append(to_row(family, mech, row, direction, res))
                    bucket_tables[
                        (family.label, mech.name, f"{row['instrument']} {direction} {row['filter_type']}")
                    ] = descriptive_partner_bucket(
                        df.loc[garch_high(df)].copy(),
                        mech,
                    )
    con.close()
    return pd.DataFrame(result_rows), bucket_tables


def emit(df: pd.DataFrame, bucket_tables: dict[tuple[str, str, str], pd.DataFrame]) -> None:
    lines = [
        "# Garch W2 Mechanism Pairing Audit",
        "",
        "**Date:** 2026-04-16",
        "**Boundary:** validated shelf only, exact filter semantics, no deployment conclusions.",
        "**Mechanisms:**",
        "- `M1 latent_expansion`: `garch_high AND overnight_range_pct < 80`",
        "- `M2 active_transition`: `garch_high AND atr_vel_regime = Expanding`",
        "",
        "2026 OOS is descriptive only in this stage.",
        "",
    ]

    if len(df) == 0:
        lines.append("No validated rows met minimum support.")
        OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
        return

    for family in FAMILIES:
        fam = df[df["family"] == family.label].copy()
        if len(fam) == 0:
            continue
        lines.extend([f"## {family.label}", ""])
        for mech in MECHANISMS:
            sub = fam[fam["mechanism"] == mech.name].copy()
            if len(sub) == 0:
                continue
            verdict, role = family_verdict(sub, mech)
            pooled = pooled_summary(sub)
            lines.extend(
                [
                    f"### {mech.name} {mech.label}",
                    "",
                    f"- Cells: **{len(sub)}**",
                    f"- Pooled `N_total`: **{pooled['n_total']}**",
                    f"- Pooled conjunction `N`: **{pooled['n_conj']}**",
                    f"- Base ExpR: **{pooled['mean_base_exp']:+.3f}**",
                    f"- Conjunction ExpR: **{pooled['mean_conj_exp']:+.3f}**"
                    if not pd.isna(pooled["mean_conj_exp"])
                    else "- Conjunction ExpR: n/a",
                    "",
                    "| Check | Support / valid cells |",
                    "|---|---:|",
                    f"| garch marginal | {pooled['g_marg_support_cells']}/{pooled['g_marg_valid_cells']} |",
                    f"| partner marginal | {pooled['p_marg_support_cells']}/{pooled['p_marg_valid_cells']} |",
                    f"| partner inside garch | {pooled['partner_inside_garch_support_cells']}/{pooled['partner_inside_garch_valid_cells']} |",
                    f"| garch inside partner | {pooled['garch_inside_partner_support_cells']}/{pooled['garch_inside_partner_valid_cells']} |",
                    "",
                    "| Instrument | Dir | Filter | N | Base ExpR | Garch ExpR | Partner ExpR | Conj N | Conj ExpR | G marg | P marg | P|G | G|P | OOS Conj N | OOS Conj ExpR |",
                    "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for _, row in sub.sort_values(["instrument", "direction", "filter_type"]).iterrows():
                lines.append(
                    f"| {row['instrument']} | {row['direction']} | {row['filter_type']} | {int(row['n_total'])} | "
                    f"{row['base_exp_r']:+.3f} | {row['exp_garch_high']:+.3f} | {row['exp_partner_fav']:+.3f} | "
                    f"{int(row['n_conj'])} | {row['exp_conj']:+.3f} | "
                    f"{'' if pd.isna(row['g_marg_lift']) else f'{row["g_marg_lift"]:+.3f}'} | "
                    f"{'' if pd.isna(row['p_marg_lift']) else f'{row["p_marg_lift"]:+.3f}'} | "
                    f"{'' if pd.isna(row['partner_inside_garch_lift']) else f'{row["partner_inside_garch_lift"]:+.3f}'} | "
                    f"{'' if pd.isna(row['garch_inside_partner_lift']) else f'{row["garch_inside_partner_lift"]:+.3f}'} | "
                    f"{int(row['n_conj_oos'])} | "
                    f"{'' if row['exp_conj_oos'] is None else f'{row["exp_conj_oos"]:+.3f}'} |"
                )
            lines.extend(
                [
                    "",
                    f"- Verdict: **{verdict}**",
                    f"- Allowed implication: **{role}**",
                    "",
                    "#### Descriptive buckets inside garch_high",
                    "",
                ]
            )
            added = False
            for key, table in bucket_tables.items():
                fam_key, mech_key, cell_key = key
                if fam_key != family.label or mech_key != mech.name or len(table) == 0:
                    continue
                added = True
                lines.append(f"**{cell_key}**")
                lines.append("")
                lines.append("| Bucket | N | ExpR | Total R |")
                lines.append("|---|---:|---:|---:|")
                for _, row in table.iterrows():
                    lines.append(f"| {row['bucket']} | {int(row['n'])} | {row['exp_r']:+.3f} | {row['total_r']:+.1f} |")
                lines.append("")

            if not added:
                lines.append("No descriptive bucket rows met support.")
                lines.append("")

    lines.extend(
        [
            "## Guardrails",
            "",
            "- This stage uses raw validated rows rejoined to canonical layers; prior report summaries are not authority.",
            "- `garch_distinct` here means the partner did not add enough on the validated shelf, not that garch is a proven deployment edge.",
            "- `partner_dominant` means the partner explains more of the local family utility than garch.",
            "- 2026 OOS rows are descriptive only and were not used to choose verdicts.",
        ]
    )

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df, bucket_tables = build()
    emit(df, bucket_tables)
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
