"""Validated-shelf provenance audit for garch partner-state representations.

Purpose:
  Test whether the current W2 partner states are principled local mechanism
  representations, or whether neighboring / alternate locked representations
  are better supported on the validated shelf.

Output:
  docs/audit/results/2026-04-16-garch-partner-state-provenance-audit.md
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

OUTPUT_MD = Path("docs/audit/results/2026-04-16-garch-partner-state-provenance-audit.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

MIN_TOTAL = 50
MIN_SIDE = 10
MIN_CONJ = 30
GARCH_HIGH = 70.0


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
class Candidate:
    mechanism: str
    name: str
    label: str
    current_w2: bool = False


CANDIDATES = [
    Candidate("M1", "OVN_NOT_HIGH_60", "overnight_range_pct < 60"),
    Candidate("M1", "OVN_NOT_HIGH_70", "overnight_range_pct < 70"),
    Candidate("M1", "OVN_NOT_HIGH_80", "overnight_range_pct < 80", current_w2=True),
    Candidate("M1", "OVN_MID_ONLY", "20 < overnight_range_pct < 80"),
    Candidate("M2", "ATRVEL_EXPANDING", "atr_vel_regime == Expanding", current_w2=True),
    Candidate("M2", "ATRVEL_GE_100", "atr_vel_ratio >= 1.00"),
    Candidate("M2", "ATRVEL_GE_105", "atr_vel_ratio >= 1.05"),
    Candidate("M2", "ATRVEL_GE_110", "atr_vel_ratio >= 1.10"),
    Candidate("M2", "ATR_PCT_GE_70", "atr_20_pct >= 70"),
    Candidate("M2", "ATR_PCT_GE_80", "atr_20_pct >= 80"),
]


def mechanism_label(mech: str) -> str:
    return {
        "M1": "latent_expansion",
        "M2": "active_transition",
    }[mech]


def sharpe_like(arr: pd.Series) -> float:
    arr = pd.Series(arr).astype(float)
    if len(arr) < 2:
        return 0.0
    sd = arr.std(ddof=1)
    return float(arr.mean() / sd) if sd and not pd.isna(sd) else 0.0


def exp_r(arr: pd.Series) -> float:
    return float(pd.Series(arr).astype(float).mean())


def compare_split(df: pd.DataFrame, on_mask: pd.Series) -> dict[str, object]:
    on = df.loc[on_mask, "pnl_r"].astype(float)
    off = df.loc[~on_mask, "pnl_r"].astype(float)
    if len(on) < MIN_SIDE or len(off) < MIN_SIDE:
        return {"valid": False, "n_on": int(len(on)), "n_off": int(len(off))}
    lift = exp_r(on) - exp_r(off)
    return {
        "valid": True,
        "n_on": int(len(on)),
        "n_off": int(len(off)),
        "n_compared": int(len(on) + len(off)),
        "lift": lift,
        "exp_on": exp_r(on),
        "exp_off": exp_r(off),
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
      d.atr_vel_regime,
      d.atr_20_pct
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
      AND d.orb_{row["orb_label"]}_break_dir = '{direction}'
      AND {filter_sql}
      AND o.trading_day {date_clause} DATE '{broad.IS_END}'
    ORDER BY o.trading_day
    """
    df = con.execute(q).df()
    if len(df) == 0:
        return df
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    for col in ["pnl_r", "gp", "overnight_range_pct", "atr_vel_ratio", "atr_20_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["atr_vel_regime"] = df["atr_vel_regime"].astype("string")
    return df


def garch_mask(df: pd.DataFrame) -> pd.Series:
    return df["gp"] >= GARCH_HIGH


def partner_mask(df: pd.DataFrame, cand: Candidate) -> tuple[pd.Series, pd.Series]:
    if cand.name == "OVN_NOT_HIGH_60":
        return df["overnight_range_pct"] < 60, df["overnight_range_pct"].notna()
    if cand.name == "OVN_NOT_HIGH_70":
        return df["overnight_range_pct"] < 70, df["overnight_range_pct"].notna()
    if cand.name == "OVN_NOT_HIGH_80":
        return df["overnight_range_pct"] < 80, df["overnight_range_pct"].notna()
    if cand.name == "OVN_MID_ONLY":
        return ((df["overnight_range_pct"] > 20) & (df["overnight_range_pct"] < 80)), df["overnight_range_pct"].notna()
    if cand.name == "ATRVEL_EXPANDING":
        return df["atr_vel_regime"] == "Expanding", df["atr_vel_regime"].notna()
    if cand.name == "ATRVEL_GE_100":
        return df["atr_vel_ratio"] >= 1.00, df["atr_vel_ratio"].notna()
    if cand.name == "ATRVEL_GE_105":
        return df["atr_vel_ratio"] >= 1.05, df["atr_vel_ratio"].notna()
    if cand.name == "ATRVEL_GE_110":
        return df["atr_vel_ratio"] >= 1.10, df["atr_vel_ratio"].notna()
    if cand.name == "ATR_PCT_GE_70":
        return df["atr_20_pct"] >= 70, df["atr_20_pct"].notna()
    if cand.name == "ATR_PCT_GE_80":
        return df["atr_20_pct"] >= 80, df["atr_20_pct"].notna()
    raise ValueError(cand.name)


def analyze_cell(df: pd.DataFrame, df_oos: pd.DataFrame, cand: Candidate) -> dict[str, object]:
    if len(df) < MIN_TOTAL:
        return {"valid": False}
    g = garch_mask(df)
    fav, valid_partner = partner_mask(df, cand)
    conj = g & fav
    g_sub = df.loc[g & valid_partner].copy()
    p_sub = df.loc[valid_partner].copy()

    partner_inside_garch = compare_split(g_sub, fav.loc[g_sub.index]) if len(g_sub) >= (2 * MIN_SIDE) else {"valid": False}
    garch_inside_partner = compare_split(p_sub, g.loc[p_sub.index]) if len(p_sub) >= (2 * MIN_SIDE) else {"valid": False}

    n_conj = int(conj.sum())
    base_exp = exp_r(df["pnl_r"])
    conj_exp = exp_r(df.loc[conj, "pnl_r"]) if n_conj else float("nan")
    conj_sr = sharpe_like(df.loc[conj, "pnl_r"]) if n_conj else float("nan")

    oos = {"n_conj_oos": 0, "exp_conj_oos": None}
    if len(df_oos) > 0:
        g_oos = garch_mask(df_oos)
        fav_oos, _ = partner_mask(df_oos, cand)
        conj_oos = g_oos & fav_oos
        oos["n_conj_oos"] = int(conj_oos.sum())
        oos["exp_conj_oos"] = exp_r(df_oos.loc[conj_oos, "pnl_r"]) if conj_oos.sum() >= 3 else None

    return {
        "valid": True,
        "n_total": int(len(df)),
        "base_exp": base_exp,
        "n_conj": n_conj,
        "conj_exp": conj_exp,
        "conj_exp_minus_base": (conj_exp - base_exp) if n_conj else float("nan"),
        "sr_conj": conj_sr,
        "support_partner_inside_garch": bool(partner_inside_garch.get("support", False)),
        "valid_partner_inside_garch": bool(partner_inside_garch["valid"]),
        "n_partner_inside_garch": int(partner_inside_garch.get("n_compared", 0)),
        "lift_partner_inside_garch": partner_inside_garch.get("lift"),
        "support_garch_inside_partner": bool(garch_inside_partner.get("support", False)),
        "valid_garch_inside_partner": bool(garch_inside_partner["valid"]),
        "n_garch_inside_partner": int(garch_inside_partner.get("n_compared", 0)),
        "lift_garch_inside_partner": garch_inside_partner.get("lift"),
        **oos,
    }


def to_row(family: Family, cand: Candidate, row: pd.Series, direction: str, res: dict[str, object]) -> dict[str, object]:
    return {
        "family": family.label,
        "session": family.session,
        "mechanism": cand.mechanism,
        "candidate": cand.name,
        "candidate_label": cand.label,
        "current_w2": cand.current_w2,
        "instrument": row["instrument"],
        "direction": direction,
        "filter_type": row["filter_type"],
        "n_total": res["n_total"],
        "base_exp": res["base_exp"],
        "n_conj": res["n_conj"],
        "conj_exp": res["conj_exp"],
        "conj_exp_minus_base": res["conj_exp_minus_base"],
        "sr_conj": res["sr_conj"],
        "valid_partner_inside_garch": res["valid_partner_inside_garch"],
        "support_partner_inside_garch": res["support_partner_inside_garch"],
        "n_partner_inside_garch": res["n_partner_inside_garch"],
        "lift_partner_inside_garch": res["lift_partner_inside_garch"],
        "valid_garch_inside_partner": res["valid_garch_inside_partner"],
        "support_garch_inside_partner": res["support_garch_inside_partner"],
        "n_garch_inside_partner": res["n_garch_inside_partner"],
        "lift_garch_inside_partner": res["lift_garch_inside_partner"],
        "n_conj_oos": res["n_conj_oos"],
        "exp_conj_oos": res["exp_conj_oos"],
    }


def summarize_candidate(sub: pd.DataFrame) -> dict[str, object]:
    if len(sub) == 0:
        return {}
    n_total = int(sub["n_total"].sum())
    n_conj = int(sub["n_conj"].sum())
    base_exp = float((sub["base_exp"] * sub["n_total"]).sum() / n_total) if n_total else float("nan")
    conj_exp = float((sub["conj_exp"] * sub["n_conj"]).sum() / n_conj) if n_conj else float("nan")
    return {
        "cells": int(len(sub)),
        "n_total": n_total,
        "n_conj": n_conj,
        "base_exp": base_exp,
        "conj_exp": conj_exp,
        "delta": conj_exp - base_exp if n_conj else float("nan"),
        "support_cells_partner_inside_garch": int(sub.loc[sub["valid_partner_inside_garch"], "support_partner_inside_garch"].sum()),
        "valid_cells_partner_inside_garch": int(sub["valid_partner_inside_garch"].sum()),
        "support_weight_partner_inside_garch": int(
            sub.loc[sub["valid_partner_inside_garch"] & sub["support_partner_inside_garch"], "n_partner_inside_garch"].sum()
        ),
        "valid_weight_partner_inside_garch": int(
            sub.loc[sub["valid_partner_inside_garch"], "n_partner_inside_garch"].sum()
        ),
        "support_cells_garch_inside_partner": int(sub.loc[sub["valid_garch_inside_partner"], "support_garch_inside_partner"].sum()),
        "valid_cells_garch_inside_partner": int(sub["valid_garch_inside_partner"].sum()),
        "support_weight_garch_inside_partner": int(
            sub.loc[sub["valid_garch_inside_partner"] & sub["support_garch_inside_partner"], "n_garch_inside_partner"].sum()
        ),
        "valid_weight_garch_inside_partner": int(
            sub.loc[sub["valid_garch_inside_partner"], "n_garch_inside_partner"].sum()
        ),
        "n_conj_oos": int(sub["n_conj_oos"].sum()),
        "exp_conj_oos": (
            float(
                (
                    sub.loc[sub["exp_conj_oos"].notna(), "exp_conj_oos"]
                    * sub.loc[sub["exp_conj_oos"].notna(), "n_conj_oos"]
                ).sum()
                / sub.loc[sub["exp_conj_oos"].notna(), "n_conj_oos"].sum()
            )
            if sub.loc[sub["exp_conj_oos"].notna(), "n_conj_oos"].sum() > 0
            else np.nan
        ),
    }


def support_share(num: int, den: int) -> float:
    return float(num / den) if den > 0 else float("nan")


def viable(summary: dict[str, object]) -> bool:
    if not summary:
        return False
    cond_share = support_share(summary["support_weight_partner_inside_garch"], summary["valid_weight_partner_inside_garch"])
    return (
        summary["n_conj"] >= MIN_CONJ
        and not pd.isna(summary["delta"])
        and summary["delta"] > 0
        and not pd.isna(cond_share)
        and cond_share >= 0.5
    )


def family_mech_verdict(mech_sub: pd.DataFrame) -> tuple[str, str, pd.DataFrame]:
    summaries = []
    for candidate, csub in mech_sub.groupby("candidate", sort=False):
        s = summarize_candidate(csub)
        s["candidate"] = candidate
        s["candidate_label"] = csub["candidate_label"].iloc[0]
        s["current_w2"] = bool(csub["current_w2"].iloc[0])
        s["support_share_partner_inside_garch"] = support_share(
            s["support_weight_partner_inside_garch"], s["valid_weight_partner_inside_garch"]
        )
        s["support_share_garch_inside_partner"] = support_share(
            s["support_weight_garch_inside_partner"], s["valid_weight_garch_inside_partner"]
        )
        s["viable"] = viable(s)
        summaries.append(s)
    s_df = pd.DataFrame(summaries).sort_values(["current_w2", "delta", "support_share_partner_inside_garch"], ascending=[False, False, False])
    if len(s_df) == 0:
        return "unclear", "no_rows", s_df

    current = s_df[s_df["current_w2"]].iloc[0] if s_df["current_w2"].any() else None
    viable_df = s_df[s_df["viable"]].sort_values(["delta", "support_share_partner_inside_garch", "n_conj"], ascending=[False, False, False])

    if len(viable_df) == 0:
        if s_df["n_conj"].max() < MIN_CONJ:
            return "unclear", "thin_conjunctions", s_df
        return "weak_mechanism", "no_viable_representation", s_df

    best = viable_df.iloc[0]
    if current is not None and bool(current["viable"]):
        if best["candidate"] == current["candidate"]:
            neighbor = viable_df[viable_df["candidate"] != current["candidate"]]
            if len(neighbor) > 0 and abs(float(neighbor.iloc[0]["delta"]) - float(current["delta"])) <= 0.05:
                return "neighbor_stable", "carry_mechanism_no_unique_cutoff", s_df
            return "supported_current", "carry_current_representation", s_df
        if abs(float(best["delta"]) - float(current["delta"])) <= 0.05:
            return "neighbor_stable", "carry_mechanism_no_unique_cutoff", s_df
        return "alternate_better", f"locked_alternate_beats_current:{best['candidate']}", s_df

    if current is not None and not bool(current["viable"]):
        return "alternate_better", f"current_demoted_locked_alternate:{best['candidate']}", s_df

    return "unclear", "mixed", s_df


def build() -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    rows = load_rows(con)
    result_rows: list[dict[str, object]] = []
    summaries: dict[tuple[str, str], pd.DataFrame] = {}

    for family in FAMILIES:
        fam_rows = rows[rows["orb_label"] == family.session].copy()
        for cand in CANDIDATES:
            for _, row in fam_rows.iterrows():
                for direction in ["long", "short"]:
                    df = load_trades(con, row, direction, is_oos=False)
                    if len(df) < MIN_TOTAL:
                        continue
                    df_oos = load_trades(con, row, direction, is_oos=True)
                    res = analyze_cell(df, df_oos, cand)
                    if not res["valid"]:
                        continue
                    result_rows.append(to_row(family, cand, row, direction, res))

    con.close()
    df = pd.DataFrame(result_rows)
    if len(df) == 0:
        return df, summaries

    for family in FAMILIES:
        fam = df[df["family"] == family.label]
        if len(fam) == 0:
            continue
        for mech in sorted(fam["mechanism"].unique()):
            verdict, implication, s_df = family_mech_verdict(fam[fam["mechanism"] == mech].copy())
            if len(s_df) == 0:
                continue
            s_df = s_df.copy()
            s_df["verdict"] = verdict
            s_df["implication"] = implication
            summaries[(family.label, mech)] = s_df
    return df, summaries


def emit(df: pd.DataFrame, summaries: dict[tuple[str, str], pd.DataFrame]) -> None:
    lines = [
        "# Garch Partner-State Provenance Audit",
        "",
        "**Date:** 2026-04-16",
        "**Boundary:** validated shelf only, exact canonical joins, no deployment conclusions.",
        "",
        "This stage asks whether the current W2 partner encodings are principled local mechanism representations, or whether a nearby / alternate locked representation is better supported.",
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
        for mech in ["M1", "M2"]:
            s_df = summaries.get((family.label, mech))
            if s_df is None or len(s_df) == 0:
                continue
            verdict = s_df["verdict"].iloc[0]
            implication = s_df["implication"].iloc[0]
            lines.extend(
                [
                    f"### {mech} {mechanism_label(mech)}",
                    "",
                    f"- Verdict: **{verdict}**",
                    f"- Allowed implication: **{implication}**",
                    "",
                    "| Candidate | Current W2 | Cells | N total | N conj | Base ExpR | Conj ExpR | Delta | P|G support | G|P support | OOS conj N | OOS conj ExpR |",
                    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for _, row in s_df.sort_values(["current_w2", "delta", "support_share_partner_inside_garch"], ascending=[False, False, False]).iterrows():
                lines.append(
                    f"| {row['candidate']} | {'yes' if row['current_w2'] else 'no'} | {int(row['cells'])} | "
                    f"{int(row['n_total'])} | {int(row['n_conj'])} | {row['base_exp']:+.3f} | "
                    f"{'' if pd.isna(row['conj_exp']) else f'{row['conj_exp']:+.3f}'} | "
                    f"{'' if pd.isna(row['delta']) else f'{row['delta']:+.3f}'} | "
                    f"{'' if pd.isna(row['support_share_partner_inside_garch']) else f'{row['support_share_partner_inside_garch']:.2f}'} | "
                    f"{'' if pd.isna(row['support_share_garch_inside_partner']) else f'{row['support_share_garch_inside_partner']:.2f}'} | "
                    f"{int(row['n_conj_oos'])} | "
                    f"{'' if pd.isna(row['exp_conj_oos']) else f'{row['exp_conj_oos']:+.3f}'} |"
                )
            lines.append("")

    lines.extend(
        [
            "## Guardrails",
            "",
            "- This is a representation audit, not a new discovery sweep.",
            "- 2026 OOS context is descriptive only and was not used to choose the preferred representation.",
            "- `supported_current` means the current W2 representation remains defensible locally; it does not mean deployment-ready.",
            "- `alternate_better` means a locked neighboring or alternate representation beat the current W2 state on the validated shelf.",
            "- Prior-day levels and prior-session carry remain in the queue, but were intentionally kept out of this stage.",
        ]
    )

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df, summaries = build()
    emit(df, summaries)
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
