"""Validated-shelf utility audit for the surviving M2 mechanism.

Purpose:
  Test whether the conservative family-local M2 representation chosen by the
  completed provenance rule improves validated utility beyond base and
  garch_high alone.

Output:
  docs/audit/results/2026-04-16-garch-w2c-m2-validated-utility-audit.md
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
from research import garch_partner_state_provenance_audit as prov

OUTPUT_MD = Path("docs/audit/results/2026-04-16-garch-w2c-m2-validated-utility-audit.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

MIN_CONJ = 30
CONCENTRATION_RISK = 0.50


@dataclass(frozen=True)
class FamilyChoice:
    family_label: str
    session: str
    candidate_name: str


FAMILY_CHOICES = [
    FamilyChoice("COMEX_SETTLE_high", "COMEX_SETTLE", "ATRVEL_EXPANDING"),
    FamilyChoice("EUROPE_FLOW_high", "EUROPE_FLOW", "ATRVEL_EXPANDING"),
    FamilyChoice("TOKYO_OPEN_high", "TOKYO_OPEN", "ATRVEL_EXPANDING"),
    FamilyChoice("SINGAPORE_OPEN_high", "SINGAPORE_OPEN", "ATRVEL_GE_110"),
]


def chosen_candidate(name: str) -> prov.Candidate:
    return next(c for c in prov.CANDIDATES if c.name == name)


def family_rows(con: duckdb.DuckDBPyConnection, session: str) -> pd.DataFrame:
    rows = prov.load_rows(con)
    return rows[rows["orb_label"] == session].copy()


def analyze_row(df: pd.DataFrame, df_oos: pd.DataFrame, cand: prov.Candidate) -> dict[str, object]:
    if len(df) < prov.MIN_TOTAL:
        return {"valid": False}
    g = prov.garch_mask(df)
    fav, valid_partner = prov.partner_mask(df, cand)
    conj = g & fav

    g_exp = prov.exp_r(df.loc[g, "pnl_r"]) if g.sum() else float("nan")
    base_exp = prov.exp_r(df["pnl_r"])
    conj_exp = prov.exp_r(df.loc[conj, "pnl_r"]) if conj.sum() else float("nan")

    g_sub = df.loc[g & valid_partner].copy()
    p_sub = df.loc[valid_partner].copy()
    partner_inside_garch = prov.compare_split(g_sub, fav.loc[g_sub.index]) if len(g_sub) >= (2 * prov.MIN_SIDE) else {"valid": False}
    garch_inside_partner = prov.compare_split(p_sub, g.loc[p_sub.index]) if len(p_sub) >= (2 * prov.MIN_SIDE) else {"valid": False}

    oos = {"n_conj_oos": 0, "conj_exp_oos": None}
    if len(df_oos) > 0:
        g_oos = prov.garch_mask(df_oos)
        fav_oos, _ = prov.partner_mask(df_oos, cand)
        conj_oos = g_oos & fav_oos
        oos["n_conj_oos"] = int(conj_oos.sum())
        oos["conj_exp_oos"] = prov.exp_r(df_oos.loc[conj_oos, "pnl_r"]) if conj_oos.sum() >= 3 else None

    return {
        "valid": True,
        "n_total": int(len(df)),
        "n_garch": int(g.sum()),
        "n_conj": int(conj.sum()),
        "base_exp": base_exp,
        "garch_exp": g_exp,
        "conj_exp": conj_exp,
        "delta_garch_vs_base": g_exp - base_exp if not pd.isna(g_exp) else float("nan"),
        "delta_conj_vs_base": conj_exp - base_exp if not pd.isna(conj_exp) else float("nan"),
        "delta_conj_vs_garch": conj_exp - g_exp if not pd.isna(conj_exp) and not pd.isna(g_exp) else float("nan"),
        "valid_partner_inside_garch": bool(partner_inside_garch["valid"]),
        "support_partner_inside_garch": bool(partner_inside_garch.get("support", False)),
        "n_partner_inside_garch": int(partner_inside_garch.get("n_compared", 0)),
        "valid_garch_inside_partner": bool(garch_inside_partner["valid"]),
        "support_garch_inside_partner": bool(garch_inside_partner.get("support", False)),
        "n_garch_inside_partner": int(garch_inside_partner.get("n_compared", 0)),
        **oos,
    }


def to_row(choice: FamilyChoice, row: pd.Series, direction: str, res: dict[str, object]) -> dict[str, object]:
    return {
        "family": choice.family_label,
        "session": choice.session,
        "candidate": choice.candidate_name,
        "instrument": row["instrument"],
        "orb_minutes": row["orb_minutes"],
        "direction": direction,
        "filter_type": row["filter_type"],
        "rr_target": row["rr_target"],
        **res,
    }


def summarize_family(sub: pd.DataFrame) -> dict[str, object]:
    n_total = int(sub["n_total"].sum())
    n_garch = int(sub["n_garch"].sum())
    n_conj = int(sub["n_conj"].sum())
    base_exp = float((sub["base_exp"] * sub["n_total"]).sum() / n_total) if n_total else float("nan")
    garch_exp = float((sub["garch_exp"] * sub["n_garch"]).sum() / n_garch) if n_garch else float("nan")
    conj_exp = float((sub["conj_exp"] * sub["n_conj"]).sum() / n_conj) if n_conj else float("nan")
    max_conj_share = float(sub["n_conj"].max() / n_conj) if n_conj else np.nan
    return {
        "cells": int(len(sub)),
        "n_total": n_total,
        "n_garch": n_garch,
        "n_conj": n_conj,
        "base_exp": base_exp,
        "garch_exp": garch_exp,
        "conj_exp": conj_exp,
        "delta_garch_vs_base": garch_exp - base_exp if n_garch else float("nan"),
        "delta_conj_vs_base": conj_exp - base_exp if n_conj else float("nan"),
        "delta_conj_vs_garch": conj_exp - garch_exp if n_conj and n_garch else float("nan"),
        "support_share_partner_inside_garch": (
            float(
                sub.loc[sub["valid_partner_inside_garch"] & sub["support_partner_inside_garch"], "n_partner_inside_garch"].sum()
                / sub.loc[sub["valid_partner_inside_garch"], "n_partner_inside_garch"].sum()
            )
            if sub.loc[sub["valid_partner_inside_garch"], "n_partner_inside_garch"].sum() > 0
            else np.nan
        ),
        "support_share_garch_inside_partner": (
            float(
                sub.loc[sub["valid_garch_inside_partner"] & sub["support_garch_inside_partner"], "n_garch_inside_partner"].sum()
                / sub.loc[sub["valid_garch_inside_partner"], "n_garch_inside_partner"].sum()
            )
            if sub.loc[sub["valid_garch_inside_partner"], "n_garch_inside_partner"].sum() > 0
            else np.nan
        ),
        "max_conj_cell_share": max_conj_share,
        "n_conj_oos": int(sub["n_conj_oos"].sum()),
        "conj_exp_oos": (
            float(
                (
                    sub.loc[sub["conj_exp_oos"].notna(), "conj_exp_oos"]
                    * sub.loc[sub["conj_exp_oos"].notna(), "n_conj_oos"]
                ).sum()
                / sub.loc[sub["conj_exp_oos"].notna(), "n_conj_oos"].sum()
            )
            if sub.loc[sub["conj_exp_oos"].notna(), "n_conj_oos"].sum() > 0
            else np.nan
        ),
    }


def family_verdict(summary: dict[str, object]) -> tuple[str, str]:
    if summary["n_conj"] < MIN_CONJ:
        return "unclear", "thin_conjunction"
    if pd.isna(summary["delta_conj_vs_garch"]):
        return "unclear", "missing_garch_or_conj"
    concentration = bool(summary["max_conj_cell_share"] > CONCENTRATION_RISK) if not pd.isna(summary["max_conj_cell_share"]) else True
    partner_support = bool(summary["support_share_partner_inside_garch"] >= 0.5) if not pd.isna(summary["support_share_partner_inside_garch"]) else False

    if summary["delta_conj_vs_garch"] > 0 and summary["delta_conj_vs_base"] > 0 and partner_support and not concentration:
        return "carry_local_m2", "validated_local_m2_candidate"
    if summary["delta_conj_vs_base"] > 0 and (partner_support or summary["delta_conj_vs_garch"] >= 0):
        return "partial_local_m2", "useful_but_not_clean"
    return "demote_local_m2", "does_not_beat_garch_cleanly"


def stage_verdict(summaries: list[dict[str, object]]) -> tuple[str, str]:
    verdicts = [s["verdict"] for s in summaries]
    if any(v == "carry_local_m2" for v in verdicts):
        return "M2_carry", "at_least_one_clean_local_family"
    if any(v == "partial_local_m2" for v in verdicts):
        return "M2_partial", "some_local_utility_but_not_clean_carry"
    return "M2_demote", "no_family_beats_garch_cleanly"


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    result_rows: list[dict[str, object]] = []
    for choice in FAMILY_CHOICES:
        cand = chosen_candidate(choice.candidate_name)
        rows = family_rows(con, choice.session)
        for _, row in rows.iterrows():
            for direction in ["long", "short"]:
                df = prov.load_trades(con, row, direction, is_oos=False)
                if len(df) < prov.MIN_TOTAL:
                    continue
                df_oos = prov.load_trades(con, row, direction, is_oos=True)
                res = analyze_row(df, df_oos, cand)
                if not res["valid"]:
                    continue
                result_rows.append(to_row(choice, row, direction, res))
    con.close()

    df = pd.DataFrame(result_rows)
    summaries = []
    if len(df) > 0:
        for choice in FAMILY_CHOICES:
            sub = df[df["family"] == choice.family_label].copy()
            if len(sub) == 0:
                continue
            s = summarize_family(sub)
            verdict, implication = family_verdict(s)
            s["family"] = choice.family_label
            s["session"] = choice.session
            s["candidate"] = choice.candidate_name
            s["verdict"] = verdict
            s["implication"] = implication
            summaries.append(s)
    return df, pd.DataFrame(summaries)


def emit(df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    lines = [
        "# Garch W2c M2 Validated Utility Audit",
        "",
        "**Date:** 2026-04-16",
        "**Boundary:** validated shelf only, conservative family-local M2 representation, no deployment conclusions.",
        "",
    ]
    if len(summary_df) == 0:
        lines.append("No carried families met minimum support.")
        OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
        return

    stage_v, stage_i = stage_verdict(summary_df.to_dict("records"))
    lines.extend(
        [
            f"- Stage verdict: **{stage_v}**",
            f"- Allowed implication: **{stage_i}**",
            "",
            "| Family | Representation | Cells | N total | N garch | N conj | Base ExpR | Garch ExpR | Conj ExpR | Δ conj-base | Δ conj-garch | P|G support | G|P support | Max conj share | OOS conj N | OOS conj ExpR | Verdict |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in summary_df.sort_values("family").iterrows():
        lines.append(
            f"| {row['family']} | {row['candidate']} | {int(row['cells'])} | {int(row['n_total'])} | {int(row['n_garch'])} | {int(row['n_conj'])} | "
            f"{row['base_exp']:+.3f} | {row['garch_exp']:+.3f} | {row['conj_exp']:+.3f} | "
            f"{row['delta_conj_vs_base']:+.3f} | {row['delta_conj_vs_garch']:+.3f} | "
            f"{'' if pd.isna(row['support_share_partner_inside_garch']) else f'{row['support_share_partner_inside_garch']:.2f}'} | "
            f"{'' if pd.isna(row['support_share_garch_inside_partner']) else f'{row['support_share_garch_inside_partner']:.2f}'} | "
            f"{'' if pd.isna(row['max_conj_cell_share']) else f'{row['max_conj_cell_share']:.2f}'} | "
            f"{int(row['n_conj_oos'])} | "
            f"{'' if pd.isna(row['conj_exp_oos']) else f'{row['conj_exp_oos']:+.3f}'} | "
            f"{row['verdict']} |"
        )

    lines.extend(["", "## Per-cell detail", ""])
    for family in summary_df["family"]:
        sub = df[df["family"] == family].copy()
        if len(sub) == 0:
            continue
        lines.extend(
            [
                f"### {family}",
                "",
                "| Instrument | ORB | Dir | Filter | RR | N | N garch | N conj | Base ExpR | Garch ExpR | Conj ExpR | Δ conj-garch |",
                "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in sub.sort_values(["instrument", "orb_minutes", "direction", "filter_type", "rr_target"]).iterrows():
            lines.append(
                f"| {row['instrument']} | {int(row['orb_minutes'])} | {row['direction']} | {row['filter_type']} | {row['rr_target']:.1f} | {int(row['n_total'])} | {int(row['n_garch'])} | {int(row['n_conj'])} | "
                f"{row['base_exp']:+.3f} | {row['garch_exp']:+.3f} | {row['conj_exp']:+.3f} | "
                f"{row['delta_conj_vs_garch']:+.3f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Guardrails",
            "",
            "- Family representation choice was frozen from the completed provenance rule; no new representation search happened in this stage.",
            "- `neighbor_stable` families kept the current canonical representation; only explicit `alternate_better` families switched.",
            "- 2026 OOS remains descriptive only.",
            "- This is still validated utility only, not deployment doctrine.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df, summary_df = build()
    emit(df, summary_df)
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
