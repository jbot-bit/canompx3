"""Validated-shelf prior-session carry conditioning audit for garch.

Purpose:
  Test whether same-day, fully-resolved LONDON_METALS state adds distinct or
  complementary value to garch_high on validated EUROPE_FLOW families.

Output:
  docs/audit/results/2026-04-16-garch-w2e-prior-session-carry-audit.md
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

from pipeline.dst import orb_utc_window
from pipeline.paths import GOLD_DB_PATH
from research import garch_broad_exact_role_exhaustion as broad
from research import garch_partner_state_provenance_audit as prov

OUTPUT_MD = Path("docs/audit/results/2026-04-16-garch-w2e-prior-session-carry-audit.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

MIN_TOTAL = 50
MIN_CONJ = 30
GARCH_HIGH = 70.0
PRIOR_SESSION = "LONDON_METALS"
TARGET_SESSION = "EUROPE_FLOW"


@dataclass(frozen=True)
class CarryState:
    name: str
    label: str
    expected_role: str  # take_pair / veto_pair


STATES = [
    CarryState("PRIOR_WIN_OPPOSED", "prior win opposed", "veto_pair"),
    CarryState("PRIOR_WIN_ALIGN", "prior win align", "take_pair"),
]


def exp_r(arr: pd.Series) -> float:
    return float(pd.Series(arr).astype(float).mean())


def sharpe_like(arr: pd.Series) -> float:
    arr = pd.Series(arr).astype(float)
    if len(arr) < 2:
        return 0.0
    sd = arr.std(ddof=1)
    return float(arr.mean() / sd) if sd and not pd.isna(sd) else 0.0


def load_validated_rows(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    rows = prov.load_rows(con)
    rows = rows[(rows["orb_label"] == TARGET_SESSION) & (rows["instrument"] == "MNQ")].copy()
    return rows.reset_index(drop=True)


def load_prior_state(con: duckdb.DuckDBPyConnection, symbol: str) -> pd.DataFrame:
    q = f"""
    SELECT
      o.trading_day,
      o.symbol,
      o.outcome,
      o.exit_ts,
      d.orb_{PRIOR_SESSION}_break_dir AS prior_dir
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    WHERE o.orb_label = '{PRIOR_SESSION}'
      AND o.symbol = '{symbol}'
      AND o.orb_minutes = 5
      AND o.entry_model = 'E2'
      AND o.confirm_bars = 1
      AND o.rr_target = 1.0
      AND o.outcome IS NOT NULL
      AND o.exit_ts IS NOT NULL
      AND d.orb_{PRIOR_SESSION}_break_dir IS NOT NULL
    ORDER BY o.trading_day
    """
    df = con.execute(q).df()
    if len(df) == 0:
        return df
    df["trading_day"] = pd.to_datetime(df["trading_day"]).dt.date
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True)
    return df


def load_target_trades(con: duckdb.DuckDBPyConnection, row: pd.Series) -> pd.DataFrame:
    filter_sql, join_sql = broad.exact_filter_sql(row["filter_type"], row["orb_label"], row["instrument"])
    q = f"""
    SELECT
      o.trading_day,
      o.pnl_r,
      d.garch_forecast_vol_pct AS gp,
      d.orb_{TARGET_SESSION}_break_dir AS target_dir
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    {join_sql}
    WHERE o.symbol = '{row["instrument"]}'
      AND o.orb_label = '{TARGET_SESSION}'
      AND o.orb_minutes = {row["orb_minutes"]}
      AND o.entry_model = '{row["entry_model"]}'
      AND o.rr_target = {row["rr_target"]}
      AND o.pnl_r IS NOT NULL
      AND d.garch_forecast_vol_pct IS NOT NULL
      AND d.orb_{TARGET_SESSION}_break_dir IS NOT NULL
      AND {filter_sql}
    ORDER BY o.trading_day
    """
    df = con.execute(q).df()
    if len(df) == 0:
        return df
    df["trading_day"] = pd.to_datetime(df["trading_day"]).dt.date
    df["target_start_ts"] = [
        orb_utc_window(day, TARGET_SESSION, int(row["orb_minutes"]))[0] for day in df["trading_day"]
    ]
    return df


def classify_state(df: pd.DataFrame, state: CarryState) -> pd.Series:
    resolved = df["resolved_before_start"]
    prior_win = df["prior_outcome"].eq("win")
    align = df["prior_dir"].eq(df["target_dir"])
    if state.name == "PRIOR_WIN_ALIGN":
        return resolved & prior_win & align
    if state.name == "PRIOR_WIN_OPPOSED":
        return resolved & prior_win & (~align)
    raise ValueError(state.name)


def analyze_row(df: pd.DataFrame, state: CarryState) -> dict[str, object]:
    if len(df) < MIN_TOTAL:
        return {"valid": False}
    g = df["gp"] >= GARCH_HIGH
    carry = classify_state(df, state)
    conj = g & carry

    n_garch = int(g.sum())
    n_carry = int(carry.sum())
    n_conj = int(conj.sum())
    if n_conj == 0 or n_carry == 0 or n_garch == 0:
        return {
            "valid": True,
            "n_total": int(len(df)),
            "n_garch": n_garch,
            "n_carry": n_carry,
            "n_conj": n_conj,
            "base_exp": exp_r(df["pnl_r"]),
            "garch_exp": exp_r(df.loc[g, "pnl_r"]) if n_garch else float("nan"),
            "carry_exp": exp_r(df.loc[carry, "pnl_r"]) if n_carry else float("nan"),
            "conj_exp": float("nan"),
            "resolved_rows": int(df["resolved_before_start"].sum()),
            "valid": True,
        }

    return {
        "valid": True,
        "n_total": int(len(df)),
        "n_garch": n_garch,
        "n_carry": n_carry,
        "n_conj": n_conj,
        "resolved_rows": int(df["resolved_before_start"].sum()),
        "base_exp": exp_r(df["pnl_r"]),
        "garch_exp": exp_r(df.loc[g, "pnl_r"]),
        "carry_exp": exp_r(df.loc[carry, "pnl_r"]),
        "conj_exp": exp_r(df.loc[conj, "pnl_r"]),
        "delta_conj_vs_base": exp_r(df.loc[conj, "pnl_r"]) - exp_r(df["pnl_r"]),
        "delta_conj_vs_garch": exp_r(df.loc[conj, "pnl_r"]) - exp_r(df.loc[g, "pnl_r"]),
        "delta_conj_vs_carry": exp_r(df.loc[conj, "pnl_r"]) - exp_r(df.loc[carry, "pnl_r"]),
        "sr_conj": sharpe_like(df.loc[conj, "pnl_r"]),
    }


def to_row(row: pd.Series, state: CarryState, res: dict[str, object]) -> dict[str, object]:
    return {
        "strategy_id": row["strategy_id"],
        "instrument": row["instrument"],
        "orb_minutes": row["orb_minutes"],
        "rr_target": row["rr_target"],
        "filter_type": row["filter_type"],
        "state": state.name,
        "state_label": state.label,
        "expected_role": state.expected_role,
        **res,
    }


def summarize(sub: pd.DataFrame, state: CarryState) -> dict[str, object]:
    n_total = int(sub["n_total"].sum())
    n_garch = int(sub["n_garch"].sum())
    n_carry = int(sub["n_carry"].sum())
    n_conj = int(sub["n_conj"].sum())
    base_exp = float((sub["base_exp"] * sub["n_total"]).sum() / n_total) if n_total else float("nan")
    garch_exp = float((sub["garch_exp"] * sub["n_garch"]).sum() / n_garch) if n_garch else float("nan")
    carry_exp = float((sub["carry_exp"] * sub["n_carry"]).sum() / n_carry) if n_carry else float("nan")
    conj_exp = float((sub["conj_exp"] * sub["n_conj"]).sum() / n_conj) if n_conj else float("nan")

    verdict = "unclear"
    if n_conj < MIN_CONJ:
        verdict = "not_testable_here"
    elif state.expected_role == "take_pair":
        if conj_exp > base_exp and conj_exp > garch_exp and conj_exp > carry_exp:
            verdict = "carry_take_pair"
        elif garch_exp >= conj_exp and garch_exp >= carry_exp:
            verdict = "garch_only"
        elif carry_exp >= conj_exp and carry_exp >= garch_exp:
            verdict = "carry_only"
    elif state.expected_role == "veto_pair":
        if conj_exp < base_exp and conj_exp < garch_exp and conj_exp < carry_exp:
            verdict = "carry_veto_pair"
        elif garch_exp <= conj_exp and garch_exp <= carry_exp:
            verdict = "garch_only"
        elif carry_exp <= conj_exp and carry_exp <= garch_exp:
            verdict = "carry_only"

    return {
        "state": state.name,
        "state_label": state.label,
        "expected_role": state.expected_role,
        "cells": int(len(sub)),
        "n_total": n_total,
        "resolved_rows": int(sub["resolved_rows"].sum()),
        "n_garch": n_garch,
        "n_carry": n_carry,
        "n_conj": n_conj,
        "base_exp": base_exp,
        "garch_exp": garch_exp,
        "carry_exp": carry_exp,
        "conj_exp": conj_exp,
        "delta_conj_vs_base": conj_exp - base_exp if n_conj else float("nan"),
        "delta_conj_vs_garch": conj_exp - garch_exp if n_conj and n_garch else float("nan"),
        "delta_conj_vs_carry": conj_exp - carry_exp if n_conj and n_carry else float("nan"),
        "verdict": verdict,
    }


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    rows = load_validated_rows(con)
    if len(rows) == 0:
        con.close()
        return pd.DataFrame(), pd.DataFrame()

    prior = load_prior_state(con, "MNQ")
    results: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        target = load_target_trades(con, row)
        if len(target) == 0:
            continue
        merged = target.merge(prior, on=["trading_day", "symbol"], how="left")
        merged["resolved_before_start"] = merged["exit_ts"].notna() & (
            merged["exit_ts"] < pd.to_datetime(merged["target_start_ts"], utc=True)
        )
        merged = merged.rename(columns={"outcome": "prior_outcome", "lm_dir": "prior_dir"})
        for state in STATES:
            res = analyze_row(merged, state)
            if res.get("valid"):
                results.append(to_row(row, state, res))
    con.close()

    df = pd.DataFrame(results)
    summaries = []
    if len(df) > 0:
        for state in STATES:
            sub = df[df["state"] == state.name].copy()
            if len(sub) == 0:
                continue
            summaries.append(summarize(sub, state))
    return df, pd.DataFrame(summaries)


def emit(df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    lines = [
        "# Garch W2e Prior-Session Carry Audit",
        "",
        "**Date:** 2026-04-16",
        "**Boundary:** validated EUROPE_FLOW shelf only, prior LONDON_METALS trade must resolve before target session start, no deployment conclusions.",
        "",
    ]
    if len(summary_df) == 0:
        lines.append("No testable validated rows were found.")
        OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.extend(
        [
            "| State | Expected role | Cells | N total | Resolved rows | N garch | N carry | N conj | Base ExpR | Garch ExpR | Carry ExpR | Conj ExpR | Δ conj-base | Δ conj-garch | Δ conj-carry | Verdict |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in summary_df.sort_values("state").iterrows():
        lines.append(
            f"| {row['state_label']} | {row['expected_role']} | {int(row['cells'])} | {int(row['n_total'])} | {int(row['resolved_rows'])} | "
            f"{int(row['n_garch'])} | {int(row['n_carry'])} | {int(row['n_conj'])} | "
            f"{row['base_exp']:+.3f} | {row['garch_exp']:+.3f} | {row['carry_exp']:+.3f} | {row['conj_exp']:+.3f} | "
            f"{row['delta_conj_vs_base']:+.3f} | {row['delta_conj_vs_garch']:+.3f} | {row['delta_conj_vs_carry']:+.3f} | {row['verdict']} |"
        )

    lines.extend(["", "## Per-row detail", ""])
    for state in summary_df["state"]:
        sub = df[df["state"] == state].copy()
        if len(sub) == 0:
            continue
        lines.extend(
            [
                f"### {sub['state_label'].iloc[0]}",
                "",
                "| Strategy | ORB | Filter | RR | N | Resolved | N garch | N carry | N conj | Base ExpR | Garch ExpR | Carry ExpR | Conj ExpR |",
                "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in sub.sort_values(["orb_minutes", "filter_type", "rr_target"]).iterrows():
            conj_exp = row["conj_exp"]
            lines.append(
                f"| {row['strategy_id']} | {int(row['orb_minutes'])} | {row['filter_type']} | {row['rr_target']:.1f} | "
                f"{int(row['n_total'])} | {int(row['resolved_rows'])} | {int(row['n_garch'])} | {int(row['n_carry'])} | {int(row['n_conj'])} | "
                f"{row['base_exp']:+.3f} | {row['garch_exp']:+.3f} | {row['carry_exp']:+.3f} | "
                f"{'' if pd.isna(conj_exp) else f'{conj_exp:+.3f}'} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Guardrails",
            "",
            "- Prior-session state is only admissible when the prior trade has non-null `exit_ts` and `exit_ts < target_session_start_ts` on that trading day.",
            "- Static session ordering is not used for admissibility.",
            "- Prior session state comes from canonical `orb_outcomes` + `daily_features`, not from ML cross-session features.",
            "- This is validated utility only, not deployment doctrine.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df, summary_df = build()
    emit(df, summary_df)
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
