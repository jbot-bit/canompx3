#!/usr/bin/env python3
"""Native MGC low-R target-shape test.

Locked by:
  docs/audit/hypotheses/2026-04-19-mgc-native-low-r-v1.yaml
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.lib import bh_fdr, ttest_1s, write_csv
from research.research_mgc_payoff_compression_audit import (
    FAMILIES,
    build_family_trade_matrix,
    load_rows,
)

HOLDOUT_START = pd.Timestamp("2026-01-01")
RESULT_PATH = Path("docs/audit/results/2026-04-19-mgc-native-low-r-v1.md")
OUTPUT_PREFIX = "mgc_native_low_r_v1"
MIN_N = 50
BH_Q = 0.10

VARIANT_COLS = {
    "LR05": "lower_0_5_pnl_r",
    "LR075": "lower_0_75_pnl_r",
}


def evaluate_variants(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for family in FAMILIES:
        family_df = trades[trades["family_id"] == family.family_id].copy()
        if family_df.empty:
            continue
        family_df["trading_day"] = pd.to_datetime(family_df["trading_day"])
        is_df = family_df[family_df["trading_day"] < HOLDOUT_START]
        oos_df = family_df[family_df["trading_day"] >= HOLDOUT_START]

        for variant_id, value_col in VARIANT_COLS.items():
            n_is, mean_is, wr_is, t_is, p_is = ttest_1s(is_df[value_col].values)
            n_oos, mean_oos, wr_oos, t_oos, p_oos = ttest_1s(oos_df[value_col].values)
            rows.append(
                {
                    "family_id": family.family_id,
                    "family_kind": family.kind,
                    "orb_label": family.orb_label,
                    "filter_type": family.filter_type or "NO_FILTER",
                    "variant_id": variant_id,
                    "value_col": value_col,
                    "n_is": n_is,
                    "avg_is": mean_is,
                    "wr_is": wr_is,
                    "t_is": t_is,
                    "p_is": p_is,
                    "n_oos": n_oos,
                    "avg_oos": mean_oos,
                    "wr_oos": wr_oos,
                    "t_oos": t_oos,
                    "p_oos": p_oos,
                }
            )
    result = pd.DataFrame(rows)
    rejected = bh_fdr(result["p_is"].fillna(1.0).tolist(), q=BH_Q)
    result["bh_survive"] = [idx in rejected for idx in range(len(result))]
    result["primary_survivor"] = (
        result["bh_survive"]
        & (result["n_is"] >= MIN_N)
        & (result["avg_is"] > 0)
    )
    return result.sort_values(
        ["primary_survivor", "avg_is", "family_kind", "orb_label", "variant_id"],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)


def build_markdown(result: pd.DataFrame) -> str:
    survivors = result[result["primary_survivor"]].copy()
    lines = [
        "# MGC Native Low-R v1",
        "",
        "Date: 2026-04-19",
        "",
        "## Scope",
        "",
        "Native follow-through on the prior MGC payoff-compression diagnostics.",
        "This pass treats conservative low-R exits as candidate native MGC",
        "families and tests them under honest global BH across the full locked matrix.",
        "",
        "Locked matrix:",
        "",
        "- 8 family rows (`3` broad + `5` warm/filtered)",
        "- 2 target variants each (`0.5R`, `0.75R`)",
        "- total K = 16 with global BH at q=0.10",
        "- pre-2026 for selection; 2026 held back as diagnostic OOS only",
        "",
        "## Executive Verdict",
        "",
    ]

    if survivors.empty:
        lines += [
            "No low-R native MGC families survive the locked 16-trial matrix under",
            "global BH + N>=50 + positive pre-2026 expectancy.",
            "",
            "That means the lower-R effect remains diagnostic, not validated. It is not",
            "strong enough yet to promote into a native MGC edge claim.",
            "",
        ]
    else:
        n_lr05 = int((survivors["variant_id"] == "LR05").sum())
        n_lr075 = int((survivors["variant_id"] == "LR075").sum())
        n_broad = int((survivors["family_kind"] == "broad").sum())
        n_warm = int((survivors["family_kind"] == "warm").sum())
        lines += [
            f"{len(survivors)} families survive the locked matrix.",
            "",
            f"- survivor split by target: `0.5R={n_lr05}`, `0.75R={n_lr075}`",
            f"- survivor split by family kind: `broad={n_broad}`, `warm={n_warm}`",
            "",
            "Interpretation:",
            "",
            "- if broad rows survive too, the unresolved issue is broader native MGC",
            "  target shape, not just translated warm rows",
            "- if only 0.5R survives, the compression problem is still tight enough that",
            "  even modestly higher targets lose the edge",
            "- 2026 remains diagnostic only and must not rescue or kill survivors by itself",
            "",
        ]

    if int(result["n_oos"].sum()) == 0:
        lines += [
            "## Holdout Status",
            "",
            "No 2026 rows were available on the locked surface after rebuilding the trade matrix.",
            "So this result is still **IS-only** in practical terms even though the holdout split",
            "logic is implemented. Do not treat any survivor here as fully validated.",
            "",
        ]

    lines += [
        "## Full Matrix",
        "",
        "| Family | Kind | Variant | N IS | Avg IS | p IS | BH | Primary | N OOS | Avg OOS |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result.itertuples(index=False):
        p_is = f"{row.p_is:.4f}" if pd.notna(row.p_is) else "NA"
        lines.append(
            f"| {row.family_id} | {row.family_kind} | {row.variant_id} | "
            f"{int(row.n_is)} | {row.avg_is:+.4f} | {p_is} | "
            f"{'Y' if row.bh_survive else 'N'} | {'Y' if row.primary_survivor else 'N'} | "
            f"{int(row.n_oos)} | {row.avg_oos:+.4f} |"
        )

    lines += [
        "",
        "## Guardrails",
        "",
        "- These exits are still diagnostic rewrites, not live-ready execution rules.",
        "- Ambiguous losses remain fail-closed.",
        "- This pass does not revive the retired GC shelf or reopen proxy discovery.",
        "",
        "## Outputs",
        "",
        "- `research/output/mgc_native_low_r_v1_matrix.csv`",
    ]
    return "\n".join(lines).replace("nan", "NA") + "\n"


def main() -> None:
    rows = load_rows(end_exclusive=None)
    trades = build_family_trade_matrix(rows)
    result = evaluate_variants(trades)
    write_csv(result, f"{OUTPUT_PREFIX}_matrix.csv")
    RESULT_PATH.write_text(build_markdown(result), encoding="utf-8")
    print(result.to_string(index=False))
    print(f"\nWrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
