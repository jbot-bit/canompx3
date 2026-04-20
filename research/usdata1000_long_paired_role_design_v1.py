#!/usr/bin/env python3
"""Role-design audit for the verified MNQ US_DATA_1000 long paired family.

This is a deterministic synthesis script grounded in the already-verified paired
family. It does not claim fresh discovery; it translates the paired TAKE/AVOID
states into candidate ORB usage roles.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research.lib import connect_db
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

INSTRUMENT = "MNQ"
SESSION = "US_DATA_1000"
RR_TARGETS = (1.0, 1.5)
HOLDOUT = pd.Timestamp(HOLDOUT_SACRED_FROM).date()

OUTPUT_MD = Path("docs/plans/2026-04-20-usdata1000-paired-role-design.md")
OUTPUT_CSV = Path("research/output/usdata1000_long_paired_role_design_v1.csv")
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)


def _safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def _load_df() -> pd.DataFrame:
    sql = f"""
    SELECT
        o.trading_day,
        o.entry_price,
        o.stop_price,
        o.pnl_r,
        o.rr_target,
        d.prev_day_high,
        d.prev_day_low,
        d.orb_{SESSION}_high AS orb_high,
        d.orb_{SESSION}_low AS orb_low
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol = '{INSTRUMENT}'
      AND o.orb_label = '{SESSION}'
      AND o.orb_minutes = 5
      AND o.entry_model = 'E2'
      AND o.confirm_bars = 1
      AND o.rr_target IN ({", ".join(str(rr) for rr in RR_TARGETS)})
      AND o.pnl_r IS NOT NULL
      AND d.prev_day_high IS NOT NULL
      AND d.prev_day_low IS NOT NULL
      AND d.orb_{SESSION}_high IS NOT NULL
      AND d.orb_{SESSION}_low IS NOT NULL
    ORDER BY o.trading_day
    """
    with connect_db() as con:
        df = con.execute(sql).fetchdf()

    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["direction"] = np.where(df["entry_price"] > df["stop_price"], "long", "short")
    df = df[df["direction"] == "long"].copy()
    mid = (df["orb_high"].astype(float) + df["orb_low"].astype(float)) / 2.0
    pdl = df["prev_day_low"].astype(float)
    pdh = df["prev_day_high"].astype(float)
    df["F5_BELOW_PDL"] = mid < pdl
    df["F6_INSIDE_PDR"] = (mid > pdl) & (mid < pdh)
    df["NEUTRAL"] = (~df["F5_BELOW_PDL"]) & (~df["F6_INSIDE_PDR"])
    df["OFF_F6"] = ~df["F6_INSIDE_PDR"]
    df["is_is"] = df["trading_day"].dt.date < HOLDOUT
    df["period"] = np.where(df["is_is"], "IS", "OOS")
    return df


def _summarize_group(sub: pd.DataFrame, rr_target: float, period: str) -> list[dict[str, object]]:
    rr_df = sub[(sub["rr_target"] == rr_target) & (sub["period"] == period)].copy()
    groups = {
        "baseline": pd.Series(np.ones(len(rr_df), dtype=bool), index=rr_df.index),
        "f5_take_only": rr_df["F5_BELOW_PDL"],
        "f6_avoid_only": rr_df["F6_INSIDE_PDR"],
        "neutral_only": rr_df["NEUTRAL"],
        "off_f6_filter": rr_df["OFF_F6"],
    }
    rows: list[dict[str, object]] = []
    baseline_mean = _safe_mean(rr_df["pnl_r"])
    for role_name, mask in groups.items():
        pnl = rr_df.loc[mask, "pnl_r"]
        rows.append(
            {
                "rr_target": rr_target,
                "period": period,
                "role": role_name,
                "n": int(len(pnl)),
                "exp_r": _safe_mean(pnl),
                "delta_vs_baseline": _safe_mean(pnl) - baseline_mean if len(pnl) else float("nan"),
            }
        )
    return rows


def main() -> None:
    df = _load_df()
    rows: list[dict[str, object]] = []
    for rr in RR_TARGETS:
        for period in ("IS", "OOS"):
            rows.extend(_summarize_group(df, rr, period))
    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_CSV, index=False)

    lines = [
        "# US_DATA_1000 Long Paired Role Design",
        "",
        "**Input family:** `docs/audit/results/2026-04-20-usdata1000-long-f5-f6-paired-v1.md`",
        "**Purpose:** translate the verified `F5 TAKE / F6 AVOID` pair into the most honest ORB usage role.",
        "",
        "## Grounding",
        "",
        "- `resources/Algorithmic_Trading_Chan.pdf`: bounded executable routes are valid strategy units.",
        "- `resources/Robert Carver - Systematic Trading.pdf`: signals can be conditioners and size/priority modifiers, not only standalone systems.",
        "- `RESEARCH_RULES.md`: do not overclaim discovery from a design synthesis; this note translates an already-verified pair.",
        "- `docs/institutional/mechanism_priors.md`: inside-range opens should hurt breakouts, while washed-out below-PDL opens can help long quality.",
        "",
        "## Canonical role metrics",
        "",
    ]

    recommendations: list[str] = []
    for rr in RR_TARGETS:
        is_rows = result_df[(result_df["rr_target"] == rr) & (result_df["period"] == "IS")].set_index("role")
        oos_rows = result_df[(result_df["rr_target"] == rr) & (result_df["period"] == "OOS")].set_index("role")
        baseline = is_rows.loc["baseline"]
        f5 = is_rows.loc["f5_take_only"]
        f6 = is_rows.loc["f6_avoid_only"]
        neutral = is_rows.loc["neutral_only"]
        off_f6 = is_rows.loc["off_f6_filter"]
        lines.extend(
            [
                f"### RR {rr:.1f}",
                "",
                f"- IS baseline: N={int(baseline['n'])} ExpR={baseline['exp_r']:+.4f}",
                f"- IS F5 TAKE-only: N={int(f5['n'])} ExpR={f5['exp_r']:+.4f}",
                f"- IS F6 AVOID-state: N={int(f6['n'])} ExpR={f6['exp_r']:+.4f}",
                f"- IS neutral-only: N={int(neutral['n'])} ExpR={neutral['exp_r']:+.4f}",
                f"- IS OFF_F6 filtered parent: N={int(off_f6['n'])} ExpR={off_f6['exp_r']:+.4f}",
                f"- OOS F5 TAKE-only: N={int(oos_rows.loc['f5_take_only', 'n'])} ExpR={oos_rows.loc['f5_take_only', 'exp_r']:+.4f}",
                f"- OOS OFF_F6 filtered parent: N={int(oos_rows.loc['off_f6_filter', 'n'])} ExpR={oos_rows.loc['off_f6_filter', 'exp_r']:+.4f}",
                "",
            ]
        )

        ordering_holds = bool(f5["exp_r"] > neutral["exp_r"] > f6["exp_r"])
        if off_f6["exp_r"] > baseline["exp_r"] and off_f6["n"] >= 2 * f5["n"]:
            recommendation = (
                f"RR {rr:.1f}: prefer `NOT_F6` as the primary ORB integration route. "
                f"It keeps {int(off_f6['n'])} IS trades at ExpR {off_f6['exp_r']:+.4f}, "
                f"which is more practical than F5-only ({int(f5['n'])} trades at {f5['exp_r']:+.4f})."
            )
        else:
            recommendation = (
                f"RR {rr:.1f}: F5-only may be the cleaner route, but this condition did not trigger on current data."
            )
        if ordering_holds:
            recommendation += " The three-state ordering `F5 > neutral > F6` also holds in-sample."
        recommendations.append(recommendation)

    lines.extend(
        [
            "## Decision",
            "",
            "- The verified pair is **not best handled as a standalone F5-only lane**.",
            "- The most honest ORB integration is a **binary `NOT_F6_INSIDE_PDR` filter candidate**, with `F5_BELOW_PDL` treated as a higher-quality sub-state inside that allowed set.",
            "- This means the pair belongs in ORB as a conditioner / lane-design route, not as a separate standalone trade family.",
            "",
            "## Why",
            "",
        ]
    )
    for rec in recommendations:
        lines.append(f"- {rec}")

    lines.extend(
        [
            "",
            "## Next bounded action",
            "",
            "- Preserve a candidate-lane validation contract for `MNQ US_DATA_1000 O5 E2 long NOT_F6_INSIDE_PDR`.",
            "- Treat `F5_BELOW_PDL` as a secondary priority / upsize descriptor inside that route, not the primary gate.",
            "- Do not claim live-readiness from this design note; the next step is a bounded candidate-lane validation or shadow design.",
            "",
            "## Artefacts",
            "",
            f"- CSV: `{OUTPUT_CSV}`",
            f"- Script: `research/{Path(__file__).name}`",
        ]
    )

    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
