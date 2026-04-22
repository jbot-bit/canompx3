"""Bounded MNQ layered candidate board on positive parent lanes.

Canonical truth only:
- daily_features
- orb_outcomes

This is an exploratory candidate board, not a promotion runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from pipeline.paths import GOLD_DB_PATH
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

MIN_IS_TRADES = 30
OOS_START = pd.Timestamp(HOLDOUT_SACRED_FROM)

OUTPUT_DIR = Path("docs/audit/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = OUTPUT_DIR / "2026-04-22-mnq-layered-candidate-board-v1.csv"
OUTPUT_MD = OUTPUT_DIR / "2026-04-22-mnq-layered-candidate-board-v1.md"


@dataclass(frozen=True)
class ParentLane:
    orb_label: str
    rr_target: float
    direction: str


PARENTS: tuple[ParentLane, ...] = (
    ParentLane("CME_PRECLOSE", 1.0, "long"),
    ParentLane("CME_PRECLOSE", 1.0, "short"),
    ParentLane("NYSE_OPEN", 1.5, "long"),
    ParentLane("NYSE_OPEN", 1.5, "short"),
    ParentLane("US_DATA_1000", 1.0, "long"),
    ParentLane("US_DATA_1000", 1.5, "short"),
)

BASE_FEATURES: tuple[str, ...] = (
    "F1_NEAR_PDH_15",
    "F2_NEAR_PDL_15",
    "F3_NEAR_PIVOT_15",
    "F3_NEAR_PIVOT_50",
    "F4_ABOVE_PDH",
    "F5_BELOW_PDL",
    "F6_INSIDE_PDR",
    "F7_GAP_UP",
    "F8_GAP_DOWN",
)

EXCLUDED_PAIRS: set[tuple[str, str]] = {
    tuple(sorted(("F4_ABOVE_PDH", "F5_BELOW_PDL"))),
    tuple(sorted(("F4_ABOVE_PDH", "F6_INSIDE_PDR"))),
    tuple(sorted(("F5_BELOW_PDL", "F6_INSIDE_PDR"))),
    tuple(sorted(("F7_GAP_UP", "F8_GAP_DOWN"))),
    tuple(sorted(("F3_NEAR_PIVOT_15", "F3_NEAR_PIVOT_50"))),
}


def signal_specs() -> list[tuple[str, tuple[str, ...]]]:
    specs: list[tuple[str, tuple[str, ...]]] = [(name, (name,)) for name in BASE_FEATURES]
    for left, right in combinations(BASE_FEATURES, 2):
        pair = tuple(sorted((left, right)))
        if pair in EXCLUDED_PAIRS:
            continue
        specs.append((f"{left}__AND__{right}", (left, right)))
    return specs


def load_parent_lane(parent: ParentLane) -> pd.DataFrame:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    session = parent.orb_label
    query = f"""
    SELECT
        o.trading_day,
        o.pnl_r,
        o.outcome,
        d.atr_20,
        d.prev_day_high,
        d.prev_day_low,
        d.prev_day_close,
        d.gap_type,
        (d.orb_{session}_high + d.orb_{session}_low) / 2.0 AS orb_mid,
        d.orb_{session}_break_dir AS break_dir
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol = 'MNQ'
      AND o.orb_minutes = 5
      AND o.entry_model = 'E2'
      AND o.confirm_bars = 1
      AND o.rr_target = {parent.rr_target}
      AND o.orb_label = '{session}'
      AND o.pnl_r IS NOT NULL
      AND d.atr_20 IS NOT NULL
      AND d.atr_20 > 0
      AND d.prev_day_high IS NOT NULL
      AND d.prev_day_low IS NOT NULL
      AND d.prev_day_close IS NOT NULL
      AND d.orb_{session}_break_dir = '{parent.direction}'
    """
    df = con.execute(query).df()
    con.close()
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pdh = out["prev_day_high"].astype(float)
    pdl = out["prev_day_low"].astype(float)
    pdc = out["prev_day_close"].astype(float)
    atr = out["atr_20"].astype(float)
    mid = out["orb_mid"].astype(float)
    pivot = (pdh + pdl + pdc) / 3.0

    out["F1_NEAR_PDH_15"] = (np.abs(mid - pdh) / atr < 0.15).astype(int)
    out["F2_NEAR_PDL_15"] = (np.abs(mid - pdl) / atr < 0.15).astype(int)
    out["F3_NEAR_PIVOT_15"] = (np.abs(mid - pivot) / atr < 0.15).astype(int)
    out["F3_NEAR_PIVOT_50"] = (np.abs(mid - pivot) / atr < 0.50).astype(int)
    out["F4_ABOVE_PDH"] = (mid > pdh).astype(int)
    out["F5_BELOW_PDL"] = (mid < pdl).astype(int)
    out["F6_INSIDE_PDR"] = ((mid > pdl) & (mid < pdh)).astype(int)
    out["F7_GAP_UP"] = (out["gap_type"] == "gap_up").astype(int)
    out["F8_GAP_DOWN"] = (out["gap_type"] == "gap_down").astype(int)
    return out


def evaluate_signal(
    df: pd.DataFrame,
    parent: ParentLane,
    signal_name: str,
    inputs: tuple[str, ...],
) -> dict[str, object] | None:
    mask = pd.Series(True, index=df.index)
    for column in inputs:
        mask &= df[column].astype(bool)

    on = df[mask]
    off = df[~mask]
    is_on = on[on["trading_day"] < OOS_START]
    is_off = off[off["trading_day"] < OOS_START]
    if len(is_on) < MIN_IS_TRADES or len(is_off) < MIN_IS_TRADES:
        return None

    oos_on = on[on["trading_day"] >= OOS_START]
    oos_off = off[off["trading_day"] >= OOS_START]

    expr_on_is = float(is_on["pnl_r"].mean())
    expr_off_is = float(is_off["pnl_r"].mean())
    delta_is = expr_on_is - expr_off_is
    winrate_on_is = float((is_on["pnl_r"] > 0).mean())
    winrate_off_is = float((is_off["pnl_r"] > 0).mean())
    welch_t, welch_p = stats.ttest_ind(
        is_on["pnl_r"].to_numpy(dtype=float),
        is_off["pnl_r"].to_numpy(dtype=float),
        equal_var=False,
    )

    expr_on_oos = float(oos_on["pnl_r"].mean()) if len(oos_on) else float("nan")
    expr_off_oos = float(oos_off["pnl_r"].mean()) if len(oos_off) else float("nan")
    delta_oos = (
        float(expr_on_oos - expr_off_oos)
        if len(oos_on) > 0 and len(oos_off) > 0
        else float("nan")
    )
    same_sign_oos = (
        bool(np.sign(delta_is) == np.sign(delta_oos))
        if np.isfinite(delta_oos) and delta_is != 0.0 and delta_oos != 0.0
        else None
    )
    role = "TAKE" if delta_is > 0 else "AVOID"

    return {
        "orb_label": parent.orb_label,
        "rr_target": parent.rr_target,
        "direction": parent.direction,
        "signal": signal_name,
        "n_on_is": int(len(is_on)),
        "n_off_is": int(len(is_off)),
        "expr_on_is": expr_on_is,
        "expr_off_is": expr_off_is,
        "delta_is": delta_is,
        "winrate_on_is": winrate_on_is,
        "winrate_off_is": winrate_off_is,
        "welch_t": float(welch_t),
        "welch_p": float(welch_p),
        "n_on_oos": int(len(oos_on)),
        "n_off_oos": int(len(oos_off)),
        "expr_on_oos": expr_on_oos,
        "expr_off_oos": expr_off_oos,
        "delta_oos": delta_oos,
        "same_sign_oos": same_sign_oos,
        "role": role,
    }


def build_board() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specs = signal_specs()
    for parent in PARENTS:
        df = add_features(load_parent_lane(parent))
        for signal_name, inputs in specs:
            row = evaluate_signal(df, parent, signal_name, inputs)
            if row is not None:
                rows.append(row)

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("candidate board produced no valid rows")

    _, bh_p, _, _ = multipletests(results["welch_p"].to_numpy(dtype=float), alpha=0.10, method="fdr_bh")
    results["bh_p"] = bh_p
    results["abs_delta_is"] = results["delta_is"].abs()
    results["priority_rank"] = (
        results["same_sign_oos"].eq(True).astype(int) * 10_000
        + (results["abs_delta_is"] * 1_000).round().astype(int)
        - results["bh_p"].fillna(1.0).mul(100).round().astype(int)
    )
    results = results.sort_values(
        ["priority_rank", "abs_delta_is", "welch_t"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return results


def _fmt_float(value: float) -> str:
    if pd.isna(value):
        return "nan"
    return f"{value:+.4f}"


def write_markdown(results: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# 2026-04-22 MNQ Layered Candidate Board v1")
    lines.append("")
    lines.append("Canonical read-only discovery board on `orb_outcomes x daily_features`.")
    lines.append("")
    lines.append(f"- Parent lanes: {len(PARENTS)}")
    lines.append(f"- Signal specs per lane: {len(signal_specs())}")
    lines.append(f"- Actual tested rows: {len(results)}")
    lines.append(f"- Holdout split: IS < {OOS_START.date()}, OOS >= {OOS_START.date()}")
    lines.append("")
    lines.append("## Top Board")
    lines.append("")
    top = results.head(15)
    lines.append("| Lane | Signal | Role | N_on_IS | ExpR_on_IS | ExpR_off_IS | Delta_IS | t | p_bh | N_on_OOS | Delta_OOS | OOS sign |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for _, row in top.iterrows():
        lane = f"{row['orb_label']} RR{row['rr_target']} {row['direction']}"
        lines.append(
            f"| {lane} | {row['signal']} | {row['role']} | {int(row['n_on_is'])} | "
            f"{_fmt_float(row['expr_on_is'])} | {_fmt_float(row['expr_off_is'])} | "
            f"{_fmt_float(row['delta_is'])} | {_fmt_float(row['welch_t'])} | "
            f"{row['bh_p']:.4f} | {int(row['n_on_oos'])} | {_fmt_float(row['delta_oos'])} | "
            f"{row['same_sign_oos']} |"
        )
    lines.append("")

    for parent in PARENTS:
        sub = results[
            (results["orb_label"] == parent.orb_label)
            & (results["rr_target"] == parent.rr_target)
            & (results["direction"] == parent.direction)
        ].copy()
        lines.append(f"## {parent.orb_label} RR{parent.rr_target} {parent.direction}")
        lines.append("")
        take = sub[sub["role"] == "TAKE"].head(8)
        avoid = sub[sub["role"] == "AVOID"].sort_values(
            ["abs_delta_is", "welch_t"],
            ascending=[False, True],
        ).head(8)
        lines.append("### Best Take States")
        lines.append("")
        if take.empty:
            lines.append("- none")
        else:
            for _, row in take.iterrows():
                lines.append(
                    f"- `{row['signal']}`: `N_on_IS={int(row['n_on_is'])}`, "
                    f"`ExpR_on_IS={_fmt_float(row['expr_on_is'])}`, "
                    f"`Delta_IS={_fmt_float(row['delta_is'])}`, "
                    f"`t={_fmt_float(row['welch_t'])}`, `BH={row['bh_p']:.4f}`, "
                    f"`N_on_OOS={int(row['n_on_oos'])}`, `Delta_OOS={_fmt_float(row['delta_oos'])}`"
                )
        lines.append("")
        lines.append("### Worst On-States (Avoid Candidates)")
        lines.append("")
        if avoid.empty:
            lines.append("- none")
        else:
            for _, row in avoid.iterrows():
                lines.append(
                    f"- `{row['signal']}`: `N_on_IS={int(row['n_on_is'])}`, "
                    f"`ExpR_on_IS={_fmt_float(row['expr_on_is'])}`, "
                    f"`Delta_IS={_fmt_float(row['delta_is'])}`, "
                    f"`t={_fmt_float(row['welch_t'])}`, `BH={row['bh_p']:.4f}`, "
                    f"`N_on_OOS={int(row['n_on_oos'])}`, `Delta_OOS={_fmt_float(row['delta_oos'])}`"
                )
        lines.append("")

    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    results = build_board()
    results.to_csv(OUTPUT_CSV, index=False)
    write_markdown(results)
    print(f"[mnq-layered-board] wrote {OUTPUT_CSV}")
    print(f"[mnq-layered-board] wrote {OUTPUT_MD}")
    print(results.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
