"""Bounded MNQ prior-day family board on live-adjacent positive parents.

Purpose:
- step one level broader than exact-cell sniping
- stay within canonical read-only truth (`daily_features`, `orb_outcomes`)
- test small mechanism families rather than new-on-new stacks

Not a promotion runner. This is a low-K shortlist builder for the next bridge.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
OUTPUT_CSV = OUTPUT_DIR / "2026-04-22-mnq-prior-day-family-board-v1.csv"
OUTPUT_MD = OUTPUT_DIR / "2026-04-22-mnq-prior-day-family-board-v1.md"


@dataclass(frozen=True)
class ParentLane:
    orb_label: str
    rr_target: float
    direction: str


PARENTS: tuple[ParentLane, ...] = (
    ParentLane("CME_PRECLOSE", 1.0, "long"),
    ParentLane("NYSE_OPEN", 1.5, "long"),
    ParentLane("US_DATA_1000", 1.0, "long"),
)

FAMILY_DEFS: dict[str, tuple[str, ...]] = {
    "TAKE_DOWNSIDE_DISPLACEMENT": ("F2_NEAR_PDL_15", "F5_BELOW_PDL"),
    "AVOID_CONGESTION": ("F3_NEAR_PIVOT_50", "F6_INSIDE_PDR"),
    "TAKE_OVERHEAD_BREAK": ("F1_NEAR_PDH_15", "F4_ABOVE_PDH"),
}


def load_parent_lane(parent: ParentLane) -> pd.DataFrame:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    session = parent.orb_label
    query = f"""
    SELECT
        o.trading_day,
        o.pnl_r,
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
    out["F3_NEAR_PIVOT_50"] = (np.abs(mid - pivot) / atr < 0.50).astype(int)
    out["F4_ABOVE_PDH"] = (mid > pdh).astype(int)
    out["F5_BELOW_PDL"] = (mid < pdl).astype(int)
    out["F6_INSIDE_PDR"] = ((mid > pdl) & (mid < pdh)).astype(int)
    return out


def evaluate_family(df: pd.DataFrame, parent: ParentLane, family_name: str, members: tuple[str, ...]) -> dict[str, object] | None:
    mask = pd.Series(False, index=df.index)
    for column in members:
        mask |= df[column].astype(bool)

    on = df[mask]
    off = df[~mask]
    is_on = on[on["trading_day"] < OOS_START]
    is_off = off[off["trading_day"] < OOS_START]
    if len(is_on) < MIN_IS_TRADES or len(is_off) < MIN_IS_TRADES:
        return None

    oos_on = on[on["trading_day"] >= OOS_START]
    oos_off = off[ooff := off["trading_day"] >= OOS_START]
    _ = ooff  # keep lint quiet in py3.13 walrus-free path

    expr_on_is = float(is_on["pnl_r"].mean())
    expr_off_is = float(is_off["pnl_r"].mean())
    delta_is = expr_on_is - expr_off_is
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
        "family": family_name,
        "members": " OR ".join(members),
        "n_on_is": int(len(is_on)),
        "n_off_is": int(len(is_off)),
        "expr_on_is": expr_on_is,
        "expr_off_is": expr_off_is,
        "delta_is": delta_is,
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
    for parent in PARENTS:
        df = add_features(load_parent_lane(parent))
        for family_name, members in FAMILY_DEFS.items():
            row = evaluate_family(df, parent, family_name, members)
            if row is not None:
                rows.append(row)

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("family board produced no valid rows")
    _, bh_p, _, _ = multipletests(results["welch_p"].to_numpy(dtype=float), alpha=0.10, method="fdr_bh")
    results["bh_p"] = bh_p
    results["abs_delta_is"] = results["delta_is"].abs()
    results = results.sort_values(
        ["same_sign_oos", "bh_p", "abs_delta_is"],
        ascending=[False, True, False],
    ).reset_index(drop=True)
    return results


def _fmt(value: float) -> str:
    if pd.isna(value):
        return "nan"
    return f"{value:+.4f}"


def write_markdown(results: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# 2026-04-22 MNQ Prior-Day Family Board v1")
    lines.append("")
    lines.append("Bounded read-only board on canonical `daily_features x orb_outcomes`.")
    lines.append("")
    lines.append("- Parents: 3 live-adjacent positive MNQ lanes")
    lines.append("- Families: 3 prior-day mechanism families")
    lines.append("- K_family: 9")
    lines.append(f"- Holdout split: IS < {OOS_START.date()}, OOS >= {OOS_START.date()}")
    lines.append("")
    lines.append("| Lane | Family | Role | N_on_IS | ExpR_on_IS | ExpR_off_IS | Delta_IS | t | BH | N_on_OOS | Delta_OOS | OOS sign |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in results.itertuples(index=False):
        lines.append(
            f"| {row.orb_label} RR{row.rr_target} {row.direction} | {row.family} | {row.role} | "
            f"{row.n_on_is} | {_fmt(row.expr_on_is)} | {_fmt(row.expr_off_is)} | {_fmt(row.delta_is)} | "
            f"{row.welch_t:+.2f} | {row.bh_p:.4f} | {row.n_on_oos} | {_fmt(row.delta_oos)} | {row.same_sign_oos} |"
        )
    lines.append("")
    lines.append("## Family definitions")
    lines.append("")
    for family_name, members in FAMILY_DEFS.items():
        lines.append(f"- `{family_name}`: `{' OR '.join(members)}`")
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    results = build_board()
    results.to_csv(OUTPUT_CSV, index=False)
    write_markdown(results)
    print(results.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
