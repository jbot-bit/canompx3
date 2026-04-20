#!/usr/bin/env python3
"""MNQ prior-day positional ORB overlays v1.

Locked by:
  docs/audit/hypotheses/2026-04-20-prior-day-direction-split-orb-overlays-v1.yaml

Purpose:
  Run a low-blast, direction-split ORB context family on a fixed MNQ O5/E2/CB1
  surface. This is a context-overlay scan, not a new entry model and not a
  standalone trade family.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from research.lib import bh_fdr, connect_db
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

INSTRUMENT = "MNQ"
SESSIONS = ("NYSE_OPEN", "COMEX_SETTLE", "US_DATA_1000")
RR_TARGETS = (1.0, 1.5)
DIRECTIONS = ("long", "short")
FEATURES = ("F1_NEAR_PDH_15", "F5_BELOW_PDL", "F6_INSIDE_PDR")
BH_Q = 0.05
MIN_N = 100
T_ABS_MIN = 3.0
DELTA_ABS_MIN = 0.05
YEAR_MIN_GROUP = 10
OOS_MIN_GROUP = 30
HOLDOUT = pd.Timestamp(HOLDOUT_SACRED_FROM).date()

OUTPUT_DIR = Path("research/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = OUTPUT_DIR / "prior_day_direction_split_orb_overlays_v1.csv"
OUTPUT_MD = Path("docs/audit/results/2026-04-20-prior-day-direction-split-orb-overlays-v1.md")


@dataclass(frozen=True)
class CellSpec:
    session: str
    rr_target: float
    direction: str
    feature: str


def _load_base_df() -> pd.DataFrame:
    unions: list[str] = []
    for session in SESSIONS:
        unions.append(
            f"""
            SELECT
                o.trading_day,
                o.symbol,
                o.orb_label,
                o.orb_minutes,
                o.entry_model,
                o.confirm_bars,
                o.rr_target,
                o.entry_price,
                o.stop_price,
                o.pnl_r,
                d.atr_20,
                d.prev_day_high,
                d.prev_day_low,
                d.prev_day_close,
                d.orb_{session}_high AS orb_high,
                d.orb_{session}_low AS orb_low
            FROM orb_outcomes o
            JOIN daily_features d
              ON o.trading_day = d.trading_day
             AND o.symbol = d.symbol
             AND o.orb_minutes = d.orb_minutes
            WHERE o.symbol = '{INSTRUMENT}'
              AND o.orb_label = '{session}'
              AND o.orb_minutes = 5
              AND o.entry_model = 'E2'
              AND o.confirm_bars = 1
              AND o.rr_target IN ({", ".join(str(rr) for rr in RR_TARGETS)})
              AND o.pnl_r IS NOT NULL
              AND d.atr_20 IS NOT NULL
              AND d.atr_20 > 0
              AND d.prev_day_high IS NOT NULL
              AND d.prev_day_low IS NOT NULL
              AND d.prev_day_close IS NOT NULL
            """
        )

    sql = " UNION ALL ".join(unions)
    with connect_db() as con:
        df = con.execute(sql).fetchdf()

    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["direction"] = np.where(df["entry_price"] > df["stop_price"], "long", "short")
    df["is_is"] = df["trading_day"].dt.date < HOLDOUT
    df["is_oos"] = ~df["is_is"]
    df["year"] = df["trading_day"].dt.year

    orb_mid = (df["orb_high"].astype(float) + df["orb_low"].astype(float)) / 2.0
    atr = df["atr_20"].astype(float)
    pdh = df["prev_day_high"].astype(float)
    pdl = df["prev_day_low"].astype(float)

    df["F1_NEAR_PDH_15"] = (np.abs(orb_mid - pdh) / atr < 0.15)
    df["F5_BELOW_PDL"] = orb_mid < pdl
    df["F6_INSIDE_PDR"] = (orb_mid > pdl) & (orb_mid < pdh)
    return df


def _welch(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    res = stats.ttest_ind(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_var=False)
    return float(np.asarray(res.statistic)), float(np.asarray(res.pvalue))


def _safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def _role(delta: float, exp_on: float, exp_off: float) -> str:
    if np.isnan(delta):
        return "NONE"
    if delta >= DELTA_ABS_MIN and exp_on > 0:
        return "TAKE"
    if delta <= -DELTA_ABS_MIN and exp_off > 0:
        return "AVOID"
    return "NONE"


def _year_consistency(df_is: pd.DataFrame, feature: str, signal_sign: float) -> tuple[int, int]:
    eligible = 0
    consistent = 0
    for _, year_df in df_is.groupby("year"):
        on = year_df.loc[year_df[feature], "pnl_r"]
        off = year_df.loc[~year_df[feature], "pnl_r"]
        if len(on) < YEAR_MIN_GROUP or len(off) < YEAR_MIN_GROUP:
            continue
        eligible += 1
        delta = float(on.mean() - off.mean())
        if np.sign(delta) == np.sign(signal_sign):
            consistent += 1
    return eligible, consistent


def _evaluate_cell(df: pd.DataFrame, spec: CellSpec) -> dict[str, object]:
    cell = df[
        (df["orb_label"] == spec.session)
        & (df["rr_target"] == spec.rr_target)
        & (df["direction"] == spec.direction)
    ].copy()
    is_df = cell[cell["is_is"]].copy()
    oos_df = cell[cell["is_oos"]].copy()

    on_is = is_df.loc[is_df[spec.feature], "pnl_r"]
    off_is = is_df.loc[~is_df[spec.feature], "pnl_r"]
    on_oos = oos_df.loc[oos_df[spec.feature], "pnl_r"]
    off_oos = oos_df.loc[~oos_df[spec.feature], "pnl_r"]

    exp_on_is = _safe_mean(on_is)
    exp_off_is = _safe_mean(off_is)
    delta_is = exp_on_is - exp_off_is
    t_is, p_is = _welch(on_is, off_is)

    exp_on_oos = _safe_mean(on_oos)
    exp_off_oos = _safe_mean(off_oos)
    delta_oos = exp_on_oos - exp_off_oos
    role = _role(delta_is, exp_on_is, exp_off_is)
    eligible_years, years_consistent = _year_consistency(is_df, spec.feature, delta_is)

    oos_dir_match: bool | None
    if len(on_oos) >= OOS_MIN_GROUP and len(off_oos) >= OOS_MIN_GROUP:
        oos_dir_match = bool(np.sign(delta_oos) == np.sign(delta_is))
    else:
        oos_dir_match = None

    return {
        "instrument": INSTRUMENT,
        "session": spec.session,
        "rr_target": spec.rr_target,
        "direction": spec.direction,
        "feature": spec.feature,
        "n_is": int(len(is_df)),
        "n_on_is": int(len(on_is)),
        "n_off_is": int(len(off_is)),
        "exp_on_is": exp_on_is,
        "exp_off_is": exp_off_is,
        "delta_is": delta_is,
        "t_is": t_is,
        "p_is": p_is,
        "n_oos": int(len(oos_df)),
        "n_on_oos": int(len(on_oos)),
        "n_off_oos": int(len(off_oos)),
        "exp_on_oos": exp_on_oos,
        "exp_off_oos": exp_off_oos,
        "delta_oos": delta_oos,
        "fire_rate_is": float(len(on_is) / len(is_df)) if len(is_df) else float("nan"),
        "role": role,
        "eligible_years": eligible_years,
        "years_consistent": years_consistent,
        "oos_dir_match": oos_dir_match,
    }


def _format_float(value: object, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (float, np.floating)):
        if np.isnan(value) or np.isinf(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    return str(value)


def main() -> None:
    df = _load_base_df()
    specs = [
        CellSpec(session=session, rr_target=rr, direction=direction, feature=feature)
        for session in SESSIONS
        for rr in RR_TARGETS
        for direction in DIRECTIONS
        for feature in FEATURES
    ]
    rows = [_evaluate_cell(df, spec) for spec in specs]
    result_df = pd.DataFrame(rows)

    pvals = result_df["p_is"].fillna(1.0).tolist()
    survivors = bh_fdr(pvals, q=BH_Q)
    ranked = sorted(enumerate(pvals), key=lambda item: item[1])
    q_values = [1.0] * len(pvals)
    running = 1.0
    for rank in range(len(ranked) - 1, -1, -1):
        idx, p_val = ranked[rank]
        q_val = min(running, p_val * len(ranked) / (rank + 1))
        running = q_val
        q_values[idx] = q_val
    result_df["q_family"] = q_values
    result_df["bh_survivor"] = [idx in survivors for idx in range(len(result_df))]

    result_df["passes_primary"] = (
        result_df["bh_survivor"]
        & (result_df["q_family"] < BH_Q)
        & (result_df["n_on_is"] >= MIN_N)
        & (result_df["n_off_is"] >= MIN_N)
        & (result_df["role"] != "NONE")
        & (result_df["delta_is"].abs() >= DELTA_ABS_MIN)
        & (result_df["t_is"].abs() >= T_ABS_MIN)
        & (
            result_df["oos_dir_match"].isna()
            | result_df["oos_dir_match"].astype(bool)
        )
    )

    result_df["verdict"] = np.where(
        result_df["passes_primary"],
        "ALIVE",
        np.where(
            result_df["bh_survivor"] & (result_df["role"] != "NONE"),
            "CONDITIONAL",
            "DEAD",
        ),
    )

    result_df.to_csv(OUTPUT_CSV, index=False)

    alive = result_df[result_df["passes_primary"]].copy()
    conditional = result_df[
        (result_df["verdict"] == "CONDITIONAL") & (~result_df["passes_primary"])
    ].copy()
    best = result_df.sort_values("delta_is", ascending=False).head(8)
    worst = result_df.sort_values("delta_is", ascending=True).head(8)

    lines = [
        "# Prior-Day Direction-Split ORB Overlays V1",
        "",
        "Locked by `docs/audit/hypotheses/2026-04-20-prior-day-direction-split-orb-overlays-v1.yaml`.",
        "",
        "## Scope",
        "",
        f"- Instrument: {INSTRUMENT}",
        f"- Sessions: {', '.join(SESSIONS)}",
        "- Aperture: O5",
        "- Entry model: E2",
        "- Confirm bars: CB1",
        f"- RR targets: {', '.join(str(v) for v in RR_TARGETS)}",
        f"- Directions: {', '.join(DIRECTIONS)}",
        f"- Features: {', '.join(FEATURES)}",
        "- Mode A: trading_day < 2026-01-01 is IS; 2026+ is descriptive OOS only",
        "",
        "## Resource grounding",
        "",
        "- `resources/Algorithmic_Trading_Chan.pdf`: intraday/systematic strategy framing justifies testing bounded, executable pattern families instead of ad hoc chart lore.",
        "- `resources/Robert Carver - Systematic Trading.pdf`: supports treating signals as conditioners/sizers rather than assuming every useful feature must be a standalone strategy.",
        "- `resources/Lopez_de_Prado_ML_for_Asset_Managers.pdf`: theory-first discipline; use a small pre-registered family instead of backtest fishing.",
        "- `resources/Two_Million_Trading_Strategies_FDR.pdf`: honest family-level multiple-testing control is mandatory.",
        "",
        "## Family verdict",
        "",
        f"- Locked family K: {len(specs)}",
        f"- Primary survivors: {int(result_df['passes_primary'].sum())}",
        f"- Conditional cells: {int((result_df['verdict'] == 'CONDITIONAL').sum())}",
        "",
    ]

    if alive.empty:
        lines.extend(
            [
                "No primary survivors.",
                "",
                "The family did not surface a context state strong enough to justify immediate TAKE/AVOID promotion under the locked gates.",
                "",
            ]
        )
    else:
        lines.extend(["### Primary survivors", ""])
        for row in alive.itertuples(index=False):
            lines.append(
                f"- {row.session} RR{row.rr_target:.1f} {row.direction} {row.feature}: "
                f"{row.role} | "
                f"IS on/off ExpR {row.exp_on_is:+.4f}/{row.exp_off_is:+.4f} | "
                f"delta {row.delta_is:+.4f} | t={row.t_is:.2f} | q={row.q_family:.4f} | "
                f"N_on/N_off={row.n_on_is}/{row.n_off_is}"
            )
        lines.append("")

    if not conditional.empty:
        lines.extend(["### Conditional cells", ""])
        for row in conditional.itertuples(index=False):
            lines.append(
                f"- {row.session} RR{row.rr_target:.1f} {row.direction} {row.feature}: "
                f"{row.role} | delta {row.delta_is:+.4f} | t={_format_float(row.t_is, 2)} | "
                f"q={_format_float(row.q_family, 4)} | N_on/N_off={row.n_on_is}/{row.n_off_is} | "
                f"OOS dir match={row.oos_dir_match}"
            )
        lines.append("")

    lines.extend(["## Strongest positive deltas", ""])
    for row in best.itertuples(index=False):
        lines.append(
            f"- {row.session} RR{row.rr_target:.1f} {row.direction} {row.feature}: "
            f"role={row.role} delta={row.delta_is:+.4f} q={row.q_family:.4f} "
            f"N_on/N_off={row.n_on_is}/{row.n_off_is}"
        )
    lines.append("")

    lines.extend(["## Strongest negative deltas", ""])
    for row in worst.itertuples(index=False):
        lines.append(
            f"- {row.session} RR{row.rr_target:.1f} {row.direction} {row.feature}: "
            f"role={row.role} delta={row.delta_is:+.4f} q={row.q_family:.4f} "
            f"N_on/N_off={row.n_on_is}/{row.n_off_is}"
        )
    lines.append("")

    lines.extend(
        [
            "## Caveats",
            "",
            "- This is a context-overlay family only. It does not validate a standalone trade class.",
            "- 2026 OOS remains descriptive unless both on/off groups reach the pre-registered 30-trade floor.",
            "- TAKE vs AVOID is derived from one signed delta per cell, not from a doubled post-hoc hypothesis set.",
            "",
            "## Artefacts",
            "",
            f"- CSV: `{OUTPUT_CSV}`",
            "- Script: `research/prior_day_direction_split_orb_overlays_v1.py`",
        ]
    )

    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_MD}")
    print(
        f"Primary survivors={int(result_df['passes_primary'].sum())} | "
        f"Conditional={int((result_df['verdict'] == 'CONDITIONAL').sum())}"
    )


if __name__ == "__main__":
    main()
