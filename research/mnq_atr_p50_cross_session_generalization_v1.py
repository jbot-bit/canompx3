"""Locked ATR_P50 cross-session generalization audit for MNQ.

Scope is frozen by
`docs/audit/hypotheses/2026-04-22-mnq-atr-p50-cross-session-generalization-v1.yaml`.

This runner tests the canonical ATR_P50 filter as an R1 binary overlay across
all MNQ sessions on the two apertures already present in the repo surface:
O15 and O30, both at E2 / CB1 / RR1.5.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from pipeline.paths import GOLD_DB_PATH
from research.filter_utils import filter_signal
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

OUTPUT_MD = Path("docs/audit/results/2026-04-22-mnq-atr-p50-cross-session-generalization-v1.md")
OUTPUT_CSV = Path("docs/audit/results/2026-04-22-mnq-atr-p50-cross-session-generalization-v1-rows.csv")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

INSTRUMENT = "MNQ"
ENTRY_MODEL = "E2"
CONFIRM_BARS = 1
RR_TARGET = 1.5
APERTURES = (15, 30)
SESSIONS = (
    "BRISBANE_1025",
    "CME_PRECLOSE",
    "CME_REOPEN",
    "COMEX_SETTLE",
    "EUROPE_FLOW",
    "LONDON_METALS",
    "NYSE_CLOSE",
    "NYSE_OPEN",
    "SINGAPORE_OPEN",
    "TOKYO_OPEN",
    "US_DATA_1000",
    "US_DATA_830",
)
FILTER_KEY = "ATR_P50"
POWER_FLOOR = 50


@dataclass(frozen=True)
class GroupStats:
    n: int
    wins: int
    expr: float
    wr: float


def _safe_float(value: object) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    return float(value)


def _welch(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    t_stat, p_val = stats.ttest_ind(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_var=False)
    return float(t_stat), float(p_val)


def _two_prop(a_wins: int, a_n: int, b_wins: int, b_n: int) -> tuple[float, float]:
    if min(a_n, b_n) == 0:
        return float("nan"), float("nan")
    pooled = (a_wins + b_wins) / (a_n + b_n)
    se = np.sqrt(pooled * (1.0 - pooled) * ((1.0 / a_n) + (1.0 / b_n)))
    if se == 0:
        return float("nan"), float("nan")
    z_val = ((a_wins / a_n) - (b_wins / b_n)) / se
    p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_val)))
    return float(z_val), float(p_val)


def _bh_fdr(p_values: pd.Series) -> pd.Series:
    valid = p_values.notna()
    out = pd.Series(np.nan, index=p_values.index, dtype=float)
    if not valid.any():
        return out

    ranked = p_values[valid].sort_values()
    n = len(ranked)
    adjusted = np.empty(n, dtype=float)
    min_so_far = 1.0
    for idx in range(n - 1, -1, -1):
        rank = idx + 1
        candidate = ranked.iloc[idx] * n / rank
        min_so_far = min(min_so_far, candidate)
        adjusted[idx] = min_so_far
    out.loc[ranked.index] = np.clip(adjusted, 0.0, 1.0)
    return out


def _group_stats(pnl: pd.Series) -> GroupStats:
    n = int(len(pnl))
    wins = int((pnl > 0).sum())
    expr = float(pnl.mean()) if n else float("nan")
    wr = float(wins / n) if n else float("nan")
    return GroupStats(n=n, wins=wins, expr=expr, wr=wr)


def _aperture_verdict(metrics: pd.DataFrame) -> tuple[str, float]:
    sgp = metrics.loc[metrics["session"] == "SINGAPORE_OPEN"].iloc[0]
    if sgp["delta_expr"] <= 0.0 or pd.isna(sgp["welch_p_raw"]) or sgp["welch_p_raw"] >= 0.05:
        return "SGP_FAILS_RAW", float("nan")

    powered = metrics[metrics["powered"]].copy()
    powered_median = float(powered["delta_expr"].median()) if not powered.empty else float("nan")
    if not powered.empty and powered_median <= 0.0:
        return "KILL_FAMILY", powered_median

    non_sgp = metrics[metrics["session"] != "SINGAPORE_OPEN"].copy()
    survivors = non_sgp[(non_sgp["delta_expr"] > 0.0) & (non_sgp["welch_q_local"] < 0.05)]
    bad_survivor = non_sgp[(non_sgp["delta_expr"] <= 0.0) & (non_sgp["welch_q_local"] < 0.05)]
    if not bad_survivor.empty:
        return "KILL_FAMILY", powered_median
    if len(survivors) >= 2:
        return "CLASS_GENERALIZES_STRONG", powered_median
    if len(survivors) >= 1:
        return "CLASS_GENERALIZES", powered_median
    return "LANE_LOCAL_ONLY", powered_median


def _overall_verdict(aperture_verdicts: dict[int, str], aperture_medians: dict[int, float]) -> str:
    if all(v == "SGP_FAILS_RAW" for v in aperture_verdicts.values()):
        return "KILL_FAMILY"
    if all(v == "KILL_FAMILY" for v in aperture_verdicts.values()):
        return "KILL_FAMILY"
    if all(v in {"KILL_FAMILY", "SGP_FAILS_RAW"} for v in aperture_verdicts.values()):
        return "KILL_FAMILY"
    if any(v == "CLASS_GENERALIZES_STRONG" for v in aperture_verdicts.values()):
        return "CLASS_GENERALIZES_STRONG"
    if any(v == "CLASS_GENERALIZES" for v in aperture_verdicts.values()):
        return "CLASS_GENERALIZES"
    if all(v in {"LANE_LOCAL_ONLY", "SGP_FAILS_RAW"} for v in aperture_verdicts.values()):
        if all(np.isnan(aperture_medians[a]) or aperture_medians[a] <= 0.0 for a in aperture_verdicts):
            return "KILL_FAMILY"
        return "LANE_LOCAL_ONLY"
    return "LANE_LOCAL_ONLY"


def _markdown_table(df: pd.DataFrame, float_cols: dict[str, int] | None = None) -> str:
    display = df.copy()
    float_cols = float_cols or {}
    for col, places in float_cols.items():
        if col in display.columns:

            def _fmt(x: object, digits: int = places) -> str:
                return "NaN" if pd.isna(x) else f"{x:.{digits}f}"

            display[col] = display[col].map(_fmt)
    cols = list(display.columns)
    rows = [cols] + [[str(row[col]) for col in cols] for _, row in display.iterrows()]
    widths = [max(len(r[i]) for r in rows) for i in range(len(cols))]

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(row))) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([fmt(rows[0]), sep, *[fmt(r) for r in rows[1:]]])


def _load_rows(con: duckdb.DuckDBPyConnection, aperture: int) -> pd.DataFrame:
    sql = """
    SELECT
        o.trading_day,
        o.orb_label AS session,
        o.orb_minutes,
        o.pnl_r,
        o.outcome,
        d.atr_20_pct
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol = ?
      AND o.orb_minutes = ?
      AND o.entry_model = ?
      AND o.confirm_bars = ?
      AND o.rr_target = ?
      AND o.pnl_r IS NOT NULL
      AND o.orb_label IN (
        'BRISBANE_1025','CME_PRECLOSE','CME_REOPEN','COMEX_SETTLE','EUROPE_FLOW',
        'LONDON_METALS','NYSE_CLOSE','NYSE_OPEN','SINGAPORE_OPEN','TOKYO_OPEN',
        'US_DATA_1000','US_DATA_830'
      )
    ORDER BY o.trading_day, o.orb_label
    """
    df = con.execute(sql, [INSTRUMENT, aperture, ENTRY_MODEL, CONFIRM_BARS, RR_TARGET]).df()
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["sample"] = np.where(df["trading_day"] < pd.Timestamp(HOLDOUT_SACRED_FROM), "IS", "OOS")
    df["aperture"] = aperture
    for session in SESSIONS:
        mask = df["session"] == session
        if mask.any():
            df.loc[mask, "fire"] = filter_signal(df.loc[mask].copy(), FILTER_KEY, session)
    df["fire"] = df["fire"].astype(int)
    return df


def _summarize_aperture(df: pd.DataFrame, aperture: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    oos_rows: list[dict[str, Any]] = []
    for session in SESSIONS:
        sub = df[df["session"] == session].copy()
        is_sub = sub[sub["sample"] == "IS"]
        oos_sub = sub[sub["sample"] == "OOS"]

        fire = is_sub[is_sub["fire"] == 1]["pnl_r"]
        nonfire = is_sub[is_sub["fire"] == 0]["pnl_r"]
        fire_stats = _group_stats(fire)
        nonfire_stats = _group_stats(nonfire)
        welch_t, welch_p = _welch(fire, nonfire)
        wr_z, wr_p = _two_prop(fire_stats.wins, fire_stats.n, nonfire_stats.wins, nonfire_stats.n)
        rows.append(
            {
                "aperture": aperture,
                "session": session,
                "n_total_is": int(len(is_sub)),
                "n_fire": fire_stats.n,
                "n_nonfire": nonfire_stats.n,
                "expR_fire": fire_stats.expr,
                "expR_nonfire": nonfire_stats.expr,
                "delta_expr": fire_stats.expr - nonfire_stats.expr,
                "wr_fire": fire_stats.wr,
                "wr_nonfire": nonfire_stats.wr,
                "wr_z": wr_z,
                "wr_p_raw": wr_p,
                "welch_t": welch_t,
                "welch_p_raw": welch_p,
                "powered": bool(fire_stats.n >= POWER_FLOOR and nonfire_stats.n >= POWER_FLOOR),
            }
        )
        fire_oos = oos_sub[oos_sub["fire"] == 1]["pnl_r"]
        nonfire_oos = oos_sub[oos_sub["fire"] == 0]["pnl_r"]
        oos_rows.append(
            {
                "aperture": aperture,
                "session": session,
                "n_total_oos": int(len(oos_sub)),
                "n_fire_oos": int(len(fire_oos)),
                "n_nonfire_oos": int(len(nonfire_oos)),
                "expR_fire_oos": _safe_float(fire_oos.mean()) if len(fire_oos) else float("nan"),
                "expR_nonfire_oos": _safe_float(nonfire_oos.mean()) if len(nonfire_oos) else float("nan"),
                "delta_expr_oos": (
                    _safe_float(fire_oos.mean()) - _safe_float(nonfire_oos.mean())
                    if len(fire_oos) and len(nonfire_oos)
                    else float("nan")
                ),
            }
        )

    summary = pd.DataFrame(rows)
    summary["welch_q_local"] = _bh_fdr(summary["welch_p_raw"])
    summary["welch_q_conservative"] = _bh_fdr(summary["welch_p_raw"])
    return summary, pd.DataFrame(oos_rows)


def _render_report(
    prereg_sha: str,
    is_rows: pd.DataFrame,
    summary_by_aperture: dict[int, pd.DataFrame],
    oos_by_aperture: dict[int, pd.DataFrame],
    aperture_verdicts: dict[int, str],
    aperture_medians: dict[int, float],
    overall_verdict: str,
) -> str:
    lines = [
        "# MNQ ATR_P50 Cross-Session Generalization Audit",
        "",
        f"**Date:** {pd.Timestamp.now(tz='Australia/Brisbane').date()}",
        f"**Prereg lock:** `docs/audit/hypotheses/2026-04-22-mnq-atr-p50-cross-session-generalization-v1.yaml` (`commit_sha={prereg_sha}`)",
        "",
        "## Verdict",
        "",
        f"`{overall_verdict}`",
        "",
        "This is a fixed-family audit of ATR_P50 as an `R1` overlay across all",
        "MNQ sessions on the two apertures the repo already uses for this class:",
        "`O15` and `O30`.",
        "",
    ]

    for aperture in APERTURES:
        summary = summary_by_aperture[aperture].copy()
        oos = oos_by_aperture[aperture].copy()
        lines.extend(
            [
                f"## Aperture O{aperture}",
                "",
                f"- IS rows: `{int((is_rows['aperture'] == aperture).sum())}`",
                f"- Powered-session median `delta_expr`: `{aperture_medians[aperture]:+.4f}`"
                if not np.isnan(aperture_medians[aperture])
                else "- Powered-session median `delta_expr`: `NaN`",
                f"- Aperture verdict: `{aperture_verdicts[aperture]}`",
                "",
                _markdown_table(
                    summary[
                        [
                            "session",
                            "n_total_is",
                            "n_fire",
                            "n_nonfire",
                            "expR_fire",
                            "expR_nonfire",
                            "delta_expr",
                            "wr_fire",
                            "wr_nonfire",
                            "welch_p_raw",
                            "welch_q_local",
                            "powered",
                        ]
                    ],
                    {
                        "expR_fire": 4,
                        "expR_nonfire": 4,
                        "delta_expr": 4,
                        "wr_fire": 4,
                        "wr_nonfire": 4,
                        "welch_p_raw": 4,
                        "welch_q_local": 4,
                    },
                ),
                "",
                "### OOS descriptive only",
                "",
                _markdown_table(
                    oos[
                        [
                            "session",
                            "n_total_oos",
                            "n_fire_oos",
                            "n_nonfire_oos",
                            "expR_fire_oos",
                            "expR_nonfire_oos",
                            "delta_expr_oos",
                        ]
                    ],
                    {
                        "expR_fire_oos": 4,
                        "expR_nonfire_oos": 4,
                        "delta_expr_oos": 4,
                    },
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Decision",
            "",
            f"- Overall family verdict: `{overall_verdict}`",
            f"- O15 verdict: `{aperture_verdicts[15]}`",
            f"- O30 verdict: `{aperture_verdicts[30]}`",
            "- OOS is reported descriptively only and does not vote in the verdict.",
            "- No threshold search, aperture widening beyond O15/O30, or role drift was performed.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        all_rows = pd.concat([_load_rows(con, aperture) for aperture in APERTURES], ignore_index=True)
    finally:
        con.close()

    OUTPUT_CSV.write_text("")
    all_rows.to_csv(OUTPUT_CSV, index=False)

    is_rows = all_rows[all_rows["sample"] == "IS"].copy()
    summary_by_aperture: dict[int, pd.DataFrame] = {}
    oos_by_aperture: dict[int, pd.DataFrame] = {}
    aperture_verdicts: dict[int, str] = {}
    aperture_medians: dict[int, float] = {}

    combined_p = []
    for aperture in APERTURES:
        summary, oos = _summarize_aperture(all_rows[all_rows["aperture"] == aperture].copy(), aperture)
        summary_by_aperture[aperture] = summary
        oos_by_aperture[aperture] = oos
        combined_p.append(summary["welch_p_raw"])

    combined_q = _bh_fdr(pd.concat(combined_p, ignore_index=True))
    offset = 0
    for aperture in APERTURES:
        summary = summary_by_aperture[aperture]
        count = len(summary)
        summary["welch_q_conservative"] = combined_q.iloc[offset : offset + count].to_numpy()
        verdict, median = _aperture_verdict(summary)
        aperture_verdicts[aperture] = verdict
        aperture_medians[aperture] = median
        offset += count

    overall_verdict = _overall_verdict(aperture_verdicts, aperture_medians)
    report = _render_report(
        prereg_sha="1be7e6db",
        is_rows=is_rows,
        summary_by_aperture=summary_by_aperture,
        oos_by_aperture=oos_by_aperture,
        aperture_verdicts=aperture_verdicts,
        aperture_medians=aperture_medians,
        overall_verdict=overall_verdict,
    )
    OUTPUT_MD.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
