#!/usr/bin/env python3
"""MGC 5-minute payoff-compression audit.

Locked by:
  docs/audit/hypotheses/2026-04-19-mgc-payoff-compression-audit.yaml

Purpose:
- stay on the narrow GC->MGC translation question
- compare broad vs warm translated MGC 5-minute families
- test whether canonical T80 time-stop or conservative lower-R handling
  materially changes the translation verdict
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import pandas as pd

from pipeline.cost_model import get_cost_spec, to_r_multiple
from research.lib import connect_db, write_csv
from trading_app.config import EARLY_EXIT_MINUTES

OVERLAP_START = "2022-06-13"
HOLDOUT_START = "2026-01-01"
RESULT_PATH = Path("docs/audit/results/2026-04-19-mgc-payoff-compression-audit.md")
OUTPUT_PREFIX = "mgc_payoff_compression_audit"
MGC_SPEC = get_cost_spec("MGC")


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    orb_label: str
    kind: str
    filter_type: str | None = None


FAMILIES: tuple[FamilySpec, ...] = (
    FamilySpec("EUROPE_FLOW_BROAD_RR1", "EUROPE_FLOW", "broad"),
    FamilySpec("NYSE_OPEN_BROAD_RR1", "NYSE_OPEN", "broad"),
    FamilySpec("US_DATA_1000_BROAD_RR1", "US_DATA_1000", "broad"),
    FamilySpec("EUROPE_FLOW_OVNRNG_50_RR1", "EUROPE_FLOW", "warm", "OVNRNG_50"),
    FamilySpec("NYSE_OPEN_OVNRNG_50_RR1", "NYSE_OPEN", "warm", "OVNRNG_50"),
    FamilySpec("US_DATA_1000_ATR_P70_RR1", "US_DATA_1000", "warm", "ATR_P70"),
    FamilySpec("US_DATA_1000_ORB_G5_RR1", "US_DATA_1000", "warm", "ORB_G5"),
    FamilySpec("US_DATA_1000_OVNRNG_10_RR1", "US_DATA_1000", "warm", "OVNRNG_10"),
)


def fetch_df(sql: str) -> pd.DataFrame:
    with connect_db() as con:
        return con.execute(sql).fetchdf()


def load_rows(end_exclusive: str | None = HOLDOUT_START) -> pd.DataFrame:
    upper_clause = ""
    if end_exclusive is not None:
        upper_clause = f"AND o.trading_day < DATE '{end_exclusive}'"
    sql = f"""
    SELECT
        o.trading_day,
        o.orb_label,
        o.entry_price,
        o.stop_price,
        o.outcome,
        o.pnl_r,
        o.mae_r,
        o.mfe_r,
        o.ambiguous_bar,
        COALESCE(o.ts_pnl_r, o.pnl_r) AS ts_pnl_r,
        COALESCE(o.ts_outcome, o.outcome) AS ts_outcome,
        d.atr_20_pct,
        d.overnight_range,
        d.prev_day_range,
        d.atr_20,
        d.orb_EUROPE_FLOW_size,
        d.orb_NYSE_OPEN_size,
        d.orb_US_DATA_1000_size
    FROM orb_outcomes o
    JOIN daily_features d
      ON d.trading_day = o.trading_day
     AND d.symbol = o.symbol
     AND d.orb_minutes = o.orb_minutes
    WHERE o.symbol = 'MGC'
      AND o.orb_minutes = 5
      AND o.entry_model = 'E2'
      AND o.confirm_bars = 1
      AND o.rr_target = 1.0
      AND o.orb_label IN ('EUROPE_FLOW', 'NYSE_OPEN', 'US_DATA_1000')
      AND o.trading_day >= DATE '{OVERLAP_START}'
      {upper_clause}
      AND o.outcome IS NOT NULL
      AND o.entry_price IS NOT NULL
      AND o.stop_price IS NOT NULL
    ORDER BY o.orb_label, o.trading_day
    """
    return fetch_df(sql)


def passes_filter(row: pd.Series, family: FamilySpec) -> bool:
    if family.filter_type is None:
        return row["orb_label"] == family.orb_label
    if row["orb_label"] != family.orb_label:
        return False
    if family.filter_type == "ATR_P70":
        return float(row["atr_20_pct"]) >= 70.0
    if family.filter_type == "OVNRNG_10":
        return float(row["overnight_range"]) >= 10.0
    if family.filter_type == "OVNRNG_50":
        return float(row["overnight_range"]) >= 50.0
    if family.filter_type == "ORB_G5":
        session_col = f"orb_{family.orb_label}_size"
        return float(row[session_col]) >= 5.0
    raise ValueError(f"Unsupported filter_type: {family.filter_type}")


def target_net_r(entry_price: float, stop_price: float, target_r: float) -> float | None:
    risk_points = abs(float(entry_price) - float(stop_price))
    if risk_points <= 0:
        return None
    return round(
        to_r_multiple(MGC_SPEC, float(entry_price), float(stop_price), risk_points * target_r),
        4,
    )


def conservative_lower_target_pnl(row: pd.Series, target_r: float) -> float | None:
    actual = row["pnl_r"]
    if pd.isna(actual):
        return None
    net_target = target_net_r(row["entry_price"], row["stop_price"], target_r)
    if net_target is None:
        return None
    if bool(row["ambiguous_bar"]) and row["outcome"] == "loss":
        return float(actual)
    if pd.notna(row["mfe_r"]) and float(row["mfe_r"]) >= net_target:
        return net_target
    return float(actual)


def build_family_trade_matrix(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for family in FAMILIES:
        mask = rows.apply(lambda r, family=family: passes_filter(r, family), axis=1)
        family_rows = rows.loc[mask].copy()
        if family_rows.empty:
            continue
        family_rows["family_id"] = family.family_id
        family_rows["family_kind"] = family.kind
        family_rows["filter_type"] = family.filter_type or "NO_FILTER"
        family_rows["lower_0_5_pnl_r"] = family_rows.apply(
            lambda r: conservative_lower_target_pnl(r, 0.5),
            axis=1,
        )
        family_rows["lower_0_75_pnl_r"] = family_rows.apply(
            lambda r: conservative_lower_target_pnl(r, 0.75),
            axis=1,
        )
        family_rows["net_target_0_5_r"] = family_rows.apply(
            lambda r: target_net_r(r["entry_price"], r["stop_price"], 0.5),
            axis=1,
        )
        family_rows["net_target_0_75_r"] = family_rows.apply(
            lambda r: target_net_r(r["entry_price"], r["stop_price"], 0.75),
            axis=1,
        )
        family_rows["reached_0_5_conservative"] = (
            (~family_rows["ambiguous_bar"].fillna(False))
            & (family_rows["mfe_r"] >= family_rows["net_target_0_5_r"])
        )
        family_rows["reached_0_75_conservative"] = (
            (~family_rows["ambiguous_bar"].fillna(False))
            & (family_rows["mfe_r"] >= family_rows["net_target_0_75_r"])
        )
        records.append(family_rows)
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def summarize_families(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (family_id, family_kind, orb_label, filter_type), grp in trades.groupby(
        ["family_id", "family_kind", "orb_label", "filter_type"],
        sort=False,
    ):
        n = len(grp)
        raw = grp["pnl_r"].astype(float)
        ts = grp["ts_pnl_r"].astype(float)
        lr05 = grp["lower_0_5_pnl_r"].astype(float)
        lr075 = grp["lower_0_75_pnl_r"].astype(float)
        rows.append(
            {
                "family_id": family_id,
                "family_kind": family_kind,
                "orb_label": orb_label,
                "filter_type": filter_type,
                "n": n,
                "avg_raw_r": raw.mean(),
                "wr_raw": (raw > 0).mean(),
                "avg_ts_r": ts.mean(),
                "wr_ts": (ts > 0).mean(),
                "time_stop_rate": (grp["ts_outcome"] == "time_stop").mean(),
                "delta_ts_r": (ts - raw).mean(),
                "avg_lr05_r": lr05.mean(),
                "wr_lr05": (lr05 > 0).mean(),
                "delta_lr05_r": (lr05 - raw).mean(),
                "avg_lr075_r": lr075.mean(),
                "wr_lr075": (lr075 > 0).mean(),
                "delta_lr075_r": (lr075 - raw).mean(),
                "reach_05_rate": grp["reached_0_5_conservative"].mean(),
                "reach_075_rate": grp["reached_0_75_conservative"].mean(),
                "avg_mfe_r": grp["mfe_r"].mean(),
                "avg_mae_r": grp["mae_r"].mean(),
                "ambiguous_rate": grp["ambiguous_bar"].fillna(False).mean(),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(["family_kind", "orb_label", "family_id"]).reset_index(drop=True)


def fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def top_findings(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    warm = summary[summary["family_kind"] == "warm"].copy()
    broad = summary[summary["family_kind"] == "broad"].copy()
    warm = warm.sort_values("delta_lr05_r", ascending=False)
    broad = broad.sort_values("delta_lr05_r", ascending=False)
    return warm, broad


def classify_verdicts(
    summary: pd.DataFrame,
    *,
    all_time_stop_zero: bool,
    no_threshold_sessions: bool,
) -> dict[str, object]:
    warm, broad = top_findings(summary)
    warm_lr05_positive = int((warm["avg_lr05_r"] > 0).sum()) if not warm.empty else 0
    broad_lr05_positive = int((broad["avg_lr05_r"] > 0).sum()) if not broad.empty else 0
    best_warm_delta = float(warm["delta_lr05_r"].max()) if not warm.empty else float("nan")
    time_stop_null = bool(all_time_stop_zero and no_threshold_sessions)
    low_rr_rescue_plausible = bool(warm_lr05_positive >= 3 and math.isfinite(best_warm_delta) and best_warm_delta > 0.03)
    no_rescue_signal = bool(warm_lr05_positive == 0 and broad_lr05_positive == 0)
    return {
        "PAYOFF_COMPRESSION_REAL": bool(time_stop_null and low_rr_rescue_plausible),
        "LOW_RR_RESCUE_PLAUSIBLE": low_rr_rescue_plausible,
        "NO_RESCUE_SIGNAL": no_rescue_signal,
        "warm_lr05_positive": warm_lr05_positive,
        "broad_lr05_positive": broad_lr05_positive,
    }


def build_markdown(summary: pd.DataFrame) -> str:
    warm, broad = top_findings(summary)
    all_time_stop_zero = bool((summary["time_stop_rate"] == 0).all()) if not summary.empty else True
    no_threshold_sessions = all(
        EARLY_EXIT_MINUTES.get(label) is None
        for label in ("EUROPE_FLOW", "NYSE_OPEN", "US_DATA_1000")
    )
    verdicts = classify_verdicts(
        summary,
        all_time_stop_zero=all_time_stop_zero,
        no_threshold_sessions=no_threshold_sessions,
    )
    warm_lr05_positive = int(verdicts["warm_lr05_positive"])
    broad_lr05_positive = int(verdicts["broad_lr05_positive"])
    lines = [
        "# MGC 5-minute Payoff-Compression Audit",
        "",
        "Date: 2026-04-19",
        "",
        "## Scope",
        "",
        "Narrow diagnostic audit of the warm `GC -> MGC` translated 5-minute families.",
        "This is not a new discovery run and not a deployment memo.",
        "",
        "Locked surface:",
        "",
        "- symbol: `MGC`",
        "- overlap era only: `2022-06-13 <= trading_day < 2026-01-01`",
        "- `orb_minutes = 5`, `entry_model = E2`, `confirm_bars = 1`, `rr_target = 1.0`",
        "- sessions: `EUROPE_FLOW`, `NYSE_OPEN`, `US_DATA_1000`",
        "- diagnostics only:",
        "  - raw canonical `pnl_r`",
        "  - canonical `ts_pnl_r` time-stop",
        "  - conservative lower-target rewrites at `0.5R` and `0.75R`",
        "",
        "## Executive Verdict",
        "",
    ]

    if warm.empty:
        lines += [
            "No warm-family rows were available under the locked surface.",
            "",
        ]
    else:
        best_warm = warm.iloc[0]
        best_broad = broad.iloc[0] if not broad.empty else None
        lines += [
            "The question is no longer whether `GC` triggers translate; they do. The",
            "question is whether narrow exit handling can recover the warm `MGC` bridge.",
            "",
            "Under the locked diagnostic:",
            "",
            f"- strongest warm lower-0.5R improvement: `{best_warm['family_id']}`",
            f"  (`avg_raw_r={best_warm['avg_raw_r']:+.4f}` -> `avg_lr05_r={best_warm['avg_lr05_r']:+.4f}`)",
        ]
        if best_broad is not None:
            lines.append(
                f"- strongest broad lower-0.5R improvement: `{best_broad['family_id']}`"
                f" (`avg_raw_r={best_broad['avg_raw_r']:+.4f}` -> `avg_lr05_r={best_broad['avg_lr05_r']:+.4f}`)"
            )
        lines += [
            "",
        ]
        if all_time_stop_zero and no_threshold_sessions:
            lines += [
                "- canonical time-stop is a null lens here:",
                "  `ts_pnl_r` is identical to raw `pnl_r` across all tested families, and these",
                "  sessions currently have no configured early-exit threshold in runtime policy",
                "  (`EARLY_EXIT_MINUTES` is `None` for all three)",
                "",
            ]
        elif all_time_stop_zero:
            lines += [
                "- canonical time-stop is a null lens here:",
                "  `ts_pnl_r` is identical to raw `pnl_r` across all tested families even though",
                "  the runtime config includes thresholds for at least one tested session",
                "",
            ]
        if broad_lr05_positive >= 2:
            lines += [
                "- lower-0.5R improvement is **not** confined to the warm translated rows:",
                f"  {broad_lr05_positive}/{len(broad)} broad session comparators also flip positive",
                "  under the same conservative rewrite",
                "",
                "That means the remaining question has widened slightly:",
                "",
                "- this is still a gold-specific 5-minute payoff-compression problem",
                "- but it now looks more like a broader `MGC` target-shape issue in these sessions,",
                "  not just a narrow proxy-rescue path",
                "",
            ]
        else:
            lines += [
                "- lower-0.5R improvement remains concentrated in the warm translated rows",
                f"  ({warm_lr05_positive}/{len(warm)} warm positive vs {broad_lr05_positive}/{len(broad)} broad positive)",
                "",
            ]
        lines += [
            "Interpretation should stay disciplined:",
            "",
            "- this is not evidence that the retired `GC` shelf should be revived",
            "- this is not evidence that `MGC` is solved by one lower target",
            "- it is evidence that `RR1.0` may still be too ambitious for 5-minute `MGC` in these",
            "  sessions, including beyond the narrow translated warm rows",
            "",
        ]

    lines += [
        "## Verdict Labels",
        "",
        f"- `PAYOFF_COMPRESSION_REAL`: {'YES' if verdicts['PAYOFF_COMPRESSION_REAL'] else 'NO'}",
        f"- `LOW_RR_RESCUE_PLAUSIBLE`: {'YES' if verdicts['LOW_RR_RESCUE_PLAUSIBLE'] else 'NO'}",
        f"- `NO_RESCUE_SIGNAL`: {'YES' if verdicts['NO_RESCUE_SIGNAL'] else 'NO'}",
        "",
        "## Recommended Next Move",
        "",
    ]
    if verdicts["LOW_RR_RESCUE_PLAUSIBLE"]:
        lines += [
            "- Treat this item as actioned and closed at the diagnostic stage.",
            "- If revisited, do it as a narrow MGC 5-minute exit-shape / lower-target prereg.",
            "- Do not reopen broad GC proxy discovery or treat this as a generic gold deployment claim.",
            "",
        ]
    else:
        lines += [
            "- Treat the translation path as structurally unresolved at 5 minutes and do not reopen it without a new mechanism.",
            "",
        ]

    lines += [
        "## Family Summary",
        "",
        "| Family | Kind | N | Raw avg R | T80 avg R | 0.5R avg | 0.75R avg | reach 0.5 | reach 0.75 | time-stop | ambiguous |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.family_id} | {row.family_kind} | {int(row.n)} | "
            f"{row.avg_raw_r:+.4f} | {row.avg_ts_r:+.4f} | {row.avg_lr05_r:+.4f} | "
            f"{row.avg_lr075_r:+.4f} | {row.reach_05_rate:.1%} | {row.reach_075_rate:.1%} | "
            f"{row.time_stop_rate:.1%} | {row.ambiguous_rate:.1%} |"
        )

    lines += [
        "",
        "## Guardrails",
        "",
        "- Lower-target rewrites are diagnostic only. They are not live-ready exits.",
        "- Ambiguous loss bars are left as losses rather than rescued.",
        "- This audit does **not** say whether MGC is dead overall. It only addresses the",
        "  warm translated 5-minute families from the prior `GC -> MGC` audit.",
        "",
        "## Outputs",
        "",
        "- `research/output/mgc_payoff_compression_audit_family_summary.csv`",
        "- `research/output/mgc_payoff_compression_audit_trade_matrix.csv`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = load_rows()
    trades = build_family_trade_matrix(rows)
    if trades.empty:
        raise SystemExit("No rows matched locked MGC payoff-compression surface.")
    summary = summarize_families(trades)
    write_csv(summary, f"{OUTPUT_PREFIX}_family_summary.csv")
    write_csv(trades, f"{OUTPUT_PREFIX}_trade_matrix.csv")
    RESULT_PATH.write_text(build_markdown(summary), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nWrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
