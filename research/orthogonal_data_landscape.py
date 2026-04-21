"""Phase 0 data landscape for the orthogonal golden-egg hunt.

Generates:
- docs/audit/2026-04-21-data-landscape.md
- outputs/negative_space_heatmap.csv

Canonical truth inputs:
- bars_1m
- daily_features
- orb_outcomes

Posture-only inputs:
- active_validated_setups (for occupied-surface fence)
- active profile live-six lanes (for negative-space exclusion only)
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import duckdb

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from trading_app.config import ALL_FILTERS, CostRatioFilter, OrbSizeFilter, OwnATRPercentileFilter
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
from trading_app.prop_profiles import ACCOUNT_PROFILES, effective_daily_lanes


ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = ROOT / "docs" / "audit" / "2026-04-21-data-landscape.md"
HEATMAP_PATH = ROOT / "outputs" / "negative_space_heatmap.csv"
CANONICAL_DB = Path("/mnt/c/Users/joshd/canompx3/gold.db")


def query_df(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None):
    return con.execute(sql, params or []).fetchdf()


def get_live_profile_lanes() -> list[str]:
    active_profiles = [p for p in ACCOUNT_PROFILES.values() if p.active]
    if not active_profiles:
        raise RuntimeError("No active profiles found.")
    # Current runtime has a single active profile; union all active profiles for safety.
    lane_ids: list[str] = []
    for profile in active_profiles:
        lane_ids.extend(l.strategy_id for l in effective_daily_lanes(profile))
    return sorted(set(lane_ids))


def live_six_filter_columns(con: duckdb.DuckDBPyConnection, lane_ids: list[str]) -> tuple[list[dict], set[str]]:
    placeholders = ",".join("?" for _ in lane_ids)
    rows = con.execute(
        f"""
        select strategy_id, instrument, orb_label, orb_minutes, entry_model, rr_target, filter_type
        from active_validated_setups
        where strategy_id in ({placeholders})
        order by strategy_id
        """,
        lane_ids,
    ).fetchall()
    used_cols: set[str] = {"symbol", "atr_20_pct"}
    specs: list[dict] = []
    for strategy_id, instrument, orb_label, orb_minutes, entry_model, rr_target, filter_type in rows:
        filt = ALL_FILTERS[filter_type]
        columns: list[str]
        if isinstance(filt, (OrbSizeFilter, CostRatioFilter)):
            columns = [f"orb_{orb_label}_size", "symbol"]
            used_cols.add(f"orb_{orb_label}_size")
            used_cols.add("symbol")
        elif isinstance(filt, OwnATRPercentileFilter):
            columns = ["atr_20_pct"]
            used_cols.add("atr_20_pct")
        else:
            columns = []
        specs.append(
            {
                "strategy_id": strategy_id,
                "instrument": instrument,
                "session": orb_label,
                "orb_minutes": orb_minutes,
                "entry_model": entry_model,
                "rr_target": rr_target,
                "filter_type": filter_type,
                "filter_class": type(filt).__name__,
                "feature_columns": columns,
            }
        )
    return specs, used_cols


def markdown_table_from_df(df, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    trimmed = df.head(max_rows)
    cols = list(trimmed.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in trimmed.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    con = duckdb.connect(str(CANONICAL_DB), read_only=True)
    instruments = sorted(ACTIVE_ORB_INSTRUMENTS)
    instrument_list = ",".join(f"'{x}'" for x in instruments)

    live_lane_ids = get_live_profile_lanes()
    live_specs, used_feature_cols = live_six_filter_columns(con, live_lane_ids)
    live_pairs = {(s["instrument"], s["session"]) for s in live_specs}

    bars_monthly = query_df(
        con,
        f"""
        select symbol, year(ts_utc) as year, month(ts_utc) as month, count(*) as row_count
        from bars_1m
        where symbol in ({instrument_list})
        group by 1, 2, 3
        order by 1, 2, 3
        """,
    )
    daily_monthly = query_df(
        con,
        f"""
        select symbol, year(trading_day) as year, month(trading_day) as month, count(*) as row_count
        from daily_features
        where symbol in ({instrument_list})
        group by 1, 2, 3
        order by 1, 2, 3
        """,
    )
    outcomes_monthly = query_df(
        con,
        f"""
        select symbol, year(trading_day) as year, month(trading_day) as month, count(*) as row_count
        from orb_outcomes
        where symbol in ({instrument_list})
        group by 1, 2, 3
        order by 1, 2, 3
        """,
    )

    monthly_stats = {
        "bars_1m": query_df(
            con,
            f"""
            with monthly as (
              select symbol, year(ts_utc) as year, month(ts_utc) as month, count(*) as row_count
              from bars_1m
              where symbol in ({instrument_list})
              group by 1, 2, 3
            )
            select symbol, min(row_count) as min_rows, median(row_count) as median_rows, max(row_count) as max_rows
            from monthly
            group by 1
            order by 1
            """,
        ),
        "daily_features": query_df(
            con,
            f"""
            with monthly as (
              select symbol, year(trading_day) as year, month(trading_day) as month, count(*) as row_count
              from daily_features
              where symbol in ({instrument_list})
              group by 1, 2, 3
            )
            select symbol, min(row_count) as min_rows, median(row_count) as median_rows, max(row_count) as max_rows
            from monthly
            group by 1
            order by 1
            """,
        ),
        "orb_outcomes": query_df(
            con,
            f"""
            with monthly as (
              select symbol, year(trading_day) as year, month(trading_day) as month, count(*) as row_count
              from orb_outcomes
              where symbol in ({instrument_list})
              group by 1, 2, 3
            )
            select symbol, min(row_count) as min_rows, median(row_count) as median_rows, max(row_count) as max_rows
            from monthly
            group by 1
            order by 1
            """,
        ),
    }

    feature_df = query_df(con, "describe daily_features")
    feature_df["used_in_live_six"] = feature_df["column_name"].isin(sorted(used_feature_cols))
    feature_df["status"] = feature_df["used_in_live_six"].map(lambda x: "used_in_live_six" if x else "canonical_but_unused")

    session_counts: list[dict] = []
    for instrument in instruments:
        total_days = con.execute(
            "select count(*) from daily_features where symbol = ? and trading_day < ?",
            [instrument, HOLDOUT_SACRED_FROM],
        ).fetchone()[0]
        for session in sorted({s["session"] for s in live_specs} | {"BRISBANE_1025", "CME_PRECLOSE", "CME_REOPEN", "COMEX_SETTLE", "EUROPE_FLOW", "LONDON_METALS", "NYSE_CLOSE", "NYSE_OPEN", "SINGAPORE_OPEN", "TOKYO_OPEN", "US_DATA_1000", "US_DATA_830"}):
            col = f"orb_{session}_size"
            if col not in set(feature_df["column_name"]):
                continue
            n_days = con.execute(
                f"""
                select count(*)
                from daily_features
                where symbol = ? and trading_day < ? and {col} is not null
                """,
                [instrument, HOLDOUT_SACRED_FROM],
            ).fetchone()[0]
            coverage = 0.0 if total_days == 0 else (100.0 * n_days / total_days)
            session_counts.append(
                {
                    "instrument": instrument,
                    "session": session,
                    "trade_days_pre_holdout": n_days,
                    "coverage_pct": round(coverage, 2),
                }
            )
    session_coverage_df = duckdb.from_df(__import__("pandas").DataFrame(session_counts)).df().sort_values(
        ["instrument", "trade_days_pre_holdout", "session"], ascending=[True, False, True]
    )

    holdout_budget_df = query_df(
        con,
        f"""
        select symbol as instrument, orb_label as session, count(distinct trading_day) as trade_days_pre_holdout
        from orb_outcomes
        where symbol in ({instrument_list})
          and trading_day < ?
          and outcome in ('win', 'loss')
        group by 1, 2
        order by 1, 3 desc, 2
        """,
        [HOLDOUT_SACRED_FROM],
    )

    regime_columns = [
        "atr_20",
        "atr_20_pct",
        "atr_vel_ratio",
        "atr_vel_regime",
        "day_of_week",
        "gap_open_points",
        "overnight_range_pct",
        "pit_range_atr",
        "garch_forecast_vol_pct",
        "prev_week_range",
        "prev_month_range",
        "is_friday",
        "is_monday",
        "is_tuesday",
        "is_nfp_day",
        "is_opex_day",
    ]
    available_regime_cols = [c for c in regime_columns if c in set(feature_df["column_name"])]

    heatmap_rows: list[dict] = []
    for instrument in instruments:
        for session in sorted({row["session"] for row in live_specs} | set(holdout_budget_df["session"].unique())):
            if (instrument, session) in live_pairs:
                continue
            col = f"orb_{session}_size"
            if col not in set(feature_df["column_name"]):
                continue
            rows = con.execute(
                f"""
                select
                  ? as instrument,
                  ? as session,
                  day_of_week,
                  coalesce(atr_vel_regime, 'UNKNOWN') as vol_regime,
                  count(*) as day_count
                from daily_features
                where symbol = ?
                  and trading_day < ?
                  and {col} is not null
                group by 1, 2, 3, 4
                having count(*) > 0
                order by day_count desc
                """,
                [instrument, session, instrument, HOLDOUT_SACRED_FROM],
            ).fetchall()
            for inst, sess, dow, vol_regime, day_count in rows:
                heatmap_rows.append(
                    {
                        "instrument": inst,
                        "session": sess,
                        "day_of_week": dow,
                        "vol_regime": vol_regime,
                        "day_count": day_count,
                    }
                )

    heatmap_rows.sort(key=lambda x: (-x["day_count"], x["instrument"], x["session"], x["day_of_week"], x["vol_regime"]))
    HEATMAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HEATMAP_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["instrument", "session", "day_of_week", "vol_regime", "day_count"])
        writer.writeheader()
        writer.writerows(heatmap_rows)

    top_heatmap = heatmap_rows[:20]
    top_heatmap_df = duckdb.from_df(__import__("pandas").DataFrame(top_heatmap)).df() if top_heatmap else __import__("pandas").DataFrame()

    bars_summary = query_df(
        con,
        f"""
        select symbol, min(ts_utc) as first_bar, max(ts_utc) as last_bar, count(*) as bar_rows
        from bars_1m
        where symbol in ({instrument_list})
        group by 1
        order by 1
        """
    )
    daily_summary = query_df(
        con,
        f"""
        select symbol, min(trading_day) as first_day, max(trading_day) as last_day, count(*) as feature_rows
        from daily_features
        where symbol in ({instrument_list})
        group by 1
        order by 1
        """
    )
    outcomes_summary = query_df(
        con,
        f"""
        select symbol, min(trading_day) as first_day, max(trading_day) as last_day, count(*) as outcome_rows
        from orb_outcomes
        where symbol in ({instrument_list})
        group by 1
        order by 1
        """
    )

    used_counter = Counter()
    for spec in live_specs:
        for col in spec["feature_columns"]:
            used_counter[col] += 1

    md = []
    md.append("# 2026-04-21 Data Landscape")
    md.append("")
    md.append("Scope: Phase 0 of the orthogonal canonical golden-egg hunt.")
    md.append("")
    md.append("Canonical truth inputs:")
    md.append("- `bars_1m`")
    md.append("- `daily_features`")
    md.append("- `orb_outcomes`")
    md.append("")
    md.append("Posture-only inputs:")
    md.append("- `active_validated_setups` for occupied-surface fencing")
    md.append("- active profile live-six lane IDs for negative-space exclusion")
    md.append("")
    md.append(f"Data mode: read-only DuckDB against `{CANONICAL_DB}`")
    md.append(f"Holdout fence: `{HOLDOUT_SACRED_FROM}`")
    md.append("")
    md.append("## 0.1 Shape and Coverage")
    md.append("")
    md.append("### Table summaries")
    md.append("")
    md.append("#### bars_1m")
    md.append(markdown_table_from_df(bars_summary))
    md.append("")
    md.append("#### daily_features")
    md.append(markdown_table_from_df(daily_summary))
    md.append("")
    md.append("#### orb_outcomes")
    md.append(markdown_table_from_df(outcomes_summary))
    md.append("")
    md.append("### Monthly density samples")
    md.append("")
    md.append("#### bars_1m monthly summary")
    md.append(markdown_table_from_df(monthly_stats["bars_1m"]))
    md.append("")
    md.append("#### daily_features monthly summary")
    md.append(markdown_table_from_df(monthly_stats["daily_features"]))
    md.append("")
    md.append("#### orb_outcomes monthly summary")
    md.append(markdown_table_from_df(monthly_stats["orb_outcomes"]))
    md.append("")
    md.append("## 0.2 Feature Inventory")
    md.append("")
    md.append(f"Daily feature column count: `{len(feature_df)}`")
    md.append(f"Used-in-live-six feature columns: `{len(used_feature_cols)}`")
    md.append("")
    md.append("Used-in-live-six columns derived from current live filter implementations:")
    for col, count in sorted(used_counter.items()):
        md.append(f"- `{col}` referenced by `{count}` live-six lane filter(s)")
    md.append("")
    md.append("Feature inventory sample:")
    md.append(markdown_table_from_df(feature_df[["column_name", "column_type", "status"]], max_rows=40))
    md.append("")
    md.append("## 0.3 Session Coverage Map")
    md.append("")
    md.append("Coverage uses non-null `daily_features.orb_{SESSION}_size` before the holdout fence as the session-availability proxy.")
    md.append(markdown_table_from_df(session_coverage_df, max_rows=36))
    md.append("")
    md.append("## 0.4 Regime Descriptors Available")
    md.append("")
    for col in available_regime_cols:
        md.append(f"- `{col}`")
    md.append("")
    md.append("## 0.5 Holdout-Window Descriptor")
    md.append("")
    md.append("Distinct trade days before the sacred holdout, from `orb_outcomes` trades only:")
    md.append(markdown_table_from_df(holdout_budget_df, max_rows=36))
    md.append("")
    md.append("## 0.6 Negative-Space Heatmap")
    md.append("")
    md.append("Heatmap file: `outputs/negative_space_heatmap.csv`")
    md.append("")
    md.append("Top uncovered cells by pre-holdout day count:")
    md.append(markdown_table_from_df(top_heatmap_df, max_rows=20))
    md.append("")
    md.append("## Live-Six Filter Spec")
    md.append("")
    md.append("Current live-six filter columns inferred from code and active lane IDs:")
    for spec in live_specs:
        md.append(f"- `{spec['strategy_id']}` → `{spec['filter_type']}` / `{spec['filter_class']}` / columns `{', '.join(spec['feature_columns']) or 'none-resolved'}`")
    md.append("")

    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "doc_path": str(DOC_PATH),
                "heatmap_path": str(HEATMAP_PATH),
                "live_lane_count": len(live_specs),
                "used_feature_columns": sorted(used_feature_cols),
                "negative_space_rows": len(heatmap_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
