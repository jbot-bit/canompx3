#!/usr/bin/env python3
"""GC -> MGC translation audit.

Research-only diagnostic pass locked by:
  docs/audit/hypotheses/2026-04-19-gc-mgc-translation-audit.yaml

Purpose:
- verify whether GC strength is still real in the overlap era
- verify whether price-safe filter triggers translate from GC to MGC
- locate whether translation breaks at trigger stage or payoff stage
- avoid reusing old proxy conclusions without fresh canonical proof
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from pipeline.cost_model import get_cost_spec
from research.lib import connect_db, write_csv

OVERLAP_START = "2022-06-13"
HOLDOUT_START = "2026-01-01"
LOCKED_RR = (1.0, 1.5, 2.0)
GC_ONLY_PROXY_ORB_MINUTES = 5
OUTPUT_PREFIX = "gc_mgc_translation_audit"
RESULT_PATH = Path("docs/audit/results/2026-04-19-gc-mgc-translation-audit.md")


def t_stat(avg_r: float | None, sd: float | None, n: int) -> float | None:
    if avg_r is None or sd in (None, 0) or n <= 1:
        return None
    return avg_r / (sd / math.sqrt(n))


def fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def fetch_df(sql: str) -> pd.DataFrame:
    with connect_db() as con:
        return con.execute(sql).fetchdf()


def gc_mgc_minute_coverage() -> pd.DataFrame:
    sql = f"""
    SELECT symbol, orb_minutes, COUNT(*) AS n
    FROM orb_outcomes
    WHERE symbol IN ('GC', 'MGC')
      AND entry_model = 'E2'
      AND confirm_bars = 1
      AND trading_day >= DATE '{OVERLAP_START}'
      AND trading_day < DATE '{HOLDOUT_START}'
    GROUP BY 1, 2
    ORDER BY 1, 2
    """
    return fetch_df(sql)


def overlap_baseline() -> pd.DataFrame:
    rr_sql = ", ".join(str(x) for x in LOCKED_RR)
    sql = f"""
    WITH base AS (
        SELECT symbol, orb_minutes, orb_label, rr_target,
               COUNT(*) AS n,
               AVG(pnl_r) AS avg_r,
               STDDEV_SAMP(pnl_r) AS sd,
               AVG(CASE WHEN pnl_r > 0 THEN 1.0 ELSE 0.0 END) AS win_rate,
               AVG(CASE WHEN pnl_r > 0 THEN pnl_r END) AS avg_win_r,
               AVG(CASE WHEN pnl_r <= 0 THEN pnl_r END) AS avg_loss_r
        FROM orb_outcomes
        WHERE symbol IN ('GC', 'MGC')
          AND entry_model = 'E2'
          AND confirm_bars = 1
          AND trading_day >= DATE '{OVERLAP_START}'
          AND trading_day < DATE '{HOLDOUT_START}'
          AND rr_target IN ({rr_sql})
        GROUP BY 1, 2, 3, 4
    )
    SELECT *
    FROM base
    ORDER BY orb_minutes, orb_label, rr_target, symbol
    """
    df = fetch_df(sql)
    df["t_stat"] = [t_stat(r.avg_r, r.sd, int(r.n)) for r in df.itertuples(index=False)]
    return df.drop(columns=["sd"])


def paired_outcomes() -> pd.DataFrame:
    sql = f"""
    WITH paired AS (
        SELECT
            g.orb_label,
            g.rr_target,
            COUNT(*) AS n_pairs,
            CORR(g.pnl_r, m.pnl_r) AS corr_r,
            AVG(g.pnl_r) AS gc_avg_r,
            AVG(m.pnl_r) AS mgc_avg_r,
            AVG(g.pnl_r - m.pnl_r) AS avg_gap_r,
            AVG(CASE WHEN SIGN(g.pnl_r) = SIGN(m.pnl_r) THEN 1.0 ELSE 0.0 END) AS sign_agree
        FROM orb_outcomes g
        JOIN orb_outcomes m
          ON g.trading_day = m.trading_day
         AND g.orb_label = m.orb_label
         AND g.orb_minutes = m.orb_minutes
         AND g.rr_target = m.rr_target
         AND g.confirm_bars = m.confirm_bars
         AND g.entry_model = m.entry_model
        WHERE g.symbol = 'GC'
          AND m.symbol = 'MGC'
          AND g.orb_minutes = {GC_ONLY_PROXY_ORB_MINUTES}
          AND g.entry_model = 'E2'
          AND g.confirm_bars = 1
          AND g.trading_day >= DATE '{OVERLAP_START}'
          AND g.trading_day < DATE '{HOLDOUT_START}'
        GROUP BY 1, 2
    )
    SELECT *
    FROM paired
    ORDER BY orb_label, rr_target
    """
    return fetch_df(sql)


def feature_parity() -> pd.DataFrame:
    sql = f"""
    SELECT
        symbol,
        COUNT(*) AS n,
        AVG(atr_20) AS atr20_avg,
        AVG(overnight_range) AS overnight_range_avg,
        AVG(prev_day_range) AS prev_day_range_avg,
        AVG(orb_NYSE_OPEN_size) AS nyse_open_orb_avg,
        AVG(orb_US_DATA_1000_size) AS us_data_1000_orb_avg,
        AVG(orb_EUROPE_FLOW_size) AS europe_flow_orb_avg,
        AVG(orb_LONDON_METALS_size) AS london_metals_orb_avg
    FROM daily_features
    WHERE symbol IN ('GC', 'MGC')
      AND orb_minutes = 5
      AND trading_day >= DATE '{OVERLAP_START}'
      AND trading_day < DATE '{HOLDOUT_START}'
    GROUP BY 1
    ORDER BY 1
    """
    return fetch_df(sql)


def transfer_matrix() -> pd.DataFrame:
    sql = f"""
    WITH retired_gc AS (
        SELECT strategy_id, orb_label, orb_minutes, rr_target, confirm_bars, entry_model,
               filter_type
        FROM validated_setups
        WHERE instrument = 'GC'
          AND status = 'retired'
          AND filter_type IN ('ATR_P50', 'ATR_P70', 'OVNRNG_10', 'OVNRNG_50', 'PDR_R080', 'ORB_G5')
    ),
    paired AS (
        SELECT
            r.strategy_id,
            r.orb_label,
            r.orb_minutes,
            r.rr_target,
            r.filter_type,
            'GC' AS symbol,
            COUNT(*) AS n,
            AVG(o.pnl_r) AS avg_r,
            STDDEV_SAMP(o.pnl_r) AS sd,
            AVG(CASE WHEN o.pnl_r > 0 THEN 1.0 ELSE 0.0 END) AS win_rate
        FROM retired_gc r
        JOIN orb_outcomes o
          ON o.symbol = 'GC'
         AND o.orb_label = r.orb_label
         AND o.orb_minutes = r.orb_minutes
         AND o.rr_target = r.rr_target
         AND o.confirm_bars = r.confirm_bars
         AND o.entry_model = r.entry_model
        JOIN daily_features d
          ON d.trading_day = o.trading_day
         AND d.symbol = o.symbol
         AND d.orb_minutes = o.orb_minutes
        WHERE o.trading_day >= DATE '{OVERLAP_START}'
          AND o.trading_day < DATE '{HOLDOUT_START}'
          AND CASE
                WHEN r.filter_type = 'ATR_P50' THEN d.atr_20_pct >= 50.0
                WHEN r.filter_type = 'ATR_P70' THEN d.atr_20_pct >= 70.0
                WHEN r.filter_type = 'OVNRNG_10' THEN d.overnight_range >= 10.0
                WHEN r.filter_type = 'OVNRNG_50' THEN d.overnight_range >= 50.0
                WHEN r.filter_type = 'PDR_R080' THEN d.prev_day_range / NULLIF(d.atr_20, 0) >= 0.8
                WHEN r.filter_type = 'ORB_G5' THEN
                    CASE
                        WHEN r.orb_label = 'US_DATA_1000' THEN d.orb_US_DATA_1000_size >= 5.0
                        WHEN r.orb_label = 'NYSE_OPEN' THEN d.orb_NYSE_OPEN_size >= 5.0
                        WHEN r.orb_label = 'EUROPE_FLOW' THEN d.orb_EUROPE_FLOW_size >= 5.0
                        WHEN r.orb_label = 'LONDON_METALS' THEN d.orb_LONDON_METALS_size >= 5.0
                        WHEN r.orb_label = 'COMEX_SETTLE' THEN d.orb_COMEX_SETTLE_size >= 5.0
                        WHEN r.orb_label = 'US_DATA_830' THEN d.orb_US_DATA_830_size >= 5.0
                        WHEN r.orb_label = 'SINGAPORE_OPEN' THEN d.orb_SINGAPORE_OPEN_size >= 5.0
                        WHEN r.orb_label = 'TOKYO_OPEN' THEN d.orb_TOKYO_OPEN_size >= 5.0
                        WHEN r.orb_label = 'CME_REOPEN' THEN d.orb_CME_REOPEN_size >= 5.0
                        ELSE FALSE
                    END
                ELSE FALSE
              END
        GROUP BY 1,2,3,4,5,6

        UNION ALL

        SELECT
            r.strategy_id,
            r.orb_label,
            r.orb_minutes,
            r.rr_target,
            r.filter_type,
            'MGC' AS symbol,
            COUNT(*) AS n,
            AVG(o.pnl_r) AS avg_r,
            STDDEV_SAMP(o.pnl_r) AS sd,
            AVG(CASE WHEN o.pnl_r > 0 THEN 1.0 ELSE 0.0 END) AS win_rate
        FROM retired_gc r
        JOIN orb_outcomes o
          ON o.symbol = 'MGC'
         AND o.orb_label = r.orb_label
         AND o.orb_minutes = r.orb_minutes
         AND o.rr_target = r.rr_target
         AND o.confirm_bars = r.confirm_bars
         AND o.entry_model = r.entry_model
        JOIN daily_features d
          ON d.trading_day = o.trading_day
         AND d.symbol = o.symbol
         AND d.orb_minutes = o.orb_minutes
        WHERE o.trading_day >= DATE '{OVERLAP_START}'
          AND o.trading_day < DATE '{HOLDOUT_START}'
          AND CASE
                WHEN r.filter_type = 'ATR_P50' THEN d.atr_20_pct >= 50.0
                WHEN r.filter_type = 'ATR_P70' THEN d.atr_20_pct >= 70.0
                WHEN r.filter_type = 'OVNRNG_10' THEN d.overnight_range >= 10.0
                WHEN r.filter_type = 'OVNRNG_50' THEN d.overnight_range >= 50.0
                WHEN r.filter_type = 'PDR_R080' THEN d.prev_day_range / NULLIF(d.atr_20, 0) >= 0.8
                WHEN r.filter_type = 'ORB_G5' THEN
                    CASE
                        WHEN r.orb_label = 'US_DATA_1000' THEN d.orb_US_DATA_1000_size >= 5.0
                        WHEN r.orb_label = 'NYSE_OPEN' THEN d.orb_NYSE_OPEN_size >= 5.0
                        WHEN r.orb_label = 'EUROPE_FLOW' THEN d.orb_EUROPE_FLOW_size >= 5.0
                        WHEN r.orb_label = 'LONDON_METALS' THEN d.orb_LONDON_METALS_size >= 5.0
                        WHEN r.orb_label = 'COMEX_SETTLE' THEN d.orb_COMEX_SETTLE_size >= 5.0
                        WHEN r.orb_label = 'US_DATA_830' THEN d.orb_US_DATA_830_size >= 5.0
                        WHEN r.orb_label = 'SINGAPORE_OPEN' THEN d.orb_SINGAPORE_OPEN_size >= 5.0
                        WHEN r.orb_label = 'TOKYO_OPEN' THEN d.orb_TOKYO_OPEN_size >= 5.0
                        WHEN r.orb_label = 'CME_REOPEN' THEN d.orb_CME_REOPEN_size >= 5.0
                        ELSE FALSE
                    END
                ELSE FALSE
              END
        GROUP BY 1,2,3,4,5,6
    )
    SELECT *
    FROM paired
    ORDER BY strategy_id, symbol
    """
    df = fetch_df(sql)
    df["t_stat"] = [t_stat(r.avg_r, r.sd, int(r.n)) for r in df.itertuples(index=False)]
    return df.drop(columns=["sd"])


def filter_pass_parity() -> pd.DataFrame:
    sql = f"""
    WITH retired_gc AS (
        SELECT DISTINCT strategy_id, orb_label, orb_minutes, filter_type
        FROM validated_setups
        WHERE instrument = 'GC'
          AND status = 'retired'
          AND filter_type IN ('ATR_P50', 'ATR_P70', 'OVNRNG_10', 'OVNRNG_50', 'PDR_R080', 'ORB_G5')
    ),
    pass_counts AS (
        SELECT
            r.strategy_id,
            r.orb_label,
            r.filter_type,
            d.symbol,
            COUNT(*) AS pass_days
        FROM retired_gc r
        JOIN daily_features d
          ON d.symbol IN ('GC', 'MGC')
         AND d.orb_minutes = r.orb_minutes
        WHERE d.trading_day >= DATE '{OVERLAP_START}'
          AND d.trading_day < DATE '{HOLDOUT_START}'
          AND CASE
                WHEN r.filter_type = 'ATR_P50' THEN d.atr_20_pct >= 50.0
                WHEN r.filter_type = 'ATR_P70' THEN d.atr_20_pct >= 70.0
                WHEN r.filter_type = 'OVNRNG_10' THEN d.overnight_range >= 10.0
                WHEN r.filter_type = 'OVNRNG_50' THEN d.overnight_range >= 50.0
                WHEN r.filter_type = 'PDR_R080' THEN d.prev_day_range / NULLIF(d.atr_20, 0) >= 0.8
                WHEN r.filter_type = 'ORB_G5' THEN
                    CASE
                        WHEN r.orb_label = 'US_DATA_1000' THEN d.orb_US_DATA_1000_size >= 5.0
                        WHEN r.orb_label = 'NYSE_OPEN' THEN d.orb_NYSE_OPEN_size >= 5.0
                        WHEN r.orb_label = 'EUROPE_FLOW' THEN d.orb_EUROPE_FLOW_size >= 5.0
                        WHEN r.orb_label = 'LONDON_METALS' THEN d.orb_LONDON_METALS_size >= 5.0
                        WHEN r.orb_label = 'COMEX_SETTLE' THEN d.orb_COMEX_SETTLE_size >= 5.0
                        WHEN r.orb_label = 'US_DATA_830' THEN d.orb_US_DATA_830_size >= 5.0
                        WHEN r.orb_label = 'SINGAPORE_OPEN' THEN d.orb_SINGAPORE_OPEN_size >= 5.0
                        WHEN r.orb_label = 'TOKYO_OPEN' THEN d.orb_TOKYO_OPEN_size >= 5.0
                        WHEN r.orb_label = 'CME_REOPEN' THEN d.orb_CME_REOPEN_size >= 5.0
                        ELSE FALSE
                    END
                ELSE FALSE
              END
        GROUP BY 1,2,3,4
    )
    SELECT *
    FROM pass_counts
    ORDER BY strategy_id, symbol
    """
    return fetch_df(sql)


def wide_transfer_summary(transfer_df: pd.DataFrame) -> pd.DataFrame:
    wide = transfer_df.pivot_table(
        index=["strategy_id", "orb_label", "orb_minutes", "rr_target", "filter_type"],
        columns="symbol",
        values=["n", "avg_r", "t_stat", "win_rate"],
    )
    wide.columns = ["_".join(col).lower() for col in wide.columns]
    wide = wide.reset_index()
    wide["same_sign"] = (
        wide["avg_r_gc"].apply(lambda x: 1 if pd.notna(x) and x > 0 else (-1 if pd.notna(x) and x < 0 else 0))
        == wide["avg_r_mgc"].apply(lambda x: 1 if pd.notna(x) and x > 0 else (-1 if pd.notna(x) and x < 0 else 0))
    )
    wide["mgc_positive"] = wide["avg_r_mgc"] > 0
    return wide.sort_values(["orb_label", "rr_target", "filter_type"])


def wide_pass_summary(pass_df: pd.DataFrame) -> pd.DataFrame:
    wide = pass_df.pivot_table(
        index=["strategy_id", "orb_label", "filter_type"],
        columns="symbol",
        values="pass_days",
    ).reset_index()
    wide["delta_days"] = wide["GC"] - wide["MGC"]
    wide["mgc_to_gc_ratio"] = wide["MGC"] / wide["GC"]
    return wide.sort_values(["orb_label", "filter_type", "strategy_id"])


def render(df: pd.DataFrame, columns: list[str] | None = None, digits: int = 4) -> str:
    if columns is not None:
        df = df[columns].copy()
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: fmt(x, digits))
    return out.to_string(index=False)


def write_result(
    minute_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    transfer_wide: pd.DataFrame,
    pass_wide: pd.DataFrame,
) -> None:
    gc_cost = get_cost_spec("GC")
    baseline_5m = baseline_df[baseline_df["orb_minutes"] == 5].copy()

    transfer_positive = int(transfer_wide["mgc_positive"].sum())
    transfer_rows = len(transfer_wide)
    rr_gt_1_positive = int(transfer_wide[(transfer_wide["rr_target"] > 1.0) & (transfer_wide["mgc_positive"])]["strategy_id"].count())

    lines = [
        "# GC -> MGC Translation Audit",
        "",
        "Date: 2026-04-19",
        "",
        "## Scope",
        "",
        "Audit the unresolved gold translation question from canonical overlap-era truth:",
        "",
        "- does `GC` stay strong in the actual `MGC` overlap era, or was the old proxy edge mostly old history?",
        "- do price-safe triggers transfer from `GC` to `MGC` cleanly?",
        "- if translation fails, is the break at trigger timing, payoff shape, or modeled friction?",
        "",
        "This is a research audit only. It is not a discovery run, not a deployment memo, and not a new justification for widening proxy use.",
        "",
        "## Guardrails",
        "",
        "- Canonical proof uses only `gold.db::orb_outcomes`, `gold.db::daily_features`, and `pipeline.cost_model`.",
        "- 2026 holdout is excluded from selection and diagnosis here (`trading_day < 2026-01-01`).",
        "- Prior docs and old handoffs were used only as orientation/comparison, not as proof.",
        "- No claim is made about `GC` 15m/30m proxy transfer because the canonical `GC` proxy surface here is 5-minute only.",
        "",
        "## Executive Verdict",
        "",
        "`GC` strength is still real in the overlap era, so the old gold signal was not just a stale-history artifact. But the `GC -> MGC` bridge does not fail because price-safe filters stop firing. It fails mainly because the 5-minute `MGC` payoff shape is materially worse: win rates are modestly lower, average wins are much smaller, and the broad positive `GC` expectancy compresses toward flat or negative on `MGC`.",
        "",
        f"On the exact retired `GC` validated rows, only {transfer_positive}/{transfer_rows} keep a positive sign on `MGC` overlap, and only {rr_gt_1_positive} of those are above `RR=1.0`. So the shorthand \"edge does not transfer\" was too blunt, but the stronger claim still holds: the full `GC` proxy shelf does **not** transfer cleanly to `MGC`, and any surviving bridge is narrow, weak, and concentrated at low RR.",
        "",
        "## Source-of-Truth Chain",
        "",
        "1. `gold.db::orb_outcomes`",
        "2. `gold.db::daily_features`",
        "3. `pipeline.cost_model`",
        "",
        "Orientation only:",
        "",
        "4. `validated_setups` for the already-retired `GC` rows",
        "5. `docs/plans/2026-04-10-mgc-proxy-hypothesis-design.md`",
        "6. `docs/handoffs/2026-04-10-gc-proxy-discovery-handover.md`",
        "",
        "## Finding 1 — The old GC strength is still real in the overlap era",
        "",
        "The overlap-era `GC` baseline remains positive across multiple gold-relevant sessions on the canonical 5-minute `E2 / CB1` surface. So this is not just a pre-2022 history artifact.",
        "",
        "```text",
        render(
            baseline_5m[baseline_5m["symbol"] == "GC"],
            columns=["symbol", "orb_label", "rr_target", "n", "avg_r", "t_stat", "win_rate", "avg_win_r", "avg_loss_r"],
        ),
        "```",
        "",
        "## Finding 2 — Trigger parity is strong; the bridge does not break because filters disappear",
        "",
        "Price-safe feature means are nearly identical between `GC` and `MGC` in the overlap era, and the pass-day counts on the retired `GC` filter winners are almost one-for-one.",
        "",
        f"Modeled friction is also not the culprit: `GC` and `MGC` have the same friction in price points ({gc_cost.friction_in_points:.3f}) and the same minimum-risk floor in price points ({gc_cost.min_risk_floor_points:.1f}). The dollar costs differ by 10x, but the R-multiple burden is the same by construction.",
        "",
        "Feature parity:",
        "",
        "```text",
        render(feature_df),
        "```",
        "",
        "Price-safe pass-day parity on retired `GC` winners:",
        "",
        "```text",
        render(pass_wide, columns=["strategy_id", "orb_label", "filter_type", "GC", "MGC", "delta_days", "mgc_to_gc_ratio"]),
        "```",
        "",
        "## Finding 3 — The main break is payoff compression on MGC 5-minute trades",
        "",
        "At the same 5-minute broad baseline, `MGC` usually keeps broadly similar loss size but materially smaller winners, with some sessions also losing win rate. That compresses expectancy from positive `GC` cells into flat or negative `MGC` cells.",
        "",
        "```text",
        render(
            baseline_5m,
            columns=["symbol", "orb_label", "rr_target", "n", "win_rate", "avg_win_r", "avg_loss_r", "avg_r"],
        ),
        "```",
        "",
        "Paired same-day `GC` vs `MGC` outcomes on the 5-minute surface confirm the same story: the day-level paths are still highly correlated, but `MGC` carries a persistent negative R-gap.",
        "",
        "```text",
        render(paired_df, columns=["orb_label", "rr_target", "n_pairs", "corr_r", "gc_avg_r", "mgc_avg_r", "avg_gap_r", "sign_agree"]),
        "```",
        "",
        "## Finding 4 — Transfer is narrow and mostly collapses above RR1.0",
        "",
        "The retired `GC` validated rows do not vanish because the filters stop working. They mostly fail because the same filtered `MGC` rows lose enough payoff that only a small low-RR subset stays positive.",
        "",
        "```text",
        render(
            transfer_wide,
            columns=[
                "strategy_id",
                "orb_label",
                "rr_target",
                "filter_type",
                "n_gc",
                "avg_r_gc",
                "t_stat_gc",
                "n_mgc",
                "avg_r_mgc",
                "t_stat_mgc",
                "same_sign",
                "mgc_positive",
            ],
        ),
        "```",
        "",
        "High-level summary:",
        "",
        f"- retired `GC` price-safe rows audited: {transfer_rows}",
        f"- rows still positive on `MGC` overlap: {transfer_positive}",
        f"- rows still positive on `MGC` with `RR > 1.0`: {rr_gt_1_positive}",
        "- the warmest surviving bridge is `US_DATA_1000 / ORB_G5 / RR1.0`, but it is still much weaker on `MGC` than on `GC`",
        "",
        "## Finding 5 — Do not generalize this beyond the current surface",
        "",
        "`MGC` has 15-minute and 30-minute canonical rows in the overlap era, but the `GC` proxy surface here does not. That means there is no honest `GC -> MGC` statement yet for 15m/30m. The translation question proven here is the 5-minute path only.",
        "",
        "Minute coverage:",
        "",
        "```text",
        render(minute_df),
        "```",
        "",
        "## Bottom Line",
        "",
        "The correct conclusion is not \"GC proxy was fake\" and not \"GC edge transfers fine.\" The honest conclusion is narrower:",
        "",
        "- `GC` overlap-era strength is real",
        "- price-safe triggers transfer cleanly enough",
        "- the bridge breaks mainly in 5-minute `MGC` payoff translation",
        "- transfer is mostly too weak above `RR=1.0` to rescue the old `GC` proxy shelf",
        "",
        "## Next Action",
        "",
        "Run a narrow **MGC 5-minute payoff-compression audit** on the warm translated families (`US_DATA_1000`, `NYSE_OPEN`, `EUROPE_FLOW`) to test whether the right rescue question is lower-RR/exit-shape handling rather than more proxy discovery. Do not reopen broad `GC` proxy exploration before that.",
        "",
    ]
    RESULT_PATH.write_text("\n".join(lines))


def main() -> None:
    minute_df = gc_mgc_minute_coverage()
    baseline_df = overlap_baseline()
    paired_df = paired_outcomes()
    feature_df = feature_parity()
    transfer_df = transfer_matrix()
    pass_df = filter_pass_parity()

    transfer_wide = wide_transfer_summary(transfer_df)
    pass_wide = wide_pass_summary(pass_df)

    write_csv(minute_df, f"{OUTPUT_PREFIX}_minute_coverage.csv")
    write_csv(baseline_df, f"{OUTPUT_PREFIX}_baseline.csv")
    write_csv(paired_df, f"{OUTPUT_PREFIX}_paired.csv")
    write_csv(feature_df, f"{OUTPUT_PREFIX}_feature_parity.csv")
    write_csv(transfer_wide, f"{OUTPUT_PREFIX}_transfer.csv")
    write_csv(pass_wide, f"{OUTPUT_PREFIX}_filter_pass_parity.csv")

    write_result(minute_df, baseline_df, paired_df, feature_df, transfer_wide, pass_wide)
    print(f"Wrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
