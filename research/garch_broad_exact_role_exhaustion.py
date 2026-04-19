"""Broad exact-filter garch role exhaustion.

Source-wide scope:
  - validated_setups
  - experimental_strategies

Included filter families only when exact semantics are available from
trading_app/config.py and daily_features:
  ORB_G*, ORB_G*_NOFRI, ATR_P*, COST_LT*, OVNRNG_*,
  X_MES_ATR60, VWAP_MID_ALIGNED, VWAP_BP_ALIGNED, PDR_R080, GAP_R015

Excluded on purpose:
  - GARCH_* self-referential filters
  - NO_FILTER (not a filter family)
  - CROSS_*_MOMENTUM and other composite/context filters
  - filters whose exact semantics are not trivial to encode

Pre-committed garch roles tested:
  - HIGH tail: >=60, >=70, >=80
  - LOW tail:  <=40, <=30, <=20
"""

from __future__ import annotations

import io
import re
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
from scipy import stats

from pipeline.cost_model import COST_SPECS
from pipeline.paths import GOLD_DB_PATH

OUTPUT_MD = Path("docs/audit/results/2026-04-16-garch-broad-exact-role-exhaustion.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

IS_END = "2026-01-01"
SEED = 20260416
MIN_TOTAL = 50
MIN_SIDE = 10
PERMUTATIONS = 1000


@dataclass(frozen=True)
class ThresholdSpec:
    side: str
    threshold: int


SPECS = [ThresholdSpec("high", t) for t in (60, 70, 80)] + [ThresholdSpec("low", t) for t in (40, 30, 20)]

ORB_RE = re.compile(r"^ORB_G(\d+)(?:_NOFRI)?$")
ATR_RE = re.compile(r"^ATR_P(\d+)$")
COST_RE = re.compile(r"^COST_LT(\d+)$")
OVN_RE = re.compile(r"^OVNRNG_(\d+)$")
PDR_RE = re.compile(r"^PDR_R(\d{3})$")
GAP_RE = re.compile(r"^GAP_R(\d{3})$")


def exact_filter_sql(filter_type: str, orb_label: str, instrument: str) -> tuple[str | None, str]:
    """Return (predicate_sql, join_sql_suffix)."""
    match = ORB_RE.match(filter_type)
    if match:
        min_size = float(match.group(1))
        sql = f"d.orb_{orb_label}_size >= {min_size:.1f}"
        if filter_type.endswith("_NOFRI"):
            sql = f"({sql} AND NOT d.is_friday)"
        return sql, ""

    match = ATR_RE.match(filter_type)
    if match:
        return f"d.atr_20_pct >= {float(match.group(1)):.1f}", ""

    match = COST_RE.match(filter_type)
    if match:
        pct = float(match.group(1))
        if instrument not in COST_SPECS:
            return None, ""
        cost_spec = COST_SPECS[instrument]
        # Canonical semantics from config.py: orb_size * point_value.
        return (
            f"(100.0 * {cost_spec.total_friction:.8f} / "
            f"NULLIF((d.orb_{orb_label}_size * {cost_spec.point_value:.8f}) + {cost_spec.total_friction:.8f}, 0)) < {pct:.1f}"
        ), ""

    match = OVN_RE.match(filter_type)
    if match:
        return f"d.overnight_range >= {float(match.group(1)):.1f}", ""

    if filter_type == "X_MES_ATR60":
        join = """
        JOIN daily_features mes
          ON o.trading_day = mes.trading_day
         AND mes.symbol = 'MES'
         AND mes.orb_minutes = o.orb_minutes
        """
        return "mes.atr_20_pct >= 60.0", join

    if filter_type == "VWAP_MID_ALIGNED":
        return (
            f"(((d.orb_{orb_label}_break_dir='long') AND "
            f"(d.orb_{orb_label}_high + d.orb_{orb_label}_low)/2.0 > d.orb_{orb_label}_vwap) "
            f"OR ((d.orb_{orb_label}_break_dir='short') AND "
            f"(d.orb_{orb_label}_high + d.orb_{orb_label}_low)/2.0 < d.orb_{orb_label}_vwap))"
        ), ""

    if filter_type == "VWAP_BP_ALIGNED":
        return (
            f"(((d.orb_{orb_label}_break_dir='long') AND d.orb_{orb_label}_high > d.orb_{orb_label}_vwap) "
            f"OR ((d.orb_{orb_label}_break_dir='short') AND d.orb_{orb_label}_low < d.orb_{orb_label}_vwap))"
        ), ""

    match = PDR_RE.match(filter_type)
    if match:
        ratio = float(match.group(1)) / 100.0
        return f"(d.prev_day_range / NULLIF(d.atr_20, 0)) >= {ratio:.4f}", ""

    match = GAP_RE.match(filter_type)
    if match:
        ratio = float(match.group(1)) / 100.0
        return f"(ABS(d.gap_open_points) / NULLIF(d.atr_20, 0)) >= {ratio:.4f}", ""

    return None, ""


def load_rows(con) -> pd.DataFrame:
    return con.execute(
        """
        SELECT 'validated' AS src, strategy_id, instrument, orb_label, orb_minutes,
               rr_target, entry_model, filter_type, sample_size
        FROM validated_setups
        UNION ALL
        SELECT 'experimental' AS src, strategy_id, instrument, orb_label, orb_minutes,
               rr_target, entry_model, filter_type, sample_size
        FROM experimental_strategies
        """
    ).df()


def in_scope(filter_type: str) -> bool:
    if filter_type.startswith("GARCH_"):
        return False
    if filter_type in {"NO_FILTER", "CROSS_SGP_MOMENTUM", "CROSS_NYSE_MOMENTUM", "CROSS_COMEX_MOMENTUM"}:
        return False
    if ORB_RE.match(filter_type):
        return True
    if ATR_RE.match(filter_type):
        return True
    if COST_RE.match(filter_type):
        return True
    if OVN_RE.match(filter_type):
        return True
    if filter_type in {"X_MES_ATR60", "VWAP_MID_ALIGNED", "VWAP_BP_ALIGNED"}:
        return True
    if PDR_RE.match(filter_type):
        return True
    if GAP_RE.match(filter_type):
        return True
    return False


def split_mask(df: pd.DataFrame, spec: ThresholdSpec) -> pd.Series:
    return df["gp"] >= spec.threshold if spec.side == "high" else df["gp"] <= spec.threshold


def sharpe(arr: np.ndarray) -> float:
    sd = arr.std(ddof=1)
    return float(arr.mean() / sd) if sd > 0 else 0.0


def load_trades(con, row: pd.Series, direction: str, *, is_oos: bool) -> pd.DataFrame:
    filter_sql, join_sql = exact_filter_sql(row["filter_type"], row["orb_label"], row["instrument"])
    if filter_sql is None:
        return pd.DataFrame()
    date_clause = ">=" if is_oos else "<"
    q = f"""
    SELECT o.trading_day, o.pnl_r, d.garch_forecast_vol_pct AS gp
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    {join_sql}
    WHERE o.symbol = '{row["instrument"]}'
      AND o.orb_minutes = {row["orb_minutes"]}
      AND o.orb_label = '{row["orb_label"]}'
      AND o.entry_model = '{row["entry_model"]}'
      AND o.rr_target = {row["rr_target"]}
      AND o.pnl_r IS NOT NULL
      AND d.garch_forecast_vol_pct IS NOT NULL
      AND d.orb_{row["orb_label"]}_break_dir = '{direction}'
      AND {filter_sql}
      AND o.trading_day {date_clause} DATE '{IS_END}'
    ORDER BY o.trading_day
    """
    try:
        df = con.execute(q).df()
    except Exception as exc:
        print(f"ERR {row['strategy_id']} {direction}: {type(exc).__name__}: {exc}")
        return pd.DataFrame()
    if len(df) == 0:
        return df
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["year"] = df["trading_day"].dt.year
    df["pnl_r"] = df["pnl_r"].astype(float)
    df["gp"] = df["gp"].astype(float)
    return df


def ntile_shape(df: pd.DataFrame) -> dict[str, object]:
    if len(df) < 80:
        return {"skip": True}
    work = df.copy()
    work["bucket"] = pd.qcut(work["gp"], 5, labels=False, duplicates="drop")
    agg = work.groupby("bucket")["pnl_r"].agg(["size", "mean"]).reset_index()
    if len(agg) < 5:
        return {"skip": True}
    bucket_means = agg["mean"].tolist()
    best_bucket = int(agg.loc[agg["mean"].idxmax(), "bucket"])
    tail_bias = float(bucket_means[-1] - bucket_means[0])
    rho, p_val = stats.spearmanr(agg["bucket"], agg["mean"])
    return {
        "skip": False,
        "best_bucket": best_bucket,
        "tail_bias": tail_bias,
        "rho": float(rho) if not np.isnan(rho) else 0.0,
        "p_val": float(p_val) if not np.isnan(p_val) else 1.0,
    }


def oos_lift(df_oos: pd.DataFrame, spec: ThresholdSpec) -> tuple[int, int, float | None]:
    if len(df_oos) == 0:
        return 0, 0, None
    mask = split_mask(df_oos, spec)
    on = df_oos.loc[mask, "pnl_r"]
    off = df_oos.loc[~mask, "pnl_r"]
    if len(on) < 3 or len(off) < 3:
        return len(on), len(off), None
    return len(on), len(off), float(on.mean() - off.mean())


def test_spec(df: pd.DataFrame, df_oos: pd.DataFrame, spec: ThresholdSpec) -> dict[str, object]:
    if len(df) < MIN_TOTAL:
        return {"skip": True}
    mask = split_mask(df, spec)
    on = df.loc[mask]
    off = df.loc[~mask]
    if len(on) < MIN_SIDE or len(off) < MIN_SIDE:
        return {"skip": True}

    expr_on = float(on["pnl_r"].mean())
    expr_off = float(off["pnl_r"].mean())
    sr_on = sharpe(on["pnl_r"].to_numpy())
    sr_off = sharpe(off["pnl_r"].to_numpy())
    lift = expr_on - expr_off
    sr_lift = sr_on - sr_off
    t_stat, p_mean = stats.ttest_ind(on["pnl_r"], off["pnl_r"], equal_var=False)

    pnl = df["pnl_r"].to_numpy()
    is_on = mask.astype(int).to_numpy()
    rng = np.random.default_rng(SEED)
    beats = 0
    for _ in range(PERMUTATIONS):
        shuffled = rng.permutation(is_on)
        if shuffled.sum() > 1 and (1 - shuffled).sum() > 1:
            perm = sharpe(pnl[shuffled == 1]) - sharpe(pnl[shuffled == 0])
            if abs(perm) >= abs(sr_lift):
                beats += 1
    p_sharpe = (beats + 1) / (PERMUTATIONS + 1)

    yr_pos = 0
    yr_total = 0
    for year in sorted(df["year"].unique()):
        sub = df[df["year"] == year]
        sub_mask = split_mask(sub, spec)
        on_y = sub.loc[sub_mask, "pnl_r"]
        off_y = sub.loc[~sub_mask, "pnl_r"]
        if len(on_y) >= 3 and len(off_y) >= 3:
            yr_total += 1
            if float(on_y.mean() - off_y.mean()) > 0:
                yr_pos += 1

    oos_on, oos_off, oos = oos_lift(df_oos, spec)
    return {
        "skip": False,
        "side": spec.side,
        "threshold": spec.threshold,
        "N": len(df),
        "N_on": len(on),
        "N_off": len(off),
        "lift": lift,
        "sr_lift": sr_lift,
        "p_mean": float(p_mean),
        "p_sharpe": float(p_sharpe),
        "yr_pos": yr_pos,
        "yr_total": yr_total,
        "oos_on": oos_on,
        "oos_off": oos_off,
        "oos_lift": oos,
    }


def bh_fdr(pvals: list[float], q: float = 0.05) -> list[bool]:
    n = len(pvals)
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    max_rank = 0
    for rank, (_, p) in enumerate(indexed, start=1):
        if not np.isnan(p) and p <= q * rank / n:
            max_rank = rank
    out = [False] * n
    for rank, (idx, _) in enumerate(indexed, start=1):
        if rank <= max_rank:
            out[idx] = True
    return out


def main() -> None:
    print("GARCH BROAD EXACT ROLE EXHAUSTION")
    print("=" * 72)
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    rows = load_rows(con)
    rows = rows[rows["filter_type"].map(in_scope)].copy()
    print(f"Rows in exact broad scope: {len(rows)}")

    results: list[dict[str, object]] = []
    shapes: list[dict[str, object]] = []

    for _, row in rows.iterrows():
        for direction in ["long", "short"]:
            df = load_trades(con, row, direction, is_oos=False)
            if len(df) < MIN_TOTAL:
                continue
            df_oos = load_trades(con, row, direction, is_oos=True)
            shape = ntile_shape(df)
            shapes.append(
                {
                    "src": row["src"],
                    "strategy_id": row["strategy_id"],
                    "instrument": row["instrument"],
                    "orb_label": row["orb_label"],
                    "rr_target": row["rr_target"],
                    "direction": direction,
                    "filter_type": row["filter_type"],
                    **shape,
                }
            )
            for spec in SPECS:
                r = test_spec(df, df_oos, spec)
                if r.get("skip"):
                    continue
                r.update(
                    {
                        "src": row["src"],
                        "strategy_id": row["strategy_id"],
                        "instrument": row["instrument"],
                        "orb_label": row["orb_label"],
                        "orb_minutes": row["orb_minutes"],
                        "rr_target": row["rr_target"],
                        "direction": direction,
                        "filter_type": row["filter_type"],
                    }
                )
                results.append(r)
    con.close()

    results_df = pd.DataFrame(results)
    shapes_df = pd.DataFrame(shapes)
    print(f"Primary tests run: {len(results_df)}")
    bh_mean = bh_fdr(results_df["p_mean"].tolist(), q=0.05) if len(results_df) else []
    bh_sharpe = bh_fdr(results_df["p_sharpe"].tolist(), q=0.05) if len(results_df) else []
    if len(results_df):
        results_df["bh_mean"] = bh_mean
        results_df["bh_sharpe"] = bh_sharpe
    print(f"BH mean survivors: {sum(bh_mean)}")
    print(f"BH sharpe survivors: {sum(bh_sharpe)}")

    if len(results_df):
        print("\nTop 15 positive local cells:")
        for _, r in results_df.sort_values("sr_lift", ascending=False).head(15).iterrows():
            print(
                f"  [{r['src']}] {r['strategy_id']} {r['direction']} {r['side']}@{int(r['threshold'])}: "
                f"sr_lift={r['sr_lift']:+.3f} lift={r['lift']:+.3f} p_sh={r['p_sharpe']:.4f}"
            )
        print("\nTop 15 negative local cells:")
        for _, r in results_df.sort_values("sr_lift", ascending=True).head(15).iterrows():
            print(
                f"  [{r['src']}] {r['strategy_id']} {r['direction']} {r['side']}@{int(r['threshold'])}: "
                f"sr_lift={r['sr_lift']:+.3f} lift={r['lift']:+.3f} p_sh={r['p_sharpe']:.4f}"
            )

    emit(rows, results_df, shapes_df)


def emit(rows: pd.DataFrame, results_df: pd.DataFrame, shapes_df: pd.DataFrame) -> None:
    lines = [
        "# Garch Broad Exact Role Exhaustion",
        "",
        "**Date:** 2026-04-16",
        "**Scope:** `validated_setups` + `experimental_strategies`, exact filter semantics only.",
        "",
        f"- Strategy rows in scope: **{len(rows)}**",
        f"- Primary tests run: **{len(results_df)}**",
        f"- BH mean survivors: **{int(results_df['bh_mean'].sum()) if len(results_df) else 0}**",
        f"- BH sharpe survivors: **{int(results_df['bh_sharpe'].sum()) if len(results_df) else 0}**",
        "",
        "**Included filter families:** ORB_G*, ORB_G*_NOFRI, ATR_P*, COST_LT*, OVNRNG_*, X_MES_ATR60, VWAP_MID_ALIGNED, VWAP_BP_ALIGNED, PDR_R080, GAP_R015.",
        "",
        "**Excluded families:** GARCH_* self-reference, NO_FILTER, CROSS_*_MOMENTUM, and filters without exact clean semantics in this pass.",
        "",
        "---",
        "",
        "## Top positive local cells",
        "",
        "| Src | Strategy | Dir | Side | Thr | sr_lift | lift | p_sharpe | OOS lift |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in results_df.sort_values("sr_lift", ascending=False).head(25).iterrows():
        oos = "n/a" if pd.isna(r["oos_lift"]) else f"{r['oos_lift']:+.3f}"
        lines.append(
            f"| {r['src']} | {r['strategy_id']} | {r['direction']} | {r['side']} | {int(r['threshold'])} | "
            f"{r['sr_lift']:+.3f} | {r['lift']:+.3f} | {r['p_sharpe']:.4f} | {oos} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Top negative local cells",
        "",
        "| Src | Strategy | Dir | Side | Thr | sr_lift | lift | p_sharpe | OOS lift |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in results_df.sort_values("sr_lift", ascending=True).head(25).iterrows():
        oos = "n/a" if pd.isna(r["oos_lift"]) else f"{r['oos_lift']:+.3f}"
        lines.append(
            f"| {r['src']} | {r['strategy_id']} | {r['direction']} | {r['side']} | {int(r['threshold'])} | "
            f"{r['sr_lift']:+.3f} | {r['lift']:+.3f} | {r['p_sharpe']:.4f} | {oos} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Shape summary by session",
        "",
        "| Session | Count | Mean tail bias | Mean best bucket |",
        "|---|---|---|---|",
    ]
    if len(shapes_df):
        agg = (
            shapes_df[~shapes_df.get("skip", False)]
            .groupby("orb_label")
            .agg(count=("strategy_id", "size"), tail_bias=("tail_bias", "mean"), best_bucket=("best_bucket", "mean"))
            .reset_index()
        )
        for _, r in agg.iterrows():
            lines.append(
                f"| {r['orb_label']} | {int(r['count'])} | {float(r['tail_bias']):+.3f} | {float(r['best_bucket']):.2f} |"
            )

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] {OUTPUT_MD}")


if __name__ == "__main__":
    main()
