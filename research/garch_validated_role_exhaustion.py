"""Institutional-grade garch role exhaustion on validated strategies.

Purpose:
  Exhaust the plausible validated-scope roles for garch without ad hoc fishing:
    1. High-tail overlay: garch_pct >= {60, 70, 80}
    2. Low-tail overlay:  garch_pct <= {40, 30, 20}
    3. Tail-shape diagnostic via NTILE(5)

Questions:
  - Does high garch help on the ACTUAL validated trade population?
  - Does low garch help somewhere instead (inverse / avoid-high role)?
  - Is the effect tail-monotone enough to justify R3/R7 (size / confluence),
    or is it too irregular and therefore only suitable for narrow R1 use?

Methodology:
  - Load each validated strategy's exact filter semantics.
  - Include X_MES_ATR60 (previous honest test skipped it for complexity).
  - Split by direction.
  - Test six pre-committed thresholds only:
      HIGH: >=60, >=70, >=80
      LOW:  <=40, <=30, <=20
  - Welch mean test + Sharpe permutation + BH-FDR across the whole family.
  - Report OOS direction where there is enough data, but do not over-weight
    thin OOS buckets.

Output:
  docs/audit/results/2026-04-16-garch-validated-role-exhaustion.md
"""

from __future__ import annotations

import io
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

from pipeline.paths import GOLD_DB_PATH
from research.filter_utils import filter_signal  # canonical delegation (research-truth-protocol.md)
from trading_app.config import ALL_FILTERS, CrossAssetATRFilter

OUTPUT_MD = Path("docs/audit/results/2026-04-16-garch-validated-role-exhaustion.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

IS_END = "2026-01-01"
SEED = 20260416
HIGH_THRESHOLDS = [60, 70, 80]
LOW_THRESHOLDS = [40, 30, 20]
MIN_TOTAL = 50
MIN_SIDE = 10
PERMUTATIONS = 1000


@dataclass(frozen=True)
class ThresholdSpec:
    side: str  # "high" or "low"
    threshold: int


SPECS = [ThresholdSpec("high", t) for t in HIGH_THRESHOLDS] + [ThresholdSpec("low", t) for t in LOW_THRESHOLDS]


def load_trades(con, row: pd.Series, direction: str, *, is_oos: bool) -> pd.DataFrame:
    """Load validated-scope trades matching the strategy's exact filter.

    Filter application delegates to canonical ``research.filter_utils.filter_signal``
    per `.claude/rules/research-truth-protocol.md` § Canonical filter delegation.
    No inline filter SQL, no hardcoded cost or threshold constants in this module.

    Cross-asset filters (CrossAssetATRFilter family, e.g. X_MES_ATR60) require
    ``cross_atr_{source_instrument}_pct`` to be injected into the feature row
    before ``filter_signal`` can evaluate. Uses a direct column-map assignment
    (equivalent semantics to ``trading_app.strategy_fitness._enrich_cross_asset_atr``
    but preserves dtypes — the fitness-tracker path round-trips through
    ``list[dict] -> DataFrame`` which collapses numpy dtypes to object).
    """
    filter_type = row["filter_type"]
    if filter_type not in ALL_FILTERS:
        print(f"  ERR unknown filter_type '{filter_type}' for {row['strategy_id']}")
        return pd.DataFrame()

    date_clause = ">=" if is_oos else "<"
    # Load d.* so filter_signal has every column any canonical filter may need.
    q = f"""
    SELECT
      o.trading_day,
      o.pnl_r,
      o.risk_dollars,
      d.*
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol = '{row["instrument"]}'
      AND o.orb_minutes = {row["orb_minutes"]}
      AND o.orb_label = '{row["orb_label"]}'
      AND o.entry_model = '{row["entry_model"]}'
      AND o.rr_target = {row["rr_target"]}
      AND o.pnl_r IS NOT NULL
      AND d.garch_forecast_vol_pct IS NOT NULL
      AND d.orb_{row["orb_label"]}_break_dir = '{direction}'
      AND o.trading_day {date_clause} DATE '{IS_END}'
    ORDER BY o.trading_day
    """
    try:
        df = con.execute(q).df()
    except Exception as exc:
        print(f"  ERR loading {row['strategy_id']} {direction}: {type(exc).__name__}: {exc}")
        return pd.DataFrame()

    if len(df) == 0:
        return df

    # Cross-asset enrichment for CrossAssetATRFilter (e.g., X_MES_ATR60 needs
    # cross_atr_MES_pct injected). Direct column-map assignment — avoids the
    # DataFrame -> list[dict] -> DataFrame round-trip pattern used by the
    # canonical fitness tracker path. Round-tripping collapses numpy/pandas
    # dtypes to object, which can silently corrupt vectorized comparisons
    # for other filters sharing the df. This path keeps the column's float64
    # dtype intact.
    filt = ALL_FILTERS[filter_type]
    if isinstance(filt, CrossAssetATRFilter):
        src = filt.source_instrument
        atr_rows = con.execute(
            """SELECT trading_day, atr_20_pct FROM daily_features
               WHERE symbol = ? AND orb_minutes = 5 AND atr_20_pct IS NOT NULL""",
            [src],
        ).fetchall()
        # Normalize trading_day key to date so the lookup works whether df's
        # trading_day is pd.Timestamp, numpy.datetime64, or plain date.
        atr_map: dict = {}
        for td, pct in atr_rows:
            key = td.date() if hasattr(td, "date") else td
            atr_map[key] = pct

        def _date_key(t):
            return t.date() if hasattr(t, "date") else t

        df[f"cross_atr_{src}_pct"] = df["trading_day"].apply(_date_key).map(atr_map)

    # Canonical filter application — delegate to filter_signal. No inline SQL.
    mask = np.asarray(filter_signal(df, filter_type, row["orb_label"])).astype(bool)
    df = df.loc[mask].reset_index(drop=True)

    if len(df) == 0:
        return df

    # Preserve the downstream consumer schema: {trading_day, pnl_r, gp, year}.
    # "gp" is aliased from canonical garch_forecast_vol_pct for readability.
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["year"] = df["trading_day"].dt.year
    df["pnl_r"] = df["pnl_r"].astype(float)
    df["gp"] = df["garch_forecast_vol_pct"].astype(float)
    return df[["trading_day", "pnl_r", "gp", "year"]]


def split_mask(df: pd.DataFrame, spec: ThresholdSpec) -> pd.Series:
    if spec.side == "high":
        return df["gp"] >= spec.threshold
    return df["gp"] <= spec.threshold


def sharpe(arr: np.ndarray) -> float:
    sd = arr.std(ddof=1)
    return float(arr.mean() / sd) if sd > 0 else 0.0


def oos_lift(df_oos: pd.DataFrame, spec: ThresholdSpec) -> tuple[int, int, float | None]:
    if len(df_oos) == 0:
        return 0, 0, None
    mask = split_mask(df_oos, spec)
    on = df_oos.loc[mask, "pnl_r"]
    off = df_oos.loc[~mask, "pnl_r"]
    if len(on) < 3 or len(off) < 3:
        return len(on), len(off), None
    return len(on), len(off), float(on.mean() - off.mean())


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
        "bucket_means": bucket_means,
        "rho": float(rho) if not np.isnan(rho) else 0.0,
        "p_val": float(p_val) if not np.isnan(p_val) else 1.0,
    }


def test_spec(df: pd.DataFrame, df_oos: pd.DataFrame, spec: ThresholdSpec) -> dict[str, object]:
    if len(df) < MIN_TOTAL:
        return {"skip": True, "reason": f"N={len(df)}"}
    mask = split_mask(df, spec)
    on = df.loc[mask]
    off = df.loc[~mask]
    if len(on) < MIN_SIDE or len(off) < MIN_SIDE:
        return {"skip": True, "reason": f"thin on/off={len(on)}/{len(off)}"}

    expr_on = float(on["pnl_r"].mean())
    expr_off = float(off["pnl_r"].mean())
    sd_on = float(on["pnl_r"].std(ddof=1))
    sd_off = float(off["pnl_r"].std(ddof=1))
    sr_on = expr_on / sd_on if sd_on > 0 else 0.0
    sr_off = expr_off / sd_off if sd_off > 0 else 0.0
    lift = expr_on - expr_off
    sr_lift = sr_on - sr_off

    t_stat, p_mean = stats.ttest_ind(on["pnl_r"], off["pnl_r"], equal_var=False)

    rng = np.random.default_rng(SEED)
    pnl = df["pnl_r"].to_numpy()
    is_on = mask.astype(int).to_numpy()
    beats = 0
    for _ in range(PERMUTATIONS):
        shuffled = rng.permutation(is_on)
        if shuffled.sum() > 1 and (1 - shuffled).sum() > 1:
            lift_perm = sharpe(pnl[shuffled == 1]) - sharpe(pnl[shuffled == 0])
            if abs(lift_perm) >= abs(sr_lift):
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

    n_oos_on, n_oos_off, lift_oos = oos_lift(df_oos, spec)
    return {
        "skip": False,
        "side": spec.side,
        "threshold": spec.threshold,
        "N": len(df),
        "N_on": len(on),
        "N_off": len(off),
        "expr_on": expr_on,
        "expr_off": expr_off,
        "lift": lift,
        "sr_on": sr_on,
        "sr_off": sr_off,
        "sr_lift": sr_lift,
        "wr_on": float((on["pnl_r"] > 0).mean()),
        "wr_off": float((off["pnl_r"] > 0).mean()),
        "t_stat": float(t_stat),
        "p_mean": float(p_mean),
        "p_sharpe": float(p_sharpe),
        "yr_pos": yr_pos,
        "yr_total": yr_total,
        "oos_on": n_oos_on,
        "oos_off": n_oos_off,
        "oos_lift": lift_oos,
    }


def bh_fdr(pvals: list[float], q: float = 0.05) -> list[bool]:
    n = len(pvals)
    if n == 0:
        return []
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


def classify_role(group: pd.DataFrame, shape: dict[str, object]) -> str:
    """Summarize the most plausible role using local evidence only."""
    high = group[group["side"] == "high"].sort_values("p_sharpe")
    low = group[group["side"] == "low"].sort_values("p_sharpe")
    best_high = high.iloc[0] if len(high) else None
    best_low = low.iloc[0] if len(low) else None

    high_good = best_high is not None and best_high["lift"] > 0 and best_high["p_sharpe"] < 0.10
    low_good = best_low is not None and best_low["lift"] > 0 and best_low["p_sharpe"] < 0.10
    high_bad = best_high is not None and best_high["lift"] < 0 and best_high["p_sharpe"] < 0.10

    if high_good and not low_good:
        if not shape.get("skip") and shape.get("best_bucket") == 4:
            return "upper-tail regime / possible R3-R7"
        return "upper-tail binary candidate"
    if low_good and not high_good:
        if not shape.get("skip") and shape.get("best_bucket") == 0:
            return "lower-tail regime / inverse-high avoid"
        return "lower-tail binary candidate"
    if high_bad and low_good:
        return "inverse-high / prefer-low"
    if high_good and low_good:
        return "non-monotone / ambiguous"
    return "null / insufficient"


def main() -> None:
    print("GARCH VALIDATED ROLE EXHAUSTION")
    print("=" * 72)
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    validated = con.execute(
        """
        SELECT strategy_id, instrument, orb_label, orb_minutes, rr_target, entry_model,
               filter_type, sample_size
        FROM validated_setups
        WHERE filter_type IN (
            'ORB_G5','ORB_G5_NOFRI','COST_LT12','OVNRNG_100',
            'ATR_P50','ATR_P70','VWAP_MID_ALIGNED','VWAP_BP_ALIGNED',
            'X_MES_ATR60'
        )
        ORDER BY instrument, orb_label, rr_target, strategy_id
        """
    ).df()
    print(f"Validated strategies in scope: {len(validated)}")

    results: list[dict[str, object]] = []
    shape_rows: list[dict[str, object]] = []

    for _, row in validated.iterrows():
        for direction in ["long", "short"]:
            df = load_trades(con, row, direction, is_oos=False)
            if len(df) < MIN_TOTAL:
                continue
            df_oos = load_trades(con, row, direction, is_oos=True)
            shape = ntile_shape(df)
            shape_row = {
                "strategy_id": row["strategy_id"],
                "direction": direction,
                "instrument": row["instrument"],
                "orb_label": row["orb_label"],
                "rr_target": row["rr_target"],
                "filter_type": row["filter_type"],
                **shape,
            }
            shape_rows.append(shape_row)
            for spec in SPECS:
                r = test_spec(df, df_oos, spec)
                if r.get("skip"):
                    continue
                r.update(
                    {
                        "strategy_id": row["strategy_id"],
                        "direction": direction,
                        "instrument": row["instrument"],
                        "orb_label": row["orb_label"],
                        "rr_target": row["rr_target"],
                        "filter_type": row["filter_type"],
                    }
                )
                results.append(r)
    con.close()

    print(f"Primary tests run: {len(results)}")
    p_mean = [float(r["p_mean"]) for r in results]
    p_sharpe = [float(r["p_sharpe"]) for r in results]
    bh_mean = bh_fdr(p_mean, q=0.05)
    bh_sharpe = bh_fdr(p_sharpe, q=0.05)
    for i, r in enumerate(results):
        r["bh_mean"] = bh_mean[i]
        r["bh_sharpe"] = bh_sharpe[i]

    results_df = pd.DataFrame(results)
    shape_df = pd.DataFrame(shape_rows)

    print(f"BH mean survivors: {sum(bh_mean)}")
    print(f"BH sharpe survivors: {sum(bh_sharpe)}")

    # Local top prints for fast terminal visibility.
    print("\nTop 12 positive by |sr_lift|:")
    for _, r in results_df.sort_values("sr_lift", ascending=False).head(12).iterrows():
        print(
            f"  {r['strategy_id']} {r['direction']} {r['side']}@{int(r['threshold'])}: "
            f"lift={r['lift']:+.3f} sr_lift={r['sr_lift']:+.3f} "
            f"p_sh={r['p_sharpe']:.4f} yrs={int(r['yr_pos'])}/{int(r['yr_total'])} "
            f"oos={('n/a' if pd.isna(r['oos_lift']) else f'{r["oos_lift"]:+.3f}')}"
        )
    print("\nTop 12 negative by sr_lift:")
    for _, r in results_df.sort_values("sr_lift", ascending=True).head(12).iterrows():
        print(
            f"  {r['strategy_id']} {r['direction']} {r['side']}@{int(r['threshold'])}: "
            f"lift={r['lift']:+.3f} sr_lift={r['sr_lift']:+.3f} "
            f"p_sh={r['p_sharpe']:.4f} yrs={int(r['yr_pos'])}/{int(r['yr_total'])} "
            f"oos={('n/a' if pd.isna(r['oos_lift']) else f'{r["oos_lift"]:+.3f}')}"
        )

    role_rows = []
    if not results_df.empty:
        for (strategy_id, direction), grp in results_df.groupby(["strategy_id", "direction"]):
            shape = shape_df[(shape_df["strategy_id"] == strategy_id) & (shape_df["direction"] == direction)]
            shape_info = shape.iloc[0].to_dict() if len(shape) else {"skip": True}
            role_rows.append(
                {
                    "strategy_id": strategy_id,
                    "direction": direction,
                    "instrument": grp["instrument"].iloc[0],
                    "orb_label": grp["orb_label"].iloc[0],
                    "rr_target": grp["rr_target"].iloc[0],
                    "filter_type": grp["filter_type"].iloc[0],
                    "role_guess": classify_role(grp, shape_info),
                    "best_bucket": None if shape_info.get("skip") else int(shape_info["best_bucket"]),
                    "tail_bias": None if shape_info.get("skip") else float(shape_info["tail_bias"]),
                    "shape_rho": None if shape_info.get("skip") else float(shape_info["rho"]),
                }
            )
    role_df = pd.DataFrame(role_rows)

    emit(results_df, shape_df, role_df)


def emit(results_df: pd.DataFrame, shape_df: pd.DataFrame, role_df: pd.DataFrame) -> None:
    lines = [
        "# Garch Validated Role Exhaustion",
        "",
        "**Date:** 2026-04-16",
        "**Purpose:** Exhaust plausible validated-scope uses of `garch_forecast_vol_pct` before making role claims.",
        "",
        "**Grounding:**",
        "- Feature timing and no-lookahead: `pipeline/build_daily_features.py` (`compute_garch_forecast`, prior-rank percentile) and `pipeline/session_guard.py`.",
        "- Role taxonomy: `docs/institutional/mechanism_priors.md` (R1/R3/R7).",
        "- Production gates: `docs/institutional/pre_registered_criteria.md` and `docs/institutional/regime-and-rr-handling-framework.md`.",
        "- Position-size interpretation: `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md`.",
        "",
        f"**Primary family size:** K = {len(results_df)} tests "
        f"({len(results_df[results_df['side'] == 'high'])} high-tail + {len(results_df[results_df['side'] == 'low'])} low-tail).",
        "",
        "**Thresholds:** HIGH {60,70,80}, LOW {40,30,20}. These were fixed before the run to test upper-tail, lower-tail, and inverse-high possibilities without open-ended fishing.",
        "",
        "---",
        "",
        "## BH-FDR summary",
        "",
        f"- Mean-test survivors: **{int(results_df['bh_mean'].sum()) if len(results_df) else 0} / {len(results_df)}**",
        f"- Sharpe-permutation survivors: **{int(results_df['bh_sharpe'].sum()) if len(results_df) else 0} / {len(results_df)}**",
        "",
        "If both counts are zero, there is no validated-scope production claim yet for either high-tail or low-tail usage.",
        "",
        "---",
        "",
        "## Top positive local cells (informational, pre-correction)",
        "",
        "| Strategy | Dir | Side | Thr | N on/off | lift | sr_lift | p_sharpe | yrs+ | OOS lift |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in results_df.sort_values("sr_lift", ascending=False).head(20).iterrows():
        oos = "n/a" if pd.isna(r["oos_lift"]) else f"{r['oos_lift']:+.3f}"
        lines.append(
            f"| {r['strategy_id']} | {r['direction']} | {r['side']} | {int(r['threshold'])} | "
            f"{int(r['N_on'])}/{int(r['N_off'])} | {r['lift']:+.3f} | {r['sr_lift']:+.3f} | "
            f"{r['p_sharpe']:.4f} | {int(r['yr_pos'])}/{int(r['yr_total'])} | {oos} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Top negative local cells (informational, pre-correction)",
        "",
        "| Strategy | Dir | Side | Thr | N on/off | lift | sr_lift | p_sharpe | yrs+ | OOS lift |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in results_df.sort_values("sr_lift", ascending=True).head(20).iterrows():
        oos = "n/a" if pd.isna(r["oos_lift"]) else f"{r['oos_lift']:+.3f}"
        lines.append(
            f"| {r['strategy_id']} | {r['direction']} | {r['side']} | {int(r['threshold'])} | "
            f"{int(r['N_on'])}/{int(r['N_off'])} | {r['lift']:+.3f} | {r['sr_lift']:+.3f} | "
            f"{r['p_sharpe']:.4f} | {int(r['yr_pos'])}/{int(r['yr_total'])} | {oos} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Tail-shape diagnostics",
        "",
        "NTILE(5) checks whether the signal is truly tail-driven. Per Carver-style continuous sizing, a forecast is easier to justify when the edge improves coherently toward a tail rather than peaking in the middle.",
        "",
        "| Strategy | Dir | Best bucket | Tail bias (Q5-Q1) | Spearman rho(bucket, meanR) |",
        "|---|---|---|---|---|",
    ]
    for _, r in shape_df[~shape_df.get("skip", False)].head(40).iterrows():
        lines.append(
            f"| {r['strategy_id']} | {r['direction']} | {int(r['best_bucket'])} | "
            f"{float(r['tail_bias']):+.3f} | {float(r['rho']):+.3f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Role classification by strategy-direction",
        "",
        "| Strategy | Dir | Session | RR | Filter | Role guess | Best bucket | Tail bias |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in role_df.sort_values(["orb_label", "rr_target", "strategy_id", "direction"]).iterrows():
        best_bucket = "n/a" if pd.isna(r["best_bucket"]) else str(int(r["best_bucket"]))
        tail_bias = "n/a" if pd.isna(r["tail_bias"]) else f"{float(r['tail_bias']):+.3f}"
        lines.append(
            f"| {r['strategy_id']} | {r['direction']} | {r['orb_label']} | {r['rr_target']} | "
            f"{r['filter_type']} | {r['role_guess']} | {best_bucket} | {tail_bias} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Honest institutional interpretation",
        "",
        "- `garch` is only production-ready as a filter/sizer if it survives validated-scope family correction. Local cells alone are not enough.",
        "- A strong **upper tail** with best bucket `Q5` supports `R3/R7` exploration better than `R1` binary skip.",
        "- A strong **lower tail** with best bucket `Q1` supports inverse-high avoidance or low-regime preference.",
        "- A best bucket in the middle argues AGAINST Carver-style continuous sizing and suggests the signal is not behaving like a clean forecast.",
        "- Even where local tail behavior is promising, production still requires pre-registration, forward OOS accumulation, Monte Carlo, and live Shiryaev-Roberts monitoring.",
        "",
    ]

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] {OUTPUT_MD}")


if __name__ == "__main__":
    main()
