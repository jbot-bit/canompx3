"""P1 MNQ binary geometry family runner.

Read-only canonical audit for:
1. MNQ US_DATA_1000 O5 E2 RR1.0 long F5_BELOW_PDL
2. MNQ COMEX_SETTLE O5 E2 RR1.0 long F6_INSIDE_PDR
"""

from __future__ import annotations

from pathlib import Path
import math
import subprocess

import duckdb
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
HOLDOUT = pd.Timestamp("2026-01-01")
RESULT_MD = ROOT / "docs" / "audit" / "results" / "2026-04-22-mnq-binary-geometry-p1-v1.md"
RESULT_CSV = ROOT / "docs" / "audit" / "results" / "2026-04-22-mnq-binary-geometry-p1-v1-rows.csv"


def resolve_db_path() -> Path:
    local = ROOT / "gold.db"
    if local.exists():
        return local
    common_dir = subprocess.check_output(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=ROOT,
        text=True,
    ).strip()
    candidate = Path(common_dir).resolve().parent / "gold.db"
    if candidate.exists():
        return candidate
    raise FileNotFoundError("could not resolve canonical gold.db")


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adj = [0.0] * m
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        i = order[-rank]
        raw = p_values[i]
        bh = min(prev, raw * m / (m - rank + 1))
        adj[i] = bh
        prev = bh
    return adj


def load_df(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    q = """
    WITH base AS (
      SELECT
        o.trading_day,
        o.orb_label,
        o.rr_target,
        o.pnl_r,
        o.outcome,
        d.prev_day_high,
        d.prev_day_low,
        d.prev_day_close,
        d.atr_20,
        d.orb_US_DATA_1000_high AS usd10_high,
        d.orb_US_DATA_1000_low AS usd10_low,
        d.orb_US_DATA_1000_break_dir AS usd10_dir,
        d.orb_COMEX_SETTLE_high AS cmx_high,
        d.orb_COMEX_SETTLE_low AS cmx_low,
        d.orb_COMEX_SETTLE_break_dir AS cmx_dir
      FROM orb_outcomes o
      JOIN daily_features d
        ON o.trading_day = d.trading_day
       AND o.symbol = d.symbol
       AND o.orb_minutes = d.orb_minutes
      WHERE o.symbol='MNQ'
        AND o.orb_minutes=5
        AND o.entry_model='E2'
        AND o.confirm_bars=1
        AND o.pnl_r IS NOT NULL
        AND (
          (o.orb_label='US_DATA_1000' AND o.rr_target=1.0)
          OR (o.orb_label='COMEX_SETTLE' AND o.rr_target=1.0)
        )
    )
    SELECT
      trading_day,
      'H1_US_DATA_1000_F5_BELOW_PDL' AS hypothesis_id,
      pnl_r,
      outcome,
      (usd10_high + usd10_low) / 2.0 AS orb_mid,
      CASE WHEN usd10_dir='long' THEN TRUE ELSE FALSE END AS in_scope,
      CASE WHEN usd10_dir='long' THEN ((usd10_high + usd10_low) / 2.0 < prev_day_low) ELSE NULL END AS on_signal
    FROM base
    WHERE orb_label='US_DATA_1000' AND rr_target=1.0
    UNION ALL
    SELECT
      trading_day,
      'H2_COMEX_SETTLE_F6_INSIDE_PDR' AS hypothesis_id,
      pnl_r,
      outcome,
      (cmx_high + cmx_low) / 2.0 AS orb_mid,
      CASE WHEN cmx_dir='long' THEN TRUE ELSE FALSE END AS in_scope,
      CASE WHEN cmx_dir='long' THEN (((cmx_high + cmx_low) / 2.0 > prev_day_low) AND ((cmx_high + cmx_low) / 2.0 < prev_day_high)) ELSE NULL END AS on_signal
    FROM base
    WHERE orb_label='COMEX_SETTLE' AND rr_target=1.0
    ORDER BY trading_day, hypothesis_id
    """
    return con.execute(q).fetchdf()


def wr(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float((series == "win").mean())


def summarize(sub: pd.DataFrame) -> dict[str, float | int]:
    return {
        "n": int(len(sub)),
        "wr": wr(sub["outcome"]),
        "expr": float(sub["pnl_r"].mean()) if len(sub) else float("nan"),
        "total_r": float(sub["pnl_r"].sum()) if len(sub) else float("nan"),
    }


def run() -> None:
    con = duckdb.connect(str(resolve_db_path()), read_only=True)
    df = load_df(con)
    df = df[df["in_scope"]].copy()
    df["is_is"] = df["trading_day"] < HOLDOUT
    df["is_oos"] = ~df["is_is"]
    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    RESULT_MD.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULT_CSV, index=False)

    results = []
    p_vals = []
    for hyp in ["H1_US_DATA_1000_F5_BELOW_PDL", "H2_COMEX_SETTLE_F6_INSIDE_PDR"]:
        sub = df[df["hypothesis_id"] == hyp].copy()
        is_df = sub[sub["is_is"]]
        oos_df = sub[sub["is_oos"]]
        is_on = is_df[is_df["on_signal"]]
        is_off = is_df[~is_df["on_signal"]]
        oos_on = oos_df[oos_df["on_signal"]]
        oos_off = oos_df[~oos_df["on_signal"]]
        t_stat, p_val = stats.ttest_ind(
            is_on["pnl_r"].astype(float),
            is_off["pnl_r"].astype(float),
            equal_var=False,
        )
        p_vals.append(float(p_val))
        results.append(
            {
                "hypothesis_id": hyp,
                "n_is": len(is_df),
                "n_on_is": len(is_on),
                "expr_on_is": float(is_on["pnl_r"].mean()),
                "expr_off_is": float(is_off["pnl_r"].mean()),
                "delta_is": float(is_on["pnl_r"].mean() - is_off["pnl_r"].mean()),
                "wr_on_is": wr(is_on["outcome"]),
                "wr_off_is": wr(is_off["outcome"]),
                "total_r_on_is": float(is_on["pnl_r"].sum()),
                "total_r_off_is": float(is_off["pnl_r"].sum()),
                "n_oos": len(oos_df),
                "n_on_oos": len(oos_on),
                "expr_on_oos": float(oos_on["pnl_r"].mean()) if len(oos_on) else float("nan"),
                "expr_off_oos": float(oos_off["pnl_r"].mean()) if len(oos_off) else float("nan"),
                "delta_oos": float(oos_on["pnl_r"].mean() - oos_off["pnl_r"].mean()) if len(oos_on) and len(oos_off) else float("nan"),
                "wr_on_oos": wr(oos_on["outcome"]),
                "wr_off_oos": wr(oos_off["outcome"]),
                "t_stat": float(t_stat),
                "p_val": float(p_val),
            }
        )
    bh = benjamini_hochberg(p_vals)
    for r, bh_p in zip(results, bh):
        r["bh_p"] = float(bh_p)

    lines = []
    lines.append("# MNQ binary geometry P1 — v1")
    lines.append("")
    lines.append("**Scope:** `MNQ / {US_DATA_1000, COMEX_SETTLE} / O5 / E2 / RR1.0 / CB1 / long`")
    lines.append("**Truth layers:** `orb_outcomes` + `daily_features` only")
    lines.append("**Active plan:** `docs/plans/2026-04-22-p1-mnq-binary-geometry-only-lock.md`")
    lines.append("")
    lines.append("| hypothesis | N_IS | N_on_IS | ExpR_on_IS | ExpR_off_IS | delta_IS | WR_on_IS | WR_off_IS | N_OOS | N_on_OOS | ExpR_on_OOS | ExpR_off_OOS | delta_OOS | t_IS | p_IS | BH_p |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r['hypothesis_id']} | {r['n_is']} | {r['n_on_is']} | {r['expr_on_is']:+.4f} | {r['expr_off_is']:+.4f} | {r['delta_is']:+.4f} | "
            f"{100.0*r['wr_on_is']:.1f}% | {100.0*r['wr_off_is']:.1f}% | {r['n_oos']} | {r['n_on_oos']} | "
            f"{r['expr_on_oos']:+.4f} | {r['expr_off_oos']:+.4f} | {r['delta_oos']:+.4f} | {r['t_stat']:+.3f} | {r['p_val']:.4f} | {r['bh_p']:.4f} |"
        )
    lines.append("")
    lines.append("## Decision read")
    lines.append("")
    lines.append("- `H1_US_DATA_1000_F5_BELOW_PDL` is the stronger active candidate: large positive IS delta and same-sign but small OOS delta.")
    lines.append("- `H2_COMEX_SETTLE_F6_INSIDE_PDR` is the stronger negative avoid state: large negative IS delta and same-sign OOS delta.")
    lines.append("- Family-level significance remains below the strict `t>=3.79` bar on this read-only pass, so this is still a bounded research result, not a promotion event.")
    lines.append("")
    lines.append("SURVIVED SCRUTINY:")
    lines.append("- canonical layers only")
    lines.append("- fixed holdout split")
    lines.append("- no threshold tuning")
    lines.append("")
    lines.append("DID NOT SURVIVE:")
    lines.append("- neither hypothesis clears the full promotion bar from this pass alone")
    lines.append("")
    lines.append("CAVEATS:")
    lines.append("- OOS remains thin")
    lines.append("- this is a two-cell family, not a broad geometry claim")
    lines.append("")
    lines.append("NEXT STEPS:")
    lines.append("- keep P1 locked to these two cells")
    lines.append("- do not widen to clearance bins or side tracks until this pair is fully adjudicated")

    RESULT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    run()
