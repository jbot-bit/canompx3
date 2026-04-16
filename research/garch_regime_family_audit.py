"""Pre-registered garch regime family audit.

Purpose:
  Turn the exploratory broad garch surface into a disciplined family-level
  audit with:
    - fixed thresholds only (HIGH >= 70, LOW <= 30)
    - session-side directional sign tests
    - session-side monotonicity / tail-bias tests
    - global shuffle-null destruction controls

This script does NOT claim production readiness. It answers whether garch
behaves like a structured regime-family variable under natural family
boundaries, using canonical trade populations loaded from orb_outcomes +
daily_features with exact filter semantics.

Pre-registration:
  docs/audit/hypotheses/2026-04-16-garch-regime-family-audit.yaml

Output:
  docs/audit/results/2026-04-16-garch-regime-family-audit.md
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
from research import garch_broad_exact_role_exhaustion as broad

OUTPUT_MD = Path("docs/audit/results/2026-04-16-garch-regime-family-audit.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

HIGH_THRESHOLD = 70
LOW_THRESHOLD = 30
SHUFFLES = 100
FAMILY_MIN_CELLS = 12
SHAPE_MIN_CELLS = 8
SEED = 20260416


@dataclass
class CellRecord:
    strategy_id: str
    src: str
    instrument: str
    orb_label: str
    orb_minutes: int
    rr_target: float
    direction: str
    filter_type: str
    pnl: np.ndarray
    gp: np.ndarray
    years: np.ndarray
    high_sr_lift: float
    high_lift: float
    high_p_sharpe: float
    high_oos_lift: float | None
    low_sr_lift: float
    low_lift: float
    low_p_sharpe: float
    low_oos_lift: float | None
    shape_skip: bool
    tail_bias: float | None
    best_bucket: int | None


def preflight(con: duckdb.DuckDBPyConnection) -> dict[str, object]:
    latest_outcomes = con.execute(
        "SELECT symbol, MAX(trading_day) AS max_day, COUNT(*) AS n FROM orb_outcomes GROUP BY symbol ORDER BY symbol"
    ).df()
    latest_features = con.execute(
        "SELECT symbol, MAX(trading_day) AS max_day, COUNT(*) AS n FROM daily_features GROUP BY symbol ORDER BY symbol"
    ).df()
    schema_df = con.execute("DESCRIBE daily_features").df()
    required = {
        "garch_forecast_vol",
        "garch_forecast_vol_pct",
        "atr_20_pct",
        "overnight_range",
        "gap_open_points",
    }
    present = set(schema_df["column_name"].astype(str))
    return {
        "orb_outcomes": latest_outcomes,
        "daily_features": latest_features,
        "required_present": sorted(required & present),
        "required_missing": sorted(required - present),
    }


def build_cells() -> tuple[list[CellRecord], dict[str, object]]:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    pf = preflight(con)

    rows = broad.load_rows(con)
    rows = rows[rows["filter_type"].map(broad.in_scope)].copy()

    cells: list[CellRecord] = []
    for _, row in rows.iterrows():
        for direction in ["long", "short"]:
            df = broad.load_trades(con, row, direction, is_oos=False)
            if len(df) < broad.MIN_TOTAL:
                continue
            df_oos = broad.load_trades(con, row, direction, is_oos=True)

            high = broad.test_spec(df, df_oos, broad.ThresholdSpec("high", HIGH_THRESHOLD))
            low = broad.test_spec(df, df_oos, broad.ThresholdSpec("low", LOW_THRESHOLD))
            if high.get("skip") or low.get("skip"):
                continue

            shape = broad.ntile_shape(df)
            cells.append(
                CellRecord(
                    strategy_id=str(row["strategy_id"]),
                    src=str(row["src"]),
                    instrument=str(row["instrument"]),
                    orb_label=str(row["orb_label"]),
                    orb_minutes=int(row["orb_minutes"]),
                    rr_target=float(row["rr_target"]),
                    direction=direction,
                    filter_type=str(row["filter_type"]),
                    pnl=df["pnl_r"].to_numpy(dtype=float),
                    gp=df["gp"].to_numpy(dtype=float),
                    years=df["year"].to_numpy(dtype=int),
                    high_sr_lift=float(high["sr_lift"]),
                    high_lift=float(high["lift"]),
                    high_p_sharpe=float(high["p_sharpe"]),
                    high_oos_lift=None if pd.isna(high["oos_lift"]) else float(high["oos_lift"]),
                    low_sr_lift=float(low["sr_lift"]),
                    low_lift=float(low["lift"]),
                    low_p_sharpe=float(low["p_sharpe"]),
                    low_oos_lift=None if pd.isna(low["oos_lift"]) else float(low["oos_lift"]),
                    shape_skip=bool(shape.get("skip", False)),
                    tail_bias=None if shape.get("skip") else float(shape["tail_bias"]),
                    best_bucket=None if shape.get("skip") else int(shape["best_bucket"]),
                )
            )
    con.close()
    return cells, pf


def directional_p(n_support: int, n_total: int) -> float:
    if n_total <= 0:
        return float("nan")
    return float(stats.binom.sf(n_support - 1, n_total, 0.5))


def sr_lift_from_arrays(pnl: np.ndarray, gp: np.ndarray, side: str) -> float | None:
    if side == "high":
        mask = gp >= HIGH_THRESHOLD
    else:
        mask = gp <= LOW_THRESHOLD
    on = pnl[mask]
    off = pnl[~mask]
    if len(on) < broad.MIN_SIDE or len(off) < broad.MIN_SIDE:
        return None
    s_on = on.std(ddof=1)
    s_off = off.std(ddof=1)
    sr_on = on.mean() / s_on if s_on > 0 else 0.0
    sr_off = off.mean() / s_off if s_off > 0 else 0.0
    return float(sr_on - sr_off)


def family_directional(cells: list[CellRecord]) -> pd.DataFrame:
    rows = []
    for sess in sorted({c.orb_label for c in cells}):
        for side in ["high", "low"]:
            fam = [c for c in cells if c.orb_label == sess]
            if side == "high":
                vals = [c.high_sr_lift for c in fam]
                lifts = [c.high_lift for c in fam]
                pvals = [c.high_p_sharpe for c in fam]
                oos = [c.high_oos_lift for c in fam if c.high_oos_lift is not None]
                support = sum(v > 0 for v in vals)
            else:
                vals = [c.low_sr_lift for c in fam]
                lifts = [c.low_lift for c in fam]
                pvals = [c.low_p_sharpe for c in fam]
                oos = [c.low_oos_lift for c in fam if c.low_oos_lift is not None]
                support = sum(v < 0 for v in vals)
            n = len(vals)
            long_vals = [v for c, v in zip(fam, vals) if c.direction == "long"]
            short_vals = [v for c, v in zip(fam, vals) if c.direction == "short"]
            bh_ct = sum(broad.bh_fdr(pvals, q=0.05)) if pvals else 0
            rows.append(
                {
                    "session": sess,
                    "side": side,
                    "n_cells": n,
                    "support": support,
                    "oppose": n - support,
                    "support_frac": support / n if n else float("nan"),
                    "p_dir": directional_p(support, n),
                    "mean_sr_lift": float(np.mean(vals)) if vals else float("nan"),
                    "mean_lift": float(np.mean(lifts)) if lifts else float("nan"),
                    "long_support": sum(v > 0 for v in long_vals) if side == "high" else sum(v < 0 for v in long_vals),
                    "long_total": len(long_vals),
                    "short_support": sum(v > 0 for v in short_vals) if side == "high" else sum(v < 0 for v in short_vals),
                    "short_total": len(short_vals),
                    "family_bh_survivors": bh_ct,
                    "oos_match_frac": (
                        sum(x > 0 for x in oos) / len(oos) if (side == "high" and oos) else
                        sum(x < 0 for x in oos) / len(oos) if (side == "low" and oos) else
                        float("nan")
                    ),
                }
            )
    out = pd.DataFrame(rows)
    eligible = out["n_cells"] >= FAMILY_MIN_CELLS
    flags = broad.bh_fdr(out.loc[eligible, "p_dir"].tolist(), q=0.05)
    out["bh_dir"] = False
    out.loc[eligible, "bh_dir"] = flags
    return out.sort_values(["session", "side"]).reset_index(drop=True)


def family_monotonicity(cells: list[CellRecord]) -> pd.DataFrame:
    rows = []
    for sess in sorted({c.orb_label for c in cells}):
        fam = [c for c in cells if c.orb_label == sess and not c.shape_skip and c.tail_bias is not None]
        for side in ["high", "low"]:
            if side == "high":
                support = sum(c.tail_bias > 0 for c in fam)
            else:
                support = sum(c.tail_bias < 0 for c in fam)
            n = len(fam)
            rows.append(
                {
                    "session": sess,
                    "side": side,
                    "n_shapes": n,
                    "support": support,
                    "oppose": n - support,
                    "support_frac": support / n if n else float("nan"),
                    "p_tail": directional_p(support, n),
                    "mean_tail_bias": float(np.mean([c.tail_bias for c in fam])) if fam else float("nan"),
                    "mean_best_bucket": float(np.mean([c.best_bucket for c in fam])) if fam else float("nan"),
                }
            )
    out = pd.DataFrame(rows)
    eligible = out["n_shapes"] >= SHAPE_MIN_CELLS
    flags = broad.bh_fdr(out.loc[eligible, "p_tail"].tolist(), q=0.05)
    out["bh_tail"] = False
    out.loc[eligible, "bh_tail"] = flags
    return out.sort_values(["session", "side"]).reset_index(drop=True)


def global_asymmetry(cells: list[CellRecord]) -> dict[str, object]:
    high_support = sum(c.high_sr_lift > 0 for c in cells)
    low_support = sum(c.low_sr_lift < 0 for c in cells)
    return {
        "n_cells": len(cells),
        "high_pos": high_support,
        "high_frac": high_support / len(cells) if cells else float("nan"),
        "high_p": directional_p(high_support, len(cells)),
        "low_neg": low_support,
        "low_frac": low_support / len(cells) if cells else float("nan"),
        "low_p": directional_p(low_support, len(cells)),
    }


def shuffle_controls(cells: list[CellRecord], rng_seed: int = SEED) -> dict[str, object]:
    rng = np.random.default_rng(rng_seed)
    real_high = sum(c.high_sr_lift > 0 for c in cells) / len(cells)
    real_low = sum(c.low_sr_lift < 0 for c in cells) / len(cells)

    high_fracs = []
    low_fracs = []
    for _ in range(SHUFFLES):
        pos_high = 0
        neg_low = 0
        for c in cells:
            shuffled = rng.permutation(c.gp)
            s_high = sr_lift_from_arrays(c.pnl, shuffled, "high")
            s_low = sr_lift_from_arrays(c.pnl, shuffled, "low")
            if s_high is not None and s_high > 0:
                pos_high += 1
            if s_low is not None and s_low < 0:
                neg_low += 1
        high_fracs.append(pos_high / len(cells))
        low_fracs.append(neg_low / len(cells))

    high_p = (sum(x >= real_high for x in high_fracs) + 1) / (len(high_fracs) + 1)
    low_p = (sum(x >= real_low for x in low_fracs) + 1) / (len(low_fracs) + 1)
    return {
        "real_high_frac": real_high,
        "real_low_frac": real_low,
        "shuf_high_median": float(np.median(high_fracs)),
        "shuf_low_median": float(np.median(low_fracs)),
        "shuf_high_range": (float(np.min(high_fracs)), float(np.max(high_fracs))),
        "shuf_low_range": (float(np.min(low_fracs)), float(np.max(low_fracs))),
        "shuf_high_p": float(high_p),
        "shuf_low_p": float(low_p),
    }


def emit(
    pf: dict[str, object],
    cells: list[CellRecord],
    directional: pd.DataFrame,
    monotone: pd.DataFrame,
    asym: dict[str, object],
    shuf: dict[str, object],
) -> None:
    lines = [
        "# Garch Regime Family Audit",
        "",
        "**Date:** 2026-04-16",
        "**Pre-registration:** `docs/audit/hypotheses/2026-04-16-garch-regime-family-audit.yaml`",
        "**Fixed tails:** `HIGH >= 70`, `LOW <= 30`",
        "**Cell inventory source:** current tradeable exact-filter row universe seeded from `validated_setups` + `experimental_strategies`; all metrics computed from canonical `orb_outcomes` + `daily_features`.",
        "",
        "## Preflight",
        "",
        "### orb_outcomes freshness / counts",
        "",
        "| Symbol | Max trading_day | Rows |",
        "|---|---|---|",
    ]
    for _, r in pf["orb_outcomes"].iterrows():
        lines.append(f"| {r['symbol']} | {r['max_day']} | {int(r['n'])} |")

    lines += [
        "",
        "### daily_features freshness / counts",
        "",
        "| Symbol | Max trading_day | Rows |",
        "|---|---|---|",
    ]
    for _, r in pf["daily_features"].iterrows():
        lines.append(f"| {r['symbol']} | {r['max_day']} | {int(r['n'])} |")

    lines += [
        "",
        f"Required columns present: `{', '.join(pf['required_present'])}`",
        f"Required columns missing: `{', '.join(pf['required_missing']) if pf['required_missing'] else 'none'}`",
        "",
        "## Global asymmetry",
        "",
        f"- Cells in scope: **{len(cells)}**",
        f"- HIGH @70 positive cells: **{asym['high_pos']} / {asym['n_cells']}** (`p={asym['high_p']:.6f}`)",
        f"- LOW @30 negative cells: **{asym['low_neg']} / {asym['n_cells']}** (`p={asym['low_p']:.6f}`)",
        "",
        "## Shuffle-null destruction control",
        "",
        f"- Real HIGH positive fraction: **{shuf['real_high_frac']:.3f}**",
        f"- Shuffled HIGH median fraction: **{shuf['shuf_high_median']:.3f}** range [{shuf['shuf_high_range'][0]:.3f}, {shuf['shuf_high_range'][1]:.3f}]",
        f"- Shuffle p (HIGH real >= shuffled): **{shuf['shuf_high_p']:.4f}**",
        "",
        f"- Real LOW negative fraction: **{shuf['real_low_frac']:.3f}**",
        f"- Shuffled LOW median fraction: **{shuf['shuf_low_median']:.3f}** range [{shuf['shuf_low_range'][0]:.3f}, {shuf['shuf_low_range'][1]:.3f}]",
        f"- Shuffle p (LOW real >= shuffled): **{shuf['shuf_low_p']:.4f}**",
        "",
        "## Session-side directional sign test",
        "",
        "| Session | Side | Cells | Support | Oppose | Support % | p_dir | BH | mean sr_lift | mean lift | family BH survivors | OOS sign match | Long support | Short support |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in directional.iterrows():
        bh = "Y" if bool(r["bh_dir"]) else "."
        oos = "n/a" if pd.isna(r["oos_match_frac"]) else f"{float(r['oos_match_frac']):.1%}"
        lines.append(
            f"| {r['session']} | {r['side']} | {int(r['n_cells'])} | {int(r['support'])} | {int(r['oppose'])} | "
            f"{float(r['support_frac']):.1%} | {float(r['p_dir']):.6f} | {bh} | "
            f"{float(r['mean_sr_lift']):+.3f} | {float(r['mean_lift']):+.3f} | "
            f"{int(r['family_bh_survivors'])} | {oos} | "
            f"{int(r['long_support'])}/{int(r['long_total'])} | {int(r['short_support'])}/{int(r['short_total'])} |"
        )

    lines += [
        "",
        "## Session-side monotonicity / tail-bias test",
        "",
        "| Session | Side | Shapes | Support | Oppose | Support % | p_tail | BH | mean tail bias | mean best bucket |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in monotone.iterrows():
        bh = "Y" if bool(r["bh_tail"]) else "."
        lines.append(
            f"| {r['session']} | {r['side']} | {int(r['n_shapes'])} | {int(r['support'])} | {int(r['oppose'])} | "
            f"{float(r['support_frac']):.1%} | {float(r['p_tail']):.6f} | {bh} | "
            f"{float(r['mean_tail_bias']):+.3f} | {float(r['mean_best_bucket']):.2f} |"
        )

    top_dir = directional[directional["bh_dir"] == True].sort_values("p_dir")
    if len(top_dir):
        lines += ["", "## Families surviving directional BH", ""]
        for _, r in top_dir.iterrows():
            lines.append(
                f"- `{r['session']} {r['side']}`: support {int(r['support'])}/{int(r['n_cells'])}, "
                f"`p_dir={float(r['p_dir']):.6f}`, mean `sr_lift={float(r['mean_sr_lift']):+.3f}`"
            )

    top_tail = monotone[monotone["bh_tail"] == True].sort_values("p_tail")
    if len(top_tail):
        lines += ["", "## Families surviving monotonicity BH", ""]
        for _, r in top_tail.iterrows():
            lines.append(
                f"- `{r['session']} {r['side']}`: support {int(r['support'])}/{int(r['n_shapes'])}, "
                f"`p_tail={float(r['p_tail']):.6f}`, mean `tail_bias={float(r['mean_tail_bias']):+.3f}`"
            )

    lines += [
        "",
        "---",
        "",
        "## Notes",
        "",
        "- This is a family audit, not a production promotion decision.",
        "- Session-side BH asks whether the regime effect clusters naturally; global BH still governs any universal-overlay headline claim.",
        "- Tail-bias is an informational structural check for R3/R7 suitability; it does not replace forward validation.",
    ]

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {OUTPUT_MD}")


def main() -> None:
    print("GARCH REGIME FAMILY AUDIT")
    print("=" * 72)
    cells, pf = build_cells()
    print(f"cells in scope: {len(cells)}")

    directional = family_directional(cells)
    monotone = family_monotonicity(cells)
    asym = global_asymmetry(cells)
    shuf = shuffle_controls(cells)

    print(f"global high pos: {asym['high_pos']}/{asym['n_cells']} p={asym['high_p']:.6f}")
    print(f"global low neg: {asym['low_neg']}/{asym['n_cells']} p={asym['low_p']:.6f}")
    print(f"shuffle high p: {shuf['shuf_high_p']:.4f}")
    print(f"shuffle low p: {shuf['shuf_low_p']:.4f}")
    print(f"directional BH survivors: {int(directional['bh_dir'].sum())}")
    print(f"tail-bias BH survivors: {int(monotone['bh_tail'].sum())}")

    emit(pf, cells, directional, monotone, asym, shuf)


if __name__ == "__main__":
    main()
