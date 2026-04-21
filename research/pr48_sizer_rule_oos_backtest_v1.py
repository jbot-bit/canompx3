"""PR #48 sizer-rule OOS backtest — IS-trained quintile thresholds applied to OOS.

Pre-reg: docs/audit/hypotheses/2026-04-21-pr48-sizer-rule-oos-backtest-v1.yaml

Closes the next bounded step flagged by the PR #48 OOS β₁ replication:
"a production sizer rule needs IS-trained thresholds applied to OOS rel_vol".

Per-instrument test (K_family=3, Pathway B):
1. Compute IS per-lane (session, direction) 5-quantile thresholds of rel_vol.
2. For each OOS trade in that lane, bucket by IS thresholds → quintile 1..5.
3. Apply pre-committed multipliers {Q1: 0.5, Q2: 0.75, Q3: 1.0, Q4: 1.25, Q5: 1.5}.
4. Paired comparison: mean(pnl_r * size_mult) vs mean(pnl_r * 1.0). Capital-neutral
   by construction (mean multiplier = 1.0).
5. Pass if delta > 0 AND paired t >= +2.0 (one-tailed).

No capital action.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from pipeline.paths import GOLD_DB_PATH
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

PREREG_PATH = "docs/audit/hypotheses/2026-04-21-pr48-sizer-rule-oos-backtest-v1.yaml"
RESULT_DOC = Path("docs/audit/results/2026-04-21-pr48-sizer-rule-oos-backtest-v1.md")
INSTRUMENTS = ["MNQ", "MES", "MGC"]
APERTURE = 5
RR = 1.5
PAIRED_T_PASS = 2.0
MIN_IS_PER_LANE = 100

# Pre-committed quintile multipliers (mean = 1.0, capital-neutral)
MULTIPLIERS = {1: 0.5, 2: 0.75, 3: 1.0, 4: 1.25, 5: 1.5}
assert abs(sum(MULTIPLIERS.values()) / 5 - 1.0) < 1e-9, "Multipliers must average 1.0"


@dataclass
class InstResult:
    instrument: str
    n_oos: int
    mean_uniform: float
    mean_weighted: float
    delta: float
    paired_t: float
    paired_p: float
    per_quintile_counts: dict[int, int]
    per_quintile_mean_pnl: dict[int, float]


def _list_sessions(con: duckdb.DuckDBPyConnection, symbol: str) -> list[str]:
    rows = con.execute(
        """
        SELECT DISTINCT orb_label FROM orb_outcomes
        WHERE symbol = ? AND orb_minutes = ? AND pnl_r IS NOT NULL
        ORDER BY orb_label
        """,
        [symbol, APERTURE],
    ).fetchall()
    return [r[0] for r in rows]


def _load_cell(
    con: duckdb.DuckDBPyConnection, symbol: str, session: str
) -> pd.DataFrame:
    rel_col = f"rel_vol_{session}"
    sql = f"""
    WITH df AS (
      SELECT d.trading_day, d.symbol, d.{rel_col} AS rel_vol
      FROM daily_features d
      WHERE d.symbol = '{symbol}' AND d.orb_minutes = {APERTURE}
    )
    SELECT o.trading_day, o.pnl_r, o.entry_price, o.stop_price, df.rel_vol
    FROM orb_outcomes o
    JOIN df ON o.trading_day = df.trading_day AND o.symbol = df.symbol
    WHERE o.symbol = '{symbol}'
      AND o.orb_label = '{session}'
      AND o.orb_minutes = {APERTURE}
      AND o.entry_model = 'E2'
      AND o.confirm_bars = 1
      AND o.rr_target = {RR}
      AND o.pnl_r IS NOT NULL
    """
    df = con.sql(sql).to_df()
    if len(df) == 0:
        return df
    df["direction"] = np.where(df["entry_price"] > df["stop_price"], "long", "short")
    df["session"] = session
    df["lane"] = df["session"] + "_" + df["direction"]
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    return df


def _is_oos_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = pd.Timestamp(HOLDOUT_SACRED_FROM)
    is_df = df.loc[df["trading_day"] < cut].reset_index(drop=True)
    oos_df = df.loc[df["trading_day"] >= cut].reset_index(drop=True)
    return is_df, oos_df


def _quintile_thresholds(is_vals: np.ndarray) -> np.ndarray | None:
    """5 cutpoints → 5 buckets. Returns 4 quantile boundaries (20%, 40%, 60%, 80%)."""
    if len(is_vals) < MIN_IS_PER_LANE:
        return None
    return np.quantile(is_vals, [0.2, 0.4, 0.6, 0.8])


def _bucket(val: float, thresholds: np.ndarray) -> int:
    """Return 1..5 based on which quintile val falls in."""
    return int(np.searchsorted(thresholds, val, side="right") + 1)


def _assign_quintiles(
    is_df: pd.DataFrame, oos_df: pd.DataFrame
) -> pd.DataFrame:
    """Return OOS df with 'quintile' column [1..5]; NaN/<100 lanes get quintile=3 (uniform)."""
    if oos_df.empty:
        return oos_df
    oos = oos_df.copy()
    oos["quintile"] = 3  # default uniform for SCAN_FALLBACK lanes

    for lane, lane_is in is_df.groupby("lane"):
        is_vals = lane_is["rel_vol"].dropna().astype(float).to_numpy()
        thresholds = _quintile_thresholds(is_vals)
        if thresholds is None:
            continue  # lane too thin; OOS trades get uniform size
        mask = oos["lane"] == lane
        if not mask.any():
            continue
        oos.loc[mask, "quintile"] = (
            oos.loc[mask, "rel_vol"]
            .astype(float)
            .apply(lambda v: _bucket(v, thresholds) if not np.isnan(v) else 3)
        )
    # clip 1..5
    oos["quintile"] = oos["quintile"].clip(lower=1, upper=5).astype(int)
    return oos


def _apply_rule(oos: pd.DataFrame) -> pd.DataFrame:
    if oos.empty:
        return oos
    oos = oos.copy()
    oos["size_mult"] = oos["quintile"].map(MULTIPLIERS).astype(float)
    oos["pnl_weighted"] = oos["pnl_r"].astype(float) * oos["size_mult"]
    return oos


def _paired_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Paired one-tailed t: H1 a > b. Returns (t, p_one_tailed)."""
    diff = a - b
    if len(diff) < 2:
        return float("nan"), float("nan")
    m = float(diff.mean())
    sd = float(diff.std(ddof=1))
    if sd == 0:
        return float("nan"), float("nan")
    t = m / (sd / np.sqrt(len(diff)))
    p = float(1.0 - stats.t.cdf(t, len(diff) - 1))
    return float(t), p


def _run_instrument(con: duckdb.DuckDBPyConnection, instrument: str) -> InstResult:
    sessions = _list_sessions(con, instrument)
    frames = [sub for s in sessions if len(sub := _load_cell(con, instrument, s)) > 0]
    if not frames:
        raise RuntimeError(f"No data for {instrument}")
    full = pd.concat(frames, ignore_index=True)
    is_df, oos_df = _is_oos_split(full)
    oos_bucketed = _assign_quintiles(is_df, oos_df)
    oos_with_rule = _apply_rule(oos_bucketed)

    n = len(oos_with_rule)
    vals_uniform = oos_with_rule["pnl_r"].astype(float).to_numpy()
    vals_weighted = oos_with_rule["pnl_weighted"].astype(float).to_numpy()
    mean_uniform = float(vals_uniform.mean()) if n else float("nan")
    mean_weighted = float(vals_weighted.mean()) if n else float("nan")
    delta = mean_weighted - mean_uniform
    t_stat, p_val = _paired_t(vals_weighted, vals_uniform)

    per_q_counts = {
        q: int((oos_with_rule["quintile"] == q).sum()) for q in range(1, 6)
    }
    per_q_mean_pnl = {
        q: float(
            oos_with_rule.loc[oos_with_rule["quintile"] == q, "pnl_r"].mean()
        ) if per_q_counts[q] > 0 else float("nan")
        for q in range(1, 6)
    }

    return InstResult(
        instrument=instrument,
        n_oos=n,
        mean_uniform=mean_uniform,
        mean_weighted=mean_weighted,
        delta=delta,
        paired_t=t_stat,
        paired_p=p_val,
        per_quintile_counts=per_q_counts,
        per_quintile_mean_pnl=per_q_mean_pnl,
    )


def _verdict(r: InstResult) -> str:
    if np.isnan(r.delta) or np.isnan(r.paired_t):
        return "SCAN_ABORT"
    if r.delta < 0:
        return "SIZER_ADVERSE"
    if r.delta > 0 and r.paired_t >= PAIRED_T_PASS:
        return "SIZER_ALIVE"
    if r.delta > 0 and r.paired_t > 0:
        return "SIZER_WEAK"
    return "SIZER_NULL"


def main() -> int:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    results: list[InstResult] = []
    try:
        for inst in INSTRUMENTS:
            results.append(_run_instrument(con, inst))
    finally:
        con.close()

    RESULT_DOC.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    parts.append("# PR #48 sizer-rule OOS backtest v1\n")
    parts.append(f"**Pre-reg:** `{PREREG_PATH}`\n")
    parts.append("**Script:** `research/pr48_sizer_rule_oos_backtest_v1.py`\n")
    parts.append(
        "**Rule (pre-committed):** IS per-lane rel_vol 5-quintile thresholds, "
        "multipliers {Q1: 0.5, Q2: 0.75, Q3: 1.0, Q4: 1.25, Q5: 1.5}; "
        "mean multiplier = 1.0 (capital-neutral).\n"
    )
    parts.append(f"**Pass criterion:** delta > 0 AND paired t >= +{PAIRED_T_PASS}.\n")
    parts.append("## Headline")
    parts.append("")
    parts.append(
        "| Instrument | N_OOS | Uniform ExpR | Sizer ExpR | Delta | Paired t | p (one-tail) | Verdict |"
    )
    parts.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for r in results:
        v = _verdict(r)
        p_str = f"{r.paired_p:.4f}" if not np.isnan(r.paired_p) else "-"
        t_str = f"{r.paired_t:+.3f}" if not np.isnan(r.paired_t) else "-"
        parts.append(
            f"| {r.instrument} | {r.n_oos} | {r.mean_uniform:+.5f} | "
            f"{r.mean_weighted:+.5f} | {r.delta:+.5f} | {t_str} | {p_str} | "
            f"**{v}** |"
        )
    parts.append("")

    parts.append("## Per-quintile OOS diagnostics (rank→pnl monotonicity check)")
    parts.append("")
    parts.append("| Instrument | Q1 N / ExpR | Q2 N / ExpR | Q3 N / ExpR | Q4 N / ExpR | Q5 N / ExpR |")
    parts.append("|---|---|---|---|---|---|")
    for r in results:
        cells = []
        for q in range(1, 6):
            cnt = r.per_quintile_counts[q]
            mean = r.per_quintile_mean_pnl[q]
            mean_s = f"{mean:+.4f}" if not np.isnan(mean) else "-"
            cells.append(f"{cnt} / {mean_s}")
        parts.append(f"| {r.instrument} | " + " | ".join(cells) + " |")
    parts.append("")

    parts.append("## Summary")
    parts.append("")
    alive = [r for r in results if _verdict(r) == "SIZER_ALIVE"]
    weak = [r for r in results if _verdict(r) == "SIZER_WEAK"]
    adverse = [r for r in results if _verdict(r) == "SIZER_ADVERSE"]
    null = [r for r in results if _verdict(r) == "SIZER_NULL"]
    parts.append(f"- SIZER_ALIVE: {len(alive)} — {', '.join(r.instrument for r in alive) or 'none'}")
    parts.append(f"- SIZER_WEAK: {len(weak)} — {', '.join(r.instrument for r in weak) or 'none'}")
    parts.append(f"- SIZER_ADVERSE: {len(adverse)} — {', '.join(r.instrument for r in adverse) or 'none'}")
    parts.append(f"- SIZER_NULL: {len(null)} — {', '.join(r.instrument for r in null) or 'none'}")
    parts.append("")
    parts.append("## Interpretation")
    parts.append("")
    if alive:
        parts.append(
            f"**Sizer rule deploy-eligible on:** {', '.join(r.instrument for r in alive)}. "
            "For each ALIVE instrument: IS-trained quintile thresholds applied to fresh OOS "
            "rel_vol produce a paired-positive, statistically-significant uplift vs uniform "
            "sizing at capital-neutral budget. Next bounded step per pre_registered_criteria.md "
            "is **shadow-deployment design** — document the exact rule, the per-lane thresholds "
            "(frozen snapshot), and the shadow monitor (Shiryaev-Roberts or fixed-horizon fire-rate)."
        )
    elif weak:
        parts.append(
            f"**Sizer rule SIZER_WEAK on:** {', '.join(r.instrument for r in weak)}. "
            "Positive delta but below the pre-committed t >= +2.0 gate. Could be insufficient "
            "OOS sample or a genuine sub-threshold edge. Re-run in 6-12 months with more OOS."
        )
    if adverse:
        parts.append(
            f"**SIZER_ADVERSE on:** {', '.join(r.instrument for r in adverse)}. "
            "Rule SUBTRACTS from OOS expectancy — investigate per-quintile pattern. "
            "Likely IS/OOS regime divergence or wrong shape (e.g., inverted-U dominates, "
            "not monotonic-up). Do NOT deploy; redesign the rule shape."
        )
    if null:
        parts.append(
            f"**SIZER_NULL on:** {', '.join(r.instrument for r in null)}. "
            "Rule effectively neutral; pattern holds but quintile-linear framing doesn't "
            "capture a discriminable edge. Could try alternative rank-basis features or "
            "different multiplier curves (future pre-regs)."
        )
    parts.append("")
    parts.append("## Methodology notes")
    parts.append("")
    parts.append(
        "- IS thresholds computed per-lane (session × direction) on pre-2026-01-01 data only. "
        "OOS `rel_vol` is searchsort-bucketed into those fixed IS thresholds — no OOS information "
        "leaks into the rule."
    )
    parts.append(
        "- Lanes with fewer than 100 IS trades fall back to uniform size=1.0 (no rule applied). "
        "Prevents sparse-lane distortion."
    )
    parts.append(
        "- Paired one-tailed t-test on per-trade weighted vs uniform P&L is the correct "
        "discriminator here (same trades, different sizes). One-sample t on the mean difference."
    )
    parts.append(
        "- Multipliers {0.5, 0.75, 1.0, 1.25, 1.5} have mean 1.0 → total capital deployed is "
        "identical to uniform baseline. This is the honest capital-neutral comparison."
    )
    parts.append("")
    parts.append("## Not done by this result")
    parts.append("")
    parts.append("- No writes to validated_setups / edge_families / lane_allocation / live_config.")
    parts.append("- No deployment or capital action.")
    parts.append("- Does NOT tune the multiplier curve (pre-committed).")
    parts.append("- Does NOT test alternative rank-basis features (atr_vel_ratio, break_delay_min) — future pre-regs.")
    parts.append(
        "- Does NOT test per-direction or per-session (pooled by instrument is the pre-reg scope)."
    )

    RESULT_DOC.write_text("\n".join(parts), encoding="utf-8")

    print("PR #48 sizer-rule OOS backtest (K=3 Pathway B)")
    for r in results:
        v = _verdict(r)
        print(
            f"  {r.instrument}: N_OOS={r.n_oos}  uniform={r.mean_uniform:+.5f}  "
            f"sizer={r.mean_weighted:+.5f}  delta={r.delta:+.5f}  "
            f"t={r.paired_t:+.3f}  -> {v}"
        )
    print(f"\nRESULT_DOC: {RESULT_DOC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
