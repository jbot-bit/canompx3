"""Bounded directional state study for MNQ US_DATA_1000.

Frozen scope:
  - instrument: MNQ
  - session: US_DATA_1000
  - entry model: E2

# e2-lookahead-policy: not-predictor
# orb_US_DATA_1000_break_dir is selected as a context column (line ~75) and used
# for direction-segmentation of already-taken trades (pandas groupby, lines ~354-355).
# No WHERE predicate uses break_dir to decide whether to take a trade.
# Direction-segmentation post-entry is permitted per backtesting-methodology.md § 6.3.
  - confirm bars: 1
  - apertures: 5, 15
  - RR: 1.0, 1.5, 2.0

Frozen directional state machine:
  - confluence_both -> long
  - near_pivot -> short
  - else -> no trade

Outputs:
  docs/audit/results/2026-04-22-mnq-usdata1000-directional-state-study-v1.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from pipeline.paths import GOLD_DB_PATH
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

SEED = 20260422
OUTPUT_MD = Path("docs/audit/results/2026-04-22-mnq-usdata1000-directional-state-study-v1.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Cell:
    aperture: int
    rr: float

    @property
    def key(self) -> str:
        rr_str = f"{self.rr:.1f}"
        return f"O{self.aperture}_RR{rr_str}"


CELLS = [
    Cell(5, 1.0),
    Cell(5, 1.5),
    Cell(5, 2.0),
    Cell(15, 1.0),
    Cell(15, 1.5),
    Cell(15, 2.0),
]


def fetch_rows(symbol: str) -> pd.DataFrame:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    q = """
    WITH base AS (
        SELECT
            o.trading_day,
            o.symbol,
            o.orb_label,
            o.orb_minutes,
            o.rr_target,
            o.confirm_bars,
            o.entry_model,
            o.pnl_r,
            d.orb_US_DATA_1000_high AS orb_high,
            d.orb_US_DATA_1000_low AS orb_low,
            d.orb_US_DATA_1000_break_dir AS break_dir,
            d.prev_day_high,
            d.prev_day_low,
            d.prev_day_close,
            d.atr_20,
            d.prev_day_range,
            d.gap_open_points,
            d.atr_20_pct,
            d.overnight_range_pct
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.trading_day = d.trading_day
         AND o.symbol = d.symbol
         AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = ?
          AND o.orb_label = 'US_DATA_1000'
          AND o.entry_model = 'E2'
          AND o.confirm_bars = 1
          AND o.orb_minutes IN (5, 15)
          AND o.rr_target IN (1.0, 1.5, 2.0)
          AND o.pnl_r IS NOT NULL
    ),
    feats AS (
        SELECT *,
          (orb_high + orb_low) / 2.0 AS orb_mid,
          ((orb_high + orb_low) / 2.0 > prev_day_low AND (orb_high + orb_low) / 2.0 < prev_day_high) AS inside_pdr,
          CASE WHEN atr_20 > 0 THEN abs(((orb_high + orb_low) / 2.0) - prev_day_low) / atr_20 END AS dist_pdl_atr,
          CASE WHEN atr_20 > 0 THEN abs(((orb_high + orb_low) / 2.0) - ((prev_day_high + prev_day_low + prev_day_close) / 3.0)) / atr_20 END AS dist_pivot_atr
        FROM base
    )
    SELECT *,
      CASE
        WHEN atr_20 > 0
         AND prev_day_close IS NOT NULL
         AND (orb_mid < prev_day_low OR dist_pdl_atr < 0.15)
         AND NOT (inside_pdr OR dist_pivot_atr < 0.50)
        THEN TRUE ELSE FALSE END AS confluence_both,
      CASE
        WHEN atr_20 > 0
         AND prev_day_close IS NOT NULL
         AND dist_pivot_atr < 0.50
        THEN TRUE ELSE FALSE END AS near_pivot,
      CASE
        WHEN break_dir = 'long'
         AND atr_20 > 0
         AND prev_day_close IS NOT NULL
         AND (orb_mid < prev_day_low OR dist_pdl_atr < 0.15)
         AND NOT (inside_pdr OR dist_pivot_atr < 0.50)
        THEN TRUE
        WHEN break_dir = 'short'
         AND atr_20 > 0
         AND prev_day_close IS NOT NULL
         AND dist_pivot_atr < 0.50
        THEN TRUE
        ELSE FALSE END AS combined_policy,
      CASE
        WHEN break_dir = 'short'
         AND atr_20 > 0
         AND prev_day_close IS NOT NULL
         AND (orb_mid < prev_day_low OR dist_pdl_atr < 0.15)
         AND NOT (inside_pdr OR dist_pivot_atr < 0.50)
        THEN TRUE
        WHEN break_dir = 'long'
         AND atr_20 > 0
         AND prev_day_close IS NOT NULL
         AND dist_pivot_atr < 0.50
        THEN TRUE
        ELSE FALSE END AS inverse_policy
    FROM feats
    """
    df = con.execute(q, [symbol]).df()
    con.close()
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    return df


def one_sample(arr: np.ndarray) -> dict[str, float | int | None]:
    n = int(len(arr))
    if n == 0:
        return {"n": 0, "avg_r": None, "wr": None, "p_zero": None}
    avg = float(arr.mean())
    wr = float((arr > 0).mean())
    if n < 2 or np.isclose(arr.std(ddof=1), 0.0):
        return {"n": n, "avg_r": avg, "wr": wr, "p_zero": None}
    p_zero = float(stats.ttest_1samp(arr, 0.0).pvalue)
    return {"n": n, "avg_r": avg, "wr": wr, "p_zero": p_zero}


def diff_test(a: np.ndarray, b: np.ndarray) -> dict[str, float | None]:
    if len(a) < 2 or len(b) < 2:
        delta = float(a.mean() - b.mean()) if len(a) and len(b) else None
        return {"delta": delta, "p": None}
    res = stats.ttest_ind(a, b, equal_var=False)
    return {"delta": float(a.mean() - b.mean()), "p": float(res.pvalue)}


def t0_tautology(cell_df: pd.DataFrame) -> dict[str, object]:
    merged = cell_df[["trading_day", "combined_policy"]].drop_duplicates().copy()
    merged["pdr_r105_fire"] = ((cell_df["prev_day_range"] / cell_df["atr_20"]) >= 1.05).astype(float)
    merged["gap_r015_fire"] = ((cell_df["gap_open_points"].abs() / cell_df["atr_20"]) >= 0.015).astype(float)
    merged["atr70_fire"] = (cell_df["atr_20_pct"] >= 70).astype(float)
    merged["ovn80_fire"] = (cell_df["overnight_range_pct"] >= 80).astype(float)

    corrs: dict[str, float] = {}
    feat = cell_df["combined_policy"].astype(float)
    for col in ["pdr_r105_fire", "gap_r015_fire", "atr70_fire", "ovn80_fire"]:
        vals = merged[col]
        if len(vals) > 10 and vals.notna().sum() > 10:
            corr = feat.reset_index(drop=True).corr(vals.reset_index(drop=True))
            corrs[col] = float(0.0 if pd.isna(corr) else corr)
    max_corr = max((abs(v) for v in corrs.values()), default=0.0)
    max_name = max(corrs, key=lambda k: abs(corrs[k])) if corrs else "none"
    status = "FAIL" if max_corr > 0.70 else "PASS"
    detail = "DUPLICATE_FILTER" if status == "FAIL" else "no tautology with deployed proxies"
    return {"status": status, "value": f"max |corr|={max_corr:.3f} ({max_name})", "detail": detail, "corrs": corrs}


def t1_wr_monotonicity(cell_df: pd.DataFrame) -> dict[str, object]:
    is_df = cell_df[cell_df["trading_day"].dt.date < HOLDOUT_SACRED_FROM]
    on = is_df[is_df["combined_policy"]]
    off = is_df[~is_df["combined_policy"]]
    if len(on) < 30 or len(off) < 30:
        return {"status": "INFO", "value": "insufficient_N", "detail": "N too small"}
    wr_spread = abs((on["pnl_r"] > 0).mean() - (off["pnl_r"] > 0).mean())
    expr_spread = abs(on["pnl_r"].mean() - off["pnl_r"].mean())
    if wr_spread < 0.03 and expr_spread > 0.05:
        return {"status": "FAIL", "value": f"WR_spread={wr_spread:.3f} ExpR_spread={expr_spread:.3f}", "detail": "ARITHMETIC_ONLY"}
    if wr_spread >= 0.05:
        return {"status": "PASS", "value": f"WR_spread={wr_spread:.3f}", "detail": "signal-like WR spread"}
    return {"status": "INFO", "value": f"WR_spread={wr_spread:.3f}", "detail": "modest WR spread"}


def t2_is_baseline(cell_df: pd.DataFrame) -> dict[str, object]:
    is_df = cell_df[cell_df["trading_day"].dt.date < HOLDOUT_SACRED_FROM]
    on = is_df[is_df["combined_policy"]]["pnl_r"].to_numpy(dtype=float)
    policy_ev = float(np.where(is_df["combined_policy"], is_df["pnl_r"], 0.0).mean())
    stats_on = one_sample(on)
    status = "PASS" if stats_on["n"] >= 100 and policy_ev > 0 and (stats_on["avg_r"] or 0) > 0 else "INFO"
    return {
        "status": status,
        "value": f"N={stats_on['n']} ExpR_on={stats_on['avg_r']:+.3f} PolicyEV/day={policy_ev:+.3f}",
        "detail": "deployable sample floor met" if status == "PASS" else "positive but below deployable floor or weak EV",
    }


def t3_oos_wfe(cell_df: pd.DataFrame) -> dict[str, object]:
    is_df = cell_df[cell_df["trading_day"].dt.date < HOLDOUT_SACRED_FROM]
    oos_df = cell_df[cell_df["trading_day"].dt.date >= HOLDOUT_SACRED_FROM]
    on_is = is_df[is_df["combined_policy"]]["pnl_r"].to_numpy(dtype=float)
    on_oos = oos_df[oos_df["combined_policy"]]["pnl_r"].to_numpy(dtype=float)
    if len(on_oos) < 10:
        return {"status": "FAIL", "value": f"N_OOS={len(on_oos)}", "detail": "insufficient OOS N for WFE"}
    if len(on_is) < 2 or len(on_oos) < 2:
        return {"status": "FAIL", "value": "insufficient_var", "detail": "insufficient variance"}

    expr_is = float(on_is.mean())
    expr_oos = float(on_oos.mean())
    sr_is = expr_is / float(np.std(on_is, ddof=1))
    sr_oos = expr_oos / float(np.std(on_oos, ddof=1))
    wfe = sr_oos / sr_is if sr_is and not math.isnan(sr_is) else float("nan")
    ev_is = float(np.where(is_df["combined_policy"], is_df["pnl_r"], 0.0).mean())
    ev_oos = float(np.where(oos_df["combined_policy"], oos_df["pnl_r"], 0.0).mean())
    sign_match = np.sign(ev_is) == np.sign(ev_oos)

    if abs(wfe) > 0.95:
        status, detail = "FAIL", f"WFE={wfe:.2f} LEAKAGE_SUSPECT"
    elif wfe < 0.50:
        status, detail = "FAIL", f"WFE={wfe:.2f} < 0.50"
    elif not sign_match:
        status, detail = "FAIL", f"WFE={wfe:.2f} but IS/OOS direction mismatch"
    else:
        status, detail = "PASS", f"WFE={wfe:.2f} healthy, sign match"
    return {"status": status, "value": f"WFE={wfe:.2f} EV_IS={ev_is:+.3f} EV_OOS={ev_oos:+.3f}", "detail": detail}


def t4_sensitivity() -> dict[str, object]:
    return {"status": "INFO", "value": "N/A", "detail": "binary frozen state machine — no theta grid"}


def t5_family(current: Cell, family_summary: dict[str, dict[str, float]]) -> dict[str, object]:
    positive = sum(1 for v in family_summary.values() if v["policy_ev_pre"] > 0)
    beats_parent = sum(1 for v in family_summary.values() if v["policy_minus_parent_pre"] > 0)
    return {
        "status": "INFO",
        "value": f"{positive}/6 positive EV, {beats_parent}/6 beat parent",
        "detail": f"family breadth context for {current.key}",
    }


def t6_null_floor(cell_df: pd.DataFrame, expected_positive: bool = True, B: int = 1000) -> dict[str, object]:
    is_df = cell_df[cell_df["trading_day"].dt.date < HOLDOUT_SACRED_FROM].copy()
    n_on = int(is_df["combined_policy"].sum())
    if n_on < 30:
        return {"status": "FAIL", "value": f"N_on={n_on}", "detail": "on-signal N < 30"}

    pnl = is_df["pnl_r"].astype(float).to_numpy()
    feat = is_df["combined_policy"].astype(int).to_numpy().copy()
    observed = float(pnl[feat == 1].mean())

    beats = 0
    for i in range(B):
        rng = np.random.default_rng(SEED + i)
        shuffled = feat.copy()
        rng.shuffle(shuffled)
        boot = float(pnl[shuffled == 1].mean()) if (shuffled == 1).any() else 0.0
        if expected_positive:
            if boot >= observed:
                beats += 1
        else:
            if boot <= observed:
                beats += 1
    p_val = (beats + 1) / (B + 1)
    status = "PASS" if p_val < 0.05 else "FAIL"
    return {"status": status, "value": f"p={p_val:.4f} ExpR_obs={observed:+.3f}", "detail": f"{B} shuffles"}


def t7_per_year(cell_df: pd.DataFrame) -> dict[str, object]:
    is_df = cell_df[cell_df["trading_day"].dt.date < HOLDOUT_SACRED_FROM].copy()
    on = is_df[is_df["combined_policy"]].copy()
    if len(on) < 30:
        return {"status": "FAIL", "value": f"N_on={len(on)}", "detail": "N insufficient"}
    on["year"] = on["trading_day"].dt.year
    yearly = {}
    for year, grp in on.groupby("year"):
        if len(grp) >= 5:
            yearly[int(year)] = float(grp["pnl_r"].mean())
        else:
            yearly[int(year)] = None
    testable = sum(1 for v in yearly.values() if v is not None)
    positive = sum(1 for v in yearly.values() if v is not None and v > 0)
    if testable == 0:
        return {"status": "FAIL", "value": "no_testable_years", "detail": "no year had N>=5"}
    frac = positive / testable
    status = "PASS" if frac >= 0.70 else ("INFO" if frac >= 0.50 else "FAIL")
    return {"status": status, "value": f"{positive}/{testable} positive years", "detail": str(yearly)}


def t8_cross_instrument(cell_df_twin: pd.DataFrame) -> dict[str, object]:
    is_df = cell_df_twin[cell_df_twin["trading_day"].dt.date < HOLDOUT_SACRED_FROM]
    on = is_df[is_df["combined_policy"]]["pnl_r"].to_numpy(dtype=float)
    off = is_df[~is_df["combined_policy"]]["pnl_r"].to_numpy(dtype=float)
    if len(on) < 30 or len(off) < 30:
        return {"status": "INFO", "value": f"N_on={len(on)} N_off={len(off)}", "detail": "twin N insufficient"}
    delta = float(on.mean() - off.mean())
    status = "PASS" if delta > 0.05 else "FAIL"
    return {"status": status, "value": f"Δ_twin={delta:+.3f}", "detail": "MES twin sign+mag check"}


def bh_adjust(pvals: dict[str, float | None]) -> dict[str, float | None]:
    items = [(k, v) for k, v in pvals.items() if v is not None]
    m = len(items)
    ranked = sorted(items, key=lambda kv: kv[1])
    adjusted: dict[str, float] = {}
    prev = 1.0
    for i, (key, p) in enumerate(reversed(ranked), start=1):
        rank = m - i + 1
        val = min(prev, p * m / rank)
        prev = val
        adjusted[key] = val
    return {k: adjusted.get(k) for k in pvals}


def summarize_cell(cell: Cell, df: pd.DataFrame, twin_df: pd.DataFrame, family_summary: dict[str, dict[str, float]]) -> dict[str, object]:
    cell_df = df[(df["orb_minutes"] == cell.aperture) & (df["rr_target"] == cell.rr)].copy()
    twin_cell_df = twin_df[(twin_df["orb_minutes"] == cell.aperture) & (twin_df["rr_target"] == cell.rr)].copy()
    is_mask = cell_df["trading_day"].dt.date < HOLDOUT_SACRED_FROM
    oos_mask = ~is_mask

    parent_pre = cell_df.loc[is_mask, "pnl_r"].to_numpy(dtype=float)
    parent_oos = cell_df.loc[oos_mask, "pnl_r"].to_numpy(dtype=float)
    on_pre = cell_df.loc[is_mask & cell_df["combined_policy"], "pnl_r"].to_numpy(dtype=float)
    on_oos = cell_df.loc[oos_mask & cell_df["combined_policy"], "pnl_r"].to_numpy(dtype=float)

    policy_ev_pre = float(np.where(cell_df.loc[is_mask, "combined_policy"], cell_df.loc[is_mask, "pnl_r"], 0.0).mean())
    policy_ev_oos = float(np.where(cell_df.loc[oos_mask, "combined_policy"], cell_df.loc[oos_mask, "pnl_r"], 0.0).mean()) if oos_mask.any() else 0.0
    inverse_ev_pre = float(np.where(cell_df.loc[is_mask, "inverse_policy"], cell_df.loc[is_mask, "pnl_r"], 0.0).mean())
    parent_avg_pre = float(parent_pre.mean())
    parent_avg_oos = float(parent_oos.mean()) if len(parent_oos) else 0.0

    conf_long = cell_df.loc[is_mask & (cell_df["break_dir"] == "long") & (cell_df["confluence_both"]), "pnl_r"].to_numpy(dtype=float)
    conf_short = cell_df.loc[is_mask & (cell_df["break_dir"] == "short") & (cell_df["confluence_both"]), "pnl_r"].to_numpy(dtype=float)
    np_short = cell_df.loc[is_mask & (cell_df["break_dir"] == "short") & (cell_df["near_pivot"]), "pnl_r"].to_numpy(dtype=float)
    np_long = cell_df.loc[is_mask & (cell_df["break_dir"] == "long") & (cell_df["near_pivot"]), "pnl_r"].to_numpy(dtype=float)

    tests = {
        "T0": t0_tautology(cell_df.loc[is_mask].copy()),
        "T1": t1_wr_monotonicity(cell_df),
        "T2": t2_is_baseline(cell_df),
        "T3": t3_oos_wfe(cell_df),
        "T4": t4_sensitivity(),
        "T5": t5_family(cell, family_summary),
        "T6": t6_null_floor(cell_df),
        "T7": t7_per_year(cell_df),
        "T8": t8_cross_instrument(twin_cell_df),
    }
    adversarial = {
        "policy_minus_parent_pre": policy_ev_pre - parent_avg_pre,
        "policy_minus_inverse_pre": policy_ev_pre - inverse_ev_pre,
        "confluence_long_minus_short_pre": (
            float(conf_long.mean() - conf_short.mean()) if len(conf_long) and len(conf_short) else None
        ),
        "near_pivot_short_minus_long_pre": (
            float(np_short.mean() - np_long.mean()) if len(np_short) and len(np_long) else None
        ),
    }

    fail_names = [name for name, result in tests.items() if result["status"] == "FAIL"]
    if tests["T6"]["status"] == "FAIL" or policy_ev_pre <= 0 or (np.isnan(policy_ev_pre) if isinstance(policy_ev_pre, float) else False):
        verdict = "KILL"
    elif policy_ev_pre <= parent_avg_pre:
        verdict = "PARK"
    elif fail_names:
        verdict = "PARK"
    else:
        verdict = "CONTINUE"

    return {
        "cell": cell,
        "parent_pre": one_sample(parent_pre),
        "parent_oos": one_sample(parent_oos),
        "on_pre": one_sample(on_pre),
        "on_oos": one_sample(on_oos),
        "policy_ev_pre": policy_ev_pre,
        "policy_ev_oos": policy_ev_oos,
        "inverse_ev_pre": inverse_ev_pre,
        "parent_avg_pre": parent_avg_pre,
        "parent_avg_oos": parent_avg_oos,
        "tests": tests,
        "adversarial": adversarial,
        "verdict": verdict,
    }


def family_policy_summary(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for cell in CELLS:
        cell_df = df[(df["orb_minutes"] == cell.aperture) & (df["rr_target"] == cell.rr)].copy()
        is_df = cell_df[cell_df["trading_day"].dt.date < HOLDOUT_SACRED_FROM]
        policy_ev_pre = float(np.where(is_df["combined_policy"], is_df["pnl_r"], 0.0).mean())
        parent_avg_pre = float(is_df["pnl_r"].mean())
        out[cell.key] = {
            "policy_ev_pre": policy_ev_pre,
            "parent_avg_pre": parent_avg_pre,
            "policy_minus_parent_pre": policy_ev_pre - parent_avg_pre,
        }
    return out


def portfolio_ev_comparison(best_result: dict[str, object]) -> tuple[str, str]:
    best_delta = float(best_result["adversarial"]["policy_minus_parent_pre"])  # type: ignore[index]
    if best_delta > 0:
        return (
            "competitive",
            "This path beats its own parent comparator on pre-2026 policy EV and may justify competing with the current queue leader.",
        )
    return (
        "inferior",
        "This path does not beat its own parent comparator on pre-2026 policy EV, so it remains portfolio-EV inferior to the queue leader `PR48 MES/MGC deployable sizer rule`.",
    )


def emit(results: list[dict[str, object]], family_bh: dict[str, float | None], comparison_label: str, comparison_text: str) -> None:
    lines: list[str] = [
        "# MNQ US_DATA_1000 Bounded Directional State Study",
        "",
        "**Date:** 2026-04-22",
        "**Pre-reg:** `docs/audit/hypotheses/2026-04-22-mnq-usdata1000-directional-state-study-v1.yaml`",
        "**Canonical truth layers:** `daily_features`, `orb_outcomes`",
        "**Holdout policy:** Mode A, `2026-01-01` sacred",
        "",
        "## Frozen State Machine",
        "",
        "- `confluence_both -> long`",
        "- `near_pivot -> short`",
        "- `else -> no trade`",
        "",
        "## Raw State Matrix",
        "",
        "| Cell | Parent IS AvgR | Policy IS AvgR(on) | Policy EV/day IS | Parent OOS AvgR | Policy EV/day OOS | Inverse EV/day IS | Verdict |",
        "|------|----------------|--------------------|------------------|-----------------|-------------------|-------------------|---------|",
    ]
    for res in results:
        cell = res["cell"]
        lines.append(
            f"| {cell.key} | {res['parent_avg_pre']:+.3f} | {res['on_pre']['avg_r']:+.3f} | {res['policy_ev_pre']:+.3f} | "
            f"{res['parent_avg_oos']:+.3f} | {res['policy_ev_oos']:+.3f} | {res['inverse_ev_pre']:+.3f} | **{res['verdict']}** |"
        )

    lines += [
        "",
        "## T0-T8 Audit",
        "",
    ]
    for res in results:
        cell = res["cell"]
        bh_val = family_bh.get(cell.key)
        lines += [
            f"### {cell.key}",
            "",
            f"**BH-FDR q (family K=6) on on-signal mean-vs-zero p:** `{bh_val:.4f}`" if bh_val is not None else "**BH-FDR q:** `N/A`",
            "",
            "| Test | Value | Status | Detail |",
            "|------|-------|--------|--------|",
        ]
        for name, result in res["tests"].items():
            lines.append(f"| {name} | {result['value']} | **{result['status']}** | {result['detail']} |")
        lines += [
            "",
            "| Adversarial Check | Value |",
            "|-------------------|-------|",
            f"| policy_minus_parent_pre | {res['adversarial']['policy_minus_parent_pre']:+.3f} |",
            f"| policy_minus_inverse_pre | {res['adversarial']['policy_minus_inverse_pre']:+.3f} |",
            f"| confluence_long_minus_short_pre | {res['adversarial']['confluence_long_minus_short_pre']} |",
            f"| near_pivot_short_minus_long_pre | {res['adversarial']['near_pivot_short_minus_long_pre']} |",
            "",
        ]

    lines += [
        "## IS/OOS Direction Match",
        "",
        "| Cell | IS Policy EV/day | OOS Policy EV/day | Direction Match |",
        "|------|------------------|-------------------|-----------------|",
    ]
    for res in results:
        direction_match = np.sign(float(res["policy_ev_pre"])) == np.sign(float(res["policy_ev_oos"]))
        lines.append(
            f"| {res['cell'].key} | {res['policy_ev_pre']:+.3f} | {res['policy_ev_oos']:+.3f} | {direction_match} |"
        )

    lines += [
        "",
        "## Portfolio EV Comparison",
        "",
        f"- Queue leader from `docs/plans/2026-04-22-recent-pr-followthrough-queue.md`: `PR48 MES/MGC deployable sizer rule`.",
        f"- Comparison verdict: **{comparison_label.upper()}**.",
        f"- {comparison_text}",
        "",
        "## Bottom Line",
        "",
    ]
    best = max(results, key=lambda r: float(r["policy_ev_pre"]))
    if all(res["verdict"] == "KILL" for res in results):
        lines.append("- Family verdict: **KILL**. No cell survives the bounded directional story.")
    elif any(res["verdict"] == "CONTINUE" for res in results):
        lines.append("- Family verdict: **CONTINUE**. At least one cell clears the bounded study and parent comparator.")
    else:
        lines.append("- Family verdict: **PARK**. The state machine improves selected-trade quality but does not beat the raw parent on portfolio EV.")
    lines.append(f"- Best pre-2026 policy EV/day cell: `{best['cell'].key}` at `{best['policy_ev_pre']:+.3f}R/day` vs parent `{best['parent_avg_pre']:+.3f}R/day`.")
    lines.append("- No OOS tuning was performed. 2026 results are descriptive only.")

    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] {OUTPUT_MD}")


def main() -> None:
    mnq = fetch_rows("MNQ")
    mes = fetch_rows("MES")
    family_summary = family_policy_summary(mnq)
    results = [summarize_cell(cell, mnq, mes, family_summary) for cell in CELLS]
    pvals = {res["cell"].key: res["on_pre"]["p_zero"] for res in results}
    family_bh = bh_adjust(pvals)
    best = max(results, key=lambda r: float(r["policy_ev_pre"]))
    comparison_label, comparison_text = portfolio_ev_comparison(best)
    emit(results, family_bh, comparison_label, comparison_text)


if __name__ == "__main__":
    main()
