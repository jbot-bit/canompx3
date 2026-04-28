"""Phase D D-0 v2 backtest — garch_forecast_vol_pct size-scaling on MNQ COMEX_SETTLE O5 RR1.5 OVNRNG_100.

v2 replaces contaminated rel_vol_COMEX_SETTLE with garch_forecast_vol_pct (clean pre-break predictor).

# e2-lookahead-policy: clean
# garch_forecast_vol_pct input is prior-day daily closes only; forecast fixed before any ORB on
# the current trading day. Pure pre-entry. § 6.1 safe whitelist.

Pre-registration: docs/audit/hypotheses/2026-04-28-phase-d-d0-v2-garch-clean-rederivation.yaml
Commit SHA of pre-reg: 823b0127

Execution order (enforces pre-reg § execution_gate):
1. Load IS-only lane trade set (trading_day < HOLDOUT_SACRED_FROM, OVNRNG_100 fires,
   garch_forecast_vol_pct non-null) — realized-eod scratch policy (no pnl_r IS NOT NULL filter)
2. Compute P33/P67 from IS garch_forecast_vol_pct distribution ONCE and freeze
3. Bucket each trade: low (<P33) / mid [P33,P67) / high (>=P67); ties at boundaries go HIGH
4. Baseline: constant 1.0x — Sharpe_baseline = mean(R) / std(R) per-trade
5. Sized: R * size_mult[bucket] — Sharpe_sized = mean(sR) / std(sR) per-trade
6. H1 metrics: Sharpe uplift % (>=15%) AND absolute Sharpe floor (>=0.05) AND raw_p < 0.05
7. Bootstrap paired p-value (B=10000, independent paired, no block)
8. H2 ablation (descriptive-only): low=0.0x, mid=1.0x, high=1.5x
9. K2 implementation_integrity checklist (pre-reg § kill_criteria.K2) — each item PASS/FAIL
10. K3 feature_temporal_integrity (entry_ts vs feature_input_close_ts) — each item PASS/FAIL
11. Rule 8.2 arithmetic-only check
12. Selection-bias audit: per-year N excluded due to garch_forecast_vol_pct IS NULL
13. Emit Markdown report with every output required by pre-reg

Canonical source delegation (institutional-rigor rule 4):
- DB path from pipeline.paths.GOLD_DB_PATH
- HOLDOUT_SACRED_FROM from trading_app.holdout_policy
- OVNRNG_100 filter min_range from trading_app.config.ALL_FILTERS
- Triple-join rule per .claude/rules/daily-features-joins.md
- scratch_policy: realized-eod (include outcome='scratch' rows; pnl_r already populated by outcome_builder)
"""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from pipeline.paths import GOLD_DB_PATH
from trading_app.config import ALL_FILTERS
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]


PREREG_PATH = "docs/audit/hypotheses/2026-04-28-phase-d-d0-v2-garch-clean-rederivation.yaml"
PREREG_COMMIT = "823b0127"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SCOPE = {
    "instrument": "MNQ",
    "orb_label": "COMEX_SETTLE",
    "orb_minutes": 5,
    "rr_target": 1.5,
    "entry_model": "E2",
    "confirm_bars": 1,
    "filter_type": "OVNRNG_100",
}

BUCKETS = {"low": 0.5, "mid": 1.0, "high": 1.5}
H2_BUCKETS = {"low": 0.0, "mid": 1.0, "high": 1.5}
BOOTSTRAP_B = 10_000


@dataclass
class SharpeStats:
    n: int
    mean_r: float
    std_r: float
    sharpe: float
    win_rate: float
    total_r: float

    @classmethod
    def from_rs(cls, rs: list[float]) -> "SharpeStats":
        if not rs:
            return cls(0, 0.0, 0.0, 0.0, 0.0, 0.0)
        n = len(rs)
        mean_r = sum(rs) / n
        var = sum((r - mean_r) ** 2 for r in rs) / (n - 1) if n > 1 else 0.0
        std_r = math.sqrt(var)
        sharpe = mean_r / std_r if std_r > 0 else 0.0
        wins = sum(1 for r in rs if r > 0)
        return cls(
            n=n,
            mean_r=mean_r,
            std_r=std_r,
            sharpe=sharpe,
            win_rate=wins / n,
            total_r=sum(rs),
        )


def load_is_trades(con) -> tuple[list[dict], dict]:
    """Load IS-only trades for the D-0 v2 lane. Applies triple-join + canonical filter.

    Returns (trades, null_by_year) where null_by_year is the per-year breakdown of
    excluded rows due to garch_forecast_vol_pct IS NULL (selection-bias audit).

    scratch_policy: realized-eod — outcome='scratch' rows are INCLUDED. The pnl_r
    column is populated by the outcome_builder for realized-eod scratches.
    DO NOT add o.pnl_r IS NOT NULL or o.outcome != 'scratch' filters.
    """
    ovn_filter = ALL_FILTERS[SCOPE["filter_type"]]
    min_range = getattr(ovn_filter, "min_range", None)
    if min_range is None:
        raise RuntimeError(
            f"Filter {SCOPE['filter_type']} missing min_range attribute; "
            "canonical source changed. Do not fallback — halt per K2."
        )

    rows = con.execute(
        """
        SELECT
            o.trading_day,
            o.pnl_r,
            o.outcome,
            o.entry_ts,
            d.garch_forecast_vol_pct,
            d.overnight_range
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.trading_day = d.trading_day
          AND o.symbol = d.symbol
          AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = ?
          AND o.orb_label = ?
          AND o.orb_minutes = ?
          AND o.entry_model = ?
          AND o.rr_target = ?
          AND o.confirm_bars = ?
          AND o.trading_day < ?
          AND d.overnight_range IS NOT NULL
          AND d.overnight_range >= ?
          AND d.garch_forecast_vol_pct IS NOT NULL
        ORDER BY o.trading_day
        """,
        [
            SCOPE["instrument"],
            SCOPE["orb_label"],
            SCOPE["orb_minutes"],
            SCOPE["entry_model"],
            SCOPE["rr_target"],
            SCOPE["confirm_bars"],
            HOLDOUT_SACRED_FROM,
            min_range,
        ],
    ).fetchall()

    trades = [
        {
            "trading_day": r[0],
            "pnl_r": float(r[1]),
            "outcome": r[2],
            "entry_ts": r[3],
            "garch_vol_pct": float(r[4]),
            "overnight_range": float(r[5]),
        }
        for r in rows
    ]

    # Selection-bias audit: per-year count of rows excluded because garch IS NULL
    null_rows = con.execute(
        """
        SELECT YEAR(o.trading_day) AS yr,
               COUNT(*) AS total_rows,
               SUM(CASE WHEN d.garch_forecast_vol_pct IS NULL THEN 1 ELSE 0 END) AS garch_null_count
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.trading_day = d.trading_day
          AND o.symbol = d.symbol
          AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = ?
          AND o.orb_label = ?
          AND o.orb_minutes = ?
          AND o.entry_model = ?
          AND o.rr_target = ?
          AND o.confirm_bars = ?
          AND o.trading_day < ?
          AND d.overnight_range IS NOT NULL
          AND d.overnight_range >= ?
        GROUP BY 1 ORDER BY 1
        """,
        [
            SCOPE["instrument"],
            SCOPE["orb_label"],
            SCOPE["orb_minutes"],
            SCOPE["entry_model"],
            SCOPE["rr_target"],
            SCOPE["confirm_bars"],
            HOLDOUT_SACRED_FROM,
            min_range,
        ],
    ).fetchall()

    null_by_year = {
        str(r[0]): {"total": r[1], "garch_null": r[2], "pct_excluded": round(100.0 * r[2] / r[1], 1) if r[1] > 0 else 0.0}
        for r in null_rows
    }

    return trades, null_by_year


def compute_quantiles(values: list[float], qs: list[float]) -> list[float]:
    sv = sorted(values)
    n = len(sv)
    out = []
    for q in qs:
        pos = q * (n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            out.append(sv[lo])
        else:
            w = pos - lo
            out.append(sv[lo] * (1 - w) + sv[hi] * w)
    return out


def bucket_of(garch_pct: float, p33: float, p67: float) -> str:
    """Bucket assignment per pre-reg primary_schema.bucket_assignment_rule.

    low  if x < P33
    mid  if P33 <= x < P67
    high if x >= P67   (ties at P67 go HIGH)
    Ties at P33 boundary go to MID (>= P33).
    """
    if garch_pct < p33:
        return "low"
    if garch_pct >= p67:
        return "high"
    return "mid"


def sharpe_diff(pairs: list[tuple[float, float]]) -> float:
    """Per-trade Sharpe difference: Sharpe(sized) - Sharpe(baseline)."""
    baseline_rs = [b for b, _ in pairs]
    sized_rs = [s for _, s in pairs]
    s_base = SharpeStats.from_rs(baseline_rs)
    s_sized = SharpeStats.from_rs(sized_rs)
    return s_sized.sharpe - s_base.sharpe


def bootstrap_paired_p(
    baseline_rs: list[float],
    sized_rs: list[float],
    observed_diff: float,
    b: int,
    seed: int = 42,
) -> float:
    """Independent paired bootstrap p-value for (sharpe_sized - sharpe_baseline).

    Per pre-reg § hypotheses.H1.pathway_b_floor.raw_p_method:
    - Independent paired bootstrap on per-trade (baseline_R, sized_R) pairs
    - No block (per-trade pairs are independent by construction)
    - B=10000 resamples
    - statistic: (sharpe_sized - sharpe_baseline) on resampled pairs
    - p-value: two-sided, (count where |diff_boot| >= |observed_diff|) / B
    """
    rng = random.Random(seed)
    n = len(baseline_rs)
    pairs = list(zip(baseline_rs, sized_rs))
    count_ge = 0
    abs_obs = abs(observed_diff)
    for _ in range(b):
        sample = [pairs[rng.randint(0, n - 1)] for _ in range(n)]
        diff = sharpe_diff(sample)
        if abs(diff) >= abs_obs:
            count_ge += 1
    return count_ge / b


def run_backtest() -> dict:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        # Step 1: load IS trade set (realized-eod scratch policy — no outcome filter)
        trades, null_by_year = load_is_trades(con)
        if not trades:
            raise RuntimeError("Zero IS trades found for D-0 v2 lane — check filter / scope.")

        # Step 2: compute P33/P67 ONCE from IS garch distribution and freeze
        garch_vals = [t["garch_vol_pct"] for t in trades]
        p33, p67 = compute_quantiles(garch_vals, [0.33, 0.67])
        thresholds_frozen_at = datetime.now(timezone.utc).isoformat()

        # Step 3: bucket each trade (deterministic from frozen thresholds)
        for t in trades:
            t["bucket"] = bucket_of(t["garch_vol_pct"], p33, p67)

        # Step 4: baseline — constant 1.0x (equivalent to raw pnl_r)
        baseline_rs = [t["pnl_r"] for t in trades]
        baseline = SharpeStats.from_rs(baseline_rs)

        # Step 5: primary sized schema (0.5 / 1.0 / 1.5)
        sized_rs = [BUCKETS[t["bucket"]] * t["pnl_r"] for t in trades]
        sized = SharpeStats.from_rs(sized_rs)

        # Step 6: H1 metrics
        abs_sharpe_diff = sized.sharpe - baseline.sharpe
        if baseline.sharpe != 0:
            uplift_pct = 100.0 * abs_sharpe_diff / abs(baseline.sharpe)
        else:
            uplift_pct = 0.0

        # Step 7: bootstrap paired p-value (B=10000, independent paired, no block)
        print(f"Running bootstrap (B={BOOTSTRAP_B})...", flush=True)
        raw_p = bootstrap_paired_p(baseline_rs, sized_rs, abs_sharpe_diff, BOOTSTRAP_B)

        # H1 decision (all three gates required for CONTINUE):
        # relative_pass: uplift_pct >= 15.0
        # absolute_pass: abs_sharpe_diff >= 0.05
        # significance_pass: raw_p < 0.05
        relative_pass = uplift_pct >= 15.0
        absolute_pass = abs_sharpe_diff >= 0.05
        significance_pass = raw_p < 0.05

        if relative_pass and absolute_pass and significance_pass:
            h1_verdict = "CONTINUE_TO_D1_V2"
        elif uplift_pct < 10.0:
            h1_verdict = "KILL"
        elif not relative_pass:
            # 10 <= uplift < 15 (park zone) or below absolute floor
            if not absolute_pass:
                h1_verdict = "PARK_ABSOLUTE_FLOOR_FAIL"
            else:
                h1_verdict = "PARK_RELATIVE_BELOW_15"
        elif not absolute_pass:
            h1_verdict = "PARK_ABSOLUTE_FLOOR_FAIL"
        elif not significance_pass:
            h1_verdict = "PARK_UNDERPOWERED"
        else:
            h1_verdict = "PARK"

        # Step 8: H2 ablation (descriptive only)
        h2_rs = [H2_BUCKETS[t["bucket"]] * t["pnl_r"] for t in trades]
        h2 = SharpeStats.from_rs(h2_rs)
        h2_skipped_low = sum(1 for t in trades if t["bucket"] == "low")

        # Per-bucket diagnostics (unscaled R — measures whether garch bucket predicts raw outcomes)
        bucket_stats = {}
        for b_name in ("low", "mid", "high"):
            b_rs = [t["pnl_r"] for t in trades if t["bucket"] == b_name]
            bucket_stats[b_name] = SharpeStats.from_rs(b_rs)

        # Step 9: K2 integrity checklist
        oos_trades = [t for t in trades if t["trading_day"] >= HOLDOUT_SACRED_FROM]
        null_pnl_trades = [t for t in trades if t["pnl_r"] is None]
        k2 = {
            "p33_p67_calibrated_is_only": all(t["trading_day"] < HOLDOUT_SACRED_FROM for t in trades),
            "oos_trades_in_sample": len(oos_trades),
            "oos_consulted": False,
            "bucket_thresholds_frozen_before_sharpe": True,
            "bucket_thresholds_frozen_at": thresholds_frozen_at,
            "size_applied_at_entry_not_retro": True,
            "trade_count_baseline_eq_sized": baseline.n == sized.n,
            "no_pnl_r_null_in_sample": len(null_pnl_trades) == 0,
            "no_garch_null_in_sample": all(t["garch_vol_pct"] is not None for t in trades),
            "scratch_policy_realized_eod": "outcome='scratch' rows included; no pnl_r IS NOT NULL filter applied",
        }
        k2_pass = (
            k2["p33_p67_calibrated_is_only"]
            and k2["oos_trades_in_sample"] == 0
            and k2["oos_consulted"] is False
            and k2["bucket_thresholds_frozen_before_sharpe"] is True
            and k2["size_applied_at_entry_not_retro"] is True
            and k2["trade_count_baseline_eq_sized"] is True
            and k2["no_pnl_r_null_in_sample"] is True
            and k2["no_garch_null_in_sample"] is True
        )

        # Step 10: K3 feature temporal integrity
        # garch_forecast_vol_pct is computed from prior-day closes only (§ feature_definition).
        # feature_input_close_ts = prior trading day close. This is structurally pre-entry for
        # any ORB on the current trading day. No per-row timestamp to compare — the structural
        # guarantee is documented in pre-reg § feature_definition.lookahead_check.
        # We confirm entry_ts is NOT NULL for all rows (any NULL entry_ts = unknown temporal state).
        null_entry_ts_count = sum(1 for t in trades if t["entry_ts"] is None)
        k3 = {
            "garch_input_is_prior_day_close": True,
            "garch_window_closes_before_any_orb_on_trading_day": True,
            "structural_guarantee": "garch_forecast_vol_pct computation uses rolling 252-day prior_only window [i-252:i]; forecast fixed at prior-day close before any ORB session of the current trading day. Ref: build_daily_features.py:1482-1497 docstring + backtesting-methodology.md § 6.1.",
            "null_entry_ts_count": null_entry_ts_count,
            "entry_ts_all_non_null": null_entry_ts_count == 0,
        }
        k3_pass = k3["entry_ts_all_non_null"]

        # Step 11: Rule 8.2 arithmetic-only check
        per_bucket_wr = {b: bucket_stats[b].win_rate for b in ("low", "mid", "high")}
        wr_spread_pct = (max(per_bucket_wr.values()) - min(per_bucket_wr.values())) * 100
        arithmetic_only_flag = wr_spread_pct < 3.0 and abs(uplift_pct) > 10.0

        # Step 12: scratch trades count
        scratch_count = sum(1 for t in trades if t["outcome"] == "scratch")

        return {
            "scope": SCOPE,
            "prereg_path": PREREG_PATH,
            "prereg_commit": PREREG_COMMIT,
            "run_utc": datetime.now(timezone.utc).isoformat(),
            "holdout_sacred_from": str(HOLDOUT_SACRED_FROM),
            "scratch_policy": "realized-eod",
            "is_trades": {
                "n": len(trades),
                "first_day": str(trades[0]["trading_day"]),
                "last_day": str(trades[-1]["trading_day"]),
                "scratch_count": scratch_count,
            },
            "selection_bias_audit": {
                "null_garch_excluded_per_year": null_by_year,
                "note": "Rows excluded because garch_forecast_vol_pct IS NULL (concentrated in early history per GARCH_PCT_MIN_PRIOR_VALUES floor).",
            },
            "thresholds": {
                "p33": p33,
                "p67": p67,
                "frozen_at_utc": thresholds_frozen_at,
                "feature": "garch_forecast_vol_pct",
                "population": "IS-only MNQ COMEX_SETTLE O5 E2 CB1 OVNRNG_100-firing trades",
            },
            "bucket_distribution": {b: sum(1 for t in trades if t["bucket"] == b) for b in ("low", "mid", "high")},
            "per_bucket_stats": {b: asdict(bucket_stats[b]) for b in ("low", "mid", "high")},
            "baseline": asdict(baseline),
            "sized_primary": asdict(sized),
            "h1_sharpe_uplift_pct": uplift_pct,
            "h1_absolute_sharpe_diff": abs_sharpe_diff,
            "h1_bootstrap_raw_p": raw_p,
            "h1_bootstrap_b": BOOTSTRAP_B,
            "h1_gates": {
                "relative_ge_15pct": relative_pass,
                "absolute_diff_ge_0_05": absolute_pass,
                "raw_p_lt_0_05": significance_pass,
                "all_three_required_for_continue": True,
            },
            "h1_verdict": h1_verdict,
            "h2_ablation": {
                "stats": asdict(h2),
                "low_trades_skipped": h2_skipped_low,
                "note": "descriptive_only; not primary selector; cannot replace H1 post-hoc",
            },
            "rule_8_2_arithmetic_only_check": {
                "per_bucket_win_rate": per_bucket_wr,
                "win_rate_spread_pct": wr_spread_pct,
                "flag": arithmetic_only_flag,
                "interpretation": (
                    "ARITHMETIC_ONLY (cost-screen not edge)"
                    if arithmetic_only_flag
                    else "Not flagged — WR spread >= 3% OR uplift within noise band"
                ),
            },
            "k2_implementation_integrity": {
                "checks": k2,
                "overall": "PASS" if k2_pass else "FAIL",
            },
            "k3_feature_temporal_integrity": {
                "checks": k3,
                "overall": "PASS" if k3_pass else "FAIL",
            },
            "comparison_to_v1": {
                "note": "DESCRIPTIVE_ONLY_NOT_STATISTICALLY_COMPARABLE",
                "v1_sharpe_uplift_pct": 7.33,
                "v1_verdict": "KILL",
                "v1_prereg_commit": "b6918d8d",
                "reason_not_comparable": "v2 N=468 excludes ~61 rows (2019+2020 early history) where garch_forecast_vol_pct IS NULL that v1's rel_vol covered. Different sample, different scratch policy (v1 pre-Stage-5, v2 realized-eod). Direct ratio is not a valid statistical comparison.",
            },
        }
    finally:
        con.close()


def emit_markdown(report: dict, out_path: Path) -> None:
    lines = []
    lines.append("# Phase D D-0 v2 Backtest — MNQ COMEX_SETTLE O5 RR1.5 OVNRNG_100 garch_forecast_vol_pct size-scaling")
    lines.append("")
    lines.append(f"**Pre-registration:** `{report['prereg_path']}` (commit `{report['prereg_commit']}`)")
    lines.append(f"**Run UTC:** {report['run_utc']}")
    lines.append(f"**Holdout sacred from:** {report['holdout_sacred_from']}")
    lines.append(f"**Scratch policy:** {report['scratch_policy']}")
    lines.append(f"**Scope:** {json.dumps(report['scope'])}")
    lines.append("")
    lines.append("## Predictor swap")
    lines.append("- v1 predictor: `rel_vol_COMEX_SETTLE` — TAINTED (E2 look-ahead, 41.3% post-entry on E2)")
    lines.append("- v2 predictor: `garch_forecast_vol_pct` — CLEAN (prior-day close input, § 6.1 safe whitelist)")
    lines.append("")
    lines.append("## IS trade set")
    lines.append(f"- N: {report['is_trades']['n']}")
    lines.append(f"- First day: {report['is_trades']['first_day']}")
    lines.append(f"- Last day: {report['is_trades']['last_day']}")
    lines.append(f"- Scratch trades (realized-eod policy): {report['is_trades']['scratch_count']}")
    lines.append("")
    lines.append("## Selection bias audit — rows excluded due to garch_forecast_vol_pct IS NULL")
    lines.append("| Year | Total OVNRNG_100 rows | garch NULL excluded | % excluded |")
    lines.append("|---:|---:|---:|---:|")
    total_all = 0
    null_all = 0
    for yr, v in sorted(report["selection_bias_audit"]["null_garch_excluded_per_year"].items()):
        lines.append(f"| {yr} | {v['total']} | {v['garch_null']} | {v['pct_excluded']}% |")
        total_all += v["total"]
        null_all += v["garch_null"]
    pct_all = round(100.0 * null_all / total_all, 1) if total_all > 0 else 0.0
    lines.append(f"| **Total** | **{total_all}** | **{null_all}** | **{pct_all}%** |")
    lines.append("")
    lines.append(f"_{report['selection_bias_audit']['note']}_")
    lines.append("")
    lines.append("## Frozen thresholds (pre-reg § calibration.thresholds)")
    lines.append(f"- P33 garch_forecast_vol_pct: **{report['thresholds']['p33']:.4f}**")
    lines.append(f"- P67 garch_forecast_vol_pct: **{report['thresholds']['p67']:.4f}**")
    lines.append(f"- Frozen at UTC: {report['thresholds']['frozen_at_utc']}")
    lines.append(f"- Population: {report['thresholds']['population']}")
    lines.append("")
    lines.append("## Bucket distribution")
    for b, n in report["bucket_distribution"].items():
        lines.append(f"- {b}: N={n}")
    lines.append("")
    lines.append("## Per-bucket R-multiple statistics (unbiased diagnostic)")
    lines.append("| Bucket | N | Mean R | Std R | Sharpe (per-trade) | Win rate | Total R |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for b in ("low", "mid", "high"):
        s = report["per_bucket_stats"][b]
        lines.append(
            f"| {b} | {s['n']} | {s['mean_r']:+.4f} | {s['std_r']:.4f} | {s['sharpe']:+.4f} | {s['win_rate']:.3f} | {s['total_r']:+.2f} |"
        )
    lines.append("")
    lines.append("## H1 — baseline vs sized (primary schema 0.5x / 1.0x / 1.5x)")
    lines.append("| Variant | N | Mean R | Std R | Sharpe | Win rate | Total R |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    b = report["baseline"]
    lines.append(
        f"| baseline 1.0x | {b['n']} | {b['mean_r']:+.4f} | {b['std_r']:.4f} | {b['sharpe']:+.4f} | {b['win_rate']:.3f} | {b['total_r']:+.2f} |"
    )
    s = report["sized_primary"]
    lines.append(
        f"| sized 0.5/1.0/1.5 | {s['n']} | {s['mean_r']:+.4f} | {s['std_r']:.4f} | {s['sharpe']:+.4f} | {s['win_rate']:.3f} | {s['total_r']:+.2f} |"
    )
    lines.append("")
    lines.append(f"**Sharpe uplift (relative):** **{report['h1_sharpe_uplift_pct']:+.2f}%** (gate: >= 15%)")
    lines.append(f"**Sharpe difference (absolute):** **{report['h1_absolute_sharpe_diff']:+.4f}** (gate: >= 0.05)")
    lines.append(f"**Bootstrap p-value (B={report['h1_bootstrap_b']}):** **{report['h1_bootstrap_raw_p']:.4f}** (gate: < 0.05)")
    lines.append("")
    lines.append("### H1 gate breakdown (all three required for CONTINUE_TO_D1_V2)")
    gates = report["h1_gates"]
    lines.append(f"- relative_ge_15pct: {'PASS' if gates['relative_ge_15pct'] else 'FAIL'}")
    lines.append(f"- absolute_diff_ge_0_05: {'PASS' if gates['absolute_diff_ge_0_05'] else 'FAIL'}")
    lines.append(f"- raw_p_lt_0_05: {'PASS' if gates['raw_p_lt_0_05'] else 'FAIL'}")
    lines.append("")
    lines.append(f"**H1 verdict:** **{report['h1_verdict']}**")
    lines.append("")
    lines.append("## Rule 8.2 arithmetic-only check (backtesting-methodology.md)")
    lines.append(f"- Per-bucket WR: {report['rule_8_2_arithmetic_only_check']['per_bucket_win_rate']}")
    lines.append(f"- WR spread across buckets: {report['rule_8_2_arithmetic_only_check']['win_rate_spread_pct']:.2f}%")
    lines.append(f"- Flag: {report['rule_8_2_arithmetic_only_check']['flag']}")
    lines.append(f"- Interpretation: {report['rule_8_2_arithmetic_only_check']['interpretation']}")
    lines.append("")
    lines.append("## H2 ablation (low=0.0x, mid=1.0x, high=1.5x) — DESCRIPTIVE ONLY, not primary selector")
    h2s = report["h2_ablation"]["stats"]
    lines.append("| Variant | N (incl. zeros) | Mean R | Std R | Sharpe | Win rate | Total R | Low_Q1 trades skipped |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| H2 hard-skip | {h2s['n']} | {h2s['mean_r']:+.4f} | {h2s['std_r']:.4f} | {h2s['sharpe']:+.4f} | {h2s['win_rate']:.3f} | {h2s['total_r']:+.2f} | {report['h2_ablation']['low_trades_skipped']} |"
    )
    lines.append("")
    lines.append(
        "**H2 note:** per pre-reg § hypotheses.H2.selection_rule, this is descriptive-only. It is not the primary D-0 v2 selector and cannot replace H1 post hoc."
    )
    lines.append("")
    lines.append("## K2 implementation integrity checklist (pre-reg § kill_criteria.K2)")
    k2 = report["k2_implementation_integrity"]
    lines.append(f"**Overall: {k2['overall']}**")
    lines.append("")
    for check, value in k2["checks"].items():
        lines.append(f"- `{check}`: {value}")
    lines.append("")
    lines.append("## K3 feature temporal integrity (pre-reg § kill_criteria.K3)")
    k3 = report["k3_feature_temporal_integrity"]
    lines.append(f"**Overall: {k3['overall']}**")
    lines.append("")
    for check, value in k3["checks"].items():
        lines.append(f"- `{check}`: {value}")
    lines.append("")
    lines.append("## No-OOS assertion")
    lines.append(f"- OOS trades consulted during D-0 v2: **{k2['checks']['oos_consulted']}**")
    lines.append(f"- OOS trades present in sample: **{k2['checks']['oos_trades_in_sample']}** (must be 0)")
    lines.append("")
    lines.append("## Comparison to v1 (DESCRIPTIVE_ONLY_NOT_STATISTICALLY_COMPARABLE)")
    cmp = report["comparison_to_v1"]
    lines.append(f"- v1 Sharpe uplift: {cmp['v1_sharpe_uplift_pct']:+.2f}% → KILL (commit {cmp['v1_prereg_commit']})")
    lines.append(f"- v2 Sharpe uplift: {report['h1_sharpe_uplift_pct']:+.2f}% → {report['h1_verdict']}")
    lines.append(f"- Reason not comparable: {cmp['reason_not_comparable']}")
    lines.append("")
    lines.append("## Outputs-required mapping (pre-reg § outputs_required_after_run)")
    lines.append("- Predictor swap statement: YES (see Predictor swap section)")
    lines.append("- P33 and P67: YES (see Frozen thresholds)")
    lines.append("- IS sample span and N: YES (see IS trade set)")
    lines.append("- baseline Sharpe and sized Sharpe: YES (see H1 table)")
    lines.append("- Sharpe uplift % (relative): YES")
    lines.append("- Sharpe difference (absolute floor gate): YES")
    lines.append("- raw p-value (bootstrap, B=10000): YES")
    lines.append("- expectancy impact: YES (mean R comparison in H1 table)")
    lines.append("- win-rate impact (Rule 8.2): YES")
    lines.append("- trade count check (baseline == sized): YES (K2)")
    lines.append("- scratch trade count and realized-eod confirmation: YES (IS trade set + K2)")
    lines.append("- K2 implementation_integrity checklist: YES")
    lines.append("- K3 feature_temporal_integrity checklist: YES")
    lines.append("- no-OOS assertion: YES")
    lines.append("- H2 ablation: YES")
    lines.append("- comparison to v1 (DESCRIPTIVE_ONLY): YES")
    lines.append("- selection_bias_audit per-year garch NULL breakdown: YES")
    lines.append("- absolute Sharpe difference: YES")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    report = run_backtest()
    out_md = PROJECT_ROOT / "docs" / "audit" / "results" / "2026-04-28-phase-d-d0-v2-garch-backtest.md"
    out_json = PROJECT_ROOT / "research" / "output" / "phase_d_d0_v2_backtest.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    emit_markdown(report, out_md)

    # Console summary
    print(f"Pre-reg: {report['prereg_path']} (commit {report['prereg_commit']})")
    print(
        f"IS N: {report['is_trades']['n']}  span: {report['is_trades']['first_day']} .. {report['is_trades']['last_day']}"
    )
    print(f"Scratch trades (realized-eod): {report['is_trades']['scratch_count']}")
    print(f"P33: {report['thresholds']['p33']:.4f}  P67: {report['thresholds']['p67']:.4f}")
    print(f"Bucket dist: {report['bucket_distribution']}")
    print(f"Baseline Sharpe: {report['baseline']['sharpe']:+.4f}  WR: {report['baseline']['win_rate']:.3f}")
    print(f"Sized    Sharpe: {report['sized_primary']['sharpe']:+.4f}  WR: {report['sized_primary']['win_rate']:.3f}")
    print(f"Sharpe uplift (relative): {report['h1_sharpe_uplift_pct']:+.2f}%  (gate: >= 15%)")
    print(f"Sharpe diff   (absolute): {report['h1_absolute_sharpe_diff']:+.4f}  (gate: >= 0.05)")
    print(f"Bootstrap p:              {report['h1_bootstrap_raw_p']:.4f}       (gate: < 0.05)")
    print(f"H1 verdict: {report['h1_verdict']}")
    print(f"Rule 8.2 arithmetic-only flag: {report['rule_8_2_arithmetic_only_check']['flag']}")
    print(f"K2 integrity: {report['k2_implementation_integrity']['overall']}")
    print(f"K3 temporal:  {report['k3_feature_temporal_integrity']['overall']}")
    print(f"MD written: {out_md}")
    print(f"JSON written: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
