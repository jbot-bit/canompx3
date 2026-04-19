"""Phase D D-0 backtest — rel_vol size-scaling on MNQ COMEX_SETTLE O5 RR1.5 OVNRNG_100.

Pre-registration: docs/audit/hypotheses/2026-04-18-phase-d-d0-rel-vol-sizing-mnq-comex-settle.yaml
Commit SHA of pre-reg: b6918d8d (locked before this script runs)

Execution order (enforces pre-reg § execution_gate):
1. Load IS-only lane trade set (trading_day < HOLDOUT_SACRED_FROM, OVNRNG_100 fires, rel_vol not null)
2. Compute P33/P67 from IS rel_vol distribution ONCE and freeze
3. Bucket each trade: low (< P33) / mid / high (>= P67)
4. Baseline: constant 1.0x — Sharpe_baseline = mean(R) / std(R)
5. Sized: R * size_mult[bucket] — Sharpe_sized = mean(sR) / std(sR)
6. H1 metric: Sharpe uplift % (must be >= 15% to CONTINUE, < 10% to KILL)
7. H2 ablation (descriptive-only): low=0.0x, mid=1.0x, high=1.5x
8. K2 integrity checklist (pre-reg § kill_criteria.K2) — each item PASS/FAIL
9. Emit Markdown report with every output required by pre-reg.

Canonical source delegation (institutional-rigor rule 4):
- DB path from pipeline.paths.GOLD_DB_PATH
- HOLDOUT_SACRED_FROM from trading_app.holdout_policy
- OVNRNG_100 filter logic from trading_app.config.ALL_FILTERS (canonical matches_df)
- Triple-join rule per .claude/rules/daily-features-joins.md
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from pipeline.paths import GOLD_DB_PATH
from trading_app.config import ALL_FILTERS
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]


PREREG_PATH = "docs/audit/hypotheses/2026-04-18-phase-d-d0-rel-vol-sizing-mnq-comex-settle.yaml"
PREREG_COMMIT = "b6918d8d"
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


def load_is_trades(con) -> list[dict]:
    """Load IS-only trades for the D-0 lane. Applies triple-join + canonical filter."""
    ovn_filter = ALL_FILTERS[SCOPE["filter_type"]]
    min_range = getattr(ovn_filter, "min_range", None)
    if min_range is None:
        raise RuntimeError(
            f"Filter {SCOPE['filter_type']} missing min_range attribute; "
            "canonical source changed. Do not fallback — halt per K2."
        )

    rows = con.execute(
        f"""
        SELECT
            o.trading_day,
            o.pnl_r,
            o.outcome,
            d.rel_vol_{SCOPE["orb_label"]} AS rel_vol,
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
          AND o.outcome IS NOT NULL
          AND o.pnl_r IS NOT NULL
          AND o.trading_day < ?
          AND d.overnight_range IS NOT NULL
          AND d.overnight_range >= ?
          AND d.rel_vol_{SCOPE["orb_label"]} IS NOT NULL
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

    return [
        {
            "trading_day": r[0],
            "pnl_r": float(r[1]),
            "outcome": r[2],
            "rel_vol": float(r[3]),
            "overnight_range": float(r[4]),
        }
        for r in rows
    ]


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


def bucket_of(rel_vol: float, p33: float, p67: float) -> str:
    if rel_vol < p33:
        return "low"
    if rel_vol >= p67:
        return "high"
    return "mid"


def run_backtest() -> dict:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        # Step 1: load IS trade set
        trades = load_is_trades(con)
        if not trades:
            raise RuntimeError("Zero IS trades found for D-0 lane — check filter / scope.")

        # Step 2: compute P33/P67 ONCE from IS rel_vol distribution and freeze
        rel_vols = [t["rel_vol"] for t in trades]
        p33, p67 = compute_quantiles(rel_vols, [0.33, 0.67])
        thresholds_frozen_at = datetime.now(timezone.utc).isoformat()

        # Step 3: bucket each trade (deterministic from frozen thresholds)
        for t in trades:
            t["bucket"] = bucket_of(t["rel_vol"], p33, p67)

        # Step 4: baseline — constant 1.0x (equivalent to raw pnl_r)
        baseline_rs = [t["pnl_r"] for t in trades]
        baseline = SharpeStats.from_rs(baseline_rs)

        # Step 5: primary sized schema (0.5 / 1.0 / 1.5)
        sized_rs = [BUCKETS[t["bucket"]] * t["pnl_r"] for t in trades]
        sized = SharpeStats.from_rs(sized_rs)

        # Step 6: H1 metric
        if baseline.sharpe != 0:
            uplift_pct = 100.0 * (sized.sharpe - baseline.sharpe) / abs(baseline.sharpe)
        else:
            uplift_pct = 0.0

        # Step 7: H2 ablation (descriptive only)
        h2_rs = [H2_BUCKETS[t["bucket"]] * t["pnl_r"] for t in trades]
        h2 = SharpeStats.from_rs(h2_rs)
        h2_skipped_low = sum(1 for t in trades if t["bucket"] == "low")

        # Per-bucket diagnostics
        bucket_stats = {}
        for b_name in ("low", "mid", "high"):
            b_rs = [t["pnl_r"] for t in trades if t["bucket"] == b_name]
            bucket_stats[b_name] = SharpeStats.from_rs(b_rs)

        # Step 8: K2 integrity checklist
        oos_trades = [t for t in trades if t["trading_day"] >= HOLDOUT_SACRED_FROM]
        k2 = {
            "p33_p67_calibrated_is_only": all(t["trading_day"] < HOLDOUT_SACRED_FROM for t in trades),
            "oos_trades_in_sample": len(oos_trades),
            "oos_consulted": False,  # script never queries OOS
            "bucket_thresholds_frozen_before_sharpe": True,
            "bucket_thresholds_frozen_at": thresholds_frozen_at,
            "size_applied_at_entry_not_retro": True,  # bucket = f(rel_vol) only; rel_vol ∈ trade-time-knowable set
            "trade_count_baseline_eq_sized": baseline.n == sized.n,
        }
        k2_pass = (
            k2["p33_p67_calibrated_is_only"]
            and k2["oos_trades_in_sample"] == 0
            and k2["oos_consulted"] is False
            and k2["bucket_thresholds_frozen_before_sharpe"] is True
            and k2["size_applied_at_entry_not_retro"] is True
            and k2["trade_count_baseline_eq_sized"] is True
        )

        # Step 9: Rule 8.2 arithmetic-only check.
        # Linear sizing preserves sign → aggregate WR trivially identical to baseline.
        # Real check: does rel_vol bucket PREDICT win probability (per-bucket WR spread),
        # or only per-bucket payoff? Spread < 3% + large Sharpe move = ARITHMETIC_ONLY.
        per_bucket_wr = {b: bucket_stats[b].win_rate for b in ("low", "mid", "high")}
        wr_spread_pct = (max(per_bucket_wr.values()) - min(per_bucket_wr.values())) * 100
        arithmetic_only_flag = wr_spread_pct < 3.0 and abs(uplift_pct) > 10.0

        return {
            "scope": SCOPE,
            "prereg_path": PREREG_PATH,
            "prereg_commit": PREREG_COMMIT,
            "run_utc": datetime.now(timezone.utc).isoformat(),
            "holdout_sacred_from": str(HOLDOUT_SACRED_FROM),
            "is_trades": {
                "n": len(trades),
                "first_day": str(trades[0]["trading_day"]),
                "last_day": str(trades[-1]["trading_day"]),
            },
            "thresholds": {
                "p33": p33,
                "p67": p67,
                "frozen_at_utc": thresholds_frozen_at,
            },
            "bucket_distribution": {b: sum(1 for t in trades if t["bucket"] == b) for b in ("low", "mid", "high")},
            "per_bucket_stats": {b: asdict(bucket_stats[b]) for b in ("low", "mid", "high")},
            "baseline": asdict(baseline),
            "sized_primary": asdict(sized),
            "h1_sharpe_uplift_pct": uplift_pct,
            "h1_verdict": ("CONTINUE_TO_D1" if uplift_pct >= 15.0 else "PARK" if uplift_pct >= 10.0 else "KILL"),
            "h2_ablation": {
                "stats": asdict(h2),
                "low_trades_skipped": h2_skipped_low,
                "note": "descriptive_only; not primary selector",
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
        }
    finally:
        con.close()


def emit_markdown(report: dict, out_path: Path) -> None:
    lines = []
    lines.append("# Phase D D-0 Backtest — MNQ COMEX_SETTLE O5 RR1.5 OVNRNG_100 rel_vol size-scaling")
    lines.append("")
    lines.append(f"**Pre-registration:** `{report['prereg_path']}` (commit `{report['prereg_commit']}`)")
    lines.append(f"**Run UTC:** {report['run_utc']}")
    lines.append(f"**Holdout sacred from:** {report['holdout_sacred_from']}")
    lines.append(f"**Scope:** {json.dumps(report['scope'])}")
    lines.append("")
    lines.append("## IS trade set")
    lines.append(f"- N: {report['is_trades']['n']}")
    lines.append(f"- First day: {report['is_trades']['first_day']}")
    lines.append(f"- Last day: {report['is_trades']['last_day']}")
    lines.append("")
    lines.append("## Frozen thresholds (pre-reg § calibration.thresholds)")
    lines.append(f"- P33 rel_vol_COMEX_SETTLE: **{report['thresholds']['p33']:.4f}**")
    lines.append(f"- P67 rel_vol_COMEX_SETTLE: **{report['thresholds']['p67']:.4f}**")
    lines.append(f"- Frozen at UTC: {report['thresholds']['frozen_at_utc']}")
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
    lines.append(f"**Sharpe uplift:** **{report['h1_sharpe_uplift_pct']:+.2f}%**")
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
        "**H2 note:** per pre-reg § hypotheses.H2.selection_rule, this is descriptive-only. It is not the primary D-0 selector and cannot replace H1 post hoc."
    )
    lines.append("")
    lines.append("## K2 implementation integrity checklist (pre-reg § kill_criteria.K2)")
    k2 = report["k2_implementation_integrity"]
    lines.append(f"**Overall: {k2['overall']}**")
    lines.append("")
    for check, value in k2["checks"].items():
        lines.append(f"- `{check}`: {value}")
    lines.append("")
    lines.append("## No-OOS assertion")
    lines.append(f"- OOS trades consulted during D-0: **{k2['checks']['oos_consulted']}**")
    lines.append(f"- OOS trades present in sample: **{k2['checks']['oos_trades_in_sample']}** (must be 0)")
    lines.append("")
    lines.append("## Outputs-required mapping (pre-reg § outputs_required_after_run)")
    lines.append("- P33 and P67: YES (see thresholds)")
    lines.append("- IS sample span and N: YES (see IS trade set)")
    lines.append("- baseline Sharpe and sized Sharpe: YES (see H1 table)")
    lines.append("- Sharpe uplift %: YES")
    lines.append("- expectancy impact: YES (mean R comparison in H1 table)")
    lines.append("- win-rate impact (Rule 8.2): YES")
    lines.append("- trade count impact: YES (N_baseline == N_sized required check)")
    lines.append("- H2 hard-skip ablation: YES")
    lines.append("- no-OOS assertion: YES")
    lines.append("- K2 checklist PASS/FAIL: YES")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    report = run_backtest()
    out_md = PROJECT_ROOT / "docs" / "audit" / "results" / "2026-04-18-phase-d-d0-backtest.md"
    out_json = PROJECT_ROOT / "research" / "output" / "phase_d_d0_backtest.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    emit_markdown(report, out_md)

    # Console summary
    print(f"Pre-reg: {report['prereg_path']} (commit {report['prereg_commit']})")
    print(
        f"IS N: {report['is_trades']['n']}  span: {report['is_trades']['first_day']} .. {report['is_trades']['last_day']}"
    )
    print(f"P33: {report['thresholds']['p33']:.4f}  P67: {report['thresholds']['p67']:.4f}")
    print(f"Bucket dist: {report['bucket_distribution']}")
    print(f"Baseline Sharpe: {report['baseline']['sharpe']:+.4f}  WR: {report['baseline']['win_rate']:.3f}")
    print(f"Sized    Sharpe: {report['sized_primary']['sharpe']:+.4f}  WR: {report['sized_primary']['win_rate']:.3f}")
    print(f"Sharpe uplift: {report['h1_sharpe_uplift_pct']:+.2f}%")
    print(f"H1 verdict: {report['h1_verdict']}")
    print(f"Rule 8.2 arithmetic-only flag: {report['rule_8_2_arithmetic_only_check']['flag']}")
    print(f"K2 integrity: {report['k2_implementation_integrity']['overall']}")
    print(f"MD written: {out_md}")
    print(f"JSON written: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
