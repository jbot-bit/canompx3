"""Phase D Volume Pilot — Stage D-1 review (one-shot verdict).

Pre-reg: docs/audit/hypotheses/2026-04-17-phase-d-d1-signal-only-shadow.yaml

Runs ONCE on the review_date (2026-05-15) against the combined shadow log
(retrospective + forward_shadow rows). Evaluates all pre-registered gates
and writes the final verdict MD.

Fail-closed guards
    - Refuses to run before REVIEW_DATE unless --allow-preview is passed
    - Refuses to overwrite the verdict MD if it already exists
    - Raises on any log-integrity violation (dupes, out-of-window rows)
    - Does NOT re-tune any threshold, does NOT re-compute buckets

Execute on review date:
    uv run python research/phase_d_volume_sizing_pilot_d1_review.py

Preview before review date (writes to a clearly-labeled preview file):
    uv run python research/phase_d_volume_sizing_pilot_d1_review.py --allow-preview
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import numpy as np

from pipeline.paths import GOLD_DB_PATH

# LOCKED from D-0 pre-reg. Do not change.
P33 = 1.0529
P67 = 1.7880

INSTRUMENT = "MNQ"
SESSION = "COMEX_SETTLE"

# Windows from pre-reg.
RETROSPECTIVE_FROM = date(2026, 1, 1)
RETROSPECTIVE_TO = date(2026, 4, 16)
FORWARD_FROM = date(2026, 4, 17)
FORWARD_TO = date(2026, 5, 15)
REVIEW_DATE = date(2026, 5, 15)
COMBINED_FROM = RETROSPECTIVE_FROM
COMBINED_TO = FORWARD_TO

# Gates from pre-reg.
GATE_PRIMARY_SHARPE_RATIO = 1.05
GATE_PRIMARY_EXPR_LIFT = 0.03
GATE_SECONDARY_CORR_FLOOR = 0.00
GATE_SECONDARY_BUCKET_MIN = 0.20
GATE_SECONDARY_BUCKET_MAX = 0.50
GATE_SECONDARY_N_MIN = 60
GATE_TAUTOLOGY_MAX_ABS_CORR = 0.70

SHADOW_LOG = Path("docs/audit/results/phase-d-d1-shadow-log.csv")
VERDICT_MD = Path("docs/audit/results/2026-05-15-phase-d-d1-verdict.md")
PREVIEW_MD = Path("docs/audit/results/2026-05-15-phase-d-d1-verdict-PREVIEW.md")


@dataclass
class Metrics:
    n: int
    expR: float
    sharpe_ann: float
    max_dd_r: float


def _load_shadow_log() -> list[dict]:
    if not SHADOW_LOG.exists():
        raise RuntimeError(f"Shadow log not found: {SHADOW_LOG}")
    with SHADOW_LOG.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _integrity_check(rows: list[dict]) -> None:
    seen_days: set[date] = set()
    for r in rows:
        td = date.fromisoformat(r["trading_day"])
        if td < COMBINED_FROM or td > COMBINED_TO:
            raise RuntimeError(f"Out-of-window row: {td} (allowed [{COMBINED_FROM}, {COMBINED_TO}])")
        if td in seen_days:
            raise RuntimeError(f"Duplicate trading_day in shadow log: {td}")
        seen_days.add(td)
        label = r.get("window_label", "")
        if label == "retrospective" and td > RETROSPECTIVE_TO:
            raise RuntimeError(f"Row tagged retrospective but td={td} > {RETROSPECTIVE_TO}")
        if label == "forward_shadow" and td < FORWARD_FROM:
            raise RuntimeError(f"Row tagged forward_shadow but td={td} < {FORWARD_FROM}")


def _compute_metrics(pnl_r: np.ndarray, trading_days: list[date], label: str) -> Metrics:
    n = len(pnl_r)
    if n == 0:
        return Metrics(n=0, expR=0.0, sharpe_ann=0.0, max_dd_r=0.0)
    expR = float(np.mean(pnl_r))
    if n > 1:
        sd = float(np.std(pnl_r, ddof=1))
    else:
        sd = 0.0
    if sd > 0:
        sharpe_per_trade = expR / sd
        years_span = (trading_days[-1] - trading_days[0]).days / 365.25
        trades_per_year = n / years_span if years_span > 0 else 0.0
        sharpe_ann = sharpe_per_trade * math.sqrt(trades_per_year)
    else:
        sharpe_ann = 0.0
    cum = np.cumsum(pnl_r)
    peak = np.maximum.accumulate(cum)
    max_dd_r = float(-(cum - peak).min()) if n > 0 else 0.0
    _ = label
    return Metrics(n=n, expR=expR, sharpe_ann=sharpe_ann, max_dd_r=max_dd_r)


def _overnight_range_lookup(trading_days: list[date]) -> dict[date, float]:
    if not trading_days:
        return {}
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        days_sql = ",".join(f"'{d.isoformat()}'" for d in trading_days)
        q = f"""
        SELECT trading_day, overnight_range
        FROM daily_features
        WHERE symbol = '{INSTRUMENT}'
          AND orb_minutes = 5
          AND trading_day IN ({days_sql})
        """
        return {td: float(ov) if ov is not None else 0.0 for td, ov in con.execute(q).fetchall()}
    finally:
        con.close()


def _write_verdict(
    out_path: Path,
    is_preview: bool,
    rows: list[dict],
    baseline: Metrics,
    scaled: Metrics,
    bucket_frac: dict[float, float],
    corr_size_pnl: float,
    tautology_corr: float,
    gate_results: dict[str, tuple[str, str]],
    verdict: str,
) -> None:
    n_retro = sum(1 for r in rows if r["window_label"] == "retrospective")
    n_fwd = sum(1 for r in rows if r["window_label"] == "forward_shadow")
    lines: list[str] = []
    a = lines.append
    banner = "**PREVIEW — NOT THE FINAL VERDICT**" if is_preview else "**FINAL VERDICT**"
    a("# Phase D Volume Pilot — Stage D-1 Verdict")
    a("")
    a(banner)
    a("")
    a(f"- **Date run:** {date.today().isoformat()}")
    a("- **Pre-reg:** `docs/audit/hypotheses/2026-04-17-phase-d-d1-signal-only-shadow.yaml`")
    a(f"- **Shadow log:** `{SHADOW_LOG.as_posix()}`")
    a("- **D-0 source:** commit 55b6ba89 — `docs/audit/results/2026-04-17-phase-d-d0-backtest.md`")
    a(f"- **Verdict:** `{verdict}`")
    a("")
    a("## Universe and locks")
    a(f"- Instrument: {INSTRUMENT}")
    a(f"- Session: {SESSION}")
    a("- Aperture: O5 | RR: 1.5 | Entry: E2 | CB: 1 | Filter: none (unfiltered)")
    a(f"- P33 locked: {P33}")
    a(f"- P67 locked: {P67}")
    a(f"- Window: [{COMBINED_FROM}, {COMBINED_TO}] combined one-shot")
    a("")
    a("## Sample composition")
    a(f"- N retrospective (2026-01-01 to 2026-04-16): {n_retro}")
    a(f"- N forward shadow (2026-04-17 to 2026-05-15): {n_fwd}")
    a(f"- N combined: {len(rows)}")
    a("")
    a("## Headline metrics")
    a("")
    a("| Metric | Baseline 1x | Scaled 3-tier | Ratio |")
    a("|---|---|---|---|")
    a(f"| N | {baseline.n} | {scaled.n} | 1.000 |")
    a(f"| ExpR | {baseline.expR:.4f} | {scaled.expR:.4f} | {scaled.expR / baseline.expR if baseline.expR else 0:.3f} |")
    a(
        f"| Sharpe_ann | {baseline.sharpe_ann:.4f} | {scaled.sharpe_ann:.4f} | "
        f"{scaled.sharpe_ann / baseline.sharpe_ann if baseline.sharpe_ann else 0:.3f} |"
    )
    a(f"| MaxDD (R) | {baseline.max_dd_r:.4f} | {scaled.max_dd_r:.4f} | {scaled.max_dd_r / baseline.max_dd_r if baseline.max_dd_r else 0:.3f} |")
    a("")
    a("## Bucket distribution")
    a(f"- 0.5x: {bucket_frac.get(0.5, 0):.1%}")
    a(f"- 1.0x: {bucket_frac.get(1.0, 0):.1%}")
    a(f"- 1.5x: {bucket_frac.get(1.5, 0):.1%}")
    a(f"- pre-reg tolerance per bucket: [{GATE_SECONDARY_BUCKET_MIN:.0%}, {GATE_SECONDARY_BUCKET_MAX:.0%}]")
    a("")
    a("## Correlations")
    a(f"- corr(size_bucket, pnl_r) = {corr_size_pnl:.4f} (floor > {GATE_SECONDARY_CORR_FLOOR})")
    a(
        f"- |corr(size_bucket, overnight_range)| = {abs(tautology_corr):.4f} "
        f"(max {GATE_TAUTOLOGY_MAX_ABS_CORR}) [T0 tautology guard]"
    )
    a("")
    a("## Gate results")
    a("")
    a("| Gate | Status | Detail |")
    a("|---|---|---|")
    for name, (status, detail) in gate_results.items():
        a(f"| {name} | {status} | {detail} |")
    a("")
    a("## Verdict interpretation")
    if verdict == "PASS":
        a("- All primary gates cleared; no automatic halt triggered.")
        a("- Next step: user approves D-2 live-deployment pre-reg drafting.")
        a("- D-2 review must include DSR at multi-K framings and Shiryaev-Roberts monitor wiring.")
    elif verdict == "PASS_WITH_DRIFT_AUDIT_REQUIRED":
        a("- Primary gates cleared, but bucket distribution fell outside [20%, 50%] tolerance.")
        a("- D-2 is BLOCKED until a mandatory Shiryaev-Roberts drift audit completes.")
        a("- Do NOT re-tune P33/P67 — that is a pre-reg violation. The audit either clears the drift or halts Phase D.")
    elif verdict == "FAIL":
        a("- At least one primary gate failed. Phase D HALTS per pre-reg kill_rules.")
        a("- No D-2. Write a postmortem. Do NOT rerun with different thresholds.")
    elif verdict == "INCONCLUSIVE":
        a(f"- N_combined < {GATE_SECONDARY_N_MIN}. Shadow does not have statistical power.")
        a("- Phase D PAUSES. Window does not extend. User decides: abandon or pre-register V2 with a longer window.")
    a("")
    if is_preview:
        a("## Preview caveat")
        a(f"- This file was generated BEFORE review_date {REVIEW_DATE} via `--allow-preview`.")
        a("- The pre-registered verdict is the one produced ON {REVIEW_DATE} without the flag.")
        a("- This preview does NOT consume the Mode A one-shot read.")
    a("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase D D-1 review — one-shot verdict writer.")
    parser.add_argument(
        "--allow-preview",
        action="store_true",
        help="Run before review_date and write to a clearly-labeled preview file. Does not consume the Mode A one-shot read.",
    )
    args = parser.parse_args()

    today = date.today()
    is_preview = today < REVIEW_DATE

    if is_preview and not args.allow_preview:
        print(
            f"Refusing to run: today {today} is before review_date {REVIEW_DATE}. "
            f"Pass --allow-preview to produce a preview-only verdict file."
        )
        return 1

    out_path = PREVIEW_MD if is_preview else VERDICT_MD
    if out_path.exists() and not is_preview:
        print(
            f"Refusing to overwrite existing verdict at {out_path}. "
            f"Mode A one-shot discipline: the verdict has already been written."
        )
        return 1

    rows = _load_shadow_log()
    _integrity_check(rows)

    trading_days = [date.fromisoformat(r["trading_day"]) for r in rows]
    pnl_r = np.array([float(r["actual_pnl_r"]) for r in rows], dtype=float)
    pnl_scaled = np.array([float(r["counterfactual_pnl_r_scaled"]) for r in rows], dtype=float)
    size_bucket = np.array([float(r["size_bucket"]) for r in rows], dtype=float)

    baseline = _compute_metrics(pnl_r, trading_days, "baseline")
    scaled = _compute_metrics(pnl_scaled, trading_days, "scaled")

    # Bucket distribution.
    bucket_counts: dict[float, int] = {0.5: 0, 1.0: 0, 1.5: 0}
    for s in size_bucket:
        bucket_counts[round(float(s), 1)] = bucket_counts.get(round(float(s), 1), 0) + 1
    n = len(rows) or 1
    bucket_frac = {k: v / n for k, v in bucket_counts.items()}

    # Correlations.
    def _corr(x: np.ndarray, y: np.ndarray) -> float:
        if len(x) < 2 or np.std(x, ddof=1) == 0 or np.std(y, ddof=1) == 0:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    corr_size_pnl = _corr(size_bucket, pnl_r)
    ovn_map = _overnight_range_lookup(trading_days)
    ovn_arr = np.array([ovn_map.get(td, 0.0) for td in trading_days], dtype=float)
    tautology_corr = _corr(size_bucket, ovn_arr)

    # Gate evaluation.
    gate_results: dict[str, tuple[str, str]] = {}

    sharpe_ratio = scaled.sharpe_ann / baseline.sharpe_ann if baseline.sharpe_ann else 0.0
    primary_sharpe_pass = sharpe_ratio >= GATE_PRIMARY_SHARPE_RATIO
    gate_results["primary Sharpe uplift >= 1.05x"] = (
        "PASS" if primary_sharpe_pass else "FAIL",
        f"ratio = {sharpe_ratio:.3f}",
    )

    expR_lift = scaled.expR - baseline.expR
    primary_expR_pass = expR_lift >= GATE_PRIMARY_EXPR_LIFT
    gate_results["primary ExpR lift >= 0.03R"] = (
        "PASS" if primary_expR_pass else "FAIL",
        f"lift = {expR_lift:+.4f}R",
    )

    corr_pass = corr_size_pnl > GATE_SECONDARY_CORR_FLOOR
    gate_results["secondary corr(size, pnl) > 0"] = (
        "PASS" if corr_pass else "FAIL",
        f"corr = {corr_size_pnl:.4f}",
    )

    bucket_pass = all(
        GATE_SECONDARY_BUCKET_MIN <= f <= GATE_SECONDARY_BUCKET_MAX for f in bucket_frac.values()
    )
    gate_results["secondary bucket distribution 20-50% each"] = (
        "PASS" if bucket_pass else "DRIFT_AUDIT_REQUIRED",
        f"{bucket_frac.get(0.5, 0):.1%} / {bucket_frac.get(1.0, 0):.1%} / {bucket_frac.get(1.5, 0):.1%}",
    )

    n_pass = len(rows) >= GATE_SECONDARY_N_MIN
    gate_results["secondary N_combined >= 60"] = (
        "PASS" if n_pass else "INCONCLUSIVE_TRIGGER",
        f"N = {len(rows)}",
    )

    tautology_pass = abs(tautology_corr) < GATE_TAUTOLOGY_MAX_ABS_CORR
    gate_results["T0 tautology |corr(size, overnight_range)| < 0.70"] = (
        "PASS" if tautology_pass else "FLAG",
        f"|corr| = {abs(tautology_corr):.4f}",
    )

    # Verdict priority per pre-reg.
    if not n_pass:
        verdict = "INCONCLUSIVE"
    elif not (primary_sharpe_pass and primary_expR_pass and corr_pass):
        verdict = "FAIL"
    elif not bucket_pass:
        verdict = "PASS_WITH_DRIFT_AUDIT_REQUIRED"
    else:
        verdict = "PASS"

    _write_verdict(
        out_path,
        is_preview,
        rows,
        baseline,
        scaled,
        bucket_frac,
        corr_size_pnl,
        tautology_corr,
        gate_results,
        verdict,
    )

    print(f"Verdict: {verdict}")
    print(f"Written to: {out_path}")
    for name, (status, detail) in gate_results.items():
        print(f"  {name}: {status} ({detail})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
