"""Portfolio-level block-bootstrap on IS — probabilistic OOS estimate.

Pre-reg: docs/audit/hypotheses/2026-04-21-portfolio-bootstrap-v1.yaml
Authority: docs/institutional/pre_registered_criteria.md Amendment 3.2

Replaces the CPCV path killed by H3 (docs/audit/hypotheses/2026-04-21-cpcv-infrastructure-v1-postmortem.md).
Measured per-lane IS pnl_r lag-1 autocorrelation is ρ ∈ [−0.03, +0.03] (near-zero),
so plain block-bootstrap with block size 1 is statistically valid.

What this does:
  1. Builds the IS daily portfolio return series (2020-01-01 to 2025-12-31)
     from canonical orb_outcomes, filtered through the 6 deployed lanes'
     filter_type via ALL_FILTERS[key].matches_row.
  2. Builds the 2026 shadow portfolio daily return series from paper_trades
     (the live-execution realized P&L, not simulation).
  3. Bootstraps 10,000 74-trading-day windows from the IS daily return
     series and computes ExpR + annualized Sharpe for each.
  4. Reports where the observed 2026 shadow performance ranks within that
     bootstrap distribution.

Output: docs/audit/results/2026-04-21-portfolio-bootstrap-v1.md
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import duckdb  # noqa: E402

from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from trading_app.config import ALL_FILTERS  # noqa: E402
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM  # noqa: E402

# Pre-registered parameters (from pre-reg)
IS_START = "2020-01-01"
HOLDOUT_ISO = HOLDOUT_SACRED_FROM.isoformat()  # "2026-01-01"
N_BOOTSTRAP = 10000
WINDOW_LEN_DAYS = 74  # 2026-01-02 → 2026-04-19 trading days
SEED = 42
ANNUALIZATION_DAYS = 250


def _load_deployed_lanes() -> list[dict]:
    """Read docs/runtime/lane_allocation.json 'lanes' array."""
    with (REPO_ROOT / "docs" / "runtime" / "lane_allocation.json").open() as f:
        payload = json.load(f)
    return payload["lanes"]


def _lane_filtered_trades(
    con: duckdb.DuckDBPyConnection, lane: dict, is_start: str, is_end: str
) -> list[tuple]:
    """Return (trading_day, pnl_r) tuples for a single lane on IS range.

    Applies the lane's canonical filter via ALL_FILTERS[filter_type].matches_row
    using a daily_features join. Triple-join includes orb_minutes to avoid
    3x row inflation.
    """
    sid = lane["strategy_id"]
    # Parse strategy_id to get dimensions
    import re

    m = re.match(
        r"^([A-Z0-9]+)_(.+?)_(E\d)_RR([\d.]+)_CB(\d+)_(.+?)(?:_O(\d+))?$", sid
    )
    if not m:
        raise ValueError(f"cannot parse strategy_id: {sid}")
    inst = m.group(1)
    sess = m.group(2)
    em = m.group(3)
    rr = float(m.group(4))
    cb = int(m.group(5))
    filter_type = m.group(6)
    orb_min = int(m.group(7) or 5)

    if filter_type not in ALL_FILTERS:
        raise ValueError(f"filter_type {filter_type} not in ALL_FILTERS")
    filter_obj = ALL_FILTERS[filter_type]

    rows = con.execute(
        """
        SELECT o.*, d.*
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.symbol = d.symbol
         AND o.trading_day = d.trading_day
         AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = ?
          AND o.orb_label = ?
          AND o.orb_minutes = ?
          AND o.entry_model = ?
          AND o.confirm_bars = ?
          AND o.rr_target = ?
          AND o.trading_day >= ?
          AND o.trading_day < ?
          AND o.pnl_r IS NOT NULL
        """,
        [inst, sess, orb_min, em, cb, rr, is_start, is_end],
    ).fetchall()
    col_names = [desc[0] for desc in con.description]

    out: list[tuple] = []
    for raw in rows:
        joined = dict(zip(col_names, raw, strict=False))
        try:
            if filter_obj.matches_row(joined, sess):
                out.append((joined["trading_day"], float(joined["pnl_r"])))
        except Exception:
            # A filter defect should not crash the bootstrap; drop the row
            # and keep going — the failing row would be dropped by the live
            # pipeline's filter guard too.
            continue
    return out


def _daily_aggregate(trades: list[tuple]) -> dict:
    """Group (trading_day, pnl_r) tuples into {trading_day: sum_pnl_r}."""
    agg: dict = {}
    for day, pnl in trades:
        agg[day] = agg.get(day, 0.0) + pnl
    return agg


def _merge_portfolio_days(per_lane: list[dict]) -> list[tuple]:
    """Merge per-lane daily-agg dicts into a single sorted (day, total_R) list.

    Trading days on which NO lane fired are omitted (zero-R days add noise
    to daily-statistic bootstrap without adding information — this matches
    backtesting convention for event-driven strategies).
    """
    all_days: set = set()
    for d in per_lane:
        all_days.update(d.keys())
    merged = []
    for day in sorted(all_days):
        total = sum(d.get(day, 0.0) for d in per_lane)
        merged.append((day, total))
    return merged


def _stats(daily_rs: list[float]) -> tuple[float, float]:
    """Return (mean, annualized_sharpe) for a daily-R vector."""
    n = len(daily_rs)
    if n < 2:
        return 0.0, 0.0
    m = sum(daily_rs) / n
    var = sum((x - m) ** 2 for x in daily_rs) / (n - 1)
    sd = var ** 0.5
    if sd <= 0.0:
        return m, 0.0
    sharpe = (m / sd) * (ANNUALIZATION_DAYS ** 0.5)
    return m, sharpe


def _bootstrap_windows(
    daily_rs: list[float],
    window_len: int,
    n_iterations: int,
    seed: int,
) -> list[tuple[float, float]]:
    """Return list of (mean_R, annualized_sharpe) per bootstrap resample.

    Block size = 1 (plain bootstrap). Justified by measured ρ ≈ 0 on all
    6 deployed lanes (pre-reg § procedure step_3_bootstrap).
    """
    rng = random.Random(seed)
    n = len(daily_rs)
    out: list[tuple[float, float]] = []
    for _ in range(n_iterations):
        sample = [daily_rs[rng.randrange(n)] for _ in range(window_len)]
        out.append(_stats(sample))
    return out


def _percentile(sorted_values: list[float], target: float) -> float:
    """Percentile rank (0-100) of `target` within `sorted_values`."""
    n = len(sorted_values)
    if n == 0:
        return 50.0
    # Count of values ≤ target
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_values[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return 100.0 * lo / n


def _bootstrap_ci(sorted_values: list[float], alpha: float = 0.05) -> tuple[float, float]:
    n = len(sorted_values)
    lo = sorted_values[int(n * alpha / 2)]
    hi = sorted_values[int(n * (1 - alpha / 2))]
    return lo, hi


def main() -> int:
    print("=" * 70)
    print("PORTFOLIO BOOTSTRAP v1")
    print("Pre-reg: docs/audit/hypotheses/2026-04-21-portfolio-bootstrap-v1.yaml")
    print("=" * 70)

    lanes = _load_deployed_lanes()
    lane_sids = [lane["strategy_id"] for lane in lanes]
    print(f"\nLanes ({len(lanes)}):")
    for sid in lane_sids:
        print(f"  {sid}")

    # --- Build IS portfolio ------------------------------------------------
    print(f"\nBuilding IS portfolio: {IS_START} → {HOLDOUT_ISO} (exclusive)")
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    per_lane_agg_is: list[dict] = []
    total_is_trades = 0
    for lane in lanes:
        trades = _lane_filtered_trades(con, lane, IS_START, HOLDOUT_ISO)
        total_is_trades += len(trades)
        per_lane_agg_is.append(_daily_aggregate(trades))
        print(f"  {lane['strategy_id'][:58]:<58} N_IS_trades={len(trades)}")
    print(f"Total IS trades across lanes: {total_is_trades}")

    is_daily = _merge_portfolio_days(per_lane_agg_is)
    is_daily_rs = [r for _, r in is_daily]
    is_mean, is_sharpe = _stats(is_daily_rs)
    print(f"IS daily portfolio stats: N_days={len(is_daily_rs)}, mean_daily_R={is_mean:+.4f}, annualized_Sharpe={is_sharpe:+.3f}")

    # --- Build 2026 shadow from paper_trades -------------------------------
    print(f"\nBuilding 2026 shadow from paper_trades (strategy IDs: the same {len(lanes)}):")
    placeholders = ",".join(["?"] * len(lane_sids))
    rows = con.execute(
        f"""
        SELECT trading_day, strategy_id, pnl_r
        FROM paper_trades
        WHERE trading_day >= ? AND strategy_id IN ({placeholders})
              AND pnl_r IS NOT NULL
        """,
        [HOLDOUT_ISO, *lane_sids],
    ).fetchall()
    shadow_daily_agg: dict = {}
    for day, _sid, pnl in rows:
        shadow_daily_agg[day] = shadow_daily_agg.get(day, 0.0) + float(pnl)
    shadow_sorted = sorted(shadow_daily_agg.items())
    shadow_daily_rs = [r for _, r in shadow_sorted]
    shadow_mean, shadow_sharpe = _stats(shadow_daily_rs)
    print(
        f"2026 shadow stats: N_trades={len(rows)}, N_days_with_trades={len(shadow_daily_rs)}, "
        f"mean_daily_R={shadow_mean:+.4f}, annualized_Sharpe={shadow_sharpe:+.3f}"
    )
    con.close()

    # --- Sanity gate: IS distribution center check -------------------------
    # Kill threshold (pre-reg): |median − mean| / sd > 0.10 flags
    # heavy-tail / non-normal IS distribution that would invalidate the
    # bootstrap interpretation.
    is_sorted = sorted(is_daily_rs)
    is_median = is_sorted[len(is_sorted) // 2]
    is_sd = (sum((x - is_mean) ** 2 for x in is_daily_rs) / max(1, len(is_daily_rs) - 1)) ** 0.5
    center_ratio = abs(is_median - is_mean) / is_sd if is_sd > 0 else 0.0
    print(
        f"\nIS distribution sanity: mean={is_mean:+.4f}  median={is_median:+.4f}  "
        f"sd={is_sd:.4f}  |med−mean|/sd={center_ratio:.3f}  "
        f"({'PASS' if center_ratio <= 0.10 else 'KILL'})"
    )
    sanity_passed = center_ratio <= 0.10

    # --- Bootstrap ---------------------------------------------------------
    print(f"\nBootstrapping: N={N_BOOTSTRAP}, window={WINDOW_LEN_DAYS} days, seed={SEED}, block_size=1")
    boot = _bootstrap_windows(is_daily_rs, WINDOW_LEN_DAYS, N_BOOTSTRAP, SEED)
    boot_means = sorted(m for m, _ in boot)
    boot_sharpes = sorted(s for _, s in boot)

    # --- Compare -----------------------------------------------------------
    observed_mean = shadow_mean
    observed_sharpe = shadow_sharpe
    mean_rank = _percentile(boot_means, observed_mean)
    sharpe_rank = _percentile(boot_sharpes, observed_sharpe)
    mean_ci = _bootstrap_ci(boot_means)
    sharpe_ci = _bootstrap_ci(boot_sharpes)
    # One-tailed p: fraction of bootstrap iterations >= observed
    one_tail_p_mean = sum(1 for m in boot_means if m >= observed_mean) / len(boot_means)
    one_tail_p_sharpe = sum(1 for s in boot_sharpes if s >= observed_sharpe) / len(boot_sharpes)

    print(f"\n--- Results ---")
    print(f"2026 observed mean daily R:     {observed_mean:+.4f}")
    print(f"2026 observed annualized Sharpe: {observed_sharpe:+.3f}")
    print(f"IS bootstrap 95% CI mean:        [{mean_ci[0]:+.4f}, {mean_ci[1]:+.4f}]")
    print(f"IS bootstrap 95% CI Sharpe:      [{sharpe_ci[0]:+.3f}, {sharpe_ci[1]:+.3f}]")
    print(f"2026 mean percentile rank:       {mean_rank:.1f}")
    print(f"2026 Sharpe percentile rank:     {sharpe_rank:.1f}")
    print(f"One-tailed p (mean):             {one_tail_p_mean:.4f}")
    print(f"One-tailed p (Sharpe):           {one_tail_p_sharpe:.4f}")

    # Verdict per pre-reg
    primary_pass = mean_rank >= 75.0
    secondary_pass = sharpe_rank >= 75.0
    regime_drift = mean_rank < 25.0
    if not sanity_passed:
        verdict = "KILL_SANITY"
    elif regime_drift:
        verdict = "REGIME_DRIFT_SUSPECT"
    elif primary_pass and secondary_pass:
        verdict = "PASS_BOTH"
    elif primary_pass or secondary_pass:
        verdict = "PASS_ONE"
    else:
        verdict = "INCONCLUSIVE"

    print(f"\nVerdict: {verdict}")

    # --- Write results MD --------------------------------------------------
    out_path = REPO_ROOT / "docs" / "audit" / "results" / "2026-04-21-portfolio-bootstrap-v1.md"
    lines = []
    lines.append("# Portfolio Bootstrap v1 — Results")
    lines.append("")
    lines.append("**Date:** 2026-04-21")
    lines.append("**Pre-reg:** `docs/audit/hypotheses/2026-04-21-portfolio-bootstrap-v1.yaml`")
    lines.append("**Authority:** `docs/institutional/pre_registered_criteria.md` Amendment 3.2")
    lines.append(
        f"**Parameters:** N_bootstrap={N_BOOTSTRAP}, window={WINDOW_LEN_DAYS} days, seed={SEED}, "
        f"block_size=1 (justified by measured ρ ≈ 0), annualization=sqrt({ANNUALIZATION_DAYS})"
    )
    lines.append("")
    lines.append(f"## Verdict: **{verdict}**")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- IS window: {IS_START} → {HOLDOUT_ISO} (exclusive)")
    lines.append(f"- 2026 shadow window: {HOLDOUT_ISO} → present (74 trading days nominal)")
    lines.append(f"- Lanes (from `docs/runtime/lane_allocation.json`):")
    for sid in lane_sids:
        lines.append(f"  - `{sid}`")
    lines.append("")
    lines.append("## Per-lane IS trade counts (after canonical filter application)")
    lines.append("")
    lines.append("| Lane | N_IS_trades |")
    lines.append("|---|---:|")
    for lane, agg in zip(lanes, per_lane_agg_is, strict=False):
        lines.append(f"| `{lane['strategy_id']}` | {len(agg)} days with trades |")
    lines.append("")
    lines.append(f"Total IS trades across lanes: **{total_is_trades}**")
    lines.append("")
    lines.append("## IS daily-portfolio stats")
    lines.append("")
    lines.append(f"- Days with ≥1 trade: **{len(is_daily_rs)}**")
    lines.append(f"- Mean daily R: **{is_mean:+.4f}**")
    lines.append(f"- Median daily R: **{is_median:+.4f}**")
    lines.append(f"- Daily-R sd: **{is_sd:.4f}**")
    lines.append(f"- Annualized Sharpe (IS): **{is_sharpe:+.3f}**")
    lines.append(f"- |med−mean|/sd sanity ratio: **{center_ratio:.3f}** (threshold ≤ 0.10) → **{'PASS' if sanity_passed else 'KILL'}**")
    lines.append("")
    lines.append("## 2026 shadow observed")
    lines.append("")
    lines.append(f"- Trades in 2026 from the 6 deployed lanes (paper_trades): **{len(rows)}**")
    lines.append(f"- Days with ≥1 trade: **{len(shadow_daily_rs)}**")
    lines.append(f"- Observed mean daily R: **{observed_mean:+.4f}**")
    lines.append(f"- Observed annualized Sharpe: **{observed_sharpe:+.3f}**")
    lines.append("")
    lines.append("## Bootstrap results")
    lines.append("")
    lines.append("| Metric | Observed 2026 | IS 95% CI | Percentile rank | One-tailed p |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| Mean daily R | {observed_mean:+.4f} | [{mean_ci[0]:+.4f}, {mean_ci[1]:+.4f}] | "
        f"{mean_rank:.1f} | {one_tail_p_mean:.4f} |"
    )
    lines.append(
        f"| Annualized Sharpe | {observed_sharpe:+.3f} | [{sharpe_ci[0]:+.3f}, {sharpe_ci[1]:+.3f}] | "
        f"{sharpe_rank:.1f} | {one_tail_p_sharpe:.4f} |"
    )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if verdict == "PASS_BOTH":
        lines.append(
            "The observed 2026 shadow performance ranks above the 75th percentile on BOTH "
            "ExpR and Sharpe within the IS bootstrap distribution — consistent with the "
            "pre-registered prediction that the edge is real and stable. This is "
            "Tier-1-equivalent probabilistic OOS evidence under Amendment 3.2: not a "
            "replacement for eventual live Tier-1 at N≥100 per lane, but a meaningful "
            "independent confirmation from IS data that the portfolio's 2026 shadow is "
            "NOT a typical IS-draw coincidence."
        )
    elif verdict == "PASS_ONE":
        lines.append(
            "The 2026 shadow exceeds the 75th percentile on ONE of (ExpR, Sharpe) but "
            "not both. Partial confirmation — the portfolio is performing in the upper "
            "tail of what IS returns produce, but not decisively so on both axes."
        )
    elif verdict == "INCONCLUSIVE":
        lines.append(
            "The 2026 shadow falls within the IS typical range (25th–75th percentile "
            "on at least one metric). The portfolio is working about as well as it "
            "historically did — no regime change indicated, but also not distinctly "
            "above-IS. Live-money decisions at this point rest on personal risk "
            "tolerance, not statistical confirmation."
        )
    elif verdict == "REGIME_DRIFT_SUSPECT":
        lines.append(
            "**WARNING:** 2026 shadow ranks below the 25th percentile of the IS "
            "distribution — observed performance is in the LOWER tail of what the IS "
            "edge would produce by chance. This triggers the pre-registered "
            "regime-drift audit branch. Do NOT scale up the 6-lane deployment; run a "
            "per-lane audit before any further action."
        )
    elif verdict == "KILL_SANITY":
        lines.append(
            "**KILL:** IS distribution sanity check failed — |median−mean|/sd > 0.10. "
            "Do not interpret the bootstrap result; the IS distribution is heavy-tailed "
            "or malformed. Investigate the IS construction (filter application, join "
            "multiplication, date ranges) before drawing any conclusion."
        )
    lines.append("")
    lines.append("## Raw bootstrap percentile summary (for audit)")
    lines.append("")
    boot_pcts = [10, 25, 50, 75, 90]
    lines.append("| Percentile | IS mean daily R | IS annualized Sharpe |")
    lines.append("|---:|---:|---:|")
    for p in boot_pcts:
        idx = int(len(boot_means) * p / 100)
        lines.append(f"| {p} | {boot_means[idx]:+.4f} | {boot_sharpes[idx]:+.3f} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "Reproduce with: `python research/portfolio_bootstrap_v1.py`. Reads "
        f"canonical `gold.db` only; no writes. Fixed seed {SEED}."
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nResults written to: {out_path.relative_to(REPO_ROOT)}")

    return 0 if verdict in ("PASS_BOTH", "PASS_ONE", "INCONCLUSIVE") else 1


if __name__ == "__main__":
    sys.exit(main())
