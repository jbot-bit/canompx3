"""
ORB_G5 deployed-lane ARITHMETIC_ONLY confirmatory audit (cross-lane).

Question
--------
PR #57 (2026-04-20) classified L1 `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` as
FILTER_CORRELATES_WITH_EDGE (filt IS t=+2.28 vs unfilt t=+1.61, Welch
fire-vs-non-fire p=0.001). Action queue item #2 (
`docs/plans/2026-04-21-post-stale-lock-action-queue.md`) proposes a pre-
break-context overlay to replace ORB_G5 selectivity.

Before writing that prereg, audit the finding against RULE 8.2 of
`.claude/rules/backtesting-methodology.md`:

    |wr_spread| < 3% AND |Δ_IS| > 0.10 → flag `arithmetic_only`

ARITHMETIC_ONLY means the filter acts as a cost-screen (excludes small-ORB
trades where fixed $-costs dominate R-math), not a directional-accuracy
filter. If flagged, the correct deployment class is "cost-gate", not
"edge", and a behavioral overlay is built on a flawed premise.

Scope is CROSS-LANE over all `topstep_50k_mnq_auto` deployed lanes that
use `ORB_G5` (3 of 6 per `docs/runtime/lane_allocation.json`):

- MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5            (O5)
- MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5           (O5)
- MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15       (O15)

The other 3 lanes are out of scope (different filter class): 2 use
`COST_LT12` (cost-gate by construction); 1 uses `ATR_P50`. They are
listed in the result MD for completeness.

Each lane is tested against TWO statistics:

1. RULE 8.2 ARITHMETIC_ONLY classification.
2. Two-proportion z-test on WR directly. If p_wr >> 0.05, WR spread is
   indistinguishable from zero and the filter has no directional content
   at all (cost-amplification is the only source of Δ_ExpR).

This is a CONFIRMATORY audit per RULE 10 — no new prereg required.

Canonical source
----------------
- `orb_outcomes` ⨝ `daily_features` on `(trading_day, symbol, orb_minutes)`.
- `trading_app/config.py:2740` — ORB_G5 filter spec: `min_size=5.0`.
- IS only: `trading_day < 2026-01-01` (Mode A holdout).
- Trade requirement: `entry_ts IS NOT NULL AND pnl_r IS NOT NULL` (eligible).

Usage
-----
    python research/audit_l1_orb_g5_arithmetic_only_check.py

Emits the result MD to
`docs/audit/results/2026-04-21-orb-g5-deployed-lane-arithmetic-check.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
from scipy import stats

from pipeline.paths import GOLD_DB_PATH

RULE_82_WR_SPREAD_PP = 3.0
RULE_82_EXPR_DELTA = 0.10


@dataclass(frozen=True)
class LaneSpec:
    strategy_id: str
    symbol: str
    orb_label: str
    orb_minutes: int
    rr_target: float
    entry_model: str = "E2"
    confirm_bars: int = 1
    min_size_pts: float = 5.0  # ORB_G5 canonical threshold


# The 3 ORB_G5 deployed lanes on topstep_50k_mnq_auto per
# docs/runtime/lane_allocation.json (2026-04-18 rebalance)
ORB_G5_DEPLOYED_LANES: tuple[LaneSpec, ...] = (
    LaneSpec(
        strategy_id="MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5",
        symbol="MNQ",
        orb_label="EUROPE_FLOW",
        orb_minutes=5,
        rr_target=1.5,
    ),
    LaneSpec(
        strategy_id="MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5",
        symbol="MNQ",
        orb_label="COMEX_SETTLE",
        orb_minutes=5,
        rr_target=1.5,
    ),
    LaneSpec(
        strategy_id="MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15",
        symbol="MNQ",
        orb_label="US_DATA_1000",
        orb_minutes=15,
        rr_target=1.5,
    ),
)


def fetch_lane_is_rows(lane: LaneSpec) -> list[tuple[date, float, str, float]]:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        size_col = f"orb_{lane.orb_label}_size"
        q = f"""
        SELECT
            o.trading_day,
            o.pnl_r,
            o.outcome,
            d.{size_col} AS orb_size_pts
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.trading_day = d.trading_day
         AND o.symbol = d.symbol
         AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = ?
          AND o.orb_label = ?
          AND o.orb_minutes = ?
          AND o.entry_model = ?
          AND o.confirm_bars = ?
          AND o.rr_target = ?
          AND o.trading_day < DATE '2026-01-01'
          AND o.entry_ts IS NOT NULL
          AND o.pnl_r IS NOT NULL
          AND d.{size_col} IS NOT NULL
        """
        return con.execute(
            q,
            [
                lane.symbol,
                lane.orb_label,
                lane.orb_minutes,
                lane.entry_model,
                lane.confirm_bars,
                lane.rr_target,
            ],
        ).fetchall()
    finally:
        con.close()


@dataclass
class GroupStats:
    label: str
    n: int
    wr_pct: float
    exp_r: float
    sd_r: float
    mean_orb_pts: float
    median_orb_pts: float
    pnls: np.ndarray


def group_stats(rows: list[tuple], label: str) -> GroupStats:
    n = len(rows)
    if n == 0:
        return GroupStats(label, 0, 0.0, 0.0, 0.0, 0.0, 0.0, np.array([], dtype=float))
    pnls = np.array([float(r[1]) for r in rows], dtype=float)
    wins = sum(1 for r in rows if r[2] == "win")
    orb_sizes = np.array([float(r[3]) for r in rows], dtype=float)
    return GroupStats(
        label=label,
        n=n,
        wr_pct=100.0 * wins / n,
        exp_r=float(pnls.mean()),
        sd_r=float(pnls.std(ddof=1)) if n > 1 else 0.0,
        mean_orb_pts=float(orb_sizes.mean()),
        median_orb_pts=float(np.median(orb_sizes)),
        pnls=pnls,
    )


def two_proportion_z(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    """Two-sample z-test for equality of proportions. Returns (z, two-sided p)."""
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = float(np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)))
    if se == 0.0:
        return float("nan"), 1.0
    z = (p1 - p2) / se
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


@dataclass
class LaneResult:
    lane: LaneSpec
    n_total: int
    fire: GroupStats
    nofire: GroupStats
    welch_t_pnl_r: float
    welch_p_pnl_r: float
    z_wr: float
    p_wr: float
    delta_wr_pp: float
    delta_exp_r: float
    rule_82_flagged: bool
    rule_82_reason: str
    size_ratio: float

    @property
    def underpowered_nofire(self) -> bool:
        # RULE 3.2: N_on < 30 is directional-only (power floor)
        return self.nofire.n < 30


def run_lane(lane: LaneSpec) -> LaneResult | None:
    rows = fetch_lane_is_rows(lane)
    if not rows:
        return None

    fire_rows = [r for r in rows if float(r[3]) >= lane.min_size_pts]
    nofire_rows = [r for r in rows if float(r[3]) < lane.min_size_pts]

    f = group_stats(fire_rows, "fire")
    nf = group_stats(nofire_rows, "nofire")

    t_pnl, p_pnl = welch(f.pnls, nf.pnls)

    wins_f = int(round(f.wr_pct * f.n / 100))
    wins_nf = int(round(nf.wr_pct * nf.n / 100))
    z_wr, p_wr = two_proportion_z(wins_f, f.n, wins_nf, nf.n)

    d_wr = f.wr_pct - nf.wr_pct
    d_er = f.exp_r - nf.exp_r

    wr_under = abs(d_wr) < RULE_82_WR_SPREAD_PP
    expr_over = abs(d_er) > RULE_82_EXPR_DELTA
    flagged = wr_under and expr_over
    reason = (
        f"|wr_spread|={abs(d_wr):.2f}pp (< {RULE_82_WR_SPREAD_PP}?  {wr_under}) "
        f"AND |Delta_ExpR|={abs(d_er):.4f} (> {RULE_82_EXPR_DELTA}?  {expr_over})"
    )

    return LaneResult(
        lane=lane,
        n_total=len(rows),
        fire=f,
        nofire=nf,
        welch_t_pnl_r=t_pnl,
        welch_p_pnl_r=p_pnl,
        z_wr=z_wr,
        p_wr=p_wr,
        delta_wr_pp=d_wr,
        delta_exp_r=d_er,
        rule_82_flagged=flagged,
        rule_82_reason=reason,
        size_ratio=(f.mean_orb_pts / nf.mean_orb_pts) if nf.mean_orb_pts > 0 else float("nan"),
    )


def render_md(results: list[LaneResult]) -> str:
    md: list[str] = []
    md.append("# ORB_G5 deployed-lane ARITHMETIC_ONLY confirmatory audit")
    md.append("")
    md.append("**Date:** 2026-04-21")
    md.append("**Branch:** `research/l1-orb-g5-arithmetic-only-check`")
    md.append("**Script:** `research/audit_l1_orb_g5_arithmetic_only_check.py`")
    md.append("**Parent claim:** PR #57 (merge `5a39ea20`) classified L1 as")
    md.append("  FILTER_CORRELATES_WITH_EDGE. This audit asks whether L1 — and the other")
    md.append("  two deployed ORB_G5 lanes — are behavioral edge or RULE 8.2 cost-screen.")
    md.append("**Rule:** `.claude/rules/backtesting-methodology.md` § RULE 8.2 ARITHMETIC_ONLY.")
    md.append("**Classification:** confirmatory audit (RULE 10: no new prereg required).")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## TL;DR")
    md.append("")
    arith_lanes = [r for r in results if r and r.rule_82_flagged]
    edge_lanes = [r for r in results if r and not r.rule_82_flagged and not r.underpowered_nofire and not np.isnan(r.p_wr) and r.p_wr < 0.05]
    under_lanes = [r for r in results if r and r.underpowered_nofire]
    unclear_lanes = [
        r for r in results
        if r and not r.rule_82_flagged and not r.underpowered_nofire
        and (np.isnan(r.p_wr) or r.p_wr >= 0.05)
    ]

    md.append("**Cross-lane verdict is heterogeneous** — the 3 deployed ORB_G5 lanes do NOT share")
    md.append("a common class under RULE 8.2:")
    md.append("")
    if arith_lanes:
        md.append(
            "- **Cost-gate (ARITHMETIC_ONLY)**: "
            + ", ".join(f"`{r.lane.strategy_id}`" for r in arith_lanes)
            + ". WR spread statistically zero (p > 0.05 on proportion z-test); all ExpR lift is "
              "cost-amplification."
        )
    if edge_lanes:
        md.append(
            "- **Behavioral content possible**: "
            + ", ".join(f"`{r.lane.strategy_id}`" for r in edge_lanes)
            + ". WR spread significant at p < 0.05 on proportion z-test; fire group has real "
              "directional-accuracy advantage beyond cost-math."
        )
    if under_lanes:
        md.append(
            "- **Underpowered (RULE 3.2)**: "
            + ", ".join(f"`{r.lane.strategy_id}` (N_nofire={r.nofire.n})" for r in under_lanes)
            + ". Non-fire bucket below 30-sample power floor — directional-only, not "
              "statistically conclusive."
        )
    if unclear_lanes:
        md.append(
            "- **No RULE 8.2 flag and no significant WR spread**: "
            + ", ".join(f"`{r.lane.strategy_id}`" for r in unclear_lanes)
            + ". Neither cost-gate nor behavioral-edge by these tests; may need different "
              "classification."
        )
    md.append("")
    md.append(
        "**Consequence for action queue item #2**: the original premise (\"replace ORB_G5 "
        "selectivity on L1 with pre-break context\") is supported ONLY for L1 EUROPE_FLOW, "
        "where ORB_G5 is a cost-gate and not a behavioral filter. On other ORB_G5 lanes the "
        "filter is carrying behavioral information or the test is underpowered. Lane-specific "
        "reframing is required — a single portfolio-level rewrite is wrong."
    )
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Per-lane summary table")
    md.append("")
    md.append(
        "| Lane | N | N_fire | N_nofire | Δ_WR (pp) | z_WR | p_WR | Δ_ExpR | Welch t | Welch p | Size ratio | RULE 8.2 |"
    )
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        if r is None:
            md.append("| — (no data) | — | — | — | — | — | — | — | — | — | — | — |")
            continue
        nofire_tag = r.nofire.n
        if r.underpowered_nofire:
            nofire_tag = f"{r.nofire.n}*"
        md.append(
            f"| {r.lane.strategy_id} | {r.n_total} | {r.fire.n} | {nofire_tag} | "
            f"{r.delta_wr_pp:+.2f} | {r.z_wr:+.2f} | {r.p_wr:.4f} | "
            f"{r.delta_exp_r:+.4f} | {r.welch_t_pnl_r:+.2f} | {r.welch_p_pnl_r:.4f} | "
            f"{r.size_ratio:.2f}x | {'**ARITH**' if r.rule_82_flagged else 'edge?'} |"
        )
    md.append("")
    md.append("`*` = N_nofire < 30 (RULE 3.2 power floor; directional-only on that cell).")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Per-lane detail")
    md.append("")
    for r in results:
        if r is None:
            continue
        md.append(f"### {r.lane.strategy_id}")
        md.append("")
        md.append(
            f"- Spec: {r.lane.symbol} × {r.lane.orb_label} × {r.lane.entry_model} × "
            f"CB={r.lane.confirm_bars} × O{r.lane.orb_minutes} × RR={r.lane.rr_target} × ORB_G5"
        )
        md.append(f"- Total N (IS, eligible): **{r.n_total:,}**")
        md.append("")
        md.append("| Group | N | WR | ExpR | σ(R) | mean_ORB | median_ORB |")
        md.append("|---|---|---|---|---|---|---|")
        md.append(
            f"| FIRE (orb_size ≥ 5) | {r.fire.n:,} | {r.fire.wr_pct:.2f}% | "
            f"{r.fire.exp_r:+.4f}R | {r.fire.sd_r:.3f} | "
            f"{r.fire.mean_orb_pts:.2f} | {r.fire.median_orb_pts:.2f} |"
        )
        md.append(
            f"| NON-FIRE (orb_size < 5) | {r.nofire.n:,} | {r.nofire.wr_pct:.2f}% | "
            f"{r.nofire.exp_r:+.4f}R | {r.nofire.sd_r:.3f} | "
            f"{r.nofire.mean_orb_pts:.2f} | {r.nofire.median_orb_pts:.2f} |"
        )
        md.append("")
        md.append(f"- **Δ_WR** = {r.delta_wr_pp:+.2f}pp")
        md.append(
            f"- **Two-proportion z on WR** = {r.z_wr:+.3f}, p = {r.p_wr:.4f} → "
            f"{'WR spread statistically zero (no directional content)' if r.p_wr > 0.05 else 'WR spread significant'}"
        )
        md.append(f"- **Δ_ExpR** = {r.delta_exp_r:+.4f}R")
        md.append(
            f"- **Welch t** on pnl_r = {r.welch_t_pnl_r:+.3f}, p = {r.welch_p_pnl_r:.4f}"
        )
        md.append(f"- **Size ratio** (mean_ORB fire / non-fire) = {r.size_ratio:.2f}x")
        md.append(f"- **RULE 8.2**: {r.rule_82_reason}")
        md.append(
            f"- **Verdict**: {'ARITHMETIC_ONLY (cost-screen, not behavioral edge)' if r.rule_82_flagged else 'NOT arithmetic_only (behavioral content possible)'}"
        )
        if r.underpowered_nofire:
            md.append(
                f"- **Caveat**: N_nofire = {r.nofire.n} is below RULE 3.2 power floor (30). "
                "Treat as directional-only evidence on this cell."
            )
        md.append("")
    md.append("---")
    md.append("")
    md.append("## Mechanism")
    md.append("")
    md.append(
        "MNQ: 1 point = $2.00 per micro contract. A 5-pt ORB stop = $10 risk; a 16-pt ORB "
        "stop = $32 risk. Fixed slippage + commission per trade (canonical `cost_model.COST_SPECS`) "
        "is a much larger fraction of small-ORB R-risk than large-ORB R-risk. `pnl_r` in "
        "`orb_outcomes` is net of canonical costs, so the fire-vs-non-fire Δ_ExpR reflects "
        "this cost amplification directly."
    )
    md.append("")
    md.append(
        "Reference: RULE 7 (`backtesting-methodology.md`) — "
        "`cost_risk_pct ∝ 1 / orb_size_pts` has near-perfect inverse correlation with ORB_G5. "
        "ORB_G5 is mechanically equivalent to an upper-bound cost-risk-pct filter."
    )
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Deployed lanes NOT in this audit (filter class differs)")
    md.append("")
    md.append(
        "- `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` — ATR_P50 filter (different class; "
        "separate audit warranted to ask whether it is behavioral vs volatility-regime-gate)."
    )
    md.append(
        "- `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` — COST_LT12 is **cost-gate by construction** "
        "(canonical filter name is explicitly cost-based)."
    )
    md.append(
        "- `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12` — same as above."
    )
    md.append("")
    md.append("Combined portfolio composition (6 lanes):")
    md.append(
        "- 3 ORB_G5 lanes — **audited here; result is heterogeneous (see TL;DR and per-lane)**"
    )
    md.append("- 2 COST_LT12 lanes — cost-gate by construction")
    md.append("- 1 ATR_P50 lane — different class (not audited here)")
    md.append("")
    md.append(
        "The heterogeneous ORB_G5 result means the portfolio's filter composition is more "
        "nuanced than either \"all cost-gate\" or \"all behavioral edge\" framings. Before "
        "any portfolio-level reclassification, each ORB_G5 lane should be revisited with the "
        "lane-specific evidence from this audit."
    )
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Follow-up (not in this PR)")
    md.append("")
    md.append(
        "1. **Scope action queue item #2 to L1 only.** The \"replace ORB_G5 selectivity with "
        "pre-break context\" premise is supported only for L1 EUROPE_FLOW, where ORB_G5 is a "
        "cost-gate. The COMEX_SETTLE ORB_G5 lane has real behavioral content (p_WR=0.012) and "
        "should not be part of a behavioral-overlay prereg. The US_DATA_1000_O15 lane is "
        "underpowered (N_nofire=4) and cannot be classified from this audit."
    )
    md.append(
        "2. **L1-specific rewrite**: the L1 behavioral-overlay prereg should compare "
        "candidate signals against the **unfiltered** L1 baseline (all trades, no ORB_G5 "
        "pre-gate), not against the filtered baseline, since ORB_G5 on L1 does not carry "
        "directional information."
    )
    md.append(
        "3. **COMEX_SETTLE ORB_G5 standalone follow-up**: the +14pp WR spread at p=0.012 is "
        "real behavioral evidence. Separate diagnostic warranted — what specifically about "
        "COMEX_SETTLE's small-ORB days causes -0.41R ExpR on N=81? Possible mechanisms: "
        "thin liquidity around settlement, NY-close overhang, overnight chop that persists."
    )
    md.append(
        "4. **US_DATA_1000_O15 power question**: only 4 non-fire days across 7 years of IS "
        "data. Either the ORB_G5 threshold rarely bites at O15 (typical mechanism: the 15m "
        "ORB is almost always ≥5 pts), or there is a sampling issue. Cannot conclude from 4 "
        "trades."
    )
    md.append(
        "5. **ATR_P50 audit on L2.** Separate question; ATR_P50 is a different filter class "
        "(volatility-regime, not size). Needs its own fire-vs-non-fire decomposition. Probably "
        "not arithmetic_only but needs explicit real-data verification."
    )
    md.append(
        "6. **Cross-session pre-break descriptive diagnostic.** Highest-EV next-step research "
        "per the earlier brutal audit: measure `pre_velocity` / `vwap` directional content on "
        "the UNFILTERED universe across 12 × 3 × 3 lane combos. Separate PR."
    )
    md.append(
        "7. **No portfolio metadata reclassification yet.** Earlier draft of this audit assumed "
        "all ORB_G5 lanes were cost-gate. Real data rejects that. Portfolio `CONFIDENCE_TIER` "
        "on `OrbSizeFilter` should remain PROVEN; a finer per-lane classification is premature "
        "until (3) and (4) resolve."
    )
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Reproduction")
    md.append("")
    md.append("```bash")
    md.append("python research/audit_l1_orb_g5_arithmetic_only_check.py")
    md.append("```")
    md.append("")
    md.append("Writes this document to `docs/audit/results/2026-04-21-orb-g5-deployed-lane-arithmetic-check.md`.")
    md.append("")
    return "\n".join(md)


def main() -> None:
    results: list[LaneResult] = []
    for lane in ORB_G5_DEPLOYED_LANES:
        r = run_lane(lane)
        results.append(r)  # type: ignore[arg-type]

    md = render_md([r for r in results if r is not None])
    out_path = Path("docs/audit/results/2026-04-21-orb-g5-deployed-lane-arithmetic-check.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")

    print("=== ORB_G5 DEPLOYED-LANE ARITHMETIC_ONLY AUDIT ===")
    for r in results:
        if r is None:
            print("  (lane produced no data)")
            continue
        tag = "ARITH" if r.rule_82_flagged else "edge?"
        wr_tag = "WR=0" if r.p_wr > 0.05 else "WR≠0"
        print(
            f"  {r.lane.strategy_id:<50} N={r.n_total:>4}  "
            f"d_WR={r.delta_wr_pp:+5.2f}pp  p_WR={r.p_wr:.3f} ({wr_tag})  "
            f"d_ExpR={r.delta_exp_r:+.3f}R  "
            f"sz={r.size_ratio:.2f}x  {tag}"
        )
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
