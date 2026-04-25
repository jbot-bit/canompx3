#!/usr/bin/env python3
"""ORB_G5 cross-session overlap decomposition on 3 deployed lanes.

Audits whether the portfolio's 3 deployed ORB_G5 lanes
(MNQ EUROPE_FLOW / COMEX_SETTLE / US_DATA_1000_O15) represent 3
statistically-independent signals or 3 re-counts of one driver.

Statistical grounding:
  - Driver statistic: max pairwise Pearson correlation on 0/1 fire masks
    (= phi coefficient on binary). Chosen over Jaccard because Jaccard is
    mechanically inflated at high marginal fire rates (two 80%-fire lanes
    under pure independence give Jaccard ~0.67).
  - Aggregate N̂ via Bailey & Lopez de Prado 2014 Appendix A.3 Eq. 9:
    `N̂ = ρ̂ + (1 − ρ̂)·M` where ρ̂ is the average pairwise correlation and
    M=3. Canonical "implied independent trials" estimator.
    Citation: `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md`
  - Per-pair max preferred over ρ̂ (average) per MEMORY.md 2026-04-20 rule:
    pooled averages hide opposite-sign / heterogeneous cells.
  - Precedent: `research/rel_vol_cross_scan_overlap_decomposition.py`
    (2026-04-19) used Jaccard on ~33%-fire-rate signals where independence
    baseline Jaccard is ~0.20. This script targets a higher-fire-rate
    filter (ORB_G5 on MNQ ~70-90%), so the fire-rate-agnostic ρ is the
    appropriate driver. Jaccard retained for transparency/reporting.

Pre-committed decision rule (locked in `docs/runtime/stages/orb_g5_cross_session_overlap.md`):
  max_pair_rho = max over (lane_i, lane_j) of Pearson(fire_mask_i, fire_mask_j)
  on Mode A IS (trading_day < 2026-01-01).

  - max_pair_rho > 0.50   -> DROP_CANDIDATE (BL14 N̂ < 2.0 for avg-rho;
                              ≥1 trial-equivalent lost)
  - 0.25 < max <= 0.50    -> PARTIAL_OVERLAP (BL14 2.0 <= N̂ < 2.5; monitor)
  - max_pair_rho <= 0.25  -> CLEAN_DIVERSIFICATION (N̂ >= 2.5)

Canonical integrity:
  - ORB_G5 fire mask via `research.filter_utils.filter_signal` -> `trading_app.config.ALL_FILTERS["ORB_G5"]`
  - NEVER re-encodes filter logic (2026-04-19 filter-delegation rule)
  - Loads lanes via canonical `research.comprehensive_deployed_lane_scan.load_lane`
  - Mode A holdout from `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`
  - Canonical DB path via `pipeline.paths.GOLD_DB_PATH`

Reads ONLY canonical tables (daily_features + orb_outcomes).
No writes to validated_setups / experimental_strategies / live_config.
No randomness; reproducible on the same DB state.

Method follows `research/rel_vol_cross_scan_overlap_decomposition.py` (2026-04-19 pattern).

Usage:
  DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db \\
    python research/orb_g5_cross_session_overlap_decomposition.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from research.comprehensive_deployed_lane_scan import load_lane
from research.filter_utils import filter_signal
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

# =============================================================================
# LANES UNDER TEST — 3 deployed ORB_G5 lanes per docs/runtime/lane_allocation.json
# =============================================================================

LANES = [
    {
        "id": "L1",
        "strategy_id": "MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5",
        "instrument": "MNQ",
        "session": "EUROPE_FLOW",
        "apt": 5,
        "rr": 1.5,
    },
    {
        "id": "L2",
        "strategy_id": "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5",
        "instrument": "MNQ",
        "session": "COMEX_SETTLE",
        "apt": 5,
        "rr": 1.5,
    },
    {
        "id": "L3",
        "strategy_id": "MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15",
        "instrument": "MNQ",
        "session": "US_DATA_1000",
        "apt": 15,
        "rr": 1.5,
    },
]

RESULT_PATH = (
    PROJECT_ROOT
    / "docs/audit/results/2026-04-20-orb-g5-cross-session-overlap-decomposition.md"
)

# Pre-committed decision-rule thresholds (stage file locked).
# Driver = max pairwise Pearson correlation on 0/1 fire masks.
RHO_DROP_THRESHOLD = 0.50
RHO_PARTIAL_THRESHOLD = 0.25


# =============================================================================
# FIRE-DAY COMPUTATION
# =============================================================================


@dataclass
class LaneDecomp:
    id: str
    strategy_id: str
    instrument: str
    session: str
    apt: int
    rr: float
    n_rows_loaded: int
    n_rows_is: int
    fire_days_is: set[date]
    fire_days_full: set[date]
    # Eligible trading-day sets — days where the lane had a trade row at all
    # (rows_loaded). Used to build the alignment grid for Pearson correlation
    # computation. A day where the lane was eligible but the filter didn't
    # fire contributes a (0) to the lane's fire mask on that day; including
    # these days is necessary for honest correlation (union-of-fires grid
    # systematically under-weights shared zeros).
    trade_days_is: set[date]
    trade_days_full: set[date]


def compute_fire_days(spec: dict) -> LaneDecomp:
    """Return fire-day sets (IS + full) for a single deployed lane.

    fire_day = (ORB_G5 filter fires at ORB close) AND (E2 took a trade, i.e.,
    pnl_r IS NOT NULL in orb_outcomes).

    ORB_G5 is direction-agnostic (size-only filter), so the overlap question
    is about shared *eligible trading days* across sessions, not shared
    direction. `load_lane` already filters on break_dir IN ('long','short')
    and pnl_r IS NOT NULL, so the DataFrame rows are real E2 trade rows.
    """
    df = load_lane(spec["session"], spec["apt"], spec["rr"], spec["instrument"])
    if len(df) == 0:
        return LaneDecomp(
            id=spec["id"],
            strategy_id=spec["strategy_id"],
            instrument=spec["instrument"],
            session=spec["session"],
            apt=spec["apt"],
            rr=spec["rr"],
            n_rows_loaded=0,
            n_rows_is=0,
            fire_days_is=set(),
            fire_days_full=set(),
            trade_days_is=set(),
            trade_days_full=set(),
        )

    # Canonical filter delegation (research-truth-protocol.md MANDATORY).
    # load_lane() aliases per-session columns to generic names (orb_size,
    # break_dir, rel_vol, ...) — rehydrate the canonical `orb_{session}_*`
    # names so ALL_FILTERS["ORB_G5"].matches_df can find them by the
    # per-session lookups it expects.
    view = df.copy()
    orb_label = spec["session"]
    _ALIAS_MAP = {
        f"orb_{orb_label}_high": "orb_high",
        f"orb_{orb_label}_low": "orb_low",
        f"orb_{orb_label}_size": "orb_size",
        f"orb_{orb_label}_break_dir": "break_dir",
        f"rel_vol_{orb_label}": "rel_vol",
    }
    for canon, alias in _ALIAS_MAP.items():
        if canon not in view.columns and alias in view.columns:
            view[canon] = view[alias]

    fire_mask = filter_signal(view, "ORB_G5", orb_label).astype(bool)

    # Fire = filter passes AND a trade was actually taken (pnl_r non-null
    # is already enforced in load_lane's WHERE clause; confirm post-hoc).
    has_trade = df["pnl_r"].notna().to_numpy()
    active = fire_mask & has_trade

    is_mask = df["is_is"].astype(bool).to_numpy()

    fire_days_full: set[date] = set(
        pd.to_datetime(df.loc[active, "trading_day"]).dt.date.tolist()
    )
    fire_days_is: set[date] = set(
        pd.to_datetime(df.loc[active & is_mask, "trading_day"]).dt.date.tolist()
    )
    # Trade-days = days where the lane was eligible to trade (had a row with
    # break_dir + pnl_r). Includes both filter-fire and filter-no-fire days.
    # Used for the alignment-grid so Pearson captures shared zeros honestly.
    trade_days_full: set[date] = set(
        pd.to_datetime(df.loc[has_trade, "trading_day"]).dt.date.tolist()
    )
    trade_days_is: set[date] = set(
        pd.to_datetime(df.loc[has_trade & is_mask, "trading_day"]).dt.date.tolist()
    )

    return LaneDecomp(
        id=spec["id"],
        strategy_id=spec["strategy_id"],
        instrument=spec["instrument"],
        session=spec["session"],
        apt=spec["apt"],
        rr=spec["rr"],
        n_rows_loaded=int(len(df)),
        n_rows_is=int(is_mask.sum()),
        fire_days_is=fire_days_is,
        fire_days_full=fire_days_full,
        trade_days_is=trade_days_is,
        trade_days_full=trade_days_full,
    )


# =============================================================================
# DECOMPOSITION ANALYSIS
# =============================================================================


def build_alignment_matrix(
    lanes: list[LaneDecomp], window: str
) -> pd.DataFrame:
    """Rows = union of trade-days across lanes, cols = lane IDs, values = 0/1.

    Uses `trade_days_*` (eligible days) for the union, not `fire_days_*`.
    This captures the shared-zero structure (days where a lane was eligible
    but filter didn't fire), which is required for honest Pearson correlation.
    A lane's cell on a union-day is:
      - 1 if the lane fired on that day (d in fire_days_*)
      - 0 if the lane was eligible but didn't fire, OR wasn't eligible
    """
    trade_attr = "trade_days_is" if window == "is" else "trade_days_full"
    fire_attr = "fire_days_is" if window == "is" else "fire_days_full"
    all_days = sorted(set().union(*(getattr(c, trade_attr) for c in lanes)))
    data = {
        c.id: [1 if d in getattr(c, fire_attr) else 0 for d in all_days]
        for c in lanes
    }
    return pd.DataFrame(data, index=pd.Index(all_days, name="trading_day"))


def pairwise_stats(lanes: list[LaneDecomp], window: str) -> pd.DataFrame:
    attr = "fire_days_is" if window == "is" else "fire_days_full"
    rows = []
    for i, a in enumerate(lanes):
        for b in lanes[i + 1 :]:
            a_set = getattr(a, attr)
            b_set = getattr(b, attr)
            inter = a_set & b_set
            union = a_set | b_set
            jaccard = len(inter) / len(union) if union else 0.0
            min_denom = min(len(a_set), len(b_set))
            overlap_pct_min = (
                100.0 * len(inter) / min_denom if min_denom > 0 else None
            )
            rows.append(
                {
                    "pair": f"{a.id} x {b.id}",
                    "A": f"{a.session}",
                    "B": f"{b.session}",
                    "|A|": len(a_set),
                    "|B|": len(b_set),
                    "|A ∩ B|": len(inter),
                    "|A ∪ B|": len(union),
                    "Jaccard": jaccard,
                    "Overlap % of min(|A|,|B|)": overlap_pct_min,
                }
            )
    return pd.DataFrame(rows)


def multi_way_overlap(align: pd.DataFrame) -> dict[int, int]:
    """Count trading_days where exactly k lanes fired simultaneously, k=1..N."""
    k_count = align.sum(axis=1)
    return {k: int((k_count == k).sum()) for k in range(1, len(align.columns) + 1)}


def pairwise_correlation(
    lanes: list[LaneDecomp], window: str
) -> pd.DataFrame:
    """Compute per-pair Pearson correlation on the 2-way union of each pair.

    Using a single multi-way union to correlate a specific pair inflates
    their shared-zeros by counting days where NEITHER lane in the pair was
    eligible (only some other lane was). BL 2014's trial-correlation
    semantics require per-pair alignment.

    Returns a symmetric m×m DataFrame with 1.0 on the diagonal and
    pairwise Pearson-on-fire-masks off-diagonal.
    """
    trade_attr = "trade_days_is" if window == "is" else "trade_days_full"
    fire_attr = "fire_days_is" if window == "is" else "fire_days_full"
    cols = [c.id for c in lanes]
    mat = pd.DataFrame(
        np.eye(len(lanes)), index=pd.Index(cols), columns=pd.Index(cols)
    )
    for i, a in enumerate(lanes):
        for b in lanes[i + 1 :]:
            a_trade = getattr(a, trade_attr)
            b_trade = getattr(b, trade_attr)
            a_fire = getattr(a, fire_attr)
            b_fire = getattr(b, fire_attr)
            union_days = sorted(a_trade | b_trade)
            if not union_days:
                rho = 0.0
            else:
                a_vec = np.array(
                    [1.0 if d in a_fire else 0.0 for d in union_days]
                )
                b_vec = np.array(
                    [1.0 if d in b_fire else 0.0 for d in union_days]
                )
                # Guard against zero-variance (constant vector)
                if a_vec.std() == 0.0 or b_vec.std() == 0.0:
                    rho = 0.0
                else:
                    rho = float(np.corrcoef(a_vec, b_vec)[0, 1])
            mat.loc[a.id, b.id] = rho
            mat.loc[b.id, a.id] = rho
    return mat


def max_pairwise_rho(corr: pd.DataFrame) -> tuple[float, tuple[str, str]]:
    """Return (max off-diagonal Pearson correlation, pair-labels)."""
    m = corr.shape[0]
    if m < 2:
        return 0.0, ("", "")
    vals = corr.values
    cols = list(corr.columns)
    best = 0.0
    best_pair: tuple[str, str] = (cols[0], cols[1])
    for i in range(m):
        for j in range(i + 1, m):
            v = float(vals[i, j])
            if v > best:
                best = v
                best_pair = (cols[i], cols[j])
    return best, best_pair


def nyholt_meff(corr: pd.DataFrame) -> tuple[float, np.ndarray]:
    """Nyholt 2004 Meff from correlation eigenvalues.

    Meff = 1 + (m - 1) * (1 - Var(lambda) / m)
    """
    m = corr.shape[0]
    if m < 2:
        return float(m), np.array([])
    eigs = np.linalg.eigvalsh(corr.values)
    eigs = np.clip(eigs, 0.0, None)
    var_l = float(np.var(eigs, ddof=0))
    meff = 1.0 + (m - 1) * (1.0 - var_l / m)
    return meff, eigs


def bailey_lopez_de_prado_implied_n(corr: pd.DataFrame) -> tuple[float, float]:
    """Bailey & Lopez de Prado 2014 Eq. 9 — implied independent trials.

    N_hat = rho_hat + (1 - rho_hat) * M

    where rho_hat is the average off-diagonal correlation across the M trials.

    Source: `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md`
    (Appendix A.3).

    Returns (rho_hat, N_hat).
    """
    m = corr.shape[0]
    if m < 2:
        return 0.0, float(m)
    # Average of off-diagonal entries (upper-triangle, excluding diag).
    vals = corr.values
    offdiag: list[float] = []
    for i in range(m):
        for j in range(i + 1, m):
            offdiag.append(float(vals[i, j]))
    rho_hat = float(np.mean(offdiag)) if offdiag else 0.0
    # Eq. 9 is valid only for rho in (-1/(M-1), 1]. Clip negatives to 0 for
    # conservative reporting — negative correlations would inflate N above M.
    rho_clipped = max(rho_hat, 0.0)
    n_hat = rho_clipped + (1.0 - rho_clipped) * m
    return rho_hat, n_hat


def verdict(max_pair_rho: float) -> tuple[str, str]:
    """Apply pre-committed decision rule (driver = max pairwise Pearson rho)."""
    if max_pair_rho > RHO_DROP_THRESHOLD:
        return (
            "DROP_CANDIDATE",
            f"max pairwise ρ {max_pair_rho:.3f} > {RHO_DROP_THRESHOLD} — "
            "RECOMMEND dropping one of the overlapping lanes to free an "
            "allocator slot (BL14 N̂ < 2.0).",
        )
    if max_pair_rho > RHO_PARTIAL_THRESHOLD:
        return (
            "PARTIAL_OVERLAP",
            f"max pairwise ρ {max_pair_rho:.3f} in "
            f"({RHO_PARTIAL_THRESHOLD}, {RHO_DROP_THRESHOLD}] — "
            "partial concentration flagged; no capital action yet.",
        )
    return (
        "CLEAN_DIVERSIFICATION",
        f"max pairwise ρ {max_pair_rho:.3f} <= {RHO_PARTIAL_THRESHOLD} — "
        "three approximately-independent signals; portfolio story defensible.",
    )


# =============================================================================
# RENDER
# =============================================================================


def _fmt(x: float | None, places: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    return f"{x:.{places}f}"


def render(
    lanes: list[LaneDecomp],
    pair_df_is: pd.DataFrame,
    pair_df_full: pd.DataFrame,
    multi_is: dict[int, int],
    multi_full: dict[int, int],
    corr_is: pd.DataFrame,
    corr_full: pd.DataFrame,
    meff_is: float,
    meff_full: float,
    eigs_is: np.ndarray,
    rho_is: float,
    n_hat_is: float,
    rho_full: float,
    n_hat_full: float,
    max_rho_is: float,
    max_rho_pair_is: tuple[str, str],
    max_rho_full: float,
) -> str:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    max_jac_is = float(pair_df_is["Jaccard"].max()) if len(pair_df_is) else 0.0
    max_jac_full = (
        float(pair_df_full["Jaccard"].max()) if len(pair_df_full) else 0.0
    )
    verdict_is_label, verdict_is_text = verdict(max_rho_is)

    lines: list[str] = []
    lines.append("# ORB_G5 cross-session overlap decomposition")
    lines.append("")
    lines.append(f"**Generated:** {ts}")
    lines.append(
        "**Script:** `research/orb_g5_cross_session_overlap_decomposition.py`"
    )
    lines.append(
        f"**IS window:** `trading_day < {HOLDOUT_SACRED_FROM.isoformat()}` "
        "(Mode A, imported from `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`)."
    )
    lines.append(
        "**Filter source:** canonical `trading_app.config.ALL_FILTERS['ORB_G5']` "
        "via `research.filter_utils.filter_signal` (no re-encoded logic)."
    )
    lines.append("")
    lines.append("## Audited claim")
    lines.append("")
    lines.append(
        "Three of six deployed MNQ lanes in the 2026-04-18 allocator DEPLOY set "
        "use the `ORB_G5` size filter: MNQ EUROPE_FLOW, MNQ COMEX_SETTLE, and "
        "MNQ US_DATA_1000 (O15 variant). Portfolio sizing treats them as three "
        "independent signals. **Adversarial question:** do they fire on the same "
        "trading days (one driver counted three times), or on disjoint day sets "
        "(three genuinely diversifying signals)?"
    )
    lines.append("")
    lines.append("## Pre-committed decision rule")
    lines.append("")
    lines.append(
        f"Locked before this run in `docs/runtime/stages/orb_g5_cross_session_overlap.md`:"
    )
    lines.append("")
    lines.append(
        "**Driver statistic:** max pairwise Pearson correlation on 0/1 fire "
        "masks (= phi coefficient on binary). Chosen over Jaccard because "
        "Jaccard is mechanically inflated at high marginal fire rates (two "
        "80%-fire lanes under pure independence give Jaccard ~0.67)."
    )
    lines.append("")
    lines.append(
        f"- **max pairwise ρ > {RHO_DROP_THRESHOLD}** (IS) "
        "-> DROP_CANDIDATE: recommend dropping one of the overlapping lanes."
    )
    lines.append(
        f"- **{RHO_PARTIAL_THRESHOLD} < max ρ <= {RHO_DROP_THRESHOLD}** (IS) "
        "-> PARTIAL_OVERLAP: flag; no capital action."
    )
    lines.append(
        f"- **max ρ <= {RHO_PARTIAL_THRESHOLD}** (IS) "
        "-> CLEAN_DIVERSIFICATION."
    )
    lines.append("")
    lines.append(
        "Thresholds mapped to Bailey-Lopez de Prado 2014 Appendix A.3 Eq. 9 "
        "`N̂ = ρ̂ + (1−ρ̂)·M` (M=3): ρ=0.5 → N̂≈2.0; ρ=0.25 → N̂≈2.5; ρ=0.0 → N̂=3.0. "
        "Per-pair max preferred over ρ̂ (average) per MEMORY.md 2026-04-20 rule "
        "(pooled averages hide heterogeneous cells)."
    )
    lines.append("")
    lines.append("## Lanes under test")
    lines.append("")
    lines.append(
        "| ID | strategy_id | Session | ORB min | RR | Rows loaded | Rows IS | Fire-days IS | Fire-days full |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for c in lanes:
        lines.append(
            f"| {c.id} | `{c.strategy_id}` | {c.session} | {c.apt} | {c.rr} | "
            f"{c.n_rows_loaded} | {c.n_rows_is} | {len(c.fire_days_is)} | {len(c.fire_days_full)} |"
        )
    lines.append("")
    lines.append(
        "`Rows loaded` = E2 trade rows from `orb_outcomes` JOIN `daily_features` "
        "(pnl_r non-null, break_dir in {long,short}). `Fire-days` = subset where "
        "ORB_G5 canonical filter also fires (orb_size >= 5.0 points)."
    )
    lines.append("")

    # ----- IS WINDOW -----
    lines.append("## Pairwise overlap matrix — Mode A IS")
    lines.append("")
    lines.append(
        "| Pair | A | B | \\|A\\| | \\|B\\| | \\|A ∩ B\\| | \\|A ∪ B\\| | Jaccard | % of min |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for _, row in pair_df_is.iterrows():
        lines.append(
            f"| {row['pair']} | {row['A']} | {row['B']} | {row['|A|']} | {row['|B|']} | "
            f"{row['|A ∩ B|']} | {row['|A ∪ B|']} | {row['Jaccard']:.3f} | "
            f"{_fmt(row['Overlap % of min(|A|,|B|)'], 1)} |"
        )
    lines.append("")
    lines.append("## Simultaneous-fire distribution — Mode A IS")
    lines.append("")
    lines.append("| # lanes firing | Trading-days with this count |")
    lines.append("|---:|---:|")
    total_is = sum(multi_is.values())
    for k in sorted(multi_is):
        lines.append(f"| {k} | {multi_is[k]} |")
    lines.append(f"| **total fire-days** | **{total_is}** |")
    lines.append("")
    sum_is = sum(len(c.fire_days_is) for c in lanes)
    if total_is > 0:
        redundancy_is = 1.0 - (total_is / sum_is)
        lines.append(
            f"Sum per-lane fire-days: {sum_is}. Union: {total_is}. "
            f"Redundancy = 1 − union/sum = {redundancy_is:.3f}."
        )
    lines.append("")
    lines.append("## Pairwise Pearson correlation — Mode A IS")
    lines.append("")
    corr_r = corr_is.round(3)
    header = "| | " + " | ".join(corr_r.columns) + " |"
    sep = "|---|" + "|".join(["---:"] * len(corr_r.columns)) + "|"
    lines.append(header)
    lines.append(sep)
    for idx, row in corr_r.iterrows():
        cells_str = " | ".join(f"{v:.3f}" for v in row.values)
        lines.append(f"| {idx} | {cells_str} |")
    lines.append("")
    lines.append("## Nyholt 2004 M-effective — Mode A IS")
    lines.append("")
    lines.append(f"**Meff = {meff_is:.3f}** (out of m={len(lanes)} lanes).")
    lines.append("")
    lines.append(
        f"Eigenvalues of the correlation matrix: "
        f"{np.array2string(eigs_is, precision=3)}."
    )
    lines.append("")
    lines.append("## Bailey-Lopez de Prado 2014 Eq. 9 implied independent trials — Mode A IS")
    lines.append("")
    lines.append(
        "Canonical formula from `docs/institutional/literature/"
        "bailey_lopez_de_prado_2014_deflated_sharpe.md` Appendix A.3 Eq. 9: "
        "`N̂ = ρ̂ + (1 − ρ̂)·M`, where `ρ̂` is the average off-diagonal "
        "correlation and `M = 3` lanes."
    )
    lines.append("")
    lines.append(
        f"**Observed:** ρ̂ = {rho_is:.3f} → **N̂ = {n_hat_is:.3f}** independent trials "
        f"(out of M = {len(lanes)})."
    )
    lines.append("")
    lines.append(
        "Interpretation: ρ̂ near 0 → N̂ ≈ M (full independence); ρ̂ near 1 → "
        "N̂ ≈ 1 (one effective signal). BL 2014 Appendix A.3 clips ρ̂ to the "
        "valid range; this implementation clips negative ρ̂ to 0 for conservative "
        "reporting."
    )
    lines.append("")

    # ----- FULL SAMPLE (reference) -----
    lines.append("## Pairwise overlap matrix — Full sample (IS + 2026 OOS)")
    lines.append("")
    lines.append(
        "_Reference only; verdict is driven by the Mode A IS table above._"
    )
    lines.append("")
    lines.append(
        "| Pair | A | B | \\|A\\| | \\|B\\| | \\|A ∩ B\\| | \\|A ∪ B\\| | Jaccard | % of min |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for _, row in pair_df_full.iterrows():
        lines.append(
            f"| {row['pair']} | {row['A']} | {row['B']} | {row['|A|']} | {row['|B|']} | "
            f"{row['|A ∩ B|']} | {row['|A ∪ B|']} | {row['Jaccard']:.3f} | "
            f"{_fmt(row['Overlap % of min(|A|,|B|)'], 1)} |"
        )
    lines.append("")
    total_full = sum(multi_full.values())
    full_multi_fmt = ", ".join(f"{k}={multi_full[k]}" for k in sorted(multi_full))
    lines.append(
        f"Full-sample summary: union fire-days {total_full} ({full_multi_fmt}); "
        f"Meff = {meff_full:.3f}; ρ̂ = {rho_full:.3f}; "
        f"N̂ = {n_hat_full:.3f} (BL 2014 Eq.9); "
        f"max pair ρ = {max_rho_full:.3f}; max Jaccard = {max_jac_full:.3f}."
    )
    _ = corr_full  # included for parity with IS table; not rendered separately
    lines.append("")

    # ----- VERDICT -----
    lines.append("## Verdict")
    lines.append("")
    lines.append(
        f"**Mode A IS max pairwise ρ: {max_rho_is:.3f}** "
        f"(pair: {max_rho_pair_is[0]} × {max_rho_pair_is[1]})"
    )
    lines.append("")
    lines.append(
        f"Supporting: avg ρ̂ = {rho_is:.3f} → N̂ = {n_hat_is:.3f} (BL14); "
        f"Nyholt Meff = {meff_is:.3f}; max pair Jaccard = {max_jac_is:.3f}."
    )
    lines.append("")
    lines.append(f"**Label:** `{verdict_is_label}`")
    lines.append("")
    lines.append(verdict_is_text)
    lines.append("")
    # Caveats section — flag any lane where ORB_G5 is near-trivial.
    lines.append("## Caveats")
    lines.append("")
    degenerate = [c for c in lanes if c.n_rows_is > 0 and len(c.fire_days_is) / c.n_rows_is >= 0.97]
    if degenerate:
        for c in degenerate:
            fr = 100.0 * len(c.fire_days_is) / c.n_rows_is
            lines.append(
                f"- **{c.id} ({c.session} O{c.apt}) ORB_G5 fire rate = {fr:.1f}%.** "
                "On this lane the filter is effectively a pass-through gate; the fire "
                "mask is a near-constant vector. Pair-ρ involving this lane has limited "
                "variance-based statistical power. Interpret its ~0 correlations as "
                "'cannot detect coupling from a constant vector' rather than strong "
                "evidence of independence."
            )
        lines.append("")
    else:
        lines.append(
            "- No lane has ORB_G5 fire rate ≥ 97%; all 3 fire masks have usable variance."
        )
        lines.append("")
    lines.append(
        "- This audit measures **decision-level** correlation (fire/no-fire), not "
        "**return-level** correlation. Two lanes with ρ=0 on fire masks could still "
        "produce correlated P&L if they fire on different days but share the same "
        "regime driver. A return-correlation follow-up (P&L per trade_day per lane) "
        "is the natural next audit if return coupling becomes the question."
    )
    lines.append("")

    max_row = pair_df_is.sort_values("Jaccard", ascending=False).iloc[0].to_dict()
    if verdict_is_label == "DROP_CANDIDATE":
        lines.append("### Which lane to drop")
        lines.append("")
        lines.append(
            f"Most-overlapping pair: **{max_row['pair']}** "
            f"({max_row['A']} vs {max_row['B']}, Jaccard={float(max_row['Jaccard']):.3f})."
        )
        lines.append("")
        lines.append(
            "Next step before acting: look up per-lane Mode A ExpR from "
            "`research/mode_a_revalidation_active_setups.py`'s output at "
            "`docs/audit/results/2026-04-19-mode-a-revalidation-of-active-setups.md`. "
            "Drop the weaker ExpR; retain the stronger. Do NOT drop based on "
            "overlap alone."
        )
    elif verdict_is_label == "PARTIAL_OVERLAP":
        lines.append(
            f"Most-overlapping pair: **{max_row['pair']}** "
            f"({max_row['A']} vs {max_row['B']}, Jaccard={float(max_row['Jaccard']):.3f}). "
            "Monitor; no capital action at this Jaccard."
        )
    lines.append("")

    # ----- REPRODUCTION -----
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```")
    lines.append(
        "DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db "
        "python research/orb_g5_cross_session_overlap_decomposition.py"
    )
    lines.append("```")
    lines.append("")
    lines.append(
        "No randomness. Fire masks computed via canonical "
        "`trading_app.config.ALL_FILTERS['ORB_G5']` through "
        "`research.filter_utils.filter_signal`. No writes to "
        "`validated_setups` / `experimental_strategies` / `live_config`."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:
    print("Loading 3 deployed ORB_G5 lanes...")
    lanes: list[LaneDecomp] = []
    for spec in LANES:
        lane = compute_fire_days(spec)
        lanes.append(lane)
        print(
            f"  {lane.id} {lane.strategy_id}: "
            f"rows={lane.n_rows_loaded} rows_is={lane.n_rows_is} "
            f"fire_is={len(lane.fire_days_is)} fire_full={len(lane.fire_days_full)}"
        )

    # IS window
    align_is = build_alignment_matrix(lanes, "is")
    pair_is = pairwise_stats(lanes, "is")
    multi_is = multi_way_overlap(align_is)
    corr_is = pairwise_correlation(lanes, "is")
    meff_is, eigs_is = nyholt_meff(corr_is)
    rho_is, n_hat_is = bailey_lopez_de_prado_implied_n(corr_is)
    max_rho_is, max_rho_pair_is = max_pairwise_rho(corr_is)
    max_jac_is = float(pair_is["Jaccard"].max()) if len(pair_is) else 0.0

    # Full sample
    align_full = build_alignment_matrix(lanes, "full")
    pair_full = pairwise_stats(lanes, "full")
    multi_full = multi_way_overlap(align_full)
    corr_full = pairwise_correlation(lanes, "full")
    meff_full, _ = nyholt_meff(corr_full)
    rho_full, n_hat_full = bailey_lopez_de_prado_implied_n(corr_full)
    max_rho_full, _ = max_pairwise_rho(corr_full)

    print()
    print(f"IS  : union_fire_days={len(align_is)}  multi={multi_is}")
    print(
        f"      max_pair_ρ={max_rho_is:.3f} ({max_rho_pair_is[0]} × {max_rho_pair_is[1]})  "
        f"avg_ρ̂={rho_is:.3f}  N̂_BL14={n_hat_is:.3f}"
    )
    print(
        f"      Meff={meff_is:.3f} (m=3)  max_pair_Jaccard={max_jac_is:.3f}"
    )
    label, text = verdict(max_rho_is)
    print(f"      VERDICT: {label}")
    print(f"      {text}")

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        render(
            lanes=lanes,
            pair_df_is=pair_is,
            pair_df_full=pair_full,
            multi_is=multi_is,
            multi_full=multi_full,
            corr_is=corr_is,
            corr_full=corr_full,
            meff_is=meff_is,
            meff_full=meff_full,
            eigs_is=eigs_is,
            rho_is=rho_is,
            n_hat_is=n_hat_is,
            rho_full=rho_full,
            n_hat_full=n_hat_full,
            max_rho_is=max_rho_is,
            max_rho_pair_is=max_rho_pair_is,
            max_rho_full=max_rho_full,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {RESULT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
