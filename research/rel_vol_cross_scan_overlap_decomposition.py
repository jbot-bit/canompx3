#!/usr/bin/env python3
"""rel_vol_HIGH_Q3 cross-lane overlap decomposition.

Audits the '5 independent (instrument, session) BH-global survivors' claim
for the rel_vol_HIGH_Q3 feature in
`docs/audit/results/2026-04-15-comprehensive-deployed-lane-scan.md`.

The claim that has propagated into memory and downstream docs is that
rel_vol_HIGH_Q3 produces 5+ independent BH-global-significant cells at
K=14,261, making it a "universal volume confirmation" signal. This audit
tests whether those cells are statistically-independent draws or whether
their per-day fire masks overlap enough to collapse effective-K.

Method:
  1. Identify the 5 (instrument, session, direction) (instrument, session,
     RR) non-twin BH-global survivors from the comprehensive scan.
  2. For each cell, reproduce the per-day fire mask using the same
     bucket_high-based methodology the scan uses (67th percentile of the
     cell's full rel_vol distribution; `bucket_high(rel_vol, 67)`).
  3. Align fires on trading_day. Compute pairwise Jaccard, pairwise
     intersection, simultaneous-fire distribution (how many cells fire
     on each day that saw any fire).
  4. Report effective number of independent signals heuristic (Nyholt 2004
     Meff from correlation eigenvalues) and a raw "unique fire days /
     total fire count" redundancy ratio.

Reads ONLY canonical tables (daily_features + orb_outcomes). No writes to
validated_setups or experimental_strategies. No randomness; reproducible
on the same DB state.

Output: docs/audit/results/2026-04-19-rel-vol-cross-scan-overlap-decomposition.md

Usage:
  DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/rel_vol_cross_scan_overlap_decomposition.py
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

from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

# Canonical replication of comprehensive_deployed_lane_scan's rel_vol_HIGH_Q3
# fire definition. Imported to guarantee fire masks match the audited scan.
from research.comprehensive_deployed_lane_scan import bucket_high, load_lane

# =============================================================================
# CELLS UNDER DECOMPOSITION
# =============================================================================

# 5 non-twin (instrument, session, direction) BH-global rel_vol_HIGH_Q3
# survivors from docs/audit/results/2026-04-15-comprehensive-deployed-lane-scan.md
# Lines 142, 144, 145, 146, 151 of the BH-Global table (filtered to non-twin).
# (MNQ COMEX_SETTLE short at RR1.0 is BH-global; RR1.5 is the deployed twin —
# included separately for reference but not counted as 6th independent cell.)
CELLS = [
    {
        "id": "C1",
        "instrument": "MES",
        "session": "TOKYO_OPEN",
        "apt": 5,
        "rr": 1.0,
        "direction": "long",
        "scan_t": 5.43,
        "scan_n_on": 276,
    },
    {
        "id": "C2",
        "instrument": "MES",
        "session": "COMEX_SETTLE",
        "apt": 5,
        "rr": 1.0,
        "direction": "short",
        "scan_t": 4.89,
        "scan_n_on": 288,
    },
    {
        "id": "C3",
        "instrument": "MNQ",
        "session": "COMEX_SETTLE",
        "apt": 5,
        "rr": 1.0,
        "direction": "short",
        "scan_t": 4.88,
        "scan_n_on": 298,
    },
    {
        "id": "C4",
        "instrument": "MGC",
        "session": "LONDON_METALS",
        "apt": 5,
        "rr": 1.0,
        "direction": "short",
        "scan_t": 4.78,
        "scan_n_on": 148,
    },
    {
        "id": "C5",
        "instrument": "MNQ",
        "session": "SINGAPORE_OPEN",
        "apt": 5,
        "rr": 1.0,
        "direction": "short",
        "scan_t": 4.27,
        "scan_n_on": 286,
    },
]

RESULT_PATH = (
    PROJECT_ROOT
    / "docs/audit/results/2026-04-19-rel-vol-cross-scan-overlap-decomposition.md"
)

# =============================================================================
# FIRE MASK COMPUTATION
# =============================================================================


def compute_fire_days(cell: dict) -> set[date]:
    """Return the set of IS trading_days where rel_vol_HIGH_Q3 fires AND
    the cell's direction matches the ORB break direction.

    Uses load_lane() + bucket_high() from comprehensive_deployed_lane_scan to
    guarantee the fire mask matches the audited scan exactly. bucket_high's
    67th percentile is computed over the full cell distribution (IS+OOS), per
    the audited scan's own convention; we do NOT second-guess that here.
    """
    df = load_lane(cell["session"], cell["apt"], cell["rr"], cell["instrument"])
    if len(df) == 0:
        return set()

    # rel_vol_HIGH_Q3 fire mask, replicated from
    # comprehensive_deployed_lane_scan._build_filters:
    if df["rel_vol"].notna().any():
        fire_all = bucket_high(df["rel_vol"], 67)
    else:
        fire_all = np.zeros(len(df), dtype=int)

    # Direction match (cell fires only on days where ORB broke in cell's
    # direction — the scan filters the test population to matching direction).
    dir_match = (df["break_dir"] == cell["direction"]).astype(int).values

    # IS window
    is_is = df["is_is"].astype(int).values

    active = (fire_all * dir_match * is_is).astype(bool)
    return set(pd.to_datetime(df.loc[active, "trading_day"]).dt.date.tolist())


# =============================================================================
# DECOMPOSITION ANALYSIS
# =============================================================================


@dataclass
class CellDecomp:
    id: str
    instrument: str
    session: str
    direction: str
    rr: float
    fire_days: set[date]
    scan_n_on: int


def build_alignment_matrix(cells: list[CellDecomp]) -> pd.DataFrame:
    """Rows = union trading_days, columns = cell IDs, values = 0/1 fire indicator."""
    all_days = sorted(set().union(*(c.fire_days for c in cells)))
    data = {c.id: [1 if d in c.fire_days else 0 for d in all_days] for c in cells}
    df = pd.DataFrame(data, index=pd.Index(all_days, name="trading_day"))
    return df


def pairwise_stats(cells: list[CellDecomp]) -> pd.DataFrame:
    rows = []
    for i, a in enumerate(cells):
        for b in cells[i + 1 :]:
            inter = a.fire_days & b.fire_days
            union = a.fire_days | b.fire_days
            jaccard = len(inter) / len(union) if union else 0.0
            min_denom = min(len(a.fire_days), len(b.fire_days))
            overlap_pct_min = 100.0 * len(inter) / min_denom if min_denom > 0 else None
            rows.append(
                {
                    "pair": f"{a.id} x {b.id}",
                    "A": f"{a.instrument} {a.session} {a.direction}",
                    "B": f"{b.instrument} {b.session} {b.direction}",
                    "|A|": len(a.fire_days),
                    "|B|": len(b.fire_days),
                    "|A ∩ B|": len(inter),
                    "|A ∪ B|": len(union),
                    "Jaccard": jaccard,
                    "Overlap % of min(|A|,|B|)": overlap_pct_min,
                }
            )
    return pd.DataFrame(rows)


def multi_way_overlap(align: pd.DataFrame) -> dict[int, int]:
    """Count trading_days where exactly k cells fired simultaneously, for k=1..N."""
    k_count = align.sum(axis=1)
    hist: dict[int, int] = {}
    for k in range(1, len(align.columns) + 1):
        hist[k] = int((k_count == k).sum())
    return hist


def pairwise_correlation(align: pd.DataFrame) -> pd.DataFrame:
    return align.corr()


def nyholt_meff(corr: pd.DataFrame) -> tuple[float, np.ndarray]:
    """Nyholt 2004 Meff estimate from correlation eigenvalues.

    Meff = 1 + (m - 1) * (1 - Var(lambda) / m)
    where lambda are the eigenvalues of the m x m correlation matrix.

    Meff in [1, m] — Meff == m means all columns are uncorrelated (full
    independence); Meff == 1 means they are perfectly correlated (one
    effective test).
    """
    m = corr.shape[0]
    if m < 2:
        return float(m), np.array([])
    eigs = np.linalg.eigvalsh(corr.values)
    # Guard against tiny negative eigenvalues from floating-point noise
    eigs = np.clip(eigs, 0.0, None)
    var_l = float(np.var(eigs, ddof=0))
    meff = 1.0 + (m - 1) * (1.0 - var_l / m)
    return meff, eigs


# =============================================================================
# RENDER
# =============================================================================


def _fmt(x: float | None, places: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    return f"{x:.{places}f}"


def render(cells: list[CellDecomp], pair_df: pd.DataFrame, align: pd.DataFrame,
           multi: dict[int, int], corr: pd.DataFrame, meff: float, eigs: np.ndarray) -> str:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines: list[str] = []
    lines.append("# rel_vol_HIGH_Q3 cross-lane overlap decomposition")
    lines.append("")
    lines.append(f"**Generated:** {ts}")
    lines.append(f"**Script:** `research/rel_vol_cross_scan_overlap_decomposition.py`")
    lines.append(
        f"**IS window:** `trading_day < {HOLDOUT_SACRED_FROM.isoformat()}` "
        "(Mode A, imported from `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`)."
    )
    lines.append("")
    lines.append("## Audited claim")
    lines.append("")
    lines.append(
        "`docs/audit/results/2026-04-15-comprehensive-deployed-lane-scan.md` § BH-FDR "
        "Survivors — Global reports 13 cells surviving at K_global=14,261. Of those, "
        "5 distinct (instrument, session, direction) lanes use the `rel_vol_HIGH_Q3` "
        "feature. The memory file `comprehensive_scan_apr15.md` and other downstream "
        "notes call this '5 independent BH-global survivors' and treat rel_vol_HIGH_Q3 "
        "as a universal volume-confirmation signal."
    )
    lines.append("")
    lines.append(
        "**Adversarial question:** are the 5 lanes independent draws, or are their "
        "per-day fires correlated enough to collapse effective-K? The Nyholt 2004 "
        "M-effective statistic answers the 'effective number of independent tests' "
        "question using correlation-eigenvalue structure."
    )
    lines.append("")
    lines.append("## Per-cell fire counts (IS)")
    lines.append("")
    lines.append("| ID | Instrument | Session | Dir | RR | Scan N_on | This-run fire-days | Match |")
    lines.append("|---|---|---|---|---:|---:|---:|---|")
    for c in cells:
        scan_row = next(x for x in CELLS if x["id"] == c.id)
        n_fire = len(c.fire_days)
        match = "✓" if abs(n_fire - scan_row["scan_n_on"]) <= 2 else "!!"
        lines.append(
            f"| {c.id} | {c.instrument} | {c.session} | {c.direction} | "
            f"{c.rr} | {scan_row['scan_n_on']} | {n_fire} | {match} |"
        )
    lines.append("")
    lines.append(
        "Match ✓ means the script's fire-days count is within ±2 of the audited scan's "
        "`N_on`. Exact match is not expected — the scan counts trades (rows in "
        "`orb_outcomes` with non-null `pnl_r`) while this script reports unique "
        "trading_days, which can differ by 0-2 when an outcome row is missing."
    )
    lines.append("")
    lines.append("## Pairwise overlap matrix")
    lines.append("")
    lines.append("| Pair | A | B | \\|A\\| | \\|B\\| | \\|A ∩ B\\| | \\|A ∪ B\\| | Jaccard | % of min |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for _, row in pair_df.iterrows():
        lines.append(
            f"| {row['pair']} | {row['A']} | {row['B']} | {row['|A|']} | {row['|B|']} | "
            f"{row['|A ∩ B|']} | {row['|A ∪ B|']} | {row['Jaccard']:.3f} | "
            f"{_fmt(row['Overlap % of min(|A|,|B|)'], 1)} |"
        )
    lines.append("")
    lines.append("**Interpretation:**")
    lines.append("")
    lines.append(
        "- Jaccard > 0.30 ≈ two cells share ≥ 30% of their fire days → likely not "
        "independent hypotheses on the same underlying driver."
    )
    lines.append(
        "- Jaccard < 0.10 with overlap-of-min < 20% ≈ weak coupling → approximately "
        "independent."
    )
    lines.append("")
    lines.append("## Simultaneous-fire distribution")
    lines.append("")
    lines.append("| # cells firing | Trading-days with this count |")
    lines.append("|---:|---:|")
    total = sum(multi.values())
    for k in sorted(multi):
        lines.append(f"| {k} | {multi[k]} |")
    lines.append(f"| **total fire-days** | **{total}** |")
    lines.append("")
    sum_fires = sum(len(c.fire_days) for c in cells)
    if total > 0:
        redundancy = 1.0 - (total / sum_fires)
        lines.append(
            f"Sum of per-cell fire-day counts: {sum_fires}. Union of fire-days: {total}. "
            f"Redundancy = 1 − union/sum = {redundancy:.3f}. Redundancy 0 ≈ fully disjoint "
            f"fire sets; 1 ≈ fully identical fire sets across all 5 cells."
        )
    lines.append("")
    lines.append("## Pairwise Pearson correlation of fire indicators")
    lines.append("")
    lines.append(
        "Correlation computed on the union-trading_day grid (rows that are not in "
        "a given cell's fire-set contribute 0). This is conservative — "
        "cross-instrument cells may have structurally disjoint eligible-day sets "
        "(e.g., MGC LONDON_METALS fires on MGC days; MNQ SINGAPORE_OPEN fires on "
        "MNQ days) and the correlation captures both co-firing and joint-absence."
    )
    lines.append("")
    corr_r = corr.round(3)
    header = "| | " + " | ".join(corr_r.columns) + " |"
    sep = "|---|" + "|".join(["---:"] * len(corr_r.columns)) + "|"
    lines.append(header)
    lines.append(sep)
    for idx, row in corr_r.iterrows():
        cells_str = " | ".join(f"{v:.3f}" for v in row.values)
        lines.append(f"| {idx} | {cells_str} |")
    lines.append("")
    lines.append("## Nyholt 2004 M-effective")
    lines.append("")
    lines.append(f"**Meff = {meff:.3f}** (out of m={len(cells)} cells).")
    lines.append("")
    lines.append(
        f"Eigenvalues of the correlation matrix: {np.array2string(eigs, precision=3)}."
    )
    lines.append("")
    lines.append(
        "Meff heuristic: Meff ≈ m (all 5 cells here ≈ 5) means the 5 lanes are "
        "approximately independent draws on the alignment grid and K_effective for "
        "the rel_vol_HIGH_Q3 family is ≈ 5, not ≈ 1. Meff ≈ 1 means a single "
        "effective test. Values in-between interpolate."
    )
    lines.append("")
    lines.append("## Verdict heuristic")
    lines.append("")
    ok_independence = meff >= (len(cells) - 0.5)
    lines.append(
        "- If Meff ≥ m − 0.5 AND max pairwise Jaccard < 0.20 → the '5 independent "
        "survivors' framing is supported; DSR / BH-FDR verdicts computed with K ≈ 5 "
        "are honest."
    )
    lines.append(
        "- If Meff is materially below m (e.g., < 0.7·m) OR max Jaccard ≥ 0.30 → "
        "the framing is overstated; effective K collapses toward Meff and downstream "
        "DSR/BH claims need re-computation with the reduced K."
    )
    lines.append("")
    max_jac = float(pair_df["Jaccard"].max()) if len(pair_df) else 0.0
    lines.append(f"Observed: Meff = {meff:.3f} / m = {len(cells)}; max pairwise Jaccard = {max_jac:.3f}.")
    lines.append("")
    if ok_independence and max_jac < 0.20:
        lines.append(
            "**Heuristic verdict:** the '5 independent BH-global survivors' framing "
            "is supported by this decomposition. Continue treating K_effective ≈ 5 "
            "for the rel_vol_HIGH_Q3 family in downstream DSR / BH-FDR reports."
        )
    else:
        lines.append(
            "**Heuristic verdict:** the '5 independent BH-global survivors' framing "
            "is NOT fully supported. Effective K is below 5. Downstream DSR / BH-FDR "
            "reports that used K=5 on the family survivors should be re-reviewed with "
            f"K_effective ≈ {meff:.1f} and the max-Jaccard pair flagged for possible "
            "co-firing driver."
        )
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```")
    lines.append(
        "DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db "
        "python research/rel_vol_cross_scan_overlap_decomposition.py"
    )
    lines.append("```")
    lines.append("")
    lines.append(
        "No randomness. Fire masks replicated from "
        "`research.comprehensive_deployed_lane_scan.bucket_high`. "
        "No writes to `validated_setups` / `experimental_strategies`."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:
    print("Loading fire masks for 5 BH-global rel_vol_HIGH_Q3 cells...")
    cells: list[CellDecomp] = []
    for spec in CELLS:
        fire = compute_fire_days(spec)
        cells.append(
            CellDecomp(
                id=spec["id"],
                instrument=spec["instrument"],
                session=spec["session"],
                direction=spec["direction"],
                rr=spec["rr"],
                fire_days=fire,
                scan_n_on=spec["scan_n_on"],
            )
        )
        print(
            f"  {spec['id']} {spec['instrument']} {spec['session']} {spec['direction']} "
            f"scan_N_on={spec['scan_n_on']} this_run={len(fire)}"
        )

    align = build_alignment_matrix(cells)
    pair_df = pairwise_stats(cells)
    multi = multi_way_overlap(align)
    corr = pairwise_correlation(align)
    meff, eigs = nyholt_meff(corr)

    print()
    print(f"Union trading_days with any fire: {len(align)}")
    print(f"Multi-way distribution: {multi}")
    print(f"Nyholt Meff = {meff:.3f} (m={len(cells)})")
    print(f"Max pairwise Jaccard = {pair_df['Jaccard'].max():.3f}")

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        render(cells, pair_df, align, multi, corr, meff, eigs), encoding="utf-8"
    )
    print(f"\nWrote {RESULT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
