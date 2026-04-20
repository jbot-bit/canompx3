#!/usr/bin/env python3
"""
6-lane correlation + mechanism-concentration audit.

Question answered (from reframing audit 2026-04-20):
  "What is the correlation structure across the 6 deployed lanes, and what
   single mechanism could draw down them all simultaneously?"

Why this audit:
  Allocator optimised per-lane score (ExpR × Sharpe etc) without seeing
  portfolio-level correlation structure. 6/6 lanes are MNQ; 3/6 share
  ORB_G5 filter; 5/6 share RR1.5; 6/6 share entry model E2. Surface
  concentration flags — but actual trade-return correlation decides
  whether concentration translates to correlated drawdown.

Literature grounding:
  - Carver 2015 Ch 11 (Systematic Trading, p165-175)
      docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md
      Plus direct PDF extract pages 183-193 (Ch 11 full body).
      Key results:
        - Subsystem correlation = 0.7 × instrument return correlation
          (dynamic trading systems, p167-168)
        - Instrument diversification multiplier HARD CAP 2.5 (p170)
          explicitly because "crisis correlations jump higher"
        - Semi-automatic trader allocation = 100% / max_concurrent_bets
          (p169)
  - Lopez de Prado 2020 (ML for Asset Managers, CPCV framing)
      portfolio correlation structure informs effective-bets calculation
      independent of per-lane Sharpe ranking.

Method:
  1. Load 6 deployed lanes from docs/runtime/lane_allocation.json
  2. For each lane, get trade days from strategy_trade_days table; join
     to orb_outcomes to get pnl_r per (lane, trading_day).
  3. Pivot to wide matrix: rows = trading_day, cols = strategy_id,
     cells = pnl_r (0.0 where the lane did not trade that day — Carver
     portfolio accounting treats no-trade days as 0 subsystem return).
  4. Compute pairwise Pearson correlation on the pnl_r matrix.
  5. Cluster by (session_region, filter_family, aperture, rr_target).
  6. Compute effective independent bets N_eff using Carver's approximation:
        diversification_multiplier = 1 / sqrt(w' @ C @ w)
        where w = equal weights 1/6 and C is the correlation matrix.
     Effective bets = 1 / sum(w_i^2 / div_mult_term_i), but simplest is
     N_eff = diversification_multiplier^2 for equal weights.
  7. Report concentration verdict.

Output:
  - Console summary with correlation matrix, eigenvalue decomposition,
    effective independent bets, and a recommendation.
  - Optional CSV artefact to research/output/.

Idempotent / read-only: no DB writes, no memory writes.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH  # noqa: E402

# Deployed-lane manifest path (canonical source for who is live)
LANE_ALLOC_PATH = PROJECT_ROOT / "docs" / "runtime" / "lane_allocation.json"

SESSION_REGION = {
    "TOKYO_OPEN": "ASIA",
    "SINGAPORE_OPEN": "ASIA",
    "EUROPE_FLOW": "EUROPE",
    "COMEX_SETTLE": "EUROPE/US",
    "NYSE_OPEN": "US",
    "US_DATA_1000": "US",
}

FILTER_FAMILY = {
    "ORB_G5": "range-quality-gate",
    "ORB_G8": "range-quality-gate",
    "ATR_P50": "vol-regime",
    "ATR_P60": "vol-regime",
    "COST_LT12": "cost-regime",
}


def parse_strategy_id(sid: str) -> dict:
    """Unpack canonical strategy_id:
    <INSTRUMENT>_<SESSION>_<ENTRY>_RR<x>_CB<y>_<FILTER>[_O<minutes>]"""
    parts = sid.split("_")
    inst = parts[0]
    # Find entry model position (Ex)
    for idx, p in enumerate(parts):
        if p.startswith("E") and len(p) == 2 and p[1].isdigit():
            entry_idx = idx
            break
    else:
        raise ValueError(f"no entry model in {sid}")
    session = "_".join(parts[1:entry_idx])
    entry = parts[entry_idx]
    rr = float(parts[entry_idx + 1].removeprefix("RR"))
    cb = int(parts[entry_idx + 2].removeprefix("CB"))
    tail = parts[entry_idx + 3 :]
    # optional trailing O{minutes}
    if tail and tail[-1].startswith("O") and tail[-1][1:].isdigit():
        orb_minutes = int(tail[-1][1:])
        filter_type = "_".join(tail[:-1])
    else:
        orb_minutes = 5
        filter_type = "_".join(tail)
    return {
        "strategy_id": sid,
        "instrument": inst,
        "session": session,
        "entry_model": entry,
        "rr_target": rr,
        "confirm_bars": cb,
        "orb_minutes": orb_minutes,
        "filter_type": filter_type,
        "region": SESSION_REGION.get(session, "?"),
        "filter_family": FILTER_FAMILY.get(filter_type, "?"),
    }


def load_lanes() -> list[dict]:
    with open(LANE_ALLOC_PATH) as f:
        data = json.load(f)
    raw = data.get("deployed_lanes") or data.get("lanes") or []
    return [parse_strategy_id(ln["strategy_id"]) for ln in raw]


def load_trade_returns(con: duckdb.DuckDBPyConnection, lanes: list[dict]) -> pd.DataFrame:
    """Return wide DataFrame indexed by trading_day, one column per strategy_id,
    cells = pnl_r on days where the canonical filter fires, NaN otherwise.

    Filter application delegated to ALL_FILTERS[filter_type].matches_row()
    (institutional-rigor rule #4: never re-encode canonical logic). Per lane we
    JOIN (trading_day, symbol, orb_minutes) from daily_features against
    orb_outcomes to get the filter-input row and the P&L, then call
    matches_row() in Python to select the trading days.
    """
    from trading_app.config import ALL_FILTERS  # canonical filter registry

    frames = []
    for ln in lanes:
        sid = ln["strategy_id"]
        filt_key = ln["filter_type"]
        if filt_key not in ALL_FILTERS:
            print(f"  WARN: filter {filt_key} not in ALL_FILTERS — lane {sid} skipped")
            frames.append(pd.DataFrame(columns=[sid]).rename_axis("trading_day"))
            continue
        filt = ALL_FILTERS[filt_key]

        # daily_features is wide: one row per (trading_day, symbol, orb_minutes)
        # with session-specific columns encoded as orb_<LABEL>_<field>. The
        # matches_row() canonical signature is (row_dict, orb_label), so we
        # pass the full wide row and the lane's session as orb_label.
        joined = con.execute(
            """
            SELECT d.*, o.pnl_r, o.outcome
              FROM daily_features d
              JOIN orb_outcomes o
                ON o.trading_day = d.trading_day
               AND o.symbol = d.symbol
               AND o.orb_minutes = d.orb_minutes
             WHERE d.symbol = ?
               AND d.orb_minutes = ?
               AND o.orb_label = ?
               AND o.rr_target = ?
               AND o.confirm_bars = ?
               AND o.entry_model = ?
               AND o.pnl_r IS NOT NULL
             ORDER BY d.trading_day
            """,
            [
                ln["instrument"],
                ln["orb_minutes"],
                ln["session"],
                ln["rr_target"],
                ln["confirm_bars"],
                ln["entry_model"],
            ],
        ).fetchdf()

        if joined.empty:
            frames.append(pd.DataFrame(columns=[sid]).rename_axis("trading_day"))
            continue

        # Apply the canonical filter row-by-row.
        match_mask = []
        for _, row in joined.iterrows():
            row_dict = row.to_dict()
            try:
                match_mask.append(bool(filt.matches_row(row_dict, ln["session"])))
            except Exception as e:
                print(f"  WARN: matches_row failed on {sid} {row_dict.get('trading_day')}: {e}")
                match_mask.append(False)

        hits = joined[match_mask][["trading_day", "pnl_r"]].set_index("trading_day")
        hits = hits.rename(columns={"pnl_r": sid})
        frames.append(hits)

    wide = pd.concat(frames, axis=1).sort_index()
    return wide


def compute_correlation_matrix(returns_wide: pd.DataFrame) -> pd.DataFrame:
    # Fill NaN with 0 — portfolio accounting: no-trade day = 0 subsystem return
    filled = returns_wide.fillna(0.0)
    return filled.corr(method="pearson")


def effective_bets(corr: pd.DataFrame) -> tuple[float, float]:
    """Carver's diversification multiplier (p170) for equal weights:
        sigma_portfolio = sqrt(w' @ C @ w)
        D = 1 / sigma_portfolio
        N_eff (equal weights) = D^2
    Capped at 2.5 per Carver (crisis-correlation-jump guard)."""
    n = corr.shape[0]
    w = np.full(n, 1.0 / n)
    sigma_p = math.sqrt(float(w @ corr.values @ w))
    div_mult = 1.0 / sigma_p
    n_eff = div_mult**2
    return div_mult, n_eff


def classify_mechanism_cluster(lane: dict) -> str:
    """Hand-grouping per mechanism_priors heuristic."""
    region = lane["region"]
    ff = lane["filter_family"]
    return f"{region}|{ff}"


def main() -> int:
    print("=" * 78)
    print("6-LANE CORRELATION + MECHANISM-CONCENTRATION AUDIT")
    print("=" * 78)
    print()
    print(f"Canonical DB: {GOLD_DB_PATH}")
    print(f"Lane manifest: {LANE_ALLOC_PATH}")
    print()

    lanes = load_lanes()
    print(f"Deployed lanes: {len(lanes)}")
    print()
    print("-" * 78)
    print("DECOMPOSITION (structural concentration flags)")
    print("-" * 78)
    import collections

    inst_counts = collections.Counter(ln["instrument"] for ln in lanes)
    reg_counts = collections.Counter(ln["region"] for ln in lanes)
    ff_counts = collections.Counter(ln["filter_family"] for ln in lanes)
    ft_counts = collections.Counter(ln["filter_type"] for ln in lanes)
    em_counts = collections.Counter(ln["entry_model"] for ln in lanes)
    rr_counts = collections.Counter(ln["rr_target"] for ln in lanes)
    ap_counts = collections.Counter(ln["orb_minutes"] for ln in lanes)

    def dump(label, counts):
        row = ", ".join(f"{k}×{v}" for k, v in sorted(counts.items(), key=lambda x: -x[1]))
        print(f"  {label:20s}: {row}")

    dump("instrument", inst_counts)
    dump("session region", reg_counts)
    dump("filter family", ff_counts)
    dump("filter type", ft_counts)
    dump("entry model", em_counts)
    dump("rr_target", rr_counts)
    dump("orb_minutes", ap_counts)
    print()

    print("Mechanism clusters (region × filter_family):")
    clusters = collections.defaultdict(list)
    for ln in lanes:
        clusters[classify_mechanism_cluster(ln)].append(ln["strategy_id"])
    for cl, members in sorted(clusters.items()):
        print(f"  {cl:30s}  ({len(members)})")
        for m in members:
            print(f"      {m}")
    print()

    with duckdb.connect(str(GOLD_DB_PATH), read_only=True) as con:
        returns_wide = load_trade_returns(con, lanes)

    print("-" * 78)
    print("TRADE-RETURN DATA COVERAGE")
    print("-" * 78)
    for sid in returns_wide.columns:
        n_days = int(returns_wide[sid].notna().sum())
        mean_r = returns_wide[sid].mean()
        std_r = returns_wide[sid].std()
        print(f"  {sid:52s} N={n_days:4d}  mean(R)={mean_r:+.3f}  std(R)={std_r:.3f}")
    # Coverage window
    idx = returns_wide.index
    if len(idx) > 0:
        print(f"  range: {idx.min()} → {idx.max()}  ({len(idx)} unique trading days)")
    print()

    print("-" * 78)
    print("CORRELATION MATRIX (Pearson, NaN-as-0 / portfolio accounting)")
    print("-" * 78)
    corr = compute_correlation_matrix(returns_wide)
    print(corr.round(3).to_string())
    print()

    # Off-diagonal distribution
    vals = corr.values
    tri_mask = np.triu(np.ones_like(vals, dtype=bool), k=1)
    off_diag = vals[tri_mask]
    print(f"  off-diagonal count : {len(off_diag)}")
    print(f"  mean corr          : {off_diag.mean():+.3f}")
    print(f"  median corr        : {np.median(off_diag):+.3f}")
    print(f"  max corr           : {off_diag.max():+.3f}")
    print(f"  min corr           : {off_diag.min():+.3f}")
    print(f"  corr >= 0.30 count : {int((off_diag >= 0.30).sum())}")
    print()

    div_mult, n_eff = effective_bets(corr)
    div_mult_capped = min(div_mult, 2.5)
    print("-" * 78)
    print("DIVERSIFICATION METRICS (Carver Ch 11)")
    print("-" * 78)
    print(f"  portfolio-sigma (equal weights, N={corr.shape[0]}): {1/div_mult:.4f}")
    print(f"  diversification multiplier D = 1/sigma_p         : {div_mult:.3f}")
    print(f"  D capped at 2.5 (crisis-correlation guard)        : {div_mult_capped:.3f}")
    print(f"  effective independent bets  N_eff = D^2           : {n_eff:.2f}")
    print(f"  nominal lanes                                     : {corr.shape[0]}")
    print(f"  fragility ratio (nominal/N_eff)                   : {corr.shape[0]/n_eff:.2f}")
    print()

    # Eigenvalue decomposition → PC1 share (variance concentration)
    eigvals, _ = np.linalg.eigh(corr.values)
    eigvals_desc = sorted(eigvals.tolist(), reverse=True)
    pc1_share = eigvals_desc[0] / sum(eigvals_desc)
    print(f"  eigenvalues (desc)   : {[round(e,3) for e in eigvals_desc]}")
    print(f"  PC1 explained variance share : {pc1_share:.1%}")
    print(f"  (PC1 >= 50% → book is effectively a single-factor bet)")
    print()

    # Verdict
    print("-" * 78)
    print("VERDICT")
    print("-" * 78)
    flags = []
    if inst_counts.most_common(1)[0][1] == len(lanes):
        flags.append(f"INSTRUMENT-CONCENTRATED: {len(lanes)}/{len(lanes)} = {inst_counts.most_common(1)[0][0]}")
    if ft_counts.most_common(1)[0][1] >= 3:
        ft_top = ft_counts.most_common(1)[0]
        flags.append(f"FILTER-CONCENTRATED: {ft_top[1]}/{len(lanes)} share filter {ft_top[0]}")
    if em_counts.most_common(1)[0][1] == len(lanes):
        flags.append(f"ENTRY-MODEL-CONCENTRATED: all {len(lanes)} = {em_counts.most_common(1)[0][0]}")
    if off_diag.mean() >= 0.30:
        flags.append(f"HIGH-MEAN-CORR: mean off-diagonal = {off_diag.mean():+.3f}")
    elif off_diag.mean() >= 0.15:
        flags.append(f"MODERATE-MEAN-CORR: mean off-diagonal = {off_diag.mean():+.3f}")
    if pc1_share >= 0.50:
        flags.append(f"SINGLE-FACTOR: PC1 explains {pc1_share:.1%}")
    elif pc1_share >= 0.35:
        flags.append(f"DOMINANT-FACTOR: PC1 explains {pc1_share:.1%}")
    if n_eff < len(lanes) * 0.5:
        flags.append(f"LOW-N-EFF: effective bets {n_eff:.1f} vs nominal {len(lanes)}")

    if not flags:
        print("  CLEAN: no concentration flags triggered.")
    else:
        print("  Flags (ordered most severe first):")
        for f in flags:
            print(f"    [FLAG] {f}")
    print()

    # ---- FALSIFICATION BATTERY ----
    print("-" * 78)
    print("FALSIFICATION #1 — per-year correlation stability")
    print("-" * 78)
    returns_wide = returns_wide.copy()
    returns_wide.index = pd.to_datetime(returns_wide.index)
    years = sorted({d.year for d in returns_wide.index})
    yearly_rows = []
    for yr in years:
        yr_mask = returns_wide.index.year == yr
        yr_data = returns_wide[yr_mask].fillna(0.0)
        if len(yr_data) < 30:
            continue
        yr_corr = yr_data.corr(method="pearson")
        yr_vals = yr_corr.values[np.triu(np.ones_like(yr_corr.values, dtype=bool), k=1)]
        yr_D = 1.0 / math.sqrt(
            float(np.full(yr_corr.shape[0], 1.0 / yr_corr.shape[0]) @ yr_corr.values @ np.full(yr_corr.shape[0], 1.0 / yr_corr.shape[0]))
        )
        yr_n_eff = yr_D**2
        yr_eigs = sorted(np.linalg.eigh(yr_corr.values)[0].tolist(), reverse=True)
        pc1 = yr_eigs[0] / sum(yr_eigs)
        yearly_rows.append(
            {
                "year": yr,
                "n_days": int(yr_mask.sum()),
                "mean_corr": round(yr_vals.mean(), 3),
                "max_corr": round(yr_vals.max(), 3),
                "n_eff": round(yr_n_eff, 2),
                "pc1_share": round(pc1, 3),
            }
        )
    yr_df = pd.DataFrame(yearly_rows)
    print(yr_df.to_string(index=False))
    print()

    print("-" * 78)
    print("FALSIFICATION #2 — conditional correlation (days where BOTH lanes traded)")
    print("-" * 78)
    cond_rows = []
    cols = list(returns_wide.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            both = returns_wide[[a, b]].dropna()
            if len(both) < 30:
                cond_rows.append({"pair_a": a, "pair_b": b, "n_both_traded": len(both), "cond_corr": None})
            else:
                cond_rows.append(
                    {
                        "pair_a": a,
                        "pair_b": b,
                        "n_both_traded": len(both),
                        "cond_corr": round(both.corr().iloc[0, 1], 3),
                    }
                )
    cond_df = pd.DataFrame(cond_rows)
    cond_valid = cond_df.dropna(subset=["cond_corr"])
    print(f"  pairs with N >= 30 both-traded: {len(cond_valid)} of {len(cond_df)}")
    print(f"  mean conditional corr   : {cond_valid['cond_corr'].mean():+.3f}")
    print(f"  max conditional corr    : {cond_valid['cond_corr'].max():+.3f}")
    print(f"  min conditional corr    : {cond_valid['cond_corr'].min():+.3f}")
    print(f"  cond corr >= 0.30 count : {int((cond_valid['cond_corr'] >= 0.30).sum())}")
    if (cond_valid["cond_corr"] >= 0.30).any():
        top_pairs = cond_valid.nlargest(3, "cond_corr")
        print("  Top 3 correlated pairs (conditional):")
        for _, row in top_pairs.iterrows():
            print(f"    {row['pair_a']} <-> {row['pair_b']}: {row['cond_corr']:+.3f} (N={row['n_both_traded']})")
    print()

    print("-" * 78)
    print("FALSIFICATION #3 — stress-period correlation (high-VIX proxies)")
    print("-" * 78)
    # Crisis months proxy: 2020-03 (COVID), 2022-06 (bear), 2024-08 (vol spike).
    crisis_windows = [
        ("2020-02-15", "2020-05-15", "COVID crash 2020-03"),
        ("2022-01-01", "2022-07-31", "2022 bear H1"),
        ("2024-08-01", "2024-08-31", "2024-08 vol spike"),
    ]
    for start, end, label in crisis_windows:
        win = returns_wide[(returns_wide.index >= start) & (returns_wide.index <= end)]
        if len(win) < 10:
            print(f"  {label:28s} [{start} - {end}] N={len(win):3d}  SKIP (too few days)")
            continue
        win_filled = win.fillna(0.0)
        win_corr = win_filled.corr()
        win_vals = win_corr.values[np.triu(np.ones_like(win_corr.values, dtype=bool), k=1)]
        D_win = 1.0 / math.sqrt(
            float(
                np.full(win_corr.shape[0], 1.0 / win_corr.shape[0])
                @ win_corr.values
                @ np.full(win_corr.shape[0], 1.0 / win_corr.shape[0])
            )
        )
        eigs = sorted(np.linalg.eigh(win_corr.values)[0].tolist(), reverse=True)
        pc1 = eigs[0] / sum(eigs)
        print(
            f"  {label:28s} [{start} → {end}] N={len(win):3d}  "
            f"mean_corr={win_vals.mean():+.3f}  max={win_vals.max():+.3f}  "
            f"N_eff={D_win**2:.2f}  PC1={pc1:.1%}"
        )
    print()

    # Write artefact
    out_dir = PROJECT_ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    corr.round(4).to_csv(out_dir / "6lane_correlation_matrix.csv")
    yr_df.to_csv(out_dir / "6lane_correlation_by_year.csv", index=False)
    cond_df.to_csv(out_dir / "6lane_conditional_correlation_pairs.csv", index=False)
    print(f"  Artefacts -> {out_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
