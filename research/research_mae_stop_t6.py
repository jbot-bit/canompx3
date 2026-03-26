"""T6 Null Floor Bootstrap — MAE Tight Stop (S0.75) Aggregate Test.

Tests whether the ExpR improvement from apply_tight_stop(stop_multiplier=0.75)
is statistically significant vs. permutation null.

Population: E2 RR2.0 G4+ (all apertures, all sessions, scratches excluded).
Method: Shuffle pnl_r+outcome 1000x, re-apply tight stop, compare deltas.
P-value: Phipson & Smyth (2010) — (exceeding + 1) / (perms + 1).

Uses EXISTING apply_tight_stop() from trading_app.config (same function as
54,810 experimental + 279 validated S0.75 strategies).

Classification: ARITHMETIC_ONLY (loss-size reduction, same family as G-filters).

@research-source: quant-audit-protocol.md T6
@entry-models: E2
@revalidated-for: E2 event-based (2026-03-27)
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.cost_model import COST_SPECS
from pipeline.paths import GOLD_DB_PATH
from trading_app.config import apply_tight_stop

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────
INSTRUMENTS = ["MGC", "MNQ", "MES"]
ENTRY_MODEL = "E2"
RR_TARGET = 2.0
G4_THRESHOLD = 4.0  # |entry - stop| >= 4.0 points
STOP_MULT = 0.75
N_PERMS = 1000
SEED = 42
SL_LEVELS_TESTED = 5  # 0.60, 0.65, 0.70, 0.75, 0.80 — for full-search K
APERTURES = [5, 15, 30]


def load_population(con: duckdb.DuckDBPyConnection, instrument: str) -> list[dict]:
    """Load all E2 RR2.0 G4+ trades as list of dicts for apply_tight_stop."""
    rows = con.execute(
        """
        SELECT mae_r, pnl_r, entry_price, stop_price, risk_dollars,
               outcome, orb_minutes
        FROM orb_outcomes
        WHERE symbol = ?
            AND entry_model = ?
            AND rr_target = ?
            AND pnl_r IS NOT NULL
            AND mae_r IS NOT NULL
            AND entry_price IS NOT NULL
            AND ABS(entry_price - stop_price) >= ?
        ORDER BY trading_day, orb_label, orb_minutes
        """,
        [instrument, ENTRY_MODEL, RR_TARGET, G4_THRESHOLD],
    ).fetchall()

    cols = ["mae_r", "pnl_r", "entry_price", "stop_price", "risk_dollars",
            "outcome", "orb_minutes"]
    return [dict(zip(cols, row)) for row in rows]


def compute_expr(outcomes: list[dict]) -> float:
    """Mean pnl_r across outcomes."""
    pnl_rs = [o["pnl_r"] for o in outcomes]
    return np.mean(pnl_rs)


def run_t6(
    trades: list[dict],
    cost_spec,
    n_perms: int = N_PERMS,
    seed: int = SEED,
) -> dict:
    """Run T6 permutation test on a trade population.

    Returns dict with observed_delta, null_mean, null_std, null_p95, p_value,
    and the full null distribution for diagnostics.
    """
    n = len(trades)
    if n == 0:
        return {"error": "empty population"}

    # Observed
    base_expr = compute_expr(trades)
    tight_outcomes = apply_tight_stop(trades, STOP_MULT, cost_spec)
    tight_expr = compute_expr(tight_outcomes)
    observed_delta = tight_expr - base_expr

    # Extract arrays for shuffling
    pnl_arr = np.array([t["pnl_r"] for t in trades])
    outcome_arr = np.array([t["outcome"] for t in trades])

    # Permutation loop
    rng = np.random.default_rng(seed)
    null_deltas = np.empty(n_perms)
    exceeding = 0

    for p in range(n_perms):
        # Shuffle pnl_r and outcome together (preserve their association)
        idx = rng.permutation(n)
        shuffled_pnl = pnl_arr[idx]
        shuffled_outcome = outcome_arr[idx]

        # Build shuffled trade list (mae_r/entry/stop/risk_d stay fixed)
        shuffled_trades = []
        for i, t in enumerate(trades):
            st = dict(t)
            st["pnl_r"] = float(shuffled_pnl[i])
            st["outcome"] = shuffled_outcome[i]
            shuffled_trades.append(st)

        # Apply tight stop to shuffled
        shuffled_tight = apply_tight_stop(shuffled_trades, STOP_MULT, cost_spec)

        perm_base = np.mean(shuffled_pnl)
        perm_tight = compute_expr(shuffled_tight)
        perm_delta = perm_tight - perm_base
        null_deltas[p] = perm_delta

        if perm_delta >= observed_delta:
            exceeding += 1

    # Phipson & Smyth p-value
    p_value = (exceeding + 1) / (n_perms + 1)

    return {
        "n": n,
        "base_expr": base_expr,
        "tight_expr": tight_expr,
        "observed_delta": observed_delta,
        "null_mean": float(np.mean(null_deltas)),
        "null_std": float(np.std(null_deltas)),
        "null_p95": float(np.percentile(null_deltas, 95)),
        "null_max": float(np.max(null_deltas)),
        "p_value": p_value,
        "exceeding": exceeding,
        "null_deltas": null_deltas,
    }


def run_friction_adjusted_t6(
    trades: list[dict],
    cost_spec,
    n_perms: int = N_PERMS,
    seed: int = SEED,
) -> dict:
    """Sensitivity check: friction-adjusted kill pnl_r instead of raw -0.75.

    When killed at 0.75R raw, actual loss = -(0.75 * risk_pts * pv + friction) / risk_d,
    which is MORE negative than -0.75.
    """
    n = len(trades)
    pv = cost_spec.point_value
    friction = cost_spec.total_friction

    def apply_friction_stop(trade_list: list[dict]) -> list[dict]:
        adjusted = []
        for t in trade_list:
            mae_r = t.get("mae_r")
            entry = t.get("entry_price")
            stop = t.get("stop_price")
            if mae_r is None or entry is None or stop is None:
                adjusted.append(t)
                continue
            risk_pts = abs(entry - stop)
            if risk_pts <= 0:
                adjusted.append(t)
                continue
            raw_risk_d = risk_pts * pv
            risk_d = raw_risk_d + friction
            max_adv_pts = mae_r * risk_d / pv
            if max_adv_pts >= STOP_MULT * risk_pts:
                new_t = dict(t)
                # Friction-adjusted: you still pay full friction on early exit
                loss_d = -(STOP_MULT * risk_pts * pv + friction)
                new_t["pnl_r"] = round(loss_d / risk_d, 4)
                if new_t.get("outcome") != "loss":
                    new_t["outcome"] = "loss"
                adjusted.append(new_t)
            else:
                adjusted.append(t)
        return adjusted

    base_expr = compute_expr(trades)
    tight = apply_friction_stop(trades)
    tight_expr = compute_expr(tight)
    observed_delta = tight_expr - base_expr

    pnl_arr = np.array([t["pnl_r"] for t in trades])
    outcome_arr = np.array([t["outcome"] for t in trades])

    rng = np.random.default_rng(seed)
    exceeding = 0

    for _ in range(n_perms):
        idx = rng.permutation(n)
        shuffled_trades = []
        for i, t in enumerate(trades):
            st = dict(t)
            st["pnl_r"] = float(pnl_arr[idx[i]])
            st["outcome"] = outcome_arr[idx[i]]
            shuffled_trades.append(st)

        shuffled_tight = apply_friction_stop(shuffled_trades)
        perm_delta = compute_expr(shuffled_tight) - np.mean(pnl_arr[idx])
        if perm_delta >= observed_delta:
            exceeding += 1

    return {
        "observed_delta": observed_delta,
        "p_value": (exceeding + 1) / (n_perms + 1),
    }


def benjamini_hochberg(p_values: list[tuple[str, float]], k_total: int) -> dict:
    """BH FDR correction. k_total allows inflating m for full-search accountability."""
    n = len(p_values)
    m = max(k_total, n)  # fail-closed: m >= n
    sorted_pv = sorted(p_values, key=lambda x: x[1])

    results = {}
    prev_adj = 1.0
    for rank_idx in range(n - 1, -1, -1):
        name, raw_p = sorted_pv[rank_idx]
        rank = rank_idx + 1
        adj_p = min(prev_adj, raw_p * m / rank)
        adj_p = min(adj_p, 1.0)
        prev_adj = adj_p
        results[name] = {
            "raw_p": raw_p,
            "adjusted_p": adj_p,
            "fdr_significant": adj_p < 0.05,
            "rank": rank,
        }
    return results


def main() -> None:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    print("=" * 60)
    print("T6 NULL FLOOR -- MAE TIGHT STOP (S0.75) AGGREGATE")
    print("=" * 60)
    print(f"Population: {ENTRY_MODEL} RR{RR_TARGET} G4+ (all apertures, all sessions)")
    print(f"Function:   trading_app.config.apply_tight_stop (production code)")
    print(f"Convention: pnl_r = -stop_multiplier (raw R, matches 279 validated)")
    print(f"Perms:      {N_PERMS} (Phipson & Smyth correction)")
    print(f"Seed:       {SEED}")
    print()

    # ── Primary test ─────────────────────────────────────────────
    results = {}
    for instrument in INSTRUMENTS:
        cost_spec = COST_SPECS[instrument]
        trades = load_population(con, instrument)
        print(f"Running {instrument} (n={len(trades):,})...", flush=True)
        r = run_t6(trades, cost_spec)
        results[instrument] = r

    # ── BH FDR correction ────────────────────────────────────────
    raw_ps = [(inst, results[inst]["p_value"]) for inst in INSTRUMENTS]
    bh_k3 = benjamini_hochberg(raw_ps, k_total=3)
    bh_k15 = benjamini_hochberg(raw_ps, k_total=SL_LEVELS_TESTED * len(INSTRUMENTS))

    # ── Primary results ──────────────────────────────────────────
    print()
    header = f"{'INSTRUMENT':10s} {'N':>8s} {'OBS_DELTA':>10s} {'NULL_MEAN':>10s} {'NULL_P95':>10s} {'P_VALUE':>8s} {'BH(K=3)':>8s} {'BH(K=15)':>9s}"
    print(header)
    print("-" * len(header))

    for inst in INSTRUMENTS:
        r = results[inst]
        print(
            f"{inst:10s} {r['n']:>8,d} {r['observed_delta']:>+10.4f} "
            f"{r['null_mean']:>+10.4f} {r['null_p95']:>+10.4f} "
            f"{r['p_value']:>8.4f} {bh_k3[inst]['adjusted_p']:>8.4f} "
            f"{bh_k15[inst]['adjusted_p']:>9.4f}"
        )

    # ── Per-aperture sensitivity ─────────────────────────────────
    print()
    print("SENSITIVITY — Per-Aperture:")
    for instrument in INSTRUMENTS:
        cost_spec = COST_SPECS[instrument]
        all_trades = load_population(con, instrument)
        parts = []
        for ap in APERTURES:
            ap_trades = [t for t in all_trades if t["orb_minutes"] == ap]
            if len(ap_trades) < 30:
                parts.append(f"O{ap}m: n={len(ap_trades)} (SKIP)")
                continue
            r = run_t6(ap_trades, cost_spec, n_perms=500, seed=SEED)
            parts.append(f"O{ap}m: n={r['n']:,} delta={r['observed_delta']:+.4f} p={r['p_value']:.4f}")
        print(f"  {instrument}: {' | '.join(parts)}")

    # ── Friction-adjusted sensitivity ────────────────────────────
    print()
    print("SENSITIVITY — Friction-Adjusted Kill pnl_r:")
    for instrument in INSTRUMENTS:
        cost_spec = COST_SPECS[instrument]
        trades = load_population(con, instrument)
        r = run_friction_adjusted_t6(trades, cost_spec, n_perms=500, seed=SEED)
        print(f"  {instrument}: delta={r['observed_delta']:+.4f} p={r['p_value']:.4f}")

    # ── Decision ─────────────────────────────────────────────────
    print()
    all_pass_k3 = all(bh_k3[inst]["fdr_significant"] for inst in INSTRUMENTS)
    all_pass_k15 = all(bh_k15[inst]["fdr_significant"] for inst in INSTRUMENTS)

    if all_pass_k3:
        verdict = "BEATS_NULL"
        if all(results[inst]["p_value"] < 0.002 for inst in INSTRUMENTS):
            verdict += " (TRIVIALLY — all p < 0.002)"
    elif any(bh_k3[inst]["fdr_significant"] for inst in INSTRUMENTS):
        passing = [inst for inst in INSTRUMENTS if bh_k3[inst]["fdr_significant"]]
        verdict = f"PARTIAL ({', '.join(passing)} pass, others fail)"
    else:
        verdict = "NO_EDGE — investigate for bugs (30/30 years positive, p>0.05 unexpected)"

    print(f"DECISION:        {verdict}")
    print(f"CLASSIFICATION:  ARITHMETIC_ONLY (loss-size reduction)")
    print(f"BH K=3 (honest): {'ALL PASS' if all_pass_k3 else 'PARTIAL/FAIL'}")
    print(f"BH K=15 (full):  {'ALL PASS' if all_pass_k15 else 'PARTIAL/FAIL'}")
    print("=" * 60)

    con.close()


if __name__ == "__main__":
    main()
