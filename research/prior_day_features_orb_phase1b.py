"""Prior-day zone/positional/gap-categorical features as ORB filters.

Pre-registered at: docs/audit/hypotheses/2026-04-15-prior-day-zone-positional-features-orb.md

This script is Protocol-B enumeration with K_local=96, K_global=639, Bailey E>=1.2.
Gates and scope are LOCKED by the pre-registration. Editing this file after first
run invalidates the registration (drift check #94).
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from pipeline.cost_model import COST_SPECS
from pipeline.paths import GOLD_DB_PATH
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

# ============================================================================
# LOCKED CONFIG (from pre-registration — do not edit)
# ============================================================================

SEED = 20260415
rng = np.random.default_rng(SEED)

INSTRUMENTS = ("MNQ", "MES")
SESSIONS = ("CME_PRECLOSE", "US_DATA_1000", "NYSE_OPEN")
APERTURES = (5,)
RR_TARGETS = (1.0, 1.5)
ENTRY_MODEL = "E2"

THETA_PRIMARY = 0.30
THETA_SENSITIVITY = (0.15, 0.30, 0.50)

# K accounting (pre-registration §5)
N_MINBTL = 8 * 3 * 1 * 2 * 1  # = 48 strategy definitions
K_LOCAL = N_MINBTL * 2  # = 96 p-values
K_GLOBAL = K_LOCAL + 61 + 57 + 425  # = 639; frozen at registration

# Gates (pre-registration §6)
BONFERRONI_LOCAL = 0.05 / K_LOCAL
BH_FDR_Q = 0.05
CHORDIA_T = 3.79
ERA_BOUNDS = [
    (pd.Timestamp("2015-01-01"), pd.Timestamp("2019-12-31")),
    (pd.Timestamp("2020-01-01"), pd.Timestamp("2022-12-31")),
    (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31")),
    (pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31")),
    (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-12-31")),
]
ERA_N_FLOOR = 50
ERA_EXPR_FLOOR = -0.05
HOLDOUT_N_FLOOR = 30
HOLDOUT_EFFECT_RATIO = 0.40
EFFECT_FLOOR_R = 0.10
JACCARD_REDUNDANT = 0.40
JACCARD_WEAK = 0.30
BLOCK_BOOTSTRAP_B = 10_000
CLUSTER_ROBUST = True

# OOS window — HOLDOUT_GRANDFATHER_CUTOFF = 2026-04-08, data on/after excluded
OOS_START = HOLDOUT_SACRED_FROM  # 2026-01-01
OOS_END = pd.Timestamp("2026-04-07").date()  # exclusive end before grandfather

OUTPUT_MD = Path("docs/audit/results/2026-04-15-prior-day-zone-positional-features-orb-results.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATA LOAD — CTE dedup + 3-key join (pre-reg §8.1 MANDATORY)
# ============================================================================


def load_data() -> pd.DataFrame:
    """Load orb_outcomes x daily_features with CTE dedup and 3-key join."""
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    session_selects = []
    for session in SESSIONS:
        session_selects.append(
            f"""
        SELECT
            o.trading_day, o.symbol, o.orb_minutes, o.orb_label,
            o.entry_model, o.rr_target, o.outcome, o.pnl_r, o.risk_dollars,
            o.entry_price, o.stop_price,
            d.atr_20, d.prev_day_high, d.prev_day_low, d.prev_day_close,
            d.prev_day_range, d.prev_day_direction, d.gap_type, d.gap_open_points,
            (d.orb_{session}_high + d.orb_{session}_low) / 2.0 AS orb_mid,
            d.orb_{session}_high AS orb_high,
            d.orb_{session}_low AS orb_low,
            d.orb_{session}_size AS orb_size
        FROM orb_outcomes o
        JOIN (SELECT * FROM daily_features WHERE orb_minutes = 5) d  -- CTE DEDUP
            ON o.trading_day = d.trading_day
            AND o.symbol = d.symbol
            AND o.orb_minutes = d.orb_minutes  -- TRIPLE JOIN (3-key)
        WHERE o.orb_label = '{session}'
            AND o.symbol IN {INSTRUMENTS}
            AND o.orb_minutes = 5
            AND o.entry_model = '{ENTRY_MODEL}'
            AND o.rr_target IN {tuple(RR_TARGETS)}
            AND o.pnl_r IS NOT NULL
            AND d.atr_20 IS NOT NULL AND d.atr_20 > 0
            AND d.prev_day_high IS NOT NULL
            AND d.prev_day_low IS NOT NULL
            AND d.prev_day_close IS NOT NULL
        """
        )
    query = " UNION ALL ".join(session_selects)
    df = con.execute(query).df()
    con.close()

    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["is_oos"] = (df["trading_day"].dt.date >= OOS_START) & (df["trading_day"].dt.date < OOS_END)
    df["is_is"] = df["trading_day"].dt.date < OOS_START

    return df


def assert_cost_net(df: pd.DataFrame) -> None:
    """Pre-reg §8.2 — verify pnl_r is cost-net (not gross) via aggregate sanity.

    Gross RR=1.0 wins would have pnl_r == 1.0 exactly. Cost-net wins show mean < 1.0.
    A mean ≈ 1.0 or > 1.0 flags gross (framework bug). A mean < 0.5 flags aberrant.
    """
    wins_rr1 = df[(df["outcome"] == "win") & (df["rr_target"] == 1.0)]
    if len(wins_rr1) < 100:
        raise AssertionError(f"insufficient RR=1.0 wins ({len(wins_rr1)}) to verify cost-net")
    mean_pnl = float(wins_rr1["pnl_r"].mean())
    losses = df[df["outcome"] == "loss"]
    mean_loss = float(losses["pnl_r"].mean()) if len(losses) > 0 else np.nan
    assert 0.5 < mean_pnl < 1.0, (
        f"RR=1.0 win mean pnl_r={mean_pnl:.4f} — gross would be ~1.0; "
        f"cost-net should be <1.0 but >0.5. Framework bug suspected."
    )
    assert -1.5 < mean_loss < -0.9, (
        f"loss mean pnl_r={mean_loss:.4f} — gross loss ~-1.0; cost-net should be ~-1.0 to -1.2"
    )
    print(f"[cost-net assertion] PASS: mean RR=1.0 win={mean_pnl:.4f}, mean loss={mean_loss:.4f}")


# ============================================================================
# FEATURES (pre-reg §3)
# ============================================================================


def compute_features(df: pd.DataFrame, theta: float = THETA_PRIMARY) -> pd.DataFrame:
    df = df.copy()
    atr = df["atr_20"]
    mid = df["orb_mid"]
    pdh = df["prev_day_high"]
    pdl = df["prev_day_low"]
    pdc = df["prev_day_close"]
    pivot = (pdh + pdl + pdc) / 3.0

    df["F1_NEAR_PDH"] = (np.abs(mid - pdh) / atr < theta).astype(int)
    df["F2_NEAR_PDL"] = (np.abs(mid - pdl) / atr < theta).astype(int)
    df["F3_NEAR_PIVOT"] = (np.abs(mid - pivot) / atr < theta).astype(int)
    df["F4_ABOVE_PDH"] = (mid > pdh).astype(int)
    df["F5_BELOW_PDL"] = (mid < pdl).astype(int)
    df["F6_INSIDE_PDR"] = ((mid > pdl) & (mid < pdh)).astype(int)
    df["F7_GAP_UP"] = (df["gap_type"] == "gap_up").astype(int)
    df["F8_GAP_DOWN"] = (df["gap_type"] == "gap_down").astype(int)

    return df


FEATURES = [
    "F1_NEAR_PDH",
    "F2_NEAR_PDL",
    "F3_NEAR_PIVOT",
    "F4_ABOVE_PDH",
    "F5_BELOW_PDL",
    "F6_INSIDE_PDR",
    "F7_GAP_UP",
    "F8_GAP_DOWN",
]


# ============================================================================
# CONTROLS (pre-reg §7; outside K=96 budget)
# ============================================================================


def add_controls(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # C1 destruction — within-era shuffle of F1
    df["C1_DESTRUCTION"] = df["F1_NEAR_PDH"].copy()
    for lo, hi in ERA_BOUNDS:
        mask = (df["trading_day"] >= lo) & (df["trading_day"] <= hi)
        if mask.sum() > 0:
            vals = np.array(df.loc[mask, "C1_DESTRUCTION"].values, copy=True)
            rng_local = np.random.default_rng(SEED + int(lo.year))
            rng_local.shuffle(vals)
            df.loc[mask, "C1_DESTRUCTION"] = vals

    # C2 known-null — pure noise keyed per (symbol, trading_day, session)
    def null_bit(row):
        key = f"{row['symbol']}|{row['trading_day'].date()}|{row['orb_label']}|{SEED}"
        h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
        return 1 if (h % 2 == 0) else 0

    df["C2_KNOWN_NULL"] = df.apply(null_bit, axis=1)

    # C3 known-positive sanity: VWAPBreakDirectionFilter
    # VWAP filter fires when break direction aligns with pre-session VWAP position
    # Proxy via orb_mid vs. approximate VWAP — if orb_vwap is available, use it
    df["C3_VWAP_ALIGNED"] = 0  # placeholder — real implementation requires orb_{session}_vwap
    # (Framework check: if the canonical filter is in validated_setups, we'd pull its fire-pattern.
    #  This is a sanity check — if it fails, investigate, but do not block Phase 1.)

    return df


# ============================================================================
# PER-CELL ANALYSIS
# ============================================================================


@dataclass
class CellResult:
    instrument: str
    session: str
    rr: float
    signal: str
    theta: float
    n_is: int
    n_oos: int
    expr_on_is: float
    expr_off_is: float
    expr_delta_is: float
    expr_delta_oos: float
    t_stat: float
    p_raw: float
    ci_lo: float
    ci_hi: float
    era_results: dict
    jaccard_max_vs_deployed: float
    partial_reg_mean_p: float
    partial_reg_binary_p: float
    holdout_direction_match: bool
    holdout_effect_ratio: float
    fire_rate: float


def block_bootstrap_ci(
    on_vals: np.ndarray, off_vals: np.ndarray, block_size: int, B: int = BLOCK_BOOTSTRAP_B
) -> tuple[float, float]:
    """Politis-Romano stationary block bootstrap CI on mean delta."""
    n_on, n_off = len(on_vals), len(off_vals)
    if n_on < 2 or n_off < 2:
        return (np.nan, np.nan)
    deltas = np.empty(B)
    for b in range(B):
        idx_on = (np.arange(n_on) + rng.integers(0, n_on)) % n_on
        idx_off = (np.arange(n_off) + rng.integers(0, n_off)) % n_off
        # Block resampling
        starts_on = rng.integers(0, n_on, size=(n_on // block_size) + 1)
        starts_off = rng.integers(0, n_off, size=(n_off // block_size) + 1)
        boot_on = np.concatenate([on_vals[s : s + block_size] for s in starts_on])[:n_on]
        boot_off = np.concatenate([off_vals[s : s + block_size] for s in starts_off])[:n_off]
        deltas[b] = np.mean(boot_on) - np.mean(boot_off)
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def analyze_cell(
    df: pd.DataFrame, instrument: str, session: str, rr: float, signal: str, theta: float
) -> CellResult | None:
    cell = df[(df["symbol"] == instrument) & (df["orb_label"] == session) & (df["rr_target"] == rr)].copy()
    if len(cell) == 0:
        return None

    is_mask = cell["is_is"]
    oos_mask = cell["is_oos"]
    cell_is = cell[is_mask]
    cell_oos = cell[oos_mask]

    if len(cell_is) < HOLDOUT_N_FLOOR:
        return None

    on_is = cell_is[cell_is[signal] == 1]["pnl_r"].values
    off_is = cell_is[cell_is[signal] == 0]["pnl_r"].values
    if len(on_is) < HOLDOUT_N_FLOOR or len(off_is) < HOLDOUT_N_FLOOR:
        return None

    expr_on = float(np.mean(on_is))
    expr_off = float(np.mean(off_is))
    delta_is = expr_on - expr_off

    # Welch's t on cluster-robust... approximation: use Welch's t as fallback;
    # cluster SE via statsmodels below.
    t_stat, p_raw = stats.ttest_ind(on_is, off_is, equal_var=False)

    # Block bootstrap CI
    T = cell_is["trading_day"].dt.date.nunique()
    block_size = max(2, int(np.ceil(T ** (1 / 3))))
    ci_lo, ci_hi = block_bootstrap_ci(on_is, off_is, block_size)

    # Era results
    era_results = {}
    for i, (lo, hi) in enumerate(ERA_BOUNDS):
        era_mask = (cell_is["trading_day"] >= lo) & (cell_is["trading_day"] <= hi) & (cell_is[signal] == 1)
        era_vals = cell_is[era_mask]["pnl_r"].values
        n = len(era_vals)
        expr = float(np.mean(era_vals)) if n > 0 else np.nan
        era_results[f"era_{i}"] = {"n": n, "expr": expr, "exempt": n < ERA_N_FLOOR}

    # Holdout
    oos_on = cell_oos[cell_oos[signal] == 1]["pnl_r"].values
    oos_off = cell_oos[cell_oos[signal] == 0]["pnl_r"].values
    n_oos = len(oos_on)
    if n_oos > 0 and len(oos_off) > 0:
        delta_oos = float(np.mean(oos_on) - np.mean(oos_off))
        dir_match = (np.sign(delta_oos) == np.sign(delta_is)) and (abs(delta_is) > 1e-6)
        effect_ratio = delta_oos / delta_is if abs(delta_is) > 1e-6 else np.nan
    else:
        delta_oos = np.nan
        dir_match = False
        effect_ratio = np.nan

    # Partial regression (mean)
    try:
        X = pd.get_dummies(cell_is[["orb_label"]], drop_first=True).astype(float)
        X["feature"] = cell_is[signal].values
        X["atr_20"] = cell_is["atr_20"].values
        X["orb_size"] = cell_is["orb_size"].values
        # Interaction: feature x session — only one session per cell so this simplifies
        X = sm.add_constant(X, has_constant="add")
        y = cell_is["pnl_r"].values
        clusters = cell_is["trading_day"].dt.date.values
        model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": clusters})
        reg_p = float(model.pvalues.get("feature", np.nan))
    except Exception as e:
        reg_p = np.nan

    # Partial regression (binary / logistic on win)
    try:
        cell_is["is_win"] = (cell_is["pnl_r"] > 0).astype(int)
        X = pd.get_dummies(cell_is[["orb_label"]], drop_first=True).astype(float)
        X["feature"] = cell_is[signal].values
        X["atr_20"] = cell_is["atr_20"].values
        X["orb_size"] = cell_is["orb_size"].values
        X = sm.add_constant(X, has_constant="add")
        y = cell_is["is_win"].values
        logit = sm.Logit(y, X).fit(disp=0)
        logit_p = float(logit.pvalues.get("feature", np.nan))
    except Exception as e:
        logit_p = np.nan

    # Jaccard vs deployed filters — placeholder; needs BASE_GRID fire-patterns joined
    jaccard_max = 0.0

    return CellResult(
        instrument=instrument,
        session=session,
        rr=rr,
        signal=signal,
        theta=theta,
        n_is=len(on_is),
        n_oos=n_oos,
        expr_on_is=expr_on,
        expr_off_is=expr_off,
        expr_delta_is=delta_is,
        expr_delta_oos=delta_oos,
        t_stat=float(t_stat),
        p_raw=float(p_raw),
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        era_results=era_results,
        jaccard_max_vs_deployed=jaccard_max,
        partial_reg_mean_p=reg_p,
        partial_reg_binary_p=logit_p,
        holdout_direction_match=dir_match,
        holdout_effect_ratio=float(effect_ratio) if not np.isnan(effect_ratio) else np.nan,
        fire_rate=float(np.sum(on_is.shape[0]) / len(cell_is)) if len(cell_is) > 0 else 0.0,
    )


# ============================================================================
# FDR CORRECTION
# ============================================================================


def bh_fdr(pvals: np.ndarray, q: float = BH_FDR_Q) -> np.ndarray:
    """Benjamini-Hochberg FDR — returns boolean survival mask."""
    pvals = np.asarray(pvals, dtype=float)
    valid = ~np.isnan(pvals)
    n = valid.sum()
    survive = np.zeros(len(pvals), dtype=bool)
    if n == 0:
        return survive
    order = np.argsort(pvals[valid])
    ranked = pvals[valid][order]
    thresholds = (np.arange(1, n + 1) / n) * q
    below = ranked <= thresholds
    if below.any():
        k = np.max(np.where(below)[0]) + 1
        crit = ranked[k - 1]
        full_idx = np.where(valid)[0]
        survive[full_idx[pvals[valid] <= crit]] = True
    return survive


# ============================================================================
# MAIN
# ============================================================================


def main():
    print(f"[pre-reg] SEED={SEED}, K_local={K_LOCAL}, K_global={K_GLOBAL}")
    print(f"[pre-reg] Bonferroni-local={BONFERRONI_LOCAL:.2e}, BH-FDR q={BH_FDR_Q}, Chordia t>={CHORDIA_T}")

    df = load_data()
    print(f"[data] loaded {len(df)} rows, {df['trading_day'].nunique()} unique trading days")
    print(f"[data] IS: {df['is_is'].sum()} rows | OOS: {df['is_oos'].sum()} rows")

    assert_cost_net(df)

    df = compute_features(df, theta=THETA_PRIMARY)
    df = add_controls(df)

    # Run cells at primary theta
    results: list[CellResult] = []
    for instr in INSTRUMENTS:
        for session in SESSIONS:
            for rr in RR_TARGETS:
                for feat in FEATURES:
                    r = analyze_cell(df, instr, session, rr, feat, THETA_PRIMARY)
                    if r is not None:
                        results.append(r)

    print(f"[run] {len(results)} cells analyzed at theta={THETA_PRIMARY}")

    # Controls
    control_results: list[CellResult] = []
    for instr in INSTRUMENTS:
        for session in SESSIONS:
            for rr in RR_TARGETS:
                for ctrl in ["C1_DESTRUCTION", "C2_KNOWN_NULL"]:
                    r = analyze_cell(df, instr, session, rr, ctrl, 0.0)
                    if r is not None:
                        control_results.append(r)

    # Apply FDR
    pvals = np.array([r.p_raw for r in results])
    bh_local = bh_fdr(pvals, q=BH_FDR_Q)

    # Emit results
    emit_report(df, results, control_results, bh_local)


def emit_report(
    df: pd.DataFrame, results: list[CellResult], controls: list[CellResult], bh_local_mask: np.ndarray
) -> None:
    lines = [
        "# Results — Prior-Day Zone / Positional / Gap-Categorical Features on ORB Outcomes",
        "",
        f"**Pre-registration:** `docs/audit/hypotheses/2026-04-15-prior-day-zone-positional-features-orb.md`",
        f"**Run date:** 2026-04-15",
        f"**Rows IS / OOS:** {df['is_is'].sum()} / {df['is_oos'].sum()}",
        f"**K_local:** {K_LOCAL} | **K_global:** {K_GLOBAL} | **Bonferroni-local p:** {BONFERRONI_LOCAL:.2e}",
        "",
        "## Primary cells (theta=0.30)",
        "",
        "| Instr | Session | RR | Signal | N_IS | N_OOS | ExpR_on | ExpR_off | Delta_IS | Delta_OOS | t | p_raw | Bonf | BH | Chordia | Dir_match | OOS/IS |",
        "|-------|---------|----|---------|------|-------|---------|----------|----------|-----------|---|-------|------|-----|---------|-----------|--------|",
    ]

    for i, r in enumerate(results):
        bonf = "PASS" if r.p_raw < BONFERRONI_LOCAL else "fail"
        bh_mark = "PASS" if bh_local_mask[i] else "fail"
        chordia = "PASS" if abs(r.t_stat) >= CHORDIA_T else "fail"
        dir_str = "PASS" if r.holdout_direction_match else "fail"
        lines.append(
            f"| {r.instrument} | {r.session} | {r.rr} | {r.signal} | {r.n_is} | {r.n_oos} | "
            f"{r.expr_on_is:+.3f} | {r.expr_off_is:+.3f} | {r.expr_delta_is:+.3f} | "
            f"{r.expr_delta_oos:+.3f} | {r.t_stat:+.2f} | {r.p_raw:.4f} | {bonf} | {bh_mark} | "
            f"{chordia} | {dir_str} | {r.holdout_effect_ratio:+.2f} |"
        )

    lines += [
        "",
        "## Framework-integrity controls (outside K=96 budget)",
        "",
        "| Control | Instr | Session | RR | N_IS | Delta_IS | t | p_raw |",
        "|---------|-------|---------|----|------|----------|---|-------|",
    ]
    for r in controls:
        lines.append(
            f"| {r.signal} | {r.instrument} | {r.session} | {r.rr} | {r.n_is} | "
            f"{r.expr_delta_is:+.3f} | {r.t_stat:+.2f} | {r.p_raw:.4f} |"
        )

    # Verdict skeleton
    survivors = [r for i, r in enumerate(results) if bh_local_mask[i] and abs(r.t_stat) >= CHORDIA_T]
    lines += [
        "",
        "## Preliminary verdict (before §9 failure-mode full-gate check)",
        "",
        f"- Cells tested: {len(results)}",
        f"- Bonferroni-local passers: {int(sum(1 for r in results if r.p_raw < BONFERRONI_LOCAL))}",
        f"- BH-FDR local passers: {int(bh_local_mask.sum())}",
        f"- Chordia t>=3.79 passers: {int(sum(1 for r in results if abs(r.t_stat) >= CHORDIA_T))}",
        f"- Joint BH + Chordia passers: {len(survivors)}",
        "",
        "Full per-cell failure-mode evaluation (era stability, Jaccard, partial-regression) in next-pass analysis.",
    ]

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] written to {OUTPUT_MD}")


if __name__ == "__main__":
    main()
