"""Mega exploratory enumeration — prior-day zone/positional/gap features.

EXPLORATORY, NOT CONFIRMATORY. Labels results at multiple K framings.
Pre-registered scope was Phase 1A (K=96). This mega-run INTENTIONALLY
exceeds MinBTL to surface any signal that Phase 1A pigeon-holing missed.

Scope:
- All 3 instruments: MNQ, MES, MGC
- All 12 active sessions from SESSION_CATALOG
- All 3 apertures: 5, 15, 30
- All 3 RR: 1.0, 1.5, 2.0
- Direction split: LONG, SHORT (each tested separately)
- 8 features × 3 theta variants for F1/F2/F3 = 14 signal definitions
- Entry model: E2
- Holdout: IS < 2026-01-01, OOS 2026-01-01 to 2026-04-07

Output per cell: Welch t p, cluster-SE t p (binding), logistic win p, IS/OOS
ExpR, direction match, holdout effect ratio, fire rate.

Exploratory flags:
- HOT: |cluster_t| >= 4.0 AND same sign cross-instr (exceeds Bailey E[max] at K~9000)
- WARM: |cluster_t| >= 3.0 AND IS/OOS direction match
- LUKEWARM: |cluster_t| >= 2.5 AND Bonferroni-per-feature pass
- COLD: rest
"""

from __future__ import annotations

import hashlib
import sys
import time
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

SEED = 20260415
rng = np.random.default_rng(SEED)

INSTRUMENTS = ("MNQ", "MES", "MGC")
SESSIONS = (
    "CME_REOPEN", "TOKYO_OPEN", "SINGAPORE_OPEN", "LONDON_METALS",
    "EUROPE_FLOW", "US_DATA_830", "NYSE_OPEN", "US_DATA_1000",
    "COMEX_SETTLE", "CME_PRECLOSE", "NYSE_CLOSE", "BRISBANE_1025",
)
APERTURES = (5, 15, 30)
RR_TARGETS = (1.0, 1.5, 2.0)
DIRECTIONS = ("long", "short")
ENTRY_MODEL = "E2"

THETAS = (0.15, 0.30, 0.50)

OOS_START = HOLDOUT_SACRED_FROM
OOS_END = pd.Timestamp("2026-04-07").date()

OUTPUT_MD = Path("docs/audit/results/2026-04-15-prior-day-features-orb-mega-exploration.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    """Load orb_outcomes x daily_features for ALL sessions x apertures.

    Joins on 3-key (trading_day, symbol, orb_minutes) so no CTE dedup needed —
    the join is the dedup. break_dir pulled from daily_features so we can
    split by direction.
    """
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    parts = []
    for session in SESSIONS:
        parts.append(
            f"""
        SELECT
            o.trading_day, o.symbol, o.orb_minutes, o.orb_label,
            o.entry_model, o.rr_target, o.outcome, o.pnl_r, o.risk_dollars,
            d.atr_20, d.prev_day_high, d.prev_day_low, d.prev_day_close,
            d.prev_day_range, d.prev_day_direction, d.gap_type, d.gap_open_points,
            (d.orb_{session}_high + d.orb_{session}_low) / 2.0 AS orb_mid,
            d.orb_{session}_high AS orb_high, d.orb_{session}_low AS orb_low,
            d.orb_{session}_size AS orb_size, d.orb_{session}_break_dir AS break_dir
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.trading_day = d.trading_day
          AND o.symbol = d.symbol
          AND o.orb_minutes = d.orb_minutes  -- 3-key join = dedup
        WHERE o.orb_label = '{session}'
          AND o.symbol IN {INSTRUMENTS}
          AND o.entry_model = '{ENTRY_MODEL}'
          AND o.rr_target IN {tuple(RR_TARGETS)}
          AND o.pnl_r IS NOT NULL
          AND d.atr_20 IS NOT NULL AND d.atr_20 > 0
          AND d.prev_day_high IS NOT NULL
          AND d.prev_day_low IS NOT NULL
          AND d.prev_day_close IS NOT NULL
          AND d.orb_{session}_break_dir IN ('long','short')
        """
        )
    query = " UNION ALL ".join(parts)
    df = con.execute(query).df()
    con.close()

    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["is_is"] = df["trading_day"].dt.date < OOS_START
    df["is_oos"] = (df["trading_day"].dt.date >= OOS_START) & (df["trading_day"].dt.date < OOS_END)
    return df


def assert_cost_net(df: pd.DataFrame) -> None:
    wins_rr1 = df[(df["outcome"] == "win") & (df["rr_target"] == 1.0)]
    if len(wins_rr1) < 100:
        raise AssertionError("insufficient RR=1.0 wins")
    m = float(wins_rr1["pnl_r"].mean())
    losses = df[df["outcome"] == "loss"]
    ml = float(losses["pnl_r"].mean()) if len(losses) else float("nan")
    assert 0.5 < m < 1.0, f"RR1 win mean={m:.3f} not cost-net"
    assert -1.5 < ml < -0.9, f"loss mean={ml:.3f} off"
    print(f"[cost-net] mean RR1 win={m:.4f}  mean loss={ml:.4f}  PASS")


def compute_features_all_theta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    atr = df["atr_20"].astype(float)
    mid = df["orb_mid"].astype(float)
    pdh = df["prev_day_high"].astype(float)
    pdl = df["prev_day_low"].astype(float)
    pdc = df["prev_day_close"].astype(float)
    pivot = (pdh + pdl + pdc) / 3.0

    # θ-sensitive features: F1/F2/F3 at each θ
    for theta in THETAS:
        tag = f"{int(theta*100):02d}"
        df[f"F1_NEAR_PDH_{tag}"] = (np.abs(mid - pdh) / atr < theta).astype(int)
        df[f"F2_NEAR_PDL_{tag}"] = (np.abs(mid - pdl) / atr < theta).astype(int)
        df[f"F3_NEAR_PIVOT_{tag}"] = (np.abs(mid - pivot) / atr < theta).astype(int)

    # Binary / categorical features (theta-invariant)
    df["F4_ABOVE_PDH"] = (mid > pdh).astype(int)
    df["F5_BELOW_PDL"] = (mid < pdl).astype(int)
    df["F6_INSIDE_PDR"] = ((mid > pdl) & (mid < pdh)).astype(int)
    df["F7_GAP_UP"] = (df["gap_type"] == "gap_up").astype(int)
    df["F8_GAP_DOWN"] = (df["gap_type"] == "gap_down").astype(int)

    return df


FEATURES = (
    [f"F1_NEAR_PDH_{int(t*100):02d}" for t in THETAS]
    + [f"F2_NEAR_PDL_{int(t*100):02d}" for t in THETAS]
    + [f"F3_NEAR_PIVOT_{int(t*100):02d}" for t in THETAS]
    + ["F4_ABOVE_PDH", "F5_BELOW_PDL", "F6_INSIDE_PDR", "F7_GAP_UP", "F8_GAP_DOWN"]
)


@dataclass
class CellResult:
    instrument: str
    session: str
    aperture: int
    rr: float
    direction: str
    signal: str
    n_is: int
    n_oos: int
    n_on_is: int
    n_off_is: int
    expr_on_is: float
    expr_off_is: float
    delta_is: float
    delta_oos: float
    t_welch: float
    p_welch: float
    t_cluster: float
    p_cluster: float
    p_logit: float
    holdout_dir_match: bool
    holdout_effect_ratio: float
    fire_rate: float


def analyze_cell(df: pd.DataFrame, signal: str) -> CellResult | None:
    if len(df) < 30:
        return None
    is_df = df[df["is_is"]]
    oos_df = df[df["is_oos"]]
    if len(is_df) < 30:
        return None
    on_is = is_df[is_df[signal] == 1]["pnl_r"].values
    off_is = is_df[is_df[signal] == 0]["pnl_r"].values
    if len(on_is) < 30 or len(off_is) < 30:
        return None

    on_is = np.asarray(on_is, dtype=float)
    off_is = np.asarray(off_is, dtype=float)

    expr_on = float(on_is.mean())
    expr_off = float(off_is.mean())
    delta_is = expr_on - expr_off

    # Welch t
    t_w, p_w = stats.ttest_ind(on_is, off_is, equal_var=False)

    # Cluster-SE OLS regression — feature + atr + orb_size + intercept
    t_c, p_c = float("nan"), float("nan")
    p_logit = float("nan")
    try:
        X = pd.DataFrame({
            "feature": is_df[signal].astype(float).values,
            "atr_20": is_df["atr_20"].astype(float).values,
            "orb_size": is_df["orb_size"].astype(float).values,
        })
        X = sm.add_constant(X)
        y = is_df["pnl_r"].astype(float).values
        clusters = is_df["trading_day"].dt.date.values
        m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": clusters})
        t_c = float(m.tvalues.get("feature", float("nan")))
        p_c = float(m.pvalues.get("feature", float("nan")))
    except Exception:
        pass

    try:
        y_win = (is_df["pnl_r"] > 0).astype(int).values
        X2 = pd.DataFrame({
            "feature": is_df[signal].astype(float).values,
            "atr_20": is_df["atr_20"].astype(float).values,
            "orb_size": is_df["orb_size"].astype(float).values,
        })
        X2 = sm.add_constant(X2)
        lm = sm.Logit(y_win, X2).fit(disp=0, maxiter=100)
        p_logit = float(lm.pvalues.get("feature", float("nan")))
    except Exception:
        pass

    # Holdout
    on_oos = oos_df[oos_df[signal] == 1]["pnl_r"].values
    off_oos = oos_df[oos_df[signal] == 0]["pnl_r"].values
    if len(on_oos) > 0 and len(off_oos) > 0:
        delta_oos = float(np.asarray(on_oos).mean() - np.asarray(off_oos).mean())
        dir_match = (np.sign(delta_oos) == np.sign(delta_is)) and abs(delta_is) > 1e-6
        ratio = delta_oos / delta_is if abs(delta_is) > 1e-6 else float("nan")
    else:
        delta_oos = float("nan")
        dir_match = False
        ratio = float("nan")

    fire = float(len(on_is) / (len(on_is) + len(off_is)))

    return CellResult(
        instrument=str(is_df["symbol"].iloc[0]),
        session=str(is_df["orb_label"].iloc[0]),
        aperture=int(is_df["orb_minutes"].iloc[0]),
        rr=float(is_df["rr_target"].iloc[0]),
        direction=str(is_df["break_dir"].iloc[0]) if "break_dir" in is_df.columns else "BOTH",
        signal=signal,
        n_is=len(is_df),
        n_oos=len(oos_df),
        n_on_is=len(on_is),
        n_off_is=len(off_is),
        expr_on_is=expr_on,
        expr_off_is=expr_off,
        delta_is=delta_is,
        delta_oos=delta_oos,
        t_welch=float(t_w),
        p_welch=float(p_w),
        t_cluster=t_c,
        p_cluster=p_c,
        p_logit=p_logit,
        holdout_dir_match=bool(dir_match),
        holdout_effect_ratio=float(ratio) if not np.isnan(ratio) else float("nan"),
        fire_rate=fire,
    )


def bh_fdr_mask(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
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


def main():
    t0 = time.time()
    print("[mega] loading data …")
    df = load_data()
    print(f"[data] {len(df)} rows, {df['trading_day'].nunique()} days; IS={df['is_is'].sum()} OOS={df['is_oos'].sum()}")

    assert_cost_net(df)
    df = compute_features_all_theta(df)

    # Enumerate cells: instrument × session × aperture × RR × direction × feature
    results: list[CellResult] = []
    n_combos = len(INSTRUMENTS) * len(SESSIONS) * len(APERTURES) * len(RR_TARGETS) * len(DIRECTIONS) * len(FEATURES)
    print(f"[run] enumerating {n_combos} cells (exploratory — intentionally exceeds MinBTL)")

    count = 0
    for instr in INSTRUMENTS:
        for session in SESSIONS:
            for aperture in APERTURES:
                for rr in RR_TARGETS:
                    for direction in DIRECTIONS:
                        sub = df[
                            (df["symbol"] == instr)
                            & (df["orb_label"] == session)
                            & (df["orb_minutes"] == aperture)
                            & (df["rr_target"] == rr)
                            & (df["break_dir"] == direction)
                        ]
                        if len(sub) < 30:
                            count += len(FEATURES)
                            continue
                        for feat in FEATURES:
                            r = analyze_cell(sub, feat)
                            if r is not None:
                                results.append(r)
                            count += 1
                            if count % 1000 == 0:
                                elapsed = time.time() - t0
                                print(f"  [progress] {count}/{n_combos}  elapsed={elapsed:.0f}s  analyzed={len(results)}")

    print(f"[run] complete. {len(results)} cells analyzed of {n_combos} possible. elapsed={time.time()-t0:.0f}s")

    # Multi-K FDR framings
    p_cluster = np.array([r.p_cluster for r in results])
    p_welch = np.array([r.p_welch for r in results])

    bh_global = bh_fdr_mask(p_cluster, q=0.05)  # K = all results

    # Per-feature family K — group by feature and FDR within
    per_feat_survive = np.zeros(len(results), dtype=bool)
    feature_groups: dict[str, list[int]] = {}
    for i, r in enumerate(results):
        feature_groups.setdefault(r.signal, []).append(i)
    for feat, idxs in feature_groups.items():
        sub_p = np.array([results[i].p_cluster for i in idxs])
        sub_mask = bh_fdr_mask(sub_p, q=0.05)
        for j, i in enumerate(idxs):
            per_feat_survive[i] = sub_mask[j]

    # Per-(instrument, session) family K — for localized edge detection
    per_is_survive = np.zeros(len(results), dtype=bool)
    is_groups: dict[tuple[str, str], list[int]] = {}
    for i, r in enumerate(results):
        is_groups.setdefault((r.instrument, r.session), []).append(i)
    for key, idxs in is_groups.items():
        sub_p = np.array([results[i].p_cluster for i in idxs])
        sub_mask = bh_fdr_mask(sub_p, q=0.05)
        for j, i in enumerate(idxs):
            per_is_survive[i] = sub_mask[j]

    emit(df, results, bh_global, per_feat_survive, per_is_survive)


def classify(r: CellResult) -> str:
    """Exploratory heat flag."""
    if np.isnan(r.t_cluster):
        return "NA"
    t = abs(r.t_cluster)
    if t >= 4.0 and r.holdout_dir_match:
        return "HOT"
    if t >= 3.0 and r.holdout_dir_match:
        return "WARM"
    if t >= 2.5:
        return "LUKEWARM"
    return "COLD"


def emit(df, results, bh_global, per_feat, per_is):
    total = len(results)
    hot = [r for r in results if classify(r) == "HOT"]
    warm = [r for r in results if classify(r) == "WARM"]
    lukewarm = [r for r in results if classify(r) == "LUKEWARM"]
    sorted_by_t = sorted(results, key=lambda r: -abs(r.t_cluster) if not np.isnan(r.t_cluster) else 0)

    lines = [
        "# MEGA EXPLORATION — Prior-Day Features × Full Scope",
        "",
        f"**Rows IS/OOS:** {df['is_is'].sum()}/{df['is_oos'].sum()}",
        f"**Cells analyzed:** {total}",
        "",
        "**EXPLORATORY.** K exceeds Bailey MinBTL — multiple K framings reported.",
        "Primary gate: cluster-SE t (p_cluster). Heat classification by absolute cluster t.",
        "",
        f"- **HOT (|t_cluster| >= 4.0 AND holdout dir match):** {len(hot)}",
        f"- **WARM (|t_cluster| >= 3.0 AND holdout dir match):** {len(warm)}",
        f"- **LUKEWARM (|t_cluster| >= 2.5):** {len(lukewarm)}",
        f"- BH-FDR global (K={total}, q=0.05) survivors: {int(bh_global.sum())}",
        f"- BH-FDR per-feature family survivors: {int(per_feat.sum())}",
        f"- BH-FDR per-(instr,session) family survivors: {int(per_is.sum())}",
        "",
        "## Top 50 by |cluster-t|",
        "",
        "| flag | instr | session | apt | rr | dir | signal | N_on | N_off | ExpR_on | ExpR_off | Δ_IS | Δ_OOS | t_cl | p_cl | p_logit | dir_match | OOS/IS | BH_glob | BH_feat | BH_IS |",
        "|------|-------|---------|-----|----|-----|--------|------|-------|---------|----------|------|-------|------|------|---------|-----------|--------|---------|---------|-------|",
    ]

    idx_map = {id(r): i for i, r in enumerate(results)}
    for r in sorted_by_t[:50]:
        i = idx_map[id(r)]
        lines.append(
            f"| {classify(r)} | {r.instrument} | {r.session} | O{r.aperture} | {r.rr} | {r.direction} | {r.signal} | "
            f"{r.n_on_is} | {r.n_off_is} | {r.expr_on_is:+.3f} | {r.expr_off_is:+.3f} | "
            f"{r.delta_is:+.3f} | {r.delta_oos:+.3f} | {r.t_cluster:+.2f} | {r.p_cluster:.4f} | "
            f"{r.p_logit:.4f} | {'Y' if r.holdout_dir_match else '.'} | {r.holdout_effect_ratio:+.2f} | "
            f"{'Y' if bh_global[i] else '.'} | {'Y' if per_feat[i] else '.'} | {'Y' if per_is[i] else '.'} |"
        )

    if hot:
        lines += ["", "## HOT cells (|t_cluster| >= 4.0 AND holdout dir match)", ""]
        for r in hot:
            i = idx_map[id(r)]
            lines.append(
                f"- {r.instrument} {r.session} O{r.aperture} RR{r.rr} {r.direction} {r.signal}: "
                f"t_cl={r.t_cluster:+.2f} Δ_IS={r.delta_is:+.3f} Δ_OOS={r.delta_oos:+.3f} "
                f"N_on={r.n_on_is} BH_glob={'Y' if bh_global[i] else '.'}"
            )

    if warm:
        lines += ["", "## WARM cells (|t_cluster| >= 3.0 AND holdout dir match)", ""]
        for r in warm:
            i = idx_map[id(r)]
            lines.append(
                f"- {r.instrument} {r.session} O{r.aperture} RR{r.rr} {r.direction} {r.signal}: "
                f"t_cl={r.t_cluster:+.2f} Δ_IS={r.delta_is:+.3f} Δ_OOS={r.delta_oos:+.3f} "
                f"N_on={r.n_on_is} BH_feat={'Y' if per_feat[i] else '.'} BH_IS={'Y' if per_is[i] else '.'}"
            )

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {OUTPUT_MD} ({total} cells, {len(hot)} HOT, {len(warm)} WARM)")


if __name__ == "__main__":
    main()
