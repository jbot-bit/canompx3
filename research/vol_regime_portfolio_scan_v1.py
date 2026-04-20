"""
Vol-regime confluence portfolio-generalization scan v1.

Pre-reg: docs/audit/hypotheses/2026-04-20-vol-regime-confluence-portfolio-test.yaml
Purpose: test whether (F_lane AND vol_regime_variant) adds marginal edge
         to 6 MNQ live allocator lanes, or is COMEX_SETTLE-specific.

17 cells (family Pathway A, BH-FDR K=17, MinBTL 5.67yr < 6.65yr available):
  4 OVN-valid lanes × 3 variants (OVN, XMES, OVN OR XMES) = 12
  2 OVN-invalid lanes × 1 variant (XMES only)             =  2
  3 LON-valid lanes × 1 variant (LON)                     =  3

Canonical delegation:
  - research.filter_utils.filter_signal for all base filters (ORB_G5, ATR_P50,
    COST_LT12). Per research-truth-protocol.md § Canonical filter delegation,
    no inline re-encoding. Vol-regime variants (OVNRNG_100, X_MES_ATR60,
    LONDON_RANGE_100) are computed directly from daily_features columns —
    not existing as canonical filters for this scope in trading_app.config,
    so inline is legitimate (not re-encoding a canonical).

Temporal-alignment gate (RULE 1.2):
  - overnight_range valid ONLY for ORB starting ≥ 17:00 Brisbane
    (excluded from SINGAPORE_OPEN, TOKYO_OPEN)
  - session_london_range valid ONLY for ORB starting ≥ 23:00 Brisbane
    (excluded from EUROPE_FLOW, SINGAPORE_OPEN, TOKYO_OPEN)
  - mes_atr_20_pct (20-day rolling, prior close) valid at all session starts

Joins:
  - Triple-join: orb_outcomes.(trading_day, symbol, orb_minutes)
    = daily_features.(trading_day, symbol, orb_minutes)
  - MES atr cross-asset injection: LEFT JOIN on (trading_day) with
    CTE guard WHERE orb_minutes=5 (RULE 9 CTE deduplication).

Holdout:
  - IS: trading_day < 2026-01-01 (HOLDOUT_SACRED_FROM)
  - OOS: 2026-01-02 through 2026-04-16 (69 days)
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from pipeline.paths import GOLD_DB_PATH
from research.filter_utils import filter_signal
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM as _HOLDOUT_DATE

HOLDOUT_SACRED_FROM = pd.Timestamp(_HOLDOUT_DATE)

# ---- Configuration ----
OOS_WINDOW_END = pd.Timestamp("2026-04-16")

OVN_THRESHOLD = 100.0
XMES_THRESHOLD = 60.0
LON_THRESHOLD = 100.0

# Lane spec: (strategy_id, orb_label, orb_minutes, rr_target, base_filter_key, ovn_valid, lon_valid)
LANES = [
    ("MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5",       "EUROPE_FLOW",    5,  1.5, "ORB_G5",    True,  False),
    ("MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15","SINGAPORE_OPEN", 15, 1.5, "ATR_P50",   False, False),
    ("MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5",       "COMEX_SETTLE",   5,  1.5, "ORB_G5",    True,  True),
    ("MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12",       "NYSE_OPEN",      5,  1.0, "COST_LT12", True,  True),
    ("MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12",      "TOKYO_OPEN",     5,  1.5, "COST_LT12", False, False),
    ("MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15",   "US_DATA_1000",   15, 1.5, "ORB_G5",    True,  True),
]

ERAS = [
    (pd.Timestamp("2019-01-01"), pd.Timestamp("2020-12-31"), "2019-2020"),
    (pd.Timestamp("2021-01-01"), pd.Timestamp("2022-12-31"), "2021-2022"),
    (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31"), "2023"),
    (pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31"), "2024-2025"),
]


@dataclass
class CellResult:
    cell_id: int
    lane: str
    orb_label: str
    orb_minutes: int
    rr_target: float
    base_filter: str
    variant: str

    # Base-filter fire (the live lane itself) — context
    n_base_is: int
    expr_base_is: float
    t_base_is: float
    n_base_oos: int
    expr_base_oos: float

    # Variant fire (F_lane AND vol_regime)
    n_variant_is: int
    expr_variant_is: float
    t_variant_is: float
    p_variant_is: float
    sd_variant_is: float
    wr_variant_is: float
    n_variant_oos: int
    expr_variant_oos: float
    t_variant_oos: float
    p_variant_oos: float
    dir_match_oos: bool

    # Marginal: variant vs base (is the variant carrying more edge?)
    marginal_delta_expr: float       # variant_expr - base_expr
    marginal_delta_wr: float         # variant_wr - base_wr
    expr_base_nonfire: float         # F_lane AND NOT variant
    n_base_nonfire: int

    # Fire rates
    fire_rate_variant_on_base: float  # P(variant fires | base fires)
    fire_rate_raw: float              # P(variant fires) on full IS universe

    # T0 tautology — correlation of variant fire mask with base fire mask
    t0_corr_base_variant: float

    # C8 ratio (live OOS ExpR / IS ExpR on variant-fire subset)
    c8_ratio: float

    # Era stability (Criterion 9)
    era_pass: bool
    era_breaches: list

    # Per-year
    yearly: dict

    # Gates
    gates_passed: dict
    all_gates_passed: bool


def build_variant_mask(df: pd.DataFrame, variant: str, ovn_valid: bool, lon_valid: bool) -> np.ndarray | None:
    """Build the vol-regime variant fire mask. Returns None if variant is invalid for this lane."""
    if variant == "ovn_only":
        if not ovn_valid:
            return None
        return (df["overnight_range"].fillna(-np.inf) >= OVN_THRESHOLD).to_numpy().astype(int)
    if variant == "xmes_only":
        return (df["mes_atr_20_pct"].fillna(-np.inf) >= XMES_THRESHOLD).to_numpy().astype(int)
    if variant == "ovn_or_xmes":
        if not ovn_valid:
            return None
        ovn = (df["overnight_range"].fillna(-np.inf) >= OVN_THRESHOLD).to_numpy().astype(int)
        xmes = (df["mes_atr_20_pct"].fillna(-np.inf) >= XMES_THRESHOLD).to_numpy().astype(int)
        return ((ovn | xmes)).astype(int)
    if variant == "london_range_only":
        if not lon_valid:
            return None
        return (df["session_london_range"].fillna(-np.inf) >= LON_THRESHOLD).to_numpy().astype(int)
    raise ValueError(f"Unknown variant: {variant}")


def _stats(pnl: np.ndarray) -> tuple[int, float, float, float, float, float]:
    """Return (N, mean, sd, t, p_two_sided, wr). For N<=1 or sd=0, t=0, p=1."""
    n = len(pnl)
    if n == 0:
        return 0, 0.0, 0.0, 0.0, 1.0, 0.0
    mean = float(pnl.mean())
    sd = float(pnl.std(ddof=1)) if n > 1 else 0.0
    wr = float((pnl > 0).mean())
    if n > 1 and sd > 0:
        t = mean / sd * np.sqrt(n)
        p = 2 * (1 - stats.t.cdf(abs(t), n - 1))
    else:
        t, p = 0.0, 1.0
    return n, mean, sd, float(t), float(p), wr


def load_lane_data(con: duckdb.DuckDBPyConnection, orb_label: str, orb_minutes: int, rr_target: float) -> pd.DataFrame:
    """Load trade outcomes + per-day features joined canonically.

    Triple-join on (trading_day, symbol, orb_minutes).
    MES atr LEFT JOIN via CTE with WHERE orb_minutes=5 guard (RULE 9 CTE).

    Returns DataFrame with one row per orb_outcomes trade (after filter ranges).
    Columns include pnl_r plus all the feature columns needed for mask computation.
    """
    q = """
    WITH mnq_feat AS (
      SELECT trading_day, symbol, orb_minutes,
             overnight_range, orb_COMEX_SETTLE_size, orb_EUROPE_FLOW_size,
             orb_NYSE_OPEN_size, orb_US_DATA_1000_size,
             orb_SINGAPORE_OPEN_size, orb_TOKYO_OPEN_size,
             atr_20, atr_20_pct,
             session_london_high, session_london_low
      FROM daily_features
      WHERE symbol = 'MNQ'
    ),
    mes_atr AS (
      SELECT trading_day, atr_20_pct AS mes_atr_20_pct
      FROM daily_features
      WHERE symbol = 'MES' AND orb_minutes = 5
    )
    SELECT o.trading_day, o.symbol, o.pnl_r,
           m.overnight_range,
           m.orb_COMEX_SETTLE_size, m.orb_EUROPE_FLOW_size, m.orb_NYSE_OPEN_size,
           m.orb_US_DATA_1000_size, m.orb_SINGAPORE_OPEN_size, m.orb_TOKYO_OPEN_size,
           m.atr_20, m.atr_20_pct,
           (m.session_london_high - m.session_london_low) AS session_london_range,
           x.mes_atr_20_pct
    FROM orb_outcomes o
    JOIN mnq_feat m
      ON o.trading_day = m.trading_day
     AND o.symbol     = m.symbol
     AND o.orb_minutes = m.orb_minutes
    LEFT JOIN mes_atr x ON o.trading_day = x.trading_day
    WHERE o.symbol      = 'MNQ'
      AND o.orb_label   = ?
      AND o.orb_minutes = ?
      AND o.entry_model = 'E2'
      AND o.confirm_bars = 1
      AND o.rr_target   = ?
      AND o.pnl_r IS NOT NULL
    ORDER BY o.trading_day
    """
    df = con.execute(q, [orb_label, orb_minutes, rr_target]).df()
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    return df


def compute_base_filter_mask(df: pd.DataFrame, base_filter: str, orb_label: str) -> np.ndarray:
    """Delegate to canonical research.filter_utils.filter_signal — NO re-encoding."""
    return filter_signal(df, base_filter, orb_label)


def bh_fdr(p_values: list[float], q: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg. Returns list of pass/fail at q level (False=survives)."""
    n = len(p_values)
    if n == 0:
        return []
    idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[idx]
    thresholds = (np.arange(1, n + 1) / n) * q
    passes_sorted = sorted_p <= thresholds
    # largest k where p_(k) <= (k/n)*q; all k' <= k are survivors
    if not passes_sorted.any():
        return [False] * n
    k_max = np.where(passes_sorted)[0].max()
    survivors_sorted = np.zeros(n, dtype=bool)
    survivors_sorted[: k_max + 1] = True
    survivors = np.zeros(n, dtype=bool)
    for i, orig_i in enumerate(idx):
        survivors[orig_i] = survivors_sorted[i]
    return survivors.tolist()


def era_stability_check(df: pd.DataFrame, variant_mask: np.ndarray, is_mask: np.ndarray) -> tuple[bool, list]:
    """Per Criterion 9: no era with N>=50 may have ExpR < -0.05."""
    breaches = []
    is_only = df[is_mask]
    var_in_is = variant_mask[is_mask]
    is_fire = is_only[var_in_is.astype(bool)]
    for s, e, label in ERAS:
        sub = is_fire[(is_fire["trading_day"] >= s) & (is_fire["trading_day"] <= e)]
        n = len(sub)
        if n == 0:
            continue
        m = float(sub.pnl_r.mean())
        if n >= 50 and m < -0.05:
            breaches.append({"era": label, "n": n, "expr": m})
    return (len(breaches) == 0), breaches


def yearly_stats(df: pd.DataFrame, variant_mask: np.ndarray, is_mask: np.ndarray) -> dict:
    out = {}
    is_only = df[is_mask]
    var_in_is = variant_mask[is_mask]
    is_fire = is_only[var_in_is.astype(bool)].copy()
    is_fire["year"] = is_fire["trading_day"].dt.year
    for y, g in is_fire.groupby("year"):
        n, m, _, t, _, wr = _stats(g.pnl_r.values)
        out[int(y)] = {"n": n, "expr": round(m, 4), "t": round(t, 2), "wr": round(wr, 4)}
    return out


def run_cell(
    con: duckdb.DuckDBPyConnection,
    cell_id: int,
    lane: str,
    orb_label: str,
    orb_minutes: int,
    rr_target: float,
    base_filter: str,
    variant: str,
    ovn_valid: bool,
    lon_valid: bool,
) -> CellResult | None:
    # Load data
    df = load_lane_data(con, orb_label, orb_minutes, rr_target)
    if len(df) == 0:
        return None

    # Canonical base-filter mask
    base_mask = compute_base_filter_mask(df, base_filter, orb_label)
    # Variant mask
    var_mask = build_variant_mask(df, variant, ovn_valid, lon_valid)
    if var_mask is None:
        return None
    # Confluence: base AND variant
    confluence_mask = (base_mask & var_mask).astype(int)
    # Base-fire AND NOT variant — for marginal comparison
    base_nonvariant = (base_mask & (1 - var_mask)).astype(int)

    # IS / OOS masks
    is_mask = (df["trading_day"] < HOLDOUT_SACRED_FROM).to_numpy()
    oos_mask = (
        (df["trading_day"] >= HOLDOUT_SACRED_FROM)
        & (df["trading_day"] <= OOS_WINDOW_END)
    ).to_numpy()

    # Base-filter IS/OOS stats (the live lane, context)
    base_is = df[is_mask & (base_mask == 1)]
    n_base_is, expr_base_is, _, t_base_is, _, wr_base_is = _stats(base_is.pnl_r.values)
    base_oos = df[oos_mask & (base_mask == 1)]
    n_base_oos, expr_base_oos, _, _, _, _ = _stats(base_oos.pnl_r.values)

    # Variant (confluence) IS/OOS stats
    conf_is = df[is_mask & (confluence_mask == 1)]
    n_v_is, expr_v_is, sd_v_is, t_v_is, p_v_is, wr_v_is = _stats(conf_is.pnl_r.values)
    conf_oos = df[oos_mask & (confluence_mask == 1)]
    n_v_oos, expr_v_oos, _, t_v_oos, p_v_oos, _ = _stats(conf_oos.pnl_r.values)

    # Base-nonfire (F_lane AND NOT variant) — should be weaker if hypothesis true
    nonfire_is = df[is_mask & (base_nonvariant == 1)]
    n_nonfire, expr_nonfire, _, _, _, _ = _stats(nonfire_is.pnl_r.values)

    # Fire rates
    fire_rate_on_base = float(confluence_mask[is_mask & (base_mask == 1)].mean()) if (is_mask & (base_mask == 1)).sum() > 0 else 0.0
    fire_rate_raw = float(var_mask[is_mask].mean()) if is_mask.sum() > 0 else 0.0

    # T0 correlation (variant mask vs base mask on IS)
    if is_mask.sum() > 1:
        t0_corr = float(np.corrcoef(base_mask[is_mask], var_mask[is_mask])[0, 1])
    else:
        t0_corr = 0.0

    # Dir match
    dir_match = (np.sign(expr_v_is) == np.sign(expr_v_oos)) if (n_v_is > 0 and n_v_oos > 0) else False

    # C8 ratio
    c8 = (expr_v_oos / expr_v_is) if expr_v_is != 0 else 0.0

    # Era stability
    era_pass, era_breaches = era_stability_check(df, confluence_mask, is_mask)

    # Yearly
    yearly = yearly_stats(df, confluence_mask, is_mask)

    # Gates (cell_pass_gates from pre-reg). Stored as strings for clean JSON/CSV:
    # "pass" | "fail" | "pending" (pending = computed in a later stage).
    def _g(v: bool) -> str:
        return "pass" if v else "fail"

    gates = {
        "t0_not_tautology": _g(abs(t0_corr) < 0.70),
        "fire_rate_band": _g(0.05 <= fire_rate_on_base <= 0.95),
        "not_arithmetic_only": _g(not (abs(wr_v_is - wr_base_is) < 0.03 and abs(expr_v_is - expr_base_is) > 0.10)),
        "chordia_t_ge_3": _g(abs(t_v_is) >= 3.0),
        "bh_fdr_primary": "pending",         # set after all cells run (Stage F)
        "dir_match": _g(dir_match),
        "c8_ratio": _g(c8 >= 0.40),
        "era_stable": _g(era_pass),
        "t4_sensitivity": "pending",         # Stage G
        "t6_null_bootstrap": "pending",      # Stage H
    }
    all_passed = all(v == "pass" for v in gates.values() if v != "pending")

    return CellResult(
        cell_id=cell_id,
        lane=lane,
        orb_label=orb_label,
        orb_minutes=orb_minutes,
        rr_target=rr_target,
        base_filter=base_filter,
        variant=variant,
        n_base_is=n_base_is, expr_base_is=expr_base_is, t_base_is=t_base_is,
        n_base_oos=n_base_oos, expr_base_oos=expr_base_oos,
        n_variant_is=n_v_is, expr_variant_is=expr_v_is, t_variant_is=t_v_is,
        p_variant_is=p_v_is, sd_variant_is=sd_v_is, wr_variant_is=wr_v_is,
        n_variant_oos=n_v_oos, expr_variant_oos=expr_v_oos, t_variant_oos=t_v_oos,
        p_variant_oos=p_v_oos, dir_match_oos=dir_match,
        marginal_delta_expr=expr_v_is - expr_base_is,
        marginal_delta_wr=wr_v_is - wr_base_is,
        expr_base_nonfire=expr_nonfire, n_base_nonfire=n_nonfire,
        fire_rate_variant_on_base=fire_rate_on_base,
        fire_rate_raw=fire_rate_raw,
        t0_corr_base_variant=t0_corr,
        c8_ratio=c8,
        era_pass=era_pass, era_breaches=era_breaches,
        yearly=yearly,
        gates_passed=gates, all_gates_passed=all_passed,
    )


def main():
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    print(f"DB: {GOLD_DB_PATH}")
    print(f"HOLDOUT_SACRED_FROM: {HOLDOUT_SACRED_FROM}")
    print(f"OOS_WINDOW_END: {OOS_WINDOW_END.date()}")
    print(f"Thresholds: OVN>={OVN_THRESHOLD}  XMES>={XMES_THRESHOLD}  LON>={LON_THRESHOLD}")
    print()

    cells = []
    cell_id = 0
    for lane, orb_label, orb_min, rr, base_f, ovn_v, lon_v in LANES:
        variants = []
        if ovn_v:
            variants.append("ovn_only")
        variants.append("xmes_only")
        if ovn_v:
            variants.append("ovn_or_xmes")
        if lon_v:
            variants.append("london_range_only")
        for v in variants:
            cell_id += 1
            r = run_cell(con, cell_id, lane, orb_label, orb_min, rr, base_f, v, ovn_v, lon_v)
            if r is None:
                print(f"[SKIP] cell {cell_id} {lane} {v}")
                continue
            cells.append(r)

    # BH-FDR at K=17 primary, using p_variant_is
    pvals = [c.p_variant_is for c in cells]
    survivors = bh_fdr(pvals, q=0.05)
    for c, surv in zip(cells, survivors):
        c.gates_passed["bh_fdr_primary"] = "pass" if surv else "fail"
        c.all_gates_passed = all(v == "pass" for v in c.gates_passed.values() if v != "pending")

    # Print summary
    print("=" * 140)
    print(f"{'ID':>3} {'Lane':<15} {'O':>2} {'RR':>3} {'Filter':<12} {'Variant':<20}  "
          f"{'N_is':>4} {'ExpR_is':>8} {'t_is':>6} {'p_is':>8}  "
          f"{'N_oos':>5} {'ExpR_oos':>9} {'dirm':>4}  "
          f"{'C8':>5} {'fire%':>6} {'T0':>6} BH  ALL")
    print("=" * 140)
    for c in cells:
        print(
            f"{c.cell_id:>3} {c.orb_label:<15} {c.orb_minutes:>2} {c.rr_target:>3.1f} "
            f"{c.base_filter:<12} {c.variant:<20}  "
            f"{c.n_variant_is:>4} {c.expr_variant_is:>+8.4f} {c.t_variant_is:>+6.2f} {c.p_variant_is:>8.4f}  "
            f"{c.n_variant_oos:>5} {c.expr_variant_oos:>+9.4f} {str(c.dir_match_oos):>4}  "
            f"{c.c8_ratio:>+5.2f} {100*c.fire_rate_variant_on_base:>5.1f}% {c.t0_corr_base_variant:>+6.2f} "
            f"{'Y' if c.gates_passed['bh_fdr_primary'] else 'n':>2}  {'Y' if c.all_gates_passed else 'n'}"
        )
    print("=" * 140)

    passers = [c for c in cells if c.all_gates_passed]
    lanes_with_pass = {c.orb_label for c in passers}
    print(f"\nCells passing ALL gates: {len(passers)}/{len(cells)}")
    print(f"Lanes with at least 1 passing variant: {len(lanes_with_pass)} / 6 -> {sorted(lanes_with_pass)}")

    # Verdict per pre-reg decision tree
    print("\n=== VERDICT (per pre-committed decision tree) ===")
    if len(lanes_with_pass) >= 4:
        print(f"  PORTFOLIO_GENERAL — {len(lanes_with_pass)}/6 lanes passing. Phase 5 should pre-reg allocator-level vol-regime conditioner.")
    elif len(lanes_with_pass) in (2, 3):
        print(f"  COEXISTS_BOTH — {len(lanes_with_pass)}/6 lanes passing. Effect is lane-dependent. Per-lane Phase 5 pre-reg.")
    elif lanes_with_pass == {"COMEX_SETTLE"}:
        print("  COMEX_SETTLE_SPECIFIC — only COMEX_SETTLE passes. Prior overlap finding was correct-arithmetic but narrow-framing.")
    elif len(lanes_with_pass) == 1:
        print(f"  SINGLE_LANE — only {sorted(lanes_with_pass)[0]} passes. Narrow finding.")
    else:
        print("  DEAD — 0 lanes pass. Confluence effect is NOT portfolio-general. Prior finding was post-hoc noise.")

    # Persist
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / "vol_regime_portfolio_scan_v1.json"
    csv_path = out_dir / "vol_regime_portfolio_scan_v1.csv"

    def _to_jsonable(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return str(obj)

    with json_path.open("w") as f:
        json.dump([asdict(c) for c in cells], f, indent=2, default=_to_jsonable)

    # CSV: flatten nested dicts as JSON strings (dicts contain only str/int/float now)
    with csv_path.open("w", newline="") as f:
        if cells:
            w = csv.DictWriter(f, fieldnames=list(asdict(cells[0]).keys()))
            w.writeheader()
            for c in cells:
                row = asdict(c)
                row["gates_passed"] = json.dumps(row["gates_passed"], default=_to_jsonable)
                row["era_breaches"] = json.dumps(row["era_breaches"], default=_to_jsonable)
                row["yearly"] = json.dumps(row["yearly"], default=_to_jsonable)
                w.writerow(row)

    print(f"\nOutputs:\n  {json_path}\n  {csv_path}")
    return cells


if __name__ == "__main__":
    main()
