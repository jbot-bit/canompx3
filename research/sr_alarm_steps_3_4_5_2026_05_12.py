"""SR alarm Steps 3/4/5 reproducer for the 3 deployed MNQ lanes (2026-05-12).

Companion to ``research/sr_alarm_decomposition_2026_05_12.py`` (Step 2).
Reproduces the numbers cited in:

  - docs/audit/results/2026-05-12-sr-alarm-nyse-open-rr1.md
  - docs/audit/results/2026-05-12-sr-alarm-comex-settle-rr1.5.md
  - docs/audit/results/2026-05-12-sr-alarm-us-data-1000-rr1.5.md
  - docs/audit/results/2026-05-12-sr-alarm-3lane-summary.md

Sections:
  Step 3 — Per-lane fire-rate-by-year + per-(year, direction) sign-flip
           rate (F3 Harris-trigger checks (b)/(c)).
  Step 4 — Live Bailey-LdP 2014 DSR per lane
           (delegates to research.audit_ovnrng50_canonical_dsr.bailey_dsr).
  Step 5 — Cross-lane common-factor decomposition
           (F5(1) atr_vel_regime distribution shift, F5(2) cost-spec drift).

Lane list source: ``docs/audit/hypotheses/2026-05-12-3lane-sr-alarm-diagnosis.yaml``
``scope.lanes[].strategy_id`` (single source of truth per the pre-reg).

Authority chain — DELEGATE, do not re-encode (institutional-rigor sec 4):
  - ``pipeline.paths.GOLD_DB_PATH`` for DB path
  - ``pipeline.cost_model.COST_SPECS`` for cost-spec drift check (F5.2)
  - ``trading_app.eligibility.builder.parse_strategy_id`` for ID parsing
  - ``trading_app.holdout_policy.HOLDOUT_SACRED_FROM`` for IS/OOS split
  - ``research.filter_utils.filter_signal`` for canonical filter application
  - ``research.audit_ovnrng50_canonical_dsr.bailey_dsr`` for DSR math
    (top-level import safe after the 2026-05-12 ``__main__`` guard refactor)

Read-only against ``gold.db``. Writes only to stdout.
"""

from __future__ import annotations

import argparse
import sys
from math import e as EULER_E
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import yaml
from scipy import stats

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline.cost_model import COST_SPECS
from pipeline.paths import GOLD_DB_PATH
from research.audit_ovnrng50_canonical_dsr import EULER_GAMMA, bailey_dsr
from research.filter_utils import filter_signal
from trading_app.config import ALL_FILTERS
from trading_app.eligibility.builder import parse_strategy_id
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PREREG_PATH = (
    _PROJECT_ROOT
    / "docs"
    / "audit"
    / "hypotheses"
    / "2026-05-12-3lane-sr-alarm-diagnosis.yaml"
)

# Cost-spec snapshot the SR-baseline calibration was anchored against
# (validated_setups promoted_at = 2026-05-10 13:37-13:45+10:00, post-F-4 fix
# per feedback_doctrine_drift_cost_specs_2026_05_01.md). The F5(2) check
# asserts pipeline.cost_model.COST_SPECS still matches this snapshot at
# diagnostic-run time; any divergence means the cost model drifted between
# alarm calibration (2026-05-10) and diagnosis (2026-05-12).
SR_BASELINE_COST_SPEC_MNQ = {
    "total_friction": 2.92,
    "commission_rt": 1.42,
    "spread_doubled": 0.5,
    "slippage": 1.0,
}


# ---------------------------------------------------------------------------
# Pre-reg yaml loader (single source of truth for lane list)
# ---------------------------------------------------------------------------


def load_lanes_from_prereg(path: Path) -> tuple[dict[str, Any], ...]:
    """Read scope.lanes[] from the pre-reg yaml, return parsed lane records."""
    if not path.exists():
        raise FileNotFoundError(f"Pre-reg yaml not found: {path}")
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw_lanes = spec.get("scope", {}).get("lanes")
    if not raw_lanes:
        raise ValueError(f"Pre-reg yaml {path} has no scope.lanes[] entries.")
    out: list[dict[str, Any]] = []
    for lane in raw_lanes:
        sid = lane["strategy_id"]
        parsed = parse_strategy_id(sid)
        # Embed the SR snapshot baseline alongside parsed dimensions so
        # downstream functions don't have to re-load the yaml.
        snapshot = lane.get("sr_state_snapshot", {})
        out.append(
            {
                "strategy_id": sid,
                "parsed": parsed,
                "expected_r": float(snapshot.get("expected_r", float("nan"))),
                "std_r": float(snapshot.get("std_r", float("nan"))),
            }
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# Canonical-layer ledger fetch (mirrors Step 2 script)
# ---------------------------------------------------------------------------


def _fetch_lane_ledger(
    con: duckdb.DuckDBPyConnection, parsed: dict[str, Any]
) -> pd.DataFrame:
    q = """
    SELECT o.trading_day, o.entry_ts, o.pnl_r, o.outcome,
           d.atr_vel_regime, d.*
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol = ?
      AND o.orb_label = ?
      AND o.orb_minutes = ?
      AND o.entry_model = ?
      AND o.rr_target = ?
      AND o.confirm_bars = ?
    ORDER BY o.trading_day, o.entry_ts
    """
    df = con.execute(
        q,
        [
            parsed["instrument"],
            parsed["orb_label"],
            parsed["orb_minutes"],
            parsed["entry_model"],
            parsed["rr_target"],
            parsed["confirm_bars"],
        ],
    ).fetchdf()
    df["pnl_r"] = df["pnl_r"].fillna(0.0)
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    return df


# ---------------------------------------------------------------------------
# Step 3 — Fire-rate-by-year + per-(year) sign-flip
# ---------------------------------------------------------------------------


def step3_fire_rate_and_signflip(
    con: duckdb.DuckDBPyConnection, lane: dict[str, Any]
) -> None:
    parsed = lane["parsed"]
    sid = lane["strategy_id"]
    df = _fetch_lane_ledger(con, parsed)
    sig = filter_signal(df, parsed["filter_type"], parsed["orb_label"])
    df = df.copy()
    df["fires"] = sig.astype(int)
    df["yr"] = df["trading_day"].dt.year

    print(f"\nLANE: {sid}")
    print(f"  filter: {parsed['filter_type']!r}  session: {parsed['orb_label']!r}")
    print(f"  Fire-rate by year:")
    by_yr = (
        df.groupby("yr")
        .agg(N_total=("fires", "size"), N_fires=("fires", "sum"))
        .reset_index()
    )
    by_yr["fire_rate"] = by_yr["N_fires"] / by_yr["N_total"]
    for _, row in by_yr.iterrows():
        print(
            f"    {int(row.yr)}: total={int(row.N_total):4d}  "
            f"fires={int(row.N_fires):4d}  fire_rate={row.fire_rate:.4f}"
        )

    fires = df.loc[df["fires"] == 1].copy()
    if fires.empty:
        print("  Per-year filtered ExpR: NO FIRED TRADES")
        return

    per_yr = (
        fires.groupby("yr")["pnl_r"]
        .agg(["mean", "count"])
        .reset_index()
    )
    pooled_sign = 1 if fires["pnl_r"].mean() > 0 else -1
    eligible = per_yr.loc[per_yr["count"] >= 10]
    flips = int(((eligible["mean"] * pooled_sign) < 0).sum())
    total = int(len(eligible))
    flip_pct = 100.0 * flips / max(1, total)

    print(f"  Per-year filtered mean_r (N>=10):")
    for _, row in per_yr.iterrows():
        flag = ""
        if row["count"] >= 10 and (row["mean"] * pooled_sign) < 0:
            flag = " [FLIP]"
        print(
            f"    {int(row.yr)}: N={int(row['count']):4d}  "
            f"mean={row['mean']:+.4f}{flag}"
        )
    print(
        f"  Pooled-finding flip rate: {flips}/{total} = {flip_pct:.1f}% "
        f"(heterogeneity-ack threshold 25% per pooled-finding-rule.md)"
    )


# ---------------------------------------------------------------------------
# Step 4 — Live Bailey DSR (delegates to canonical bailey_dsr)
# ---------------------------------------------------------------------------


def _scan_cell_sr_samples(
    con: duckdb.DuckDBPyConnection, parsed: dict[str, Any]
) -> np.ndarray:
    """V[ŜR_n] sample: same (instrument, session) over all (apt, rr, filter).

    Mirrors the per-(instrument, session) scan used by
    research/audit_ovnrng50_canonical_dsr._run_ovnrng50_analysis. Confirm_bars
    locked to the lane's confirm_bars; entry_model locked to the lane's.
    """
    samples: list[float] = []
    for orb_m in (5, 15, 30):
        for rr_t in (1.0, 1.5, 2.0):
            q = """
            SELECT o.trading_day, o.pnl_r, d.*
            FROM orb_outcomes o
            JOIN daily_features d
              ON o.trading_day = d.trading_day AND o.symbol = d.symbol
             AND o.orb_minutes = d.orb_minutes
            WHERE o.symbol = ? AND o.orb_label = ? AND o.orb_minutes = ?
              AND o.entry_model = ? AND o.rr_target = ? AND o.confirm_bars = ?
              AND o.pnl_r IS NOT NULL
            """
            df_c = con.execute(
                q,
                [
                    parsed["instrument"],
                    parsed["orb_label"],
                    orb_m,
                    parsed["entry_model"],
                    rr_t,
                    parsed["confirm_bars"],
                ],
            ).fetchdf()
            if len(df_c) < 50:
                continue
            m = df_c["pnl_r"].mean()
            s = df_c["pnl_r"].std(ddof=1)
            if s > 0:
                samples.append(m / s)
            for fk in ALL_FILTERS.keys():
                try:
                    sg = filter_signal(df_c, fk, parsed["orb_label"])
                    sub = df_c.loc[sg == 1]
                    if len(sub) < 50:
                        continue
                    m2 = sub["pnl_r"].mean()
                    s2 = sub["pnl_r"].std(ddof=1)
                    if s2 > 0:
                        samples.append(m2 / s2)
                except Exception:
                    continue
    return np.array(samples)


def step4_live_dsr(
    con: duckdb.DuckDBPyConnection,
    lane: dict[str, Any],
    n_trials_at_discovery: int,
    deployment_dsr: float,
) -> None:
    parsed = lane["parsed"]
    sid = lane["strategy_id"]
    df = _fetch_lane_ledger(con, parsed)
    sig = filter_signal(df, parsed["filter_type"], parsed["orb_label"])
    sel = df.loc[sig == 1].copy()
    T = int(len(sel))
    if T < 30:
        print(f"\nLANE: {sid}  Step 4 SKIPPED: T={T} < 30 (insufficient).")
        return

    pnl = sel["pnl_r"].values
    mean_r = float(pnl.mean())
    std_r = float(pnl.std(ddof=1))
    sr_nonann = mean_r / std_r if std_r > 0 else float("nan")
    skew = float(stats.skew(pnl))
    kurt_pearson = float(stats.kurtosis(pnl, fisher=False))

    print(f"\nLANE: {sid}")
    print(
        f"  T={T}  mean={mean_r:+.4f}  std={std_r:.4f}  "
        f"SR_nonann={sr_nonann:+.4f}  SR_ann={sr_nonann * np.sqrt(252):+.3f}"
    )
    print(f"  skew={skew:+.3f}  kurt(Pearson)={kurt_pearson:+.3f}")

    sr_arr = _scan_cell_sr_samples(con, parsed)
    if len(sr_arr) < 2:
        print(f"  Scan cells N>=50: {len(sr_arr)} (insufficient for V[SR_n])")
        return
    V = float(sr_arr.var(ddof=1))
    print(
        f"  Scan cells N>=50: {len(sr_arr)}  "
        f"V[SR_n]={V:.6f}  sqrt(V)={np.sqrt(V):.4f}"
    )

    z1 = stats.norm.ppf(1 - 1 / n_trials_at_discovery)
    z2 = stats.norm.ppf(1 - 1 / (n_trials_at_discovery * EULER_E))
    emax_z = (1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2
    sr_0 = float(np.sqrt(V) * emax_z)
    dsr_live = bailey_dsr(sr_nonann, sr_0, T, skew, kurt_pearson)

    print(
        f"  E[max] z @ N={n_trials_at_discovery}: {emax_z:.4f}  "
        f"SR_0={sr_0:.4f}  LIVE_DSR={dsr_live:.4f}"
    )
    print(f"  deployment_DSR (validated_setups): {deployment_dsr:.6e}")
    verdict = "DSR_STABLE" if dsr_live >= 0.50 else "DSR_DECAYED (cross-check, Amendment 2.1)"
    print(f"  Verdict: {verdict}")


# ---------------------------------------------------------------------------
# Step 5 — Cross-lane common-factor: regime shift + cost-spec drift
# ---------------------------------------------------------------------------


def step5_regime_shift(
    con: duckdb.DuckDBPyConnection, lanes: tuple[dict[str, Any], ...]
) -> None:
    holdout = pd.Timestamp(HOLDOUT_SACRED_FROM)
    print()
    print("F5(1) atr_vel_regime distribution shift (full vs OOS vs recent-60):")
    for lane in lanes:
        parsed = lane["parsed"]
        sid = lane["strategy_id"]
        df = _fetch_lane_ledger(con, parsed)
        full = df["atr_vel_regime"].value_counts(normalize=True).sort_index().round(3).to_dict()
        oos = df.loc[df["trading_day"] >= holdout]
        oos_dist = (
            oos["atr_vel_regime"].value_counts(normalize=True).sort_index().round(3).to_dict()
            if len(oos) > 0
            else {}
        )
        rec = df.tail(60)
        rec_dist = rec["atr_vel_regime"].value_counts(normalize=True).sort_index().round(3).to_dict()
        full_stable = full.get("Stable", 0.0)
        oos_stable = oos_dist.get("Stable", 0.0)
        delta_pp = (oos_stable - full_stable) * 100
        print(f"  {sid}")
        print(f"    full   N={len(df):4d}: {full}")
        print(f"    OOS    N={len(oos):4d}: {oos_dist}")
        print(f"    rec60  N={len(rec):4d}: {rec_dist}")
        print(
            f"    Stable share full -> OOS: {full_stable:.3f} -> {oos_stable:.3f} "
            f"({delta_pp:+.1f}pp)"
        )


def step5_cost_spec_drift() -> None:
    print()
    print("F5(2) cost-spec drift check vs SR-baseline snapshot (2026-05-10):")
    mnq = COST_SPECS["MNQ"]
    drift_count = 0
    for field, expected in SR_BASELINE_COST_SPEC_MNQ.items():
        actual = getattr(mnq, field)
        match = "OK" if abs(actual - expected) < 1e-6 else "DRIFTED"
        if match == "DRIFTED":
            drift_count += 1
        print(
            f"  MNQ.{field}: expected={expected}  actual={actual}  -> {match}"
        )
    if drift_count == 0:
        print("  Verdict: NOT_FIRED (no cost-spec drift since SR-baseline snapshot).")
    else:
        print(
            f"  Verdict: FIRED ({drift_count} field(s) drifted) — re-derive SR baseline."
        )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prereg",
        type=Path,
        default=PREREG_PATH,
        help="Pre-reg yaml path (default: docs/audit/hypotheses/2026-05-12-3lane-sr-alarm-diagnosis.yaml).",
    )
    parser.add_argument(
        "--steps",
        default="3,4,5",
        help="Comma-separated step numbers to run (default: 3,4,5).",
    )
    args = parser.parse_args()
    requested = {int(s.strip()) for s in args.steps.split(",") if s.strip()}

    lanes = load_lanes_from_prereg(args.prereg)
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    print("=" * 78)
    print("SR ALARM STEPS 3/4/5 REPRODUCER — 2026-05-12")
    print(f"DB:        {GOLD_DB_PATH}")
    print(f"Pre-reg:   {args.prereg.relative_to(_PROJECT_ROOT)}")
    print(f"Holdout:   {HOLDOUT_SACRED_FROM}")
    print(f"Lanes:     {len(lanes)} from pre-reg scope.lanes[]")
    print(f"Steps:     {sorted(requested)}")
    print("=" * 78)

    if 3 in requested:
        print()
        print("-" * 78)
        print("Step 3 — F3 fire-rate-by-year + per-year sign-flip rate")
        print("-" * 78)
        for lane in lanes:
            step3_fire_rate_and_signflip(con, lane)

    if 4 in requested:
        print()
        print("-" * 78)
        print("Step 4 — F4 Bailey-LdP 2014 DSR live recompute")
        print("-" * 78)
        # Load deployment-time DSR + n_trials_at_discovery from validated_setups
        # for each lane (CANONICAL READ from canonical layer; not a research input).
        for lane in lanes:
            sid = lane["strategy_id"]
            row = con.execute(
                "SELECT n_trials_at_discovery, dsr_score "
                "FROM validated_setups WHERE strategy_id = ?",
                [sid],
            ).fetchone()
            if row is None:
                print(f"\nLANE: {sid}  Step 4 SKIPPED: not in validated_setups.")
                continue
            n_trials, deployment_dsr = int(row[0]), float(row[1])
            step4_live_dsr(con, lane, n_trials, deployment_dsr)

    if 5 in requested:
        print()
        print("-" * 78)
        print("Step 5 — F5 cross-lane common-factor decomposition")
        print("-" * 78)
        step5_regime_shift(con, lanes)
        step5_cost_spec_drift()

    print()
    print("=" * 78)
    print("DONE.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
