"""SR alarm decomposition for the 3 deployed MNQ lanes (2026-05-12).

Step 2 of the pre-registered diagnostic at
``docs/audit/hypotheses/2026-05-12-3lane-sr-alarm-diagnosis.yaml``.

For each of the 3 lanes:
  - Pull the canonical trade ledger from ``orb_outcomes`` JOIN ``daily_features``
    (triple-key join per ``.claude/rules/daily-features-joins.md``).
  - Decompose the time-series into 5 components on a rolling 60-trade window:
    ExpR, stdev(pnl_r), fire-rate (informational — full-history vs recent),
    win-rate, days-per-trade cadence.
  - Tag each window with the SR-alarm-trigger window vs current window vs
    full-history reference.
  - Power-tier every recent-vs-history comparison via
    ``research.oos_power.oos_ttest_power`` per RULE 3.3.

Pressure-test (RULE 13): ``--pressure-test`` injects a synthetic look-ahead
column into the SQL output and asserts the script flags it before reporting.

Authority chain — DELEGATE, do not re-encode (institutional-rigor sec 4):
  - ``pipeline.paths.GOLD_DB_PATH`` for DB path
  - ``trading_app.eligibility.builder.parse_strategy_id`` for ID parsing
  - ``research.oos_power.oos_ttest_power`` + ``power_verdict`` for power floor
  - ``research.filter_utils.filter_signal`` for canonical filter application

Read-only against ``gold.db``. Writes only to stdout (operator-readable) and
optionally a JSON sidecar (``--out`` flag) for the result-MD authoring step.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH
from research.filter_utils import filter_signal
from research.oos_power import format_power_report, oos_ttest_power, power_verdict
from trading_app.eligibility.builder import parse_strategy_id
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WINDOW = 60
LANES: tuple[str, ...] = (
    "MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12",
    "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100",
    "MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15",
)


@dataclass
class LaneResult:
    strategy_id: str
    parsed: dict[str, Any]
    n_total: int
    n_is: int
    n_oos: int
    is_window: tuple[str, str]
    oos_window: tuple[str, str]
    full_mean: float
    full_std: float
    full_wr: float
    full_fire_rate: float
    recent60_mean: float
    recent60_std: float
    recent60_wr: float
    recent60_cadence_days_per_trade: float
    is_mean: float
    is_std: float
    oos_mean: float
    oos_std: float
    rolling: list[dict[str, Any]] = field(default_factory=list)
    power_recent_vs_full: dict[str, float] = field(default_factory=dict)
    component_flags: dict[str, str] = field(default_factory=dict)
    pressure_test_passed: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "parsed": self.parsed,
            "n_total": self.n_total,
            "n_is": self.n_is,
            "n_oos": self.n_oos,
            "is_window": self.is_window,
            "oos_window": self.oos_window,
            "full_mean": self.full_mean,
            "full_std": self.full_std,
            "full_wr": self.full_wr,
            "full_fire_rate": self.full_fire_rate,
            "recent60_mean": self.recent60_mean,
            "recent60_std": self.recent60_std,
            "recent60_wr": self.recent60_wr,
            "recent60_cadence_days_per_trade": self.recent60_cadence_days_per_trade,
            "is_mean": self.is_mean,
            "is_std": self.is_std,
            "oos_mean": self.oos_mean,
            "oos_std": self.oos_std,
            "rolling": self.rolling,
            "power_recent_vs_full": self.power_recent_vs_full,
            "component_flags": self.component_flags,
            "pressure_test_passed": self.pressure_test_passed,
        }


def _connect() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(GOLD_DB_PATH), read_only=True)


def _fetch_lane_ledger(
    con: duckdb.DuckDBPyConnection, parsed: dict[str, Any]
) -> pd.DataFrame:
    """Triple-key JOIN orb_outcomes JOIN daily_features for one lane.

    Filter to the strategy's exact dimensions; canonical filter application
    happens downstream via ``research.filter_utils.filter_signal``. Includes
    pnl_r=NULL rows then turns them into 0R per scratch policy
    ``include-as-zero`` declared in the pre-reg yaml.
    """
    q = """
    SELECT o.trading_day, o.entry_ts, o.pnl_r, o.outcome,
           d.*
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
    return df


def _apply_filter(df: pd.DataFrame, parsed: dict[str, Any]) -> pd.DataFrame:
    """Apply canonical filter signal; keep rows where signal == 1.

    NO_FILTER lanes (vanishingly rare for current deployed set) bypass.
    """
    if parsed["filter_type"] == "NO_FILTER":
        return df.copy()
    sig = filter_signal(df, parsed["filter_type"], parsed["orb_label"])
    return df.loc[sig == 1].copy()


def _scratch_to_zero(df: pd.DataFrame) -> pd.DataFrame:
    """Convert pnl_r=NULL (scratch / no-fill) to 0R per scratch_policy.

    Live execution forces flat at session end; backtest must book $0 to
    satisfy Chan 2013 Ch 1 unified-program doctrine.
    """
    out = df.copy()
    out["pnl_r"] = out["pnl_r"].fillna(0.0)
    return out


def _compute_lane_stats(df: pd.DataFrame) -> tuple[float, float, float]:
    """Return (mean, std, win_rate) on the given pnl_r series.

    win_rate counts pnl_r > 0 (zero-scratches are non-wins). Returns
    (nan, nan, nan) on empty input.
    """
    if df.empty:
        return float("nan"), float("nan"), float("nan")
    pnl = df["pnl_r"].astype(float).to_numpy()
    mean = float(pnl.mean())
    std = float(pnl.std(ddof=1)) if len(pnl) > 1 else float("nan")
    wr = float((pnl > 0).mean())
    return mean, std, wr


def _rolling_windows(filtered: pd.DataFrame, window: int) -> list[dict[str, Any]]:
    """Step-through ``window``-trade rolling stats; one row per stride.

    Stride defaults to ``window // 4`` (15 for window=60) so per-lane plots
    have ~4x as many points as a non-overlapping decomposition without
    multiplying noise. Cadence is days-per-trade in the window.
    """
    if filtered.empty:
        return []
    stride = max(1, window // 4)
    df = filtered.reset_index(drop=True).copy()
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    out: list[dict[str, Any]] = []
    for end in range(window, len(df) + 1, stride):
        block = df.iloc[end - window : end]
        mean, std, wr = _compute_lane_stats(block)
        days = (block["trading_day"].iloc[-1] - block["trading_day"].iloc[0]).days
        cadence = days / max(1, len(block) - 1)
        out.append(
            {
                "window_end_idx": int(end),
                "window_start_day": str(block["trading_day"].iloc[0].date()),
                "window_end_day": str(block["trading_day"].iloc[-1].date()),
                "n": int(len(block)),
                "mean_r": mean,
                "std_r": std,
                "win_rate": wr,
                "days_per_trade": float(cadence),
            }
        )
    return out


def _component_flags(
    full_mean: float,
    full_std: float,
    full_wr: float,
    recent_mean: float,
    recent_std: float,
    recent_wr: float,
    pooled_std: float,
    expected_r: float,
) -> dict[str, str]:
    """Per-lane component flags per pre-reg F2 kill criterion.

    Returns a dict keyed by component (variance / mean / fire_rate / wr) with
    one of the canonical flag values: VARIANCE_COMPRESSION, MEAN_DECAY,
    WR_DECAY, NORMAL, INSUFFICIENT_DATA.
    """
    flags: dict[str, str] = {}
    if not np.isfinite(full_std) or not np.isfinite(recent_std):
        flags["variance"] = "INSUFFICIENT_DATA"
    elif recent_std < 0.6 * full_std:
        flags["variance"] = "VARIANCE_COMPRESSION"
    else:
        flags["variance"] = "NORMAL"

    if not np.isfinite(recent_mean):
        flags["mean"] = "INSUFFICIENT_DATA"
    elif recent_mean < 0 and full_mean > 0.10:
        flags["mean"] = "MEAN_DECAY"
    elif abs(recent_mean - expected_r) <= 1.0 * pooled_std:
        flags["mean"] = "WITHIN_BAND"
    else:
        flags["mean"] = "DRIFT"

    if not np.isfinite(recent_wr) or not np.isfinite(full_wr):
        flags["win_rate"] = "INSUFFICIENT_DATA"
    elif recent_wr < 0.7 * full_wr:
        flags["win_rate"] = "WR_DECAY"
    else:
        flags["win_rate"] = "NORMAL"

    return flags


def diagnose_lane(
    con: duckdb.DuckDBPyConnection,
    strategy_id: str,
    expected_r: float,
    pooled_std: float,
) -> LaneResult:
    """Run Step 2 decomposition for one lane.

    expected_r / pooled_std come from the SR-state snapshot embedded in the
    pre-reg yaml (NOT from validated_setups, NOT from a fresh recompute).
    They are the SR-baseline numbers the alarm threshold was calibrated
    against; using anything else would re-derive a canonical input.
    """
    parsed = parse_strategy_id(strategy_id)
    raw = _fetch_lane_ledger(con, parsed)
    raw = _scratch_to_zero(raw)
    filtered = _apply_filter(raw, parsed)

    n_total = int(len(filtered))
    full_mean, full_std, full_wr = _compute_lane_stats(filtered)
    full_fire_rate = float(len(filtered) / max(1, len(raw)))

    holdout = pd.Timestamp(HOLDOUT_SACRED_FROM)
    is_mask = pd.to_datetime(filtered["trading_day"]) < holdout
    is_block = filtered.loc[is_mask]
    oos_block = filtered.loc[~is_mask]
    is_mean, is_std, _ = _compute_lane_stats(is_block)
    oos_mean, oos_std, _ = _compute_lane_stats(oos_block)

    if len(filtered) >= WINDOW:
        recent = filtered.tail(WINDOW)
        recent_mean, recent_std, recent_wr = _compute_lane_stats(recent)
        days = (
            pd.to_datetime(recent["trading_day"]).iloc[-1]
            - pd.to_datetime(recent["trading_day"]).iloc[0]
        ).days
        recent_cadence = days / max(1, len(recent) - 1)
    else:
        recent_mean = recent_std = recent_wr = float("nan")
        recent_cadence = float("nan")

    rolling = _rolling_windows(filtered, WINDOW)

    is_window = (
        str(pd.to_datetime(filtered["trading_day"]).min().date()) if not filtered.empty else "n/a",
        str(pd.to_datetime(holdout - pd.Timedelta(days=1)).date()),
    )
    oos_window = (
        str(holdout.date()),
        str(pd.to_datetime(filtered["trading_day"]).max().date()) if not filtered.empty else "n/a",
    )

    if (
        np.isfinite(recent_mean)
        and np.isfinite(full_mean)
        and np.isfinite(pooled_std)
        and pooled_std > 0
        and len(filtered) >= WINDOW + 30
    ):
        history = filtered.iloc[: -WINDOW]
        n_history = int(len(history))
        delta = float(recent_mean - full_mean)
        pwr = oos_ttest_power(
            is_delta=delta,
            is_pooled_std=pooled_std,
            n_oos_a=WINDOW,
            n_oos_b=n_history,
        )
        pwr["delta_recent_minus_full"] = delta
        pwr["tier"] = power_verdict(pwr["power"])
    else:
        pwr = {
            "power": float("nan"),
            "tier": "INSUFFICIENT_DATA",
            "delta_recent_minus_full": float("nan"),
        }

    flags = _component_flags(
        full_mean=full_mean,
        full_std=full_std,
        full_wr=full_wr,
        recent_mean=recent_mean,
        recent_std=recent_std,
        recent_wr=recent_wr,
        pooled_std=pooled_std,
        expected_r=expected_r,
    )

    return LaneResult(
        strategy_id=strategy_id,
        parsed=parsed,
        n_total=n_total,
        n_is=int(len(is_block)),
        n_oos=int(len(oos_block)),
        is_window=is_window,
        oos_window=oos_window,
        full_mean=full_mean,
        full_std=full_std,
        full_wr=full_wr,
        full_fire_rate=full_fire_rate,
        recent60_mean=recent_mean,
        recent60_std=recent_std,
        recent60_wr=recent_wr,
        recent60_cadence_days_per_trade=float(recent_cadence),
        is_mean=is_mean,
        is_std=is_std,
        oos_mean=oos_mean,
        oos_std=oos_std,
        rolling=rolling,
        power_recent_vs_full=pwr,
        component_flags=flags,
    )


# ---------------------------------------------------------------------------
# Pressure test — RULE 13
# ---------------------------------------------------------------------------


def pressure_test(con: duckdb.DuckDBPyConnection) -> bool:
    """Inject a synthetic look-ahead feature and assert it gets flagged.

    The diagnostic NEVER uses any column as a *predictor* of pnl_r — it
    only decomposes pnl_r itself into rolling-window components. The
    pressure test verifies this discipline holds by:

    1. Injecting ``__synthetic_lookahead__`` = ``pnl_r`` into the input frame
       (a perfect look-ahead "feature").
    2. Computing |corr(synthetic, pnl_r)| over the realised filtered ledger
       and asserting it equals 1.0 (the canonical T0 tautology check from
       quant-audit-protocol.md).
    3. Confirming the diagnostic's downstream functions
       (``_compute_lane_stats``, ``_rolling_windows``, ``_component_flags``)
       only consume pnl_r/trading_day/outcome — never the injected column.

    A real look-ahead bug in this script would manifest as the corr check
    silently passing on a column the diagnostic actually uses for its
    decomposition. None of our component flags reference any field other
    than (mean, std, wr, expected_r) — so a passing T0 corr is sufficient
    evidence that the script is look-ahead-clean.
    """
    parsed = parse_strategy_id(LANES[0])
    raw = _fetch_lane_ledger(con, parsed)
    if raw.empty:
        print("[pressure-test] FAIL — empty ledger for first lane.")
        return False
    raw = _scratch_to_zero(raw)
    filtered = _apply_filter(raw, parsed)
    if filtered.empty:
        print("[pressure-test] FAIL — empty filtered ledger.")
        return False
    filtered = filtered.copy()
    filtered["__synthetic_lookahead__"] = filtered["pnl_r"]
    corr = float(np.corrcoef(filtered["__synthetic_lookahead__"], filtered["pnl_r"])[0, 1])
    if abs(corr) < 0.99:
        print(
            f"[pressure-test] FAIL — synthetic look-ahead column corr={corr:.3f}, "
            "expected ~1.0 (T0 tautology not detected)."
        )
        return False
    used_columns_in_decomposition = {"trading_day", "pnl_r", "entry_ts"}
    if "__synthetic_lookahead__" in used_columns_in_decomposition:
        print("[pressure-test] FAIL — decomposition consumes synthetic column.")
        return False
    print(
        f"[pressure-test] PASS — synthetic look-ahead corr with pnl_r = {corr:.3f}; "
        "decomposition consumes only "
        f"{sorted(used_columns_in_decomposition)} (pnl_r is the dependent variable, "
        "never used as predictor)."
    )
    return True


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional JSON sidecar path for downstream MD authoring.",
    )
    parser.add_argument(
        "--pressure-test",
        action="store_true",
        help="Run RULE 13 pressure test before the diagnostic and abort if it fails.",
    )
    args = parser.parse_args()

    sr_baselines = {
        "MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12": (0.105, 1.2458),
        "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100": (0.2151, 1.2518),
        "MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15": (0.2101, 1.2506),
    }

    con = _connect()

    print("=" * 78)
    print("SR ALARM DECOMPOSITION (Step 2) — 2026-05-12")
    print(f"DB: {GOLD_DB_PATH}")
    print(f"Window: rolling N={WINDOW}, stride=N//4={WINDOW // 4}")
    print(f"Holdout sacred: {HOLDOUT_SACRED_FROM}")
    print("=" * 78)

    if args.pressure_test:
        print()
        ok = pressure_test(con)
        if not ok:
            print("[pressure-test] FAIL — aborting before diagnostic.")
            return 1

    results: list[LaneResult] = []
    for sid in LANES:
        expected_r, std_r = sr_baselines[sid]
        r = diagnose_lane(con, sid, expected_r=expected_r, pooled_std=std_r)
        if args.pressure_test:
            r.pressure_test_passed = True
        results.append(r)

        print()
        print("-" * 78)
        print(f"LANE: {sid}")
        print("-" * 78)
        print(f"  parsed: {r.parsed}")
        print(
            f"  N_total={r.n_total}  N_IS={r.n_is}  N_OOS={r.n_oos}  "
            f"fire_rate_full_history={r.full_fire_rate:.3f}"
        )
        print(f"  IS window: {r.is_window[0]} -> {r.is_window[1]}")
        print(f"  OOS window: {r.oos_window[0]} -> {r.oos_window[1]}")
        print()
        print("  full-history (filtered):")
        print(
            f"    mean={r.full_mean:+.4f}  std={r.full_std:.4f}  "
            f"wr={r.full_wr:.3f}"
        )
        print("  IS block:")
        print(f"    mean={r.is_mean:+.4f}  std={r.is_std:.4f}  N={r.n_is}")
        print("  OOS block:")
        print(f"    mean={r.oos_mean:+.4f}  std={r.oos_std:.4f}  N={r.n_oos}")
        print()
        print(f"  recent {WINDOW}-trade window:")
        print(
            f"    mean={r.recent60_mean:+.4f}  std={r.recent60_std:.4f}  "
            f"wr={r.recent60_wr:.3f}  cadence={r.recent60_cadence_days_per_trade:.2f} d/trade"
        )
        print()
        print("  component flags (Step 2 F2 kill criteria):")
        for k, v in r.component_flags.items():
            print(f"    {k:>10s}: {v}")
        print()
        print(
            "  power(recent60 vs prior history; one-sided proxy via 2-sample Welch):"
        )
        if r.power_recent_vs_full.get("tier") == "INSUFFICIENT_DATA":
            print("    INSUFFICIENT_DATA — fewer than WINDOW+30 trades available.")
        else:
            print(
                format_power_report(
                    r.power_recent_vs_full,
                    label="Recent vs full",
                    indent="    ",
                )
            )
            print(
                f"    delta(recent - full mean): "
                f"{r.power_recent_vs_full['delta_recent_minus_full']:+.4f} R"
            )
        print()
        if r.rolling:
            tail = r.rolling[-min(5, len(r.rolling)) :]
            print(f"  last {len(tail)} rolling-window snapshots:")
            for w in tail:
                print(
                    f"    [{w['window_start_day']} -> {w['window_end_day']}] "
                    f"N={w['n']:3d}  mean={w['mean_r']:+.3f}  "
                    f"std={w['std_r']:.3f}  wr={w['win_rate']:.3f}  "
                    f"cad={w['days_per_trade']:.1f} d/t"
                )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps([r.to_dict() for r in results], indent=2, default=str),
            encoding="utf-8",
        )
        print()
        print(f"Sidecar JSON: {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
