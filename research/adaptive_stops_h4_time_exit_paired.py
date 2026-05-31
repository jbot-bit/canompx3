"""H4 — time-based no-progress exit, paired-deltaR vs the incumbent price stop.

READ-ONLY hypothesis reader executing the LOCKED H4 pre-reg
``docs/audit/hypotheses/2026-05-31-adaptive-stops-h4-time-based-exit-stack-v1.yaml``
(Pathway B, K=1). Writes NOTHING to ``experimental_strategies`` /
``validated_setups`` / any DB table. Opens DuckDB ``read_only=True``.

The question (pre-reg ``decision_rule``)
----------------------------------------
For each deployed E2 lane, score every traded (post-filter) trade under TWO exit
rules, paired on the same trade:

  Arm A (baseline) — the incumbent price-based stop. This is the trade's already
    -realized ``pnl_r`` from ``orb_outcomes`` (opposite-ORB-boundary stop). No
    re-simulation.
  Arm B (time-exit) — a time-based no-progress exit: if the trade has NOT reached
    >= +0.5R favorable MFE within 120s (wall-clock) of ``entry_ts``, exit at the
    window-close path price (path-R at entry+120s); otherwise the trade's existing
    exit stands (B == A).

Report per-lane and pooled realized mean ``pnl_r`` delta (B - A), IS and OOS, with
the OOS slice carrying its one-sample power tier (RULE 3.3) and a per-year ΔR
breakdown (pooled-finding-rule / RULE 12 regime-concentration flag).

Why bars_1m and not ``mfe_r`` (pre-reg data-contract amendment 2026-05-31)
--------------------------------------------------------------------------
``orb_outcomes.mfe_r`` is a MAGNITUDE — it says a trade eventually reached, e.g.,
+1.2R, but NOT *when*. H4's rule is a TIMING condition ("+0.5R within 120s"), so
the magnitude column cannot answer it. The first-2-bar (≈120s on 1m bars) intra
-trade path is reconstructed from canonical ``bars_1m``, anchored on the actual
E2 fill ``entry_ts`` (NOT any break-bar timestamp — backtesting-methodology §6.3:
the E2 stop-market fill precedes the daily_features close-outside break bar ~41%
of the time). This is a READ — no new sim engine, no DB write.

Canonical delegation (institutional-rigor §4 — re-encodes nothing)
------------------------------------------------------------------
- Deployed-lane source:     ``trading_app.prop_profiles.ACCOUNT_PROFILES``
- Lane-dimension resolver:  ``trading_app.account_survival._load_strategy_snapshot``
- Filtered outcome loader:  ``trading_app.strategy_fitness._load_strategy_outcomes``
- IS boundary:              ``trading_app.holdout_policy.HOLDOUT_SACRED_FROM``
- Cost / friction:          ``pipeline.cost_model.get_cost_spec`` (BOTH arms)
- OOS power tiers:          ``research.oos_power.one_sample_power`` / ``power_verdict``
- DB path:                  ``pipeline.paths.GOLD_DB_PATH``

mae_r / mfe_r are RULE 6.3-banned as backtest PREDICTORS. They are NOT read here.
The time condition is reconstructed from the bars_1m OHLC path, and no value
enters any rule that SELECTS a trade or promotes a strategy — this reader makes a
RELATIVE measurement (B - A) on already-taken trades only.

Run gate
--------
The pre-reg is LOCKED but carries ``do_not_run_until_committed: true``. This
script refuses to run unless the LOCKED pre-reg file is committed (clean in git),
mirroring H0's ``--allow-draft`` operator escape for dry-runs.

Usage
-----
    python research/adaptive_stops_h4_time_exit_paired.py            # MNQ+MES
    python research/adaptive_stops_h4_time_exit_paired.py --instrument MNQ
    python research/adaptive_stops_h4_time_exit_paired.py --csv out.csv --md out.md
    python research/adaptive_stops_h4_time_exit_paired.py --allow-uncommitted   # dry-run
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb

# Windows console defaults to cp1252, which cannot encode the result glyphs
# (Δ, ≈). Reconfigure stdout/stderr to UTF-8 so the report prints on Windows;
# fail-open if the stream does not support reconfigure (e.g. piped/captured).
for _stream in (sys.stdout, sys.stderr):
    _reconfig = getattr(_stream, "reconfigure", None)
    if _reconfig is not None:
        try:
            _reconfig(encoding="utf-8")
        except (ValueError, OSError):
            pass

from pipeline.cost_model import get_cost_spec
from pipeline.paths import GOLD_DB_PATH
from research.oos_power import one_sample_power, power_verdict
from trading_app.account_survival import _load_strategy_snapshot
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
from trading_app.prop_profiles import ACCOUNT_PROFILES
from trading_app.strategy_fitness import _load_strategy_outcomes

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Pre-committed parameters — FROZEN by the LOCKED pre-reg (Howard extract line
# 188). NOT swept. The `forbidden` list in the pre-reg (no parameter_sweep /
# grid_search / multi_window_argmax) is what keeps K=1 honest; these are module
# constants, never CLI args, so they cannot be tuned from the command line.
# ---------------------------------------------------------------------------
MFE_PROGRESS_THRESHOLD_R: float = 0.5      # +0.5R favorable MFE
NO_PROGRESS_WINDOW_SECONDS: int = 120      # 120s wall-clock from entry_ts
BAR_SECONDS: int = 60                      # bars_1m is 1-minute OHLCV

# Deployed entry geometry is E2 for every deployed lane (pre-reg scope.entry_model).
ENTRY_MODEL: str = "E2"

# Instruments in scope. MGC is EXCLUDED by the pre-reg
# (howard_illegitimate_uses_guard: "NOT an MGC basis ... MES/MNQ deployed lanes
# only"), so even though prop_profiles carries a deployed MGC lane it is dropped.
SCOPE_INSTRUMENTS: frozenset[str] = frozenset({"MNQ", "MES"})

# N>=50 traded-trade floor for a per-lane verdict (pre-reg decision_rule
# .insufficient_n + backtesting-methodology N>=50). Below this: descriptive only.
MIN_N_FOR_VERDICT: int = 50

LOCKED_PREREG_REL = (
    "docs/audit/hypotheses/2026-05-31-adaptive-stops-h4-time-based-exit-stack-v1.yaml"
)


@dataclass
class TradeScore:
    """Per-trade paired score (B - A). Pure measurement; never a selector."""

    trading_day: date
    split: str  # IS | OOS
    direction: str  # long | short
    baseline_r: float  # Arm A — booked pnl_r (incumbent price stop)
    timeexit_r: float  # Arm B — time-based no-progress exit realized R
    delta_r: float  # B - A
    reached_progress: bool  # did MFE hit +0.5R within the window?


@dataclass
class LaneReport:
    strategy_id: str
    instrument: str
    orb_label: str
    orb_minutes: int
    rr_target: float
    confirm_bars: int
    filter_type: str
    entry_model: str = ENTRY_MODEL
    n_is: int = 0
    n_oos: int = 0
    dropped_incomplete_window: int = 0  # bars window missing/partial — never scored
    mean_baseline_is: float | None = None
    mean_timeexit_is: float | None = None
    mean_delta_is: float | None = None
    mean_baseline_oos: float | None = None
    mean_timeexit_oos: float | None = None
    mean_delta_oos: float | None = None
    oos_power_tier: str | None = None
    progress_rate_is: float | None = None  # share reaching +0.5R within window (IS)
    year_delta: dict[int, dict] = field(default_factory=dict)
    year_consistency: str | None = None  # CONSISTENT | SIGN_INCONSISTENT | THIN
    verdict: str = "INSUFFICIENT_N"
    notes: str = ""


# Per-year N floor for the regime-concentration flag (RULE 3.2: N<30 -> low power).
YEAR_MIN_N: int = 30


# ---------------------------------------------------------------------------
# bars_1m intra-trade path reconstruction (the only new logic; a READ).
# ---------------------------------------------------------------------------
def _as_utc(ts) -> datetime | None:
    """Normalize a DB timestamp to a tz-aware UTC datetime. None-safe."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
    # duckdb may hand back numpy datetime64 / pandas Timestamp; .to_pydatetime exists on the latter.
    to_py = getattr(ts, "to_pydatetime", None)
    if to_py is not None:
        dt = to_py()
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    return None


def _load_window_bars(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
    entry_ts: datetime,
    window_end: datetime,
) -> list[tuple]:
    """OHLC bars whose ts_utc covers [entry-bar .. window_end).

    The entry-bar is the bar whose ts_utc is the latest bar start <= entry_ts.
    We fetch from one bar-width before entry_ts (to guarantee we capture the
    entry-bar even if entry_ts is mid-bar) up to but excluding window_end.
    """
    lo = entry_ts - timedelta(seconds=BAR_SECONDS)
    return con.execute(
        """
        SELECT ts_utc, open, high, low, close
        FROM bars_1m
        WHERE symbol = ?
          AND ts_utc >= ?
          AND ts_utc < ?
        ORDER BY ts_utc
        """,
        [instrument, lo, window_end],
    ).fetchall()


def _time_exit_r(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
    trade: dict,
) -> tuple[float, bool, str] | None:
    """Arm B realized R for one trade via the bars_1m path. Returns
    (timeexit_r, reached_progress, direction) or None if the window is incomplete.

    Convention (user-confirmed): path-R at window close. If +0.5R favorable MFE
    is reached within the window, the trade's existing exit stands (B == A). If
    NOT, the time-exit books the path-R at the close of the window-boundary bar,
    with the SAME round-turn friction the baseline paid (cost-neutral).
    """
    entry_ts = _as_utc(trade.get("entry_ts"))
    entry_price = trade.get("entry_price")
    stop_price = trade.get("stop_price")
    baseline_r = trade.get("pnl_r")
    if entry_ts is None or entry_price is None or stop_price is None or baseline_r is None:
        return None

    risk_per_r = abs(float(entry_price) - float(stop_price))
    if risk_per_r <= 0:
        return None
    # Direction from geometry: long => stop below entry; short => stop above.
    direction = "long" if float(stop_price) < float(entry_price) else "short"

    window_end = entry_ts + timedelta(seconds=NO_PROGRESS_WINDOW_SECONDS)
    bars = _load_window_bars(con, instrument, entry_ts, window_end)
    if not bars:
        return None

    # Identify the entry-bar: latest bar whose start ts_utc <= entry_ts.
    entry_bar_idx = None
    for i, b in enumerate(bars):
        b_ts = _as_utc(b[0])
        if b_ts is not None and b_ts <= entry_ts:
            entry_bar_idx = i
    if entry_bar_idx is None:
        return None  # no bar at/before entry_ts in the fetched window

    window_bars = bars[entry_bar_idx:]
    if not window_bars:
        return None
    # Require a complete window: at least the entry bar plus the next bar so that
    # 120s of wall-clock is actually covered (1m bars => 2 bars). A partial window
    # is DROPPED and counted — never silently treated as "no progress".
    expected_bars = max(1, NO_PROGRESS_WINDOW_SECONDS // BAR_SECONDS)
    if len(window_bars) < expected_bars:
        return None

    ep = float(entry_price)
    # Favorable MFE within the window (long: high above entry; short: entry above low).
    if direction == "long":
        best_favorable = max(float(b[2]) for b in window_bars) - ep  # max high - entry
    else:
        best_favorable = ep - min(float(b[3]) for b in window_bars)  # entry - min low
    mfe_r = best_favorable / risk_per_r
    reached_progress = mfe_r >= MFE_PROGRESS_THRESHOLD_R

    if reached_progress:
        # Trade made the cut — its real exit stands. B == A.
        return float(baseline_r), True, direction

    # No progress: exit at the window-close path price (close of the last
    # window bar), converted to R, with identical round-turn friction.
    window_close = float(window_bars[-1][4])
    if direction == "long":
        gross_r = (window_close - ep) / risk_per_r
    else:
        gross_r = (ep - window_close) / risk_per_r

    cost_spec = get_cost_spec(instrument)
    # Friction in R: round-turn friction dollars / (risk distance in points * point value).
    # risk_dollars for 1R is recoverable from the booked trade; prefer it when present
    # to keep the R-scale identical to how pnl_r was computed.
    risk_dollars = trade.get("risk_dollars")
    if risk_dollars and float(risk_dollars) > 0:
        friction_r = cost_spec.total_friction / float(risk_dollars)
    else:
        # Fallback: risk distance (points) * canonical point value (dollars/point).
        risk_dollars_calc = risk_per_r * cost_spec.point_value
        friction_r = cost_spec.total_friction / risk_dollars_calc if risk_dollars_calc > 0 else 0.0
    net_r = gross_r - friction_r
    return net_r, False, direction


# ---------------------------------------------------------------------------
# Per-lane scoring + stats.
# ---------------------------------------------------------------------------
def _mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def _score_lane(con: duckdb.DuckDBPyConnection, snapshot: dict) -> LaneReport:
    instrument = snapshot["instrument"]
    common = dict(
        con=con,
        instrument=instrument,
        orb_label=snapshot["orb_label"],
        orb_minutes=int(snapshot["orb_minutes"]),
        entry_model=ENTRY_MODEL,
        rr_target=float(snapshot["rr_target"]),
        confirm_bars=int(snapshot["confirm_bars"]),
        filter_type=snapshot["filter_type"],
    )
    is_out = _load_strategy_outcomes(**common, end_date=HOLDOUT_SACRED_FROM - timedelta(days=1))
    oos_out = _load_strategy_outcomes(**common, start_date=HOLDOUT_SACRED_FROM)

    rep = LaneReport(
        strategy_id=snapshot["strategy_id"],
        instrument=instrument,
        orb_label=snapshot["orb_label"],
        orb_minutes=int(snapshot["orb_minutes"]),
        rr_target=float(snapshot["rr_target"]),
        confirm_bars=int(snapshot["confirm_bars"]),
        filter_type=snapshot["filter_type"],
    )

    is_scores = _score_trades(con, instrument, is_out, "IS", rep)
    oos_scores = _score_trades(con, instrument, oos_out, "OOS", rep)
    rep.n_is = len(is_scores)
    rep.n_oos = len(oos_scores)

    if is_scores:
        rep.mean_baseline_is = _mean([s.baseline_r for s in is_scores])
        rep.mean_timeexit_is = _mean([s.timeexit_r for s in is_scores])
        rep.mean_delta_is = _mean([s.delta_r for s in is_scores])
        rep.progress_rate_is = _mean([1.0 if s.reached_progress else 0.0 for s in is_scores])
        rep.year_delta = _year_delta(is_scores)
        rep.year_consistency = _year_consistency(rep.year_delta)
    if oos_scores:
        rep.mean_baseline_oos = _mean([s.baseline_r for s in oos_scores])
        rep.mean_timeexit_oos = _mean([s.timeexit_r for s in oos_scores])
        rep.mean_delta_oos = _mean([s.delta_r for s in oos_scores])
        rep.oos_power_tier = _oos_power_tier(is_scores, len(oos_scores))

    _assign_verdict(rep)
    return rep


def _score_trades(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
    outcomes: list[dict],
    split: str,
    rep: LaneReport,
) -> list[TradeScore]:
    scores: list[TradeScore] = []
    for o in outcomes:
        if o.get("outcome") not in ("win", "loss"):
            continue  # resolved trades only (mirror H0); scratches excluded
        result = _time_exit_r(con, instrument, o)
        if result is None:
            rep.dropped_incomplete_window += 1
            continue
        timeexit_r, reached, direction = result
        baseline_r = float(o["pnl_r"])
        td = o.get("trading_day")
        td_date = td.date() if hasattr(td, "date") else td
        scores.append(
            TradeScore(
                trading_day=td_date,
                split=split,
                direction=direction,
                baseline_r=baseline_r,
                timeexit_r=timeexit_r,
                delta_r=timeexit_r - baseline_r,
                reached_progress=reached,
            )
        )
    return scores


def _year_delta(scores: list[TradeScore]) -> dict[int, dict]:
    by_year: dict[int, list[float]] = {}
    for s in scores:
        yr = getattr(s.trading_day, "year", None)
        if yr is None:
            continue
        by_year.setdefault(yr, []).append(s.delta_r)
    return {yr: {"n": len(v), "delta": sum(v) / len(v)} for yr, v in by_year.items()}


def _year_consistency(year_delta: dict[int, dict]) -> str | None:
    if not year_delta:
        return None
    powered = [d["delta"] for d in year_delta.values() if d["n"] >= YEAR_MIN_N]
    if len(powered) < 2:
        return "THIN"
    signs = {(1 if d > 1e-9 else -1 if d < -1e-9 else 0) for d in powered} - {0}
    return "SIGN_INCONSISTENT" if len(signs) > 1 else "CONSISTENT"


def _oos_power_tier(is_scores: list[TradeScore], n_oos: int) -> str | None:
    """One-sample power tier for the IS paired-ΔR effect at the OOS N (RULE 3.3).

    cohen_d = |t_IS| / sqrt(N_IS), where t is the paired-ΔR one-sample t against 0.
    """
    vals = [s.delta_r for s in is_scores]
    if len(vals) < 2 or n_oos < 2:
        return None
    mean_d = sum(vals) / len(vals)
    std_d = statistics.pstdev(vals)
    if std_d <= 0:
        return None
    t_is = abs(mean_d) * (len(vals) ** 0.5) / std_d
    cohen_d = t_is / (len(vals) ** 0.5)
    return power_verdict(one_sample_power(cohen_d, n_oos))


def _assign_verdict(rep: LaneReport) -> None:
    """Per-lane PROCEED / KILL / INSUFFICIENT_N on the IS paired-ΔR.

    PROCEED  — IS mean ΔR strictly > 0 at N_IS >= 50 (time-exit measurably less
               value-destroying than the price stop on this lane). Routes to
               heavyweight Chordia + independent positive-EV validation, NEVER to
               capital (edge_claim_summary "less bad, not good").
    KILL     — IS mean ΔR <= 0 at N_IS >= 50 (time-exit monotone-non-improving).
    INSUFFICIENT_N — N_IS < 50; descriptive only, no verdict.

    Regime-concentration is ANNOTATED (year_consistency), never flips the verdict.
    OOS never kills on its own (oos_handling: sign change is refutation only when
    power tier == CAN_REFUTE).
    """
    if rep.n_is < MIN_N_FOR_VERDICT or rep.mean_delta_is is None:
        rep.verdict = "INSUFFICIENT_N"
        rep.notes = f"N_IS={rep.n_is} < {MIN_N_FOR_VERDICT} floor; descriptive only."
        return
    regime = ""
    if rep.year_consistency == "SIGN_INCONSISTENT":
        regime = (" CAVEAT: per-year ΔR sign INCONSISTENT across powered years — "
                  "any positive pooled ΔR may be regime-concentrated (pooled-finding-rule).")
    if rep.mean_delta_is > 0:
        rep.verdict = "PROCEED"
        rep.notes = (
            f"IS mean ΔR={rep.mean_delta_is:+.4f} > 0 (time-exit less "
            f"value-destroying than price stop). Routes to heavyweight Chordia + "
            f"independent positive-EV validation, NOT capital. "
            f"progress-rate(IS)={rep.progress_rate_is:.1%}, "
            f"year-consistency={rep.year_consistency}.{regime}"
        )
    else:
        rep.verdict = "KILL"
        rep.notes = (
            f"IS mean ΔR={rep.mean_delta_is:+.4f} <= 0 (time-exit non-improving vs "
            f"price stop). Single-layer no-progress exit dead for this lane; does "
            f"NOT kill the full 7-layer Howard stack. "
            f"progress-rate(IS)={rep.progress_rate_is:.1%}, "
            f"year-consistency={rep.year_consistency}.{regime}"
        )


# ---------------------------------------------------------------------------
# Deployed-lane derivation (volatile-data rule: re-derive from prop_profiles).
# ---------------------------------------------------------------------------
def _deployed_strategy_ids(instrument_filter: str | None) -> list[str]:
    seen: dict[str, None] = {}
    for profile in ACCOUNT_PROFILES.values():
        for lane in profile.daily_lanes:
            inst = lane.instrument
            if inst not in SCOPE_INSTRUMENTS:
                continue  # MGC and any non-MNQ/MES excluded by pre-reg scope
            if instrument_filter and inst != instrument_filter:
                continue
            seen.setdefault(lane.strategy_id, None)
    return list(seen.keys())


# ---------------------------------------------------------------------------
# Pooled (cross-lane) summary — reported WITH the per-lane table, never alone.
# ---------------------------------------------------------------------------
def _pooled(reports: list[LaneReport]) -> dict:
    """Trade-count-weighted pooled IS ΔR + flip rate (pooled-finding-rule)."""
    verdict_lanes = [r for r in reports if r.n_is >= MIN_N_FOR_VERDICT and r.mean_delta_is is not None]
    if not verdict_lanes:
        return {"n_lanes": 0}
    total_n = sum(r.n_is for r in verdict_lanes)
    pooled_delta = sum(r.mean_delta_is * r.n_is for r in verdict_lanes) / total_n
    pooled_sign = 1 if pooled_delta > 0 else -1 if pooled_delta < 0 else 0
    flips = sum(
        1 for r in verdict_lanes
        if (1 if r.mean_delta_is > 0 else -1 if r.mean_delta_is < 0 else 0) not in (pooled_sign, 0)
    )
    flip_rate = 100.0 * flips / len(verdict_lanes)
    return {
        "n_lanes": len(verdict_lanes),
        "total_n_is": total_n,
        "pooled_delta_is": pooled_delta,
        "flip_rate_pct": flip_rate,
        "n_proceed": sum(1 for r in verdict_lanes if r.verdict == "PROCEED"),
        "n_kill": sum(1 for r in verdict_lanes if r.verdict == "KILL"),
    }


# ---------------------------------------------------------------------------
# Output.
# ---------------------------------------------------------------------------
def _fmt(v, spec="+.4f") -> str:
    return format(v, spec) if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)) else "n/a"


def _print_report(reports: list[LaneReport], pooled: dict) -> None:
    print("\n=== H4 TIME-BASED NO-PROGRESS EXIT — paired-ΔR (READ-ONLY, K=1) ===")
    print(f"IS boundary (HOLDOUT_SACRED_FROM): {HOLDOUT_SACRED_FROM}")
    print(f"Pre-committed: +{MFE_PROGRESS_THRESHOLD_R}R MFE within {NO_PROGRESS_WINDOW_SECONDS}s "
          f"(no sweep). N floor for verdict: {MIN_N_FOR_VERDICT}\n")
    for r in reports:
        print(f"--- {r.strategy_id}  [{r.entry_model}] ---")
        print(f"    N_IS={r.n_is}  N_OOS={r.n_oos}  dropped(incomplete window)={r.dropped_incomplete_window}")
        print(f"    IS  mean: A(price)={_fmt(r.mean_baseline_is)}  B(time)={_fmt(r.mean_timeexit_is)}  "
              f"ΔR={_fmt(r.mean_delta_is)}")
        if r.mean_delta_oos is not None:
            print(f"    OOS mean: A(price)={_fmt(r.mean_baseline_oos)}  B(time)={_fmt(r.mean_timeexit_oos)}  "
                  f"ΔR={_fmt(r.mean_delta_oos)}  power_tier={r.oos_power_tier}")
        if r.year_delta:
            yr = "  ".join(f"{y}:{d['delta']:+.4f}(n={d['n']})" for y, d in sorted(r.year_delta.items()))
            print(f"    per-year ΔR: {yr}   consistency={r.year_consistency}")
        print(f"    VERDICT: {r.verdict}")
        print(f"    {r.notes}\n")
    if pooled.get("n_lanes"):
        print("=== POOLED (reported WITH per-lane table, never alone) ===")
        print(f"    lanes with verdict: {pooled['n_lanes']}  total N_IS: {pooled['total_n_is']}")
        print(f"    pooled IS ΔR (N-weighted): {_fmt(pooled['pooled_delta_is'])}  "
              f"flip-rate: {pooled['flip_rate_pct']:.1f}%")
        print(f"    PROCEED lanes: {pooled['n_proceed']}  KILL lanes: {pooled['n_kill']}")
        if pooled["flip_rate_pct"] >= 25:
            print("    HETEROGENEITY: flip-rate >= 25% — pooled framing misleads; read per-lane.")
    print()


def _write_csv(reports: list[LaneReport], path: Path) -> None:
    cols = [
        "strategy_id", "instrument", "orb_label", "orb_minutes", "rr_target",
        "confirm_bars", "filter_type", "entry_model", "n_is", "n_oos",
        "dropped_incomplete_window", "mean_baseline_is", "mean_timeexit_is",
        "mean_delta_is", "mean_baseline_oos", "mean_timeexit_oos", "mean_delta_oos",
        "oos_power_tier", "progress_rate_is", "year_consistency", "verdict", "notes",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in reports:
            w.writerow([getattr(r, c) for c in cols])
    print(f"Wrote per-lane CSV: {path}")


def _repo_rel(path: Path) -> str:
    """Repo-relative POSIX path; falls back to the basename if outside ROOT."""
    resolved = path if path.is_absolute() else (Path.cwd() / path)
    try:
        return resolved.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def _write_md(reports: list[LaneReport], pooled: dict, path: Path, csv_path: Path | None) -> None:
    flip = pooled.get("flip_rate_pct", 0.0)
    # The per-cell breakdown is the companion CSV (one row per lane); point the
    # pooled-finding-rule frontmatter at it. Fall back to this MD if no CSV.
    breakdown = _repo_rel(csv_path) if csv_path is not None else _repo_rel(path)
    lines = [
        "---",
        "pooled_finding: true",
        f"per_cell_breakdown_path: {breakdown}",
        f"flip_rate_pct: {flip:.1f}",
    ]
    if flip >= 25:
        lines.append("heterogeneity_ack: true")
    lines += [
        "---",
        "",
        "# H4 — Time-Based No-Progress Exit (paired-ΔR vs price stop)",
        "",
        "## Pre-registration",
        "",
        f"LOCKED: `{LOCKED_PREREG_REL}` (Pathway B, K=1). Pre-committed parameter "
        f"(+{MFE_PROGRESS_THRESHOLD_R}R MFE within {NO_PROGRESS_WINDOW_SECONDS}s), no sweep.",
        "",
        "## Method",
        "",
        "- Canonical layers: `bars_1m` (intra-trade MFE timing) + `orb_outcomes` "
        "(booked baseline, via `_load_strategy_outcomes`).",
        "- Arm A: incumbent price stop = booked `pnl_r`. Arm B: time-based "
        f"no-progress exit (path-R at entry+{NO_PROGRESS_WINDOW_SECONDS}s if MFE "
        f"< +{MFE_PROGRESS_THRESHOLD_R}R within window; else existing exit stands).",
        "- Window anchored on E2 fill `entry_ts` (NOT break-bar — §6.3). Cost "
        "applied identically to both arms (cost-neutral).",
        "- Read-only; no DB write; MGC excluded by pre-reg scope.",
        "",
        "## Per-lane results",
        "",
        "| Lane | N_IS | N_OOS | A(price) IS | B(time) IS | ΔR IS | ΔR OOS | OOS power | year-consist | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for r in reports:
        lines.append(
            f"| {r.strategy_id} | {r.n_is} | {r.n_oos} | {_fmt(r.mean_baseline_is)} | "
            f"{_fmt(r.mean_timeexit_is)} | {_fmt(r.mean_delta_is)} | {_fmt(r.mean_delta_oos)} | "
            f"{r.oos_power_tier or 'n/a'} | {r.year_consistency or 'n/a'} | {r.verdict} |"
        )
    lines += ["", "## Pooled summary", ""]
    if pooled.get("n_lanes"):
        lines += [
            f"- Lanes with verdict (N_IS>={MIN_N_FOR_VERDICT}): {pooled['n_lanes']}",
            f"- Pooled IS ΔR (N-weighted): {_fmt(pooled['pooled_delta_is'])}",
            f"- Flip-rate: {pooled['flip_rate_pct']:.1f}%"
            + (" — HETEROGENEITY: pooled framing misleads, read per-lane." if flip >= 25 else ""),
            f"- PROCEED lanes: {pooled['n_proceed']}  |  KILL lanes: {pooled['n_kill']}",
        ]
    else:
        lines.append("- No lane met the N_IS>=50 floor; descriptive only.")
    lines += [
        "",
        "## Classification use",
        "",
        "PROCEED = the time-exit is measurably *less value-destroying* than the "
        "price stop on that lane (Howard \"less bad, not good\"). It routes to "
        "heavyweight Chordia review + independent positive-EV validation, NEVER to "
        "capital. KILL = single-layer no-progress exit dead for these lanes (does "
        "NOT kill the full 7-layer Howard stack — untested here).",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote result MD: {path}")


# ---------------------------------------------------------------------------
# Run gate + main.
# ---------------------------------------------------------------------------
def _prereg_committed() -> bool:
    """True iff the LOCKED pre-reg exists and is clean (committed) in git."""
    prereg = ROOT / LOCKED_PREREG_REL
    if not prereg.exists():
        return False
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--", LOCKED_PREREG_REL],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if out.returncode != 0:
        return False
    return out.stdout.strip() == ""  # no porcelain line => clean/committed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instrument", choices=["MNQ", "MES"], default=None,
                    help="Restrict to one instrument (MGC excluded by pre-reg scope).")
    ap.add_argument("--csv", default=None, help="Write per-lane CSV to this path.")
    ap.add_argument("--md", default=None, help="Write result MD to this path.")
    ap.add_argument("--allow-uncommitted", action="store_true",
                    help="Operator dry-run before the LOCKED pre-reg is committed.")
    args = ap.parse_args(argv)

    if not args.allow_uncommitted and not _prereg_committed():
        print(
            "REFUSING TO RUN: the LOCKED H4 pre-reg "
            f"({LOCKED_PREREG_REL}) is not committed/clean. The pre-reg's "
            "do_not_run_until_committed gate bars any run until the promotion "
            "commit lands. Operator dry-run: --allow-uncommitted.",
            file=sys.stderr,
        )
        return 2

    strategy_ids = _deployed_strategy_ids(args.instrument)
    if not strategy_ids:
        print("No deployed MNQ/MES E2 lanes matched.", file=sys.stderr)
        return 1

    try:
        con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    except duckdb.IOException as exc:
        print(
            f"ABORT: cannot open {GOLD_DB_PATH} read-only — a writer holds the "
            f"file. Run when no writer is active, or reap stale handles "
            f"(scripts/tools/reap_stale_claude_processes.py --apply). DuckDB said: {exc}",
            file=sys.stderr,
        )
        return 1
    try:
        reports: list[LaneReport] = []
        for sid in strategy_ids:
            try:
                snapshot = _load_strategy_snapshot(con, sid)
            except ValueError as exc:
                print(f"  SKIP {sid}: {exc}", file=sys.stderr)
                continue
            reports.append(_score_lane(con, snapshot))
    finally:
        con.close()

    reports.sort(key=lambda r: r.strategy_id)
    pooled = _pooled(reports)
    _print_report(reports, pooled)
    if args.csv:
        _write_csv(reports, Path(args.csv))
    if args.md:
        _write_md(reports, pooled, Path(args.md), Path(args.csv) if args.csv else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
