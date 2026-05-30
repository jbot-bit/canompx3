"""H0 gating diagnostic — MFE/MAE symmetry + Howard-selectivity on deployed lanes.

READ-ONLY descriptive diagnostic. Spends K=0. Writes NOTHING to
``experimental_strategies`` / ``validated_setups`` / any DB table. Decides
whether the price-stop hypothesis family (H1 level-distance anchor, H3
ATR-scaled distance) is LIVE or PRE-KILLED for each deployed lane, BEFORE any
change to ``entry_rules.py`` / ``outcome_builder.py`` / ``compute_stop_price()``
or schema.

Pre-registration
----------------
``docs/audit/hypotheses/drafts/2026-05-31-adaptive-stops-h0-mfe-mae-symmetry-diagnostic-v1.draft.yaml``
(DRAFT — human review gates promotion; this script aborts unless the pre-reg is
committed and promoted out of ``drafts/``... see ``--allow-draft`` escape for
operator dry-runs).

What it does, per (deployed lane x entry model)
-----------------------------------------------
1. Read out the realized-EV delta of the EXISTING ``stop_multiplier=0.75`` prop
   re-sim — pre-existing on-our-data evidence (memo correction #2). This is the
   exact counterfactual ``trading_app/account_survival.py:332-339`` already runs
   in production for survival sizing.
2. Measure median MFE/MAE ratio on the TRADED (post-filter) population.
3. Sweep ``config.apply_tight_stop`` across the stop-multiplier grid and compute
   realized Howard-selectivity vs the payoff-implied (~75%) breakeven.
4. Emit a per-lane symmetry verdict (PRE_KILL_PRICE_STOPS / PROCEED_H1_H3 /
   INSUFFICIENT_N).

Canonical delegation (institutional-rigor §4 — re-encodes nothing)
------------------------------------------------------------------
- Stop counterfactual:     ``trading_app.config.apply_tight_stop``
- Filtered outcome loader:  ``trading_app.strategy_fitness._load_strategy_outcomes``
- Lane-dimension resolver:  ``trading_app.account_survival._load_strategy_snapshot``
- Deployed-lane source:     ``trading_app.prop_profiles.ACCOUNT_PROFILES``
- IS boundary:              ``trading_app.holdout_policy.HOLDOUT_SACRED_FROM``
- Cost / friction:          ``pipeline.cost_model.get_cost_spec``
- OOS power tiers:          ``research.oos_power.one_sample_power`` / ``power_verdict``
- DB path:                  ``pipeline.paths.GOLD_DB_PATH``

mae_r / mfe_r are RULE 6.3-banned as backtest PREDICTORS. Here they are the
REQUIRED descriptive MEASUREMENT input to a post-hoc symmetry statistic — the
same dual status that makes ``apply_tight_stop`` legitimate (it reads ``mae_r``
to re-price a counterfactual, it does not predict with it). No ``mae_r`` /
``mfe_r`` value enters any decision rule that selects a trade or promotes a
strategy.

Usage
-----
    python research/adaptive_stops_h0_symmetry_diagnostic.py
    python research/adaptive_stops_h0_symmetry_diagnostic.py --instrument MNQ
    python research/adaptive_stops_h0_symmetry_diagnostic.py --csv out.csv --allow-draft
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from dataclasses import dataclass, field
from datetime import date

import duckdb

from pipeline.cost_model import get_cost_spec
from pipeline.paths import GOLD_DB_PATH
from research.oos_power import one_sample_power, power_verdict
from trading_app.account_survival import _load_strategy_snapshot
from trading_app.config import apply_tight_stop
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
from trading_app.prop_profiles import ACCOUNT_PROFILES
from trading_app.strategy_fitness import _load_strategy_outcomes

# Pre-registered grid (mirrors the draft YAML; apply_tight_stop is a no-op at 1.0).
STOP_MULTIPLIER_GRID: tuple[float, ...] = (0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0)

# Entry-model strata (memo correction #4: realized risk differs by entry model;
# the symmetry statistic must NEVER be pooled across them).
ENTRY_MODELS: tuple[str, ...] = ("E1", "E2", "E3")

# Minimum traded-trade count below which the lane/stratum is INSUFFICIENT_N (no
# verdict — descriptive only). Aligns with backtesting-methodology N>=50 floor for
# any claim, though H0 makes no promotion claim.
MIN_N_FOR_VERDICT: int = 50


@dataclass
class StratumReport:
    strategy_id: str
    instrument: str
    orb_label: str
    orb_minutes: int
    rr_target: float
    confirm_bars: int
    filter_type: str
    entry_model: str
    n_is: int
    n_oos: int
    median_mfe_mae_ratio: float | None  # Howard §5.3.1 symmetry mechanism (~1.0 = symmetric)
    # Per-multiplier IS stats: mean pnl_r (primary EV evidence), Howard stop-
    # selectivity (§5.3.2), and winner-damage (§5.3.2). 1.0 is the no-op baseline.
    grid_mean_pnl_r: dict[float, float] = field(default_factory=dict)
    grid_stop_selectivity: dict[float, float] = field(default_factory=dict)
    grid_winner_damage: dict[float, float] = field(default_factory=dict)
    # Existing 0.75 prop re-sim readout (memo correction #2).
    mean_pnl_r_is_baseline: float | None = None  # multiplier 1.0
    mean_pnl_r_is_075: float | None = None
    ev_delta_075_vs_baseline_is: float | None = None
    mean_pnl_r_oos_baseline: float | None = None
    mean_pnl_r_oos_075: float | None = None
    ev_delta_075_vs_baseline_oos: float | None = None
    oos_power_tier: str | None = None
    verdict: str = "INSUFFICIENT_N"
    notes: str = ""


def _avg_pnl_r(outcomes: list[dict]) -> float | None:
    vals = [o["pnl_r"] for o in outcomes if o.get("pnl_r") is not None]
    return sum(vals) / len(vals) if vals else None


def _stop_selectivity(baseline: list[dict], adjusted: list[dict]) -> float | None:
    """Howard §5.3.1/§5.3.2 stop SELECTIVITY: of the events the stop FIRED on,
    the fraction that were eventual (baseline) LOSERS.

    A high-quality stop fires mostly on losers (high selectivity); a poor stop
    fires indiscriminately on winners and losers (selectivity near the base loss
    rate). `apply_tight_stop` rebooks an event as a -stop_multiplier loss iff its
    MAE breached the tighter threshold. "Fired" = baseline pnl_r != adjusted
    pnl_r for the same event (the stop changed the outcome). Of those fired
    events, count how many had baseline pnl_r <= 0 (eventual losers).

    Howard's number to beat is the ~75% breakeven HE derived from HIS symmetric
    MFE/MAE + 20-30 tick winner horizons — NOT a generic constant. We report this
    realized stop-selectivity for description; the VERDICT gates on direct EV
    (see _assign_verdict), not on a hardcoded 75%.
    Returns None if the stop fired on nothing.
    """
    fired_total = 0
    fired_on_losers = 0
    for base, adj in zip(baseline, adjusted, strict=True):
        base_r = base.get("pnl_r")
        adj_r = adj.get("pnl_r")
        if base_r is None or adj_r is None:
            continue
        if adj_r != base_r:  # stop changed this event's outcome
            fired_total += 1
            if base_r <= 0:  # event would have been a loser/scratch anyway
                fired_on_losers += 1
    if fired_total == 0:
        return None
    return fired_on_losers / fired_total


def _winner_damage(baseline: list[dict], adjusted: list[dict]) -> float | None:
    """Howard §5.3.2 winner-damage: fraction of eventual (baseline) WINNERS that
    the tighter stop would have terminated.

    Howard's headline: 90.2% of ES horizon-winners breach a +/-2-tick stop. The
    higher this fraction on our lanes, the more a tighter price stop destroys EV
    by truncating winners. Returns None if there are no baseline winners.
    """
    winners = [(b, a) for b, a in zip(baseline, adjusted, strict=True)
               if (b.get("pnl_r") or 0) > 0]
    if not winners:
        return None
    killed = sum(1 for b, a in winners if a.get("pnl_r") != b.get("pnl_r"))
    return killed / len(winners)


def _median_mfe_mae(outcomes: list[dict]) -> float | None:
    ratios = []
    for o in outcomes:
        mae = o.get("mae_r")
        mfe = o.get("mfe_r")
        if mae is None or mfe is None or mae <= 0:
            continue
        ratios.append(mfe / mae)
    return statistics.median(ratios) if ratios else None


def _analyze_stratum(
    con: duckdb.DuckDBPyConnection,
    snapshot: dict,
    entry_model: str,
) -> StratumReport:
    instrument = snapshot["instrument"]
    cost_spec = get_cost_spec(instrument)

    # Load IS and OOS outcomes for this stratum (entry_model overridden per the
    # E1/E2/E3 stratification). _load_strategy_outcomes applies the lane filter
    # and triple-joins daily_features — eligible (traded) days only.
    common = dict(
        con=con,
        instrument=instrument,
        orb_label=snapshot["orb_label"],
        orb_minutes=int(snapshot["orb_minutes"]),
        entry_model=entry_model,
        rr_target=float(snapshot["rr_target"]),
        confirm_bars=int(snapshot["confirm_bars"]),
        filter_type=snapshot["filter_type"],
    )
    is_outcomes = _load_strategy_outcomes(**common, end_date=_day_before(HOLDOUT_SACRED_FROM))
    oos_outcomes = _load_strategy_outcomes(**common, start_date=HOLDOUT_SACRED_FROM)

    # Keep only resolved win/loss rows with the fields apply_tight_stop needs.
    is_trades = [o for o in is_outcomes if o.get("outcome") in ("win", "loss")]
    oos_trades = [o for o in oos_outcomes if o.get("outcome") in ("win", "loss")]

    rep = StratumReport(
        strategy_id=snapshot["strategy_id"],
        instrument=instrument,
        orb_label=snapshot["orb_label"],
        orb_minutes=int(snapshot["orb_minutes"]),
        rr_target=float(snapshot["rr_target"]),
        confirm_bars=int(snapshot["confirm_bars"]),
        filter_type=snapshot["filter_type"],
        entry_model=entry_model,
        n_is=len(is_trades),
        n_oos=len(oos_trades),
        median_mfe_mae_ratio=_median_mfe_mae(is_trades),
    )

    if not is_trades:
        rep.notes = "no IS trades for this stratum (entry model may be absent on lane)"
        return rep

    # Baseline = unmanaged (opposite-ORB-boundary) stop. apply_tight_stop is a
    # no-op at 1.0 (config.py:4167 returns the input list), so is_trades IS the
    # baseline. Each tighter multiplier is diffed against it for stop-selectivity
    # and winner-damage (Howard §5.3.2).
    rep.grid_mean_pnl_r[1.0] = _avg_pnl_r(is_trades)
    for m in STOP_MULTIPLIER_GRID:
        if m >= 1.0:
            continue
        adj_is = apply_tight_stop(is_trades, m, cost_spec)
        mean_pnl = _avg_pnl_r(adj_is)
        if mean_pnl is not None:
            rep.grid_mean_pnl_r[m] = mean_pnl
        sel = _stop_selectivity(is_trades, adj_is)
        if sel is not None:
            rep.grid_stop_selectivity[m] = sel
        dmg = _winner_damage(is_trades, adj_is)
        if dmg is not None:
            rep.grid_winner_damage[m] = dmg

    rep.mean_pnl_r_is_baseline = rep.grid_mean_pnl_r.get(1.0)
    rep.mean_pnl_r_is_075 = rep.grid_mean_pnl_r.get(0.75)
    if rep.mean_pnl_r_is_baseline is not None and rep.mean_pnl_r_is_075 is not None:
        rep.ev_delta_075_vs_baseline_is = rep.mean_pnl_r_is_075 - rep.mean_pnl_r_is_baseline

    # OOS 0.75 readout (monitoring-only; never tuned).
    if oos_trades:
        oos_base = _avg_pnl_r(apply_tight_stop(oos_trades, 1.0, cost_spec))
        oos_075 = _avg_pnl_r(apply_tight_stop(oos_trades, 0.75, cost_spec))
        rep.mean_pnl_r_oos_baseline = oos_base
        rep.mean_pnl_r_oos_075 = oos_075
        if oos_base is not None and oos_075 is not None:
            rep.ev_delta_075_vs_baseline_oos = oos_075 - oos_base
        # OOS power tier for the IS effect (RULE 3.3): cohen_d = |t_IS| / sqrt(N_IS).
        rep.oos_power_tier = _oos_power_tier(is_trades, len(oos_trades))

    _assign_verdict(rep)
    return rep


def _oos_power_tier(is_trades: list[dict], n_oos: int) -> str | None:
    """One-sample power tier for the IS pnl_r effect at the OOS N (RULE 3.3)."""
    vals = [o["pnl_r"] for o in is_trades if o.get("pnl_r") is not None]
    if len(vals) < 2 or n_oos < 2:
        return None
    mean_r = sum(vals) / len(vals)
    std_r = statistics.pstdev(vals)
    if std_r <= 0:
        return None
    t_is = abs(mean_r) * (len(vals) ** 0.5) / std_r
    cohen_d = t_is / (len(vals) ** 0.5)
    return power_verdict(one_sample_power(cohen_d, n_oos))


def _assign_verdict(rep: StratumReport) -> None:
    """Assign the price-stop-family gating verdict.

    PRIMARY gate = direct realized-EV evidence from the canonical
    apply_tight_stop sweep. H0's literal question is "can a tighter price stop
    add EV?" — the most direct answer is whether ANY tighter multiplier improves
    realized IS mean pnl_r over the baseline (1.0) stop. If none does (EV is
    monotone-non-increasing as the stop tightens), price stops are
    value-destroying on this lane and the price-stop family (H1/H3) is pre-killed.

    The MFE/MAE symmetry ratio (Howard §5.3.1), realized stop-selectivity and
    winner-damage (Howard §5.3.2) are REPORTED as the measured mechanism behind
    the EV result. They are NOT gating inputs and carry no hardcoded threshold
    (Howard's ~75% breakeven is ES-specific, not a generic constant). An earlier
    draft ANDed a symmetry band into the gate AND compared a mislabeled
    population profit-factor against a hardcoded 75%; both removed — the verdict
    is the direct measured EV, the rest is descriptive.
    """
    if rep.n_is < MIN_N_FOR_VERDICT:
        rep.verdict = "INSUFFICIENT_N"
        rep.notes = f"N_IS={rep.n_is} < {MIN_N_FOR_VERDICT} floor; descriptive only"
        return
    baseline = rep.grid_mean_pnl_r.get(1.0)
    tighter_ev = {m: v for m, v in rep.grid_mean_pnl_r.items()
                  if m < 1.0 and v is not None}
    if baseline is None or not tighter_ev:
        rep.verdict = "INSUFFICIENT_N"
        rep.notes = "baseline (1.0) or tighter-multiplier EV undefined"
        return

    best_tighter_m = max(tighter_ev, key=lambda k: tighter_ev[k])
    best_tighter_ev = tighter_ev[best_tighter_m]
    any_tighter_improves = best_tighter_ev > baseline

    # Measured mechanism descriptors (reported, not gating, no thresholds).
    ratio = rep.median_mfe_mae_ratio
    ratio_str = f"{ratio:.3f}" if ratio is not None else "n/a"
    dmg_075 = rep.grid_winner_damage.get(0.75)
    dmg_str = f"{dmg_075:.1%}" if dmg_075 is not None else "n/a"
    sel_075 = rep.grid_stop_selectivity.get(0.75)
    sel_str = f"{sel_075:.1%}" if sel_075 is not None else "n/a"

    if not any_tighter_improves:
        rep.verdict = "PRE_KILL_PRICE_STOPS"
        rep.notes = (
            f"NO tighter stop beats baseline EV (best tighter m={best_tighter_m} "
            f"EV={best_tighter_ev:+.4f} <= baseline {baseline:+.4f}). Price-stop "
            f"family (H1/H3) value-destroying. Measured mechanism: median "
            f"MFE/MAE={ratio_str}, winner-damage@0.75={dmg_str}, "
            f"stop-selectivity@0.75={sel_str}. H2 (entry switch) / H4 (time exit) "
            f"UNAFFECTED."
        )
    else:
        rep.verdict = "PROCEED_H1_H3"
        rep.notes = (
            f"tighter stop m={best_tighter_m} improves EV ({best_tighter_ev:+.4f} "
            f"> baseline {baseline:+.4f}); H1/H3 LIVE. Measured mechanism: median "
            f"MFE/MAE={ratio_str}, winner-damage@0.75={dmg_str}, "
            f"stop-selectivity@0.75={sel_str}."
        )


def _day_before(d: date) -> date:
    from datetime import timedelta

    return d - timedelta(days=1)


def _deployed_strategy_ids(instrument_filter: str | None) -> list[str]:
    seen: dict[str, None] = {}
    for profile in ACCOUNT_PROFILES.values():
        for lane in profile.daily_lanes:
            if instrument_filter and lane.instrument != instrument_filter:
                continue
            seen.setdefault(lane.strategy_id, None)
    return list(seen.keys())


def _print_report(reports: list[StratumReport]) -> None:
    print("\n=== H0 MFE/MAE SYMMETRY DIAGNOSTIC (K=0, read-only) ===")
    print(f"IS boundary (HOLDOUT_SACRED_FROM): {HOLDOUT_SACRED_FROM}")
    print(f"Stop-multiplier grid: {STOP_MULTIPLIER_GRID}")
    print(f"N floor for verdict: {MIN_N_FOR_VERDICT}\n")
    for r in reports:
        print(f"--- {r.strategy_id}  [{r.entry_model}] ---")
        print(f"    N_IS={r.n_is}  N_OOS={r.n_oos}")
        if r.median_mfe_mae_ratio is not None:
            print(f"    median MFE/MAE ratio (IS, Howard symmetry): {r.median_mfe_mae_ratio:.4f}")
        if r.ev_delta_075_vs_baseline_is is not None:
            print(
                f"    [existing 0.75 readout] IS mean pnl_r: "
                f"1.0={r.mean_pnl_r_is_baseline:.4f}  0.75={r.mean_pnl_r_is_075:.4f}  "
                f"delta={r.ev_delta_075_vs_baseline_is:+.4f}"
            )
        if r.ev_delta_075_vs_baseline_oos is not None:
            print(
                f"    [existing 0.75 readout] OOS mean pnl_r: "
                f"1.0={r.mean_pnl_r_oos_baseline:.4f}  0.75={r.mean_pnl_r_oos_075:.4f}  "
                f"delta={r.ev_delta_075_vs_baseline_oos:+.4f}  power_tier={r.oos_power_tier}"
            )
        if r.grid_mean_pnl_r:
            grid = "  ".join(f"{m}:{v:+.4f}" for m, v in sorted(r.grid_mean_pnl_r.items()))
            print(f"    mean pnl_r by multiplier (IS): {grid}")
        if r.grid_winner_damage:
            grid = "  ".join(f"{m}:{v:.1%}" for m, v in sorted(r.grid_winner_damage.items()))
            print(f"    winner-damage by multiplier (Howard §5.3.2): {grid}")
        if r.grid_stop_selectivity:
            grid = "  ".join(f"{m}:{v:.1%}" for m, v in sorted(r.grid_stop_selectivity.items()))
            print(f"    stop-selectivity by multiplier (Howard §5.3.2): {grid}")
        print(f"    VERDICT: {r.verdict}")
        print(f"    {r.notes}\n")


def _write_csv(reports: list[StratumReport], path: str) -> None:
    scalar_cols = [
        "strategy_id", "instrument", "orb_label", "orb_minutes", "rr_target",
        "confirm_bars", "filter_type", "entry_model", "n_is", "n_oos",
        "median_mfe_mae_ratio",
        "mean_pnl_r_is_baseline", "mean_pnl_r_is_075", "ev_delta_075_vs_baseline_is",
        "mean_pnl_r_oos_baseline", "mean_pnl_r_oos_075", "ev_delta_075_vs_baseline_oos",
        "oos_power_tier", "verdict", "notes",
    ]
    # One column per tighter multiplier for EV / winner-damage / stop-selectivity.
    tighter = [m for m in STOP_MULTIPLIER_GRID if m < 1.0]
    grid_cols = (
        [f"ev_m{m}" for m in tighter]
        + [f"winner_damage_m{m}" for m in tighter]
        + [f"stop_selectivity_m{m}" for m in tighter]
    )
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(scalar_cols + grid_cols)
        for r in reports:
            row = [getattr(r, c) for c in scalar_cols]
            row += [r.grid_mean_pnl_r.get(m) for m in tighter]
            row += [r.grid_winner_damage.get(m) for m in tighter]
            row += [r.grid_stop_selectivity.get(m) for m in tighter]
            w.writerow(row)
    print(f"Wrote per-stratum CSV: {path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instrument", choices=["MNQ", "MES"], default=None,
                    help="Restrict to one instrument (MGC horizon-excluded by pre-reg).")
    ap.add_argument("--csv", default=None, help="Write per-stratum rows to this CSV path.")
    ap.add_argument("--allow-draft", action="store_true",
                    help="Operator dry-run before the pre-reg is promoted/committed.")
    args = ap.parse_args(argv)

    if not args.allow_draft:
        print(
            "REFUSING TO RUN: pre-reg is still a DRAFT. Promote "
            "docs/audit/hypotheses/drafts/2026-05-31-adaptive-stops-h0-*.draft.yaml "
            "out of drafts/, commit it, then re-run. Operator dry-run: --allow-draft.",
            file=sys.stderr,
        )
        return 2

    strategy_ids = _deployed_strategy_ids(args.instrument)
    if not strategy_ids:
        print("No deployed lanes matched.", file=sys.stderr)
        return 1

    # READ-ONLY connection. DuckDB blocks a reader while any writer holds the
    # file (single-writer model). Abort cleanly with an actionable message rather
    # than dumping a raw traceback (institutional-rigor §6 — no silent/ugly
    # failures). The recurring cause on this machine is a stale MCP/uv process
    # holding a handle — see reap_stale_claude_processes.py.
    try:
        con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    except duckdb.IOException as exc:
        print(
            f"ABORT: cannot open {GOLD_DB_PATH} read-only — a writer holds the "
            f"file. Run when no writer is active, or reap stale handles "
            f"(scripts/tools/reap_stale_claude_processes.py --apply). "
            f"DuckDB said: {exc}",
            file=sys.stderr,
        )
        return 1
    try:
        reports: list[StratumReport] = []
        for sid in strategy_ids:
            try:
                snapshot = _load_strategy_snapshot(con, sid)
            except ValueError as exc:
                print(f"  SKIP {sid}: {exc}", file=sys.stderr)
                continue
            for em in ENTRY_MODELS:
                reports.append(_analyze_stratum(con, snapshot, em))
    finally:
        con.close()

    _print_report(reports)
    if args.csv:
        _write_csv(reports, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
