#!/usr/bin/env python3
"""
Build the REGIME-tier shadow-monitoring universe (READ-ONLY).

Enumerates EVERY active REGIME-tier strategy (validated sample_size 30-99) for
the live instruments. This is the lane universe that `regime_shadow_runner.py`
drives forward into the shadow ledger.

RECORD-ALL (operator decision, 2026-06-03): every active REGIME-tier lane is
included unconditionally (`included=True`). Fitness is an ATTRIBUTE recorded per
lane (FIT / WATCH / DECAY / STALE / ERROR), NEVER an inclusion gate — the goal is
to shadow-record every would-have trade, and DECAY/STALE lanes are the MOST
informative signal for a regime turn. Fitness is delegated to
trading_app.strategy_fitness.compute_fitness; this module never re-derives
rolling metrics or re-classifies status itself.

REGIME tier boundary is canonical (RESEARCH_RULES.md / docs/ARCHITECTURE.md):
    30 <= sample_size <= 99  ->  REGIME
    sample_size >= 100       ->  CORE (excluded — never shadowed here)

This module performs NO writes. It reads validated_setups + outcomes (read-only)
and emits a universe report (and optionally a YAML snapshot). It changes zero
live allocation, profiles, or capital logic.

Usage:
    python -m scripts.tools.regime_shadow_universe                 # print report
    python -m scripts.tools.regime_shadow_universe --write-yaml    # also snapshot
"""

from __future__ import annotations

import argparse
import datetime
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb

from pipeline.log import get_logger
from pipeline.paths import GOLD_DB_PATH

logger = get_logger(__name__)

# Canonical REGIME tier boundary — imported, NEVER re-encoded. The 30/99 bound
# lives in trading_app.config (REGIME_MIN_SAMPLES / CORE_MIN_SAMPLES) and is
# applied via classify_strategy (RESEARCH_RULES.md:55, docs/ARCHITECTURE.md:112).
from trading_app.config import (  # noqa: E402  (import after logger by design)
    CORE_MIN_SAMPLES,
    REGIME_MIN_SAMPLES,
)

# Inclusive REGIME sample-size band derived from the canonical constants.
REGIME_MIN_SAMPLE = REGIME_MIN_SAMPLES
REGIME_MAX_SAMPLE = CORE_MIN_SAMPLES - 1

# Instruments live for ORB (canonical source). Imported lazily in build to keep
# import-time side effects minimal and to fail loud if the canon changes shape.
_INSTRUMENT_FALLBACK = ("MNQ", "MES", "MGC")

# Default universe snapshot path (generated artifact).
UNIVERSE_YAML = Path("docs/runtime/regime_shadow_universe.yaml")


@dataclass(frozen=True)
class RegimeLane:
    """One REGIME-tier shadow-universe lane.

    A frozen record of the canonical strategy identity plus the recorded fitness
    status (an ATTRIBUTE, not an inclusion gate — see module docstring). Under
    record-ALL every active REGIME lane has `included=True`; `reason` carries the
    recorded status so the snapshot stays fully auditable (no silent drops).
    """

    strategy_id: str
    instrument: str
    orb_label: str
    orb_minutes: int
    rr_target: float
    entry_model: str
    confirm_bars: int
    filter_type: str
    sample_size: int
    fitness_status: str
    rolling_sample: int
    rolling_exp_r: float | None
    included: bool
    reason: str
    # F1: per-lane forward-monitoring boundary. The earliest trading_day this
    # lane may be recorded as shadow — its OWN first-eligible date, NOT the shared
    # global forward_start. A lane that JOINS the universe on a later run (its
    # sample_size crosses into the 30-99 band) gets first_seen=as_of_date so its
    # "N trades since start" window is comparable to other lanes. Preserved
    # (never advanced) across refreshes, mirroring the global forward_start
    # preserve-on-refresh discipline. The effective write boundary is
    # max(forward_start, first_seen) (threaded by the runner) so a lane can never
    # write earlier than the global forward-only floor.
    first_seen: datetime.date


def _active_instruments() -> tuple[str, ...]:
    """Resolve the live ORB instruments from canonical config (fail-open to the
    known triple if the canon import is unavailable in a stripped env)."""
    try:
        from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS

        return tuple(ACTIVE_ORB_INSTRUMENTS)
    except Exception:  # pragma: no cover - canon import always present in repo
        logger.warning("ACTIVE_ORB_INSTRUMENTS unavailable; using fallback %s", _INSTRUMENT_FALLBACK)
        return _INSTRUMENT_FALLBACK


def _query_regime_strategies(
    con: duckdb.DuckDBPyConnection,
    instruments: tuple[str, ...],
) -> list[dict]:
    """Read active REGIME-tier rows from validated_setups (read-only).

    Tier boundary is applied in SQL against the canonical sample_size column.
    Only status='active' rows are candidates — retired setups are never
    shadowed.
    """
    rows = con.execute(
        """
        SELECT strategy_id, instrument, orb_label, orb_minutes, rr_target,
               entry_model, confirm_bars, filter_type, sample_size
        FROM validated_setups
        WHERE status = 'active'
          AND sample_size BETWEEN ? AND ?
          AND instrument IN (SELECT UNNEST(?::VARCHAR[]))
        ORDER BY instrument, orb_label, strategy_id
        """,
        [REGIME_MIN_SAMPLE, REGIME_MAX_SAMPLE, sorted(instruments)],
    ).fetchall()
    cols = [d[0] for d in con.description]
    return [dict(zip(cols, r, strict=False)) for r in rows]


def _load_prior_first_seen(
    universe_yaml: Path | str | None = None,
    forward_start: datetime.date | None = None,
) -> dict[str, datetime.date]:
    """Read the per-lane `first_seen` map from an existing universe snapshot.

    F1 boundary durability. Returns {strategy_id: first_seen} for every lane in
    the prior YAML so `build_universe` PRESERVES (never advances) a lane's
    first-seen date across a refresh — mirroring the global forward_start
    preserve-on-refresh discipline.

    Legacy migration (provable no-op): a prior-YAML lane that PREDATES this field
    has no `first_seen`. We back-derive `first_seen = forward_start` for it, so
    `max(forward_start, first_seen) == forward_start` — identical to the
    pre-F1 global-boundary behaviour, no retroactive row deletion. If the YAML
    carries no `forward_start` either, such legacy lanes are omitted from the map
    and `build_universe` will treat them as newly-seen (as_of_date) — which is
    correct because without a persisted boundary there is no earlier window to
    preserve.
    """
    path = Path(universe_yaml) if universe_yaml else UNIVERSE_YAML
    if not path.exists():
        return {}
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if forward_start is None:
        fs = data.get("forward_start")
        forward_start = datetime.date.fromisoformat(str(fs)) if fs else None

    out: dict[str, datetime.date] = {}
    for lane in data.get("lanes", []) or []:
        sid = lane.get("strategy_id")
        if not sid:
            continue
        raw = lane.get("first_seen")
        if raw:
            out[sid] = datetime.date.fromisoformat(str(raw))
        elif forward_start is not None:
            # Legacy lane (pre-F1): back-derive to the global boundary (no-op).
            out[sid] = forward_start
    return out


def build_universe(
    db_path: Path | str | None = None,
    as_of_date: datetime.date | None = None,
    *,
    prior_first_seen: dict[str, datetime.date] | None = None,
    universe_yaml: Path | str | None = None,
) -> list[RegimeLane]:
    """Build the REGIME shadow universe (RECORD-ALL).

    READ-ONLY. Returns one RegimeLane per active REGIME-tier strategy. EVERY
    active REGIME lane is included (`included=True`) — fitness is recorded as an
    attribute, never an inclusion gate (operator decision 2026-06-03). A lane
    whose fitness cannot be computed is STILL included and flagged ERROR rather
    than dropped (no silent loss of a would-record lane).

    Delegates classification entirely to compute_fitness — never re-derives.

    F1 per-lane boundary: each lane carries `first_seen`. For a lane already
    present in the prior universe snapshot, `first_seen` is PRESERVED (read via
    `prior_first_seen`, or loaded from `universe_yaml` when not supplied). A
    newly-seen lane gets `first_seen = as_of_date`. Preservation means a lane
    that drops out and rejoins keeps its original monitoring-start, so per-lane
    "N trades since start" windows stay comparable.
    """
    from trading_app.strategy_fitness import compute_fitness

    path = Path(db_path) if db_path else GOLD_DB_PATH
    if as_of_date is None:
        as_of_date = datetime.date.today()

    if prior_first_seen is None:
        prior_first_seen = _load_prior_first_seen(universe_yaml)

    instruments = _active_instruments()

    with duckdb.connect(str(path), read_only=True) as con:
        candidates = _query_regime_strategies(con, instruments)

    from trading_app.config import classify_strategy

    lanes: list[RegimeLane] = []
    for c in candidates:
        # Tripwire: the SQL band must agree with the canonical classifier. If a
        # future edit drifts the band, fail loud rather than silently shadow a
        # CORE/INVALID strategy.
        tier = classify_strategy(c["sample_size"])
        if tier != "REGIME":
            raise ValueError(
                f"tier drift: {c['strategy_id']} sample_size={c['sample_size']} "
                f"classified {tier!r}, expected REGIME. SQL band vs classify_strategy disagree."
            )
        try:
            score = compute_fitness(c["strategy_id"], db_path=path, as_of_date=as_of_date)
            status = score.fitness_status
            rolling_sample = score.rolling_sample
            rolling_exp_r = score.rolling_exp_r
        except (ValueError, duckdb.Error, KeyError) as exc:
            # Fitness uncomputable -> STILL included (record-ALL), flagged ERROR.
            logger.warning("fitness failed for %s: %s", c["strategy_id"], exc)
            status, rolling_sample, rolling_exp_r = "ERROR", 0, None

        # RECORD-ALL: every active REGIME lane is included; fitness is recorded,
        # not gated on. ERROR lanes are flagged but never dropped.
        included = True
        if status == "ERROR":
            reason = "fitness uncomputable (flagged) — recorded"
        else:
            reason = f"{status} — recorded"

        # F1: preserve an existing lane's first_seen; a newly-seen lane starts at
        # as_of_date. Never advance a preserved value (prior_first_seen wins).
        first_seen = prior_first_seen.get(c["strategy_id"], as_of_date)

        lanes.append(
            RegimeLane(
                strategy_id=c["strategy_id"],
                instrument=c["instrument"],
                orb_label=c["orb_label"],
                orb_minutes=c["orb_minutes"],
                rr_target=c["rr_target"],
                entry_model=c["entry_model"],
                confirm_bars=c["confirm_bars"],
                filter_type=c["filter_type"],
                sample_size=c["sample_size"],
                fitness_status=status,
                rolling_sample=rolling_sample,
                rolling_exp_r=rolling_exp_r,
                included=included,
                reason=reason,
                first_seen=first_seen,
            )
        )
    return lanes


def write_universe_yaml(
    lanes: list[RegimeLane],
    *,
    forward_start: datetime.date,
    path: Path | str | None = None,
    as_of_date: datetime.date | None = None,
) -> Path:
    """Write the universe snapshot YAML (generated artifact).

    G2 boundary durability: the `forward_start` is the immutable forward-only
    boundary. If the target YAML ALREADY carries a `forward_start`, that
    persisted value is PRESERVED and the `forward_start` argument is ignored —
    a universe refresh must never silently move the boundary forward and re-open
    the relabel window. The lane list and `generated` date are always refreshed.
    """
    import yaml

    out = Path(path) if path else UNIVERSE_YAML
    if as_of_date is None:
        as_of_date = datetime.date.today()

    # Preserve an already-persisted boundary (never clobber).
    effective_start = forward_start
    if out.exists():
        existing = yaml.safe_load(out.read_text(encoding="utf-8")) or {}
        prior = existing.get("forward_start")
        if prior:
            effective_start = datetime.date.fromisoformat(str(prior))
            if effective_start != forward_start:
                logger.info(
                    "preserving persisted forward_start=%s (ignoring %s)",
                    effective_start.isoformat(),
                    forward_start.isoformat(),
                )

    def _lane_payload(lane: RegimeLane) -> dict:
        # Stringify first_seen to an ISO date (consistent with forward_start /
        # generated, and round-trip-safe via _load_prior_first_seen's
        # fromisoformat). asdict would otherwise emit a native date object.
        d = asdict(lane)
        d["first_seen"] = lane.first_seen.isoformat()
        return d

    payload = {
        "generated": as_of_date.isoformat(),
        "forward_start": effective_start.isoformat(),
        "regime_tier": {"min_sample": REGIME_MIN_SAMPLE, "max_sample": REGIME_MAX_SAMPLE},
        "lanes": [_lane_payload(lane) for lane in lanes],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return out


def _print_report(lanes: list[RegimeLane]) -> None:
    print("=" * 78)
    print("REGIME SHADOW UNIVERSE (record-ALL)")
    print("=" * 78)
    by_inst: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for x in lanes:
        by_inst[x.instrument] = by_inst.get(x.instrument, 0) + 1
        by_status[x.fitness_status] = by_status.get(x.fitness_status, 0) + 1
    print(
        f"REGIME-tier lanes (all recorded): {len(lanes)}  ({', '.join(f'{k}={v}' for k, v in sorted(by_inst.items()))})"
    )
    print(f"fitness mix (attribute): {', '.join(f'{k}={v}' for k, v in sorted(by_status.items()))}")
    print("-" * 78)
    for x in lanes:
        print(
            f"  {x.strategy_id:<48s} N={x.sample_size:<3d} "
            f"{x.fitness_status:<6s} roll_n={x.rolling_sample:<3d}  {x.reason}"
        )
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build REGIME shadow universe (read-only, record-ALL)")
    parser.add_argument("--write-yaml", action="store_true", help="Write universe snapshot YAML")
    parser.add_argument(
        "--forward-start", type=str, default=None, help="ISO date for the persisted forward boundary (default: today)"
    )
    args = parser.parse_args()

    forward_start = datetime.date.fromisoformat(args.forward_start) if args.forward_start else datetime.date.today()
    lanes = build_universe()
    _print_report(lanes)

    if args.write_yaml:
        out = write_universe_yaml(lanes, forward_start=forward_start)
        print(f"Universe snapshot written: {out}  (forward_start={forward_start.isoformat()})")


if __name__ == "__main__":
    main()
