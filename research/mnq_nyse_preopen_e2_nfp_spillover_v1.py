"""Bounded standalone runner for the MNQ NYSE_PREOPEN E2 NFP-spillover v1 prereg.

Consumes ``docs/audit/hypotheses/2026-05-25-mnq-nyse-preopen-e2-nfp-spillover-v1.yaml``
(SHA ``40f032aa3ecc99d3fdf5721e6af09d0de13885a94cbea54861c13a60f2be6436`` at
promotion) and produces a K=27 strict-Chordia verdict over the proposed
NYSE_PREOPEN session on MNQ E2 CB1, partitioned by NFP-day class.

Stage 4a contract
-----------------

This module is the runner ONLY. Stage 4a SHIPS the runner and its unit
tests. Stage 4b (separate, gated by user approval) runs the runner against
canonical layers and emits the verdict markdown + CSV.

This module:

- NEVER writes to ``gold.db``.
- NEVER writes to ``experimental_strategies``.
- NEVER writes to ``validated_setups``.
- NEVER writes to allocator / live-config / lane-allocation state.
- Reads canonical layers only: ``orb_outcomes`` JOIN ``daily_features``.
- Reads the prereg YAML and computes its SHA via
  ``trading_app.hypothesis_loader``.

Verdict gates (per prereg)
--------------------------

K_family = 27 = 3 ORB apertures (5/15/30) x 3 RR targets (1.0/1.5/2.0)
                x 3 NFP-splits (all_days / nfp_days_only / non_nfp_days).
Entry model = E2, confirm_bars = 1, instrument = MNQ, session = NYSE_PREOPEN.

Per-cell PASS requires ALL of:

1. ``t_IS >= 3.79`` (strict Chordia, no-theory bound -- prereg locks
   ``theory_grant: false``)
2. ``N_IS_on >= 100``
3. ``ExpR_IS > 0``
4. BH-FDR at K=27 family ``q < 0.05``
5. NOT killed by DST-imbalance (``N_EST < 30 OR N_EDT < 30`` => UNVERIFIED,
   not DEAD)
6. NOT killed by OOS dir-flip when OOS power tier is CAN_REFUTE

OOS power floor: ``research.oos_power.one_sample_power`` + ``power_verdict``
applied per cell. ``dir_match=False`` only kills when tier == ``CAN_REFUTE``.

Authority
---------

- ``docs/audit/hypotheses/2026-05-25-mnq-nyse-preopen-e2-nfp-spillover-v1.yaml``
- ``docs/institutional/pre_registered_criteria.md``
- ``.claude/rules/backtesting-methodology.md`` RULES 1, 3, 3.3, 4, 6, 9
- ``.claude/rules/hypothesis-prereg-discipline.md``
- ``pipeline.calendar_filters.is_nfp_day``
- ``pipeline.market_calendar.is_nyse_holiday`` (wired in Stage 2 -- holiday
  contamination is excluded upstream of this runner via
  ``pipeline.build_daily_features.compute_orb_range`` returning all-None
  for ``orb_label='NYSE_PREOPEN'`` on NYSE-closed days)

@research-source: docs/audit/hypotheses/2026-05-25-mnq-nyse-preopen-e2-nfp-spillover-v1.yaml
@canonical-source: research/mnq_nyse_preopen_e2_nfp_spillover_v1.py
@revalidated-for: Lane B Stage 4a (2026-05-28)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
from statsmodels.stats.multitest import multipletests

from pipeline.calendar_filters import is_nfp_day
from pipeline.market_calendar import is_nyse_holiday
from pipeline.paths import GOLD_DB_PATH
from research.oos_power import one_sample_power, one_sample_tstat, power_verdict
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
from trading_app.hypothesis_loader import (
    HypothesisLoaderError,
    compute_file_sha,
    load_hypothesis_metadata,
)

PREREG_FILENAME = "2026-05-25-mnq-nyse-preopen-e2-nfp-spillover-v1.yaml"
PROMOTED_PREREG_SHA = (
    "40f032aa3ecc99d3fdf5721e6af09d0de13885a94cbea54861c13a60f2be6436"
)

ORB_MINUTES = (5, 15, 30)
RR_TARGETS = (1.0, 1.5, 2.0)
SPLITS = ("all_days", "nfp_days_only", "non_nfp_days")

INSTRUMENT = "MNQ"
SESSION = "NYSE_PREOPEN"
ENTRY_MODEL = "E2"
CONFIRM_BARS = 1

K_FAMILY = len(ORB_MINUTES) * len(RR_TARGETS) * len(SPLITS)  # == 27

# Per-prereg promotion gate (Chordia strict no-theory)
CHORDIA_T_STRICT = 3.79
MIN_N_IS_ON = 100
MIN_N_PER_DST_REGIME = 30  # below either => UNVERIFIED, not DEAD


@dataclass(frozen=True)
class CellSpec:
    orb_minutes: int
    rr_target: float
    split: str  # one of SPLITS

    @property
    def cell_id(self) -> str:
        return f"O{self.orb_minutes}_RR{self.rr_target}_{self.split}"


@dataclass(frozen=True)
class OutcomeRow:
    trading_day: date
    pnl_r: float
    entry_ts: datetime | None


@dataclass(frozen=True)
class CellStats:
    spec: CellSpec
    n_is_on: int
    expr_is: float
    sharpe_is: float
    t_is: float
    p_one_sided: float
    n_oos_on: int
    expr_oos: float
    n_est_is: int
    n_edt_is: int
    dir_match_oos: bool
    oos_power: float
    oos_power_tier: str


@dataclass(frozen=True)
class CellVerdict:
    cell: CellStats
    bh_q: float
    bh_reject: bool
    dst_balance_verdict: str  # "BALANCED" | "EST_THIN" | "EDT_THIN"
    pass_chordia_strict: bool
    verdict_label: str
    verdict_reason: str


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def prereg_path() -> Path:
    return repo_root() / "docs" / "audit" / "hypotheses" / PREREG_FILENAME


def enumerate_cells() -> list[CellSpec]:
    """Enumerate all K=27 cells in the prereg's lock order."""
    cells: list[CellSpec] = []
    for o in ORB_MINUTES:
        for rr in RR_TARGETS:
            for split in SPLITS:
                cells.append(CellSpec(orb_minutes=o, rr_target=rr, split=split))
    if len(cells) != K_FAMILY:
        raise RuntimeError(
            f"Cell enumeration drift: produced {len(cells)} cells, expected {K_FAMILY}. "
            "Prereg schema has changed -- refuse to compute a contaminated verdict."
        )
    return cells


def fetch_canonical_cell_rows(
    con: duckdb.DuckDBPyConnection,
    orb_minutes: int,
    rr_target: float,
) -> list[OutcomeRow]:
    """Read all canonical orb_outcomes rows for the prereg's (orb, rr) cell.

    Returns one dict per row with the columns the runner needs. Does NOT apply
    the NFP split here -- the split is a sample partition applied downstream
    by ``apply_split``.

    Holiday contamination is excluded UPSTREAM in this runner's contract: the
    NYSE_PREOPEN ORB row is produced by ``pipeline.build_daily_features``
    only on days where ``is_nyse_holiday(trading_day) == False`` (Stage 2
    wiring, ``296c54f2``). Days when NYSE cash is closed therefore have no
    entry/exit because ``daily_features.orb_high IS NULL``, and no
    corresponding ``orb_outcomes`` row exists (or, if one exists, ``pnl_r``
    will be NULL and the row is excluded by ``pnl_r IS NOT NULL`` below).
    """
    rows = con.execute(
        """
        SELECT trading_day, pnl_r, entry_ts
        FROM orb_outcomes
        WHERE symbol = ?
          AND orb_label = ?
          AND entry_model = ?
          AND confirm_bars = ?
          AND orb_minutes = ?
          AND rr_target = ?
          AND pnl_r IS NOT NULL
        ORDER BY trading_day
        """,
        [INSTRUMENT, SESSION, ENTRY_MODEL, CONFIRM_BARS, orb_minutes, rr_target],
    ).fetchall()
    return [OutcomeRow(trading_day=r[0], pnl_r=float(r[1]), entry_ts=r[2]) for r in rows]


def apply_split(rows: list[OutcomeRow], split: str) -> list[OutcomeRow]:
    """Partition rows by NFP-day class per the prereg's canonical split source.

    ``split_source_canonical: pipeline.calendar_filters.is_nfp_day``.
    """
    if split == "all_days":
        return list(rows)
    if split == "nfp_days_only":
        return [r for r in rows if is_nfp_day(r.trading_day)]
    if split == "non_nfp_days":
        return [r for r in rows if not is_nfp_day(r.trading_day)]
    raise ValueError(f"Unknown split: {split!r}; expected one of {SPLITS}")


def split_by_holdout(
    rows: list[OutcomeRow],
) -> tuple[list[OutcomeRow], list[OutcomeRow]]:
    """Partition rows by ``trading_day < HOLDOUT_SACRED_FROM`` (strict Mode A)."""
    is_rows = [r for r in rows if r.trading_day < HOLDOUT_SACRED_FROM]
    oos_rows = [r for r in rows if r.trading_day >= HOLDOUT_SACRED_FROM]
    return is_rows, oos_rows


def count_dst_regimes(rows: list[OutcomeRow]) -> tuple[int, int]:
    """Return (N_EST, N_EDT) for the rows.

    Canonical authority: ``pipeline.dst.is_us_dst(trading_day)`` -> EDT when
    True. Imported lazily so unit tests with a fake date don't pay the import
    cost and to avoid pulling the full dst module surface into module scope.
    """
    from pipeline.dst import is_us_dst

    n_est = 0
    n_edt = 0
    for r in rows:
        if is_us_dst(r.trading_day):
            n_edt += 1
        else:
            n_est += 1
    return n_est, n_edt


def compute_cell_stats(
    spec: CellSpec,
    all_rows: list[OutcomeRow],
) -> CellStats:
    """Compute one cell's stats from the unfiltered (orb, rr) row set."""
    split_rows = apply_split(all_rows, spec.split)
    is_rows, oos_rows = split_by_holdout(split_rows)

    n_is_on = len(is_rows)
    n_oos_on = len(oos_rows)

    is_pnl = np.array([r.pnl_r for r in is_rows], dtype=float) if is_rows else np.array([])
    oos_pnl = np.array([r.pnl_r for r in oos_rows], dtype=float) if oos_rows else np.array([])

    expr_is = float(is_pnl.mean()) if is_pnl.size else float("nan")
    expr_oos = float(oos_pnl.mean()) if oos_pnl.size else float("nan")

    if is_pnl.size >= 2 and is_pnl.std(ddof=1) > 0:
        std_is = float(is_pnl.std(ddof=1))
        sharpe_is = expr_is / std_is
        t_is, p_one = one_sample_tstat(expr_is, std_is, n_is_on)
    else:
        sharpe_is = float("nan")
        t_is = float("nan")
        p_one = float("nan")

    n_est_is, n_edt_is = count_dst_regimes(is_rows)

    dir_match = False
    if not (np.isnan(expr_is) or np.isnan(expr_oos)):
        dir_match = (expr_is > 0 and expr_oos > 0) or (expr_is < 0 and expr_oos < 0)

    if n_is_on >= 2 and not np.isnan(t_is) and not np.isnan(expr_is):
        cohen_d = abs(t_is) / (n_is_on ** 0.5) if n_is_on > 0 else 0.0
        oos_pwr = one_sample_power(cohen_d, n_oos_on) if n_oos_on >= 2 else 0.0
    else:
        oos_pwr = 0.0

    return CellStats(
        spec=spec,
        n_is_on=n_is_on,
        expr_is=expr_is,
        sharpe_is=sharpe_is,
        t_is=t_is,
        p_one_sided=p_one,
        n_oos_on=n_oos_on,
        expr_oos=expr_oos,
        n_est_is=n_est_is,
        n_edt_is=n_edt_is,
        dir_match_oos=dir_match,
        oos_power=oos_pwr,
        oos_power_tier=power_verdict(oos_pwr),
    )


def compose_bh_fdr(p_values: Iterable[float], alpha: float = 0.05) -> tuple[list[bool], list[float]]:
    """Run BH-FDR at the given alpha over the cell p-values.

    NaN p-values (insufficient sample) are treated as ``p=1.0`` so they fail
    the gate cleanly without poisoning the rejection set order. Returns
    ``(rejects, q_values)`` lists aligned with the input order.
    """
    p_arr = np.array([1.0 if (p is None or np.isnan(p)) else float(p) for p in p_values])
    if p_arr.size == 0:
        return [], []
    rejects, q_values, _, _ = multipletests(p_arr, alpha=alpha, method="fdr_bh")
    return list(rejects), list(q_values)


def grade_cell(cell: CellStats, bh_q: float, bh_reject: bool) -> CellVerdict:
    """Apply the prereg's per-cell promotion gate."""
    if cell.n_est_is >= MIN_N_PER_DST_REGIME and cell.n_edt_is >= MIN_N_PER_DST_REGIME:
        dst_balance = "BALANCED"
    elif cell.n_est_is < MIN_N_PER_DST_REGIME:
        dst_balance = "EST_THIN"
    else:
        dst_balance = "EDT_THIN"

    chordia_pass = (
        not np.isnan(cell.t_is)
        and cell.t_is >= CHORDIA_T_STRICT
        and cell.n_is_on >= MIN_N_IS_ON
        and cell.expr_is > 0.0
        and bh_reject
    )

    if dst_balance != "BALANCED":
        verdict_label = "UNVERIFIED_DST_IMBALANCE"
        verdict_reason = f"{dst_balance}: N_EST={cell.n_est_is} N_EDT={cell.n_edt_is} (floor {MIN_N_PER_DST_REGIME})"
    elif not chordia_pass:
        reasons: list[str] = []
        if np.isnan(cell.t_is) or cell.t_is < CHORDIA_T_STRICT:
            reasons.append(f"t_IS={cell.t_is:.3f}<{CHORDIA_T_STRICT}")
        if cell.n_is_on < MIN_N_IS_ON:
            reasons.append(f"N_IS_on={cell.n_is_on}<{MIN_N_IS_ON}")
        if not (cell.expr_is > 0.0):
            reasons.append(f"ExpR_IS={cell.expr_is:.4f}<=0")
        if not bh_reject:
            reasons.append(f"BH q={bh_q:.4f}>=0.05")
        verdict_label = "FAIL_CHORDIA_STRICT"
        verdict_reason = "; ".join(reasons) if reasons else "FAIL_CHORDIA_STRICT"
    elif (
        not cell.dir_match_oos
        and cell.oos_power_tier == "CAN_REFUTE"
        and not (np.isnan(cell.expr_oos))
    ):
        verdict_label = "DEAD_OOS_REFUTES"
        verdict_reason = (
            f"OOS sign opposes IS at CAN_REFUTE power={cell.oos_power:.2f}; "
            f"ExpR_IS={cell.expr_is:.3f} ExpR_OOS={cell.expr_oos:.3f}"
        )
    elif not cell.dir_match_oos:
        verdict_label = "CONDITIONAL_OOS_UNDERPOWERED"
        verdict_reason = (
            f"IS clears strict gate; OOS dir_match=False at tier={cell.oos_power_tier} "
            f"(power={cell.oos_power:.2f}); RULE 3.3 forbids hard kill"
        )
    else:
        verdict_label = "PASS_CHORDIA_STRICT"
        verdict_reason = (
            f"t_IS={cell.t_is:.3f}>={CHORDIA_T_STRICT}; N_IS_on={cell.n_is_on}; "
            f"ExpR_IS={cell.expr_is:.3f}; BH q={bh_q:.4f}; OOS dir_match=True "
            f"at tier={cell.oos_power_tier}"
        )

    return CellVerdict(
        cell=cell,
        bh_q=bh_q,
        bh_reject=bool(bh_reject),
        dst_balance_verdict=dst_balance,
        pass_chordia_strict=chordia_pass and dst_balance == "BALANCED",
        verdict_label=verdict_label,
        verdict_reason=verdict_reason,
    )


def grade_family(stats: list[CellStats]) -> list[CellVerdict]:
    """Apply BH-FDR at K=27 across the family, then per-cell grading."""
    if len(stats) != K_FAMILY:
        raise RuntimeError(
            f"Family-grade called with {len(stats)} cells; K_FAMILY={K_FAMILY}. "
            "Refuse to compose a partial verdict."
        )
    rejects, q_values = compose_bh_fdr(s.p_one_sided for s in stats)
    return [grade_cell(s, q, r) for s, q, r in zip(stats, q_values, rejects)]


def compute_full_verdict(con: duckdb.DuckDBPyConnection) -> list[CellVerdict]:
    """Compute all 27 cell verdicts against a live (or in-memory) DuckDB.

    Stage 4a does NOT invoke this from the CLI. It is reachable from unit
    tests against a synthetic in-memory seed, and from Stage 4b when the
    verdict-emission step is authorised.
    """
    # Cache rows per (orb_minutes, rr_target) -- the same row set is reused
    # across the 3 splits, so fetch each (orb, rr) once.
    fetched: dict[tuple[int, float], list[OutcomeRow]] = {}
    for o in ORB_MINUTES:
        for rr in RR_TARGETS:
            fetched[(o, rr)] = fetch_canonical_cell_rows(con, o, rr)

    stats: list[CellStats] = []
    for spec in enumerate_cells():
        all_rows = fetched[(spec.orb_minutes, spec.rr_target)]
        stats.append(compute_cell_stats(spec, all_rows))
    return grade_family(stats)


def load_promoted_prereg() -> dict[str, object]:
    """Load the promoted prereg and refuse if SHA does not match the lock."""
    path = prereg_path()
    if not path.exists():
        raise HypothesisLoaderError(
            f"Promoted prereg missing at {path}. Stage 4a's git mv from drafts/ "
            "did not land, or the file was moved back."
        )
    meta = load_hypothesis_metadata(path)
    sha = compute_file_sha(path)
    if sha != PROMOTED_PREREG_SHA:
        raise HypothesisLoaderError(
            f"Promoted prereg SHA drift: file SHA {sha} != locked {PROMOTED_PREREG_SHA}. "
            "Body changed post-promotion; refuse to run."
        )
    if meta["name"] != "mnq_nyse_preopen_e2_nfp_spillover_v1":
        raise HypothesisLoaderError(
            f"Promoted prereg name drift: got {meta['name']!r}, expected "
            "'mnq_nyse_preopen_e2_nfp_spillover_v1'."
        )
    if meta["total_expected_trials"] != K_FAMILY:
        raise HypothesisLoaderError(
            f"Promoted prereg N drift: total_expected_trials={meta['total_expected_trials']}, "
            f"runner K_FAMILY={K_FAMILY}. Schema diverged; refuse to run."
        )
    if meta.get("metadata", {}).get("theory_grant", None) is not False:
        raise HypothesisLoaderError(
            "Promoted prereg theory_grant is not explicitly False. "
            "This runner ONLY supports the no-theory strict Chordia path."
        )
    return meta


def check_prereq(con: duckdb.DuckDBPyConnection | None = None) -> list[str]:
    """Return a list of human-readable PASS/FAIL lines for the runner prereqs.

    Stage 4a's main user-facing surface. Touches gold.db read-only IF a
    connection is supplied. If ``con is None``, only the prereg + canonical
    constants are checked.
    """
    lines: list[str] = []
    # 1. Prereg promoted + SHA stable
    try:
        meta = load_promoted_prereg()
        lines.append(f"PASS prereg loaded ({meta['name']}, K={meta['total_expected_trials']})")
        lines.append(f"PASS prereg SHA locked at {PROMOTED_PREREG_SHA[:16]}...")
    except HypothesisLoaderError as e:
        lines.append(f"FAIL prereg: {e}")
        return lines

    # 2. NFP calendar source resolvable
    try:
        nfp_test = is_nfp_day(date(2024, 6, 7))  # known NFP Friday
        if nfp_test is True:
            lines.append("PASS pipeline.calendar_filters.is_nfp_day reachable")
        else:
            lines.append("FAIL is_nfp_day returned False on known NFP Friday 2024-06-07")
    except Exception as e:  # noqa: BLE001
        lines.append(f"FAIL is_nfp_day: {e}")

    # 3. NYSE holiday source wired (used upstream by build_daily_features)
    try:
        if is_nyse_holiday(date(2024, 7, 4)) is True:
            lines.append("PASS pipeline.market_calendar.is_nyse_holiday wired (July 4 2024 -> True)")
        else:
            lines.append("FAIL is_nyse_holiday returned False on July 4 2024 (NYSE-closed)")
    except Exception as e:  # noqa: BLE001
        lines.append(f"FAIL is_nyse_holiday: {e}")

    # 4. OOS power helper
    try:
        pwr = one_sample_power(0.3, 84)
        tier = power_verdict(pwr)
        lines.append(f"PASS research.oos_power reachable (d=0.3 n=84 -> power={pwr:.2f} tier={tier})")
    except Exception as e:  # noqa: BLE001
        lines.append(f"FAIL oos_power: {e}")

    # 5. Canonical layer coverage (gold.db, optional)
    if con is None:
        lines.append("SKIP canonical-coverage check (no gold.db connection supplied)")
    else:
        try:
            covered = con.execute(
                """
                SELECT COUNT(*) AS n, COUNT(DISTINCT trading_day) AS n_td
                FROM orb_outcomes
                WHERE symbol = ?
                  AND orb_label = ?
                  AND entry_model = ?
                  AND confirm_bars = ?
                """,
                [INSTRUMENT, SESSION, ENTRY_MODEL, CONFIRM_BARS],
            ).fetchone()
            n = int(covered[0]) if covered else 0
            n_td = int(covered[1]) if covered else 0
            if n > 0:
                lines.append(
                    f"PASS orb_outcomes coverage: n={n} rows, n_td={n_td} trading days "
                    f"({INSTRUMENT} {SESSION} {ENTRY_MODEL} CB{CONFIRM_BARS})"
                )
            else:
                lines.append(
                    f"FAIL orb_outcomes coverage: 0 rows for {INSTRUMENT} {SESSION} {ENTRY_MODEL} CB{CONFIRM_BARS}"
                )
        except Exception as e:  # noqa: BLE001
            lines.append(f"FAIL canonical-coverage: {e}")

    return lines


def dry_run_cells() -> list[str]:
    """Enumerate the K=27 cells without computing the verdict."""
    return [f"{i + 1:>2}. {spec.cell_id}" for i, spec in enumerate(enumerate_cells())]


def build_parser() -> argparse.ArgumentParser:
    description = (__doc__ or "").split("\n", 1)[0] or "NYSE_PREOPEN Stage 4a runner"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--check-prereq",
        action="store_true",
        help="Validate that the prereg is promoted, SHA-locked, and canonical "
        "sources reachable. Touches gold.db read-only if available. Does not "
        "compute the verdict.",
    )
    parser.add_argument(
        "--dry-run-cells",
        action="store_true",
        help="Print the K=27 cell enumeration and exit. No DB access, no verdict.",
    )
    parser.add_argument(
        "--db",
        default=None,
        help=f"Override DuckDB path. Defaults to pipeline.paths.GOLD_DB_PATH ({GOLD_DB_PATH}).",
    )
    return parser


def _open_read_only(db_path: Path) -> duckdb.DuckDBPyConnection | None:
    """Open gold.db read-only; return None if the lock is held by another process."""
    try:
        return duckdb.connect(str(db_path), read_only=True)
    except duckdb.IOException as e:
        msg = str(e)
        print(f"gold.db locked or unavailable: {msg}", file=sys.stderr)
        return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not (args.check_prereq or args.dry_run_cells):
        # Stage 4a refuses to compute the verdict from the CLI.
        print(
            "Stage 4a runner: pass --check-prereq to validate prerequisites, or "
            "--dry-run-cells to print the K=27 enumeration. The verdict-emission "
            "path is Stage 4b and not exposed here.",
            file=sys.stderr,
        )
        return 2

    if args.dry_run_cells:
        for line in dry_run_cells():
            print(line)
        return 0

    db_override: str | None = args.db
    db_path = Path(db_override) if db_override else GOLD_DB_PATH
    con = _open_read_only(db_path)
    try:
        for line in check_prereq(con):
            print(line)
    finally:
        if con is not None:
            con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
