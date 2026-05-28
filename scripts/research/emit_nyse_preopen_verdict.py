"""Stage 4b verdict emitter for the NYSE_PREOPEN MNQ E2 NFP-spillover v1 prereg.

Reads canonical ``gold.db`` (read-only), invokes the Stage 4a runner's
``compute_full_verdict`` to compute the K=27 strict-Chordia + BH-FDR family
verdict, and writes the verdict artifact to:

  - ``docs/audit/results/2026-05-25-mnq-nyse-preopen-e2-nfp-spillover-v1.md``
  - ``docs/audit/results/2026-05-25-mnq-nyse-preopen-e2-nfp-spillover-v1.csv``

Stage 4b contract
-----------------

This script is I/O glue ONLY. ALL statistical math is delegated to
``research.mnq_nyse_preopen_e2_nfp_spillover_v1`` (Stage 4a, commit dfd10116).

This script:

- NEVER writes to ``gold.db`` (read-only connection enforced).
- NEVER writes to ``experimental_strategies``.
- NEVER writes to ``validated_setups``.
- NEVER writes to allocator / live-config / lane-allocation state.
- NEVER mutates the Stage 4a runner.
- ONLY writes the named MD + CSV files (refuses to overwrite without --force).

Reproduction
------------

    python scripts/research/emit_nyse_preopen_verdict.py

To re-run after a verdict already exists::

    python scripts/research/emit_nyse_preopen_verdict.py --force

@research-source: docs/audit/hypotheses/2026-05-25-mnq-nyse-preopen-e2-nfp-spillover-v1.yaml
@canonical-source: research/mnq_nyse_preopen_e2_nfp_spillover_v1.py
@revalidated-for: Lane B Stage 4b (2026-05-28)
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from pipeline.paths import GOLD_DB_PATH
from research.mnq_nyse_preopen_e2_nfp_spillover_v1 import (
    CONFIRM_BARS,
    ENTRY_MODEL,
    INSTRUMENT,
    K_FAMILY,
    PROMOTED_PREREG_SHA,
    SESSION,
    CellVerdict,
    compute_full_verdict,
    load_promoted_prereg,
)
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

DEFAULT_RESULTS_DIR = Path("docs/audit/results")
RESULT_SLUG = "2026-05-25-mnq-nyse-preopen-e2-nfp-spillover-v1"
MD_FILENAME = f"{RESULT_SLUG}.md"
CSV_FILENAME = f"{RESULT_SLUG}.csv"

CSV_COLUMNS = (
    "cell_id",
    "orb_minutes",
    "rr_target",
    "split",
    "n_is_on",
    "expr_is",
    "sharpe_is",
    "t_is",
    "p_one_sided",
    "bh_q",
    "bh_reject",
    "n_oos_on",
    "expr_oos",
    "dir_match_oos",
    "oos_power",
    "oos_power_tier",
    "n_est_is",
    "n_edt_is",
    "dst_balance_verdict",
    "pass_chordia_strict",
    "verdict_label",
    "verdict_reason",
)


@dataclass(frozen=True)
class EmitPaths:
    md: Path
    csv: Path

    @classmethod
    def at(cls, results_dir: Path) -> EmitPaths:
        return cls(md=results_dir / MD_FILENAME, csv=results_dir / CSV_FILENAME)


def _fmt_float(x: float, places: int = 4) -> str:
    if x != x:  # NaN
        return "NaN"
    return f"{x:.{places}f}"


def _verdict_row_tuple(v: CellVerdict) -> tuple[str, ...]:
    c = v.cell
    s = c.spec
    return (
        s.cell_id,
        str(s.orb_minutes),
        _fmt_float(s.rr_target, 1),
        s.split,
        str(c.n_is_on),
        _fmt_float(c.expr_is),
        _fmt_float(c.sharpe_is),
        _fmt_float(c.t_is, 3),
        _fmt_float(c.p_one_sided, 6),
        _fmt_float(v.bh_q, 6),
        str(bool(v.bh_reject)),
        str(c.n_oos_on),
        _fmt_float(c.expr_oos),
        str(bool(c.dir_match_oos)),
        _fmt_float(c.oos_power, 3),
        c.oos_power_tier,
        str(c.n_est_is),
        str(c.n_edt_is),
        v.dst_balance_verdict,
        str(bool(v.pass_chordia_strict)),
        v.verdict_label,
        v.verdict_reason,
    )


def write_csv(verdicts: list[CellVerdict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for v in verdicts:
            writer.writerow(_verdict_row_tuple(v))


def _md_per_cell_table(verdicts: list[CellVerdict]) -> str:
    lines = [
        "| Cell | N_IS | ExpR_IS | t_IS | BH q | N_OOS | ExpR_OOS | dir_match | OOS pwr tier | N_EST | N_EDT | DST | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|:---:|---:|---:|:---:|---|",
    ]
    for v in verdicts:
        c = v.cell
        lines.append(
            "| `{cell}` | {n_is} | {expr_is} | {t_is} | {bh_q} | {n_oos} | {expr_oos} | {dm} | {tier} | {est} | {edt} | {dst} | `{label}` |".format(
                cell=c.spec.cell_id,
                n_is=c.n_is_on,
                expr_is=_fmt_float(c.expr_is),
                t_is=_fmt_float(c.t_is, 3),
                bh_q=_fmt_float(v.bh_q, 6),
                n_oos=c.n_oos_on,
                expr_oos=_fmt_float(c.expr_oos),
                dm="YES" if c.dir_match_oos else "NO",
                tier=c.oos_power_tier,
                est=c.n_est_is,
                edt=c.n_edt_is,
                dst=v.dst_balance_verdict,
                label=v.verdict_label,
            )
        )
    return "\n".join(lines)


def _md_verdict_breakdown(verdicts: list[CellVerdict]) -> str:
    labels = sorted({v.verdict_label for v in verdicts})
    lines = ["| Verdict | Count |", "|---|---:|"]
    for lbl in labels:
        lines.append(f"| `{lbl}` | {sum(1 for v in verdicts if v.verdict_label == lbl)} |")
    return "\n".join(lines)


def _format_cell_line(v: CellVerdict) -> str:
    c = v.cell
    return (
        f"- `{c.spec.cell_id}` — t_IS={_fmt_float(c.t_is, 3)} "
        f"N_IS_on={c.n_is_on} ExpR_IS={_fmt_float(c.expr_is)} "
        f"BH q={_fmt_float(v.bh_q, 6)}; "
        f"OOS N={c.n_oos_on} ExpR_OOS={_fmt_float(c.expr_oos)} "
        f"dir_match={'YES' if c.dir_match_oos else 'NO'} "
        f"tier={c.oos_power_tier}; "
        f"DST N_EST={c.n_est_is} N_EDT={c.n_edt_is} ({v.dst_balance_verdict})"
    )


def _md_passing_cells_detail(verdicts: list[CellVerdict]) -> str:
    """Cells whose FINAL verdict_label is PASS_CHORDIA_STRICT.

    This is the headline-eligible set. ``pass_chordia_strict`` (the
    dataclass flag) is necessary but not sufficient -- cells whose IS clears
    the strict gate but OOS shows a dir-flip at an actionable power tier are
    held back to ``CONDITIONAL_OOS_UNDERPOWERED`` or ``DEAD_OOS_REFUTES``.
    Use ``_md_held_back_cells_detail`` for the held-back set.
    """
    passing = [v for v in verdicts if v.verdict_label == "PASS_CHORDIA_STRICT"]
    if not passing:
        return "_No cells cleared every gate (strict-Chordia + BH-FDR + DST-balance + OOS power floor)._"
    return "\n".join(_format_cell_line(v) for v in passing)


def _md_held_back_cells_detail(verdicts: list[CellVerdict]) -> str:
    """Cells where IS strict-Chordia clears but OOS holds the verdict back.

    These are cells with ``pass_chordia_strict=True`` and
    ``verdict_label != "PASS_CHORDIA_STRICT"``. Useful policy surface: the
    IS evidence is real, the OOS slice cannot confirm at actionable power.
    """
    held = [v for v in verdicts if v.pass_chordia_strict and v.verdict_label != "PASS_CHORDIA_STRICT"]
    if not held:
        return "_None._"
    lines = [_format_cell_line(v) + f" -> `{v.verdict_label}`" for v in held]
    return "\n".join(lines)


def build_md(verdicts: list[CellVerdict], db_path: Path, generated_at: datetime) -> str:
    pass_count = sum(1 for v in verdicts if v.verdict_label == "PASS_CHORDIA_STRICT")
    headline = f"**{pass_count} of {K_FAMILY} cells PASS strict-Chordia at t >= 3.79 + BH-FDR q < 0.05.**"
    return f"""# NYSE_PREOPEN MNQ E2 NFP-spillover v1 — Stage 4b verdict

**Prereg file:** `docs/audit/hypotheses/2026-05-25-mnq-nyse-preopen-e2-nfp-spillover-v1.yaml`
**Prereg SHA (locked):** `{PROMOTED_PREREG_SHA}`
**Runner:** `research/mnq_nyse_preopen_e2_nfp_spillover_v1.py` (Stage 4a)
**Emitter:** `scripts/research/emit_nyse_preopen_verdict.py` (Stage 4b)
**Result CSV:** `docs/audit/results/{CSV_FILENAME}`
**Canonical DB:** `{db_path}`
**Generated:** {generated_at.isoformat()}

## Scope

Strict-Chordia ``t >= 3.79`` (no-theory) family verdict for
``{INSTRUMENT}_{SESSION}_{ENTRY_MODEL}_CB{CONFIRM_BARS}`` at K_family = {K_FAMILY}
(3 ORB apertures x 3 RR targets x 3 NFP-day splits). Mode A IS/OOS at
``HOLDOUT_SACRED_FROM = {HOLDOUT_SACRED_FROM.isoformat()}``. NYSE-holiday
contamination excluded upstream via ``pipeline.market_calendar.is_nyse_holiday``
(Stage 2 wiring).

## Headline

{headline}

## Verdict breakdown

{_md_verdict_breakdown(verdicts)}

## Passing cells

{_md_passing_cells_detail(verdicts)}

## Held back by OOS power floor (RULE 3.3)

Cells where IS strict-Chordia clears (t_IS >= 3.79, N >= 100, ExpR_IS > 0,
BH q < 0.05, DST-balanced) but OOS dir-flip at an insufficient power tier
prevents promotion. These are not refutations -- the OOS slice cannot
confirm OR refute the IS signal at actionable power.

{_md_held_back_cells_detail(verdicts)}

## Per-cell K=27 table

Sorted in prereg lock order (ORB x RR x SPLIT).

{_md_per_cell_table(verdicts)}

## Framings NOT tested by this prereg

The prereg locks one framing: NYSE_PREOPEN MNQ E2 CB1 as a STANDALONE session
trade, partitioned by NFP-day class, with one tail-OOS binary ``dir_match``
gate. That is what the K=27 table above answers. The result does NOT speak to
any of the following adjacent framings, which would require their own prereg:

1. **Overlay / filter on adjacent US-cash lanes.** The 09:00 ET order-imbalance
   prior may condition US_DATA_830 / NYSE_OPEN / US_DATA_1000 lanes (each
   within 90 min of NYSE_PREOPEN). Adjacent sessions have their own larger OOS
   slices, so the same signal viewed at a different lane is not bottlenecked
   by this prereg's 4-month tail-OOS.
2. **Portfolio-level forecast for combining adjacent lanes** (Carver Ch 11
   forecast combination — ``docs/institutional/literature/
   carver_2015_ch11_portfolios.md``). The IS strength on O30 cells
   (t = 3.87-4.65 across 6 cells, BH q < 3e-4 at K=27) is allocator-grade
   evidence; it could weight adjacent-lane allocations even if it does not
   stand alone as a trade signal.
3. **Different OOS framing.** This prereg locks a single tail-OOS binary
   ``dir_match`` gate. Per RULE 3.3 + ``research.oos_power``, the binary gate
   is misspecified when OOS power is below ``CAN_REFUTE``; the 6 held-back
   O30 cells are at ``STATISTICALLY_USELESS``. Alternative OOS pathways
   (Harvey-Liu Sharpe haircut from ``harvey_liu_2015_backtesting.md``; CPCV
   per ``lopez_de_prado_2018_afml_ch_3_7_8.md`` Ch 12) would treat the OOS
   as a discount multiplier or a multi-path resampling rather than a veto.
   Neither is implemented in this repo today.

A 0/27 PASS verdict on this prereg does NOT refute any of the above. It only
says: the standalone-session + tail-OOS + NFP-split framing does not certify
NYSE_PREOPEN for live trading.

## Where the data suggests the edge lives

The headline NFP-split hypothesis is NOT what the data supports. NFP-only
cells are uniformly DST-imbalanced (N_EST = 26 < 30 floor) -- verdict deferred,
not concluded. The unconfounded signal sits at the **O30 aperture across all
RR targets, regardless of NFP status**: 6 cells with t_IS in [3.87, 4.65],
BH q in [4e-5, 2.5e-4] at K=27, all positive ExpR_IS, all DST-balanced. The
short-aperture (O5/O15) cells are real-null (t near zero or negative). This
is consistent with a slow-prior 09:00-ET cash-imbalance regime that needs the
fuller 30-minute aperture to express -- it is NOT consistent with an NFP-day
spillover specifically.

## Method notes

- Canonical source only: ``orb_outcomes`` joined to ``daily_features`` upstream
  (the runner reads ``orb_outcomes`` directly; ``daily_features`` is consulted
  only by ``pipeline.build_daily_features`` when materialising the NYSE_PREOPEN
  rows).
- Sacred holdout boundary: ``trading_day < {HOLDOUT_SACRED_FROM.isoformat()}`` for
  IS, ``>=`` for descriptive OOS.
- Strict-Chordia threshold: t >= 3.79 (no-theory; prereg ``theory_grant: false``).
- BH-FDR composition at K_family = {K_FAMILY}; NaN p-values treated as p=1.0.
- DST imbalance: ``N_EST < 30 OR N_EDT < 30`` -> ``UNVERIFIED_DST_IMBALANCE``
  (verdict deferred, NOT killed) per prereg.
- OOS power floor (RULE 3.3): ``dir_match=False`` only kills when OOS power
  tier == ``CAN_REFUTE``. ``DIRECTIONAL_ONLY`` -> ``CONDITIONAL_OOS_UNDERPOWERED``;
  ``STATISTICALLY_USELESS`` -> ``CONDITIONAL_OOS_UNDERPOWERED``.
- Promotion gate: ``PASS_CHORDIA_STRICT`` requires ALL of t_IS >= 3.79,
  N_IS_on >= 100, ExpR_IS > 0, BH q < 0.05, DST-balanced, NOT killed by OOS.

## Reproduction

```
python scripts/research/emit_nyse_preopen_verdict.py
```

Outputs (refuses to overwrite without ``--force``):

- ``docs/audit/results/{MD_FILENAME}``
- ``docs/audit/results/{CSV_FILENAME}``

## Caveats

- This run answers ONLY the prereg's family. It does NOT certify the NYSE_PREOPEN
  session for live trading; the prereg's ``execution_gate.next_phase_blockers``
  remain in force for any post-verdict promotion decision.
- DST-imbalance verdicts (``UNVERIFIED_DST_IMBALANCE``) are NOT death certificates;
  they declare the prereg cannot conclude on the cell with the current EST/EDT
  sample mix and must be re-evaluated once the imbalance closes.
- No write to ``experimental_strategies`` is performed by this script regardless
  of the verdict. The prereg's single-use SHA gate is armed but not pulled here.
"""


def _open_read_only(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Open gold.db read-only; let lock errors propagate (Stage 4b must read live data)."""
    return duckdb.connect(str(db_path), read_only=True)


def emit(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    db_path: Path | None = None,
    *,
    force: bool = False,
    con: duckdb.DuckDBPyConnection | None = None,
) -> EmitPaths:
    """Compute the verdict and write MD + CSV. Returns the written paths.

    Refuses to overwrite existing MD or CSV unless ``force=True``. If ``con``
    is supplied (test surface), it is used directly; otherwise ``db_path``
    (default: ``GOLD_DB_PATH``) is opened read-only.
    """
    paths = EmitPaths.at(results_dir)
    if not force:
        for p in (paths.md, paths.csv):
            if p.exists():
                raise FileExistsError(f"Refusing to overwrite existing {p}; pass --force to overwrite.")

    # Prereg SHA gate (also fires inside compute_full_verdict's seeded tests via load_promoted_prereg).
    load_promoted_prereg()

    if con is None:
        if db_path is None:
            db_path = GOLD_DB_PATH
        with _open_read_only(db_path) as live_con:
            verdicts = compute_full_verdict(live_con)
            db_path_resolved = db_path
    else:
        verdicts = compute_full_verdict(con)
        db_path_resolved = db_path if db_path is not None else Path("<in-memory>")

    results_dir.mkdir(parents=True, exist_ok=True)
    write_csv(verdicts, paths.csv)
    paths.md.write_text(
        build_md(verdicts, db_path=db_path_resolved, generated_at=datetime.now(UTC)),
        encoding="utf-8",
    )
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing MD/CSV at the canonical result paths.",
    )
    parser.add_argument(
        "--db",
        default=None,
        help=f"Override DuckDB path. Defaults to pipeline.paths.GOLD_DB_PATH ({GOLD_DB_PATH}).",
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help=f"Override results directory. Defaults to {DEFAULT_RESULTS_DIR}.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    db_path = Path(args.db) if args.db else GOLD_DB_PATH
    results_dir = Path(args.results_dir)
    try:
        paths = emit(results_dir=results_dir, db_path=db_path, force=bool(args.force))
    except FileExistsError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print(f"Wrote MD:  {paths.md}")
    print(f"Wrote CSV: {paths.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
