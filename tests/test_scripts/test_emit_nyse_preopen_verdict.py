"""Unit tests for the Stage 4b NYSE_PREOPEN verdict emitter.

Stage 4b contract: the emitter is I/O glue around the Stage 4a runner. These
tests exercise the emitter against the SAME synthetic in-memory DuckDB seed
the Stage 4a tests use, so the math is canonical-source delegated. Tests do
NOT touch canonical gold.db.
"""

from __future__ import annotations

import csv
from collections.abc import Generator
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pytest

from research.mnq_nyse_preopen_e2_nfp_spillover_v1 import (
    K_FAMILY,
    ORB_MINUTES,
    PROMOTED_PREREG_SHA,
    RR_TARGETS,
)
from scripts.research.emit_nyse_preopen_verdict import (
    CSV_COLUMNS,
    CSV_FILENAME,
    MD_FILENAME,
    EmitPaths,
    build_parser,
    emit,
    main,
)


# ---------------------------------------------------------------------------
# Synthetic in-memory DuckDB seed (mirrors the Stage 4a fixture exactly).
# ---------------------------------------------------------------------------


def _seed_synthetic_db(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE orb_outcomes (
            trading_day DATE,
            symbol VARCHAR,
            orb_label VARCHAR,
            orb_minutes INTEGER,
            rr_target DOUBLE,
            confirm_bars INTEGER,
            entry_model VARCHAR,
            entry_ts TIMESTAMPTZ,
            pnl_r DOUBLE
        )
        """
    )
    is_dates: list[date] = []
    for i in range(200):
        if i < 100:
            d = date(2022, 1, 3 + (i % 28))
        else:
            d = date(2022, 7, 1 + (i % 30))
        is_dates.append(d)
    oos_dates = [date(2026, 1, 5) + timedelta(days=i) for i in range(30)]
    all_dates = is_dates + oos_dates

    for o in ORB_MINUTES:
        for rr in RR_TARGETS:
            for i, d in enumerate(all_dates):
                pnl = 0.10 + (1.0 if i % 2 == 0 else -1.0)
                con.execute(
                    "INSERT INTO orb_outcomes VALUES (?, 'MNQ', 'NYSE_PREOPEN', ?, ?, 1, 'E2', NULL, ?)",
                    [d, o, rr, pnl],
                )


@pytest.fixture
def seeded_con() -> Generator[duckdb.DuckDBPyConnection, None, None]:
    con = duckdb.connect(":memory:")
    _seed_synthetic_db(con)
    try:
        yield con
    finally:
        con.close()


# ---------------------------------------------------------------------------
# CSV layout
# ---------------------------------------------------------------------------


class TestCsv:
    def test_csv_has_header_and_27_rows(self, seeded_con: duckdb.DuckDBPyConnection, tmp_path: Path) -> None:
        paths = emit(results_dir=tmp_path, con=seeded_con)
        with paths.csv.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        assert rows[0] == list(CSV_COLUMNS)
        assert len(rows) == 1 + K_FAMILY  # header + 27 cell rows

    def test_csv_column_contract_stable(self) -> None:
        # Lock the canonical CSV column set so downstream consumers
        # (Pinecone upsert, allocator review) can rely on it.
        assert "cell_id" in CSV_COLUMNS
        assert "t_is" in CSV_COLUMNS
        assert "bh_q" in CSV_COLUMNS
        assert "verdict_label" in CSV_COLUMNS
        assert "pass_chordia_strict" in CSV_COLUMNS
        assert "dst_balance_verdict" in CSV_COLUMNS


# ---------------------------------------------------------------------------
# MD layout
# ---------------------------------------------------------------------------


class TestMd:
    def test_md_header_and_required_sections(self, seeded_con: duckdb.DuckDBPyConnection, tmp_path: Path) -> None:
        paths = emit(results_dir=tmp_path, con=seeded_con)
        text = paths.md.read_text(encoding="utf-8")
        # Title
        assert "NYSE_PREOPEN MNQ E2 NFP-spillover v1 — Stage 4b verdict" in text
        # Prereg SHA lock surfaced
        assert PROMOTED_PREREG_SHA in text
        # Sections present
        assert "## Scope" in text
        assert "## Headline" in text
        assert "## Verdict breakdown" in text
        assert "## Passing cells" in text
        assert "## Held back by OOS power floor" in text
        assert "## Per-cell K=27 table" in text
        assert "## Framings NOT tested by this prereg" in text
        assert "## Where the data suggests the edge lives" in text
        assert "## Method notes" in text
        assert "## Reproduction" in text
        # Headline declares pass count out of K_FAMILY
        assert f"of {K_FAMILY} cells PASS strict-Chordia" in text

    def test_md_per_cell_table_has_27_rows(self, seeded_con: duckdb.DuckDBPyConnection, tmp_path: Path) -> None:
        paths = emit(results_dir=tmp_path, con=seeded_con)
        text = paths.md.read_text(encoding="utf-8")
        # The per-cell table renders each cell_id as a backtick token; count occurrences.
        # 27 cells x 1 occurrence in the table + 1 occurrence per passing cell in detail.
        # Be conservative: assert each unique cell_id appears at least once.
        for o in ORB_MINUTES:
            for rr in RR_TARGETS:
                for split in ("all_days", "nfp_days_only", "non_nfp_days"):
                    cell_id = f"O{o}_RR{rr}_{split}"
                    assert f"`{cell_id}`" in text, f"Missing cell {cell_id} in MD"


# ---------------------------------------------------------------------------
# Overwrite protection
# ---------------------------------------------------------------------------


class TestOverwriteProtection:
    def test_refuses_to_overwrite_md(self, seeded_con: duckdb.DuckDBPyConnection, tmp_path: Path) -> None:
        emit(results_dir=tmp_path, con=seeded_con)
        # Second invocation without --force must raise
        with pytest.raises(FileExistsError):
            emit(results_dir=tmp_path, con=seeded_con)

    def test_refuses_to_overwrite_csv_only(self, seeded_con: duckdb.DuckDBPyConnection, tmp_path: Path) -> None:
        # CSV exists but MD does not -> still refuse, fail-closed.
        (tmp_path / CSV_FILENAME).write_text("placeholder", encoding="utf-8")
        with pytest.raises(FileExistsError):
            emit(results_dir=tmp_path, con=seeded_con)

    def test_force_overrides(self, seeded_con: duckdb.DuckDBPyConnection, tmp_path: Path) -> None:
        emit(results_dir=tmp_path, con=seeded_con)
        # Second invocation with force=True must succeed
        paths = emit(results_dir=tmp_path, con=seeded_con, force=True)
        assert paths.md.exists()
        assert paths.csv.exists()


# ---------------------------------------------------------------------------
# Read-only DB enforcement
# ---------------------------------------------------------------------------


class TestReadOnlyDb:
    def test_emit_writes_no_db_rows(self, seeded_con: duckdb.DuckDBPyConnection, tmp_path: Path) -> None:
        before = seeded_con.execute("SELECT COUNT(*) FROM orb_outcomes").fetchone()
        assert before is not None
        n_before = int(before[0])
        emit(results_dir=tmp_path, con=seeded_con)
        after = seeded_con.execute("SELECT COUNT(*) FROM orb_outcomes").fetchone()
        assert after is not None
        assert int(after[0]) == n_before, "Emitter must not write to orb_outcomes"

    def test_read_only_connection_open_then_write_raises(self, tmp_path: Path) -> None:
        # Build a real on-disk DuckDB so we can re-open it read-only.
        db_path = tmp_path / "tmp_gold.db"
        # Seed with the canonical schema
        with duckdb.connect(str(db_path)) as wcon:
            _seed_synthetic_db(wcon)
        # Re-open read-only and confirm write attempts raise
        rocon = duckdb.connect(str(db_path), read_only=True)
        try:
            with pytest.raises(duckdb.Error):
                rocon.execute(
                    "INSERT INTO orb_outcomes VALUES (?, 'MNQ', 'NYSE_PREOPEN', 5, 1.0, 1, 'E2', NULL, 0.1)",
                    [date(2099, 1, 1)],
                )
        finally:
            rocon.close()


# ---------------------------------------------------------------------------
# CLI surface + argparse contract
# ---------------------------------------------------------------------------


class TestCli:
    def test_build_parser_has_force_and_db_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        assert args.force is False
        assert args.db is None
        args = parser.parse_args(["--force", "--db", "C:/tmp/x.db", "--results-dir", "out/"])
        assert args.force is True
        assert args.db == "C:/tmp/x.db"
        assert args.results_dir == "out/"

    def test_main_writes_artifacts(self, tmp_path: Path) -> None:
        # Write a real DuckDB at tmp_path and point --db at it; main()
        # opens read-only and emits MD + CSV to --results-dir.
        db_path = tmp_path / "tmp_gold.db"
        with duckdb.connect(str(db_path)) as wcon:
            _seed_synthetic_db(wcon)
        rc = main(["--db", str(db_path), "--results-dir", str(tmp_path)])
        assert rc == 0
        assert (tmp_path / MD_FILENAME).exists()
        assert (tmp_path / CSV_FILENAME).exists()

    def test_main_refuses_overwrite_returns_2(self, tmp_path: Path) -> None:
        db_path = tmp_path / "tmp_gold.db"
        with duckdb.connect(str(db_path)) as wcon:
            _seed_synthetic_db(wcon)
        rc = main(["--db", str(db_path), "--results-dir", str(tmp_path)])
        assert rc == 0
        # Second run, no --force
        rc2 = main(["--db", str(db_path), "--results-dir", str(tmp_path)])
        assert rc2 == 2


# ---------------------------------------------------------------------------
# Stage 4a runner regression check
# ---------------------------------------------------------------------------


class TestStage4aIntegrity:
    def test_csv_columns_match_runner_dataclass_fields(self) -> None:
        # Build_md and write_csv must keep the column set aligned with what
        # the runner's CellVerdict surfaces. This locks the I/O contract to
        # the canonical-source dataclass.
        from research.mnq_nyse_preopen_e2_nfp_spillover_v1 import CellStats, CellVerdict

        cell_stats_fields = {f for f in CellStats.__dataclass_fields__}
        cell_verdict_fields = {f for f in CellVerdict.__dataclass_fields__}
        # Sanity: the union spans the fields the CSV reports (excluding spec
        # subfields, which are flattened into orb_minutes / rr_target / split).
        flattened_csv_cols = set(CSV_COLUMNS) - {"orb_minutes", "rr_target", "split", "cell_id"}
        for col in flattened_csv_cols:
            assert col in cell_stats_fields or col in cell_verdict_fields, (
                f"CSV column {col!r} not present on CellStats or CellVerdict; I/O drift would silently lose a field."
            )


# ---------------------------------------------------------------------------
# EmitPaths convenience
# ---------------------------------------------------------------------------


class TestEmitPaths:
    def test_at_constructs_expected_filenames(self, tmp_path: Path) -> None:
        paths = EmitPaths.at(tmp_path)
        assert paths.md == tmp_path / MD_FILENAME
        assert paths.csv == tmp_path / CSV_FILENAME


# ---------------------------------------------------------------------------
# Build_md robustness against empty / NaN cells
# ---------------------------------------------------------------------------


class TestBuildMdRobustness:
    def test_build_md_no_passing_cells_shows_explicit_message(
        self, seeded_con: duckdb.DuckDBPyConnection, tmp_path: Path
    ) -> None:
        # The synthetic seed produces ExpR_IS ~= 0.1 across all cells with
        # alternating wins/losses; t_IS values come out below the 3.79 strict
        # hurdle, so no cells should PASS. Verify the MD renders the
        # "no cells cleared" message rather than crashing on the empty list.
        paths = emit(results_dir=tmp_path, con=seeded_con)
        text = paths.md.read_text(encoding="utf-8")
        # If no cells pass, "Passing cells" section shows the explicit message.
        passing_section = text.split("## Passing cells", 1)[1].split("## ", 1)[0]
        # "Passing cells" section is rendered against verdict_label == PASS_CHORDIA_STRICT.
        # Either the seed produced a pass label (acceptable) ...
        if "PASS_CHORDIA_STRICT" not in passing_section:
            assert (
                "_No cells cleared every gate (strict-Chordia + BH-FDR + DST-balance + OOS power floor)._"
                in passing_section
            )
