"""
Cross-session conditional probability table.

Pre-computes P(session_B outcome | session_A outcome, direction_relation)
from historical orb_outcomes data.

Read-only consumer of the database -- never writes to DB.

Usage:
    table = build_cascade_table(Path("gold.db"))
    entry = table[("0900", "loss", "opposite")]
    print(entry)  # {"1000_wr": 0.52, "n": 148}
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb


def build_cascade_table(
    db_path: Path | str,
    orb_minutes: int = 5,
) -> dict[tuple[str, str, str], dict]:
    """Build cross-session conditional probability table from historical data.

    Joins orb_outcomes for session pairs on the same trading_day and computes:
      P(session_B win | session_A outcome, direction_relation)

    Returns nested dict keyed by (session_A_label, outcome_A, direction_relation):
      {
        ("0900", "loss", "opposite"): {"1000_wr": 0.52, "n": 148},
        ("0900", "win", "same"):      {"1000_wr": 0.61, "n": 203},
        ...
      }

    Direction relation is "same" if both sessions break the same direction,
    "opposite" if they break in opposite directions.
    """
    # Session pairs to analyze: (earlier, later)
    pairs = [
        ("0900", "1000"),
        ("0900", "1100"),
        ("1000", "1100"),
        ("1800", "2300"),
        ("2300", "0030"),
    ]

    table = {}
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        for sess_a, sess_b in pairs:
            rows = con.execute("""
                SELECT
                    da.orb_{sa}_outcome AS outcome_a,
                    da.orb_{sa}_break_dir AS dir_a,
                    da.orb_{sb}_break_dir AS dir_b,
                    da.orb_{sb}_outcome AS outcome_b
                FROM daily_features da
                WHERE da.symbol = 'MGC'
                  AND da.orb_minutes = ?
                  AND da.orb_{sa}_outcome IS NOT NULL
                  AND da.orb_{sb}_outcome IS NOT NULL
                  AND da.orb_{sa}_break_dir IS NOT NULL
                  AND da.orb_{sb}_break_dir IS NOT NULL
            """.format(sa=sess_a, sb=sess_b), [orb_minutes]).fetchall()

            # Group by (outcome_a, direction_relation)
            groups: dict[tuple[str, str], list[str]] = {}
            for outcome_a, dir_a, dir_b, outcome_b in rows:
                if not all([outcome_a, dir_a, dir_b, outcome_b]):
                    continue
                same = dir_a.upper() == dir_b.upper()
                rel = "same" if same else "opposite"
                key = (outcome_a, rel)
                if key not in groups:
                    groups[key] = []
                groups[key].append(outcome_b)

            # Compute conditional win rates
            for (outcome_a, rel), outcomes_b in groups.items():
                n = len(outcomes_b)
                if n < 5:
                    continue  # Too few samples
                wins = sum(1 for o in outcomes_b if o == "win")
                wr = wins / n if n > 0 else 0.0
                table_key = (sess_a, outcome_a, rel)
                table[table_key] = {
                    f"{sess_b}_wr": round(wr, 4),
                    "n": n,
                }

    finally:
        con.close()

    return table


def lookup_cascade(
    table: dict,
    session_a: str,
    outcome_a: str,
    direction_relation: str,
) -> dict | None:
    """Look up conditional probability from the cascade table.

    Returns dict with "{session_b}_wr" and "n" keys, or None if not found.
    """
    return table.get((session_a, outcome_a, direction_relation))
