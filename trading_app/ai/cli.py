"""
Command-line interface for AI queries.

Usage:
    python -m trading_app.ai.cli "Show me all CORE strategies for CME_REOPEN session"
    python -m trading_app.ai.cli --export-csv top10.csv "Top 10 by Sharpe"
    python -m trading_app.ai.cli --show-intent "What columns are in orb_outcomes?"
"""

import argparse
import io
import os
import sys

from pathlib import Path

# Fix Windows cp1252 console encoding -- replace unencodable chars
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding=sys.stdout.encoding, errors="replace"
    )


def _load_env(env_path: Path) -> None:
    """Load key=value pairs from .env file into os.environ."""
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def main():
    parser = argparse.ArgumentParser(
        description="AI-powered query interface for MGC trading data"
    )
    parser.add_argument("query", help="Question in plain English")
    parser.add_argument(
        "--db",
        default=str(Path(__file__).parent.parent.parent / "gold.db"),
        help="Path to DuckDB database (default: gold.db)",
    )
    parser.add_argument(
        "--export-csv",
        metavar="FILE",
        help="Export query results to CSV",
    )
    parser.add_argument(
        "--show-intent",
        action="store_true",
        help="Show extracted query intent (debug mode)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ANTHROPIC_API_KEY"),
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)",
    )

    args = parser.parse_args()

    # Load .env if API key not already set
    _load_env(Path(args.db).parent / ".env")
    if not args.api_key:
        args.api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not args.api_key:
        print("Error: ANTHROPIC_API_KEY required. Set env var or use --api-key.")
        sys.exit(1)

    if not Path(args.db).exists():
        print(f"Error: Database not found: {args.db}")
        sys.exit(1)

    from trading_app.ai.query_agent import QueryAgent

    agent = QueryAgent(db_path=args.db, api_key=args.api_key)
    result = agent.query(args.query)

    # Show intent if requested
    if args.show_intent and result.intent:
        print("=== QUERY INTENT ===")
        print(f"  Template: {result.intent.template.value}")
        print(f"  Parameters: {result.intent.parameters}")
        print(f"  Reason: {result.intent.explanation}")
        print()

    # Show warnings
    if result.warnings:
        print("=== WARNINGS ===")
        for w in result.warnings:
            print(f"  [!] {w}")
        print()

    # Show explanation
    print("=== ANSWER ===")
    print(result.explanation)
    print()

    # Show data summary
    if result.data is not None and not result.data.empty:
        print(f"=== DATA ({len(result.data)} rows) ===")
        print(result.data.to_string(index=False, max_rows=20))
        print()

    # Show grounding references
    if result.grounding_refs:
        print("=== GROUNDING REFS ===")
        for ref in result.grounding_refs:
            print(f"  - {ref}")
        print()

    # Export CSV if requested
    if args.export_csv and result.data is not None:
        result.data.to_csv(args.export_csv, index=False)
        print(f"Exported {len(result.data)} rows to {args.export_csv}")


if __name__ == "__main__":
    main()
