"""
Generate markdown snapshots of trading system state for Pinecone Assistant sync.

Four snapshot generators query gold-db and produce markdown documents:
  1. Portfolio state (strategy counts, edge families by instrument)
  2. Fitness report (breakdowns by session, entry model, aperture; top strategies)
  3. Live config (LIVE_PORTFOLIO specs, tiers, gates)
  4. Research index (research/output/ file listing with summaries)

Usage:
    python scripts/tools/pinecone_snapshots.py              # generate all snapshots
    python scripts/tools/pinecone_snapshots.py --snapshot portfolio_state
    python scripts/tools/pinecone_snapshots.py --list        # list available snapshots
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Force unbuffered stdout (Windows cp1252 buffering issue)
sys.stdout.reconfigure(line_buffering=True)

import duckdb

from pipeline.paths import GOLD_DB_PATH


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def save_snapshot(content: str, snapshot_name: str) -> Path:
    """Save snapshot content to scripts/tools/{snapshot_name}. Return path."""
    output_path = PROJECT_ROOT / "scripts" / "tools" / snapshot_name
    output_path.write_text(content, encoding="utf-8")
    return output_path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# 1. Portfolio State Snapshot
# ---------------------------------------------------------------------------

def generate_portfolio_state_snapshot() -> str:
    """Strategy counts and edge family summary by instrument."""
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        # Strategy counts by instrument
        strat_rows = con.execute("""
            SELECT instrument,
                COUNT(*) FILTER (WHERE status='active') as active_count,
                COUNT(*) FILTER (WHERE status='active' AND fdr_significant=true) as fdr_count,
                COUNT(*) FILTER (WHERE status='active' AND sample_size >= 100) as core_count,
                COUNT(*) FILTER (WHERE status='active' AND sample_size BETWEEN 30 AND 99) as regime_count
            FROM validated_setups
            GROUP BY instrument ORDER BY instrument
        """).fetchall()

        # Edge family counts by instrument
        family_rows = con.execute("""
            SELECT instrument,
                COUNT(*) as family_count,
                COUNT(*) FILTER (WHERE robustness_status='ROBUST') as robust_count,
                COUNT(*) FILTER (WHERE robustness_status='WHITELISTED') as whitelisted_count
            FROM edge_families
            GROUP BY instrument ORDER BY instrument
        """).fetchall()

        # Total active
        total_active = sum(r[1] for r in strat_rows)
        total_fdr = sum(r[2] for r in strat_rows)
        total_core = sum(r[3] for r in strat_rows)
        total_regime = sum(r[4] for r in strat_rows)
        total_families = sum(r[1] for r in family_rows)
        total_robust = sum(r[2] for r in family_rows)
        total_whitelisted = sum(r[3] for r in family_rows)

    finally:
        con.close()

    lines = [
        "# Portfolio State Snapshot",
        "",
        f"Generated: {_utc_now_iso()}",
        "",
        "## Validated Strategies by Instrument",
        "",
        "| Instrument | Active | FDR | CORE (N>=100) | REGIME (30-99) |",
        "|------------|--------|-----|---------------|----------------|",
    ]
    for inst, active, fdr, core, regime in strat_rows:
        lines.append(f"| {inst} | {active} | {fdr} | {core} | {regime} |")
    lines.append(f"| **TOTAL** | **{total_active}** | **{total_fdr}** | **{total_core}** | **{total_regime}** |")

    lines += [
        "",
        "## Edge Families by Instrument",
        "",
        "| Instrument | Families | ROBUST | WHITELISTED |",
        "|------------|----------|--------|-------------|",
    ]
    for inst, fam_count, robust, whitelisted in family_rows:
        lines.append(f"| {inst} | {fam_count} | {robust} | {whitelisted} |")
    lines.append(f"| **TOTAL** | **{total_families}** | **{total_robust}** | **{total_whitelisted}** |")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Fitness Report Snapshot
# ---------------------------------------------------------------------------

def generate_fitness_report_snapshot() -> str:
    """Active strategy breakdown by session, entry model, aperture; top 10 by Sharpe."""
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        # Breakdown by instrument, session, entry model, aperture
        breakdown_rows = con.execute("""
            SELECT instrument, orb_label, entry_model, orb_minutes,
                COUNT(*) as count,
                ROUND(AVG(expectancy_r), 3) as avg_exp_r,
                ROUND(AVG(sharpe_ann), 2) as avg_sharpe
            FROM validated_setups WHERE status='active'
            GROUP BY instrument, orb_label, entry_model, orb_minutes
            ORDER BY instrument, orb_label, entry_model, orb_minutes
        """).fetchall()

        # Top 10 by Sharpe
        top_rows = con.execute("""
            SELECT strategy_id, instrument, orb_label, entry_model,
                orb_minutes, sample_size, expectancy_r, sharpe_ann,
                win_rate, fdr_significant
            FROM validated_setups
            WHERE status='active'
            ORDER BY sharpe_ann DESC NULLS LAST
            LIMIT 10
        """).fetchall()

    finally:
        con.close()

    lines = [
        "# Fitness Report Snapshot",
        "",
        f"Generated: {_utc_now_iso()}",
        "",
        "## Strategy Breakdown (Active Only)",
        "",
        "| Instrument | Session | Entry | Aperture | Count | Avg ExpR | Avg Sharpe |",
        "|------------|---------|-------|----------|-------|----------|------------|",
    ]
    for inst, orb, entry, aperture, count, avg_exp, avg_sh in breakdown_rows:
        lines.append(
            f"| {inst} | {orb} | {entry} | {aperture}m | {count} | "
            f"{avg_exp:+.3f} | {avg_sh:.2f} |"
        )

    lines += [
        "",
        "## Top 10 Strategies by Annualized Sharpe",
        "",
        "| Strategy ID | Instrument | Session | Entry | Aperture | N | ExpR | Sharpe | WR | FDR |",
        "|-------------|------------|---------|-------|----------|---|------|--------|----|----|",
    ]
    for (sid, inst, orb, entry, aperture, n, exp_r, sharpe, wr, fdr) in top_rows:
        fdr_mark = "Y" if fdr else "N"
        lines.append(
            f"| {sid} | {inst} | {orb} | {entry} | {aperture}m | "
            f"{n} | {exp_r:+.3f} | {sharpe:.2f} | {wr:.0%} | {fdr_mark} |"
        )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Live Config Snapshot
# ---------------------------------------------------------------------------

def generate_live_config_snapshot() -> str:
    """Live portfolio specs, tier counts, and gates from trading_app.live_config."""
    from trading_app.live_config import (
        LIVE_PORTFOLIO,
        LIVE_MIN_EXPECTANCY_R,
        LIVE_MIN_EXPECTANCY_DOLLARS_MULT,
    )

    # Count by tier
    tier_counts: dict[str, int] = {}
    for spec in LIVE_PORTFOLIO:
        tier_counts[spec.tier] = tier_counts.get(spec.tier, 0) + 1

    lines = [
        "# Live Config Snapshot",
        "",
        f"Generated: {_utc_now_iso()}",
        "",
        "## Portfolio Gates",
        "",
        f"- **LIVE_MIN_EXPECTANCY_R:** {LIVE_MIN_EXPECTANCY_R}",
        f"- **LIVE_MIN_EXPECTANCY_DOLLARS_MULT:** {LIVE_MIN_EXPECTANCY_DOLLARS_MULT}",
        "",
        "## Tier Summary",
        "",
        "| Tier | Count |",
        "|------|-------|",
    ]
    for tier in sorted(tier_counts):
        lines.append(f"| {tier.upper()} | {tier_counts[tier]} |")
    lines.append(f"| **TOTAL** | **{len(LIVE_PORTFOLIO)}** |")

    lines += [
        "",
        "## Strategy Specs",
        "",
        "| # | Family ID | Tier | Session | Entry | Filter | Regime Gate |",
        "|---|-----------|------|---------|-------|--------|-------------|",
    ]
    for i, spec in enumerate(LIVE_PORTFOLIO, 1):
        gate = spec.regime_gate or "always-on"
        lines.append(
            f"| {i} | {spec.family_id} | {spec.tier.upper()} | "
            f"{spec.orb_label} | {spec.entry_model} | {spec.filter_type} | {gate} |"
        )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Research Index Snapshot
# ---------------------------------------------------------------------------

def generate_research_index_snapshot() -> str:
    """Scan research/output/ for .md and .txt files, group by prefix, show summary."""
    research_dir = PROJECT_ROOT / "research" / "output"

    if not research_dir.exists():
        return "# Research Index Snapshot\n\nNo research/output/ directory found.\n"

    # Collect all .md and .txt files
    files = sorted(
        [f for f in research_dir.iterdir()
         if f.is_file() and f.suffix in (".md", ".txt")],
        key=lambda f: f.name,
    )

    if not files:
        return "# Research Index Snapshot\n\nNo .md or .txt files in research/output/.\n"

    # Group by common prefix (split on first underscore or use full name)
    groups: dict[str, list[Path]] = {}
    for f in files:
        stem = f.stem
        # Group by prefix: everything before the first underscore, or full name if no underscore
        if "_" in stem:
            prefix = stem.split("_")[0]
        else:
            prefix = stem
        groups.setdefault(prefix, []).append(f)

    lines = [
        "# Research Index Snapshot",
        "",
        f"Generated: {_utc_now_iso()}",
        "",
        f"**Total files:** {len(files)} ({sum(1 for f in files if f.suffix == '.md')} .md, "
        f"{sum(1 for f in files if f.suffix == '.txt')} .txt)",
        f"**Groups:** {len(groups)}",
        "",
    ]

    for prefix in sorted(groups):
        group_files = groups[prefix]
        lines.append(f"## {prefix} ({len(group_files)} files)")
        lines.append("")
        lines.append("| File | Size | Summary |")
        lines.append("|------|------|---------|")

        for f in group_files:
            size_kb = f.stat().st_size / 1024
            # Read first 3 non-empty lines as summary
            try:
                with open(f, "r", encoding="utf-8", errors="replace") as fh:
                    first_lines = []
                    for raw_line in fh:
                        stripped = raw_line.strip()
                        if stripped:
                            # Remove markdown headers and truncate
                            clean = stripped.lstrip("#").strip()
                            if clean:
                                first_lines.append(clean[:80])
                        if len(first_lines) >= 3:
                            break
                summary = " / ".join(first_lines) if first_lines else "(empty)"
            except Exception:
                summary = "(unreadable)"

            # Escape pipe chars in summary for markdown table
            summary = summary.replace("|", "\\|")
            lines.append(f"| {f.name} | {size_kb:.1f} KB | {summary} |")

        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

SNAPSHOT_REGISTRY = {
    "portfolio_state": (generate_portfolio_state_snapshot, "_snapshot_portfolio_state.md"),
    "fitness_report": (generate_fitness_report_snapshot, "_snapshot_fitness_report.md"),
    "live_config": (generate_live_config_snapshot, "_snapshot_live_config.md"),
    "research_index": (generate_research_index_snapshot, "_snapshot_research_index.md"),
}


def generate_all_snapshots() -> dict[str, Path]:
    """Generate all snapshots and save to disk. Returns {name: path}."""
    results = {}
    for name, (gen_func, filename) in SNAPSHOT_REGISTRY.items():
        content = gen_func()
        path = save_snapshot(content, filename)
        results[name] = path
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate markdown snapshots of trading system state"
    )
    parser.add_argument("--snapshot", choices=list(SNAPSHOT_REGISTRY.keys()),
                        help="Generate a single snapshot (default: all)")
    parser.add_argument("--list", action="store_true",
                        help="List available snapshot names")
    args = parser.parse_args()

    if args.list:
        print("Available snapshots:")
        for name, (_, filename) in SNAPSHOT_REGISTRY.items():
            print(f"  {name:20s} -> {filename}")
        return

    if args.snapshot:
        gen_func, filename = SNAPSHOT_REGISTRY[args.snapshot]
        content = gen_func()
        path = save_snapshot(content, filename)
        print(f"Generated {args.snapshot} -> {path}")
        print(f"--- first 30 lines ---")
        for line in content.splitlines()[:30]:
            print(line)
    else:
        print("Generating all snapshots...")
        results = generate_all_snapshots()
        for name, path in results.items():
            print(f"  {name:20s} -> {path}")
        print(f"\nDone. {len(results)} snapshots generated.")

        # Print first 30 lines of each
        for name, path in results.items():
            content = path.read_text(encoding="utf-8")
            print(f"\n{'='*70}")
            print(f"  {name}")
            print(f"{'='*70}")
            for line in content.splitlines()[:30]:
                print(line)


if __name__ == "__main__":
    main()
