#!/usr/bin/env python3
"""
Parameter stability heatmap generator for live portfolio strategies.

For each live spec, queries the experimental_strategies grid to build
RR × CB matrices (or RR × aperture for E2/E3). Flags isolated peaks
where the chosen parameter has no stable neighbors — a red flag for
data mining.

Output: HTML report with color-coded cells.
  - Green:  ≥80% of baseline ExpR (STABLE)
  - Yellow: 50-80% of baseline (OK)
  - Orange: 20-50% of baseline (WEAK)
  - Red:    ≤20% or sign-flipped (UNSTABLE)
  - Gray:   N < 20 (insufficient data)

Usage:
    python scripts/reports/parameter_stability_heatmap.py
    python scripts/reports/parameter_stability_heatmap.py --instrument MGC
    python scripts/reports/parameter_stability_heatmap.py --family TOKYO_OPEN_E2_ORB_G5
"""

import argparse
import html as html_mod
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.build_daily_features import VALID_ORB_MINUTES  # noqa: E402
from pipeline.db_config import configure_connection  # noqa: E402
from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from trading_app.config import TRADEABLE_INSTRUMENTS  # noqa: E402
from trading_app.live_config import LIVE_PORTFOLIO  # noqa: E402

# Grid axes
RR_STEPS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
CB_STEPS = [1, 2, 3, 4, 5]
APERTURES = VALID_ORB_MINUTES

# Stability thresholds (fraction of baseline ExpR retained)
STABLE_THRESHOLD = 0.80
OK_THRESHOLD = 0.50
WEAK_THRESHOLD = 0.20
MIN_N = 20


def _cell_color(baseline_expr: float, cell_expr: float | None, cell_n: int) -> str:
    """Return CSS color class for a cell based on stability vs baseline."""
    if cell_expr is None or cell_n < MIN_N:
        return "gray"
    if baseline_expr <= 0:
        # Baseline is negative — any positive neighbor is better
        if cell_expr > 0:
            return "green"
        return "red"
    ratio = cell_expr / baseline_expr
    if ratio >= STABLE_THRESHOLD:
        return "green"
    if ratio >= OK_THRESHOLD:
        return "yellow"
    if ratio >= WEAK_THRESHOLD:
        return "orange"
    return "red"


def _stability_score(baseline_expr: float, neighbors: list[dict]) -> str:
    """Compute overall stability verdict from neighbor cells."""
    if not neighbors:
        return "NO_NEIGHBORS"
    valid = [n for n in neighbors if n["N"] >= MIN_N and n["ExpR"] is not None]
    if not valid:
        return "INSUFFICIENT_DATA"
    if baseline_expr <= 0:
        return "NEGATIVE_BASELINE"

    ratios = [n["ExpR"] / baseline_expr for n in valid]
    worst = min(ratios)
    avg = sum(ratios) / len(ratios)

    if worst >= OK_THRESHOLD and avg >= STABLE_THRESHOLD:
        return "STABLE"
    if worst >= WEAK_THRESHOLD and avg >= OK_THRESHOLD:
        return "OK"
    if worst < WEAK_THRESHOLD or avg < OK_THRESHOLD:
        return "ISOLATED_PEAK"
    return "WEAK"


def query_grid(
    con, instrument: str, orb_label: str, entry_model: str, filter_type: str,
) -> list[dict]:
    """Fetch all experimental_strategies rows matching the spec's grid slice."""
    rows = con.execute("""
        SELECT rr_target, confirm_bars, orb_minutes,
               expectancy_r, win_rate, sample_size, sharpe_ratio,
               sharpe_haircut, strategy_id
        FROM experimental_strategies
        WHERE instrument = ?
          AND orb_label = ?
          AND entry_model = ?
          AND filter_type = ?
        ORDER BY rr_target, confirm_bars, orb_minutes
    """, [instrument, orb_label, entry_model, filter_type]).fetchall()

    return [
        {
            "rr": r[0], "cb": r[1], "om": r[2],
            "ExpR": r[3], "WR": r[4], "N": r[5], "Sharpe": r[6],
            "SharpeH": r[7], "strategy_id": r[8],
        }
        for r in rows
    ]


def build_heatmap_data(
    grid: list[dict], baseline_rr: float, baseline_cb: int, baseline_om: int,
) -> dict:
    """Build heatmap matrices from grid data.

    Returns dict with:
      - rr_cb_matrix: {om: {(rr, cb): cell_dict}} for E1 (CB varies)
      - rr_om_matrix: {(rr, om): cell_dict} for E2/E3 (CB always 1)
      - baseline: the baseline cell
      - neighbors: list of adjacent cells for stability scoring
    """
    # Index grid by (rr, cb, om)
    indexed = {}
    for g in grid:
        key = (g["rr"], g["cb"], g["om"])
        indexed[key] = g

    baseline = indexed.get((baseline_rr, baseline_cb, baseline_om))
    if baseline is None:
        return {"baseline": None, "matrices": {}, "neighbors": [], "verdict": "NO_BASELINE"}

    # Determine if CB varies (E1) or not (E2/E3)
    unique_cb = sorted({g["cb"] for g in grid})
    cb_varies = len(unique_cb) > 1

    if cb_varies:
        # Build per-aperture RR×CB matrices
        matrices = {}
        for om in APERTURES:
            matrix = {}
            for rr in RR_STEPS:
                for cb in unique_cb:
                    cell = indexed.get((rr, cb, om))
                    if cell:
                        matrix[(rr, cb)] = cell
            if matrix:
                matrices[om] = matrix
    else:
        # Build single RR×aperture matrix
        matrices = {}
        matrix = {}
        for rr in RR_STEPS:
            for om in APERTURES:
                cell = indexed.get((rr, baseline_cb, om))
                if cell:
                    matrix[(rr, om)] = cell
        if matrix:
            matrices["rr_x_om"] = matrix

    # Find neighbors (±1 step on each axis)
    neighbors = []
    rr_idx = RR_STEPS.index(baseline_rr) if baseline_rr in RR_STEPS else -1
    for delta in [-1, 1]:
        # RR neighbors
        adj_rr_idx = rr_idx + delta
        if 0 <= adj_rr_idx < len(RR_STEPS):
            adj = indexed.get((RR_STEPS[adj_rr_idx], baseline_cb, baseline_om))
            if adj:
                neighbors.append(adj)
        # CB neighbors (E1 only)
        if cb_varies:
            cb_idx = CB_STEPS.index(baseline_cb) if baseline_cb in CB_STEPS else -1
            adj_cb_idx = cb_idx + delta
            if 0 <= adj_cb_idx < len(CB_STEPS):
                adj = indexed.get((baseline_rr, CB_STEPS[adj_cb_idx], baseline_om))
                if adj:
                    neighbors.append(adj)
        # Aperture neighbors
        om_idx = APERTURES.index(baseline_om) if baseline_om in APERTURES else -1
        adj_om_idx = om_idx + delta
        if 0 <= adj_om_idx < len(APERTURES):
            adj = indexed.get((baseline_rr, baseline_cb, APERTURES[adj_om_idx]))
            if adj:
                neighbors.append(adj)

    baseline_expr = baseline.get("ExpR") or 0.0
    verdict = _stability_score(baseline_expr, neighbors)

    return {
        "baseline": baseline,
        "matrices": matrices,
        "neighbors": neighbors,
        "verdict": verdict,
        "cb_varies": cb_varies,
    }


def _esc(text: str) -> str:
    """HTML-escape text."""
    return html_mod.escape(str(text))


def render_html_report(
    results: list[dict], output_path: Path,
) -> None:
    """Render all heatmap results as a single HTML report."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Count verdicts
    verdicts = [r["verdict"] for r in results]
    n_stable = verdicts.count("STABLE") + verdicts.count("OK")
    n_isolated = verdicts.count("ISOLATED_PEAK")
    n_weak = verdicts.count("WEAK")
    n_total = len(results)

    html_parts = [f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Parameter Stability Heatmap</title>
<style>
body {{ font-family: monospace; margin: 20px; background: #1a1a2e; color: #e0e0e0; }}
h1, h2, h3 {{ color: #f0f0f0; }}
.summary {{ background: #16213e; padding: 15px; border-radius: 8px; margin: 10px 0; }}
.spec-block {{ background: #0f3460; padding: 12px; border-radius: 8px; margin: 15px 0; }}
table {{ border-collapse: collapse; margin: 8px 0; }}
th, td {{ padding: 4px 8px; border: 1px solid #333; text-align: right; font-size: 12px; }}
th {{ background: #16213e; }}
td.green {{ background: #1b5e20; color: #a5d6a7; }}
td.yellow {{ background: #f57f17; color: #fff9c4; }}
td.orange {{ background: #e65100; color: #ffe0b2; }}
td.red {{ background: #b71c1c; color: #ef9a9a; }}
td.gray {{ background: #424242; color: #9e9e9e; }}
td.baseline {{ border: 3px solid #fff; font-weight: bold; }}
.verdict-STABLE {{ color: #4caf50; font-weight: bold; }}
.verdict-OK {{ color: #8bc34a; }}
.verdict-WEAK {{ color: #ff9800; }}
.verdict-ISOLATED_PEAK {{ color: #f44336; font-weight: bold; }}
.verdict-NO_BASELINE {{ color: #9e9e9e; }}
.verdict-INSUFFICIENT_DATA {{ color: #9e9e9e; }}
.verdict-NEGATIVE_BASELINE {{ color: #9e9e9e; }}
.verdict-NO_NEIGHBORS {{ color: #9e9e9e; }}
.legend {{ display: flex; gap: 15px; margin: 10px 0; }}
.legend span {{ padding: 2px 8px; border-radius: 3px; font-size: 12px; }}
</style></head><body>
<h1>Parameter Stability Heatmap</h1>
<p>Generated: {ts}</p>
<div class="summary">
<b>Summary:</b> {n_total} spec/instrument combos analyzed |
{n_stable} stable | {n_weak} weak | {n_isolated} isolated peaks
</div>
<div class="legend">
<span style="background:#1b5e20;color:#a5d6a7">STABLE (>=80%)</span>
<span style="background:#f57f17;color:#fff9c4">OK (50-80%)</span>
<span style="background:#e65100;color:#ffe0b2">WEAK (20-50%)</span>
<span style="background:#b71c1c;color:#ef9a9a">UNSTABLE (<20%)</span>
<span style="background:#424242;color:#9e9e9e">N<20</span>
</div>
"""]

    for r in results:
        spec_id = _esc(r["spec_id"])
        inst = _esc(r["instrument"])
        verdict = r["verdict"]
        baseline = r.get("baseline")

        html_parts.append(f'<div class="spec-block">')
        html_parts.append(f'<h3>{inst} / {spec_id} — '
                          f'<span class="verdict-{verdict}">{verdict}</span></h3>')

        if baseline is None:
            html_parts.append("<p>No experimental data found for this spec.</p></div>")
            continue

        bl_expr = baseline.get("ExpR") or 0.0
        bl_n = baseline.get("N", 0)
        bl_sharpe = baseline.get("Sharpe") or 0.0
        bl_sh = baseline.get("SharpeH")
        sh_str = f", SharpeH={bl_sh:.3f}" if bl_sh is not None else ""
        html_parts.append(
            f'<p>Baseline: RR{baseline["rr"]} CB{baseline["cb"]} O{baseline["om"]} '
            f'| ExpR={bl_expr:+.4f}, Sharpe={bl_sharpe:.3f}{sh_str}, '
            f'N={bl_n}</p>'
        )

        # Render matrices
        for matrix_key, matrix in r.get("matrices", {}).items():
            if r.get("cb_varies"):
                # RR × CB matrix for specific aperture
                om = matrix_key
                html_parts.append(f"<p><b>Aperture: {om}m</b></p>")
                cb_vals = sorted({k[1] for k in matrix.keys()})
                html_parts.append("<table><tr><th>RR \\ CB</th>")
                for cb in cb_vals:
                    html_parts.append(f"<th>CB{cb}</th>")
                html_parts.append("</tr>")
                for rr in RR_STEPS:
                    html_parts.append(f"<tr><th>RR{rr}</th>")
                    for cb in cb_vals:
                        cell = matrix.get((rr, cb))
                        if cell is None:
                            html_parts.append('<td class="gray">-</td>')
                        else:
                            color = _cell_color(bl_expr, cell.get("ExpR"), cell.get("N", 0))
                            is_bl = (rr == baseline["rr"] and cb == baseline["cb"]
                                     and om == baseline["om"])
                            bl_class = " baseline" if is_bl else ""
                            expr = cell.get("ExpR")
                            n = cell.get("N", 0)
                            val = f"{expr:+.3f}" if expr is not None else "-"
                            title = f"N={n}, WR={cell.get('WR', 0):.0%}"
                            html_parts.append(
                                f'<td class="{color}{bl_class}" title="{title}">'
                                f'{val}<br><small>N={n}</small></td>'
                            )
                    html_parts.append("</tr>")
                html_parts.append("</table>")
            else:
                # RR × aperture matrix
                html_parts.append("<p><b>RR x Aperture</b></p>")
                om_vals = sorted({k[1] for k in matrix.keys()})
                html_parts.append("<table><tr><th>RR \\ OM</th>")
                for om in om_vals:
                    html_parts.append(f"<th>{om}m</th>")
                html_parts.append("</tr>")
                for rr in RR_STEPS:
                    html_parts.append(f"<tr><th>RR{rr}</th>")
                    for om in om_vals:
                        cell = matrix.get((rr, om))
                        if cell is None:
                            html_parts.append('<td class="gray">-</td>')
                        else:
                            color = _cell_color(bl_expr, cell.get("ExpR"), cell.get("N", 0))
                            is_bl = (rr == baseline["rr"] and om == baseline["om"])
                            bl_class = " baseline" if is_bl else ""
                            expr = cell.get("ExpR")
                            n = cell.get("N", 0)
                            val = f"{expr:+.3f}" if expr is not None else "-"
                            title = f"N={n}, WR={cell.get('WR', 0):.0%}"
                            html_parts.append(
                                f'<td class="{color}{bl_class}" title="{title}">'
                                f'{val}<br><small>N={n}</small></td>'
                            )
                    html_parts.append("</tr>")
                html_parts.append("</table>")

        html_parts.append("</div>")

    html_parts.append("</body></html>")
    output_path.write_text("\n".join(html_parts), encoding="utf-8")


def find_best_variant(con, instrument: str, spec) -> dict | None:
    """Find the best validated variant for a spec/instrument."""
    row = con.execute("""
        SELECT rr_target, confirm_bars, orb_minutes, expectancy_r,
               win_rate, sample_size, sharpe_ratio, strategy_id
        FROM validated_setups
        WHERE instrument = ?
          AND orb_label = ?
          AND entry_model = ?
          AND filter_type = ?
          AND status = 'active'
        ORDER BY expectancy_r DESC
        LIMIT 1
    """, [instrument, spec.orb_label, spec.entry_model,
          spec.filter_type]).fetchone()

    if row is None:
        return None
    return {
        "rr": row[0], "cb": row[1], "om": row[2], "ExpR": row[3],
        "WR": row[4], "N": row[5], "Sharpe": row[6], "strategy_id": row[7],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Parameter stability heatmap generator")
    parser.add_argument("--instrument", default=None,
                        help="Instrument to analyze (default: all tradeable)")
    parser.add_argument("--family", default=None,
                        help="Specific family_id (default: all live specs)")
    parser.add_argument("--db", default=None, help="Database path override")
    parser.add_argument("--output", default=None,
                        help="Output HTML path (default: parameter_stability.html)")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else GOLD_DB_PATH
    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        sys.exit(1)

    instruments = [args.instrument] if args.instrument else TRADEABLE_INSTRUMENTS
    specs = LIVE_PORTFOLIO
    if args.family:
        specs = [s for s in specs if s.family_id == args.family]
        if not specs:
            print(f"ERROR: No spec found with family_id={args.family}")
            sys.exit(1)

    output_path = Path(args.output) if args.output else Path("parameter_stability.html")

    print("=" * 70)
    print("PARAMETER STABILITY HEATMAP GENERATOR")
    print(f"Database: {db_path}")
    print(f"Instruments: {', '.join(instruments)}")
    print(f"Specs: {len(specs)}")
    print("=" * 70)

    con = duckdb.connect(str(db_path), read_only=True)
    configure_connection(con)

    results = []
    n_isolated = 0

    try:
        for inst in instruments:
            for spec in specs:
                # Find the best validated variant as baseline
                best = find_best_variant(con, inst, spec)
                if best is None:
                    continue

                # Query the full experimental grid for this spec
                grid = query_grid(
                    con, inst, spec.orb_label, spec.entry_model, spec.filter_type)
                if not grid:
                    continue

                heatmap = build_heatmap_data(
                    grid, best["rr"], best["cb"], best["om"])

                result = {
                    "spec_id": spec.family_id,
                    "instrument": inst,
                    "baseline": heatmap["baseline"],
                    "matrices": heatmap.get("matrices", {}),
                    "cb_varies": heatmap.get("cb_varies", False),
                    "verdict": heatmap["verdict"],
                    "n_neighbors": len(heatmap["neighbors"]),
                }
                results.append(result)

                marker = "!!" if heatmap["verdict"] == "ISOLATED_PEAK" else "  "
                print(f"  {marker} {inst} {spec.family_id}: {heatmap['verdict']} "
                      f"({len(heatmap['neighbors'])} neighbors)")
                if heatmap["verdict"] == "ISOLATED_PEAK":
                    n_isolated += 1
    finally:
        con.close()

    render_html_report(results, output_path)
    print()
    print(f"Report written to: {output_path}")
    print(f"Total: {len(results)} combos analyzed, {n_isolated} isolated peaks")

    sys.exit(1 if n_isolated > 0 else 0)


if __name__ == "__main__":
    main()
