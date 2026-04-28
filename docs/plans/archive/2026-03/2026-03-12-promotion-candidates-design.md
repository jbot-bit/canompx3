---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Strategy Promotion Candidates Report — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Auto-surface validated strategies that aren't in the live portfolio as a scorecard HTML report, opened automatically after every rebuild, so the PM can review and decide.

**Architecture:** A single new script `scripts/tools/generate_promotion_candidates.py` that queries gold.db for FDR+WF strategies not covered by `LIVE_PORTFOLIO`, scores them with year-by-year, PBO, WFE, dollar gate, family robustness, and decay trend, then generates a self-contained HTML report (same dark-mode style as `generate_trade_sheet.py`). Integrated into the rebuild chain as step 8.5 and the `/post-rebuild` skill.

**Tech Stack:** Python, DuckDB, HTML (f-string template), existing pipeline imports.

**Institutional Grounding:**
- Lopez de Prado AFML Ch.18: Embargo → Paper → Graduation lifecycle. This fills the Embargo→Graduation gap.
- Scorecard pattern (Point72/AQR style): auto-generated metrics → PM binary go/no-go.
- No auto-promotion. Final decision always manual.

---

### Task 1: Core Query — Find Uncovered Candidates

**Files:**
- Create: `scripts/tools/generate_promotion_candidates.py`
- Reference: `trading_app/live_config.py:113-210` (LIVE_PORTFOLIO), `pipeline/check_drift.py:1886-1948` (check_uncovered pattern)

**Step 1: Write the test scaffold**

Create `tests/test_scripts/test_generate_promotion_candidates.py`:

```python
"""Tests for promotion candidate report generation."""
import pytest

def test_find_uncovered_candidates_excludes_live_portfolio():
    """Candidates list must NOT include any (orb_label, entry_model, filter_type) already in LIVE_PORTFOLIO."""
    from scripts.tools.generate_promotion_candidates import find_uncovered_candidates
    from trading_app.live_config import LIVE_PORTFOLIO
    from pipeline.paths import GOLD_DB_PATH

    candidates = find_uncovered_candidates(GOLD_DB_PATH)
    covered = {(s.orb_label, s.entry_model, s.filter_type) for s in LIVE_PORTFOLIO}
    for c in candidates:
        key = (c["orb_label"], c["entry_model"], c["filter_type"])
        assert key not in covered, f"Candidate {c['strategy_id']} is already in LIVE_PORTFOLIO"

def test_candidates_are_fdr_wf_robust():
    """Every candidate must be FDR-significant, WF-passed, and in a ROBUST family."""
    from scripts.tools.generate_promotion_candidates import find_uncovered_candidates
    from pipeline.paths import GOLD_DB_PATH

    candidates = find_uncovered_candidates(GOLD_DB_PATH)
    for c in candidates:
        assert c["fdr_significant"] is True, f"{c['strategy_id']} not FDR-sig"
        assert c["wf_passed"] is True, f"{c['strategy_id']} not WF-passed"
        assert c["robustness_status"] == "ROBUST", f"{c['strategy_id']} not ROBUST"

def test_candidates_sorted_by_expr_desc():
    """Candidates must be sorted by ExpR descending."""
    from scripts.tools.generate_promotion_candidates import find_uncovered_candidates
    from pipeline.paths import GOLD_DB_PATH

    candidates = find_uncovered_candidates(GOLD_DB_PATH)
    if len(candidates) > 1:
        exprs = [c["expectancy_r"] for c in candidates]
        assert exprs == sorted(exprs, reverse=True)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scripts/test_generate_promotion_candidates.py -v`
Expected: FAIL with ImportError (module doesn't exist yet)

**Step 3: Write `find_uncovered_candidates()` function**

Create `scripts/tools/generate_promotion_candidates.py` with:

```python
#!/usr/bin/env python3
"""
Strategy Promotion Candidate Report.

Surfaces validated strategies that passed full institutional gates
(FDR, walk-forward, ROBUST edge family) but are NOT yet in the live portfolio.
Generates a scorecard HTML for PM review.

Institutional grounding: Lopez de Prado AFML Ch.18 strategy lifecycle.
Scorecard pattern: auto-generated metrics, manual go/no-go.

Usage:
    python scripts/tools/generate_promotion_candidates.py
    python scripts/tools/generate_promotion_candidates.py --format terminal
    python scripts/tools/generate_promotion_candidates.py --no-open
"""

import argparse
import json
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb

from pipeline.asset_configs import get_active_instruments
from pipeline.cost_model import get_cost_spec
from pipeline.paths import GOLD_DB_PATH
from trading_app.live_config import (
    LIVE_MIN_EXPECTANCY_DOLLARS_MULT,
    LIVE_MIN_EXPECTANCY_R,
    LIVE_PORTFOLIO,
)


def find_uncovered_candidates(db_path: Path) -> list[dict]:
    """Find FDR+WF+ROBUST strategies not covered by LIVE_PORTFOLIO.

    Returns list of candidate dicts sorted by expectancy_r DESC.
    Each candidate is the best-ExpR strategy per uncovered
    (instrument, orb_label, entry_model, filter_type) combo.
    """
    covered = {(s.orb_label, s.entry_model, s.filter_type) for s in LIVE_PORTFOLIO}

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(
            """
            SELECT vs.strategy_id, vs.instrument, vs.orb_label, vs.entry_model,
                   vs.filter_type, vs.orb_minutes, vs.rr_target, vs.confirm_bars,
                   vs.sample_size, vs.win_rate, vs.expectancy_r, vs.sharpe_ann,
                   vs.max_drawdown_r, vs.years_tested, vs.all_years_positive,
                   vs.yearly_results, vs.fdr_significant, vs.fdr_adjusted_p,
                   vs.wf_passed, vs.wf_windows, vs.wfe, vs.skewness,
                   vs.kurtosis_excess, vs.stop_multiplier,
                   ef.robustness_status, ef.member_count, ef.pbo,
                   ef.cv_expectancy, ef.trade_tier
            FROM validated_setups vs
            INNER JOIN edge_families ef
              ON vs.strategy_id = ef.head_strategy_id
            WHERE LOWER(vs.status) = 'active'
              AND vs.fdr_significant = TRUE
              AND vs.wf_passed = TRUE
              AND ef.robustness_status = 'ROBUST'
              AND vs.expectancy_r >= ?
            ORDER BY vs.expectancy_r DESC
            """,
            [LIVE_MIN_EXPECTANCY_R],
        ).fetchall()

        cols = [desc[0] for desc in con.description]
    finally:
        con.close()

    # Filter out covered combos, keep best per (instrument, orb_label, entry_model, filter_type)
    seen = set()
    candidates = []
    for row in rows:
        d = dict(zip(cols, row, strict=False))
        combo_key = (d["orb_label"], d["entry_model"], d["filter_type"])
        if combo_key in covered:
            continue
        inst_key = (d["instrument"], d["orb_label"], d["entry_model"], d["filter_type"])
        if inst_key in seen:
            continue
        seen.add(inst_key)
        candidates.append(d)

    return candidates
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_scripts/test_generate_promotion_candidates.py -v`
Expected: All 3 PASS

**Step 5: Commit**

```bash
git add scripts/tools/generate_promotion_candidates.py tests/test_scripts/test_generate_promotion_candidates.py
git commit -m "feat: add promotion candidate query (find_uncovered_candidates)"
```

---

### Task 2: Scorecard Data Enrichment — Year-by-Year, Decay Trend, Dollar Gate

**Files:**
- Modify: `scripts/tools/generate_promotion_candidates.py`
- Test: `tests/test_scripts/test_generate_promotion_candidates.py`

**Step 1: Write the test**

Add to test file:

```python
def test_enrich_candidate_has_required_fields():
    """Enriched candidate must have year_by_year, decay_trend, dollar_gate fields."""
    from scripts.tools.generate_promotion_candidates import (
        find_uncovered_candidates,
        enrich_candidate,
    )
    from pipeline.paths import GOLD_DB_PATH

    candidates = find_uncovered_candidates(GOLD_DB_PATH)
    if not candidates:
        pytest.skip("No uncovered candidates in current DB")
    enriched = enrich_candidate(candidates[0])
    assert "year_by_year" in enriched
    assert "decay_slope" in enriched
    assert "dollar_gate_results" in enriched
    assert isinstance(enriched["year_by_year"], list)
    assert isinstance(enriched["dollar_gate_results"], dict)
```

**Step 2: Run test, verify fail**

**Step 3: Implement `enrich_candidate()`**

Add to `generate_promotion_candidates.py`:

```python
def enrich_candidate(candidate: dict) -> dict:
    """Add year-by-year breakdown, decay trend, and dollar gate results.

    Modifies candidate dict in-place and returns it.
    """
    # Year-by-year from stored JSON
    yearly_raw = candidate.get("yearly_results")
    if yearly_raw:
        yearly = json.loads(yearly_raw) if isinstance(yearly_raw, str) else yearly_raw
        years = []
        avg_rs = []
        for yr in sorted(yearly.keys()):
            y = yearly[yr]
            n = y.get("n", y.get("trades", 0))
            wr = y.get("win_rate", 0)
            avg_r = y.get("avg_r", y.get("expectancy_r", 0))
            years.append({"year": yr, "n": n, "win_rate": wr, "avg_r": avg_r, "total_r": round(avg_r * n, 2)})
            avg_rs.append(avg_r)
        candidate["year_by_year"] = years
        # Decay slope: simple linear regression of avg_r over years
        if len(avg_rs) >= 3:
            x = list(range(len(avg_rs)))
            x_mean = sum(x) / len(x)
            y_mean = sum(avg_rs) / len(avg_rs)
            num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, avg_rs))
            den = sum((xi - x_mean) ** 2 for xi in x)
            candidate["decay_slope"] = round(num / den, 4) if den > 0 else 0.0
        else:
            candidate["decay_slope"] = 0.0
    else:
        candidate["year_by_year"] = []
        candidate["decay_slope"] = 0.0

    # Dollar gate per instrument
    instruments = get_active_instruments()
    dollar_results = {}
    for inst in instruments:
        try:
            spec = get_cost_spec(inst)
            # Approximate with median risk from the strategy
            exp_r = candidate["expectancy_r"]
            # Use sample_size as rough signal — precise median_risk_points
            # requires experimental_strategies join, so we approximate
            min_dollars = LIVE_MIN_EXPECTANCY_DOLLARS_MULT * spec.total_friction
            # We store pass/fail and threshold, not exact dollars (would need median_risk_pts)
            dollar_results[inst] = {
                "rt_cost": round(spec.total_friction, 2),
                "min_required": round(min_dollars, 2),
                "point_value": spec.point_value,
            }
        except Exception:
            dollar_results[inst] = {"rt_cost": None, "min_required": None, "point_value": None}
    candidate["dollar_gate_results"] = dollar_results

    return candidate
```

**Step 4: Run test, verify pass**

**Step 5: Commit**

```bash
git add scripts/tools/generate_promotion_candidates.py tests/test_scripts/test_generate_promotion_candidates.py
git commit -m "feat: add scorecard enrichment (year-by-year, decay, dollar gate)"
```

---

### Task 3: LiveStrategySpec Code Generator

**Files:**
- Modify: `scripts/tools/generate_promotion_candidates.py`
- Test: `tests/test_scripts/test_generate_promotion_candidates.py`

**Step 1: Write the test**

```python
def test_generate_spec_code_produces_valid_python():
    """Generated LiveStrategySpec code must be syntactically valid Python."""
    from scripts.tools.generate_promotion_candidates import generate_spec_code

    code = generate_spec_code(
        orb_label="CME_PRECLOSE",
        entry_model="E2",
        filter_type="ORB_G8",
    )
    assert "LiveStrategySpec(" in code
    assert "CME_PRECLOSE" in code
    assert "E2" in code
    assert "ORB_G8" in code
    # Must be valid Python
    compile(code, "<test>", "eval")
```

**Step 2: Run test, verify fail**

**Step 3: Implement `generate_spec_code()`**

```python
def generate_spec_code(orb_label: str, entry_model: str, filter_type: str) -> str:
    """Generate copy-paste LiveStrategySpec Python code."""
    family_id = f"{orb_label}_{entry_model}_{filter_type}"
    return (
        f'LiveStrategySpec("{family_id}", "core", '
        f'"{orb_label}", "{entry_model}", "{filter_type}", None)'
    )
```

**Step 4: Run test, verify pass**

**Step 5: Commit**

```bash
git add scripts/tools/generate_promotion_candidates.py tests/test_scripts/test_generate_promotion_candidates.py
git commit -m "feat: add LiveStrategySpec code generator for promotion candidates"
```

---

### Task 4: HTML Report Generation

**Files:**
- Modify: `scripts/tools/generate_promotion_candidates.py`

This is the largest task. Generates a self-contained HTML report matching `generate_trade_sheet.py` dark-mode style.

**Step 1: Write the test**

```python
def test_generate_html_report_contains_required_sections():
    """HTML report must contain header, candidate cards, and spec code blocks."""
    from scripts.tools.generate_promotion_candidates import (
        find_uncovered_candidates,
        enrich_candidate,
        generate_html,
    )
    from pipeline.paths import GOLD_DB_PATH

    candidates = find_uncovered_candidates(GOLD_DB_PATH)
    if not candidates:
        pytest.skip("No uncovered candidates in current DB")
    enriched = [enrich_candidate(c) for c in candidates[:3]]
    html = generate_html(enriched)
    assert "PROMOTION CANDIDATES" in html
    assert "LiveStrategySpec" in html
    assert "Year-by-Year" in html or "year-by-year" in html.lower()
    assert "</html>" in html
```

**Step 2: Run test, verify fail**

**Step 3: Implement `generate_html()`**

Build the HTML with these sections per candidate card:
1. **Header row**: strategy_id, instrument, session, ExpR, Sharpe, sample_size
2. **Badges**: ROBUST, FDR, WF, all_years_positive
3. **Year-by-year table**: year, N, WR%, AvgR, TotalR
4. **Metrics grid**: PBO, WFE, family_size, decay_slope, skewness, kurtosis
5. **Dollar gate table**: per-instrument RT cost and minimum required
6. **Spec code block**: copy-paste LiveStrategySpec

Use the same CSS variables and dark-mode palette as `generate_trade_sheet.py`:
- Background: `#0d1117`
- Card bg: `#161b22`
- Border: `#30363d`
- Blue accent: `#58a6ff`
- Green: `#3fb950`
- Red: `#f85149`
- Yellow: `#d29922`
- Gray text: `#8b949e`

Group candidates by session (orb_label), sorted by ExpR within each group.

Add a **summary banner** at the top: "N new candidates found across M sessions. Review and add to live_config.py."

Add **decay warning** badge (red) if decay_slope < -0.05 (declining 0.05R/year).

The function signature: `def generate_html(candidates: list[dict]) -> str`

**Step 4: Run test, verify pass**

**Step 5: Commit**

```bash
git add scripts/tools/generate_promotion_candidates.py tests/test_scripts/test_generate_promotion_candidates.py
git commit -m "feat: add HTML scorecard report for promotion candidates"
```

---

### Task 5: Terminal Format Output

**Files:**
- Modify: `scripts/tools/generate_promotion_candidates.py`

**Step 1: Write the test**

```python
def test_terminal_output_includes_summary_line():
    """Terminal format must print summary count."""
    from scripts.tools.generate_promotion_candidates import (
        find_uncovered_candidates,
        enrich_candidate,
        format_terminal,
    )
    from pipeline.paths import GOLD_DB_PATH

    candidates = find_uncovered_candidates(GOLD_DB_PATH)
    if not candidates:
        pytest.skip("No uncovered candidates")
    enriched = [enrich_candidate(c) for c in candidates[:3]]
    output = format_terminal(enriched)
    assert "candidate" in output.lower()
    assert any(c["strategy_id"] in output for c in enriched)
```

**Step 2: Run test, verify fail**

**Step 3: Implement `format_terminal()`**

Compact table output for CLI use. Show: strategy_id, instrument, session, ExpR, N, WR%, PBO, WFE, family_size, robustness, decay_slope. One line per candidate. Summary line at top and bottom.

**Step 4: Run test, verify pass**

**Step 5: Commit**

```bash
git add scripts/tools/generate_promotion_candidates.py tests/test_scripts/test_generate_promotion_candidates.py
git commit -m "feat: add terminal format output for promotion candidates"
```

---

### Task 6: CLI main() and Browser Open

**Files:**
- Modify: `scripts/tools/generate_promotion_candidates.py`

**Step 1: Implement `main()`**

Follow the exact pattern from `generate_trade_sheet.py`:

```python
def main():
    parser = argparse.ArgumentParser(description="Generate promotion candidate scorecard")
    parser.add_argument("--format", choices=["html", "terminal"], default="html",
                        help="Output format (default: html)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output HTML path. Default: promotion_candidates.html")
    parser.add_argument("--db-path", type=Path, default=None, help="Path to gold.db")
    parser.add_argument("--no-open", action="store_true", help="Don't open in browser")
    args = parser.parse_args()

    db_path = args.db_path or GOLD_DB_PATH
    output_path = Path(args.output) if args.output else PROJECT_ROOT / "promotion_candidates.html"

    print("Promotion Candidate Report")
    print(f"  DB: {db_path}")

    candidates = find_uncovered_candidates(db_path)
    enriched = [enrich_candidate(c) for c in candidates]

    if not enriched:
        print("\n  No uncovered ROBUST candidates found. Portfolio is fully covered.")
        return

    print(f"  Found {len(enriched)} promotion candidates\n")

    if args.format == "terminal":
        print(format_terminal(enriched))
        return

    html = generate_html(enriched)
    output_path.write_text(html, encoding="utf-8")
    print(f"  Written to {output_path}")

    if not args.no_open:
        webbrowser.open(str(output_path))
        print("  Opened in browser.")


if __name__ == "__main__":
    main()
```

**Step 2: Manual test**

Run: `python scripts/tools/generate_promotion_candidates.py --format terminal`
Verify: Shows candidate table with data from gold.db

Run: `python scripts/tools/generate_promotion_candidates.py`
Verify: Opens HTML report in browser

**Step 3: Commit**

```bash
git add scripts/tools/generate_promotion_candidates.py
git commit -m "feat: add CLI main() with html/terminal format and browser open"
```

---

### Task 7: Wire Into Rebuild Chain

**Files:**
- Modify: `scripts/tools/run_rebuild_with_sync.sh` (add step 8.5)
- Modify: `.claude/commands/post-rebuild.md` (add step 4.5)

**Step 1: Add to rebuild shell script**

After step 8 (health check), before step 9 (Pinecone sync), insert:

```bash
# Step 8.5: Surface promotion candidates (auto-opens in browser for PM review)
echo ""
echo "Step 8.5: Surfacing promotion candidates..."
python scripts/tools/generate_promotion_candidates.py --no-open --format terminal
```

Note: `--no-open` in the shell script (non-interactive rebuild). The terminal output is sufficient for the build log. The PM runs the HTML version manually or it opens when running `/post-rebuild`.

**Step 2: Add to post-rebuild skill**

After Step 4 (audit gates), before Step 5 (Pinecone sync), add:

```markdown
### Step 4.5: Surface Promotion Candidates

```bash
python scripts/tools/generate_promotion_candidates.py
```

Opens scorecard HTML in browser. Review candidates and decide which to add to `live_config.py`.
If no candidates found, report "Portfolio fully covered" and continue.
```

**Step 3: Manual test**

Run: `python scripts/tools/generate_promotion_candidates.py --format terminal`
Verify: No errors, shows candidates or "no candidates" message

**Step 4: Commit**

```bash
git add scripts/tools/run_rebuild_with_sync.sh .claude/commands/post-rebuild.md
git commit -m "feat: wire promotion candidates into rebuild chain and post-rebuild skill"
```

---

### Task 8: Update REPO_MAP and Final Verification

**Files:**
- Run: `python scripts/tools/gen_repo_map.py`
- Run: Full test suite + drift checks

**Step 1: Regenerate REPO_MAP**

```bash
python scripts/tools/gen_repo_map.py
```

**Step 2: Run drift + tests**

```bash
python pipeline/check_drift.py
python -m pytest tests/ -x -q
```

**Step 3: Final commit**

```bash
git add REPO_MAP.md
git commit -m "docs: update REPO_MAP after promotion candidates feature"
```

---

## Integration Summary

After implementation, the promotion workflow is:

1. **Rebuild runs** → outcomes → discovery → validation → edge families → RR locks
2. **Step 8.5 fires automatically** → `generate_promotion_candidates.py` queries gold.db
3. **Terminal output in build log**: "Found N promotion candidates"
4. **`/post-rebuild` opens HTML scorecard** in browser → PM sees candidates with full scorecard
5. **PM decides** → copies `LiveStrategySpec` code from report → pastes into `live_config.py`
6. **Next drift check** → check 43 no longer warns for promoted strategies

No new tables. No new state. No auto-promotion. Just institutional-grade surfacing with manual final decision.
