# /discover Skill — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a `/discover` slash command that runs AI-assisted strategy research on any (instrument, session) combo, with memory of past findings, statistical rigor, and structured reporting.

**Architecture:** A Claude Code slash command (`.claude/commands/discover.md`) that orchestrates a Python research module (`research/discover.py`) + Pinecone vector memory for past findings + claude-mem for session state. The Python module generalizes `edge_hunter.py` to accept any instrument/session and outputs structured JSON. The slash command interprets results, checks memory, and reports in RESEARCH_RULES.md format.

**Tech Stack:** Python (DuckDB, scipy, numpy), Claude Code slash commands, Pinecone MCP (vector search), claude-mem MCP (session state)

---

### Task 1: Create the generalized edge scanner module

**Files:**
- Create: `research/discover.py`

**Step 1: Write the module**

This generalizes `edge_hunter.py` into a callable module with CLI interface. Key changes from `edge_hunter.py`:
- Accepts any instrument + session via CLI args
- Returns structured JSON (not print output)
- Adds ATR regime scan, prior-day signal, and seasonality tests
- Auto-enforces DST splits for affected sessions
- BH FDR across all tests in a single run

```python
"""
Generalized edge discovery scanner.

Runs a structured battery of pre-entry predictor tests on any (instrument, session)
combination. All tests use correct joins (orb_minutes match), zero look-ahead,
and Benjamini-Hochberg FDR correction.

Usage:
    python research/discover.py --instrument MGC --session 1000
    python research/discover.py --instrument MES --session 0030 --entry-model E0
    python research/discover.py --instrument MGC --all-sessions
    python research/discover.py --instrument MGC --session 1000 --json
"""
import argparse
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.paths import GOLD_DB_PATH
from pipeline.dst import DST_AFFECTED_SESSIONS


# === CORRECT JOIN TEMPLATE ===
SAFE_JOIN = """
    FROM orb_outcomes o
    JOIN daily_features d
        ON o.trading_day = d.trading_day
        AND o.symbol = d.symbol
        AND o.orb_minutes = d.orb_minutes
"""

# Sessions ordered by time-of-day (Brisbane time)
ALL_SESSIONS = ['0900', '1000', '1100', '1130', '1800', '2300', '0030']

# Which sessions happen before each session (for cross-session predictors)
EARLIER_SESSIONS = {
    '0900': [],
    '1000': ['0900'],
    '1100': ['0900', '1000'],
    '1130': ['0900', '1000', '1100'],
    '1800': ['0900', '1000', '1100', '1130'],
    '2300': ['0900', '1000', '1100', '1130', '1800'],
    '0030': ['0900', '1000', '1100', '1130', '1800', '2300'],
}


def get_connection():
    return duckdb.connect(str(GOLD_DB_PATH), read_only=True)


def get_pnl_array(con, instrument, session, em, rr, cb, size_min, extra_where=""):
    """Get pnl_r array with safe join."""
    size_col = f"orb_{session}_size"
    result = con.execute(f"""
        SELECT o.pnl_r {SAFE_JOIN}
        WHERE o.symbol = ? AND o.orb_label = ?
          AND o.entry_model = ? AND o.rr_target = ? AND o.confirm_bars = ?
          AND d.{size_col} >= ?
          AND o.outcome IN ('win','loss','early_exit') AND o.pnl_r IS NOT NULL
          {extra_where}
    """, [instrument, session, em, rr, cb, size_min]).fetchnumpy()
    return result.get('pnl_r', np.array([]))


def audit_join(con, instrument, session, em, rr, cb, size_min):
    """Verify join doesn't inflate rows. Returns (n_raw, n_joined, n_filtered)."""
    size_col = f"orb_{session}_size"

    n_raw = con.execute("""
        SELECT COUNT(*) FROM orb_outcomes
        WHERE symbol = ? AND orb_label = ?
          AND entry_model = ? AND rr_target = ? AND confirm_bars = ?
          AND outcome IN ('win','loss','early_exit') AND pnl_r IS NOT NULL
    """, [instrument, session, em, rr, cb]).fetchone()[0]

    n_joined = con.execute(f"""
        SELECT COUNT(*) {SAFE_JOIN}
        WHERE o.symbol = ? AND o.orb_label = ?
          AND o.entry_model = ? AND o.rr_target = ? AND o.confirm_bars = ?
          AND o.outcome IN ('win','loss','early_exit') AND o.pnl_r IS NOT NULL
    """, [instrument, session, em, rr, cb]).fetchone()[0]

    n_filtered = con.execute(f"""
        SELECT COUNT(*) {SAFE_JOIN}
        WHERE o.symbol = ? AND o.orb_label = ?
          AND o.entry_model = ? AND o.rr_target = ? AND o.confirm_bars = ?
          AND d.{size_col} >= ?
          AND o.outcome IN ('win','loss','early_exit') AND o.pnl_r IS NOT NULL
    """, [instrument, session, em, rr, cb, size_min]).fetchone()[0]

    if n_joined > n_raw:
        raise ValueError(f"JOIN INFLATION: raw={n_raw} joined={n_joined}")
    if n_filtered > n_joined:
        raise ValueError(f"FILTER INFLATION: joined={n_joined} filtered={n_filtered}")

    return n_raw, n_joined, n_filtered


def test_binary_split(con, name, instrument, session, em, rr, cb, size_min,
                      col_expr, true_label="true", false_label="false"):
    """Test a binary predictor. Returns result dict or None."""
    true_arr = get_pnl_array(con, instrument, session, em, rr, cb, size_min,
                             f"AND ({col_expr}) = true")
    false_arr = get_pnl_array(con, instrument, session, em, rr, cb, size_min,
                              f"AND ({col_expr}) = false")

    n_t, n_f = len(true_arr), len(false_arr)
    if n_t < 20 or n_f < 20:
        return None

    m_t, m_f = float(np.mean(true_arr)), float(np.mean(false_arr))
    t_stat, p = stats.ttest_ind(true_arr, false_arr, equal_var=False)

    return {
        'name': name, 'type': 'binary',
        'n_true': int(n_t), 'mean_true': round(m_t, 4),
        'n_false': int(n_f), 'mean_false': round(m_f, 4),
        'delta': round(m_f - m_t, 4),
        't': round(float(t_stat), 4), 'p': round(float(p), 6),
        'true_label': true_label, 'false_label': false_label,
    }


def apply_bh_fdr(results, alpha=0.05):
    """Apply Benjamini-Hochberg FDR correction to results list (in-place)."""
    results.sort(key=lambda x: x['p'])
    n_tests = len(results)
    for i, r in enumerate(results):
        rank = i + 1
        r['bh_rank'] = rank
        r['bh_threshold'] = round(alpha * rank / n_tests, 6)
        r['bh_significant'] = r['p'] <= r['bh_threshold']
    return results


def scan_session(con, instrument, session, entry_model='E1', rr=2.0, cb=2,
                 size_min=4.0):
    """Run full predictor battery on a single (instrument, session) combo.

    Returns dict with metadata and results list.
    """
    # Audit
    try:
        n_raw, n_joined, n_filtered = audit_join(
            con, instrument, session, entry_model, rr, cb, size_min)
    except ValueError as e:
        return {'error': str(e), 'instrument': instrument, 'session': session}

    if n_filtered < 30:
        return {
            'instrument': instrument, 'session': session,
            'entry_model': entry_model, 'rr': rr, 'cb': cb,
            'n_filtered': n_filtered, 'skipped': True,
            'reason': f'N={n_filtered} < 30 minimum',
            'results': [],
        }

    baseline = get_pnl_array(con, instrument, session, entry_model, rr, cb, size_min)
    base_expr = float(np.mean(baseline)) if len(baseline) > 0 else 0.0
    base_wr = float(np.mean(baseline > 0)) if len(baseline) > 0 else 0.0

    results = []

    # ── 1. Day of week ──
    for dow_name, dow_val in [('Mon', 1), ('Tue', 2), ('Wed', 3), ('Thu', 4), ('Fri', 5)]:
        on_day = get_pnl_array(con, instrument, session, entry_model, rr, cb, size_min,
                               f"AND EXTRACT(dow FROM o.trading_day) = {dow_val}")
        off_day = get_pnl_array(con, instrument, session, entry_model, rr, cb, size_min,
                                f"AND EXTRACT(dow FROM o.trading_day) != {dow_val}")
        if len(on_day) >= 15 and len(off_day) >= 30:
            t_stat, p = stats.ttest_ind(on_day, off_day, equal_var=False)
            results.append({
                'name': f'DOW_{dow_name}', 'type': 'dow',
                'n_true': int(len(on_day)), 'mean_true': round(float(np.mean(on_day)), 4),
                'n_false': int(len(off_day)), 'mean_false': round(float(np.mean(off_day)), 4),
                'delta': round(float(np.mean(on_day) - np.mean(off_day)), 4),
                't': round(float(t_stat), 4), 'p': round(float(p), 6),
            })

    # ── 2. Prior-day signal (prev trading day outcome for same session) ──
    for signal in ['win', 'loss']:
        arr = get_pnl_array(con, instrument, session, entry_model, rr, cb, size_min,
                            f"AND d.prev_day_{session}_outcome = '{signal}'")
        other = get_pnl_array(con, instrument, session, entry_model, rr, cb, size_min,
                              f"AND d.prev_day_{session}_outcome IS NOT NULL AND d.prev_day_{session}_outcome != '{signal}'")
        if len(arr) >= 20 and len(other) >= 20:
            t_stat, p = stats.ttest_ind(arr, other, equal_var=False)
            results.append({
                'name': f'prev_day_{signal}', 'type': 'prior_day',
                'n_true': int(len(arr)), 'mean_true': round(float(np.mean(arr)), 4),
                'n_false': int(len(other)), 'mean_false': round(float(np.mean(other)), 4),
                'delta': round(float(np.mean(arr) - np.mean(other)), 4),
                't': round(float(t_stat), 4), 'p': round(float(p), 6),
            })

    # ── 3. Earlier-session outcomes (cross-session context) ──
    for earlier in EARLIER_SESSIONS.get(session, [])[:3]:
        win_arr = get_pnl_array(con, instrument, session, entry_model, rr, cb, size_min,
                                f"AND d.orb_{earlier}_outcome = 'win'")
        loss_arr = get_pnl_array(con, instrument, session, entry_model, rr, cb, size_min,
                                 f"AND d.orb_{earlier}_outcome = 'loss'")
        if len(win_arr) >= 20 and len(loss_arr) >= 20:
            t_stat, p = stats.ttest_ind(win_arr, loss_arr, equal_var=False)
            results.append({
                'name': f'{earlier}_outcome_win_vs_loss', 'type': 'cross_session',
                'n_true': int(len(win_arr)), 'mean_true': round(float(np.mean(win_arr)), 4),
                'n_false': int(len(loss_arr)), 'mean_false': round(float(np.mean(loss_arr)), 4),
                'delta': round(float(np.mean(win_arr) - np.mean(loss_arr)), 4),
                't': round(float(t_stat), 4), 'p': round(float(p), 6),
            })

    # ── 4. ATR regime (compressed spring) ──
    low_atr = get_pnl_array(con, instrument, session, entry_model, rr, cb, size_min,
                            "AND d.atr_20 IS NOT NULL AND d.atr_20 < d.atr_20_sma50")
    high_atr = get_pnl_array(con, instrument, session, entry_model, rr, cb, size_min,
                             "AND d.atr_20 IS NOT NULL AND d.atr_20 >= d.atr_20_sma50")
    if len(low_atr) >= 20 and len(high_atr) >= 20:
        t_stat, p = stats.ttest_ind(low_atr, high_atr, equal_var=False)
        results.append({
            'name': 'ATR_below_sma50', 'type': 'regime',
            'n_true': int(len(low_atr)), 'mean_true': round(float(np.mean(low_atr)), 4),
            'n_false': int(len(high_atr)), 'mean_false': round(float(np.mean(high_atr)), 4),
            'delta': round(float(np.mean(low_atr) - np.mean(high_atr)), 4),
            't': round(float(t_stat), 4), 'p': round(float(p), 6),
        })

    # ── 5. RSI regime ──
    for rsi_col in ['rsi_14_at_0900']:
        extreme = get_pnl_array(con, instrument, session, entry_model, rr, cb, size_min,
                                f"AND (d.{rsi_col} < 30 OR d.{rsi_col} >= 70) AND d.{rsi_col} IS NOT NULL")
        middle = get_pnl_array(con, instrument, session, entry_model, rr, cb, size_min,
                               f"AND d.{rsi_col} >= 30 AND d.{rsi_col} < 70 AND d.{rsi_col} IS NOT NULL")
        if len(extreme) >= 20 and len(middle) >= 20:
            t_stat, p = stats.ttest_ind(middle, extreme, equal_var=False)
            results.append({
                'name': 'RSI_middle_vs_extreme', 'type': 'indicator',
                'n_true': int(len(extreme)), 'mean_true': round(float(np.mean(extreme)), 4),
                'n_false': int(len(middle)), 'mean_false': round(float(np.mean(middle)), 4),
                'delta': round(float(np.mean(middle) - np.mean(extreme)), 4),
                't': round(float(t_stat), 4), 'p': round(float(p), 6),
            })

    # ── 6. DST regime (for affected sessions) ──
    if session in DST_AFFECTED_SESSIONS:
        dst_col = 'us_dst' if session != '1800' else 'uk_dst'
        r = test_binary_split(con, f'DST_{dst_col}', instrument, session,
                              entry_model, rr, cb, size_min,
                              f"d.{dst_col}", "DST_on", "DST_off")
        if r:
            r['type'] = 'dst'
            results.append(r)

    # ── 7. Earlier-session ORB size ──
    for earlier in EARLIER_SESSIONS.get(session, [])[:2]:
        ecol = f"orb_{earlier}_size"
        small = get_pnl_array(con, instrument, session, entry_model, rr, cb, size_min,
                              f"AND d.{ecol} < 4 AND d.{ecol} IS NOT NULL")
        big = get_pnl_array(con, instrument, session, entry_model, rr, cb, size_min,
                            f"AND d.{ecol} >= 4 AND d.{ecol} IS NOT NULL")
        if len(small) >= 20 and len(big) >= 20:
            t_stat, p = stats.ttest_ind(big, small, equal_var=False)
            results.append({
                'name': f'{earlier}_size_big_vs_small', 'type': 'cross_session',
                'n_true': int(len(big)), 'mean_true': round(float(np.mean(big)), 4),
                'n_false': int(len(small)), 'mean_false': round(float(np.mean(small)), 4),
                'delta': round(float(np.mean(big) - np.mean(small)), 4),
                't': round(float(t_stat), 4), 'p': round(float(p), 6),
            })

    # ── 8. Monthly seasonality ──
    for month, month_name in [(1, 'Jan'), (6, 'Jun'), (12, 'Dec')]:
        in_month = get_pnl_array(con, instrument, session, entry_model, rr, cb, size_min,
                                 f"AND EXTRACT(month FROM o.trading_day) = {month}")
        out_month = get_pnl_array(con, instrument, session, entry_model, rr, cb, size_min,
                                  f"AND EXTRACT(month FROM o.trading_day) != {month}")
        if len(in_month) >= 10 and len(out_month) >= 30:
            t_stat, p = stats.ttest_ind(in_month, out_month, equal_var=False)
            results.append({
                'name': f'month_{month_name}', 'type': 'seasonality',
                'n_true': int(len(in_month)), 'mean_true': round(float(np.mean(in_month)), 4),
                'n_false': int(len(out_month)), 'mean_false': round(float(np.mean(out_month)), 4),
                'delta': round(float(np.mean(in_month) - np.mean(out_month)), 4),
                't': round(float(t_stat), 4), 'p': round(float(p), 6),
            })

    # ── Apply BH FDR ──
    if results:
        apply_bh_fdr(results)

    return {
        'instrument': instrument,
        'session': session,
        'entry_model': entry_model,
        'rr': rr, 'cb': cb, 'size_min': size_min,
        'n_raw': n_raw, 'n_filtered': n_filtered,
        'baseline_expr': round(base_expr, 4),
        'baseline_wr': round(base_wr, 4),
        'n_tests': len(results),
        'n_bh_significant': sum(1 for r in results if r.get('bh_significant')),
        'results': results,
        'skipped': False,
    }


def run_discovery(instrument, sessions=None, entry_model='E1', rr=2.0, cb=2,
                  size_min=4.0, output_json=False):
    """Run discovery across one or more sessions."""
    if sessions is None:
        sessions = ALL_SESSIONS

    con = get_connection()
    all_scans = []

    for session in sessions:
        scan = scan_session(con, instrument, session, entry_model, rr, cb, size_min)
        all_scans.append(scan)

        if not output_json:
            if scan.get('skipped'):
                print(f"\n--- {instrument} {session} {entry_model} RR{rr} CB{cb} | SKIPPED: {scan['reason']} ---")
                continue
            if scan.get('error'):
                print(f"\n--- {instrument} {session} | ERROR: {scan['error']} ---")
                continue

            print(f"\n{'='*70}")
            print(f"{instrument} {session} {entry_model} RR{rr} CB{cb} G{int(size_min)}+ | "
                  f"N={scan['n_filtered']} ExpR={scan['baseline_expr']:+.4f} WR={scan['baseline_wr']:.1%}")
            print(f"{'='*70}")

            if not scan['results']:
                print("  No testable predictors (all splits below N threshold)")
                continue

            print(f"  {scan['n_tests']} tests, {scan['n_bh_significant']} BH-significant\n")
            for r in scan['results']:
                sig = " BH-SIG" if r.get('bh_significant') else ""
                raw = "***" if r['p'] < 0.005 else "**" if r['p'] < 0.01 else "*" if r['p'] < 0.05 else ""
                print(f"  {r['name']:<35} delta={r['delta']:>+.4f}  "
                      f"p={r['p']:.4f} {raw:<4} "
                      f"N={r['n_true']}+{r['n_false']}{sig}")

    con.close()

    if output_json:
        print(json.dumps(all_scans, indent=2, default=str))

    return all_scans


def main():
    parser = argparse.ArgumentParser(description="Edge discovery scanner")
    parser.add_argument("--instrument", required=True, help="MGC, MNQ, MES, M2K")
    parser.add_argument("--session", default=None, help="Session label (e.g. 1000, 0900). Omit for all.")
    parser.add_argument("--all-sessions", action="store_true", help="Scan all sessions")
    parser.add_argument("--entry-model", default="E1", help="Entry model (default: E1)")
    parser.add_argument("--rr", type=float, default=2.0, help="RR target (default: 2.0)")
    parser.add_argument("--cb", type=int, default=2, help="Confirm bars (default: 2)")
    parser.add_argument("--size-min", type=float, default=4.0, help="Min ORB size filter (default: 4.0)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    sessions = None
    if args.session:
        sessions = [args.session]
    elif args.all_sessions:
        sessions = ALL_SESSIONS

    run_discovery(
        instrument=args.instrument,
        sessions=sessions,
        entry_model=args.entry_model,
        rr=args.rr, cb=args.cb,
        size_min=args.size_min,
        output_json=args.json,
    )


if __name__ == "__main__":
    main()
```

**Step 2: Test it runs**

```bash
python research/discover.py --instrument MGC --session 1000 --entry-model E1
```

Expected: Prints table of predictor results with BH FDR correction.

**Step 3: Test JSON output**

```bash
python research/discover.py --instrument MGC --session 1000 --json | python -c "import json,sys; d=json.load(sys.stdin); print(f'Tests: {d[0][\"n_tests\"]}, BH-sig: {d[0][\"n_bh_significant\"]}')"
```

**Step 4: Commit**

```bash
git add research/discover.py
git commit -m "feat: generalized edge discovery scanner (research/discover.py)"
```

---

### Task 2: Create the /discover slash command

**Files:**
- Create: `.claude/commands/discover.md`

**Step 1: Write the slash command**

```markdown
Research edge discovery for instrument and session: $ARGUMENTS

## Instructions

You are running an AI-assisted strategy research workflow. Follow these steps exactly.

### Step 1: Parse arguments
Parse $ARGUMENTS for instrument (required) and session (optional).
Examples: "MGC 1000", "MES 0030", "MNQ", "MGC all"
Default entry model: E0 for 1000, E1 for all others.

### Step 2: Check research memory
Use claude-mem:mem-search to check if this (instrument, session) combination has been researched before.
Search query: "discover {instrument} {session} edge research"
If previous findings exist, summarize them before running new scans.

### Step 3: Check current validated state
Use the gold-db MCP server to pull current strategy counts:
- Call query_trading_db with template "validated_summary" for this instrument
- Note how many validated strategies exist for this session already

### Step 4: Run the discovery scanner
Execute the scanner and capture JSON output:
```bash
python research/discover.py --instrument {INSTRUMENT} --session {SESSION} --entry-model {EM} --json
```
If no session specified, add `--all-sessions`.

### Step 5: Interpret results
For each scan result, apply RESEARCH_RULES.md labels:
- BH-significant with p<0.005: "validated finding"
- BH-significant with p<0.05: "promising hypothesis"
- Not BH-significant: "statistical observation" (mention but don't recommend action)
- Baseline ExpR <= 0: "NO-GO — negative baseline"

CRITICAL: For DST-affected sessions (0900, 1800, 0030, 2300), always report winter AND summer splits separately.

### Step 6: Report
Format findings as:

**{INSTRUMENT} {SESSION} Discovery Report**

| Predictor | Delta | p-value | BH-sig? | N | Label |
|-----------|-------|---------|---------|---|-------|
| ... | ... | ... | ... | ... | ... |

Key findings: [2-3 bullet summary]
Recommended actions: [specific next steps, if any survive FDR]

### Step 7: Save to memory
Save findings using claude-mem:mem-search save_memory with key "discover_{instrument}_{session}_{date}".
Include: n_tests, n_significant, top findings, recommended actions.

### Rules (from RESEARCH_RULES.md)
- NEVER say "significant" without p-value
- NEVER say "edge" without BH FDR confirmation
- Sample size labels: <30 INVALID, 30-99 REGIME, 100+ CORE
- RSI/MACD/Bollinger are "guilty until proven" — flag if they appear significant
- Always include year-by-year breakdown for any BH-significant finding
```

**Step 2: Test the slash command**

From Claude Code, run: `/discover MGC 1000`
Verify it follows all 7 steps.

**Step 3: Commit**

```bash
git add .claude/commands/discover.md
git commit -m "feat: /discover slash command for AI-assisted edge research"
```

---

### Task 3: Run drift checks and full test suite

**Step 1: Drift check**

```bash
python pipeline/check_drift.py
```

Expected: All 28+ checks pass.

**Step 2: Full test suite**

```bash
python -m pytest tests/ -x -q
```

Expected: All pass.

**Step 3: Final commit if cleanup needed**

---

### Task 4: Index existing research into memory (one-time)

This is a manual step. After the skill is built, run `/discover` for each active combo to populate the memory baseline:

```bash
# Key combos to index first
/discover MGC 0900
/discover MGC 1000
/discover MGC 1100
/discover MES 1000
/discover MES 0030
/discover MNQ 1000
/discover MNQ 1100
/discover M2K 1000
```

Each run saves findings to claude-mem, building the research memory.

---

## Notes

- `discover.py` uses parameterized queries (not f-strings) for the main data access to prevent SQL injection
- The `prev_day_{session}_outcome` column may not exist for all sessions — the scanner handles this gracefully (empty array = skipped test)
- ATR regime test uses `atr_20_sma50` which may not exist in all builds — check `daily_features` schema before first run
- The slash command is the AI orchestration layer; `discover.py` is the statistical engine. They're intentionally separate so the Python module can also be used standalone
- Pinecone integration deferred to Task 5 (future) — claude-mem provides sufficient memory for now
