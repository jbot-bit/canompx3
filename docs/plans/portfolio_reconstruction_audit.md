EXECUTE: PORTFOLIO RECONSTRUCTION AUDIT
========================================
Three sequential parts. READ ONLY for Parts 1-2.
Part 3 is a CONFIG change (not new class).
No pipeline runs. No validation changes.
No FDR re-run until Part 1 is complete.

VERIFIED AGAINST SCHEMA: 2026-03-27
------------------------------------
Key facts (proven, not assumed):
- risk_dollars = risk_pts * point_value + total_friction
  (MNQ: risk_pts * 2.0 + 2.74)
  Therefore: orb_size_pts = (risk_dollars - 2.74) / 2.0
  NOT risk_dollars / 2.0 (that overstates by 1.37 pts)
- pnl_r is POST-friction (costs baked in)
- stop_multiplier NOT in orb_outcomes — applied post-hoc
  in strategy_discovery.apply_tight_stop() using MAE
  orb_outcomes stores 1.0x stop outcomes ONLY
- 268/470 MNQ strategies use stop_multiplier = 0.75
  Their pnl_r in orb_outcomes is WRONG (1.0x not 0.75x)
  yearly_results JSON in validated_setups is the TRUTH
  for per-strategy returns (pre-computed with correct
  stop_mult, direction filter, and size filter applied)
- direction column does NOT exist in orb_outcomes,
  validated_setups, or experimental_strategies
  Direction inferred from: stop_price < entry_price = LONG
  DIR_LONG filter matches daily_features.orb_{sess}_break_dir
- strategy_trade_days: only 35/470 MNQ covered (7.4%)
  UNUSABLE as primary Jaccard source
- G-filters are POINT thresholds (not dollar):
  G4=4pts, G5=5pts, G6=6pts, G8=8pts
  Applied via: daily_features.orb_{session}_size >= min_size
- Position sizing: 2% risk per trade (global, not per-strategy)
  Defined in trading_app/portfolio.py:153
- LIVE_PORTFOLIO in live_config.py is DEPRECATED
  Actual authority: trading_app/prop_profiles.py
- All 470 MNQ FDR-significant have status = 'active'

================================================
PART 1 — COMEX_SETTLE MISSING JACCARD +
          COMPLETE THE INDEPENDENCE TABLE
================================================

STEP 1A — Why is COMEX_SETTLE "—"?

Check edge_families for COMEX_SETTLE MNQ.
Note: edge_families has no `session` column —
session derived by joining validated_setups
on head_strategy_id.

```sql
-- Get COMEX_SETTLE families via validated_setups join
SELECT ef.family_hash, ef.instrument, ef.member_count,
       ef.robustness_status, ef.head_strategy_id,
       vs.orb_label
FROM edge_families ef
JOIN validated_setups vs ON ef.head_strategy_id = vs.strategy_id
WHERE ef.instrument = 'MNQ'
AND vs.orb_label = 'COMEX_SETTLE';
```

If 0 families: edge builder excluded COMEX_SETTLE.
Check if active strategies exist:
```sql
SELECT COUNT(*), status
FROM validated_setups
WHERE instrument = 'MNQ'
AND orb_label = 'COMEX_SETTLE'
AND fdr_significant = true
GROUP BY status;
```
If active strategies exist but no families → builder gap.
If no active strategies → no families expected (state why).

STEP 1B — Compute Jaccard for COMEX_SETTLE

Jaccard is NOT built into the family builder
(families use exact hash equality, not Jaccard).
Prior audit computed Jaccard ad-hoc from orb_outcomes.

IMPORTANT: For CORRECT trade-day sets, you MUST
apply the strategy's filter. Naive orb_outcomes
join returns ALL break-day outcomes regardless of
filter eligibility. For Jaccard between families
with DIFFERENT filter_types, this inflates overlap.

Approach for Jaccard trade-day extraction:
```python
# For each family head, get its ACTUAL trade days
# by applying its filter to orb_outcomes

# For NO_FILTER / DIR_LONG:
#   NO_FILTER: all break days (both directions)
#   DIR_LONG: only days where stop_price < entry_price
#
# For G-filters (ORB_G4 through ORB_G8):
#   Join daily_features for orb_{session}_size >= threshold
#   G4: >= 4.0, G5: >= 5.0, G6: >= 6.0, G8: >= 8.0
#
# For ATR70_VOL:
#   Join daily_features for atr_20_pct >= 70
#   AND rel_vol_{session} >= 1.2
#
# For VOL_RV12_N20:
#   Join daily_features for rel_vol_{session} >= 1.2
#
# For X_MES_ATR60/70, X_MGC_ATR70:
#   Cross-asset ATR filter — requires MES/MGC daily_features
#
# SIMPLIFICATION: Use family_hash equality to identify
# SAME-family strategies (Jaccard = 1.0 by definition).
# Only compute pairwise Jaccard BETWEEN family heads
# in the same session. Since we're computing overlap
# of TRADING DAYS (not returns), and most families
# in the same session share the same orb_minutes +
# entry_model, the key differentiator is the filter.
#
# APPROXIMATION ACCEPTABLE: Use orb_outcomes join
# (without filter) as UPPER BOUND on trade-day overlap.
# If upper-bound Jaccard < 0.5, true Jaccard is also < 0.5.
# If upper-bound Jaccard > 0.5, note as approximate.

import duckdb
from itertools import combinations

con = duckdb.connect('gold.db', read_only=True)

# Get COMEX_SETTLE family heads
heads = con.execute("""
    SELECT ef.family_hash, vs.orb_minutes,
           vs.entry_model, vs.rr_target, vs.confirm_bars,
           vs.filter_type, vs.stop_multiplier
    FROM edge_families ef
    JOIN validated_setups vs
        ON ef.head_strategy_id = vs.strategy_id
    WHERE ef.instrument = 'MNQ'
    AND vs.orb_label = 'COMEX_SETTLE'
""").fetchall()

# For each head, get trade day set from orb_outcomes
# NOTE: This is the UNFILTERED set (upper bound)
# stop_multiplier does NOT affect which days trade,
# only the outcome — so it doesn't affect Jaccard
family_days = {}
for h in heads:
    fhash, om, em, rr, cb, ft, sm = h
    days = con.execute("""
        SELECT DISTINCT trading_day
        FROM orb_outcomes
        WHERE symbol = 'MNQ'
        AND orb_label = 'COMEX_SETTLE'
        AND orb_minutes = ?
        AND entry_model = ?
        AND rr_target = ?
        AND confirm_bars = ?
        AND outcome IN ('win', 'loss')
    """, [om, em, rr, cb]).fetchall()
    family_days[fhash] = set(d[0] for d in days)

# Pairwise Jaccard
for (a, b) in combinations(family_days.keys(), 2):
    inter = len(family_days[a] & family_days[b])
    union = len(family_days[a] | family_days[b])
    j = inter / union if union > 0 else 0
    if j > 0.5:
        print(f"  {a[:12]} vs {b[:12]}: J={j:.3f}")
```

For DIR_LONG families, add direction filter:
```sql
AND oo.stop_price < oo.entry_price  -- LONG only
```

STEP 1C — Complete the independence table

Once COMEX_SETTLE Jaccard is computed,
produce the complete table with no "—" gaps:

| session | raw | families | jaccard_gt05_pct |
  true_independent | concentration_flag |

concentration_flag:
"CRITICAL" if true_independent <= 2
"HIGH"     if true_independent 3-5
"MODERATE" if true_independent 6-10
"OK"       if true_independent > 10

Output complete table. No missing sessions.

================================================
PART 2 — PORTFOLIO CONSTRUCTION ON TRUE N
================================================

The portfolio was sized assuming diversification
across validated strategies. True independent N ≈ 60.
This changes three things that must be computed:

STEP 2A — Correlated drawdown exposure

DO NOT use orb_outcomes join for returns.
Use yearly_results JSON from validated_setups.
This is the ONLY source that correctly reflects
stop_multiplier, direction filter, and size filter.

For session-level daily correlation, use one family
head per session as proxy. Extract daily R from
yearly_results (which has per-year aggregates)
or use orb_outcomes for the specific family head
(since family heads have known filter_type, and
we can apply direction/filter correctly for ONE
strategy rather than trying to do it in bulk).

```python
import duckdb, json
con = duckdb.connect('gold.db', read_only=True)

# Get one representative family head per session
# Prefer NO_FILTER heads (simplest join, no filter needed)
# Fall back to heads where filter is easy to apply
heads = con.execute("""
    SELECT vs.strategy_id, vs.orb_label, vs.orb_minutes,
           vs.entry_model, vs.rr_target, vs.confirm_bars,
           vs.filter_type, vs.stop_multiplier, vs.yearly_results
    FROM edge_families ef
    JOIN validated_setups vs
        ON ef.head_strategy_id = vs.strategy_id
    WHERE ef.instrument = 'MNQ'
    AND ef.robustness_status IN ('ROBUST', 'WHITELISTED')
    ORDER BY vs.orb_label, ef.head_expectancy_r DESC
""").fetchall()

# Group by session, take top head per session
session_heads = {}
for h in heads:
    sess = h[1]
    if sess not in session_heads:
        session_heads[sess] = h

# For correlation: use yearly_results to get per-year R
# This gives annual granularity (not daily), but is the
# only reliable source for S075/filtered strategies.
# For DAILY correlation, we'd need to reconstruct from
# orb_outcomes with correct filter application per head.

# OPTION A: Annual correlation (simpler, safe)
print("=== Annual R correlation matrix (from yearly_results) ===")
import numpy as np
sessions = sorted(session_heads.keys())
year_range = range(2016, 2026)
matrix = {}
for sess in sessions:
    yr = json.loads(session_heads[sess][8])
    matrix[sess] = [yr.get(str(y), {}).get('total_r', 0) for y in year_range]

for i, s1 in enumerate(sessions):
    for j, s2 in enumerate(sessions):
        if j > i:
            r = np.corrcoef(matrix[s1], matrix[s2])[0, 1]
            if abs(r) > 0.3:
                print(f"  {s1} vs {s2}: r={r:.3f}")

# OPTION B: Daily correlation (if needed, for specific heads)
# Only do this for heads with filter_type = 'NO_FILTER'
# or where the filter is a pure size gate (G-filters)
# For G-filter heads: join daily_features to apply size gate
```

CONFIRMED: pnl_r in orb_outcomes is POST-friction
(risk_dollars includes $2.74 friction; pnl_r deducts
friction from both numerator and denominator).

STEP 2B — Concentration by true independent unit

Use yearly_results JSON for each family head.
Sum total_r across all years per head.

```python
import duckdb, json
con = duckdb.connect('gold.db', read_only=True)

# Get all ROBUST + WHITELISTED family heads
heads = con.execute("""
    SELECT ef.family_hash, ef.head_strategy_id,
           ef.robustness_status, ef.member_count,
           vs.orb_label, vs.yearly_results
    FROM edge_families ef
    JOIN validated_setups vs
        ON ef.head_strategy_id = vs.strategy_id
    WHERE ef.instrument = 'MNQ'
    AND ef.robustness_status IN ('ROBUST', 'WHITELISTED')
""").fetchall()

total_portfolio_r = 0
unit_r = []
for h in heads:
    fhash, sid, robust, members, sess, yr_json = h
    yr = json.loads(yr_json) if yr_json else {}
    r = sum(v.get('total_r', 0) for v in yr.values())
    total_portfolio_r += r
    unit_r.append((sess, fhash[:12], sid, r, members))

# Sort by R contribution
unit_r.sort(key=lambda x: x[3], reverse=True)
cumulative = 0
for sess, fh, sid, r, m in unit_r:
    cumulative += r
    pct = r / total_portfolio_r * 100 if total_portfolio_r else 0
    cum_pct = cumulative / total_portfolio_r * 100 if total_portfolio_r else 0
    print(f"  {sess:25s} {fh} R={r:>8.1f} ({pct:>5.1f}%) cum={cum_pct:>5.1f}%")
```

If any single independent unit > 20% of total R:
"WARNING: CONCENTRATION — [unit] drives >20%
 of total portfolio R. Not diversified."

If top 3 independent units > 50% of total R:
"WARNING: TOP-HEAVY PORTFOLIO — 3 units drive
 majority of return. True diversification
 requires different session mix."

STEP 2C — Position sizing implication

Current rule: 2% risk per trade (global).
Source: trading_app/portfolio.py line 153.
Applied identically to all strategies regardless
of independence.

With 470 strategies, if multiple fire on the same
day across correlated families, actual portfolio
risk = 2% × concurrent positions.

With true independent N ≈ 60:
- Max concurrent positions per session = 1
  (if trading family heads only)
- Max concurrent INDEPENDENT positions across
  sessions = number of sessions with trades that day
- Portfolio risk is bounded by: 2% × max_concurrent

Check prop_profiles.py for whether it already
accounts for this:
```python
# Read trading_app/prop_profiles.py for position limits
# Does it specify max concurrent positions?
# Does it reference family heads or raw strategies?
```

If position sizing assumes N=470 diversification:
"WARNING: POSITION SIZING BUILT ON 470 strategies.
 TRUE INDEPENDENT EXPOSURE IS ~60 units.
 Concurrent risk is [X]x higher than designed."

If position sizing uses 1 contract per strategy
(prop standard):
"POSITION SIZING IS 1-LOT PER STRATEGY.
 True risk is bounded by concurrent sessions,
 not strategy count. N=470 vs N=60 is cosmetic
 for 1-lot sizing."

NOTE: LIVE_PORTFOLIO in live_config.py is DEPRECATED.
Actual authority: trading_app/prop_profiles.py

================================================
PART 3 — THE 23 EXPOSED STRATEGIES:
          APPLY MINIMUM ORB FLOOR
================================================

Prior audit confirmed: 23 NO_FILTER + DIR_LONG
strategies have zero protection if low-vol returns.
(16 NO_FILTER + 7 DIR_LONG, all SINGAPORE_OPEN)

CRITICAL: G-filters (OrbSizeFilter) ALREADY implement
minimum ORB size gates. G4 = min 4 pts, G5 = min 5 pts,
G6 = min 6 pts, G8 = min 8 pts. These are POINT thresholds
applied via daily_features.orb_{session}_size >= min_size.

DO NOT build a new MinimumORBFilter class.
The fix is a CONFIG CHANGE: promote the 23
strategies to require G5+ (or appropriate threshold).

STEP 3A — Define the kill switch threshold

CRITICAL FORMULA:
  orb_size_pts = (risk_dollars - 2.74) / 2.0
  NOT risk_dollars / 2.0
  Because risk_dollars = risk_pts * point_value + total_friction
  Confirmed: every row in orb_outcomes satisfies
  risk_dollars = ABS(entry - stop) * 2.0 + 2.74

```python
import duckdb
con = duckdb.connect('gold.db', read_only=True)

# Breakeven ORB size analysis for exposed strategies
# Split NO_FILTER (both dirs) vs DIR_LONG (long only)

# NO_FILTER strategies (16 total, various sessions)
no_filter_sessions = con.execute("""
    SELECT DISTINCT orb_label, orb_minutes, entry_model,
           rr_target, confirm_bars
    FROM validated_setups
    WHERE instrument = 'MNQ' AND fdr_significant = true
    AND filter_type = 'NO_FILTER'
""").fetchall()

print("=== NO_FILTER: ORB size vs net R (all directions) ===")
for sess_params in no_filter_sessions[:5]:  # top 5
    sess, om, em, rr, cb = sess_params
    result = con.execute("""
        SELECT ROUND((risk_dollars - 2.74) / 2.0, 0) as orb_bin,
               COUNT(*) as n_trades,
               AVG(CASE WHEN outcome='win' THEN 1.0 ELSE 0.0 END) as wr,
               AVG(pnl_r) as avg_net_r,
               AVG(2.74 / risk_dollars) as avg_friction_pct
        FROM orb_outcomes
        WHERE symbol = 'MNQ'
        AND orb_label = ?
        AND orb_minutes = ? AND entry_model = ?
        AND rr_target = ? AND confirm_bars = ?
        AND outcome IN ('win', 'loss')
        GROUP BY ROUND((risk_dollars - 2.74) / 2.0, 0)
        ORDER BY orb_bin
    """, [sess, om, em, rr, cb]).fetchall()
    print(f"\n  {sess} O{om} {em} RR{rr}:")
    for r in result:
        print(f"    {r[0]:>4.0f} pts | n={r[1]:>4} | WR={r[2]:.3f} | net_r={r[3]:>+.4f} | friction={r[4]:.1%}")

# DIR_LONG strategies (7 total, all SINGAPORE_OPEN)
print("\n=== DIR_LONG: ORB size vs net R (LONG only) ===")
result = con.execute("""
    SELECT ROUND((risk_dollars - 2.74) / 2.0, 0) as orb_bin,
           COUNT(*) as n_trades,
           AVG(CASE WHEN outcome='win' THEN 1.0 ELSE 0.0 END) as wr,
           AVG(pnl_r) as avg_net_r,
           AVG(2.74 / risk_dollars) as avg_friction_pct
    FROM orb_outcomes
    WHERE symbol = 'MNQ'
    AND orb_label = 'SINGAPORE_OPEN'
    AND orb_minutes = 30 AND entry_model = 'E2'
    AND rr_target = 2.0 AND confirm_bars = 1
    AND outcome IN ('win', 'loss')
    AND stop_price < entry_price  -- LONG direction only
    GROUP BY ROUND((risk_dollars - 2.74) / 2.0, 0)
    ORDER BY orb_bin
""").fetchall()
for r in result:
    print(f"  {r[0]:>4.0f} pts | n={r[1]:>4} | WR={r[2]:.3f} | net_r={r[3]:>+.4f} | friction={r[4]:.1%}")

# Find: at what orb_size_pts does avg_net_r cross zero?
# Map to nearest G-filter threshold (G4=4, G5=5, G6=6, G8=8)
```

MNQ cost model reference (from pipeline/cost_model.py):
  commission_rt = $1.24
  spread_doubled = $0.50
  slippage = $1.00
  total_friction = $2.74
  point_value = $2.00
  friction_in_points = 1.37 pts
  tick_size = 0.25

STEP 3B — Define the config change (DESIGN ONLY)

DO NOT implement yet. Present the proposal:

For the 23 NO_FILTER + DIR_LONG strategies:
- Option A: Promote all 23 to require G[threshold from 3A]
  This means RE-RUNNING discovery + validation with the
  new filter_type. The strategies would get new strategy_ids.
  Validated counts would change.
- Option B: Add a portfolio-level minimum ORB floor
  in prop_profiles.py (NOT live_config.py which is deprecated)
  that rejects trades below threshold at execution time.
  No re-validation needed. Simpler. But filter not in backtest.
- Option C: Archive the 23 and accept reduced strategy count.
  The 23 contribute < X% of portfolio R (compute from Part 2B).
  If contribution is marginal, archiving is cheapest.

Present tradeoffs. Wait for user decision.

STEP 3C — Retrospective impact analysis (READ ONLY)

For the 23 exposed strategies, simulate
the proposed G-filter threshold retrospectively.

NOTE: For S075 strategies among the 23, use
yearly_results JSON for accurate R impact.
For the orb_size threshold analysis, orb_outcomes
is acceptable because we're looking at ORB SIZE
distribution (which doesn't change with stop_mult),
not at returns (which do).

```python
import duckdb, json
con = duckdb.connect('gold.db', read_only=True)

THRESHOLD = 5  # Replace with value from Step 3A

# For ORB SIZE distribution by year (stop_mult irrelevant):
result = con.execute("""
    WITH exposed AS (
        SELECT DISTINCT orb_label, orb_minutes,
               entry_model, rr_target, confirm_bars, filter_type
        FROM validated_setups
        WHERE instrument = 'MNQ'
        AND fdr_significant = true
        AND filter_type IN ('NO_FILTER', 'DIR_LONG')
    ),
    trades AS (
        SELECT e.filter_type,
               EXTRACT(YEAR FROM oo.trading_day) as yr,
               (oo.risk_dollars - 2.74) / 2.0 as orb_size_pts,
               oo.stop_price, oo.entry_price
        FROM exposed e
        JOIN orb_outcomes oo
            ON oo.symbol = 'MNQ'
            AND oo.orb_label = e.orb_label
            AND oo.orb_minutes = e.orb_minutes
            AND oo.rr_target = e.rr_target
            AND oo.confirm_bars = e.confirm_bars
            AND oo.entry_model = e.entry_model
        WHERE oo.outcome IN ('win', 'loss')
        -- Apply DIR_LONG filter where needed:
        AND (e.filter_type != 'DIR_LONG'
             OR oo.stop_price < oo.entry_price)
    )
    SELECT yr,
           COUNT(*) as total_trades,
           SUM(CASE WHEN orb_size_pts < ? THEN 1 ELSE 0 END) as blocked,
           ROUND(100.0 * SUM(CASE WHEN orb_size_pts < ? THEN 1 ELSE 0 END)
                 / COUNT(*), 1) as pct_blocked
    FROM trades
    GROUP BY yr
    ORDER BY yr
""", [THRESHOLD, THRESHOLD]).fetchall()

print(f"Threshold: G{THRESHOLD} ({THRESHOLD} pts)")
print(f"{'Year':>6} | {'Total':>6} | {'Blocked':>7} | {'Pct':>6}")
for r in result:
    print(f"{int(r[0]):>6} | {int(r[1]):>6} | {int(r[2]):>7} | {r[3]:>5.1f}%")
```

Target: blocks >80% of low-vol era trades (2016-2017),
blocks <10% of high-vol era trades (2020+).

If filter doesn't achieve this:
"WARNING: G[X] THRESHOLD DOES NOT CLEANLY
 SEPARATE REGIMES — review threshold
 or consider session-specific floors"

================================================
OUTPUT FORMAT — STRICT:
================================================
Part 1: Complete independence table
        (no missing sessions)
Part 2: Correlated session pairs,
        concentration table,
        position sizing discrepancy
Part 3: Minimum viable ORB from trade data,
        config change PROPOSAL (not implemented),
        retrospective simulation table

THEN STOP.
Write: "READY FOR YOUR REVIEW —
confirm before I interpret."

================================================
HONESTY FLAGS:
================================================
"WARNING: COMEX_SETTLE JACCARD STILL MISSING —
 independence table incomplete,
 portfolio construction cannot proceed"

"WARNING: CONCENTRATION — single unit >20% of R"

"WARNING: TOP-HEAVY — top 3 units >50% of R"

"WARNING: POSITION SIZING DISCREPANCY —
 true exposure differs from designed exposure"

"WARNING: G[X] FILTER DOES NOT CLEANLY
 SEPARATE REGIMES — threshold needs revision"

================================================
KNOWN SCHEMA TRAPS (DO NOT REPEAT):
================================================
These were found by schema audit on 2026-03-27.
Any executor running this plan MUST NOT:

1. Use risk_dollars / 2.0 for orb_size_pts
   CORRECT: (risk_dollars - 2.74) / 2.0
   (risk_dollars includes $2.74 friction)

2. Use orb_outcomes.pnl_r for S075 strategies
   CORRECT: validated_setups.yearly_results JSON
   (orb_outcomes has 1.0x stop outcomes only;
   S075 outcomes computed post-hoc in discovery)

3. Query orb_outcomes for DIR_LONG without
   AND stop_price < entry_price
   (direction column doesn't exist anywhere;
   long = stop below entry)

4. Assume strategy_trade_days has full coverage
   ACTUAL: 35/470 MNQ strategies (7.4%)
   (unusable as primary source)

5. Reference LIVE_PORTFOLIO in live_config.py
   ACTUAL AUTHORITY: trading_app/prop_profiles.py
   (live_config.py LIVE_PORTFOLIO is deprecated)

6. Reference expected_r or exp_r as column name
   CORRECT: expectancy_r (in validated_setups)

7. Reference daily_strategy_returns table
   (does not exist — phantom from vanilla Claude)

8. Reference gross_r, friction_r, or_size columns
   in orb_outcomes (do not exist — phantom)

9. Reference validation_status in validated_setups
   CORRECT: status (VARCHAR, currently all 'active')

10. Assume edge_families has a session column
    CORRECT: join via head_strategy_id to
    validated_setups.orb_label for session
