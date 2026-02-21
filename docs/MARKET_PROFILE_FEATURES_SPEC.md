# Market Profile Features Spec

## Purpose

Add prior-session Market Profile context to `daily_features` so ORB strategies can be
filtered/enriched by Value Area, Point of Control, and Opening Type. These are established
structural features with clear market mechanisms — testable as pre-entry filters alongside
existing ORB size filters (G4/G5/G6+).

**Research questions this enables:**
1. Do ORB breaks that open *above VAH* (acceptance) have higher follow-through than inside-VA opens?
2. Does "Open Drive" opening type predict ORB success better than pure ORB size?
3. Is prior POC a better exit target than a fixed R-multiple?
4. Does IB size (first 60-min range) as a compression filter beat or complement ORB_G4+?

---

## Definitions

### Value Area (VA)
The price range in which ~70% of the prior session's volume traded.
- **VAH** — Value Area High (upper boundary)
- **VAL** — Value Area Low (lower boundary)
- **POC** — Point of Control (price level with highest volume)

### Opening Scenario (4 types)
Where today's open sits relative to prior session's VA:
- `above_vah` — opened above yesterday's VAH (bullish acceptance / 80% rule long setup)
- `below_val` — opened below yesterday's VAL (bearish acceptance / 80% rule short setup)
- `inside_va` — opened inside value (mean reversion environment, contested ground)
- `at_boundary` — within 1 ATR% tick of VAH or VAL (decision point)

### Opening Type (4 types — Steidlmayer)
How the first 30-60 min of the session behaves relative to the session open:
- `OD`  — Open Drive: immediate directional move, no pullback to open. OTF conviction.
- `OTD` — Open Test Drive: probes beyond a level, finds no business, reverses hard through open.
- `ORR` — Open Rejection Reverse: moves one direction, rejected, returns through open. Lower conviction.
- `OA`  — Open Auction: two-sided rotation, no committed OTF participant.

### Initial Balance (IB)
The price range formed in the first 60 minutes of the regular session.
- `IB_high`, `IB_low`, `IB_size`
- `IB_compression` = IB_size / prior_atr (similar to ORB_G4+ but on 60-min range)

---

## New Columns for `daily_features`

All columns are per `(trading_day, symbol)`. Since daily_features has 3 rows per day
(orb_minutes=5/15/30), these columns repeat identically across all three rows —
they are session-level context, not ORB-specific.

### Prior Session Value Area
| Column | Type | Description |
|--------|------|-------------|
| `prior_vah` | DOUBLE | Yesterday's Value Area High |
| `prior_val` | DOUBLE | Yesterday's Value Area Low |
| `prior_poc` | DOUBLE | Yesterday's Point of Control |
| `prior_va_width` | DOUBLE | prior_vah - prior_val |
| `prior_va_width_pct` | DOUBLE | prior_va_width / prior_poc (normalised) |

### Opening Scenario
| Column | Type | Description |
|--------|------|-------------|
| `open_vs_prior_va` | VARCHAR | 'above_vah' / 'inside_va' / 'below_val' / 'at_boundary' |
| `open_dist_from_vah` | DOUBLE | open_price - prior_vah (negative = below VAH) |
| `open_dist_from_val` | DOUBLE | open_price - prior_val |
| `open_dist_from_poc` | DOUBLE | open_price - prior_poc |

### Initial Balance (session-specific, 60-min)
| Column | Type | Description |
|--------|------|-------------|
| `ib_{label}_high` | DOUBLE | IB high for session (e.g. ib_1000_high) |
| `ib_{label}_low` | DOUBLE | IB low for session |
| `ib_{label}_size` | DOUBLE | IB range |
| `ib_{label}_compression` | DOUBLE | ib_size / atr_14 (like ORB_G filter) |

### Opening Type (session-specific)
| Column | Type | Description |
|--------|------|-------------|
| `open_type_{label}` | VARCHAR | OD / OTD / ORR / OA for session (e.g. open_type_1000) |

---

## Computation Algorithm

### Step 1: Volume-at-Price Profile (per trading_day, symbol)

```sql
-- Distribute each bar's volume evenly across its tick range
-- tick_size from asset_configs (MGC=0.1, MES=0.25, MNQ=0.25)
WITH bar_ticks AS (
    SELECT
        b.trading_day,
        b.symbol,
        UNNEST(GENERATE_SERIES(
            CAST(ROUND(b.low  / t.tick_size) AS INTEGER),
            CAST(ROUND(b.high / t.tick_size) AS INTEGER)
        )) AS tick_idx,
        b.volume / GREATEST(1,
            CAST(ROUND((b.high - b.low) / t.tick_size) AS INTEGER) + 1
        ) AS vol_per_tick
    FROM bars_1m b
    JOIN tick_sizes t USING (symbol)
    -- Restrict to prior trading day's bars only
),
vap AS (
    SELECT
        trading_day,
        symbol,
        tick_idx * tick_size AS price_level,
        SUM(vol_per_tick) AS total_vol
    FROM bar_ticks
    GROUP BY 1, 2, 3
)
```

### Step 2: Find POC

```sql
poc AS (
    SELECT DISTINCT ON (trading_day, symbol)
        trading_day, symbol, price_level AS poc
    FROM vap
    ORDER BY trading_day, symbol, total_vol DESC
)
```

### Step 3: Expand from POC to Capture 70% of Volume

```sql
-- Running cumulative sum expanding outward from POC
-- When cumsum >= 0.70 * total_volume → stop; record VAH and VAL
-- This requires iterative expansion — best implemented in Python
-- (DuckDB recursive CTEs can do it but it's verbose)
```

**Implementation note:** The 70% expansion is easiest in Python with pandas:
```python
def compute_value_area(vap_df, target_pct=0.70):
    """
    vap_df: DataFrame with columns [price_level, total_vol], sorted by price_level.
    Returns (val, poc, vah).
    """
    total_vol = vap_df['total_vol'].sum()
    poc_price = vap_df.loc[vap_df['total_vol'].idxmax(), 'price_level']

    poc_idx = vap_df.index[vap_df['price_level'] == poc_price][0]
    up_idx = poc_idx
    dn_idx = poc_idx
    cumvol = vap_df.loc[poc_idx, 'total_vol']

    while cumvol / total_vol < target_pct:
        up_vol = vap_df.loc[up_idx + 1, 'total_vol'] if up_idx + 1 < len(vap_df) else 0
        dn_vol = vap_df.loc[dn_idx - 1, 'total_vol'] if dn_idx > 0 else 0
        if up_vol >= dn_vol:
            up_idx += 1
            cumvol += up_vol
        else:
            dn_idx -= 1
            cumvol += dn_vol

    vah = vap_df.loc[up_idx, 'price_level']
    val = vap_df.loc[dn_idx, 'price_level']
    return val, poc_price, vah
```

### Step 4: Opening Scenario Classification

```python
def classify_open_vs_va(open_price, vah, val, boundary_ticks=2, tick_size=0.1):
    boundary = boundary_ticks * tick_size
    if open_price > vah + boundary:
        return 'above_vah'
    elif open_price < val - boundary:
        return 'below_val'
    elif abs(open_price - vah) <= boundary or abs(open_price - val) <= boundary:
        return 'at_boundary'
    else:
        return 'inside_va'
```

### Step 5: Opening Type Classification (per session)

Uses bars_1m for the first 30-60 minutes of the session:

```python
def classify_open_type(bars_first_30min, session_open_price):
    """
    bars_first_30min: DataFrame of 1-min bars in first 30 min of session.
    Returns: 'OD', 'OTD', 'ORR', or 'OA'
    """
    opens_up = bars_first_30min.iloc[-1]['close'] > session_open_price

    # Check if price ever returned to open
    returned_to_open = (
        (bars_first_30min['low'] <= session_open_price).any() and
        (bars_first_30min['high'] >= session_open_price).any()
    )

    # OD: moved away from open and never came back
    if not returned_to_open:
        return 'OD'

    # Check for hard reversal through open
    crossed_open = (
        bars_first_30min['high'].max() > session_open_price and
        bars_first_30min['low'].min() < session_open_price
    )

    if crossed_open:
        # Determine if it probed one side then reversed (OTD/ORR)
        # OTD: opens, tests beyond key level, reverses; OTD > ORR in conviction
        # For simplicity: classify by magnitude of reversal
        range_above = bars_first_30min['high'].max() - session_open_price
        range_below = session_open_price - bars_first_30min['low'].min()
        if max(range_above, range_below) / min(range_above + 0.001, range_below + 0.001) > 2.5:
            return 'OTD'  # Strongly tested one side
        else:
            return 'ORR'  # Two-sided but one side rejected

    return 'OA'  # Two-sided, no committed direction
```

### Step 6: IB Classification

```python
def compute_ib(bars_first_60min, atr_14):
    ib_high = bars_first_60min['high'].max()
    ib_low = bars_first_60min['low'].min()
    ib_size = ib_high - ib_low
    ib_compression = ib_size / atr_14 if atr_14 > 0 else None
    return ib_high, ib_low, ib_size, ib_compression
```

---

## Research Hypotheses (Testable Post-Implementation)

### H1: Opening Scenario as ORB Filter
**Hypothesis:** ORB LONG trades taken when `open_vs_prior_va = 'above_vah'` have higher
ExpR than the unfiltered baseline. Market is in acceptance above value — momentum aligns.

**Test:** Split orb_outcomes by `open_vs_prior_va`, compare ExpR by scenario.
**Expected:** above_vah LONG > inside_va LONG. below_val SHORT > inside_va SHORT.
**Mechanism:** Opening outside VA = OTF participant with directional commitment.
Inside VA = contested ground, both sides active.

### H2: Open Drive as Quality Filter
**Hypothesis:** ORB trades where `open_type = 'OD'` have higher ExpR and win rate.
OD = "other timeframe" (institutional) conviction from the open.

**Test:** Split orb_outcomes by `open_type_{session}`, compare ExpR.
**Expected:** OD >> OA. ORR worst (initial extreme holds <50% of the time).
**Mechanism:** OD days are where price was committed from the first tick.

### H3: IB Compression as Second-Timeframe Filter
**Hypothesis:** Sessions where `ib_compression < 0.5` (IB < 50% of ATR) produce
higher-quality ORB breaks than IB_compression > 1.0.

**Test:** Decile IB_compression, compare ORB ExpR per decile.
**Expected:** Tight IB → explosive expansion when break occurs.
**Mechanism:** IB is the 60-min ORB. Tight IB + tight 5-min ORB = two-timeframe squeeze.

### H4: POC as Exit Target vs R-Multiple
**Hypothesis:** Exiting at prior POC (instead of fixed R-multiple) improves total P&L
by letting winners run to a structural level with high volume acceptance.

**Test:** Simulate exit at POC vs current R-multiple exits. Compare total R captured.
**Expected:** Sessions with POC far from entry (room to run) benefit most.
**Mechanism:** POC is the fairest price — market gravitates back to it. Breakout trades
use POC as a magnet on the other side of the range.

---

## Implementation Plan

### Phase A: Prior Session VA (1-2 days work)
1. New script: `pipeline/build_market_profile.py`
   - Reads bars_1m for each trading_day
   - Computes VAP using tick-level distribution
   - Computes POC, VAH, VAL via 70% expansion
   - Writes to new table: `market_profile` (trading_day, symbol, poc, vah, val)

2. Update `build_daily_features.py`:
   - JOIN `market_profile` for prior trading_day
   - Add `prior_poc`, `prior_vah`, `prior_val`, `prior_va_width`, `open_vs_prior_va`

### Phase B: Opening Type + IB (1 day work)
3. Add to `build_daily_features.py`:
   - For each active session label: compute IB (first 60 min bars)
   - Classify `open_type_{label}` from first 30-min bars
   - Add all new columns

### Phase C: Research
4. New script: `research/research_market_profile.py`
   - Tests H1/H2/H3 with BH FDR correction
   - Reports survival status per hypothesis

### Phase D: Wiring (if research survives BH)
5. Add MP-based filters to `config.py` (e.g. `MP_ABOVE_VAH`, `MP_OD_ONLY`)
6. Include in strategy grid via `get_filters_for_grid()`

---

## Tick Sizes by Instrument
| Symbol | Tick Size | Notes |
|--------|-----------|-------|
| MGC | 0.10 | Micro Gold, $0.10/tick = $1 |
| MES | 0.25 | Micro S&P, $0.25/tick = $1.25 |
| MNQ | 0.25 | Micro Nasdaq, $0.25/tick = $0.50 |
| MCL | 0.01 | Micro Crude (dead — skip) |

---

## Caveats and Risks

- **Compute cost:** VAP requires tick-level distribution of every bar. For 10yr of bars_1m
  at 390+ bars/day, this is ~1.4M bars → ~140M tick assignments. Run once, store results.
- **Tick approximation:** Distributing bar volume equally across its high-low range is an
  approximation. Real volume profile requires tick-by-tick data. This is a known limitation.
- **Prior day definition:** Brisbane trading day (09:00→09:00). Need to confirm this aligns
  with CME settlement day for each session.
- **First trading day of year:** No prior-day data available. Rows will have NULL for VA columns.
- **Overnight session:** bars_1m includes overnight bars. Decide: use full 24-hr session
  volume for VA, or only RTH (Regular Trading Hours)? Full session is more data; RTH is
  more relevant for day traders. **Recommend: full session for MGC (gold trades 23h/day);
  RTH for MES/MNQ (09:30–16:00 ET dominates volume).**
