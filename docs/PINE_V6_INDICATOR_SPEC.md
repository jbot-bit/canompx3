# Pine Script v6 — ORB Breakout Indicator Spec

Complete reference for building TradingView indicators from our validated ORB breakout system.
Give this entire doc to any AI building Pine v6 indicators.

---

## 1. Instruments

All CME micro futures. Use TradingView continuous front-month symbols.

| Instrument | TV Symbol | Point Value | Tick Size | Min Ticks (risk floor) |
|-----------|-----------|-------------|-----------|----------------------|
| MGC | MGC1! | $10/pt | 0.1 | 10 ticks = 1.0 pts = $10 |
| MNQ | MNQ1! | $2/pt | 0.25 | 10 ticks = 2.5 pts = $5 |
| MES | MES1! | $5/pt | 0.25 | 10 ticks = 2.5 pts = $12.50 |
| M2K | M2K1! | $5/pt | 0.1 | 10 ticks = 1.0 pts = $5 |

---

## 2. Sessions — Times & DST

Sessions are institutional events. Times shift with DST (US and Europe).
Brisbane (UTC+10) has NO DST. ET shifts between UTC-5 (winter) and UTC-4 (summer).

### Session Schedule

| Session | Event | ET Time | Brisbane WINTER | Brisbane SUMMER |
|---------|-------|---------|-----------------|-----------------|
| CME_REOPEN | CME Globex electronic reopen | 18:00 (6 PM) | 09:00 next day | 08:00 next day |
| TOKYO_OPEN | Tokyo Stock Exchange open | 19:00 / 20:00 | 10:00 | 10:00 |
| SINGAPORE_OPEN | SGX/HKEX open | 20:00 / 21:00 | 11:00 | 11:00 |
| LONDON_METALS | London metals AM session | 03:00 | 18:00 | 17:00 |
| US_DATA_830 | US economic data release | 08:30 | 23:30 | 22:30 |
| NYSE_OPEN | NYSE cash equity open | 09:30 | 00:30 next day | 23:30 |
| US_DATA_1000 | ISM/Consumer Confidence | 10:00 | 01:00 | 00:00 |
| COMEX_SETTLE | COMEX gold settlement | 13:30 | 04:30 | 03:30 |
| CME_PRECLOSE | CME equity pre-settlement | 15:45 (3:45 PM) | 06:45 | 05:45 |
| NYSE_CLOSE | NYSE closing bell | 16:00 (4:00 PM) | 07:00 | 06:00 |

### DST Rules for Pine Script
- **US DST**: 2nd Sunday March → 1st Sunday November (ET goes UTC-4)
- **EU DST**: Last Sunday March → Last Sunday October (London goes UTC+0 → UTC+1)
- **Tokyo/Singapore/Brisbane**: No DST ever
- **Impact**: US sessions (US_DATA_830 through NYSE_CLOSE) shift 1 hour in Brisbane time between winter/summer. LONDON_METALS shifts with EU DST.
- **Pine implementation**: Use `timestamp()` with "America/New_York" timezone for US sessions, "Europe/London" for LONDON_METALS, "Asia/Tokyo" for TOKYO_OPEN, "Asia/Singapore" for SINGAPORE_OPEN.

---

## 3. ORB (Opening Range Breakout) Definition

The ORB is the high-low range of the first N minutes after a session opens.

### Apertures (ORB durations)
| Aperture | Meaning |
|----------|---------|
| O5 | High/Low of first 5 minutes after session open |
| O15 | High/Low of first 15 minutes after session open |
| O30 | High/Low of first 30 minutes after session open |

**Default ORB duration per session** (from pipeline, all are 5 min base — but strategies use 5/15/30):
The pipeline computes ORBs for ALL three apertures (5, 15, 30) for every session. Different strategies use different apertures.

### How to Draw the ORB
1. At session open time, start tracking high and low
2. After N minutes (5, 15, or 30 based on user selection), freeze the ORB high and ORB low
3. Draw horizontal lines at ORB_HIGH and ORB_LOW extending forward
4. ORB midpoint = (ORB_HIGH + ORB_LOW) / 2

### ORB Size
`orb_size = ORB_HIGH - ORB_LOW` (in points)

This is the key input for G-filters (see section 5).

---

## 4. Entry Models

Two active entry models. User should be able to select which one.

### E2 — Stop-Market Entry (PRIMARY)
- **Long**: Place buy-stop at ORB_HIGH
- **Short**: Place sell-stop at ORB_LOW
- **Fills when**: Price trades through the ORB boundary
- **This is the industry-standard honest entry** — no hindsight bias
- **Confirm bars (CB)**: Wait for CB bars to close beyond ORB before placing stop
  - CB1 = place stop after 1 bar closes beyond ORB (most common)
  - CB2-5 = wait for 2-5 bars

### E1 — Market After Confirm Bar
- **Long**: After CB bars close above ORB_HIGH, enter at market on next bar open
- **Short**: After CB bars close below ORB_LOW, enter at market on next bar open
- **More conservative** than E2 — waits for confirmation

### E0 — DEAD, do not implement
### E3 — RETIRED, do not implement

---

## 5. Filters (Go/No-Go for Today)

These determine whether today qualifies for a trade. Display as badges/labels.

### G-Filters (ORB Size Minimum)
Minimum ORB size in POINTS (not percentage) for a trade to qualify.

| Filter | Rule | Meaning |
|--------|------|---------|
| NO_FILTER | Always passes | Trade every day |
| ORB_G4 | orb_size >= 4.0 pts | Minimum 4 point ORB |
| ORB_G5 | orb_size >= 5.0 pts | Minimum 5 point ORB |
| ORB_G6 | orb_size >= 6.0 pts | Minimum 6 point ORB |
| ORB_G8 | orb_size >= 8.0 pts | Minimum 8 point ORB |

**Display**: After ORB forms, show which G-levels pass. Example: "G4+ G5+ G6+ G8-" means ORB is between 6 and 8 points.

**NOTE**: These thresholds are in raw points (e.g., 4.0 points on MNQ = $8 risk per contract). They are NOT percentages.

### VOL_RV12_N20 (Relative Volume Filter)
- **Rule**: Today's realized volume >= 1.2x the 20-day average volume
- **Calculation**: Compare current session's volume to the average of the same session's volume over the past 20 trading days
- **Pine approach**: Use `ta.sma(volume, 20)` on the session's volume, check if today's volume >= 1.2 * that average
- **Display**: "VOL 1.2x+" badge when passes, "LOW VOL" when fails

### Composite Filters (less common, nice-to-have)
| Filter | Rule |
|--------|------|
| ORB_G4_NOFRI | G4 + skip Fridays |
| ORB_G5_FAST10 | G5 + break must happen within 10 min of ORB end |
| ORB_G4_CONT | G4 + break bar must close in break direction |
| DIR_LONG | Only take long breakouts |
| DIR_SHORT | Only take short breakouts |

---

## 6. Trade Management

### Stop Loss
- **Standard**: Opposite side of ORB from entry
  - Long entry at ORB_HIGH → stop at ORB_LOW
  - Short entry at ORB_LOW → stop at ORB_HIGH
- **Risk = ORB size** (in points)
- **Tight stop (0.75x)**: Stop at entry - 0.75 * orb_size (tighter, less risk per trade)

### Take Profit (RR Targets)
RR = reward-to-risk ratio. TP distance = RR * orb_size.

| RR | Long TP | Short TP |
|----|---------|----------|
| 1.0 | entry + orb_size | entry - orb_size |
| 1.5 | entry + 1.5 * orb_size | entry - 1.5 * orb_size |
| 2.0 | entry + 2.0 * orb_size | entry - 2.0 * orb_size |
| 2.5 | entry + 2.5 * orb_size | entry - 2.5 * orb_size |
| 3.0 | entry + 3.0 * orb_size | entry - 3.0 * orb_size |
| 4.0 | entry + 4.0 * orb_size | entry - 4.0 * orb_size |

**Display**: Draw horizontal lines at each RR level. Color-code (green for TP, red for SL).

### Time Stop (Early Exit)
If the trade hasn't hit TP by this many minutes after entry, exit at market.

| Session | Time Stop (minutes) |
|---------|-------------------|
| CME_REOPEN | 38 |
| TOKYO_OPEN | 39 |
| SINGAPORE_OPEN | 31 |
| LONDON_METALS | 36 |
| US_DATA_830 | 49 |
| NYSE_OPEN | 59 |
| US_DATA_1000 | 45 |
| COMEX_SETTLE | 39 |
| CME_PRECLOSE | 16 |
| NYSE_CLOSE | 111 |

**Display**: Vertical line at the time-stop deadline.

---

## 7. What the Indicator Should Show

### ESSENTIAL (build these)
1. **ORB box** — shaded rectangle from session open to ORB end, between ORB_HIGH and ORB_LOW
2. **Session vertical lines** — at each session open time (DST-aware)
3. **Entry level** — dotted line at E2 stop price (= ORB boundary)
4. **RR target lines** — horizontal lines at each RR level (user-selectable which RR to show)
5. **Stop loss line** — red line at opposite ORB boundary
6. **G-filter badge** — label showing "G4/G5/G6/G8" pass/fail after ORB forms
7. **Time stop vertical** — line at the time-stop deadline

### NICE TO HAVE
8. **VOL_RV12_N20 badge** — "HIGH VOL" / "LOW VOL" label
9. **ORB size label** — show the ORB range in points inside or near the box
10. **Direction arrow** — arrow when price breaks ORB boundary (entry signal)
11. **ATR contraction warning** — when ATR is declining (compressed spring AVOID signal)
12. **Prior session result** — small label showing if previous session's ORB trade won/lost

### USER INPUTS (Pine input() fields)
- Session selector (dropdown: all 11 sessions)
- Aperture (5 / 15 / 30 minutes)
- Entry model (E1 / E2)
- Confirm bars (1-5)
- RR target to display (1.0 / 1.5 / 2.0 / 2.5 / 3.0 / 4.0)
- G-filter level (None / G4 / G5 / G6 / G8)
- Show time stop (yes/no)
- Stop multiplier (1.0 / 0.75)
- Timezone (Brisbane / ET / UTC)

---

## 8. Strategy Classification (for reference labels)

| Class | Sample Size | Label Color |
|-------|-----------|-------------|
| CORE | >= 100 trades | Green |
| REGIME | 30-99 trades | Yellow |
| INVALID | < 30 trades | Red / don't show |

---

## 9. Known NO-GOs (do NOT build indicators for these)

- **DOW filters**: Day-of-week effects are noise (0 BH FDR survivors). Don't add day-of-week filtering.
- **Calendar overlays**: NFP/OPEX/FOMC = noise. Don't add economic calendar filtering.
- **E0 entry model**: DEAD. Fill-on-touch bias. Don't implement.
- **double_break**: LOOK-AHEAD bias. Cannot be used as a real-time filter.
- **MCL/SIL/M6E/MBT**: No ORB edge on these instruments.

---

## 10. Example: What a Complete Trade Looks Like

**Setup**: MNQ NYSE_OPEN O15 E2 CB1 ORB_G8 RR1.5

1. **09:30 ET**: NYSE opens. Start tracking high/low.
2. **09:45 ET** (15 min later): ORB forms. High=20150, Low=20100. ORB size = 50 pts.
3. **Filter check**: 50 pts >= 8 (G8 passes). Show green "G8+" badge.
4. **Entry**: Place buy-stop at 20150.25 (ORB_HIGH + 1 tick) and sell-stop at 20099.75 (ORB_LOW - 1 tick).
5. **09:52 ET**: Price breaks above 20150. Buy-stop fills at ~20150.25.
6. **Stop loss**: 20100 (ORB_LOW). Risk = 50.25 pts = $100.50.
7. **Take profit**: 20150.25 + (1.5 * 50) = 20225.25. Reward = $150.
8. **Time stop**: 09:52 + 59 min = 10:51 ET. If price hasn't hit TP by then, exit at market.
9. **Result**: Price hits 20225.25 at 10:15 ET. Win +1.5R.
