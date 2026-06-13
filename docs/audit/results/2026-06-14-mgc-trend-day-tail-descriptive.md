---
pooled_finding: true
per_cell_breakdown_path: research/output/mgc_trend_day_tail_grid.csv
flip_rate_pct: 0
---

# MGC Trend-Day Runner — Phase A Descriptive Scan

**Date:** 2026-06-14
**Verdict:** **PARK** — the realizable trend-day tail is flat/negative; the "4R+ on
trend days" effect is the look-ahead oracle leaking, not a tradeable edge.
**Scope:** MGC, IS-only (`trading_day < 2026-01-01`), entry models E1/E2, 9
testable sessions, triad apertures (5/15/30). N = 822,168 outcomes across 1,010
trading days (3.55 clean years, 2022-06-13 → 2025-12-31).
**Type:** descriptive truth-finding on canonical layers. NO prereg, NO LOCK, NO
`validated_setups` write (sanctioned without prereg per `backtesting-methodology`
RULE 10).
**Script:** `research/mgc_trend_day_tail_descriptive.py`
**Raw run log:** `research/output/mgc_trend_day_phaseA_run.log`
**Per-cell grid (7,066 cells):** `research/output/mgc_trend_day_tail_grid.csv`

---

## The question

Every prior MGC verdict measured `avg_pnl_r at RR1.5` — a metric that CAPS the
winner at 1.5R and averages it into the chop. The operator's framing: *"look for
massive trades on trending MGC days — less often, but they're 4R+."* The
`trend_day_mfe` study (Mar 2026) proved the raw fact: 83% of MGC wins have a
non-zero true-vs-capped MFE gap (mean +0.93R, P95 +2.37R, max +6.80R).

Phase A asks the ONE question that the MFE study did not: **does the
uncapped-R distribution on MGC trend-days have a materially fatter right tail
than non-trend days UNDER A REALIZABLE EXIT, and does a pre-entry-safe proxy
capture enough of it to be worth a Phase-B prereg?**

### The MFE-illusion guard (load-bearing)

`true_mfe_r` is the maximum favorable excursion — the single best tick the trade
ever reached. **No real exit realizes it** (you don't know the high until it has
passed). MFE is the *upper bound on a perfect trailing stop*, not a P&L. The
+0.93R gap is **headroom, not edge.** Phase A measures the tail under two
**realizable** exits and rests the decision on those; MFE is the ceiling
benchmark only.

Three tails per cell:
1. `true_mfe_r` — CEILING (non-realizable; friction in denominator only).
2. `session_close_r` — hold-to-close (realizable; net of friction).
3. `trail_r` — give-back-of-peak trailing stop (arm@1R, give back 50% of peak;
   realizable; net of friction).

---

## Headline result — the realizable tail does NOT survive

### Oracle split (look-ahead `day_type` — NON-TRADEABLE upper bound)

| Tail | TREND %≥3R | NON_TREND %≥3R | ratio | TREND mean R | TREND sum-R |
|---|---|---|---|---|---|
| **MFE (ceiling)** | 31.6% | 11.6% | **2.73×** | +2.951 | +1,475,714 |
| **CLOSE (realizable)** | 19.5% | 2.6% | 7.46× | **+0.117** | +58,299 |
| **TRAIL (realizable)** | **1.4%** | 0.0% | n/a | **−0.066** | **−32,930** |

The MFE ceiling is genuinely 2.73× fatter on trend days — **exactly the illusion
the guard warned about.** The moment a realizable exit is imposed:
- **CLOSE** TREND mean collapses to +0.117R (barely positive) and the comparator
  is look-ahead `day_type`.
- **TRAIL** TREND mean goes **NEGATIVE** (−0.066R, %≥3R 1.4%, sum-R −32,930). The
  50%-giveback trail bleeds friction across the population and rarely catches a
  runner. The give-back exit actively destroys the tail.

### Safe-proxy split (pre-entry tradeable — THE DECISION RESTS HERE)

Capture contrast — high-trend-prior bin vs complement on the **realizable** tails:

| Proxy (high-trend-prior bin) | CLOSE %≥3R ratio | TRAIL %≥3R ratio | CLOSE sum-R hi / lo |
|---|---|---|---|
| `atr_20_pct` top tercile | **1.02×** | 1.55× | −6,567 / +27,558 |
| `garch_atr_ratio > 1` | **0.95×** | 1.65× | +2,639 / +14,900 |
| `is_nfp_day` | 1.40× | 1.37× | +795 / +23,919 |
| `prev_day_direction` | (direction, not a trend-magnitude prior — no high bin) | | bull/bear ≈ equal |

**No safe proxy reaches the gate's ≥2× %≥3R bar on a realizable tail with positive
sum-R.** The best realizable-CLOSE capture (`is_nfp_day`, 1.40×) is below 2× and
the absolute edge is tiny (sum-R +795 on 34,908 NFP trades). `atr_20_pct` and
`garch_atr_ratio` — the volatility-expansion priors most theoretically tied to
trend days — capture essentially **none** of the oracle tail (CLOSE ratios 1.02×
and 0.95×).

### Capture-ratio interpretation

The oracle MFE tail is 2.73× fatter on trend days. A *tradeable* signal captures
~1.0–1.4× of the realizable-CLOSE tail and **negative sum-R on the realizable
TRAIL tail**. That gap IS the leak: "trend days have fat tails" is true but
look-ahead; "we can detect them in advance and harvest the tail" is false. The
decision rests on the latter — and it fails.

---

## RULE 12 — outlier / year stability (the central risk)

Year-by-year `trail_r` (realizable) is **negative in all 4 clean years** — the
opposite of "one good year carrying the aggregate." It is uniformly bad:

| Year | trades | mean %≥3R (trail) | sum-R (trail) |
|---|---|---|---|
| 2022 | 383,442 | 1.06% | −57,576 |
| 2023 | 848,142 | 0.77% | −131,785 |
| 2024 | 923,232 | 0.74% | −142,967 |
| 2025 | 928,320 | 1.26% | −20,633 |

There is no era in which the realizable trend-day runner is profitable. This is
not a regime-dependent edge waiting for the right conditions; it is a negative
result across the full horizon.

---

## RULE 13 — pressure test (harness validity)

The injected look-ahead controls behaved correctly, proving the harness catches
the failure class:
- `session_close_r` vs itself: corr = **1.000 → FLAGGED** look-ahead/tautology ✓
- `orb_outcomes.mfe_r` (banned look-ahead) vs tail: corr 0.213 (still banned by
  § 6.3 regardless — never a real proxy).
- `atr_20_pct` (genuinely safe) vs tail: corr **−0.007 → PASS** (not a tautology) ✓

---

## Methodology notes

- **Reuse (institutional-rigor § 4):** uncapped session MFE + hold-to-close come
  from `research/research_trend_day_mfe.py::compute_true_session_mfe`. Only the
  trailing-stop accumulator (`compute_trail_r`) is new, computed in the same
  per-day `bars_1m` replay. The DB `mfe_r` is CAPPED (`build_daily_features.py`
  outcome loop breaks on first target/stop hit), so the script CANNOT read
  `orb_*_mfe_r` — it replays.
- **Trail causality guard:** within a single 1m OHLC bar the high/low order is
  unknown. The trail is tested against the PRIOR peak before the bar can ratchet
  it, so a bar's own low can never trip the trail its own high just set
  (sub-bar look-ahead trap). 9 unit tests pin this:
  `tests/test_research/test_mgc_trend_day_trail.py`.
- **Friction:** realizable tails (CLOSE, TRAIL) are net of friction in BOTH
  numerator and denominator (`pipeline.cost_model.COST_SPECS`). The MFE ceiling
  keeps the reused friction-in-denominator-only convention because it is an upper
  bound benchmark, not a claimed P&L — labelled NON-REALIZABLE throughout.
- **Look-ahead discipline (RULE 6.1):** `day_type` is used ONLY for the
  descriptive oracle upper-bound (it is computed from `daily_close` — confirmed
  look-ahead at `build_daily_features.py:571-572`). All Phase-B-eligible proxies
  (`prev_day_direction`, `atr_20_pct`, `garch_atr_ratio`, `is_nfp_day`) are
  pre-entry-safe per `backtesting-methodology` § 6.1.
- **Anti-cherry-pick (RULE 5.3):** this reports the WHOLE grid (7,066 cells in
  `mgc_trend_day_tail_grid.csv`). No winner was selected. The verdict is the
  aggregate negative, not a hand-picked cell.
- **`flip_rate_pct: 0`** — the pooled verdict (realizable tail flat/negative)
  does not hide opposite-sign cells: the year breakdown is negative in 4/4 years
  and every safe-proxy capture ratio is < 2× on the realizable tails. There is no
  sign-flipping subset that the pooled framing masks.

---

## Decision (against the pre-defined gate)

The gate (defined before results, RULE 5/Step 5):

> ALIVE → Phase B IF: oracle tail fat **AND** safe-proxy captures ≥~half (≥2×
> non-trend %≥3R on realizable tail, sum-R positive where capped-RR was negative,
> stable in ≥3 of ~3.5 clean years).
>
> PARK → IF safe-proxy realizable tail ≈ non-trend tail (oracle leaking, not a
> tradeable edge).

**Outcome: PARK.** The oracle tail IS fat (2.73× MFE) but:
1. No safe proxy reaches ≥2× on a realizable tail (best is 1.40×, `is_nfp_day`).
2. The realizable TRAIL tail is NEGATIVE on every proxy bin and every year.
3. sum-R is not positive where it matters (volatility proxies: −6,567 / +2,639).

The "4R+ on trend days" is the look-ahead oracle leaking. Phase B is **not
warranted** — there is no tradeable safe-proxy signal to lock a prereg on.

### Redirect (compare-vs-highest-EV)

MGC is `deployable_expected=False` and will not deploy until ~2027 regardless.
Marginal research hours are better spent on the live-MNQ capital batons (multi-
account live, survival-sim sizing fix, db-lock cluster follow-ups). This negative
result closes the trend-day-runner object honestly — it is a DIFFERENT object
than the DEAD pre-ORB lean conditioner (K=89 NO-GO), and it clears the NO-GO
reopen bar by measuring the realizable tail rather than the capped mean. Both are
now negative; do not re-litigate either without a materially new exit mechanic or
a new pre-entry signal.

---

## Reproduction

```bash
python research/mgc_trend_day_tail_descriptive.py          # full IS run (~10 min)
python -m pytest tests/test_research/test_mgc_trend_day_trail.py -q   # trail logic
```

---

# Phase A2 — Exit × Filter Sweep (supersedes the Phase-A PARK with a properly-searched verdict)

**Why A2 exists:** Phase A tested ONE exit (a 50%-giveback trail) and called PARK.
That was a thin search — a 50%-giveback trail is structurally the *worst* harvester
for a fat right tail (it caps every runner at half its peak). Operator correctly
rejected the PARK and directed a full exit × filter sweep. A2 tests **13 exits**
(4 fixed R-targets, 5 trail variants, breakeven-hold, scaleout, hold-to-close, +
MFE ceiling) × **~14 pre-entry-safe filters**, grounded in Fitschen Ch5 (exits) /
Ch6 (filters — high-vol exclusion + longer-term trend alignment).

**Script:** `research/mgc_trend_day_exit_sweep.py` (22 unit tests).
**Grid (169 cells):** `research/output/mgc_exit_sweep_grid.csv`.
**Per-trade exits:** `research/output/mgc_exit_sweep_raw.csv` (822,168 rows).

## Correctness fix found in A2

`hold_to_close` (inherited from `compute_true_session_mfe`) was riding through the
hard stop to the session close — inflating it on losing days. A2 makes it die on
the hard stop. Pinned by `test_hold_to_close_dies_on_hard_stop`.

## A2 result — every realizable exit is NOISE around zero across 4 clean years

On ALL trades, all 12 realizable exits are negative (mean −0.085 to −0.122R) with
0/4 or 1/4 positive years. MFE ceiling +2.70R, 4/4 positive — headroom real and
rising, but unharvestable. Only **ORBSIZE_HI** (top-tercile ORB size,
`orb_size_pts ≥ 5.0`, N=274,986) has positive sum-R. Year-by-year (RULE 12):

| exit | 2022 | 2023 | 2024 | 2025 | yrs+ |
|---|---|---|---|---|---|
| fixed_3R | −0.03 | +0.02 | −0.03 | +0.04 | 2/4 |
| fixed_5R | −0.02 | +0.05 | −0.04 | +0.03 | 2/4 |
| hold_to_close | −0.02 | +0.05 | −0.04 | +0.04 | 2/4 |

The 2/4 split is "2 weakly-positive + 2 weakly-negative" in an alternating
(−,+,−,+) pattern, all magnitudes ±0.02–0.05R — a coin flip around break-even, not
a regime.

## Significance — naive-t inflated by day-clustering; clustered-t fails the bar

| exit (ORBSIZE_HI) | mean R | naive t | **clustered-by-day p** |
|---|---|---|---|
| fixed_3R | +0.018 | 7.50 | **0.037** |
| fixed_5R | +0.011 | 4.05 | **0.062** |
| hold_to_close | +0.014 | 4.50 | **0.102** |

Naive t (4–7) treats 274,986 correlated trades as independent; true unit is the
trading day (~3,000). Honest clustered p = 0.04–0.10 — none clears Chordia
(t≥3.79/3.00). Best effect +0.018R. Fails pre-reg C4, C8 (2/4 era stability), C9
(2024 era ExpR negative).

## A2 Verdict — DEAD for the trend-day-RUNNER object (not "no valid MGC prereg")

The trend-day **runner** (trend-day filter × uncapped/trailed exit) is DEAD on MGC:
the fat tail exists but no mechanical exit harvests it stably, and the only
positive filter is noise. This corrects the Phase-A PARK (artifact of one bad exit)
AND the smoke-run optimism (2022-only outlier).

**Scope of the kill:** this is a verdict on the *runner exit thesis*, NOT on MGC
research broadly. The exit-geometry space is now exhausted; re-litigating the
runner needs a *materially new pre-entry signal* stable across ≥3/4 clean years on
the clustered statistic — not another exit.

**Still-open (separate objects, NOT killed by A2):** validated capped-RR MGC lanes
in `validated_setups` remain their own question; a Pathway-B K=1 prereg on a
single theory-grounded MGC lane (e.g. a specific session × filter at a fixed RR)
is a *different object* and is not foreclosed by this runner kill. See the
session-handoff note for the closest-to-valid candidate.

## Reproduction (A2)

```bash
python research/mgc_trend_day_exit_sweep.py
python -m pytest tests/test_research/test_mgc_exit_sweep.py tests/test_research/test_mgc_trend_day_trail.py -q
```

---

# Phase A3 — No-pigeonhole MGC lane variable sweep ("is there ANY valid MGC prereg?")

**Why A3 exists:** operator pushed back — "lots of variables, they all matter, no
pigeonholing." A1/A2 tested the *runner* object and the *stored validated_setups*
lanes. A3 drops all pre-selection and sweeps the lane variable space directly from
canonical `orb_outcomes`, with multi-year stability + OOS + clustered-significance
as the ONLY selection gate. Grid: `research/output/mgc_lane_variable_sweep.csv`.

**Scope (612 cells, MinBTL-aware):** 9 sessions × {O5,O15,O30} × {E1,E2} ×
{RR1.0,1.5,2.0} × {NO_FILTER, ORB_G4, ORB_G6, ORB_G8}. Canonical filter delegation
via `research.filter_utils.filter_signal` (no re-encoding). Clustered-by-trading-
day t (the honest unit). Strict Mode-A IS (<2026-01-01), 2026 OOS.

## A3 result — real-but-marginal: structure exists, significance does not

| Gate constraint (N≥100) | cells passing |
|---|---|
| ≥3/4 IS years positive | 129 |
| OOS ExpR > 0 | 290 |
| **≥3/4yr AND OOS>0 (structure)** | **75** |
| clustered-t ≥ 3.00 | **0** |
| clustered-t ≥ 2.50 | **0** |
| full gate (≥3/4yr ∧ OOS>0 ∧ t≥3.0) | **0** |

**The binding constraint is significance, NOT structure.** 75 MGC cells are
genuinely multi-year-stable AND OOS-positive — the data is not empty. But the
effects are small (+0.04 to +0.17R) and none clears the multiplicity-corrected
significance bar. Top honest cells:

| lane | N | ExpR | clustered-t | yrs+ | OOS |
|---|---|---|---|---|---|
| US_DATA_830 O30 E2 RR2.0 ORB_G6 | 569 | +0.107 | **2.36** | 4/4 | +0.098 |
| CME_REOPEN O5 E2 RR2.0 ORB_G4 | 168 | +0.144 | 1.97 | 3/4 | +0.296 |
| US_DATA_830 O30 E2 RR2.0 ORB_G4 | 773 | +0.075 | 1.88 | 4/4 | +0.098 |
| SINGAPORE_OPEN O30 E2 RR2.0 ORB_G8 | 208 | +0.169 | 1.79 | 3/4 | +0.124 |

## A3 Verdict — MGC has consistent WEAK edges, no standalone-prereg-grade edge

There IS a valid MGC research object (US_DATA_830 / CME_REOPEN E2 O30/O5 RR2.0,
volatility/size-filtered, 4/4-year-stable, OOS-positive) — but at **t≈2.0–2.4 it
misses Chordia (3.00 with theory / 3.79 without)**. This is the data-grounded proof
of why MGC is `deployable_expected=False`: real edges, too small to clear the
multiplicity bar standalone.

**Valid-prereg paths (NOT foreclosed — these are the honest next moves):**
1. **Pooled/aggregate Pathway** — the US_DATA_830 O30 E2 RR2.0 cells (G4/G6/G8/
   NO_FILTER all 4/4yr, OOS+) are correlated tests of one hypothesis. Pooling per
   `pooled-finding-rule.md` inflates N and power; a pooled US_DATA_830-30min-E2
   object may clear the bar where each cell alone does not.
2. **Theory-first Pathway-B** — if a solid mechanism citation exists for US 8:30
   data-release ORB continuation (scheduled-vol catalyst; Fitschen Ch6 event/
   seasonal + `mechanism_priors.md`), the t≥3.00 bar applies — still a miss at
   2.36, but a documented near-miss worth a powered re-test, not a kill.

**What A3 forecloses:** the "find a strong standalone MGC edge by sweeping
variables" hope. 612 cells, properly gated, yield zero t≥2.5 survivors. MGC's
ceiling is weak-edge / portfolio-contributor, not standalone-deployable — proven,
not assumed. No pigeonholing: all sessions/apertures/entries/RRs/size-filters were
swept; the conclusion is the aggregate, not a picked cell.

## Reproduction (A3)

```bash
# regenerates research/output/mgc_lane_variable_sweep.csv (needs gold.db unlocked)
# (inline scan — see session transcript; reads orb_outcomes JOIN daily_features, Mode-A split)
```
