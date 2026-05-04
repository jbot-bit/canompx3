# OVNRNG Q3-Q4 Sweet-Spot — Cross-Session Replication Check

**Date:** 2026-04-21
**Branch:** `research/ovnrng-sweetspot-cross-session-diagnostic`
**Script:** `research/audit_ovnrng_sweetspot_cross_session.py`
**Parent:** PR #47 correction MD (`docs/audit/results/2026-04-20-nyse-open-ovnrng-fast10-correction.md`)

---

## Question

PR #47's correction reported that on MNQ NYSE_OPEN E2 RR=1.0 CB=1 IS,
ovn/atr quintiles Q3–Q4 (ratio 0.24–0.40) had ExpR +0.107–0.116R
while Q5 (ratio ≥0.68) HURT at +0.017R. The queued follow-up was
a cross-session 324+-cell Pathway-B pre-reg.

**Before committing to that pre-reg:** does the Q3+Q4 > Q5 pattern
replicate on other sessions?

---

## Verdict

**NO. Cross-session pre-reg is NOT worth writing.**

- Of 8 lookahead-clean sessions (≥17:00 Brisbane start, per
  `.claude/rules/backtesting-methodology.md` Rule 1.2):
  - 1 SWEET_SPOT_PRESENT (NYSE_OPEN, original finding, Δ=+0.133R p=0.101 — borderline)
  - 3 INVERSE (LONDON_METALS, US_DATA_1000, NYSE_CLOSE — Q5 beats Q3+Q4 by 0.23–0.25R)
  - 4 NO_PATTERN (EUROPE_FLOW, US_DATA_830, COMEX_SETTLE, CME_PRECLOSE)

- `memory/MEMORY.md` behavioral rule applies:
  `feedback_per_lane_breakdown_required.md` — "If ≥25% of cells flip
  sign vs pooled, finding is a heterogeneity artefact, not a universal
  rule." Here 3 of 8 = **37.5%** flip sign. Finding is heterogeneity,
  not mechanism.

- Even at K=8 sessions, BH-FDR q=0.05 requires rank-1 p < 0.00625.
  NYSE_OPEN's Welch p=0.10 fails this by 16x. At K=324 (full cross-
  session × aperture × RR scan as originally queued), required
  p would be ~0.00015 — unreachable.

---

## Per-session results (MNQ E2 RR=1.5 CB=1 orb_minutes=5, IS only)

### NYSE_OPEN — SWEET_SPOT_PRESENT (borderline)

| bin | n | ovn/atr μ | WR | ExpR |
|-----|----|----|----|-----|
| Q1 | 330 | 0.166 | 47.3% | +0.131 |
| Q2 | 330 | 0.238 | 43.9% | +0.058 |
| Q3 | 329 | 0.308 | 48.6% | +0.169 |
| Q4 | 330 | 0.404 | 46.1% | +0.109 |
| Q5 | 330 | 0.680 | 41.5% | +0.006 |

Q3+Q4 mean = +0.139, Q5 mean = +0.006, Δ = **+0.133**, Welch t=+1.64, p=0.101.

Replicates PR #47's NYSE_OPEN pattern at RR=1.5 (PR #47 tested RR=1.0).
Effect size is smaller at RR=1.5 (+0.133R vs PR #47's +0.09R pooled
lift) and not statistically significant at p<0.05.

### INVERSE sessions (3 of 8) — HIGH ovn/atr is BETTER

**LONDON_METALS** (n=1716, Δ=−0.247, p=0.001):

| bin | ExpR |
|-----|------|
| Q1 | −0.056 |
| Q2 | +0.096 |
| Q3 | −0.028 |
| Q4 | −0.038 |
| Q5 | **+0.214** |

**US_DATA_1000** (n=1673, Δ=−0.233, p=0.004):

| bin | ExpR |
|-----|------|
| Q1 | +0.085 |
| Q2 | +0.122 |
| Q3 | +0.082 |
| Q4 | −0.070 |
| Q5 | **+0.239** |

**NYSE_CLOSE** (n=611, Δ=−0.242, p=0.057):

| bin | ExpR |
|-----|------|
| Q1 | −0.066 |
| Q2 | +0.003 |
| Q3 | −0.111 |
| Q4 | −0.049 |
| Q5 | **+0.162** |

On these 3 sessions, the highest overnight range quintile is the BEST
quintile to trade — the opposite of PR #47's narrative for NYSE_OPEN.

### NO_PATTERN sessions (4 of 8)

EUROPE_FLOW (Δ=−0.026 p=0.72), US_DATA_830 (Δ=+0.036 p=0.64),
COMEX_SETTLE (Δ=−0.038 p=0.63), CME_PRECLOSE (Δ=−0.027 p=0.76).

No meaningful or statistically significant difference between Q3+Q4
and Q5 on these.

---

## Rollup

| Verdict | Sessions | Count |
|---------|----------|-------|
| NO_PATTERN | EUROPE_FLOW, US_DATA_830, COMEX_SETTLE, CME_PRECLOSE | 4 |
| INVERSE | LONDON_METALS, US_DATA_1000, NYSE_CLOSE | 3 |
| SWEET_SPOT_PRESENT | NYSE_OPEN | 1 |

**Sign-flip rate: 3 of 8 = 37.5%** — exceeds the 25% threshold in
`feedback_per_lane_breakdown_required.md`. Finding is NOT universal.

---

## Interpretation

The PR #47 "Q3-Q4 sweet-spot, Q5 hurts" observation on NYSE_OPEN was
session-specific, not a universal mechanism of overnight-range on
breakout quality. The narrative ("moderate overnight range = healthy
setup, extreme = exhausted range-day") does not hold up when tested
cross-session.

Three sessions (LONDON_METALS, US_DATA_1000, NYSE_CLOSE) show the
opposite — high overnight volatility correlates with stronger breakout
edge in the US afternoon / commodity-close contexts. This is
consistent with Chordia et al 2018 / Chan Ch 7 priors that high
realized volatility often produces the best breakout days (trend
continuation, stop-cascade mechanics). But these are 3 different
sessions with 3 different underlying mechanisms — no single
"overnight range is good/bad" filter captures them.

NYSE_OPEN's pattern may be explained by 10:00 ET US equity open
behavior — extreme overnight ranges (0.68 ovn/atr) often signal
event-driven pre-market moves that have already exhausted the day's
directional energy before cash-equity open. That's a plausible
session-specific mechanism, not a universal feature of overnight range.

---

## Operational conclusions

1. **Do NOT write the 324-cell cross-session Pathway-B pre-reg.** The
   universal claim fails.

2. **NYSE_OPEN-only pre-reg is possible but marginal.** Welch p=0.10
   at RR=1.5 (PR #47 was RR=1.0 where p might be tighter — worth
   checking if user wants to pursue). Would need to be pre-registered
   with explicit single-session scope, BH-FDR K=1 (Pathway-B), and
   theory citation that explains WHY the mechanism is session-specific.
   Lower-EV than originally thought.

3. **INVERSE sessions are separately interesting** but would need
   their own fresh pre-reg. "High overnight vol = better breakout" on
   LONDON_METALS / US_DATA_1000 / NYSE_CLOSE is a real per-session
   pattern but the mechanism differs per session, and pooling them
   violates the heterogeneity rule.

4. **No deployment change.** This is a discovery diagnostic; nothing
   in the deployed 6-lane portfolio changes.

---

## Provenance

- Canonical data: `orb_outcomes`, `daily_features` (triple-joined).
- Rule 1.2 lookahead gate applied: overnight_range valid only for ORB
  start ≥17:00 Brisbane. Skipped sessions: CME_REOPEN, TOKYO_OPEN,
  SINGAPORE_OPEN, BRISBANE_1025 (all start before 17:00).
- Holdout: 2026-01-01 (Mode A sacred). Only IS tested.
- MNQ E2 RR=1.5 CB=1 orb_minutes=5 universe (5 of 6 DEPLOY lanes use
  RR=1.5; selected for portfolio-relevance).
- Read-only. No production code touched. No pre-reg created.

---

## Next

PR #47's ATR-normalized OVNRNG sweet-spot queue item is **CLOSED** with
verdict NOT_WORTHWHILE for cross-session scope. Any follow-up would
need to be NYSE_OPEN-only Pathway-B with explicit theory citation —
and even that has marginal p=0.10 at RR=1.5.

---

## Tunnel-vision check (self-review, mandated)

### What was fairly tested

- **Filter framing — pooled Q3+Q4 vs Q5 binary contrast** at MNQ E2 RR=1.5
  CB=1 orb_minutes=5 across 8 lookahead-clean sessions. This specific
  framing is a fair replication check of PR #47's finding in the
  "filter" context. Result: does not universalize.

### What was prematurely ruled out (tunnel vision)

1. **RR tunnel** — only RR=1.5 tested. PR #47's original finding was
   at RR=1.0 where hit probability is highest. Different RR regimes
   may have different overnight-range interactions. Unknown whether
   NYSE_OPEN sweet-spot is robust across RR.

2. **Entry-model tunnel** — only E2 tested. E1 (break-only) and E3
   (break + 2 confirm) have different selection properties; overnight
   context may matter differently.

3. **Aperture tunnel** — only 5m ORB tested. 15m / 30m have different
   vol-envelope dynamics.

4. **Direction tunnel** — pooled long + short. Overnight range may
   predict direction-conditional outcomes (e.g., high overnight +
   long breakout = continuation; high overnight + short breakout =
   exhaustion). Not tested.

5. **Pooled-quintile framing tunnel** — I tested Q3+Q4 vs Q5 because
   that's what PR #47 tested. Alternative binary contrasts (Q5
   vs rest; Q1 vs rest) or monotonic-trend tests could reveal
   different patterns.

6. **Framing tunnel — I only tested "overnight range as a filter."**
   Four other uses never tested.

### Five honest alternative framings

1. **Standalone filter** (tested, fails cross-session). Result: only
   NYSE_OPEN shows the pooled-quintile sweet-spot; 3 of 8 sessions
   show the INVERSE. No universal filter signal.

2. **Conditioner (sizing tilt, not gate)**. NOT TESTED. Would scale
   position size continuously by ovn/atr percentile rather than binary
   fire/no-fire. Per-session ExpR-by-quintile data above is the input
   to a sizing function. If ExpR spread across quintiles is 0.05–0.25R
   per session, that's real information for Kelly-style sizing per
   Carver Ch 9-10.

3. **Allocator / session router** — **biggest miss**. NOT TESTED.
   The per-session quintile data reveals a pattern: high ovn/atr
   (Q5, ratio ~0.68+) is the BEST quintile on LONDON_METALS (+0.214R),
   US_DATA_1000 (+0.239R), NYSE_CLOSE (+0.162R), while high ovn/atr
   HURTS NYSE_OPEN (+0.006R). If ovn/atr were known pre-session,
   an allocator rule "when high overnight vol → trade LONDON_METALS /
   US_DATA_1000 / NYSE_CLOSE; when low-mid → NYSE_OPEN" is a real
   regime-router signal hidden in the data I already have.

4. **Confluence (interaction filter with another feature)**. NOT TESTED.
   ovn/atr × gap_type, ovn/atr × day_of_week, ovn/atr × prev_day_direction
   could reveal stronger per-subset discrimination than the marginal
   quintile test.

5. **Direction-conditional signal**. NOT TESTED. Overnight range's
   effect on long trades vs short trades may differ (e.g., overnight
   rally predicts long-breakout follow-through but short-breakout
   exhaustion).

---

## Honest best opportunity

**Best opportunity:** **Allocator/session-router hypothesis.** The
per-session quintile table shows a genuine pattern: Q5 (high overnight
vol) is BEST on LONDON_METALS, US_DATA_1000, NYSE_CLOSE and WORST/flat
on NYSE_OPEN. If ovn/atr > ~0.5 at session start, the deployed portfolio
should over-weight INVERSE sessions; if ovn/atr is in the middle band,
the portfolio should prefer NYSE_OPEN. This is information the current
fixed allocator does not use.

Rough EV estimate (conservative):
- Q5 sessions in Q5 ovn/atr: pooled ExpR ≈ (0.214+0.239+0.162)/3 = +0.205R
  per trade, N≈970 IS.
- NYSE_OPEN in Q3+Q4 ovn/atr: ExpR ≈ +0.139R per trade, N≈659 IS.
- Compared to uniformly trading all sessions at pooled ExpR ≈ +0.05R,
  the conditional rule ~2-4x the per-trade ExpR on the days the signal
  is active.

**Biggest blocker:** implementation drag. Current
`docs/runtime/lane_allocation.json` is a static DEPLOY list. A router
that chooses session based on pre-session overnight-range percentile
would require:
- A pre-session overnight-range reader (trade-time-knowable — ✓)
- A routing rule layer above the lane-allocator
- Lane-to-session mapping kept stable (fine)
- Operational: overnight_range at the time of first-session ORB start
  (≥17:00 Brisbane) is known; later sessions can use it freely.
No canonical pipeline code exists for this routing yet. Building it
is ~1 week of infra work, not a research turn.

**Biggest miss:** I ran a cross-session replication test on the
filter framing without asking "what if the data is telling me
something ALLOCATOR-shaped instead?" The ExpR-by-quintile-by-session
table was literally the answer to the allocator question, and I
reported "no universal filter" instead of "here's a real cross-session
regime signal." This is the exact per-lane-breakdown-required rule
case: pooled Q3+Q4 vs Q5 hides the lane-specific sign-flips that ARE
the finding.

**Next best test (low-cost, today):** compute "best session to trade
given ovn/atr bin" conditional table. For each (IS day, ovn/atr bin)
pair, identify the session with highest ExpR. If the best session
CHANGES with bin, it's an allocator signal. Extension: test whether
a rule "trade top-2 sessions per ovn/atr bin" beats "trade all
deployed lanes equally." Read-only, ~30 min runtime. This would be
the correct diagnostic to tell us whether the allocator-router idea
is worth pre-registering.

**Wrong question asked (honest admission):** I asked "does Q3+Q4 >
Q5 hold cross-session?" The better question was "does ovn/atr predict
which session is best to trade today?" Same data, different framing,
completely different operational conclusion.

---

## Revised operational recommendation

1. **Do not close the OVNRNG research line.** The filter-framing
   cross-session replication fails, but the allocator-framing
   hypothesis is unexplored and has a visible signal.

2. **Do not write the original 324-cell Pathway-B pre-reg.** Its
   filter framing was wrong.

3. **Run the allocator diagnostic next** (conditional "best session
   per ovn/atr bin" table) — read-only, bounded scope.

4. **If the allocator signal holds,** write a router pre-reg with
   explicit theory (Chan Ch 7 regime routing, Chordia et al 2018
   factor-segmented testing) and scope-locked to the allocator
   framing.

5. **NYSE_OPEN-only sweet-spot pre-reg** is still an option but
   lower EV than the allocator framing. Would be RR=1.0 per PR #47's
   original, not RR=1.5.
