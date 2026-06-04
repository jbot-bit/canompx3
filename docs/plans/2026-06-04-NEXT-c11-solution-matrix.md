# NEXT: C11 Solution Matrix + Reframing (read-only) — RESUME POINT

**Status 2026-06-04:** attribution DONE + a partial edge-vs-cap run DONE (banked below).
The full solution matrix + the 3 reframing analyses are NOT yet done. This file is
the one-shot resume point. **All work read-only. Do NOT edit/commit/launch/change
budget or live config until operator approves a fix.**

## Where to work (git awareness)
- Worktree: `C:\Users\joshd\canompx3-c11-attribution`, branch `c11-attribution-analysis`
  (off `origin/main` de1f9089). Main lease is FREE (operator force-released PID 11180).
- Worktree has NO `.venv` (per-worktree isolation not bootstrapped) and NO local
  `gold.db`. Run python via the **main venv** against the **shared canonical DB**:
  `cd C:/Users/joshd/canompx3-c11-attribution && C:/Users/joshd/canompx3/.venv/Scripts/python.exe <<'PYEOF' ... PYEOF`
  (the DB auto-resolves to `C:\Users\joshd\canompx3\gold.db` — see "[DB] INFO" line).
- Main HEAD has since moved to 888d1d46 (peer commits). Irrelevant to read-only analysis.

## Canonical method (do NOT re-encode the DD walk)
Monkeypatch `account_survival._load_lane_trade_paths` to inject a cap/stop override, then
call the REAL `_load_profile_daily_scenarios(pid)` + `_max_observed_rolling_drawdown(sc, horizon_days=STRICT_DD_HORIZON_DAYS)`.
This reuses the gate's own calendar-alignment/common-start logic — verified: it
reproduces the gate's $2,038.84 exactly. profile id = `topstep_50k_mnq_auto`.
Budget = `STRICT_DD_BUDGET_FRACTION (0.80)` × $2,000 MLL = **$1,600**.

```python
import trading_app.account_survival as A
ORIG = A._load_lane_trade_paths
def patch(cap=None, stop=None, only_sid=None):
    def p(con, sid, *, as_of_date, effective_stop_multiplier=None, max_orb_size_pts=None):
        # apply override only to target lane if only_sid set, else all lanes
        c = cap if (only_sid is None or sid==only_sid) else max_orb_size_pts
        s = stop if (only_sid is None or sid==only_sid) else effective_stop_multiplier
        return ORIG(con, sid, as_of_date=as_of_date, effective_stop_multiplier=s, max_orb_size_pts=c)
    return p
A._load_lane_trade_paths = patch(cap=..., stop=..., only_sid=...)   # reset to ORIG after
```

## BANKED RESULTS (authoritative, canonical)

### Strict 90d DD vs uniform cap (all lanes) — the cliff
| cap | DD | vs $1600 | vs $2000 |
|---|---|---|---|
| CURRENT(143/50/44) | **$2,038.84** | FAIL | FAIL |
| 100pt | $2,059.11 | FAIL | FAIL |
| 80pt | $2,067.71 | FAIL (WORSE) | FAIL |
| **60pt** | **$1,525.21** | **PASS** | PASS |
| 50pt | $902.28 | PASS | PASS |
| 45pt | $875.62 | PASS | PASS |
| 40pt | $743.37 | PASS | PASS |

**KEY: non-monotonic.** 80/100pt make DD WORSE than current (they strip large WINS,
lowering equity peaks → wider rolling DD). Clearance is a CLIFF at ≤60pt, not a gradient.
Memory's "80pt necessary but not sufficient" is WRONG — 80pt is counterproductive.

### Per-lane edge (current caps, $/trade as ExpR proxy, R-stats TODO)
| lane | cap | N | ExpR$ | WR | Sharpe* |
|---|---|---|---|---|---|
| COMEX_SETTLE OVNRNG_100 | 49.8 | 535 | 8.97 | 49.5% | 2.11 |
| **US_DATA_1000 VWAP_MID** | 143.2 | 795 | 23.71 | 50.8% | 2.36 |
| TOKYO_OPEN COST_LT08 | 44.2 | 460 | 11.71 | 51.7% | 2.83 |
*Sharpe here = mean/pstdev(pnl_dollars)×√252 — proxy, NOT the validator's Sharpe. Redo with pnl_r.

### **THE REFRAMING FINDING (highest EV — this changes the question)**
US_DATA_1000 lane edge vs cap — **the edge SHARPENS as you cap tighter, up to a point:**
| cap | N | ExpR$ | WR | Sharpe* | skipped |
|---|---|---|---|---|---|
| none | 819 | 20.17 | 50.4% | 1.72 | 0 |
| 143.2 | 795 | 23.71 | 50.8% | 2.36 | 24 |
| 100 | 724 | 23.86 | 51.4% | 2.71 | 95 |
| **80** | 631 | **26.74** | 52.8% | **3.43** | 188 |
| **70** | 561 | **26.99** | 53.5% | **3.78** | 258 |
| 60 | 466 | 18.44 | 51.7% | 2.92 | 353 |
| 55 | 414 | 17.51 | 51.9% | 2.97 | 405 |
| 50 | 361 | 17.68 | 52.6% | 3.24 | 458 |
| 45 | 303 | 17.70 | 53.5% | 3.55 | 516 |
| 40 | 244 | 13.53 | 52.5% | 3.04 | 575 |

**ExpR PEAKS at 70-80pt ($27, Sharpe 3.78), then DROPS at 60pt.** The big-risk trades
(>80pt) are net edge-DILUTIVE — capping them at 70-80 IMPROVES the lane. But the
DD-clearing cap (≤60) is PAST the edge peak — at 60pt ExpR falls to $18.

**THE TENSION:** the cap that clears C11 (≤60pt) is NOT the cap that maximizes edge
(70-80pt). 70-80pt sharpens edge but FAILS C11. This is the real decision.

## WHAT'S LEFT TO RUN (the operator's matrix) — ranked by EV

### Tier 1 (highest EV — run first)
1. **Recompute ExpR/WR/Sharpe in R-MULTIPLES not $** (pnl_r per trade), so it matches
   the validator. The $ proxy conflates position size. Need `pnl_r` from outcomes.
2. **US-only cap vs all-lane cap** at {70,60,55,50}: does capping ONLY US_DATA_1000
   clear C11 while leaving COMEX/TOKYO untouched? (smaller diff = safer). Use `only_sid=US`.
3. **The edge-vs-DD frontier table:** for each cap {80,70,60,55,50}, report BOTH
   book-level ExpR AND strict DD + C11 pass. Find the cap that clears C11 with max edge.

### Tier 2
4. **Stop multiplier** {0.75,0.60,0.50} × cap — does a tighter stop clear C11 at a
   HIGHER (edge-preserving) cap? `apply_tight_stop` already wired in `_load_lane_trade_paths`.
   NOTE: memory says 0.75 stop made DD WORSE ($2,038 at one point) — verify, don't trust.
5. **Remove/size-down US_DATA_1000 entirely:** 2-lane book DD + edge. Is US the problem
   lane or the best lane? (per-lane table says it's the highest ExpR$ lane — removing it
   may HURT, flag this.)
6. **Reduced size on US lane only** vs whole book (sizing, not cap).

### Tier 3 (reference only — do not tune against)
7. Topstep $50k @ official $2,000 (fraction=1.00) as reference. NOTE: $2,038 fails even
   at full $2,000 by ~$39, so the budget knob alone never clears — confirmed.
8. Topstep $100k / self-funded IF configured (`get_profile` other ids). Memory: $100k
   cap80 clears ~$630 headroom. Verify configured first.

## THE 3 REFRAMING QUESTIONS (operator asked — answer BEFORE recommending a fix)

**Q: are we asking the wrong question?**
We tested "what cap clears the DD gate." Misleading because: (a) DD is non-monotonic in
cap so "tighter=safer" is FALSE; (b) the cap that clears DD destroys edge (60pt: ExpR
$18 vs 70pt peak $27). Better questions:
1. **"What's the smallest change that clears C11 WITHOUT moving past the edge peak?"**
   → points to stop-tightening or sizing at a 70-80pt cap, not a 60pt cap.
2. **"Is the DD even driven by the cap, or by lane CORRELATION / a specific regime
   window?"** → maybe the fix is de-correlating lanes or a regime filter, not a cap.
   HIGHEST EV — untested, could clear C11 with ZERO edge loss.

**Where's the REAL edge?** US_DATA_1000 has the highest ExpR$ AND its edge sharpens
when you remove >80pt trades (the fat-risk trades are dilutive noise). The edge lives in
the **≤80pt risk band** of that lane. It's being misread as "the problem lane" when it's
actually the best lane carrying a dilutive tail. Removing it = throwing out the edge.

**3 alternative framings — were they tested?**
1. **Cap as EXECUTION filter (skip fat-risk entries live), edge unchanged** — TESTED
   (that's what max_orb_size_pts does). Clears at ≤60 but past edge peak.
2. **Stop-tightening to clear DD at an edge-preserving cap** — NOT tested (Tier 2.4).
3. **Lane de-correlation / regime-gating the drawdown window** — NOT tested, NOT ruled
   out. The $2,038 DD may be one bad regime window across correlated lanes, fixable
   without any cap. HIGHEST-EV unexplored path.

## Guardrails (operator-locked)
- Read-only. No edit/commit/launch/budget/live-config change until approval.
- Do NOT tune against 2026 holdout (all scenarios must confirm 2026 holdout untouched —
  `_load_profile_daily_scenarios` uses full history; verify the DD window isn't 2026-driven).
- Don't pick a param that barely clears C11. Prefer robust clearance + edge preserved.
- Flag any scenario where DD "improves" only by deleting winning variance (60pt vs 70pt
  is exactly this — 60pt clears DD by killing the $27-ExpR band).

## Output owed (when matrix done)
Verdict: best path / second-best fallback / no-go cases / exact evidence /
smallest implementation plan + tests. Then WAIT for approval.
Implementation type per scenario: config / code / account decision / no-go.
