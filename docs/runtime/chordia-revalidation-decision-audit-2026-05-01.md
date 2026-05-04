# Chordia revalidation — three-decision audit (2026-05-01)

**Status:** durable audit doc per save-as-you-go. Grounded in actual code/files; no
made-up references. Each finding cites file:line.

---

## Decision 1 — Doctrine action on 4 FAIL_BOTH lanes

### Question
Is "research-provisional + signal-only continuation" actually protective of capital?

### Audit findings

**1.1 Profile state verified** (`trading_app/prop_profiles.py`)
- `topstep_50k_mnq_auto`: firm=topstep, is_express_funded=True, account_size=50000
- F-1 XFA Scaling Plan IS active for this profile.

**1.2 Signal-only F-1 wiring verified** (`trading_app/live/session_orchestrator.py:237-266`)
- `_apply_signal_only_f1_seed` seeds EOD balance with $0.00 in signal-only mode.
- Verbatim docstring: "NOT a bypass: F-1 still enforces the cap; the seed just lets
  `can_enter` evaluate the ladder against a known balance."
- Result: F-1 enforces position-size cap (2 lots for 50K bottom tier) even in
  signal-only.

**1.3 B6 wiring gap historical context**
- Original (pre-fix) state: signal-only refused EVERY entry with "EOD XFA balance
  unknown" — last real `SIGNAL_ENTRY` was 2026-04-06.
- B6 fix landed: `_apply_signal_only_f1_seed` is the canonical day-1 default seed.
- Current state: signal-only sessions can fire entries; F-1 caps to bottom-tier
  position size; trades go to `live_signals.jsonl` for OOS evidence accumulation.

**1.4 Memory-file caveat** (`memory/f1_xfa_active_correction.md`)
- File is referenced in `MEMORY.md` index and project handoff plan.
- File does NOT exist on disk in current worktree's `memory/` dir.
- Original content recovered from session log `17541b2a-...jsonl:614` (created
  2026-04-24) — describes B6 wiring gap that's since been fixed.
- Reference to this file in handoff/MEMORY.md is stale.

### Verdict

**APPROVE doctrine action** — research-provisional + signal-only continuation is
correctly protective. No real broker connection in signal-only → no actual capital
at risk → live OOS evidence keeps accumulating in `live_signals.jsonl`.

**No blast radius beyond doctrine ledger.**

---

## Decision 2 — 2026-05-01 monthly rebalance recommendation

### Question
Should the new 7-lane recommendation in `lane_allocation.json` ship?

### Audit findings — CRITICAL DISCOVERY

**2.1 The rebalance ALREADY shipped to the live config path** (verified
`trading_app/prop_profiles.py:497-531` and `:1097-1157`)
- `topstep_50k_mnq_auto.daily_lanes = ()` (empty hardcoded tuple).
- Line 1112: `path = ... / "docs" / "runtime" / "lane_allocation.json"`.
- `load_allocation_lanes` filters to `status in ("DEPLOY", "PROVISIONAL")` (line 1134).
- `effective_daily_lanes(profile)` returns whatever JSON has, since hardcoded is empty.
- **Conclusion: `lane_allocation.json` IS the live deployment for this profile.**

**2.2 What I unintentionally did at Step 4 of the original handoff plan**
- I ran `python scripts/tools/rebalance_lanes.py --profile topstep_50k_mnq_auto`.
- That script overwrote `lane_allocation.json` from rebalance_date 2026-04-18
  (4 DEPLOY lanes) to rebalance_date 2026-05-01 (7 DEPLOY lanes).
- I described this as "a recommendation" in my Step 4 summary. **That was wrong.**
  The write IS the deployment change.
- Backup of pre-rebalance state was saved to `/tmp/lane_alloc_before.json` —
  reversible.

**2.3 Lane-level overlap of rebalance with Chordia audit**

| Original 4 (Chordia-audited) | New 7 (current live) | Overlap |
|---|---|---|
| MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5 | (kept, ranked #1) | ✓ same lane, FAIL_BOTH t=2.276 |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5 | DROPPED | n/a — not in new 7 |
| MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12 | DROPPED | n/a — replaced by NYSE_OPEN ORB_G5 |
| MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12 | DROPPED | n/a — replaced by TOKYO_OPEN COST_LT08 |
| | MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100 | NEW — never Chordia-audited |
| | MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08 | NEW — never Chordia-audited |
| | MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15 | NEW — never Chordia-audited |
| | MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5 | NEW — never Chordia-audited |
| | MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30 | NEW — never Chordia-audited |
| | MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15 | NEW — never Chordia-audited |

**Net effect:** 1 of 4 Chordia-FAIL_BOTH lanes survived (the worst-failing one),
plus 6 lanes that have never been Chordia-audited got promoted.

**2.4 Mitigating factors**
- **Drift check 94 enforces** lanes exist in `validated_setups` with `status='active'`.
  The 6 new lanes presumably already cleared that gate (they wouldn't be in the
  allocator's input pool otherwise).
- **F-1 enforces position cap** even in signal-only mode (verified Decision 1).
- **No real broker connection in signal-only** → no actual capital at risk regardless
  of which lane set is loaded.
- **`check_allocation_staleness`** in `pre_session_check.py:720` blocks after 60d /
  warns after 35d.

**2.5 Aggravating factors**
- Decision 2 is shipping a 7-lane DEPLOY set without ANY Chordia gate (the 6 new
  lanes have no audit; the 1 surviving original lane FAILED its audit).
- `prop_profiles.py:519-530` notes describe the 7-lane profile as having lanes "validated
  via BH FDR + walk-forward + Criterion 11 Monte Carlo" — but Criterion 4 (Chordia) is
  not mentioned, and we just demonstrated H1 fails it at strict.
- The session-disposition document I committed earlier
  (`docs/runtime/session-disposition-2026-05-01-hq-trades-prereg-batch.md`) describes
  the rebalance as a "recommendation" — also inaccurate per this finding.

### Verdict

**REVERT the rebalance to the pre-2026-05-01 state** until the 7-lane set has either
(a) collective Chordia audit, or (b) explicit user GO accepting that the new 6 are
unaudited. This is the lower-risk default per `institutional-rigor.md` Rule 7
("ground in local resources before training memory") and Rule 8 ("Verify before
claiming").

**Concrete reversal:** `cp /tmp/lane_alloc_before.json docs/runtime/lane_allocation.json`.
Live state returns to the 4 lanes audited by this Chordia gate (which fail strict but
were the originally deployed set).

**Important caveat:** the original 4 ALSO fail strict Chordia. Reverting doesn't fix
the underlying portfolio-wide audit failure — it just stops the unaudited rebalance
from shipping on top of it. The 4 vs 7 question is "do we ship 4 known-to-fail-strict
lanes or 1 known-to-fail-strict + 6 never-audited lanes."

---

## Decision 3 — H3 NYSE_OPEN steel-man override

### Question
Should H3 NYSE_OPEN E2 RR1.0 CB1 COST_LT12 retain `has_theory: True` for the entry
mechanism even though the COST_LT12 filter overlay isn't grounded?

### Audit findings

**3.1 Chan Ch 7 grounding for entry mechanism** (verbatim from
`docs/institutional/literature/chan_2013_ch7_intraday_momentum.md`):
- p.155: "the triggering of stops. Such triggers often lead to the so-called breakout
  strategies. We'll see one example that involves an entry at the market open..."
- p.155: "There is an additional cause of momentum that is mainly applicable to the
  short time frame: the triggering of stops."
- p.157: "The execution of these stop orders often leads to momentum because a
  cascading effect may trigger stop orders placed further away from the open price as well."
- p.156: FSTX (Dow Jones STOXX 50 futures) gap-momentum strategy — APR 13%, Sharpe 1.4
  over 8 years on a European equity-index future.

**This grounds:**
- Stop-cascade-breakout mechanism on equity-index futures: ✓ direct, verbatim
- "Entry at the market open" use case: ✓ directly named (p.155)
- Equity-index intraday momentum result as benchmark: ✓ FSTX p.156 case study

**3.2 What it does NOT ground:**
- COST_LT12 cost-ratio filter — outside Chan's scope (Chan covers entry mechanisms,
  not cost filtering).
- NYSE-specific session narrative — Chan's case is European (FSTX), not NYSE. But the
  CLASS argument (equity-index intraday momentum at session open) generalizes.

**3.3 Steel-man strength assessment**
- The entry mechanism (E2 = stop-market on first range-cross of ORB at NYSE open) is
  EXACTLY Chan's p.155 stop-cascade-breakout-at-market-open description.
- The instrument (MNQ = micro Nasdaq E-mini) is an equity-index future, the same
  CLASS as FSTX.
- The session (NYSE_OPEN) is a market-open session, matching Chan's "entry at the
  market open" framing.
- The RR=1.0 target is conservative (no asymmetry assumption).
- The COST_LT12 overlay doesn't change the entry mechanism; it just abstains on
  high-cost days.

**Steel-man verdict: STRONG.** This is the cleanest mechanism-to-literature mapping
of the 4 lanes audited.

**3.4 Counter-steel-man**
- Chordia 2018's threshold relaxation requires "strong pre-registered economic theory
  support" — what counts as "strong"?
- The pre-reg cited "Fitschen Ch 5-7 + Chan 2013 Ch 7" — Fitschen Ch 5-7 was
  fabricated. Chan Ch 7 alone is one source, not multiple.
- Project doctrine per `pre_registered_criteria.md` Criterion 4: relaxation is
  binary, not gradient. Either has theory or doesn't.
- Risk of double-counting: if H3 keeps `has_theory=True` then so should H1 EUROPE_FLOW
  (Chan Ch 7 mechanism applies to European-session equity-index futures equally — it's
  literally the FSTX case study). H1 t=2.276 fails BOTH thresholds; flipping H3 doesn't
  save H1.

**3.5 If the steel-man is accepted:**
- H3 verdict shifts: t=3.412, threshold=3.00 → PASS_PROTOCOL_A
- H1 EUROPE_FLOW also legitimately deserves the steel-man (same Chan Ch 7 mechanism,
  even more directly via FSTX which IS a European-session lane). H1 t=2.276 → still
  FAIL_BOTH at theory-grounded threshold 3.00.
- H2 COMEX_SETTLE: not equity-index — Chan Ch 7 doesn't apply. Stays FAIL_BOTH at
  strict.
- H4 TOKYO_OPEN: equity-index (NKD-proxy via MNQ-Asia-session) but Chan's FSTX example
  is European, not Asian. Generalizing is a stretch. Tentatively stays FAIL_BOTH at
  strict.

**3.6 Re-revised verdict matrix if steel-man accepted for equity-index sessions:**

| Hyp | Lane | t | Steel-man applies? | Threshold | New verdict |
|---|---|---|---|---:|---|
| H1 | EUROPE_FLOW + ORB_G5 | 2.276 | YES (FSTX is European) | 3.00 | **FAIL_BOTH (still)** |
| H2 | COMEX_SETTLE + ORB_G5 | 3.276 | NO (commodity session) | 3.79 | **FAIL_BOTH** |
| H3 | NYSE_OPEN + COST_LT12 | 3.412 | YES (NYSE = equity index open) | 3.00 | **PASS_PROTOCOL_A** |
| H4 | TOKYO_OPEN + COST_LT12 | 3.268 | partial (Asia-equity, not in Chan example) | 3.79 (conservative) | **FAIL_BOTH** |

Even with the most generous steel-man, only 1 of 4 lanes passes (H3).

### Verdict

**MODIFY proposed default.** The steel-man for H3 IS strong (mechanism-to-literature
cleanest match of the 4). Consistency demands the same steel-man for H1 EUROPE_FLOW
(Chan FSTX literally is the European-session equity-index case). Honest update:

- H1 EUROPE_FLOW: `has_theory=True` (Chan Ch 7 grounded), still FAIL_BOTH on raw t.
- H2 COMEX_SETTLE: `has_theory=False`, FAIL_BOTH.
- H3 NYSE_OPEN: `has_theory=True`, PASS_PROTOCOL_A.
- H4 TOKYO_OPEN: `has_theory=False` (Asian-session not in Chan FSTX example), FAIL_BOTH.

**1-of-4 PASS, 3-of-4 FAIL.** Better than the current 4-of-4 FAIL but only marginally.
Still a portfolio-wide finding requiring user attention.

This re-revision needs runner re-run with these per-lane `has_theory` flags before
result-doc finalization.

---

## Recommended next actions (in order)

1. **Decision 2 (URGENT):** revert `lane_allocation.json` to the 2026-04-18 state via
   `cp /tmp/lane_alloc_before.json docs/runtime/lane_allocation.json`, OR get explicit
   user GO to keep the 7-lane unaudited set live.

2. **Decision 3 (re-run):** apply the per-lane steel-man flags (H1=True, H2=False,
   H3=True, H4=False) and re-run `research/chordia_revalidation_deployed_2026_05_01.py`.
   Update the result doc and pre-reg yaml to v3.

3. **Decision 1 (no action needed):** doctrine action stands — research-provisional +
   signal-only continuation IS protective.

4. **Audit-trail commit:** once Decisions 2-3 resolved, commit the runner + result
   doc + pre-reg + Fitschen extract upgrade + this audit doc + design doc as one
   `[judgment]` audit-correction commit. Drift check before commit.

5. **Adversarial-audit-gate trigger:** per `.claude/rules/adversarial-audit-gate.md`,
   any `[judgment]` commit touching `trading_app/live/`, `risk_manager.py`, or
   `pipeline/` requires evidence-auditor dispatch. The audit-correction commit doesn't
   touch those paths (only docs + research script + extract MD), so the gate doesn't
   fire. Documenting for completeness.

## Save-as-you-go log

- 2026-05-01 — design doc written
  (`docs/runtime/chordia-revalidation-honest-grounding-design-2026-05-01.md`)
- 2026-05-01 — Fitschen extract upgraded with Ch 5/6/7 audit
- 2026-05-01 — runner updated to use canonical `chordia_threshold(has_theory)`
- 2026-05-01 — pre-reg yaml updated with audit-correction block
- 2026-05-01 — result doc rewritten as v2 (4-of-4 FAIL_BOTH)
- 2026-05-01 — this decision-audit doc written
- 2026-05-01 — discovered the rebalance IS deployment, not recommendation
- PENDING — re-run with H1/H3 steel-man `has_theory=True`, others False
- PENDING — user decision on Decision 2 reversal
