# Chordia revalidation — honest grounding design (2026-05-01)

**Status:** in-progress design note. Save-as-you-go per user instruction. Captures the
audit findings, the design decisions for cascading the corrections, and the iteration
plan. Updated as work progresses.

## What changed and why

The 2026-05-01 Chordia revalidation pre-reg
(`docs/audit/hypotheses/2026-05-01-chordia-revalidation-deployed-lanes.yaml`) declared
`has_theory: true` for all 4 deployed lanes citing "Fitschen Ch 5-7 + Chan 2009 Ch 1
§1.4 + Chan 2013 Ch 7" as theory grounding. User audit caught this as suspicious;
direct PDF extraction of `resources/Building_Reliable_Trading_Systems.pdf` Ch 5/6/7
(pages 82-87, 89-101, plus TOC + chapter titles) **falsified the citations**:

- **Ch 5 (Exits, pp.65-87):** exit-mechanic methodology. No session narratives.
- **Ch 6 (Filters, pp.89-105):** four filter classes — day-of-week, seasonal,
  volatility, longer-term-trend. No session-specific institutional-flow stories. The
  size-gate filter polarity Fitschen tests positive (skip-HIGH-vol) is OPPOSITE to
  what `ORB_G{N}` filters do (skip-LOW-vol).
- **Ch 7 (Money Management, pp.107-117):** position-sizing methodology by title;
  read-confirmed not session-mechanism content.

Combined with the prior `chan_2009_ch1_intraday_session_handling.md` audit (which
showed Chan 2009 has no §1.4 — phantom citation), the pre-reg's theory citations
collapse for all 4 lanes:

| Hyp | Lane | Original cite | Honest grounding |
|-----|------|---------------|-------------------|
| H1 | EUROPE_FLOW + ORB_G5 | Fitschen Ch 5-7 | Ch 3 grounds intraday-trend-follow on equity indices, NOT session OR `ORB_G{N}` polarity → `has_theory=False` |
| H2 | COMEX_SETTLE + ORB_G5 | Fitschen Ch 5-7 | Same → `has_theory=False` |
| H3 | NYSE_OPEN + COST_LT12 | Fitschen Ch 5-7 + Chan Ch 7 | Chan Ch 7 grounds equity-index intraday momentum; COST_LT12 not in any taxonomy → `has_theory=False` |
| H4 | TOKYO_OPEN + COST_LT12 | Fitschen Ch 5-7 + Chan 2009 Ch 1 §1.4 | Both citations phantom → `has_theory=False` |

## Cascading impact

| What | Was | Becomes |
|------|-----|---------|
| Chordia threshold | 3.00 (theory-grounded) | 3.79 (strict) |
| H1 t=2.276 | FAIL_BOTH | FAIL_BOTH (unchanged — already failing weakest threshold) |
| H2 t=3.276 | PASS_PROTOCOL_A | **FAIL_BOTH** (drops below strict 3.79) |
| H3 t=3.412 | PASS_PROTOCOL_A | **FAIL_BOTH** (drops below strict 3.79) |
| H4 t=3.268 | PASS_PROTOCOL_A | **FAIL_BOTH** (drops below strict 3.79) |
| Verdict count | 1 FAIL / 3 PASS | **4 FAIL / 0 PASS** |

This is a portfolio-wide failure of the defensive Chordia honesty gate.

## Design — what to do, in iteration order

### Iteration 1: Update runner verdict logic (current)

The runner's `verdict =` block at lines 185-190 hardcodes thresholds via the
`CHORDIA_T_WITH_THEORY` and `CHORDIA_T_WITHOUT_THEORY` constants but does not
*select* between them based on `lane.has_theory`. It currently always uses
`CHORDIA_T_WITH_THEORY` (3.00) as the FAIL/PROTOCOL_A boundary. That's a bug
discovered during this audit — the runner ignores the per-lane `has_theory` flag.

**Fix:** select the lane's threshold via `chordia_threshold(lane.has_theory)` from
`trading_app.chordia` (canonical helper). The verdict bands then become:
- t < lane_threshold → FAIL_BOTH
- t >= lane_threshold → PASS

If `has_theory=True`, "PASS" splits into PASS_PROTOCOL_A (t<3.79) and PASS_CHORDIA
(t>=3.79). If `has_theory=False`, "PASS" is just PASS_CHORDIA (no PROTOCOL_A band
exists for non-theory lanes).

This is the canonical-delegation way (per `chordia.chordia_threshold`).

### Iteration 2: Re-run runner, capture honest verdict table

Expected: 4 FAIL_BOTH (since all 4 t-stats are below 3.79).

### Iteration 3: Fix the pre-reg yaml's literature citations

Update `docs/audit/hypotheses/2026-05-01-chordia-revalidation-deployed-lanes.yaml`:
- Each H1-H4 hypothesis: change `has_theory: true` → `has_theory: false`
- Each `economic_theory:` block: replace fabricated narratives with HONEST text
  pointing to the audit finding
- `kill_criteria` sections: update to reflect strict threshold
- Add `audit_correction_2026_05_01:` block at top documenting the falsification

### Iteration 4: Update result doc with honest verdicts

Rewrite `docs/audit/results/2026-05-01-chordia-revalidation-deployed-lanes.md`:
- Verdict table → all 4 FAIL_BOTH
- Doctrine action expanded: not 1-lane downgrade but 4-lane portfolio-wide downgrade
- "What this gate doesn't do" updated
- New "Audit correction" section explaining the literature-grounding fix

### Iteration 5: Check downstream pre-regs that depend on this

- `2026-05-01-carver-stage2-vol-targeted-sizing.yaml` — gated on PASS_CHORDIA, was
  already MOOT per old verdicts (no PASS_CHORDIA), still MOOT now (4 FAIL_BOTH).
  Decision: no edit needed; mark gate-status in result doc cross-ref.
- `2026-05-01-mnq-garch-p70-cross-session-companion.yaml` — H2 COMEX_SETTLE was its
  "home lane". With H2 now FAIL_BOTH the GARCH companion's premise needs reconsidering.
  Decision: do NOT delete, but add audit-correction-aware status note.
- `2026-05-01-aperture-extension-o15-o30-london-usdata.yaml` — already self-corrects on
  Ch 5-7 phantom (lines 34-38). Verify the self-correction is consistent with my
  upgraded extract; if so leave as-is.

### Iteration 6: Doctrine implications

If 4-of-4 deployed lanes fail strict Chordia, the **entire signal-only TopStep
deployment** is on shaky ground. This is a doctrine-level finding that must be flagged
clearly to user. The institutional response per `pre_registered_criteria.md` Criterion
4 is research-provisional + signal-only continuation, but the SCALE of the finding
(portfolio-wide, not single lane) deserves explicit user attention.

**Open question requiring user decision (not a code change):**
- Continue running signal-only on the 4 lanes for OOS evidence accumulation, OR
- Pause signals on the 4 lanes pending clean re-derivation under strict discovery, OR
- Audit one more time — is `has_theory=False` *too* strict? Could Chan Ch 7's
  equity-index intraday momentum support H3 NYSE_OPEN as `has_theory=True` for the
  ENTRY mechanism while leaving COST_LT12 ungrounded?

The third option deserves consideration because Chan Ch 7's FSTX example IS a
genuine equity-index intraday momentum result (Sharpe 1.4, p.156). NYSE_OPEN is an
equity-index session and the E2 entry is a stop-cascade breakout (which Chan
explicitly grounds at p.155). The strongest honest reading might be: H3 entry
mechanism IS theory-grounded, but the COST_LT12 filter overlay isn't — so the LANE
as a configured whole is mixed-grounding.

This nuance is appropriate for the result-doc discussion section. Conservative
default for the verdict gate: stick with `has_theory=False` (the *whole lane* must
be grounded for the lane to count as theory-grounded).

### Iteration 7: Save state, verify drift, await user decision on doctrine action

- Save all artifacts to disk
- `python pipeline/check_drift.py --fast` → expect green
- Stop. Report verdict + doctrine question. Do NOT touch allocator code or
  `lane_allocation.json` or `prop_profiles.py`. Decision belongs to user.

## Anti-pigeon-hole self-check

Before locking the new `has_theory=False` flags:

1. **Did I just flip the flag because the user pushed back?** — No. The flag flip
   follows from PDF-verified non-existence of the cited content. Different lanes
   (e.g., a hypothetical "MNQ_LONDON_METALS_E2_RR1.5_CB1_HIGH_VOL_GATE_FILTER") would
   get `has_theory=True` because Ch 6 explicitly grounds high-vol-exclusion filters.
   The audit is mechanism-by-mechanism, not blanket.

2. **Am I using "no theory" as a way to make the verdict more dramatic?** — No. The
   pre-reg's failure_policy explicitly handles FAIL_BOTH via doctrine ledger only;
   there's no allocator code change either way. The drama is the SCALE of failure
   (4-of-4), not a doctrine escalation.

3. **Is there a steel-man for `has_theory=True` I'm missing?** — Steel-man for H3:
   Chan Ch 7 p.155 grounds stop-cascade-breakout mechanism on intraday equity index
   futures. NYSE_OPEN E2 stop-market entry IS that mechanism on an equity index
   future. The COST_LT12 overlay attenuates but doesn't reverse the entry mechanism.
   Defensible reading: H3 entry mechanism `has_theory=True`, filter overlay
   `has_theory=N/A`. Compromise verdict: H3 keeps theory grounding for the entry,
   reports the mixed-grounding caveat, threshold stays at 3.00 — H3 PASS_PROTOCOL_A.

   This compromise IS defensible. It doesn't change H1/H2/H4 which lack equity-index
   intraday momentum grounding (H1 EUROPE is a EU session, H2 COMEX is a metals
   session, H4 TOKYO is an Asian session — Chan's FSTX result is European index but
   the session-mechanism story is what's missing across all of them).

4. **What happens if I'm too aggressive with has_theory=False?** — Lanes get
   downgraded to research-provisional that arguably shouldn't be. But the project's
   doctrine specifically says "research-provisional + signal-only" is the
   protective downgrade — it does NOT remove lanes from live OOS accrual. It just
   blocks them from real-money exposure (which is already the case in signal-only).
   So aggressive grounding has low cost.

5. **What happens if I'm too lenient with has_theory=True?** — Lanes pass a defensive
   gate that they shouldn't, real-money exposure could eventually flow toward them
   via Carver Stage-2 sizing or rebalance ranking, capital at risk on noise. This is
   the high-cost direction.

**Decision:** asymmetric cost favors strict `has_theory=False` default. H3
compromise reading documented but the verdict gate uses `False` for the lane as a
configured whole.

## Update log (timestamps for save-as-you-go)

- 2026-05-01 — initial draft of design note written (this file)
- 2026-05-01 — Fitschen extract upgraded with Ch 5/6/7 audit
- 2026-05-01 — runner LANE table updated to `has_theory=False`; verdict-logic bug
  discovered (runner ignores per-lane flag). Iteration 1 fix in flight.
