# Chordia Unlock - Theory Feasibility Scan (2026-05-02)

**Status:** completed, read-only literature scan
**Scope:** classify the four filter classes in `chordia_audit_unlock_pass_chordia_strategies`
as theory-grantable, class-grounded-only, or unsupported before any new pre-regs are written.
**Inputs used:** live queue state, live lane-allocation truth, local literature extracts only.

## Purpose

[MEASURED] The open P1 queue item is
`chordia_audit_unlock_pass_chordia_strategies` in
`docs/runtime/action-queue.yaml`.

[MEASURED] The prior "resume the 7-lane book" framing in
`docs/runtime/handoff-2026-05-02.md` is stale against current live truth:
`docs/runtime/lane_allocation.json` now shows a 2026-05-01 rebalance with
**1 DEPLOY lane** and 16 paused lanes after the allocator Chordia gate.

[GROUNDED] Under `docs/runtime/chordia_audit_log.yaml`, a theory grant must
point to a real file under `docs/institutional/literature/`. Semantic
plausibility alone is not enough.

[INFERRED] The cheapest next step is therefore not a broad portfolio audit or
an MCP build. It is a theory-feasibility scan on the blocked PASS_CHORDIA
filters so only auditable candidates get promoted into Pathway-A pre-reg work.

## Sources Consulted

[GROUNDED] Core ORB / intraday-momentum grounding:
- `docs/institutional/literature/fitschen_2013_path_of_least_resistance.md`
- `docs/institutional/literature/chan_2013_ch7_intraday_momentum.md`

[GROUNDED] Portfolio / sizing context reviewed for scope only:
- `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md`
- `docs/institutional/literature/carver_2015_ch11_portfolios.md`
- `docs/institutional/literature/carver_2015_ch12_speed_and_size.md`

[MEASURED] Live orientation surfaces:
- `docs/runtime/action-queue.yaml`
- `docs/audit/results/2026-05-01-chordia-audit-unlock-triage.md`
- `docs/runtime/chordia_audit_log.yaml`
- `docs/runtime/lane_allocation.json`

## Filter-Class Verdicts

| Filter class | Verdict | Why |
|---|---|---|
| `COST_LT*` | `NO_THEORY_GRANT` | [GROUNDED] Fitschen's filter taxonomy does not include cost-ratio filters, and no loaded local extract gives an economic-theory basis for "lower friction relative to ORB risk" as a standalone alpha mechanism. [INFERRED] It is a valid execution-quality gate, but not a theory grant. |
| `X_MES_ATR60` | `UNSUPPORTED` | [GROUNDED] None of the loaded local extracts grounds cross-asset MES volatility as a causal theory for MNQ session edge. [UNSUPPORTED] No direct local citation yet for "MES ATR regime conditions MNQ breakout quality." |
| `OVNRNG_100` | `CLASS_GROUNDED_ONLY` | [GROUNDED] Fitschen Ch 6 supports volatility/context filters as a class. [GROUNDED] Chan supports overnight-event/open momentum as a class. [UNSUPPORTED] None of the loaded extracts directly grounds the specific project claim that a 100-point overnight range threshold is the causal mechanism. |
| `VWAP_MID_ALIGNED` | `INFERRED_BUT_NOT_YET_GROUNDED` | [GROUNDED] Chan supports intraday momentum from overnight/news/open imbalances and stop cascades, especially around open/data-release windows. [UNSUPPORTED] None of the loaded local extracts explicitly grounds VWAP-mid alignment as the mechanism. [INFERRED] This filter may be defensible as a fair-value / participation anchor, but that theory is not yet locally cited. |

## What Survived

[GROUNDED] The project's **core ORB breakout premise** remains well grounded by
Fitschen Ch 3 and Chan Ch 7. That supports the base intraday-momentum family.

[GROUNDED] `COST_LT*` can still pass the strict no-theory Chordia path because
the live t-stats are already above the 3.79 hurdle for the relevant blocked
candidates.

[GROUNDED] `OVNRNG_100` is not a fantasy filter class. It has class-level
support as a volatility/context gate, but the loaded literature does not yet
justify a clean `has_theory: true` entry in `chordia_audit_log.yaml`.

## What Did Not Survive

[UNSUPPORTED] `X_MES_ATR60` should not receive a theory grant from the current
local literature set.

[UNSUPPORTED] `VWAP_MID_ALIGNED` should not receive a theory grant yet from the
current local literature set, even though the session-level momentum story is
plausible.

[UNSUPPORTED] The older "7-lane corrected book is the next audit surface"
framing should not be treated as current operating truth. The Chordia gate has
already collapsed the live book to 1 active lane.

## Immediate Implications For The 8 Blocked PASS_CHORDIA Names

[MEASURED] The triage doc groups the 8 blocked names into four distinct filter
classes: `VWAP_MID_ALIGNED`, `OVNRNG_100`, `X_MES_ATR60`, and `COST_LT*`.

[INFERRED] Based on the evidence loaded in this scan:
- `COST_LT*` candidates can proceed only on the strict `t >= 3.79` path.
- `X_MES_ATR60` candidates should also stay on the strict `t >= 3.79` path.
- `OVNRNG_100` candidates are not ready for `has_theory: true` unless a more
  direct local citation is extracted.
- `VWAP_MID_ALIGNED` candidates are the best theory-upside candidates, but they
  still need an actual local fair-value / VWAP / participation reference before
  a doctrine grant would be honest.

## Recommended Next Step

[INFERRED] Do **not** start with MCP work and do **not** restart the stale
7-lane vestigialness path as if it were current.

[INFERRED] The next highest-EV move is:
1. Write a compact follow-up note that explicitly marks the 2026-05-02 runtime
   handoff as stale against live lane-allocation truth.
2. Draft Pathway-A pre-regs only for the strict-t candidates that do not depend
   on a theory grant.
3. Separately, extract one more local literature source only if we want to try
   to rescue `VWAP_MID_ALIGNED` or `OVNRNG_100` into `has_theory: true`.

## Bottom Line

[MEASURED] The repo's active capital-EV thread is the Chordia unlock queue
item, not generic MCP improvement.

[GROUNDED] Current local literature supports the ORB momentum base, but not all
four blocked filter mechanisms equally.

[INFERRED] The honest posture today is conservative:
- `COST_LT*`: no theory grant
- `X_MES_ATR60`: unsupported
- `OVNRNG_100`: class-grounded only
- `VWAP_MID_ALIGNED`: plausible but not yet grounded

[INFERRED] That means the autonomous continuation path should narrow around
strict-t unlocks first, then pursue additional literature only where it changes
the decision.
