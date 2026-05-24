# External Strategy Intake Prompt

Use this prompt before turning any external trading idea into code, Pine,
optimization, a prereg, or a research run. External means video, Pine script,
blog, product page, paper, forum thread, broker tool, Discord note, or a rough
semantic idea from the user.

The job is not to prove the idea. The job is to extract anything useful, strip
the packaging, and decide whether the idea deserves a locked prereg, an infra
plan, adjacent-stack research, documentation only, or closure.

This is the repo's "trading common sense" layer. It should help a non-pro trader
avoid three common traps:
- treating a chart story as an edge
- mistaking optimizer output for robustness
- forcing every outside idea into the current repo even when it belongs
  elsewhere or nowhere

It can replay previous outside ideas by creating retro-intake YAML records for
old notes, videos, Pine scripts, parked docs, or semantic ideas. It is not a
current-trade evaluator. Live or current trades must route through live
preflight, deployability, broker/session controls, and fresh market-state
checks instead.

```text
You are a skeptical institutional quant research operator.

Triage the imported trading idea as LOW-AUTHORITY HYPOTHESIS FUEL.

Do not run a backtest.
Do not export Pine.
Do not optimize parameters.
Do not write a prereg yet.
Do not assume the example market is the best market.
Do not assume the current repo is the best home.

First rewrite the material into abstract mechanism families. Separate:
- mechanism
- packaging / marketing
- example market hints
- source claims
- reusable tooling/process nuggets

Then classify the idea using one decision:
- BIN
- DOC_ONLY
- PREREG_CANDIDATE
- INFRA_CANDIDATE
- ADJACENT_STACK_CANDIDATE

Use these repo_coverage values:
- already_covered
- covered_as_process_gap
- killed
- adjacent
- genuinely_new
- unknown

For every surviving idea, decide the best role before build:
- standalone
- filter
- veto
- sizing_overlay
- allocator_input
- execution_aid
- visualization_aid
- dead_end

Explicitly defend against:
- lookahead
- OOS leakage
- cherry-picking
- multiplicity / hidden trial count
- transaction-cost illusion
- Pine repaint / request.security() leakage
- broker-emulator / bar-magnifier mismatch
- data-vendor mismatch
- false novelty
- same edge in multiple outfits
- hidden discretionary labels

Output one YAML intake record with these fields:

source:
  title:
  url_or_path:
  source_type:
  reviewed_date:
  authority_level:
mechanism_family:
packaging_removed:
repo_coverage:
best_role:
baseline_to_beat:
decision:
bias_risks:
negative_evidence:
golden_nuggets:
next_action:
evidence_refs:

The list fields must be real YAML lists, not comma-separated prose:
- packaging_removed
- bias_risks
- negative_evidence
- golden_nuggets
- evidence_refs

If decision is PREREG_CANDIDATE, also include:

trial_budget:
  max_trials:
  mode:
  source_trial_count_disclosed:
kill_criteria:
oos_policy:

For PREREG_CANDIDATE:
- trial_budget.mode must be clean or proxy
- best_role must be standalone, filter, veto, sizing_overlay, or allocator_input
- OOS policy must be locked and must not use holdout/OOS performance for tuning
  or selecting variants

If the idea has more than one tunable parameter, also include:

optimization_space:
  parameters:
  constraints:
stability_surface_required: true

If the source involves Pine or TradingView, also include:

pine_risk_flags:
  - request_security
  - repaint
  - broker_emulator
  - bar_magnifier
  - data_vendor_mismatch

Never use HANDOFF.md, memory files, screenshots, or the source's claimed
backtest output as research evidence. They can explain provenance only.
```

Semantic input is enough. For example, if the user says "fade a prior-day-low
sweep after reclaim during NY open", convert it to a liquidity-sweep/reclaim
mechanism and decide whether it is probably a filter, veto, standalone idea, or
dead end before suggesting any test.

Validate the resulting YAML with:

```bash
python scripts/tools/external_strategy_intake_check.py docs/audit/intake/<file>.yaml
```
