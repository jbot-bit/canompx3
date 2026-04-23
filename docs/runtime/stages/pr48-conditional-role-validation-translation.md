# PR48 Conditional-Role Validation Translation

**Status:** active bounded stage
**Date:** 2026-04-23
**Purpose:** translate the recovered `PR48` conditional-role shortlist into the current validation/allocation system without pretending the research docs are already promotion-ready.

## Why this stage exists

The recovered PR48 work is real and reproducible, but current repo plumbing still assumes promotion objects look like standalone validated lanes in `validated_setups`.

PR48 does not naturally fit that shape yet:

- `MES:q45_exec` is a role-aware executable filter arm
- `MGC:cont_exec` is an executable sizing arm
- `DUO:mes_q45_plus_mgc_cont_exec` is a bounded portfolio combination
- `MNQ:shadow_addon` is explicitly a shadow add-on, not a direct live promotion

This stage prevents two failure modes:

1. broad generic confluence rediscovery instead of resuming the real branch
2. premature shelf/live promotion from result docs alone

## Inputs

- `docs/institutional/conditional-edge-framework.md`
- `docs/audit/results/2026-04-22-pr48-conditional-role-implementation-v1.md`
- `docs/audit/results/2026-04-22-pr48-role-followthrough-v1.md`
- `docs/audit/results/2026-04-22-pr48-promotion-shortlist-v1.md`
- `docs/audit/results/2026-04-23-pr48-conditional-edge-recovery-audit.md`
- `trading_app/strategy_validator.py`
- `trading_app/validated_shelf.py`
- `trading_app/lane_allocator.py`
- `docs/runtime/lane_allocation.json`

## Exact questions

1. What is the honest intermediate object for a recovered conditional-role candidate in this repo:
   research-only result doc, shadow routing candidate, bounded execution sleeve, or a new validated-surface type?
2. Which PR48 shortlist arms can be compared to the current live book as free-slot adds, same-session replacements, or shadow-only candidates?
3. What minimal schema or report surface is required so conditional-role outputs are not mislabeled as standard standalone promotions?

## Rules

- Use canonical layers for truth: `daily_features`, `orb_outcomes`
- Treat `validated_setups`, `lane_allocation.json`, docs, and memory as downstream surfaces, not proof
- Do not rerun broad confluence discovery
- Do not auto-promote a PR48 arm into `validated_setups` without an explicit translation rule
- Do not use thin OOS to retune thresholds or executable maps

## Required outputs

1. A role-by-role translation table:
   candidate, intended role, parent comparator, live-book interaction, honest next state
2. A measured statement of whether the current allocator can consume the object natively
3. If not, the smallest safe bridge:
   report-only surface, shadow route, or new schema field
4. A final verdict for each shortlist arm:
   `SHADOW_ONLY`, `NEEDS_BRIDGE`, `READY_FOR_BOUNDED_TRANSLATION`, or `DEAD`

## Non-goals

- No fresh factor scan
- No pooled multi-confluence program reboot
- No live deployment recommendation from this stage alone
