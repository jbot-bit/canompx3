# Research Truth Protocol (enforced 2026-03-24)

Canonical authority: `RESEARCH_RULES.md` § Discovery Layer Discipline.
This file enforces the protocol in Claude Code sessions.

## Layer Classification

| Layer | Status | Safe for discovery? |
|-------|--------|---------------------|
| `bars_1m` | CANONICAL | YES |
| `daily_features` | CANONICAL | YES |
| `orb_outcomes` | CANONICAL | YES — primary trade-level truth |
| `validated_setups` | DERIVED | **NO** — may be stale/contaminated |
| `edge_families` | DERIVED | **NO** |
| `live_config` / LIVE_PORTFOLIO | HARDCODED | **NO** — deployment state, not research truth |
| Docs, comments, memory files | META | **NO** — verify against canonical layers first |

## Research Claim Requirements

Every research claim must include:
1. Source layer (must be canonical)
2. Data state timestamp (e.g., "orb_outcomes through 2026-03-23")
3. Exact query or script path
4. Sample size (N)
5. p-value (two-tailed t-test, exact)
6. K used for BH FDR (report both global K and instrument/family K; use instrument/family K for promotion decisions, global K for headline claims; never swap K post-hoc)
7. WFE (if walk-forward was performed)

## Hard Rules

- If docs conflict with canonical data → docs are STALE. Mark them, do not trust them.
- 2026 holdout is sacred. Do not use for discovery. Forward-test = monitoring only.
- No edits before read-only audit. Non-trivial changes require PASS 1 (audit) before PASS 2 (implement).
- Derived layers marked `DISCOVERY SAFETY: UNSAFE` in their module docstrings.
- **NEVER simulate strategy P&L without applying the strategy's filter.** Querying `orb_outcomes` with only instrument+session+entry_model+RR but WITHOUT the filter_type (COST_LT, VOL_RV, ATR70_VOL, OVNRNG, ORB_G, etc.) produces UNFILTERED results that overcount trades and misrepresent both P&L and risk. Every simulation MUST join `daily_features` and apply the exact filter condition. If the filter column is missing or broken, say UNVERIFIED — do not substitute `1=1`.
