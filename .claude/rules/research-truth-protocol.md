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

## Validated Universe Rule (MANDATORY — added 2026-04-04)

**NEVER run research queries against the full unfiltered `orb_outcomes` table.**

`orb_outcomes` contains every possible outcome for every session, entry model, RR target, and confirm_bars — 3M+ rows of noise. Testing a new feature against this undifferentiated mass is testing noise against noise. Any "signal" found is meaningless because:
1. Most of those parameter combos are NOT validated and have negative expectancy
2. The massive N (millions) makes even 0.01R random fluctuations "statistically significant"
3. Filters are not applied — you're mixing filtered and unfiltered trade populations

**Research queries MUST be scoped to one of:**
- The 124 validated strategies (join `validated_setups` to get the strategy dimensions, then query `orb_outcomes` with those exact filters applied via `daily_features`)
- A specific hypothesis about a specific session+instrument+filter combo
- The deployed portfolio (strategy IDs from `prop_profiles.ACCOUNT_PROFILES`)

**Template for valid research query:**
```sql
-- Get outcomes for VALIDATED strategies only, with filters applied
SELECT o.*, d.prev_day_close, d.prev_day_low, d.prev_day_range
FROM orb_outcomes o
JOIN daily_features d
    ON o.trading_day = d.trading_day AND o.symbol = d.symbol AND o.orb_minutes = d.orb_minutes
JOIN validated_setups v
    ON o.symbol = v.instrument AND o.orb_label = v.orb_label
    AND o.orb_minutes = 5 AND o.entry_model = v.entry_model
    AND o.confirm_bars = v.confirm_bars AND o.rr_target = v.rr_target
WHERE v.status = 'active'
  AND [apply v.filter_type condition from daily_features]
```

**If you catch yourself writing `FROM orb_outcomes WHERE symbol IN ('MGC','MNQ','MES')` without a validated_setups join or explicit filter application — STOP. You are about to test noise.**

## Hard Rules

- If docs conflict with canonical data → docs are STALE. Mark them, do not trust them.
- 2026 holdout is sacred. Do not use for discovery. Forward-test = monitoring only.
- No edits before read-only audit. Non-trivial changes require PASS 1 (audit) before PASS 2 (implement).
- Derived layers marked `DISCOVERY SAFETY: UNSAFE` in their module docstrings.
- **NEVER simulate strategy P&L without applying the strategy's filter.** Querying `orb_outcomes` with only instrument+session+entry_model+RR but WITHOUT the filter_type (COST_LT, VOL_RV, ATR70_VOL, OVNRNG, ORB_G, etc.) produces UNFILTERED results that overcount trades and misrepresent both P&L and risk. Every simulation MUST join `daily_features` and apply the exact filter condition. If the filter column is missing or broken, say UNVERIFIED — do not substitute `1=1`.
