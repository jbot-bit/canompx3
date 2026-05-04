# DST Pipeline Remediation — Execution Plan

## Status: ACTIVE — Feb 2026

---

## What's Broken (and What's NOT)

### NOT broken — no rebuild needed:
- `bars_1m` / `bars_5m` — Raw UTC data, correct
- `daily_features` — ORB computation is correct for stated clock times
- `orb_outcomes` (689K rows) — Trade outcomes are correct for the Brisbane clock times they represent

### WHAT IS broken — analysis layer:
- `experimental_strategies` — blended avgR/Sharpe/WR across winter+summer
- `validated_setups` — validated on blended numbers, no DST visibility
- `edge_families` — downstream of blended validated_setups
- All TRADING_RULES.md stats for 0900/1800/0030/2300

### Root cause:
`strategy_discovery.py` aggregates ALL orb_outcomes for a session into one metric set. It doesn't know or care about DST regime. The validator then validates those blended numbers. Neither component tags winter vs summer.

---

## Remediation Tracks

### Track A: Bake DST into Production Validator (PRIORITY 1)

**Goal:** Every strategy at an affected session (0900/1800/0030/2300) gets winter/summer split metrics stored permanently.

**Changes required:**

1. **`trading_app/db_manager.py`** — Add columns to `validated_setups`:
   ```sql
   dst_winter_n        INTEGER,
   dst_winter_avg_r    DOUBLE,
   dst_summer_n        INTEGER,
   dst_summer_avg_r    DOUBLE,
   dst_verdict         TEXT,      -- STABLE/WINTER-DOM/SUMMER-DOM/WINTER-ONLY/SUMMER-ONLY/LOW-N/CLEAN
   ```

2. **`trading_app/strategy_validator.py`** — Add DST analysis phase:
   - New function `compute_dst_split()` that:
     - Queries `orb_outcomes` for the strategy's session/instrument/entry_model/rr/cb/filter
     - Tags each trading day as winter or summer (using `pipeline/dst.py` logic)
     - Computes winter and summer metrics separately
     - Returns verdict per the classification in the research script
   - Called AFTER Phase 4b (walk-forward) for affected sessions only
   - NOT a rejection gate initially — INFO phase that populates columns
   - Prints winter/summer split in console output

3. **`trading_app/strategy_discovery.py`** — Add DST columns to `experimental_strategies`:
   - Same columns as above
   - Computed during discovery so validator has visibility before promotion
   - This means discovery also needs the `is_winter()` function

**Files touched:** `db_manager.py`, `strategy_validator.py`, `strategy_discovery.py`, `pipeline/dst.py` (import only)

**Execution:** Claude Code can do this — it's code changes + a full re-run of discovery+validation.

### Track B: Volume Check + New Session Candidates (PRIORITY 2)

**Goal:** Determine if edge differences are volume-driven and evaluate 09:30/19:00 for pipeline addition.

**Script:** `research/research_volume_dst_analysis.py` (prompt already written at `research/PROMPT_volume_dst_and_new_sessions.md`)

**Depends on:** Database access (C:/db/gold.db). Claude Code task.

**Decision gates:**
- If volume explains the winter/summer split → dynamic sessions preferred
- If volume is similar but edge differs → structural clock-time effect, keep fixed
- If 09:30/19:00 have sufficient volume + low overlap with existing sessions → ADD to pipeline

### Track C: Add New Sessions to Pipeline (PRIORITY 3, conditional on Track B)

**Only if Track B confirms viability.** Changes needed:
1. Add `0930` and `1900` to `SESSION_CATALOG` in `pipeline/dst.py`
2. Add to enabled sessions in `pipeline/asset_configs.py`
3. Rebuild `daily_features` for new sessions
4. Rebuild `orb_outcomes` for new sessions
5. Run discovery + validation for new sessions

---

## Execution Order

```
Step 1: Track A — DST integration into validator/discovery
        → Code changes to db_manager.py, strategy_validator.py, strategy_discovery.py
        → Re-run: discovery --all, then validator --all (3 instruments)
        → Outcome: Every validated strategy has DST split visible

Step 2: Track B — Volume analysis (parallel with Step 1 if possible)
        → Run research_volume_dst_analysis.py
        → Review results, make GO/NO-GO on new sessions

Step 3: Track C — Add new sessions (conditional)
        → Only if Step 2 says GO
        → Add 0930/1900 to pipeline, rebuild features+outcomes, run discovery+validation

Step 4: Update documentation
        → TRADING_RULES.md session playbooks with clean split numbers
        → CLAUDE.md remediation status → DONE
        → ROADMAP.md steps marked complete
```

---

## Decision: Should we REJECT strategies based on DST split?

**Current recommendation: NO — INFO only for now.**

Reason: The strategy revalidation already showed NO validated strategies are broken. All 10 red flags are MES 0900 experimental (never in production). Making DST a rejection gate would not change current production but would add complexity.

**Future option:** If a strategy passes all 6 phases but has `dst_verdict = WINTER-ONLY` or `SUMMER-ONLY`, flag it in validation_notes with a WARNING but still promote. The live_config tier system (CORE/HOT/REGIME) already handles conditional trading.

---

## What Does NOT Need Rebuilding (Confirmed)

| Component | Status | Reason |
|-----------|--------|--------|
| `bars_1m` | CLEAN | Raw UTC data |
| `bars_5m` | CLEAN | Derived from bars_1m |
| `daily_features` | CLEAN | Correct ORB at stated time |
| `orb_outcomes` | CLEAN | Correct outcomes for stated time |
| `experimental_strategies` | NEEDS DST COLUMNS | Re-run discovery with DST tagging |
| `validated_setups` | NEEDS DST COLUMNS | Re-run validation with DST split |
| `edge_families` | RE-RUN AFTER | Downstream of validated_setups |
