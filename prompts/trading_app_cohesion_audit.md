# Trading App Cohesion Audit — Reusable Prompt

Paste this into a new Claude session (or use as a custom instruction) to get a full logical-cohesion audit of trading_app.

---

## THE PROMPT

```
You are auditing the `trading_app/` package for logical cohesion, completeness, and user clarity. The goal: if a trader opens this system, can they understand what's going on, clearly and concisely? Does every piece connect? Are there dead ends, orphan modules, missing links, or confusing overlaps?

**IMPORTANT: This is a DESIGN audit, not an implementation task. Do NOT write code. Produce findings only.**

## Audit Scope

Audit the full `trading_app/` package and its `scripts/` entry points. The system is a multi-instrument futures ORB (Opening Range Breakout) backtesting + live trading pipeline for MGC, MNQ, MES.

## What You're Looking For

Run each check below. For each, report: PASS (clean), WARN (concern but functional), or FAIL (broken/confusing). Include evidence (file:line or specific example) for every WARN and FAIL.

### 1. FLOW CONTINUITY — "Can I trace the full lifecycle?"

Trace the complete path a strategy takes from raw data to live execution. Verify each handoff:

- [ ] `pipeline/` → `bars_1m` → `daily_features` → data is clean and complete
- [ ] `outcome_builder.py` reads canonical layers, writes `orb_outcomes` — no gaps
- [ ] `strategy_discovery.py` reads `orb_outcomes`, writes `experimental_strategies` — all filter_types from `config.py` are covered
- [ ] `strategy_validator.py` reads `experimental_strategies`, promotes to `validated_setups` — all 7 phases wired correctly
- [ ] `strategy_fitness.py` reads `validated_setups`, produces FIT/WATCH/DECAY/STALE — thresholds match docs
- [ ] `live_config.py` reads `validated_setups` + fitness → `LIVE_PORTFOLIO` — selection logic is sound
- [ ] `portfolio.py` reads validated strategies → position sizing → `Portfolio` object — Kelly math correct
- [ ] `execution_engine.py` receives Portfolio, processes bars → state machine works (CONFIRMING → ARMED → ENTERED → EXITED)
- [ ] `session_orchestrator.py` (live/) ties execution_engine + broker + monitoring together — all wired
- [ ] `pre_session_check.py` gates live trading — actually blocks on failure

**Key question:** Is there ANY point where the chain breaks, where output format doesn't match expected input, or where a module reads from the wrong source?

### 2. CONFIG CONSISTENCY — "Does everything agree on the rules?"

- [ ] Entry models (E1/E2/E3) defined in `config.py` — are they used identically in `outcome_builder`, `entry_rules`, `execution_engine`, and `strategy_discovery`?
- [ ] Filter types in `config.py` ALL_FILTERS — does every module that applies filters use this exact list? Any module using a hardcoded subset?
- [ ] Session definitions — does `pipeline/dst.py` SESSION_CATALOG match what `config.py`, `live_config.py`, and `session_orchestrator.py` expect?
- [ ] Cost model — is `pipeline/cost_model.COST_SPECS` the single source everywhere? Any hardcoded costs?
- [ ] RR targets, confirm_bars, orb_minutes — are the valid ranges consistent across discovery, validation, and execution?
- [ ] Prop firm rules in `prop_profiles.py` — do `risk_manager.py`, `circuit_breaker.py`, and `lane_allocator.py` all read from this source?

**Key question:** If I change a config value in ONE place, does it propagate everywhere, or are there shadow copies that would silently diverge?

### 3. MODULE ROLE CLARITY — "What does each piece do, and ONLY that?"

For each module, check:
- [ ] Does it have a clear, single responsibility?
- [ ] Does its name accurately describe what it does?
- [ ] Is there overlap with another module? (e.g., `portfolio.py` vs `prop_portfolio.py` vs `rolling_portfolio.py` — what's the difference? Would a new user know?)
- [ ] Are there modules that exist but nothing calls them? (orphans)
- [ ] Are there modules in `ml/` that are dead but not marked clearly enough?

Flag specifically:
- `portfolio.py` vs `prop_portfolio.py` vs `rolling_portfolio.py` — role boundaries
- `live_config.py` vs `portfolio.py` — who decides what to trade?
- `strategy_fitness.py` vs `rolling_portfolio.py` — overlapping monitoring?
- `scoring.py` vs `strategy_discovery.py` — who ranks strategies?
- `execution_spec.py` vs `config.py` — who defines execution rules?
- `market_state.py` vs `live/live_market_state.py` — same thing?
- `mcp_server.py` (root) vs `ai/mcp_server.py` — two MCP servers?

### 4. ENTRY POINT CLARITY — "How do I use this thing?"

A trader sitting down should be able to answer: "What do I run, and in what order?"

- [ ] Is there a single, obvious entry point for each workflow? (backtest, validate, go-live, monitor)
- [ ] Are CLI arguments consistent across modules? (e.g., `--instrument` everywhere, or sometimes `--symbol`?)
- [ ] Do error messages tell the user what to do, not just what went wrong?
- [ ] Is the discovery → validation → deployment chain documented in a way a trader (not a dev) would follow?
- [ ] `scripts/run_live_session.py` — does it clearly explain the three modes (signal-only, demo, live)?

### 5. ERROR HANDLING & FAIL-SAFETY — "What happens when things go wrong?"

- [ ] If the DB is missing or stale, does `pre_session_check.py` catch it?
- [ ] If a broker connection drops mid-session, does `session_orchestrator.py` handle it? (reconnect? halt? alert?)
- [ ] If a strategy goes DECAY/STALE, does `live_config.py` automatically remove it?
- [ ] If `circuit_breaker.py` triggers, does it actually halt all new entries? Can it be overridden accidentally?
- [ ] Are there any bare `except Exception` blocks that swallow errors silently?
- [ ] Does `instance_lock.py` prevent two bots on the same account reliably?

### 6. MONITORING & OBSERVABILITY — "Can I tell what's happening right now?"

- [ ] `bot_state.json` — is it written atomically? What happens if the dashboard reads mid-write?
- [ ] `performance_monitor.py` — does it track what a trader actually cares about? (P&L, drawdown, trade count, win rate)
- [ ] `notifications.py` — what events trigger alerts? Are the critical ones covered? (entry, exit, halt, error, reconnect)
- [ ] `cusum_monitor.py` / `sprt_monitor.py` — are these wired into anything, or standalone research tools?
- [ ] `weekly_review.py` — does it pull from canonical layers or derived?
- [ ] Is there a way to see "what happened last night" in one command?

### 7. DEAD CODE & TECHNICAL DEBT — "What can I delete?"

- [ ] `ml/` — is it clearly fenced off? Can any import path accidentally pull it in?
- [ ] `nested/` — is this active research or abandoned?
- [ ] `regime/` — same question
- [ ] `analysis/asia_session_analyzer.py` — used by anything?
- [ ] Any scripts in `scripts/` that reference deprecated tables or removed features?
- [ ] Any TODO/FIXME/HACK comments that indicate known broken things?

### 8. DOCUMENTATION ↔ REALITY — "Do the docs match the code?"

- [ ] `TRADING_RULES.md` — do the entry models, sessions, and filters described there match `config.py` exactly?
- [ ] `docs/ARCHITECTURE.md` — does the module dependency graph match reality?
- [ ] `docs/STRATEGY_BLUEPRINT.md` — does the NO-GO registry match what's actually disabled?
- [ ] MCP server templates in `mcp_server.py` — do all 18 templates return correct data?
- [ ] `prop_profiles.py` ACCOUNT_PROFILES — do they reflect current account state?

## Output Format

Produce a structured report with:

1. **Executive Summary** (3-5 sentences): Overall cohesion grade (A-F) and the top 3 issues.

2. **Flow Map**: ASCII diagram showing the actual data flow you traced, with any breaks marked.

3. **Findings Table**:
   | # | Category | Severity | Finding | Evidence | Suggested Fix |
   Each finding gets one row. Sort by severity (FAIL → WARN → PASS).

4. **Orphan Inventory**: List of modules that nothing calls or that duplicate another module's job.

5. **"If I Were a Trader" Narrative**: Write 2 paragraphs from a trader's perspective describing what it's like to open this system fresh. What's clear? What's confusing? What's missing?

6. **Priority Fix List**: Top 5 things to fix, ordered by impact on user clarity.

## Constraints

- Query the database (`gold.db`) where needed — don't guess table contents.
- Read actual imports (`grep -r "from trading_app" | sort`) to verify dependency claims.
- If a module exists but has no callers, say so with evidence.
- Do NOT recommend adding ML features (ML is dead per project rules).
- Do NOT recommend changes to the pipeline/ package (out of scope).
- Respect the authority hierarchy: Code > CLAUDE.md > TRADING_RULES.md > RESEARCH_RULES.md.
```

---

## Usage Notes

- Paste the prompt above into a fresh Claude session with this repo mounted.
- The audit takes ~10-15 minutes of tool use (file reads, greps, DB queries).
- Re-run after any major refactor to catch regressions.
- Pair with `python pipeline/check_drift.py` for the mechanical side — this prompt covers the human/logical side.
