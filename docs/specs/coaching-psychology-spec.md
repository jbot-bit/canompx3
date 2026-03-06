# AI Trading Coach — Psychology Grounding Spec

> **Authority:** This spec defines the psychological framework for all AI coaching prompts.
> When coaching_digest.py or trading_coach.py system prompts conflict with this spec, this spec wins.

**Sources:** Tendler (Mental Game of Trading, 2021), Douglas (Trading in the Zone, 2000), Elder (New Trading for a Living, 2014), Van Tharp (Trade Your Way to Financial Freedom, 2006), Schwager (Market Wizards, 2012), Faith (Way of the Turtle, 2007)

---

## 1. Performance Model: The Inchworm (Tendler)

All trading performance exists on a bell curve with three zones:

| Zone | Label | Mistake Type | Cause |
|------|-------|-------------|-------|
| **C-Game** | Obvious mistakes | Emotional hijacking overrides known rules | Performance flaws — emotions too intense or energy too low |
| **B-Game** | Marginal mistakes | Impulse to make C-game error but retained control | Mix of technical weakness + emotional noise |
| **A-Game** | Learning mistakes | Unavoidable gaps in tactical knowledge | Knowledge gaps only — no emotional component |

**Key insight:** "You can't escape the gravitational force of your C-game by focusing solely on improving your trading knowledge. If you improve only on the technical mistakes, your performance flaws will continue to generate the same level of excess emotion. You'll make different, but still obvious, mistakes." — Tendler

**Progress = raising the floor (eliminating C-game), which frees bandwidth to stretch A-game.**

The AI coach grades trades on this model, NOT binary right/wrong.

---

## 2. Five Emotion Categories

### 2a. GREED
**Nature:** Desire for profit taken to an extreme that overrides strategy.
**Root cause:** Weakness in confidence — urgency to prove oneself, especially after drawdowns.

**Data-detectable signs:**
- Position size increase during winning streaks
- Profit target widened (order modifications)
- Manual management instead of letting trades run
- Taking marginal trades after break-even/losing periods

**Escalation (10 levels):**
- 1-3: Checking PnL, calculating how much more needed, feeling excited
- 4-6: Wondering if should trail stop, staying at screen unnecessarily, irritable
- 7-8: Can't stop thinking about trade, give up trading plan but maintain stops
- 9-10: Believe market is controllable, give up all control, only focused on money NOW

### 2b. FEAR (4 sub-types)

**FOMO:** Chasing entries after seeing others profit. Taking trades outside strategy. Antsy, nervous stomach.

**Fear of Losing:** Need for trade to be green immediately. Moving stop to breakeven prematurely. Early profit-taking. Sizing down to reduce pain exposure.
- Root: "Every loss feels like a step backward. One trade outcome is treated as representative of the whole year."

**Fear of Mistakes:** Hesitating and second-guessing. Binary coding of performance (correct vs incorrect). Over-analysis before entry.
- Root: Conflating mistakes with personal failure.

**Fear of Failure:** Identity threat — "if I fail at trading, I'm a failure as a person."

**Data-detectable signs:**
- Cluster of exits near entry price (breakeven exits)
- Early exits on winners (before target)
- Reduced position sizes after losses
- Post-loss session shutdown (no further trades despite remaining time)
- No trades when setups present (requires setup detection)

### 2c. TILT / ANGER (5 sub-types)

**Hating to Lose:** Competitive drive makes losses feel unacceptable. Anger AT the loss.

**Mistake Tilt:** Self-directed anger for failing to execute known rules. Compounds the original error.

**Injustice Tilt:** Perceived unfairness. Two coding errors create the illusion:
1. Attributing mistakes to bad luck ("variance screwing me")
2. Attributing good luck to skill ("I'm good")
- "Traders remember bad luck more than good luck. You're really skilled at spotting bad luck and terrible at spotting good luck." — Tendler

**Revenge Trading:** Impulsive re-entry after losses. Mixture of injustice tilt + mistake tilt + need to be right.

**Entitlement Tilt:** Market "owes" you profits because of preparation/skill/past performance.

**Data-detectable signs:**
- Rapid re-entry after stop-out (< 2 min = strong signal)
- Same-instrument re-entry after loss (consecutive entries)
- Position size increase after losses (tilt level 6+)
- Stop moved or removed entirely
- Trade frequency spike within session
- Session PnL trajectory: loss then rapid-fire cluster
- Trading to margin call (tilt level 10)

**Escalation (10 levels):**
- 1-3: Mouse gripping, tension in head, mediocre trade taken and stopped
- 4-6: Increase size to make back losses, internal fight ("stop!" / "I can make this back"), fixate on price not analysis
- 7-10: Don't care, trade to margin call, cursing, smashing things, wanting to quit

**Accumulated tilt:** Some traders carry residual anger from previous sessions. Baseline tilt at session open is already elevated.

### 2d. CONFIDENCE (bidirectional)

Confidence fails in BOTH directions — overconfidence and underconfidence.

**Overconfidence signs:** Bigger sizes, ignoring stops, more trades than normal, euphoria, thinking about money not execution.

**Underconfidence signs:** Hesitation on entries, only taking "perfect" trades, need for external validation, questioning strategy viability.

**Sub-patterns:**
- **Perfectionism:** Expectation of 100% correct execution. Every deviation triggers fear or tilt.
- **Desperation:** Extreme underconfidence + financial pressure. NEED to make money overrides strategy.
- **Hope/Wishing:** Passive substitutes for decision-making. Holding losers because "it might come back."

**Cognitive biases affecting confidence:**
- Illusion of control (overestimating influence on outcomes)
- Black-and-white thinking (genius when winning, fraud when losing)
- Gambler's fallacy ("I'm due for a win")
- Recency effect (overweighting recent results)

### 2e. DISCIPLINE (6 sub-types)

**Impatience:** Micro (can't wait for setup) and macro (can't wait for strategy to prove out).

**Boredom:** Slow markets lead to attention drift. Miss setups when they appear.

**Results-Orientation:** Every trade evaluated by PnL, not by whether it followed the plan.

**Distractibility:** External (social media, news) and internal (mind wandering).

**Laziness:** Insufficient preparation. Skipping pre-market analysis, not reviewing trades.

**Procrastination:** Delaying review of losing trades, not fixing known problems.

---

## 3. The Mapping System (Real-Time Detection)

### What to Map (per emotional event)
1. **Triggers** — what initiated the cascade
2. **Thoughts** — internal dialogue
3. **Emotions** — what you feel
4. **Behaviors** — physical/observable actions
5. **Actions** — trading actions taken
6. **Decision-making changes** — how process shifted
7. **Perception changes** — how you see market/opportunities differently
8. **Trading mistakes** — actual execution errors

### Severity Scale
Rate both emotional/mental level AND technical/tactical level on 1-10 independently. Minimum 3 severity levels mapped.

### AI Coach Application
The coach should:
1. Detect which emotion category is active from trade data patterns
2. Estimate severity level from the intensity of data signals
3. Name the specific sub-type (not generic "you tilted")
4. Reference the trader's own historical pattern at that level
5. Provide the pre-defined correction for that level

---

## 4. Data-Detectable Composite Patterns

| Pattern | Likely Diagnosis | Severity |
|---------|-----------------|----------|
| Loss -> re-entry <2min -> larger size -> loss -> re-entry | Revenge trading spiral | Tilt 6+ |
| Win streak -> larger size -> target widened -> big loss | Greed -> Overconfidence -> Tilt cascade | Multi-stage |
| Loss -> no trades for extended period -> forced entry | Fear of losing -> Boredom -> Impatience | Fear 5+ |
| Multiple small wins taken early -> missed full targets | Fear of losing / Lack of confidence | Fear 7-8 |
| Trading outside hours/strategy after drawdown | Desperation / Discipline collapse | Critical |
| Session PnL up -> increasing size -> blowup | Overconfidence cascade | Confidence 8+ |
| Consistent early exits despite positive expectancy | Fear of losing pattern | Fear 3-5 |

---

## 5. Trade Grading Rubric (replaces generic A-F)

| Grade | Definition | Criteria |
|-------|-----------|----------|
| **A** | A-game execution | Followed plan, appropriate size, held to target/stop, no emotional interference |
| **B** | B-game execution | Had impulse to deviate but controlled it. Minor timing/sizing suboptimality. |
| **C** | C-game entry, managed recovery | Emotional entry or sizing error, but recognized and managed position correctly |
| **D** | C-game throughout | Emotional override of known rules. Revenge entry, oversized, moved stop, chased. |
| **F** | Discipline collapse | Complete emotional hijacking. Trading to margin, no plan, gambling. |

**Key:** Grade reflects PROCESS, not outcome. An A-grade trade can lose money. An F-grade trade can make money.

---

## 6. Intervention Framework

### Yerkes-Dodson Threshold
Performance improves with arousal up to a threshold, then declines sharply. "The further you pass your emotional threshold, the more knowledge you lose. Things that are brand new are the first to go." — Tendler

### Per-Level Interventions
- **Level 1-3:** Awareness only. Name the emotion. Continue trading with heightened self-monitoring.
- **Level 4-5:** Reduce size. Extend time between trades. Review the map.
- **Level 6-7:** Consider stopping for 15-30 min. Review trading plan. Do NOT make new entries.
- **Level 8-9:** Stop trading. Walk away. Do NOT re-enter this session.
- **Level 10:** Emergency stop. Shut platform. The session is over.

### Joe's Process-Focus Tool (Anti-Revenge)
Track "strategy compliance rate" per batch of N trades. Only surface PnL stats at batch boundaries, not per-trade. This prevents outcome-fixation from triggering emotional cascades.

---

## 7. Foundational Truths (Douglas)

Five truths the trader must internalize:

1. **Anything can happen** — any single trade outcome is random
2. **You don't need to know what happens next** to make money
3. **There is a random distribution** between wins and losses for any given set of variables
4. **An edge is nothing more than a higher probability** of one thing happening over another
5. **Every moment in the market is unique** — patterns repeat, but each instance is independent

The AI coach should reference these when a trader exhibits:
- Outcome fixation (truth 1, 3)
- Prediction addiction (truth 2)
- Single-trade significance (truth 4)
- Pattern-matching delusion (truth 5)

---

## 7b. Practitioner Principles (Market Wizards + Turtle + Livermore)

12 named principles extracted from elite traders, in priority order for system traders:

| # | Principle | Source | Core Insight |
|---|-----------|--------|-------------|
| 1 | DEFENSE FIRST | Jones | Think about what you can lose before what you can make |
| 2 | INDIFFERENCE THRESHOLD | Hite | Size so individual outcomes don't matter emotionally |
| 3 | BET QUALITY INDEPENDENCE | Hite | Judge trades by process, not outcome. Winning system violations are BAD trades |
| 4 | SITTING TIGHT | Livermore | Money is in the holding, not the trading |
| 5 | RIGHTNESS IS IRRELEVANT | Faith | Win rate doesn't measure quality. 10 losses while following system = perfect |
| 6 | THE SKIPPED TRADE | Hite/Faith | The trade you skip is the one that pays (recency bias) |
| 7 | SYSTEM-PERSONALITY FIT | Van Tharp | Best system you won't follow is worse than mediocre system you will |
| 8 | EGO SEPARATION | Schwartz | Needing to be right is the most expensive emotion |
| 9 | PROCESS AMNESIA | Faith | Reset after each trade. Previous outcome = zero predictive value |
| 10 | THE MONEY PARADOX | Faith | Caring about money degrades performance |
| 11 | SIZE KILLS | Livermore | Wrong size turns correct direction into bankruptcy |
| 12 | COMPULSION TO TRADE | Livermore | No signal = no trade. Boredom is expensive |

### Additional Detectable Patterns from Practitioners

| Pattern | Detection Method | Root Cause |
|---------|-----------------|------------|
| Trigger Freeze | Signal-to-entry ratio drops after drawdowns | Loss aversion + recency bias |
| Skipped Winners | P&L of system signals vs actual entries diverges | Recency bias + outcome bias |
| Loss Holding | Avg losing duration > avg winning duration | Sunk cost + ego attachment |
| Non-Signal Trading | Trades outside valid signal periods | Boredom + compulsion |
| Lifestyle Sizing | Risk-per-trade creep during winning periods | Spending pressure |
| Anchor Exit | Exit prices cluster near entry prices | Anchoring bias |
| Circuit Breaker Violation | Monthly loss exceeds preset threshold | Discipline framework absent |

### Key Quote for System Traders
"People don't change. That is why this whole game works." — Larry Hite. The edge in systematic trading exists because cognitive biases are permanent features of human decision-making. Your system exploits these biases in others. Your discipline prevents them in yourself.

---

## 8. Risk Management as Psychology (Elder + Jones + Hite)

### The Indifference Threshold (Hite)
"Never risk more than 1% of total equity on any trade. By only risking 1%, I am indifferent to any individual trade." Risk per trade should be small enough that the outcome is emotionally irrelevant. Our system uses 1-2% per trade via R-multiples.

### The 2% Rule (Elder)
Never risk more than 2% of account on a single trade. This is psychology disguised as math — it prevents any single trade from having emotional significance.

### The 6% Rule / Monthly Circuit Breaker (Elder + Jones)
If account is down 6% for the month, stop trading until next month. Jones: "I want to make sure that I never have a double-digit loss in any month."

### Drawdown Discipline Protocol (synthesized from all sources)
1. **Before any trade:** Know stop, size, max loss for period. (Kovner, Jones, Hite)
2. **During drawdown:** Reduce size. Shift focus from P&L to system adherence. Survive, don't recover fast. (Schwartz: tactical withdrawal)
3. **After consecutive losses:** Do NOT skip next signal. Do NOT increase size. (Faith: 17 cocoa losses before the big winner)
4. **At circuit breaker:** Stop trading. No exceptions. Review: system losses (acceptable) vs behavioral losses (fixable). (Jones)
5. **Recovery:** Scale back in with minimum size. Win back confidence before size. (Schwartz: "pull back, lick my wounds, come back")
6. **Post-recovery:** Resist "make up for lost time" oversizing. The Winner's Curse follows every recovery. (Livermore)

### AI Coach Application
- Track per-trade risk as % of account
- Track monthly drawdown %
- Flag when approaching 2% per-trade or 6% monthly
- These are hard stops, not suggestions
- After 3+ consecutive losses: coach must reference the Drawdown Protocol, not just say "take a break"

---

## 9. Profile Model (what the AI tracks per trader)

```json
{
  "inchworm": {
    "c_game_patterns": ["revenge re-entry", "size escalation after loss"],
    "b_game_patterns": ["impulse to chase but waited", "considered moving stop but held"],
    "a_game_indicators": ["calm execution", "losses don't register emotionally"]
  },
  "primary_emotion": "tilt",
  "tilt_subtype": "revenge_trading",
  "fear_subtype": null,
  "greed_subtype": null,
  "confidence_direction": "cycles_both",
  "discipline_weaknesses": ["impatience", "results_orientation"],
  "escalation_speed": "fast",
  "recovery_pattern": "slow",
  "accumulated_tilt_risk": "medium",
  "session_tendencies": {
    "early_session": "overconfident after overnight prep",
    "mid_session": "impatient if no setups",
    "late_session": "desperate if behind"
  },
  "known_triggers": ["2 consecutive losses", "missed trade that would have been winner"],
  "effective_interventions": ["batch compliance tracking", "15min break after 2nd loss"],
  "compliance_rate_baseline": 0.72
}
```

---

## 10. What the AI Coach MUST NOT Do

1. **Never praise without evidence.** "Great job!" means nothing without citing specific process execution.
2. **Never comfort after losses.** Validate the emotion, then redirect to process. "The loss hurts. Let's look at whether the entry followed your rules."
3. **Never diagnose without data.** If the data doesn't show a pattern, say "I don't have enough data to identify this yet."
4. **Never suggest the trader is "fixed."** Psychology is ongoing. The Inchworm never stops moving.
5. **Never conflate outcome with process.** A losing A-grade trade is better than a winning F-grade trade.
6. **Never use generic advice.** "Take a break" is useless. "You've hit level 6 on your tilt map — your historical pattern shows size escalation is next. Reduce to minimum size or stop for 15 minutes" is actionable.
7. **Never ignore accumulated tilt.** If the previous 3 sessions were negative, today's session starts with elevated baseline.

---

## 11. Gap Analysis — Current Project State

### GAP A: coaching_digest.py — Grading Rubric Undefined
- SYSTEM_PROMPT says "grade A/B/C/D/F" but gives zero criteria
- Claude invents grading logic each time — inconsistent across sessions
- No behavioral pattern taxonomy — coach guesses at "revenge trading" from vibes
- **Fix:** Replace SYSTEM_PROMPT with framework-grounded prompt using Section 5 rubric + Section 2 taxonomy

### GAP B: trading_coach.py — No Intervention Protocols
- Interactive coach can answer questions but has zero tilt-aware interventions
- Missing: "You're down 3R in 4 trades. Revenge tilt is active. Historically when you're down >2R, your next 3 trades average -0.8R."
- Missing: Implementation intentions ("If session PnL hits -2R, then I will...")
- **Fix:** Inject Inchworm model + trader's mapped patterns + intervention protocols into system prompt

### GAP C: discipline_data.py — Triggers Not Grounded
Current `DEVIATION_TRIGGERS` are arbitrary:
```python
("chart_pattern", "narrative", "felt_reversal", "chasing_loss", "fomo_late", "sized_up", "other")
```
Should map to Tendler's 5 categories: greed, fear (FOMO/loss/mistakes/failure), tilt (hate-loss/mistake/injustice/revenge/entitlement), confidence (over/under), discipline (impatience/boredom/results-fixation/distraction).

Missing triggers: overconfidence, anchoring, escalation_of_commitment, hope_holding.
Missing quantification: cost per trigger type, frequency per trigger, trigger-session correlations.

### GAP D: discipline.py — Pre-Session Priming is Shallow
- No process-vs-outcome framing ("Today: execute 80% of signals, not make $X")
- No worst-case rehearsal (reduces anxiety per Pennebaker 2011)
- No session time limit pre-commitment
- Cooling is reactive only — no early warning for building tilt (trade frequency spike, size drift, running P&L trajectory)

### GAP E: trade_matcher.py — No Behavioral Metrics
Computes: hold_seconds, direction, VWAP entry, fees, pnl.
Missing:
- Re-entry speed (revenge detection)
- Size trajectory (escalation detection)
- Consecutive loss count at entry
- Session P&L at entry
- Same-instrument re-entry flag
- Deviation cost (manual vs system signal comparison)

### GAP F: session_orchestrator.py — Behavioral Circuit Breakers Missing
Has: max_daily_loss_r, max concurrent positions, CUSUM drift.
Missing:
- Revenge trading guard (2+ consecutive losses -> require pause)
- Overconfidence guard (2+ consecutive wins -> reduce size)
- Session fatigue (>8 trades and <40% win rate -> skip remaining)
- Max re-entries per ORB
- Max deviation count per session

### GAP G: trader_profile.json — Emotional Profile is Empty
```json
"emotional_profile": {"tilt_indicators": [], "calm_indicators": []}
```
Should contain: tilt_type tracking with frequency/cost/interventions_tried/effectiveness, trigger->outcome conditioning (P(revenge|3+losses)), session tendencies per ORB session, personality traits (loss-averse vs gain-seeking).

---

## Implementation Priority

### Phase 1: Prompt Grounding (no code changes)
1. Rewrite `coaching_digest.py` SYSTEM_PROMPT using this spec
2. Rewrite `trading_coach.py` build_chat_system_prompt() with Inchworm + interventions
3. Expand trader_profile.json schema to Section 9 model

### Phase 2: Behavioral Metrics (trade_matcher changes)
4. Add re-entry speed, size trajectory, consecutive losses, session P&L at entry
5. Add composite pattern detection (revenge spiral, overconfidence cascade, etc.)

### Phase 3: Discipline Integration (UI + discipline_data changes)
6. Replace DEVIATION_TRIGGERS with Tendler taxonomy
7. Add pre-session process framing + session limits
8. Add early-warning tilt detection (trade frequency, size drift, P&L trajectory)

### Phase 4: Live Trading Guards (session_orchestrator changes)
9. Behavioral circuit breakers (revenge guard, overconfidence guard, fatigue guard)
10. Real-time tilt state inference from trade sequence
11. Affect labeling integration (1-tap emotional state before next trade when tilt detected)

### Behavioral Metrics to Add to Trade Records
- `time_since_last_exit_seconds`: Time between previous exit and this entry (revenge detection)
- `size_vs_baseline_pct`: Position size relative to trader's normal (escalation detection)
- `session_pnl_at_entry`: Running session P&L when this trade was entered (desperation detection)
- `consecutive_losses_at_entry`: How many losses in a row before this trade
- `same_instrument_reentry`: Boolean — did trader just lose on this same instrument?
