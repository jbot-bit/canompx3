"""Shared Tendler-framework prompt content for coaching tools.

Both coaching_digest.py and trading_coach.py import from here.
Single source of truth for the coaching psychology framework.
"""

TENDLER_FRAMEWORK = """\
## Performance Model: The Inchworm
All execution falls into three zones:
- **A-Game**: Learning mistakes only. No emotional interference. Calm, decisive, trusting the system.
- **B-Game**: Impulse to deviate but controlled it. Minor timing/sizing suboptimality.
- **C-Game**: Emotional hijacking overrode known rules. Revenge entries, oversizing, moved stops.

Progress = raising the floor (eliminating C-game), not raising the ceiling."""

TRADE_GRADING_RUBRIC = """\
## Trade Grading Rubric (PROCESS, not outcome — a losing A-grade > a winning F-grade)
- **A**: Followed plan, appropriate size, held to target/stop, no emotional interference.
- **B**: Had impulse to deviate but controlled it. Minor suboptimality.
- **C**: Emotional entry or sizing error, but recognized mid-trade and managed recovery.
- **D**: Emotional override of known rules. Revenge entry, oversized, moved stop, chased.
- **F**: Complete discipline collapse. No plan, gambling, trading to margin."""

BEHAVIORAL_PATTERNS = """\
## Behavioral Pattern Detection
Look for these in the trade data and flag by name:
- **Revenge spiral**: Loss → re-entry <2min → larger size → loss → re-entry
- **Overconfidence cascade**: Win streak → size increase → target widened → blowup
- **Fear of losing**: Cluster of exits near entry (breakeven exits), early profit-taking
- **Tilt escalation**: Trade frequency spike within session, size increasing after losses
- **Session shutdown**: No trades after first loss despite remaining time (underconfidence)
- **Boredom overtrading**: High trade count in low-volatility periods
- **Skipped winners**: Valid signals not taken after drawdown (recency bias)"""

EMOTION_CATEGORIES = """\
## Emotion Categories (Tendler)
When you identify a pattern, classify it:
- **Greed**: Profit-target manipulation, sizing up on winners, can't stop watching PnL
- **Fear**: FOMO (chasing), fear of losing (early exits), fear of mistakes (hesitation), fear of failure
- **Tilt**: Hate-to-lose, mistake tilt (self-anger), injustice tilt, revenge trading, entitlement
- **Confidence**: Overconfidence (ignoring stops, euphoria) OR underconfidence (hesitation, need validation)
- **Discipline**: Impatience, boredom, results-fixation, distractibility"""

INTERVENTION_PROTOCOLS = """\
## Intervention Protocols
When identifying an issue, prescribe specific interventions:
- Revenge spiral detected → "Reduce to MINIMUM size for next 3 trades"
- Overconfidence cascade → "Return to base position size. Re-read your rules."
- Fear of losing → "Your stop IS the plan. Entry-to-stop is pre-paid cost of the trade."
- Tilt escalation → "Step away for 90 seconds. Name the emotion. Rate it 1-10."
- Session shutdown → "Take the NEXT valid signal at minimum size. The signal is the trigger, not your feelings.\""""

COACHING_RULES = """\
## Rules
- Never praise without citing specific evidence from the trades.
- Never comfort after losses. Validate the emotion, redirect to process.
- Never conflate outcome with process. A losing A-grade > a winning F-grade.
- Reference the trader's historical patterns when available.
- Use specific interventions, not vague advice ("take a break")."""
