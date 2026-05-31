<!-- VERBATIM SNAPSHOT — DO NOT EDIT THE QUOTED BODY -->
Source URL: https://help.tradeify.co/en/articles/10495897-rules-trailing-max-drawdowns
Article ID: 10495897
Scraped: 2026-05-31
Updated (per article): April 10, 2026
Page title: Rules: Trailing Max Drawdowns | Tradeify Help Center
Fetch: curl -sL -A "<browser UA>" | html2text (structural tags->\n, no word changed)
Content images: none (all rules captured as HTML text)
==============================================================================

- Rules: Trailing Max Drawdowns | Tradeify Help Center
- Skip to main content

Tradeify HomeDiscordEnglishEnglish

Tradeify HomeDiscordEnglishEnglish

Search for articles...

- All Collections

- Accounts & Rules

- Trading Rules

- Rules: Trailing Max Drawdowns

Rules: Trailing Max Drawdowns
Dynamic account protection mechanism that adjusts as your profits grow.

April 10, 2026

Table of contents

Max Trailing Drawdown

Summary: The Max Trailing Drawdown is your account's maximum allowed decline from its highest balance (high water mark). Hitting this limit FAILS your account permanently. Unlike the Daily Loss Limit, there is no recovery from a drawdown breach.

Applies to: All Tradeify accounts (Growth, Select, and Lightning). All accounts use End-of-Day (EOD) trailing drawdown.

Critical Concept: Hard Breach

The Max Trailing Drawdown is a "hard breach" - if your account balance drops to or below your drawdown limit at ANY point, your account fails immediately. This is different from the Daily Loss Limit (soft breach) which only pauses trading.

End-of-Day (EOD) Trailing Drawdown

All Tradeify accounts use EOD Trailing Drawdown. Your drawdown limit only updates at the end of each trading day, not during the day.

How it works:

- Your drawdown limit trails your highest-ever end-of-day balance (the "high water mark") - it only moves up, never down

- If your EOD balance sets a new all-time high, the drawdown limit moves up accordingly

- If your EOD balance is lower than a previous day, the drawdown limit stays where it is - it does NOT recalculate from the lower balance

Example:

- Starting balance: $100,000, Drawdown: $3,500, Limit: $96,500

- Day 1: End with $103,000 → New limit: $99,500

- Day 2: End with $101,000 → Limit stays at $99,500 (only moves up, never down)

CRITICAL: Even though EOD drawdown only UPDATES at end of day, it is ENFORCED in real-time. If your balance hits the drawdown limit during trading, your account fails immediately - even if you might have recovered by end of day.

When Does Drawdown Lock?

On Sim Funded accounts (not Evaluations), your drawdown locks in place once you profit beyond the drawdown amount by $100. After locking, the drawdown limit becomes a fixed floor that never moves up again.

Account Size

Growth

Lightning

Select Flex

Select Daily

25K

$26,100

$26,100

$26,100

$26,100

50K

$52,100

$52,100

$52,100

$52,100

100K

$103,600

$104,100

$103,100

$102,600

150K

$155,100

$155,350

$154,600

$153,600

The table shows the EOD balance required to trigger the drawdown lock. Once locked, the drawdown floor becomes fixed at $100 above starting balance (e.g., $50,100 for 50K, $100,100 for 100K, $150,100 for 150K) and never moves up again.

Example: A 50K Growth account with $2,000 drawdown locks when EOD balance reaches $52,100. The drawdown floor becomes $50,100 and never moves up again. This same mechanism applies to all funded account types including Select Flex and Select Daily.

Frequently Asked Questions

Q: Can I trade aggressively during the day knowing EOD drawdown only updates at end of day?

A: No. While EOD drawdown only UPDATES at end of day, it is ENFORCED in real-time. If your balance hits the current drawdown limit during trading, your account fails immediately.

Q: Does drawdown lock on Evaluation accounts?

A: No. Drawdown only locks on Sim Funded accounts. Evaluation accounts do not have drawdown locking.

Q: Can I recover from hitting my drawdown limit?

A: No. Hitting the Max Trailing Drawdown is a hard breach - your account fails permanently. This is different from the Daily Loss Limit which only pauses trading.

Q: How do I check my current drawdown limit?

A: Your drawdown limit is not displayed on the Tradeify dashboard. You need to check it directly in your trading platform:

- Tradovate: Open the Accounts widget and add the "Dis Drawdown Net Liq" column

- NinjaTrader: Go to Control Center → Accounts and add the "Trailing Max Drawdown" column

- Rithmic: Check your drawdown in the Tradeify dashboard

- Tradesea: Check your drawdown in the Tradeify dashboard or Tradesea's account panel

For step-by-step setup instructions with screenshots, see our Drawdown Widget article. Monitor your drawdown closely, especially after profitable days when the limit has moved up.

Q: How does drawdown work on Select accounts?

A: Both Select Flex and Select Daily funded accounts use EOD trailing drawdown with the same locking mechanism as Growth and Lightning accounts. When your EOD balance exceeds your drawdown amount by $100, the drawdown locks at $100 above your starting balance. For example, a 50K Select account locks when EOD balance reaches $52,100, and the drawdown floor becomes $50,100 permanently.

Related Articles
Rules: Daily Loss Limit

Growth Funded: Account Payout Policy

Essential Trading Rules Overview

Select Flex and Select Daily Payout Policies

Tradeify Elite Program

Did this answer your question?😞😐😃

Table of contents
