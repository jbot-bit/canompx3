<!-- VERBATIM SNAPSHOT — DO NOT EDIT THE QUOTED BODY -->
Source URL: https://help.tradeify.co/en/articles/10468321-rules-daily-loss-limit
Article ID: 10468321
Scraped: 2026-05-31
Updated (per article): April 10, 2026
Page title: Rules: Daily Loss Limit | Tradeify Help Center
Fetch: curl -sL -A "<browser UA>" | html2text (structural tags->\n, no word changed)
Content images: none (all rules captured as HTML text)
==============================================================================

- Rules: Daily Loss Limit | Tradeify Help Center
- Skip to main content

Tradeify HomeDiscordEnglishEnglish

Tradeify HomeDiscordEnglishEnglish

Search for articles...

- All Collections

- Accounts & Rules

- Trading Rules

- Rules: Daily Loss Limit

Rules: Daily Loss Limit
Daily maximum loss thresholds for account protection and risk management.

April 10, 2026

Table of contents

Daily Loss Limit (DLL)

Summary: The Daily Loss Limit pauses your trading for the day when you reach a specified loss threshold. Unlike the Max Trailing Drawdown, hitting the DLL does NOT fail your account - you can resume trading the next session.

Applies to: Growth, Lightning, and Select Daily Funded accounts. Select Flex accounts do not have a DLL.

Critical Warning

NEVER use the DLL as a stop loss. The DLL is not a hard stop - losses may exceed the limit before the system triggers protection. Using the DLL as a stop loss strategy in volatile markets may cause slippage that breaches your Max Trailing Drawdown, which WILL fail your account. Tradeify is not responsible for losses if you rely on the DLL as your stop.

How the DLL Works

The Daily Loss Limit is a "soft breach" - it pauses trading for the day but does NOT fail your account:

- When your daily losses reach the DLL, trading is paused

- You can resume trading the NEXT market session (after 6:00 PM ET)

- Your account remains active as long as you haven't hit the Max Trailing Drawdown

DLL vs Max Trailing Drawdown

These are two separate risk limits that work differently:

- Daily Loss Limit (DLL): Resets each trading day. Hitting it pauses trading until next session. Soft breach - account stays active.

- Max Trailing Drawdown: Tracks your highest balance. Hitting it FAILS your account permanently. Hard breach.

Important: If your Max Trailing Drawdown is closer than your DLL, you could hit the drawdown first and fail your account before the DLL triggers.

Initial DLL Amounts

When you start a Growth, Lightning, or Select Daily Funded account, your DLL is:

Account Size

Growth/Lightning DLL

Select Daily DLL

25K

$600*

$500

50K

$1,250

$1,000

100K

$2,500

$1,250

150K

$3,000**

$1,750

*25K Lightning accounts do not have Daily Loss Limit protection.

**150k Lightning funded accounts have $3,750 DLL if purchased before March 31, 2026. 

DLL Increases at 6% Profit

When your account reaches 6% profit (balance reaches the threshold below), your DLL increases to match your drawdown amount. This larger DLL takes effect the NEXT trading session, not immediately.

Account

Balance Required

Lightning DLL Becomes

Growth DLL Becomes

25K

$26,500

N/A

$1,000

50K

$53,000

$2,000

$2,000

100K

$106,000

$4,000

$3,500

150K

$159,000

$6,000

$5,000

Legacy Accounts (Purchased Before September 12, 2025)

If your account was purchased before September 12, 2025 at 8:00 AM EST, you have the legacy DLL rules. When you reach the 6% profit threshold, your DLL is completely REMOVED (not just increased).

Example Scenario

Starting a 100K Growth account:

- Starting balance: $100,000

- Daily Loss Limit: $2,500

- Max Trailing Drawdown: $3,500 from high water mark

If you lose $2,500 in one day, the DLL triggers and trading pauses. You can trade again tomorrow.

Warning: If your high water mark is $101,000 and current balance is $99,000, your drawdown is only $2,000 away. Losing $2,500 would hit the drawdown BEFORE the DLL, failing your account.

Frequently Asked Questions

Q: Do Select accounts have a DLL?

A: It depends on your payout policy. Select Daily funded accounts have a DLL ($500 for 25K, $1,000 for 50K, $1,250 for 100K, $1,750 for 150K). Select Flex (5-Day) funded accounts do not have a DLL - only the Max Trailing Drawdown applies.

Q: Does the DLL reset each day?

A: Yes. The DLL resets at the start of each trading session (6:00 PM ET). Yesterday's losses don't count toward today's limit.

Q: When does the increased DLL take effect?

A: When you reach the 6% profit threshold, your new DLL takes effect the NEXT trading session - not immediately. If you reach $53,000 on a 50K account during Monday's session, the increased DLL applies starting Monday evening at 6:00 PM ET.

Q: Can I still fail my account if I hit the DLL?

A: Hitting the DLL alone does not fail your account. However, if your Max Trailing Drawdown is closer than your DLL, you could breach the drawdown first, which DOES fail your account.

Q: What's the difference between legacy and new DLL rules?

A: For accounts purchased before September 12, 2025, the DLL is completely removed at 6% profit. For accounts purchased after, the DLL is increased (to $1,000/$2,000/$3,500-4,000/$5,000-6,000 for 25K/50K/100K/150K) but not removed.

Related Articles
Rules: Trailing Max Drawdowns

Lightning Funded Accounts

Essential Trading Rules Overview

SELECT vs Growth: Choosing Your Evaluation Type

Tradeify 3.0 – Program Updates & Improvements

Did this answer your question?😞😐😃

Table of contents
