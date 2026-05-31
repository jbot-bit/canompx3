<!-- VERBATIM SNAPSHOT — DO NOT EDIT THE QUOTED BODY -->
Source URL: https://help.tradeify.co/en/articles/10495868-rules-hedging-trading-micros-or-minis
Article ID: 10495868
Scraped: 2026-05-31
Updated (per article): April 2, 2026
Page title: Rules: Hedging & Trading Micros or Minis | Tradeify Help Center
Fetch: curl -sL -A "<browser UA>" | html2text (structural tags->\n, no word changed)
Content images: none (all rules captured as HTML text)
==============================================================================

- Rules: Hedging & Trading Micros or Minis | Tradeify Help Center
- Skip to main content

Tradeify HomeDiscordEnglishEnglish

Tradeify HomeDiscordEnglishEnglish

Search for articles...

- All Collections

- Accounts & Rules

- Trading Rules

- Rules: Hedging & Trading Micros or Minis

Rules: Hedging & Trading Micros or Minis
Position management guidelines, hedging policies, and micro/mini contract usage rules.

April 2, 2026

Table of contents

Hedging and Contract Mixing Rules

Summary: Tradeify prohibits hedging (opposing positions on the same instrument) and mixing MINI and MICRO contracts. These rules apply across ALL your accounts.

Applies to: All Tradeify account types (Evaluation, Sim Funded, and Live)

Why These Rules Exist

Tradeify funds disciplined, consistent traders who demonstrate sustainable strategies. Hedging and contract mixing can mask risk exposure and make it difficult to evaluate genuine trading skill. These rules ensure fair and transparent trading across all programs. Additionally, they ensure that real-world capital risk behavior is accurately reflected to maintain a realistic trading environment.

Additionally, there is a 10-second grace period provided to close conflicting positions before a violation is flagged. This helps ensure traders can correct mistakes quickly without immediate penalties. Traders are also encouraged to closely monitor account activity and utilize platform tools to track and prevent unintentional violations, enhancing compliance.

The Two Rules

There are two separate rules you must follow when trading multiple instruments:

- Rule 1 - No Hedging: You cannot hold opposite positions (long AND short) on the SAME instrument at the same time.

- Rule 2 - No Mixing Contract Types: You cannot hold MINI and MICRO contracts at the same time, even on different instruments.

Contract Type Reference

Before trading multiple instruments, check which contract type each symbol belongs to. You can only hold contracts from ONE type at a time.

Contract Type

Symbols

MINI Contracts

ES, NQ, YM, RTY, CL, GC, SI, HG, NG, 6E, 6B, 6J

MICRO Contracts

MES, MNQ, MYM, M2K, MCL, MGC, M6E, M6A

Rule 1: No Hedging

The no hedging rule prohibits holding opposite positions (long and short) on the SAME instrument at the same time. This applies whether the positions are on the same account or different accounts.

Examples of hedging violations:

- Long ES + Short ES = NOT ALLOWED (same instrument, opposite directions)

- Long NQ + Short NQ = NOT ALLOWED (same instrument, opposite directions)

For example:

- A trader opened an MNQ micro position at 4:36:02 AM and later started trading NQ mini contracts while the MNQ was still open, causing a timing-based violation.

Rule 2: No Mixing MINI and MICRO Contracts

The no mixing rule prohibits holding MINI and MICRO contracts at the same time. This applies even if the contracts are for DIFFERENT underlying instruments and even if they are in the SAME direction.

Examples of mixing violations:

- Long MES + Long NQ = NOT ALLOWED (MES is MICRO, NQ is MINI)

- Long ES + Long MCL = NOT ALLOWED (ES is MINI, MCL is MICRO)

- Long MES + Short NQ = NOT ALLOWED (MES is MICRO, NQ is MINI)

What IS Allowed

These combinations are permitted because they use the SAME contract type (all MINI or all MICRO):

- Long MES + Long MNQ = ALLOWED (Both MICRO)

- Long MES + Short MNQ = ALLOWED (Both MICRO, different directions is fine for different instruments)

- Long ES + Long NQ = ALLOWED (Both MINI)

- Long ES + Short NQ = ALLOWED (Both MINI)

- Long ES + Long CL = ALLOWED (Both MINI)

- Long ES + Long GC = ALLOWED (Both MINI)

Cross-Account Rules

These rules apply across ALL your Tradeify accounts, not just within a single account. Our monitoring systems track positions across all accounts linked to the same trader.

Violations can also occur if activities in a failed account overlap with those in an active one. For instance, trading in a failed account alongside a successful active account may trigger the system's hedging alert, leading to breaches.

You cannot:

- Go long on one account and short on another account on the same instrument

- Hold MINI contracts on one account and MICRO contracts on another account simultaneously

- Use multiple accounts to circumvent hedging or contract mixing rules

Switching Between Contract Types

You CAN switch between MINI and MICRO contracts across different trading sessions. The rule only prohibits holding both types at the same time.

- Trade MES in the morning, close all positions, then trade ES in the evening = ALLOWED

- Hold MES overnight while opening ES the next day = NOT ALLOWED (both held simultaneously)

Additional strategies to prevent violations include:

- Flatten all positions in one account before trading in another to avoid cross-account hedging alerts.

- Avoid overlap between MINI and MICRO contract types, ensuring active positions are thoroughly reviewed before opening new ones.

- Confirm correct trade symbols to avoid unintended hedging violations.

- Use platform tools to synchronize multi-account strategies effectively.- Close all open positions across all accounts to avoid intentional or unintentional hedged positions.

- Trade only one contract type (either MINI or MICRO) to ensure compliance with the rules.

- Verify account status to ensure all accounts are flat before initiating new trades.

Consequences of Violations

Violations are detected automatically and will result in account review.

Consequences may include:

- Disqualification (for Evaluation accounts)

- Payout denial

- Account set to FAILED status

- All involved accounts set to FAILED (for cross-account violations)- Profits generated during the violation period may be removed if the account is later reinstated as a discretionary gesture

Automated Hedging Detection (Evaluation Accounts)

Evaluation accounts now have automated hedging detection. An evaluation account will be breached only if all three of the following conditions are met simultaneously:

- Opposing positions exist (hedging behavior)

- The hedge duration exceeds 10 seconds

- Profit generated from the hedge exceeds $250

All three conditions must be met for a breach to occur. Meeting only one or two of these conditions does not trigger a breach. For example, if opposing positions exist for more than 10 seconds but generate less than $250 in profit, the account will not be breached.

This automated detection applies only to Evaluation accounts. Sim Funded and Live accounts are subject to manual review for hedging violations as described above.

Frequently Asked Questions

Q: Can I trade MES in the morning and ES in the afternoon?

A: Yes, as long as you close all MES positions before opening any ES positions. You can switch between MINI and MICRO contracts - you just cannot hold both at the same time.

Q: I have multiple accounts. Do these rules apply across all of them?

A: Yes. Hedging and contract mixing rules apply across ALL your Tradeify accounts. Our systems monitor positions across all accounts linked to you.

Q: Can I go long ES on one account and short NQ on another?

A: Yes, this is allowed. ES and NQ are different instruments, so this is not hedging. Both are MINI contracts, so there is no mixing violation.

Q: What happens if I accidentally violate these rules?

A: Violations are detected automatically. Your account will be reviewed and may be set to FAILED status. If you believe a violation was caused by a platform error, contact support immediately.

Related Articles
Rules: Supported Trading Products / Assets

Trading Commission Fees

Lightning Funded Accounts

How to Select the Correct Trading Contract

Essential Trading Rules Overview

Did this answer your question?😞😐😃

Table of contents
