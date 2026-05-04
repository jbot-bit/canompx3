Copyright 2023. Intercom Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.Copyright 2023. Intercom Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. [Skip to main content](https://help.topstep.com/en/articles/13747047-understanding-hedging#main-content)

# Understanding Hedging

Cross-account hedging is prohibited at Topstep. This article explains what hedging is, why we prohibit it, and how to ensure your trading stays compliant.

Updated yesterday

Table of contents

[What is Cross-Account Hedging?](https://help.topstep.com/en/articles/13747047-understanding-hedging#h_9ffa6bd3d2)[Why Does Topstep Prohibit Hedging?](https://help.topstep.com/en/articles/13747047-understanding-hedging#h_4336a9ec03)[How Topstep Detects Hedging](https://help.topstep.com/en/articles/13747047-understanding-hedging#h_18c543b804)[What Happens If Hedging is Detected?](https://help.topstep.com/en/articles/13747047-understanding-hedging#h_edf36ba38a)[Copy Trading and Technical Errors](https://help.topstep.com/en/articles/13747047-understanding-hedging#h_ab28d0e1a7)[How to Avoid Hedging Violations](https://help.topstep.com/en/articles/13747047-understanding-hedging#h_b59f24858a)[Frequently Asked Questions](https://help.topstep.com/en/articles/13747047-understanding-hedging#h_bd7eb545dd)[Additional Resources](https://help.topstep.com/en/articles/13747047-understanding-hedging#h_aefa1b4aaa)

* * *

[What is Cross-Account Hedging?](https://help.topstep.com/en/articles/13747047-understanding-hedging#h_83a6ff742e)

[Why Does Topstep Prohibit Hedging?](https://help.topstep.com/en/articles/13747047-understanding-hedging#h_4336a9ec03)

[How Topstep Detects Hedging](https://help.topstep.com/en/articles/13747047-understanding-hedging#h_18c543b804)

[What Happens If Hedging is Detected?](https://help.topstep.com/en/articles/13747047-understanding-hedging#h_edf36ba38a)

[Copy Trading and Technical Errors](https://help.topstep.com/en/articles/13747047-understanding-hedging#h_ab28d0e1a7)

[How to Avoid Hedging Violations](https://help.topstep.com/en/articles/13747047-understanding-hedging#h_b59f24858a)

[Frequently Asked Questions](https://help.topstep.com/en/articles/13747047-understanding-hedging#h_bd7eb545dd)

[Additional Resources](https://help.topstep.com/en/articles/13747047-understanding-hedging#h_aefa1b4aaa)

# What is Cross-Account Hedging?

Cross-account hedging occurs when you hold **opposite positions across multiple accounts at the same time**. This means you're simultaneously long and short the same instrument (or highly correlated/fungible instruments) across different Combines, Express Funded Accounts (XFAs), or Live Funded Accounts (LFAs).

### Simple Example:

**Account A:** Long 5 contracts of ES

**Account B:** Short 5 contracts of ES

​

When positions are hedged like this, you're protected from market risk. If the market moves up, Account A profits while Account B loses. If the market moves down, Account B profits while Account A loses. The net result is roughly break-even across both accounts. This enables a clever trader to show simulated profits from the winning side while only sacrificing the cost of the Trading Combine / Express Funded Account.

### Why This is Problematic

Hedging defeats the purpose of evaluating your trading skill. It creates an unfair advantage by allowing you to profit regardless of market direction, which isn't genuine trading; it's exploiting the program structure. Additionally, this is not allowed in Live markets, so while traders could exploit Topstep’s program for profit while in a simulated market, this type of trading is prohibited by the Exchange (CME) in live markets.

* * *

# Why Does Topstep Prohibit Hedging?

Our mission is to **prepare traders for live markets**. Real exchanges strictly prohibit hedging, wash trading, and other manipulative practices.

​

**CME Group Rule 534** states:

​ _"Performing, alone or in concert with any other persons, including between connected accounts, or accounts held with different Parties, trades, or combinations of trades, the purpose of which is to manipulate, abuse, or give User an unfair advantage while using the Site or Services, for example, by engaging in any short term or high-frequency trades or simultaneously entering into opposite positions"_

​

By maintaining the same standards as professional exchanges, we ensure that:

- Success reflects genuine trading skill

- All traders compete on a level playing field

- You're prepared for the rules you'll face in live markets

- Capital is allocated to skilled traders, not loophole exploiters


* * *

# How Topstep Detects Hedging

We use advanced monitoring systems to identify hedging patterns across your accounts. Our systems analyze:

- **Position Timing:** Are opposite positions opened and closed in coordinated patterns?

- **Position Size:** Are positions sized to offset each other's risk?

- **Duration:** How long are opposite positions held simultaneously?

- **Intent:** Does the pattern suggest intentional risk elimination rather than independent trading? Our enforcement is designed to be **fair and intelligent**—we distinguish between genuine trading activity and exploitative behavior.


* * *

# **What Happens If Hedging is Detected?**

See below for an overview of our Hedging Detection process:

TopstepX Hedging Tutorial Final

Copy link

[Open video in Loom](https://www.loom.com/share/538ea47c9ba44b8091fbff2afb66fe08)

0

1.2×

6 min⚡️7 min 56 sec6 min 20 sec5 min 17 sec4 min 13 sec3 min 44 sec3 min 10 sec2 min 32 sec

![](https://cdn.loom.com/sessions/thumbnails/538ea47c9ba44b8091fbff2afb66fe08-36d0bf00d61cec0c.jpg)

Copy link

[Open video in Loom](https://www.loom.com/share/538ea47c9ba44b8091fbff2afb66fe08)

0

1.2×

6 min⚡️7 min 56 sec6 min 20 sec5 min 17 sec4 min 13 sec3 min 44 sec3 min 10 sec2 min 32 sec

Enforcement occurs in stages, depending on severity:

### **1\. First Hedging Attempt — Real-Time Warning**

If our system detects that you’ve become hedged across accounts:

- You’ll receive a **real-time modal notification**

- You will have **a brief window** to **un-hedge your positions** using the options in the modal notification:









![](https://downloads.intercomcdn.com/i/o/bjnr216i/2234310438/9c56444b13830c36cb6db388c8b9/image.png?expires=1775649600&signature=3448def47f04fd3f582861406a9cb96037b2f01c6d0695268d3b41b27bc4d8a4&req=diIkEsp%2FnYVcUfMW3nq%2BgScrBKF6h0uVxlRKjUYt6SZzp5Cr5j%2FDUsIuTp30%0A%2FAoc6Ooyr3qfhRiQKS9P57SD9js%3D%0A)

- If you **successfully un-hedge within the time window**, you may continue trading


### **If You Do Not Un-Hedge**

- Your hedged positions will be **automatically liquidated**

- Your account will be **flagged for hedging behavior**

- You will receive a **follow-up email notification**


This first instance serves as a **warning and educational step**.

* * *

### **2\. Second Hedging Attempt (Same Day)**

If hedging occurs again **on the same trading day**:

- You will have **a brief window** to un-hedge

- If you **do not un-hedge in time**:





  - A **Temporary Hedging Violation** will be issued and you will be prohibited from trading for the remainder of the trading day.

  - The violation will apply **across the hedged accounts**

  - **Please note:** There will not be a timer shown on this violation. Please un-hedge immediately to continue trading.


* * *

### **3\. Next Trading Day — Required Acknowledgement**

On your next login after a Temporary Hedging Violation:

- A **modal notification will appear upon login**

- You must **acknowledge the Terms of Use** related to hedging by typing "I agree" in the text box

- The modal will include:





  - Explanation of the hedging policy

  - Details of when the hedging occurred

  - Requirement to acknowledge before trading


You **will not be able to trade** until this acknowledgment is completed.

* * *

### **4\. Future Hedging Attempts — Permanent Violation**

After acknowledgement:

- Any future hedging attempt will trigger:





  - A real-time hedging alert pop-up with **a brief window** to un-hedge


- If you **do not un-hedge within the allotted time**:





  - A **Permanent Hedging Violation** will be issued and accounts involved in hedging will be closed.


* * *

### Important Notes

- Monitoring applies **in real-time**

- Time windows may be **adjusted by Risk & Leadership teams**

- Violations apply **across all accounts involved in hedging**

- This policy applies to **Combine, Express, and Live Accounts**


* * *

# Copy Trading and Technical Errors

### We Understand Technical Issues Happen

We recognize that copy trading software and brief execution errors can occasionally create temporary opposite positions. Our systems are designed to distinguish between:

- **Short-lived technical glitches** corrected immediately

- **Sustained hedged positions** indicate intentional risk elimination


### Your Responsibility

You remain **fully responsible** for all activity across your accounts, including positions created through:

- Copy trading software

- Automated trading systems

- Any third-party tools


If your copy trading tool creates a hedged position, you're responsible for monitoring and correcting it immediately.

### No Exceptions

Once a hedged position meets our violation criteria (based on size, duration, and intent), enforcement will occur, no exceptions.

* * *

# How to Avoid Hedging Violations

### Best Practices

1. **Trade a single account** to completely avoid the possibility of hedging

2. **Trade each account independently** based on your own analysis

3. **Don't coordinate positions** across accounts (or other traders) to offset risk

4. **Monitor copy trading tools** to ensure they're not creating opposite positions

5. **Close any accidental hedges immediately** if they occur

6. **Contact Support** before implementing any strategy you're unsure about


If you're unsure whether a specific trading approach complies with our policy, **contact Support before implementing it**. We're here to help you trade with confidence.

* * *

# Frequently Asked Questions

### Can I trade the same instrument across multiple accounts?

Yes! You can trade the same markets across different accounts. What's prohibited is holding **opposite positions simultaneously** in a way that eliminates market risk.

​

### Will I get a warning before enforcement?

Enforcement is applied in stages based on detection history:

- **1st detection:** If the hedged position is not removed, it will be liquidated.

- **Subsequent hedging attempt on the same trading day as the 1st detection:** If the hedged position is not removed, positions will be liquidated, and the account will be blocked for the remainder of the trading day.

- **Hedging attempt on any trading day after the 1st detection:** If the hedged position is not removed, this constitutes a permanent violation and zero-tolerance enforcement applies.


### What if I disagree with an enforcement decision?

All audit decisions are final. We encourage you to review our trading policies to ensure full compliance moving forward. If you have general questions about the policy itself, we are happy to clarify.

​

### Does this apply to both XFAs and LFAs?

Yes. Cross-account hedging is prohibited across all Combines, Express Funded Accounts (XFAs), and Live Funded Accounts (LFAs).

* * *

# Additional Resources

- [Topstep Terms of Use](https://www.topstep.com/terms-of-use/)

- [CME Group Rule 534](https://www.cmegroup.com/rulebook/files/cme-group-Rule-534.pdf)

- [Prohibited Conduct Policy](https://www.topstep.com/prohibited-conduct)

- [Contact Support](https://support.topstep.com/)

​


**Our commitment:** Topstep maintains a fair trading environment where success is based on skill and discipline. We appreciate your commitment to trading with integrity as we prepare you for live markets.

**Still Have Questions?**

Our Support team is here to help. If you need clarification on whether a specific trading approach complies with our hedging policy, reach out before implementing it. **Contact Support:** [support.topstep.com](http://support.topstep.com/)

* * *

Related Articles

[Trading Combine® Parameters](https://help.topstep.com/en/articles/8284197-trading-combine-parameters) [Prohibited Conduct](https://help.topstep.com/en/articles/10296582-prohibited-conduct) [Prohibited Trading Strategies at Topstep](https://help.topstep.com/en/articles/10305426-prohibited-trading-strategies-at-topstep) [What is Responsible Trading?](https://help.topstep.com/en/articles/10406542-what-is-responsible-trading)

Did this answer your question?

😞😐😃

Table of contents

[What is Cross-Account Hedging?](https://help.topstep.com/en/articles/13747047-understanding-hedging#h_9ffa6bd3d2)[Why Does Topstep Prohibit Hedging?](https://help.topstep.com/en/articles/13747047-understanding-hedging#h_4336a9ec03)[How Topstep Detects Hedging](https://help.topstep.com/en/articles/13747047-understanding-hedging#h_18c543b804)[What Happens If Hedging is Detected?](https://help.topstep.com/en/articles/13747047-understanding-hedging#h_edf36ba38a)[Copy Trading and Technical Errors](https://help.topstep.com/en/articles/13747047-understanding-hedging#h_ab28d0e1a7)[How to Avoid Hedging Violations](https://help.topstep.com/en/articles/13747047-understanding-hedging#h_b59f24858a)[Frequently Asked Questions](https://help.topstep.com/en/articles/13747047-understanding-hedging#h_bd7eb545dd)[Additional Resources](https://help.topstep.com/en/articles/13747047-understanding-hedging#h_aefa1b4aaa)