# Anti-Overfit Protocol (Mandatory)

## Goal
Prevent data-mining bias, p-hacking, curve-fitting, and false discoveries.

## 1) Pre-register before testing
For each idea, write down *before running tests*:
- Hypothesis statement
- Expected mechanism (why it should work)
- Exact rule/thresholds to test
- Allowed variant count (max 3 unless explicitly approved)

No post-hoc rewriting.

## 2) Dataset discipline
Use fixed time splits:
- Train: <= 2023
- Dev: 2024 (for limited tuning)
- Test: 2025 (one-way gate)
- Forward: 2026+ (shadow/live validation)

Rules:
- Never tune on Test after viewing results.
- If changed after Test view, mark as new hypothesis and restart split process.

## 3) Multiple-comparison control
Within each hypothesis family, control false positives with BH-FDR (q=0.10).
If family scan is large, no candidate is promotable without FDR pass.

## 4) Minimum sample gates
Default minimums (unless explicitly overridden):
- N_on >= 80
- N_off >= 80
- Test N_on >= 40, Test N_off >= 40
- At least 3 years with evaluable uplift

## 5) Promotion gates (A-tier)
Candidate is promotable only if all pass:
- avg_on >= +0.20
- uplift >= +0.25
- OOS uplift >= +0.15
- Practical tradability (frequency fit for intended mode)

## 6) Complexity penalty
Prefer simpler models when performance is similar.
- Fewer conditions > many stacked conditions
- Stable across years > highest in-sample score

## 7) Audit trail required
For every result, store:
- script path + commit hash
- exact query/parameters
- split definitions used
- pass/fail reason

No undocumented “it looked good” conclusions.

## 8) Kill policy
Auto-KILL if:
- test uplift <= 0
- unstable sign flip across years
- only works in tiny niche with no practical execution path

---

This protocol is mandatory for promotion decisions.
