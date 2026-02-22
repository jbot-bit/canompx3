# Forward Gate Status (2026-02-22)

This is a live forward-window snapshot (2026 YTD only).

## Current state
All candidates remain **PENDING** because none has reached minimum forward sample target yet.

- A0: n=3 / 60
- A1: n=11 / 60
- A2: n=2 / 60
- A3: n=3 / 60
- B1: n=7 / 100
- B2: n=10 / 100

## Read
- Current values are not decision-grade due to low sample.
- No PROMOTE/KILL decisions should be made yet.
- Continue shadow logging until target_n is reached.

## Rule remains
PROMOTE/KILL only when target_n is reached per candidate and hard gates are evaluated.
