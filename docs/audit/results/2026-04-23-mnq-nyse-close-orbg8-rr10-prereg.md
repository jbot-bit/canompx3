# MNQ NYSE_CLOSE ORB_G8 RR1.0 prereg

Date: 2026-04-23

## Scope

Execute the exact native `MNQ NYSE_CLOSE O5 E2 CB1 RR1.0 ORB_G8` prereg using canonical filter delegation only.

## Result

**Outcome:** `KILL`

Era-stability kill triggered on year(s): 2025.

## IS Summary

| Partition | N | Fire Rate | ExpR | Win Rate |
|---|---:|---:|---:|---:|
| baseline | 805 | 1.000 | +0.0838 | 0.595 |
| on_signal | 720 | 0.894 | +0.1107 | 0.603 |
| off_signal | 85 | 0.106 | -0.1439 | 0.529 |

- On-signal one-sample t-stat: `3.285`
- On-signal one-tailed p-value: `0.0005`
- IS uplift vs off-signal: `+0.2545R`

## IS Year Map (on-signal only)

| Year | N_on | ExpR_on |
|---|---:|---:|
| 2019 | 43 | +0.3201 |
| 2020 | 126 | +0.2168 |
| 2021 | 91 | +0.0975 |
| 2022 | 122 | +0.1637 |
| 2023 | 97 | +0.0194 |
| 2024 | 109 | +0.1565 |
| 2025 | 132 | -0.0697 |

## OOS Summary

| Partition | N | Fire Rate | ExpR | Win Rate |
|---|---:|---:|---:|---:|
| baseline | 42 | 1.000 | +0.4832 | 0.786 |
| on_signal | 42 | 1.000 | +0.4832 | 0.786 |
| off_signal | 0 | NA | NA | NA |

## Interpretation

- The exact `ORB_G8` path is statistically strong in-sample (`t=3.285`, `p=0.0005`) and fires on most IS rows (`720/805`, `89.4%`).
- It still fails its own prereg because 2025 on-signal performance is negative at meaningful size (`N_on=132`, `ExpR=-0.0697`).
- 2026 OOS is not a clean selector test for this path because `ORB_G8` fires on every observed OOS row (`42/42`), so there is no off-signal contrast.
- This closes the exact native `ORB_G8` route. It does not prove the broad NYSE_CLOSE RR1.0 family is dead, but it does remove the strongest locked native candidate from the open queue.
