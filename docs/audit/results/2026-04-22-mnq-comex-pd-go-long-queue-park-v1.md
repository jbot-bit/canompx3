# MNQ COMEX_SETTLE PD_GO_LONG queue park v1

Date: 2026-04-22
Branch: `wt-codex-mnq-hiroi-scan`
Candidate: `transfer::COMEX_SETTLE::1.0::long::PD_GO_LONG`

## Question

Should the current mechanism-distinct queued transfer candidate advance into a
new exact bridge on:

- `MNQ COMEX_SETTLE O5 E2 RR1.0 long`
- `PD_GO_LONG`

This row was reviewed after the NYSE short exact-cell park because it was the
next non-NYSE queued transfer candidate and kept the frontier broad.

## Transfer-board truth

From `docs/audit/results/2026-04-22-mnq-geometry-transfer-board-v1.md`:

- `COMEX_SETTLE RR1.0 long PD_GO_LONG`
  - `N_on_IS=390`
  - `ExpR_on_IS=+0.1383`
  - `ExpR_off_IS=+0.0213`
  - `Delta_IS=+0.1171`
  - `t=+1.82`
  - `BH=0.1569`
  - `N_on_OOS=17`
  - `Delta_OOS=+0.2189`
  - `same_sign_oos=True`

Directionally this row is alive, but the key question is not whether it is
positive in isolation. The key question is whether it is the right next bridge
for this parent lane.

## Existing promoted-lane check

That same lane already has a stronger promoted transfer result:

- `docs/audit/results/2026-04-22-mnq-comex-pd-clear-long-take-v1.md`
  - `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_PD_CLEAR_LONG`
  - promoted and active in `validated_setups`
  - `Delta_IS=+0.1704`
  - `Delta_OOS=+0.2336`

The accepted result doc also recorded the decisive component split on this
lane:

- `PD_CLEAR_LONG` is the positive transfer state
- `PD_DISPLACE_LONG` is weak / negative
- the broader `PD_GO_LONG` union is weaker than `PD_CLEAR_LONG` here

That means the current queued union row is not an unsolved transfer story. It
is a weaker umbrella around a lane that was already honestly solved by the
bounded `PD_CLEAR_LONG` member.

## Why This Should Not Advance

Advancing `PD_GO_LONG` now would be bad queue discipline for three reasons:

1. It is dominated by an already-promoted exact family member on the same lane.
2. The accepted `PD_CLEAR_LONG` result already concluded this session does
   **not** want the broader union.
3. Spending another exact bridge on the broader union would reopen a lane that
   has already been resolved tightly enough for the current mechanism class.

So this is not a fresh transfer opportunity. It is residual route-map context.

## Decision

Park `transfer::COMEX_SETTLE::1.0::long::PD_GO_LONG`.

Do **not** write a prereg for the broader union on this iteration. The honest
read is:

- the row is still positive as context
- but the lane has already been solved more cleanly by `PD_CLEAR_LONG`
- this queued union row is therefore lower-EV than it first appears

## Queue consequence

- frontier decision: `parked`
- queue reason:
  - dominated by already-promoted `PD_CLEAR_LONG` on the same lane
  - accepted result doc already says the broader union is weaker here
- next focus:
  - move to the unsolved `EUROPE_FLOW RR1.0 long` transfer pair and compare
    `PD_GO_LONG` vs `PD_DISPLACE_LONG` instead of revisiting COMEX

## Verdict

`transfer::COMEX_SETTLE::1.0::long::PD_GO_LONG` is parked.

This is not a mechanism kill. It is a bounded queue decision: the candidate
remains useful as context on the transfer board, but it should not consume the
next bridge because the stronger exact family member on the same lane already
promoted.
