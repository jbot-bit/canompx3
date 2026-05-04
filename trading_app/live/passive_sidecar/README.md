# Passive Sidecar

This module is prohibited for Live Funded accounts until written confirmation
is received from Topstep that passive, non-executing tooling is permitted.

DO NOT activate without that confirmation.

## Design rules

- local-device only
- non-executing only
- user-hub state consumption only in v1
- no order placement
- no order cancellation
- no order modification

`LIVE_PASSIVE_SIDECAR_ALLOWED=true` is not a product decision. It is only a
hard gate that must remain false until the written policy answer is in hand.
