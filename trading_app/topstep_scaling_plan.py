"""TopStep XFA Scaling Plan enforcer.

@canonical-source docs/research-input/topstep/images/xfa_scaling_chart.png
@canonical-source docs/research-input/topstep/topstep_scaling_plan_article.md
@article-id 8284223
@scraped 2026-04-08
@audit-finding F-1 BLOCKER

The TopStep Express Funded Account (XFA) Scaling Plan caps the maximum
mini-equivalent position size based on end-of-day account balance:

@verbatim "Your maximum number of contracts allowed to trade under the
           scaling plan does not increase throughout the trading day. If
           your earnings meet or exceed the required amount to scale up,
           you still need to wait until the following session to trade
           the next Scaling Plan level."

@verbatim "Errors in the Scaling Plan corrected in less than 10 seconds
           will be ignored. If traders leave on too many contracts for
           10 seconds or more, even if only by a few seconds, their
           account may be reviewed."

The 10-second grace period is for FAT-FINGER MANUAL ERRORS only. Sustained
automated exposure for >= 10s = account review (TopStep risk team
investigates and may close the account).

Canonical ladder (visually parsed from xfa_scaling_chart.png):

  $50K XFA
    Below $1,500           → 2 lots
    $1,500 to $2,000       → 3 lots
    Above $2,000           → 5 lots

  $100K XFA
    Below $1,500           → 3 lots
    $1,500 to $2,000       → 4 lots
    $2,000 to $3,000       → 5 lots
    Above $3,000           → 10 lots

  $150K XFA
    Below $1,500           → 3 lots
    $1,500 to $2,000       → 4 lots
    $2,000 to $3,000       → 5 lots
    $3,000 to $4,500       → 10 lots
    Above $4,500           → 15 lots

"lots" = mini-equivalent. On TopstepX, 1 lot = 1 mini = 10 micros (per
@verbatim "Micros and Minis are calculated using a 10:1 ratio: 1 Mini
           contract = 10 Micro contracts").

Day-1 of any new XFA: balance = $0 (XFA starts at $0 per
docs/research-input/topstep/topstep_mll_article.md), so max position
is the bottom-row tier (2 lots for 50K, 3 lots for 100K/150K).
"""

from __future__ import annotations

# Tier ladder: account_size (USD) → list of (min_balance, max_lots) tuples,
# sorted by min_balance ascending. The function `max_lots_for_xfa` walks the
# ladder and picks the highest tier whose threshold is met.
#
# These values are visually parsed from
# docs/research-input/topstep/images/xfa_scaling_chart.png on 2026-04-08.
# Re-verify after any TopStep policy announcement (quarterly cadence).

SCALING_PLAN_LADDER: dict[int, list[tuple[float, int]]] = {
    50_000: [
        (0.0, 2),
        (1500.0, 3),
        (2000.0, 5),
    ],
    100_000: [
        (0.0, 3),
        (1500.0, 4),
        (2000.0, 5),
        (3000.0, 10),
    ],
    150_000: [
        (0.0, 3),
        (1500.0, 4),
        (2000.0, 5),
        (3000.0, 10),
        (4500.0, 15),
    ],
}


def max_lots_for_xfa(account_size: int, eod_balance: float) -> int:
    """Return the max mini-equivalent lots allowed for an XFA at this balance.

    @canonical-source docs/research-input/topstep/images/xfa_scaling_chart.png
    @canonical-source docs/research-input/topstep/topstep_scaling_plan_article.md

    Parameters:
        account_size: XFA size in USD (50_000, 100_000, or 150_000)
        eod_balance: End-of-day account balance in USD. For an XFA, this is
            the broker equity (XFA starts at $0 and grows with profit).
            Per the canonical no-intraday-scaling rule, the bot must use
            the EOD balance from the previous session, not the live equity.

    Returns:
        Maximum mini-equivalent lot count allowed for today's session.

    Raises:
        KeyError: if account_size is not a known XFA tier (50K/100K/150K).
        ValueError: if eod_balance is below 0 (sentinel for missing data).
    """
    if account_size not in SCALING_PLAN_LADDER:
        raise KeyError(
            f"TopStep Scaling Plan ladder is only defined for "
            f"account sizes {list(SCALING_PLAN_LADDER.keys())}; got {account_size}"
        )
    if eod_balance < 0:
        raise ValueError(
            f"eod_balance must be >= 0 (got {eod_balance}). For a fresh XFA "
            f"the canonical starting balance is $0, not negative."
        )

    ladder = SCALING_PLAN_LADDER[account_size]
    # Walk ladder ascending; pick the highest tier whose threshold is met.
    max_lots = ladder[0][1]
    for threshold, lots in ladder:
        if eod_balance >= threshold:
            max_lots = lots
    return max_lots


def micros_to_mini_equivalent(micros: int) -> int:
    """Convert micro contracts to mini-equivalent (TopstepX 10:1 ratio).

    @canonical-source docs/research-input/topstep/topstep_scaling_plan_article.md
    @verbatim "On TopstepX, Micros and Minis are calculated using a 10:1
               ratio: 1 Mini contract = 10 Micro contracts"

    The Scaling Plan caps mini-equivalents, not raw contract counts. A 50K
    XFA at Level 1 (2 lots) can hold 2 minis OR 20 micros OR any
    combination summing to 2 mini-equivalents.

    Returns the CEILING of micros/10 because partial-mini exposure still
    counts toward the limit (e.g., 11 micros = 1.1 mini → 2 mini).
    """
    if micros < 0:
        raise ValueError(f"micros must be >= 0 (got {micros})")
    return (micros + 9) // 10  # ceiling division


def lots_for_position(instrument: str, contracts: int) -> int:
    """Convert a (instrument, contracts) pair to mini-equivalent lots.

    Convention: instruments starting with 'M' (MNQ, MES, MGC, MCL, etc.)
    are micros (10:1). Their full-size parents (NQ, ES, GC, CL) are minis (1:1).

    @canonical-source docs/research-input/topstep/topstep_scaling_plan_article.md
    """
    if instrument.upper().startswith("M") and len(instrument) > 1 and instrument[1].isalpha():
        # All M{X} symbols where X is alpha are micros (MNQ, MES, MGC, MCL, M2K, M6E, MBT)
        return micros_to_mini_equivalent(contracts)
    return contracts  # full-size minis count 1:1


def total_open_lots(active_trades: list, instrument: str | None = None) -> int:
    """Sum mini-equivalent lots across active ENTERED trades.

    @canonical-source docs/research-input/topstep/topstep_scaling_plan_article.md

    @future-followup The canonical referenced article on net position
    calculation across simultaneous long+short
    (https://intercom.help/topstep-llc/en/articles/8284209) has not been
    fetched yet. The Scaling Plan rule may use NET (long minus short)
    rather than GROSS (long plus short) exposure. This implementation
    uses GROSS as the conservative interpretation — gross >= net always,
    so a gross-based check cannot under-count exposure.

    Parameters:
        active_trades: list of trade objects with `state` (state.value
            should be "ENTERED" to count), `contracts`, and either
            `strategy.instrument` or `instrument` attribute.
        instrument: if provided, only sum lots for that instrument; if
            None, sum across all instruments. The Scaling Plan caps total
            mini-equivalent net position regardless of instrument, so the
            usual call site uses None.

    Returns:
        Total mini-equivalent lot count across matching active trades.
    """
    total = 0
    for t in active_trades:
        if not hasattr(t, "state") or t.state.value != "ENTERED":
            continue
        # Resolve instrument with fallback chain
        t_inst = None
        if hasattr(t, "strategy") and hasattr(t.strategy, "instrument"):
            t_inst = t.strategy.instrument
        elif hasattr(t, "instrument"):
            t_inst = t.instrument
        if instrument is not None and t_inst != instrument:
            continue
        if t_inst is None:
            continue
        contracts = getattr(t, "contracts", 0)
        if contracts <= 0:
            continue
        total += lots_for_position(t_inst, contracts)
    return total
