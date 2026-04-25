"""Carry input construction for `mf_futures`."""

from __future__ import annotations

from .kernel import annualized_carry_from_curve
from .models import CarryInputSlice, FrontNextPair


def build_carry_input_slice(pair: FrontNextPair) -> CarryInputSlice:
    """Build an honest carry input slice from a deterministic front/next pair."""
    front_contract = pair.front.contract_symbol if pair.front is not None else None
    next_contract = pair.next_contract.contract_symbol if pair.next_contract is not None else None
    front_price = pair.front.settlement if pair.front is not None else None
    next_price = pair.next_contract.settlement if pair.next_contract is not None else None

    price_spread = None
    price_ratio = None
    if front_price is not None and next_price is not None:
        price_spread = front_price - next_price
        price_ratio = front_price / next_price if next_price != 0 else None

    annualized_carry = None
    carry_available = False
    unavailable_reason = pair.unavailable_reason
    if (
        pair.carry_available
        and front_price is not None
        and next_price is not None
        and pair.calendar_gap_days is not None
    ):
        annualized_carry = annualized_carry_from_curve(
            front_price=front_price,
            next_price=next_price,
            days_between_expiries=pair.calendar_gap_days,
        )
        carry_available = True

    return CarryInputSlice(
        trading_day=pair.trading_day,
        symbol=pair.symbol,
        front_contract=front_contract,
        next_contract=next_contract,
        front_price=front_price,
        next_price=next_price,
        price_spread=price_spread,
        price_ratio=price_ratio,
        selection_metric=pair.selection_metric,
        contract_gap_months=pair.contract_gap_months,
        calendar_gap_days=pair.calendar_gap_days,
        annualized_carry=annualized_carry,
        carry_available=carry_available,
        unavailable_reason=unavailable_reason,
    )
