"""Backwards-compat re-export. Real implementation in tradovate/order_router.py."""

from .tradovate.order_router import OrderResult, OrderSpec  # noqa: F401
from .tradovate.order_router import TradovateOrderRouter as OrderRouter  # noqa: F401
