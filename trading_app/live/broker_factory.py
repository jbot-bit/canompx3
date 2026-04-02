"""Factory for creating broker components from a broker name."""

import logging
import os
from typing import TypedDict

from .broker_base import BrokerAuth, BrokerContracts, BrokerFeed, BrokerPositions, BrokerRouter

log = logging.getLogger(__name__)

# Valid broker names — canonical source for dispatcher
VALID_BROKERS = ("projectx", "tradovate")


class BrokerComponents(TypedDict):
    auth: BrokerAuth
    feed_class: type[BrokerFeed]
    router_class: type[BrokerRouter]
    contracts_class: type[BrokerContracts]
    positions_class: type[BrokerPositions]


# Re-export ABCs for type-checking convenience
__all__ = [
    "VALID_BROKERS",
    "get_broker_name",
    "create_broker_components",
    "BrokerComponents",
    "BrokerAuth",
    "BrokerContracts",
    "BrokerFeed",
    "BrokerPositions",
    "BrokerRouter",
]


def get_broker_name() -> str:
    """Read broker name from env, default to 'projectx'."""
    return os.environ.get("BROKER", "projectx").lower()


def create_broker_components(
    broker: str | None = None,
    demo: bool = True,
) -> BrokerComponents:
    """Create all broker components for the given broker.

    Returns dict with keys: auth, feed_class, router_class, contracts_class, positions_class.
    Feed/router/contracts/positions are CLASSES (not instances) — caller instantiates with args.
    Auth is an INSTANCE (needs to be shared across components).
    """
    if broker is None:
        broker = get_broker_name()

    if broker not in VALID_BROKERS:
        raise ValueError(f"Unknown broker: '{broker}'. Valid: {sorted(VALID_BROKERS)}")

    if broker == "projectx":
        from .projectx.auth import ProjectXAuth
        from .projectx.contract_resolver import ProjectXContracts
        from .projectx.data_feed import ProjectXDataFeed
        from .projectx.order_router import ProjectXOrderRouter
        from .projectx.positions import ProjectXPositions

        auth = ProjectXAuth()
        log.info("Broker: ProjectX (TopstepX)")
        return {
            "auth": auth,
            "feed_class": ProjectXDataFeed,
            "router_class": ProjectXOrderRouter,
            "contracts_class": ProjectXContracts,
            "positions_class": ProjectXPositions,
        }

    elif broker == "tradovate":
        from .tradovate.auth import TradovateAuth
        from .tradovate.contracts import TradovateContracts
        from .tradovate.order_router import TradovateOrderRouter
        from .tradovate.positions import TradovatePositions

        # Tradovate reads demo mode from TRADOVATE_DEMO env var
        auth = TradovateAuth()
        log.info("Broker: Tradovate (Tradeify/MFFU)")
        return {
            "auth": auth,
            "feed_class": None,  # No Tradovate feed — use ProjectX master feed
            "router_class": TradovateOrderRouter,
            "contracts_class": TradovateContracts,
            "positions_class": TradovatePositions,
        }

    raise AssertionError("unreachable")
