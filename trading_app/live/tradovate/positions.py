"""Tradovate position queries -- stub for future API access."""

import logging

from ..broker_base import BrokerAuth, BrokerPositions

log = logging.getLogger(__name__)


class TradovatePositions(BrokerPositions):
    def __init__(self, auth: BrokerAuth, demo: bool = True):
        super().__init__(auth)
        self.demo = demo

    def query_open(self, account_id: int) -> list[dict]:
        log.warning("Tradovate position query not implemented (no API access for prop firms). Returning empty.")
        return []
