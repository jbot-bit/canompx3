"""Rithmic broker adapter — order routing via Protocol Buffer over WebSocket.

Uses async_rithmic library (v1.5.9+) for Rithmic R | Protocol API.
Supports Bulenox, Elite Trader Funding, and other Rithmic-based prop firms.

Architecture: order-only adapter (like Tradovate). Market data from ProjectX.
Server-side brackets: stops/targets survive client crash.
"""

from .auth import RithmicAuth
from .contracts import RithmicContracts
from .order_router import RithmicOrderRouter
from .positions import RithmicPositions

__all__ = [
    "RithmicAuth",
    "RithmicContracts",
    "RithmicOrderRouter",
    "RithmicPositions",
]
