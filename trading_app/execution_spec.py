"""
Execution specification dataclass.

Encapsulates entry rules and order parameters for a trading strategy.
Future extension point for TCA (slippage benchmarking, limit orders).
"""

from dataclasses import dataclass, asdict
import json


@dataclass(frozen=True)
class ExecutionSpec:
    """
    Execution specification for a trading strategy.

    Attributes:
        confirm_bars: Number of consecutive closes required (1, 2, or 3)
        order_type: Order type ('market', 'limit', 'stop')
        limit_offset_pct: Offset for limit orders (future)
        benchmark: Slippage benchmark ('arrival', 'vwap', 'twap') (future)
    """

    confirm_bars: int
    order_type: str = "market"

    # Future extension points for TCA
    limit_offset_pct: float | None = None
    benchmark: str | None = None

    def __post_init__(self):
        """Validate spec parameters."""
        self.validate()

    def validate(self) -> None:
        """
        Validate spec parameters.

        Raises:
            ValueError: If parameters are invalid
        """
        if self.confirm_bars not in [1, 2, 3]:
            raise ValueError(f"confirm_bars must be 1, 2, or 3, got {self.confirm_bars}")

        if self.order_type not in ["market", "limit", "stop"]:
            raise ValueError(
                f"order_type must be market/limit/stop, got {self.order_type}"
            )

        if self.limit_offset_pct is not None and self.limit_offset_pct < 0:
            raise ValueError(
                f"limit_offset_pct must be >= 0, got {self.limit_offset_pct}"
            )

        if self.benchmark is not None and self.benchmark not in [
            "arrival",
            "vwap",
            "twap",
        ]:
            raise ValueError(
                f"benchmark must be arrival/vwap/twap, got {self.benchmark}"
            )

    def to_json(self) -> str:
        """Serialize to JSON for storage."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> "ExecutionSpec":
        """Deserialize from JSON."""
        return cls(**json.loads(json_str))

    def __str__(self) -> str:
        """Human-readable string representation."""
        parts = [f"CB{self.confirm_bars}", self.order_type.upper()]
        if self.limit_offset_pct is not None:
            parts.append(f"offset={self.limit_offset_pct:.2%}")
        if self.benchmark is not None:
            parts.append(f"bench={self.benchmark}")
        return "_".join(parts)
