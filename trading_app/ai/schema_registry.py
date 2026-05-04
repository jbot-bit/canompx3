"""Named JSON-schema outputs for read-only AI extraction surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StructuredOutputSchema:
    schema_id: str
    description: str
    schema: dict[str, Any]

    def response_format(self) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self.schema_id,
                "strict": True,
                "schema": self.schema,
            },
        }


SCHEMA_REGISTRY: dict[str, StructuredOutputSchema] = {
    "grounded_research_assessment": StructuredOutputSchema(
        schema_id="grounded_research_assessment",
        description="Structured assessment of repo-grounded research claims.",
        schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "summary": {"type": "string"},
                "supported_claims": {"type": "array", "items": {"type": "string"}},
                "unsupported_claims": {"type": "array", "items": {"type": "string"}},
                "evidence_refs": {"type": "array", "items": {"type": "string"}},
                "next_steps": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "summary",
                "supported_claims",
                "unsupported_claims",
                "evidence_refs",
                "next_steps",
            ],
        },
    )
}


def list_schemas() -> list[str]:
    return sorted(SCHEMA_REGISTRY)


def get_schema(schema_id: str) -> StructuredOutputSchema:
    try:
        return SCHEMA_REGISTRY[schema_id]
    except KeyError as exc:
        raise KeyError(f"Unknown structured-output schema: {schema_id}") from exc
