"""Minimal newline-delimited stdio MCP server for local repo tools.

This deliberately avoids the Python MCP server stack in environments where the
installed SDK hangs during `initialize` over stdio. It implements the small
subset Codex needs for tool-based local servers:

- `initialize`
- `notifications/initialized`
- `ping`
- `tools/list`
- `tools/call`
"""

from __future__ import annotations

import json
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: JsonDict
    handler: Callable[..., Any]


class StdioToolServer:
    def __init__(self, name: str, instructions: str, tools: list[ToolSpec]) -> None:
        self.name = name
        self.instructions = instructions
        self.tools = tools
        self._tool_index = {tool.name: tool for tool in tools}

    def tool_names(self) -> list[str]:
        return [tool.name for tool in self.tools]

    def list_tools_payload(self) -> list[JsonDict]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
            }
            for tool in self.tools
        ]

    def run(self) -> None:
        while True:
            try:
                raw = input()
            except EOFError:
                break

            if not raw.strip():
                continue

            request_id: int | str | None = None
            try:
                message = json.loads(raw)
                request_id = message.get("id")
                response = self._handle_message(message)
            except Exception as exc:
                if request_id is None:
                    continue
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32603,
                        "message": str(exc),
                        "data": traceback.format_exc(limit=5),
                    },
                }

            if response is not None:
                print(json.dumps(response, ensure_ascii=False), flush=True)

    def _handle_message(self, message: JsonDict) -> JsonDict | None:
        method = message.get("method")
        request_id = message.get("id")
        params = message.get("params") or {}

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": params.get("protocolVersion", "2024-11-05"),
                    "capabilities": {
                        "tools": {"listChanged": False},
                    },
                    "serverInfo": {
                        "name": self.name,
                        "version": "1.0.0",
                    },
                    "instructions": self.instructions,
                },
            }

        if method == "notifications/initialized":
            return None

        if method == "ping":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {},
            }

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": self.list_tools_payload(),
                },
            }

        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments") or {}
            tool = self._tool_index.get(name)
            if tool is None:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
                        "isError": True,
                    },
                }

            try:
                result = tool.handler(**arguments)
                text = json.dumps(result, ensure_ascii=False)
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [{"type": "text", "text": text}],
                        "structuredContent": result,
                        "isError": False,
                    },
                }
            except Exception as exc:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                        "isError": True,
                    },
                }

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32601,
                "message": f"Method not found: {method}",
            },
        }
