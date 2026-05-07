from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path


def _read_json_line(proc: subprocess.Popen[str]) -> dict[str, object]:
    assert proc.stdout is not None
    raw = proc.stdout.readline()
    assert raw
    return json.loads(raw)


def test_simple_stdio_server_initializes_and_lists_tools(tmp_path: Path) -> None:
    script = tmp_path / "probe_server.py"
    script.write_text(
        "\n".join(
            [
                "from scripts.tools.simple_mcp_stdio import StdioToolServer, ToolSpec",
                "",
                "def ping() -> dict[str, str]:",
                "    return {'pong': 'ok'}",
                "",
                "server = StdioToolServer(",
                "    'probe',",
                "    instructions='probe server',",
                "    tools=[",
                "        ToolSpec(",
                "            'ping',",
                "            'Ping tool',",
                "            {'type': 'object', 'properties': {}, 'additionalProperties': False},",
                "            ping,",
                "        )",
                "    ],",
                ")",
                "",
                "if __name__ == '__main__':",
                "    server.run()",
            ]
        ),
        encoding="utf-8",
    )

    root = Path(__file__).resolve().parents[2]
    env = dict(**{"PYTHONPATH": str(root)}, **dict())
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        cwd=root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    try:
        assert proc.stdin is not None
        proc.stdin.write(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "t"}},
                }
            )
            + "\n"
        )
        proc.stdin.flush()
        initialize = _read_json_line(proc)
        assert initialize["result"]["serverInfo"]["name"] == "probe"

        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}) + "\n")
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}) + "\n")
        proc.stdin.flush()
        tools = _read_json_line(proc)
        assert tools["result"]["tools"][0]["name"] == "ping"

        proc.stdin.write(
            json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "ping", "arguments": {}}})
            + "\n"
        )
        proc.stdin.flush()
        call = _read_json_line(proc)
        assert call["result"]["structuredContent"] == {"pong": "ok"}
        assert call["result"]["isError"] is False
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
