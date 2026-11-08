"""Minimal MCP stdio client for the selected MCP-Bench pilot servers."""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
MCP_PYTHON = ROOT / ".venv_mcp_live" / "Scripts" / "python.exe"


@dataclass(frozen=True)
class MCPServerConfig:
    server_id: str
    cwd: Path
    args: list[str]


MCP_SERVER_CONFIGS: dict[str, MCPServerConfig] = {
    "unit_converter": MCPServerConfig(
        server_id="unit_converter",
        cwd=ROOT / "external" / "mcp-bench" / "mcp_servers" / "unit-converter-mcp",
        args=["-m", "unit_converter_mcp.server"],
    ),
    "time": MCPServerConfig(
        server_id="time",
        cwd=ROOT / "external" / "mcp-bench" / "mcp_servers" / "time-mcp",
        args=["-m", "mcp_server_time"],
    ),
    "wikipedia": MCPServerConfig(
        server_id="wikipedia",
        cwd=ROOT / "external" / "mcp-bench" / "mcp_servers" / "wikipedia-mcp",
        args=["-m", "wikipedia_mcp", "--log-level", "ERROR"],
    ),
}


def call_mcp_tool(server_id: str, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Call a selected MCP-Bench server over stdio and return a JSON-safe result."""
    return asyncio.run(_call_mcp_tool(server_id, tool_name, args))


async def _call_mcp_tool(server_id: str, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    config = MCP_SERVER_CONFIGS[server_id]
    params = StdioServerParameters(
        command=str(MCP_PYTHON),
        args=config.args,
        cwd=str(config.cwd),
        env=None,
    )
    async with stdio_client(params, errlog=sys.stderr) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tool_result = await session.call_tool(tool_name, args)
            return {
                "server_id": server_id,
                "tool_name": tool_name,
                "content": _content_to_json(tool_result.content),
                "is_error": bool(getattr(tool_result, "isError", False)),
            }


def _content_to_json(content: list[Any]) -> Any:
    if not content:
        return None
    if len(content) == 1:
        return _item_to_json(content[0])
    return [_item_to_json(item) for item in content]


def _item_to_json(item: Any) -> Any:
    text = getattr(item, "text", None)
    if text is not None:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    if hasattr(item, "model_dump"):
        return item.model_dump()
    return str(item)
