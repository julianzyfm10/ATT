"""Persistent stdio MCP sessions for the offline Core Local servers."""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
MCP_PYTHON = ROOT / ".venv_mcp_live" / "Scripts" / "python.exe"


@dataclass(frozen=True)
class PersistentServerConfig:
    server_id: str
    command: str
    args: tuple[str, ...]
    cwd: Path


def _node_executable() -> str:
    discovered = shutil.which("node")
    if discovered:
        return discovered
    bundled = Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies" / "node" / "bin" / "node.exe"
    if bundled.exists():
        return str(bundled)
    raise FileNotFoundError("Node.js is required for Math MCP but was not found")


def core_local_server_configs() -> dict[str, PersistentServerConfig]:
    servers = ROOT / "external" / "mcp-bench" / "mcp_servers"
    return {
        "unit_converter": PersistentServerConfig("unit_converter", str(MCP_PYTHON), ("-m", "unit_converter_mcp.server"), servers / "unit-converter-mcp"),
        "time": PersistentServerConfig("time", str(MCP_PYTHON), ("-m", "mcp_server_time"), servers / "time-mcp"),
        "math": PersistentServerConfig("math", _node_executable(), ("build/index.js",), servers / "math-mcp"),
        "scientific_computing": PersistentServerConfig("scientific_computing", str(MCP_PYTHON), ("src/server.py",), servers / "scientific_computation_mcp"),
        "medical_calculator": PersistentServerConfig("medical_calculator", str(MCP_PYTHON), ("-m", "medcalc"), servers / "medcalc"),
    }


class PersistentMCPClient:
    """Own one event loop and keep each requested stdio server alive per case."""

    def __init__(self, *, timeout: float = 60.0) -> None:
        self.timeout = timeout
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="core-local-mcp-loop", daemon=True)
        self._thread.start()
        self._stack: AsyncExitStack | None = None
        self._sessions: dict[str, Any] = {}
        self._closed = False
        self._submit(self._initialize())

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _initialize(self) -> None:
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()

    def _submit(self, coroutine: Any) -> Any:
        if self._closed:
            raise RuntimeError("Persistent MCP client is closed")
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        return future.result(timeout=self.timeout)

    async def _session(self, server_id: str) -> Any:
        if server_id in self._sessions:
            return self._sessions[server_id]
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        config = core_local_server_configs()[server_id]
        if not config.cwd.exists():
            raise FileNotFoundError(f"MCP server directory missing: {config.cwd}")
        if self._stack is None:
            raise RuntimeError("Persistent MCP exit stack is not initialized")
        params = StdioServerParameters(
            command=config.command,
            args=list(config.args),
            cwd=str(config.cwd),
            env=None,
        )
        read, write = await self._stack.enter_async_context(stdio_client(params, errlog=sys.stderr))
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._sessions[server_id] = session
        return session

    async def _list_tools(self, server_id: str) -> list[dict[str, Any]]:
        session = await self._session(server_id)
        response = await session.list_tools()
        return [tool.model_dump(mode="json") for tool in response.tools]

    def list_tools(self, server_id: str) -> list[dict[str, Any]]:
        return self._submit(self._list_tools(server_id))

    async def _call_tool(self, server_id: str, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        session = await self._session(server_id)
        result = await session.call_tool(tool_name, args)
        structured = getattr(result, "structuredContent", None)
        content = structured if structured is not None else _content_to_json(result.content)
        return {
            "server_id": server_id,
            "tool_name": tool_name,
            "content": content,
            "is_error": bool(getattr(result, "isError", False)),
        }

    def call_tool(self, server_id: str, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        return self._submit(self._call_tool(server_id, tool_name, args))

    async def _aclose(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._sessions.clear()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._submit(self._aclose())
        finally:
            self._closed = True
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5.0)
            self._loop.close()

    def __enter__(self) -> "PersistentMCPClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def _content_to_json(content: list[Any]) -> Any:
    if not content:
        return None
    values = [_item_to_json(item) for item in content]
    return values[0] if len(values) == 1 else values


def _item_to_json(item: Any) -> Any:
    text = getattr(item, "text", None)
    if text is not None:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    if hasattr(item, "model_dump"):
        return item.model_dump(mode="json")
    return str(item)

