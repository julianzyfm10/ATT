"""Single-task persistent stdio MCP worker.

AnyIO cancel scopes used by the MCP SDK must be entered and exited by the same
async task.  A synchronous caller therefore submits operations to one long-
lived worker task instead of creating a new coroutine task per operation.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import queue
import shutil
import sys
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
MCP_PYTHON = ROOT / ".venv_mcp_live" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


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
    def __init__(self, *, timeout: float = 60.0) -> None:
        self.timeout = timeout
        self._requests: queue.Queue[tuple[str, tuple[Any, ...], concurrent.futures.Future[Any]]] = queue.Queue()
        self._ready = threading.Event()
        self._closed = False
        self._thread = threading.Thread(target=self._thread_main, name="core-local-mcp-worker", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise TimeoutError("Persistent MCP worker did not initialize")

    def _thread_main(self) -> None:
        asyncio.run(self._worker())

    async def _worker(self) -> None:
        sessions: dict[str, Any] = {}
        async with AsyncExitStack() as stack:
            self._ready.set()
            while True:
                operation, args, future = await asyncio.to_thread(self._requests.get)
                if operation == "close":
                    future.set_result(None)
                    break
                try:
                    if operation == "list_tools":
                        session = await self._session(args[0], sessions, stack)
                        response = await session.list_tools()
                        value = [tool.model_dump(mode="json") for tool in response.tools]
                    elif operation == "call_tool":
                        server_id, tool_name, tool_args = args
                        session = await self._session(server_id, sessions, stack)
                        result = await session.call_tool(tool_name, tool_args)
                        structured = getattr(result, "structuredContent", None)
                        content = structured if structured is not None else _content_to_json(result.content)
                        value = {"server_id": server_id, "tool_name": tool_name, "content": content, "is_error": bool(getattr(result, "isError", False))}
                    else:
                        raise ValueError(f"Unknown persistent MCP operation: {operation}")
                except BaseException as exc:
                    future.set_exception(exc)
                else:
                    future.set_result(value)

    async def _session(self, server_id: str, sessions: dict[str, Any], stack: AsyncExitStack) -> Any:
        if server_id in sessions:
            return sessions[server_id]
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        config = core_local_server_configs()[server_id]
        params = StdioServerParameters(command=config.command, args=list(config.args), cwd=str(config.cwd), env=None)
        read, write = await stack.enter_async_context(stdio_client(params, errlog=sys.stderr))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        sessions[server_id] = session
        return session

    def _request(self, operation: str, *args: Any) -> Any:
        if self._closed:
            raise RuntimeError("Persistent MCP client is closed")
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        self._requests.put((operation, args, future))
        return future.result(timeout=self.timeout)

    def list_tools(self, server_id: str) -> list[dict[str, Any]]:
        return self._request("list_tools", server_id)

    def call_tool(self, server_id: str, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        return self._request("call_tool", server_id, tool_name, args)

    def close(self) -> None:
        if self._closed:
            return
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        self._requests.put(("close", (), future))
        future.result(timeout=self.timeout)
        self._thread.join(timeout=self.timeout)
        self._closed = True

    def __enter__(self) -> "PersistentMCPClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def _content_to_json(content: list[Any]) -> Any:
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

