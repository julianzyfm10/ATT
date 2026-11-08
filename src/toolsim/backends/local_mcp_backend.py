"""Real local MCP backend with persistent per-case stdio sessions."""

from __future__ import annotations

from typing import Any

from toolsim.backends.sandbox_backend import SandboxBackend
from toolsim.core.world_state import WorldState
from toolsim.mcp_live.persistent_worker import PersistentMCPClient


class LocalMCPBackend(SandboxBackend):
    def __init__(self, session_id: str | None = None, *, timeout: float = 60.0) -> None:
        super().__init__(session_id=session_id)
        self.client = PersistentMCPClient(timeout=timeout)

    def get_backend_name(self) -> str:
        return "local_mcp"

    def create_state(self) -> WorldState:
        state = super().create_state()
        state.resources["local_mcp_session"] = self.session_id
        state.policies.setdefault("backend", {})
        state.policies["backend"].update({"realism": "local_mcp", "network": "offline", "transport": "stdio"})
        return state

    def clone_state(self, state: WorldState) -> WorldState:
        cloned = super().clone_state(state)
        cloned.resources["local_mcp_session"] = self.session_id
        cloned.policies.setdefault("backend", {})
        cloned.policies["backend"].update({"realism": "local_mcp", "network": "offline", "transport": "stdio"})
        return cloned

    def list_core_local_tools(self, server_id: str) -> list[dict[str, Any]]:
        return self.client.list_tools(server_id)

    def call_core_local_tool(
        self,
        state: WorldState,
        server_id: str,
        tool_name: str,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        del state
        result = self.client.call_tool(server_id, tool_name, args)
        is_error = bool(result.get("is_error"))
        return {
            "success": not is_error,
            "output": result.get("content"),
            "error": str(result.get("content")) if is_error else None,
        }

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "LocalMCPBackend":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


