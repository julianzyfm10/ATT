from __future__ import annotations

from typing import Any, Optional

from toolsim.backends.sandbox_backend import SandboxBackend
from toolsim.core.world_state import WorldState
from toolsim.mcp_live.client import call_mcp_tool


class MCPLiveBackend(SandboxBackend):
    """Backend that calls selected MCP-Bench servers through MCP stdio."""

    def __init__(self, session_id: Optional[str] = None) -> None:
        super().__init__(session_id=session_id)

    def get_backend_name(self) -> str:
        return "live_mcp"

    def create_state(self) -> WorldState:
        state = super().create_state()
        state.resources["mcp_live_session"] = self.session_id
        state.policies.setdefault("backend", {})
        state.policies["backend"]["realism"] = "live_mcp"
        state.policies["backend"]["mcp_source"] = "Accenture/mcp-bench"
        return state

    def clone_state(self, state: WorldState) -> WorldState:
        cloned = super().clone_state(state)
        cloned.resources["mcp_live_session"] = self.session_id
        cloned.policies.setdefault("backend", {})
        cloned.policies["backend"]["realism"] = "live_mcp"
        cloned.policies["backend"]["mcp_source"] = "Accenture/mcp-bench"
        return cloned

    def call_mcp_tool(self, state: WorldState, server_id: str, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        result = call_mcp_tool(server_id, tool_name, args)
        return {
            "success": not bool(result.get("is_error")),
            "state_changed": False,
            "observation": {
                "mcp": True,
                "server_id": server_id,
                "tool_name": tool_name,
                "input": dict(args),
                "output": result.get("content"),
                "exception": None if not result.get("is_error") else result.get("content"),
            },
            "error": None if not result.get("is_error") else str(result.get("content")),
        }
