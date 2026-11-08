"""Synthetic dependency-locked tools for Experiment 2 difficulty curves."""

from __future__ import annotations

from typing import Any

from toolsim.core.tool_spec import ToolExecutionResult, ToolMetadata, ToolSpec

SECRET_ENTITY = "dependency_secret"
LOOKUP_ENTITY = "dependency_lookup"
PREPARED_ENTITY = "dependency_prepared"
RESULT_ENTITY = "dependency_result"
AUDIT_ENTITY = "dependency_audit"


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required}


class DependencyLookupTool(ToolSpec):
    tool_name = "dependency.lookup"
    description = (
        "Look up the hidden dependency token and target value for a synthetic dependency case. "
        "Downstream dependency tools require the token returned by this observation."
    )
    input_schema = _schema({
        "case_id": {"type": "string", "description": "Synthetic dependency case id."},
    }, ["case_id"])
    metadata = ToolMetadata(
        name=tool_name,
        version="0.1",
        domain="dependency",
        description=description,
        input_schema=input_schema,
    )

    def execute(self, state_or_context: Any, args: dict[str, Any]) -> ToolExecutionResult:
        state = self.get_state_from_input(state_or_context)
        case_id = args.get("case_id")
        if not case_id:
            return ToolExecutionResult(success=False, error="Missing required argument: case_id")
        secret = state.get_entity(SECRET_ENTITY, case_id)
        if secret is None:
            return ToolExecutionResult(success=False, error=f"Unknown dependency case: {case_id!r}")
        lookup = {
            "case_id": case_id,
            "dependency_token": secret["dependency_token"],
            "target_value": secret["target_value"],
            "requires_prepare": bool(secret.get("requires_prepare")),
            "requires_audit": bool(secret.get("requires_audit")),
        }
        state.set_entity(LOOKUP_ENTITY, case_id, lookup)
        observation = dict(lookup)
        if secret.get("hide_prepare_hint"):
            observation.pop("requires_prepare", None)
        if secret.get("hide_audit_hint"):
            observation.pop("requires_audit", None)
        if not secret.get("hide_prepare_hint"):
            observation["next_required_tool"] = "dependency.prepare" if lookup["requires_prepare"] else "dependency.commit"
        return ToolExecutionResult(
            success=True,
            observation=observation,
            state_changed=True,
        )


class DependencyPrepareTool(ToolSpec):
    tool_name = "dependency.prepare"
    description = "Prepare a dependency case before commit. Requires the token returned by dependency.lookup."
    input_schema = _schema({
        "case_id": {"type": "string"},
        "dependency_token": {"type": "string"},
    }, ["case_id", "dependency_token"])
    metadata = ToolMetadata(
        name=tool_name,
        version="0.1",
        domain="dependency",
        description=description,
        input_schema=input_schema,
    )

    def execute(self, state_or_context: Any, args: dict[str, Any]) -> ToolExecutionResult:
        state = self.get_state_from_input(state_or_context)
        case_id = args.get("case_id")
        token = args.get("dependency_token")
        secret = _secret_or_error(state, case_id)
        if isinstance(secret, ToolExecutionResult):
            return secret
        token_error = _token_error(state, case_id, token)
        if token_error:
            return token_error
        prepared = {
            "case_id": case_id,
            "dependency_token": token,
            "prepared": True,
            "target_value": secret["target_value"],
        }
        state.set_entity(PREPARED_ENTITY, case_id, prepared)
        return ToolExecutionResult(
            success=True,
            observation={**prepared, "next_required_tool": "dependency.commit"},
            state_changed=True,
        )


class DependencyCommitTool(ToolSpec):
    tool_name = "dependency.commit"
    description = (
        "Commit the dependency result. Requires dependency.lookup first, and also dependency.prepare "
        "when the lookup observation says requires_prepare=true."
    )
    input_schema = _schema({
        "case_id": {"type": "string"},
        "dependency_token": {"type": "string"},
        "value": {"type": "string", "description": "Use the target_value returned by dependency.lookup."},
    }, ["case_id", "dependency_token", "value"])
    metadata = ToolMetadata(
        name=tool_name,
        version="0.1",
        domain="dependency",
        description=description,
        input_schema=input_schema,
    )

    def execute(self, state_or_context: Any, args: dict[str, Any]) -> ToolExecutionResult:
        state = self.get_state_from_input(state_or_context)
        case_id = args.get("case_id")
        token = args.get("dependency_token")
        value = args.get("value")
        secret = _secret_or_error(state, case_id)
        if isinstance(secret, ToolExecutionResult):
            return secret
        token_error = _token_error(state, case_id, token)
        if token_error:
            return token_error
        if secret.get("requires_prepare") and state.get_entity(PREPARED_ENTITY, case_id) is None:
            observation = None
            if secret.get("repair_hint_on_prepare_failure"):
                observation = {
                    "case_id": case_id,
                    "repair_tool": "dependency.prepare",
                    "retry_tool": "dependency.commit",
                    "reason": "prepare_missing",
                }
            return ToolExecutionResult(
                success=False,
                error="dependency.prepare must succeed before dependency.commit",
                observation=observation,
            )
        if value != secret["target_value"]:
            return ToolExecutionResult(success=False, error="value must match target_value from dependency.lookup")
        result = {
            "case_id": case_id,
            "value": value,
            "dependency_token": token,
            "committed": True,
        }
        state.set_entity(RESULT_ENTITY, case_id, result)
        observation = dict(result)
        if secret.get("requires_audit") and not secret.get("hide_audit_hint"):
            observation["next_required_tool"] = "dependency.audit"
        elif not secret.get("requires_audit"):
            observation["next_required_tool"] = "final"
        return ToolExecutionResult(
            success=True,
            observation=observation,
            state_changed=True,
        )


class DependencyAuditTool(ToolSpec):
    tool_name = "dependency.audit"
    description = "Audit a committed dependency result. Requires dependency.commit to have succeeded first."
    input_schema = _schema({
        "case_id": {"type": "string"},
    }, ["case_id"])
    metadata = ToolMetadata(
        name=tool_name,
        version="0.1",
        domain="dependency",
        description=description,
        input_schema=input_schema,
    )

    def execute(self, state_or_context: Any, args: dict[str, Any]) -> ToolExecutionResult:
        state = self.get_state_from_input(state_or_context)
        case_id = args.get("case_id")
        if not case_id:
            return ToolExecutionResult(success=False, error="Missing required argument: case_id")
        result = state.get_entity(RESULT_ENTITY, case_id)
        if result is None:
            return ToolExecutionResult(success=False, error="dependency.commit must succeed before dependency.audit")
        audit = {"case_id": case_id, "status": "passed", "audited_value": result.get("value")}
        state.set_entity(AUDIT_ENTITY, case_id, audit)
        return ToolExecutionResult(success=True, observation=audit, state_changed=True)


def _secret_or_error(state: Any, case_id: Any) -> dict[str, Any] | ToolExecutionResult:
    if not case_id:
        return ToolExecutionResult(success=False, error="Missing required argument: case_id")
    secret = state.get_entity(SECRET_ENTITY, case_id)
    if secret is None:
        return ToolExecutionResult(success=False, error=f"Unknown dependency case: {case_id!r}")
    return secret


def _token_error(state: Any, case_id: str, token: Any) -> ToolExecutionResult | None:
    lookup = state.get_entity(LOOKUP_ENTITY, case_id)
    if lookup is None:
        return ToolExecutionResult(success=False, error="dependency.lookup must succeed before this tool")
    if token != lookup.get("dependency_token"):
        return ToolExecutionResult(success=False, error="dependency_token must match the token returned by dependency.lookup")
    return None


DEPENDENCY_TOOLS: dict[str, ToolSpec] = {
    tool.tool_name: tool
    for tool in [
        DependencyLookupTool(),
        DependencyPrepareTool(),
        DependencyCommitTool(),
        DependencyAuditTool(),
    ]
}
