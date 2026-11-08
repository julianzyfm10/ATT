"""Deterministic oracle runner used to qualify the Core Local environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from toolsim.backends.local_mcp_backend import LocalMCPBackend
from toolsim.backends.mock_backend import MockBackend
from toolsim.backends.sandbox_backend import SandboxBackend
from toolsim.core.environment import ToolEnvironment
from toolsim.core.world_state import WorldState
from toolsim.execution.stateful_executor import ExecutorConfig, StatefulExecutor
from toolsim.execution.stateful_tracer import TraceRecorder
from toolsim.runners.core_local_dataset import CoreLocalCase, CoreLocalStateEvaluator, dependency_compliance
from toolsim.tools.core_local_mcp_tools import registry_for_servers


@dataclass
class CoreLocalOracleResult:
    case_id: str
    backend: str
    final_correct: bool
    dependency_compliant: bool
    calls: int
    failed_calls: int
    goal_details: list[dict[str, Any]]
    trace: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "backend": self.backend,
            "final_correct": self.final_correct,
            "dependency_compliant": self.dependency_compliant,
            "calls": self.calls,
            "failed_calls": self.failed_calls,
            "goal_details": self.goal_details,
            "trace": self.trace,
        }


def backend_for_name(name: str, case_id: str):
    session_id = f"core_{case_id}_{name}"[:96]
    if name == "mock":
        return MockBackend()
    if name == "sandbox":
        return SandboxBackend(session_id=session_id)
    if name == "local_mcp":
        return LocalMCPBackend(session_id=session_id, timeout=90.0)
    raise ValueError(f"Unknown Core Local backend: {name}")


def run_oracle_case(case: CoreLocalCase, backend_name: str) -> CoreLocalOracleResult:
    backend = backend_for_name(backend_name, case.case_id)
    try:
        state = backend.create_state()
        tracer = TraceRecorder()
        registry = registry_for_servers(case.servers)
        executor = StatefulExecutor(registry, tracer=tracer, config=ExecutorConfig(), backend=backend)
        environment = ToolEnvironment(state=state, backend=backend)
        step_values: dict[str, dict[str, Any]] = {}
        for oracle_step in case.oracle_plan:
            args = _resolve(oracle_step.get("args", {}), step_values)
            tool_name = f"mcp.{oracle_step['server_id']}.{oracle_step['tool_name']}"
            record = executor.execute(tool_name, state, args, environment=environment)
            step_values[oracle_step["step_id"]] = {
                "output": record.observation.get("output"),
                "error": record.error,
                "success": record.success,
            }
        evaluator = CoreLocalStateEvaluator(case)
        evaluation = evaluator.evaluate(state, [])
        trace = tracer.get_records()
        return CoreLocalOracleResult(
            case_id=case.case_id,
            backend=backend_name,
            final_correct=evaluation.all_passed,
            dependency_compliant=dependency_compliance(case, state),
            calls=len(trace),
            failed_calls=sum(not item.success for item in trace),
            goal_details=[item.to_dict() for item in evaluation.details],
            trace=[item.to_dict() for item in trace],
        )
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()


def _resolve(value: Any, steps: dict[str, dict[str, Any]]) -> Any:
    if isinstance(value, dict) and "$ref" in value:
        reference = str(value["$ref"])
        step_id, _, path = reference.partition(".")
        current: Any = steps[step_id]
        for part in path.split(".") if path else []:
            current = current[part]
        transform = value.get("transform")
        if transform == "hour":
            text = str(current)
            return int(text.split("T", 1)[-1].split(":", 1)[0])
        return current
    if isinstance(value, dict):
        return {key: _resolve(item, steps) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve(item, steps) for item in value]
    return value

