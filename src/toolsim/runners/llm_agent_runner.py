"""LLM-driven tool-use runner for stateful/stateless experiments."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from toolsim.backends.base import BaseBackend
from toolsim.backends.mock_backend import MockBackend
from toolsim.core.environment import ToolEnvironment
from toolsim.core.tool_spec import ToolSpec
from toolsim.core.world_state import WorldState
from toolsim.evaluators.evaluator import CallLevelEvaluator, StateEvaluationResult, StateLevelEvaluator
from toolsim.execution.stateful_executor import ExecutorConfig, ExecutionRecord, StatefulExecutor, create_default_tool_registry
from toolsim.execution.stateful_tracer import TraceRecorder
from toolsim.llm.types import LLMClient, LLMMessage, LLMResponse
from toolsim.runners.experiment_runner import ExperimentResult
from toolsim.runners.stateless_baseline import STATELESS_TOOLS, StatelessStateLevelEvaluator


@dataclass
class LLMAgentConfig:
    max_steps: int = 8
    temperature: float = 0.0
    max_tokens: int | None = None
    include_tool_schemas: bool = True
    include_state_summary: bool = True
    stop_on_goals_passed: bool = True
    reject_finish_when_goals_unmet: bool = True
    response_format: dict[str, Any] = field(default_factory=lambda: {"type": "json_object"})


@dataclass
class LLMAgentStep:
    step_index: int
    messages: list[dict[str, str]]
    llm_response: dict[str, Any] | None
    parsed_action: dict[str, Any] | None
    execution_record: dict[str, Any] | None
    parse_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "messages": self.messages,
            "llm_response": self.llm_response,
            "parsed_action": self.parsed_action,
            "execution_record": self.execution_record,
            "parse_error": self.parse_error,
        }


@dataclass
class LLMAgentRunResult:
    mode: str
    task_name: str
    task_description: str
    experiment_result: ExperimentResult
    llm_steps: list[LLMAgentStep]
    parse_error_count: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_llm_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "task_name": self.task_name,
            "task_description": self.task_description,
            "experiment_result": self.experiment_result.to_dict(),
            "llm_steps": [step.to_dict() for step in self.llm_steps],
            "parse_error_count": self.parse_error_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_llm_latency_ms": self.total_llm_latency_ms,
        }


class LLMAgentRunner:
    """Run a closed-loop LLM policy against toolsim tools."""

    def __init__(
        self,
        client: LLMClient,
        *,
        config: LLMAgentConfig | None = None,
        executor_config: ExecutorConfig | None = None,
    ) -> None:
        self.client = client
        self.config = config or LLMAgentConfig()
        self.executor_config = executor_config or ExecutorConfig()

    def run(
        self,
        *,
        task_name: str,
        task_description: str,
        mode: str,
        initial_state: WorldState | None,
        goals: list[dict[str, Any]] | None,
        tool_registry: dict[str, ToolSpec] | None = None,
        state_evaluator: StateLevelEvaluator | None = None,
        backend: BaseBackend | None = None,
    ) -> LLMAgentRunResult:
        tools = tool_registry or tool_registry_for_mode(mode)
        evaluator = state_evaluator or _state_evaluator_for_mode(mode)
        backend = backend or MockBackend()
        state = backend.clone_state(initial_state) if initial_state is not None else backend.create_state()
        environment = ToolEnvironment(state=state, backend=backend)
        tracer = TraceRecorder()
        executor = StatefulExecutor(tools, tracer=tracer, config=self.executor_config, backend=backend)

        llm_steps: list[LLMAgentStep] = []
        history: list[dict[str, Any]] = []
        parse_error_count = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        llm_latency = 0.0
        debug_trace = os.environ.get("AGENTINFERKIT_LLM_TOOL_DEBUG") == "1"
        backend_name = backend.get_backend_name()

        for step_index in range(self.config.max_steps):
            messages = self._build_messages(
                task_name=task_name,
                task_description=task_description,
                mode=mode,
                tools=tools,
                history=history,
                state=state,
            )
            _debug_event(
                debug_trace,
                "LLM_START",
                task_name=task_name,
                backend_name=backend_name,
                step_index=step_index,
            )
            llm_started = time.perf_counter()
            response = self.client.complete(
                messages,
                response_format=self.config.response_format,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            _debug_event(
                debug_trace,
                "LLM_DONE",
                task_name=task_name,
                backend_name=backend_name,
                step_index=step_index,
                elapsed_ms=(time.perf_counter() - llm_started) * 1000.0,
            )
            prompt_tokens += response.usage.prompt_tokens
            completion_tokens += response.usage.completion_tokens
            total_tokens += response.usage.total_tokens
            llm_latency += response.latency_ms

            action, parse_error = parse_llm_tool_action(response.content)
            if parse_error is not None:
                _debug_event(
                    debug_trace,
                    "LLM_PARSE_ERROR",
                    task_name=task_name,
                    backend_name=backend_name,
                    step_index=step_index,
                )
                parse_error_count += 1
                history.append({
                    "type": "parse_error",
                    "content": response.content,
                    "error": parse_error,
                })
                llm_steps.append(LLMAgentStep(
                    step_index=step_index,
                    messages=[message.to_dict() for message in messages],
                    llm_response=response.to_dict(),
                    parsed_action=None,
                    execution_record=None,
                    parse_error=parse_error,
                ))
                continue

            requested_tool_name = action.get("tool_name")
            is_final_action = requested_tool_name in {None, "", "final"}
            if is_final_action:
                _debug_event(
                    debug_trace,
                    "FINAL_ACTION",
                    task_name=task_name,
                    backend_name=backend_name,
                    step_index=step_index,
                )
                llm_steps.append(LLMAgentStep(
                    step_index=step_index,
                    messages=[message.to_dict() for message in messages],
                    llm_response=response.to_dict(),
                    parsed_action=action,
                    execution_record=None,
                ))
                if (
                    self.config.reject_finish_when_goals_unmet
                    and goals is not None
                    and not _goals_passed(state, goals, evaluator)
                ):
                    history.append({
                        "type": "finish_rejected",
                        "reason": "finish=true was rejected because the target goals are not satisfied yet",
                    })
                    continue
                break

            tool_name = str(action.get("tool_name", ""))
            args = action.get("args")
            if not isinstance(args, dict):
                args = {}
            _debug_event(
                debug_trace,
                "TOOL_START",
                task_name=task_name,
                backend_name=backend_name,
                step_index=step_index,
                tool_name=tool_name,
            )
            tool_started = time.perf_counter()
            record = executor.execute(tool_name, state, args, environment=environment)
            _debug_event(
                debug_trace,
                "TOOL_DONE",
                task_name=task_name,
                backend_name=backend_name,
                step_index=step_index,
                tool_name=tool_name,
                elapsed_ms=(time.perf_counter() - tool_started) * 1000.0,
                success=record.success,
            )
            history.append({
                "type": "tool_call",
                "tool_name": tool_name,
                "args": args,
                "success": record.success,
                "error": record.error,
                "observation": record.observation,
            })
            llm_steps.append(LLMAgentStep(
                step_index=step_index,
                messages=[message.to_dict() for message in messages],
                llm_response=response.to_dict(),
                parsed_action=action,
                execution_record=record.to_dict(),
            ))

            if self.config.stop_on_goals_passed and _goals_passed(state, goals, evaluator):
                break

        trace = tracer.get_records()
        state_metrics: StateEvaluationResult | None = evaluator.evaluate(state, goals) if goals is not None else None
        experiment_result = ExperimentResult(
            final_state=state,
            trace=trace,
            call_metrics=CallLevelEvaluator().evaluate(trace),
            state_metrics=state_metrics,
            all_calls_succeeded=all(record.success for record in trace),
            final_state_hash=state.compute_hash(),
        )
        return LLMAgentRunResult(
            mode=mode,
            task_name=task_name,
            task_description=task_description,
            experiment_result=experiment_result,
            llm_steps=llm_steps,
            parse_error_count=parse_error_count,
            total_prompt_tokens=prompt_tokens,
            total_completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            total_llm_latency_ms=llm_latency,
        )

    def _build_messages(
        self,
        *,
        task_name: str,
        task_description: str,
        mode: str,
        tools: dict[str, ToolSpec],
        history: list[dict[str, Any]],
        state: WorldState,
    ) -> list[LLMMessage]:
        system = _system_prompt(mode)
        user_payload: dict[str, Any] = {
            "task_name": task_name,
            "task_description": task_description,
            "mode": mode,
            "history": history,
            "response_schema": {
                "thought": "short reasoning string",
                "tool_name": "tool name to call, or final",
                "args": "object arguments for the tool",
                "finish": "boolean, true only when no more tool call is needed",
            },
        }
        if self.config.include_tool_schemas:
            user_payload["tools"] = [_tool_schema(tool) for tool in tools.values()]
        if self.config.include_state_summary:
            user_payload["state_summary"] = _state_summary(state)
        return [
            LLMMessage("system", system),
            LLMMessage("user", json.dumps(user_payload, ensure_ascii=False, sort_keys=True)),
        ]


def parse_llm_tool_action(content: str) -> tuple[dict[str, Any], str | None]:
    text = content.strip()
    if not text:
        return {}, "empty LLM response"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            return {}, "LLM response did not contain a JSON object"
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            return {}, f"invalid JSON tool action: {exc}"
    if not isinstance(parsed, dict):
        return {}, "LLM JSON response was not an object"
    if "tool_name" not in parsed and parsed.get("finish") is not True:
        return {}, "LLM JSON response missing tool_name"
    if "args" in parsed and not isinstance(parsed["args"], dict):
        return {}, "LLM JSON response args must be an object"
    parsed.setdefault("args", {})
    parsed.setdefault("finish", False)
    return parsed, None


def _system_prompt(mode: str) -> str:
    return (
        "You are an autonomous tool-use agent inside AgentInferKit. "
        "Choose exactly one next tool call per turn. Respond only with valid json. "
        "The json object must contain thought, tool_name, args, and finish. "
        "Use finish=true only when the task goals are already satisfied. "
        f"The execution mode is {mode}. In stateful mode, search requires explicit indexing "
        "and tools may enforce workflow/state constraints. In stateless mode, some tools are "
        "more permissive and search may inspect current state directly."
        )


def _debug_event(
    enabled: bool,
    event: str,
    *,
    task_name: str,
    backend_name: str,
    step_index: int,
    tool_name: str | None = None,
    elapsed_ms: float | None = None,
    success: bool | None = None,
) -> None:
    if not enabled:
        return
    fields = [
        f"event={event}",
        f"task={task_name}",
        f"backend={backend_name}",
        f"step={step_index}",
    ]
    if tool_name is not None:
        fields.append(f"tool={tool_name}")
    if elapsed_ms is not None:
        fields.append(f"elapsed_ms={elapsed_ms:.2f}")
    if success is not None:
        fields.append(f"success={success}")
    print("[agent_debug] " + " ".join(fields), flush=True)


def tool_registry_for_mode(mode: str) -> dict[str, ToolSpec]:
    if mode == "stateless":
        return dict(STATELESS_TOOLS)
    if mode == "stateful":
        return create_default_tool_registry()
    raise ValueError("mode must be either 'stateful' or 'stateless'")


def _tool_registry_for_mode(mode: str) -> dict[str, ToolSpec]:
    return tool_registry_for_mode(mode)


def _state_evaluator_for_mode(mode: str) -> StateLevelEvaluator:
    if mode == "stateless":
        return StatelessStateLevelEvaluator()
    if mode == "stateful":
        return StateLevelEvaluator()
    raise ValueError("mode must be either 'stateful' or 'stateless'")


def _tool_schema(tool: ToolSpec) -> dict[str, Any]:
    metadata = tool.get_metadata()
    schema = metadata.input_schema or getattr(tool, "input_schema", {}) or {"type": "object", "properties": {}}
    return {
        "name": metadata.name,
        "description": metadata.description or getattr(tool, "description", ""),
        "domain": metadata.domain,
        "input_schema": schema,
    }


def _state_summary(state: WorldState) -> dict[str, Any]:
    return {
        "entity_counts": {entity_type: len(bucket) for entity_type, bucket in sorted(state.entities.items())},
        "resources": state.resources,
        "clock": state.clock,
        "version": state.version,
        "pending_effect_count": len(state.pending_effects),
    }


def _goals_passed(
    state: WorldState,
    goals: list[dict[str, Any]] | None,
    evaluator: StateLevelEvaluator,
) -> bool:
    if not goals:
        return False
    return evaluator.evaluate(state, goals).all_passed
