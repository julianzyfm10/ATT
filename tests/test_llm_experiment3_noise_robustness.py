"""Tests for Experiment 3."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from toolsim.core.world_state import WorldState
from toolsim.faults import FaultProfile
from toolsim.llm.types import LLMMessage, LLMResponse, LLMUsage
from toolsim.runners.experiment_runner import ExperimentRunner
from toolsim.runners.llm_agent_runner import LLMAgentConfig
from toolsim.runners.llm_experiment3_noise_robustness import (
    Experiment3Case,
    LLMExperiment3NoiseRobustnessRunner,
    _task_prompt,
)
from toolsim.tools.toolsandbox_runtime import ensure_toolsandbox_state


class FakeLLMClient:
    model = "fake-deepseek"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        response_format=None,
        max_tokens=None,
        temperature=None,
    ) -> LLMResponse:
        del messages, response_format, max_tokens, temperature
        content = self.responses[self.calls]
        self.calls += 1
        return LLMResponse(
            content=content,
            model=self.model,
            latency_ms=1.0,
            usage=LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )


def _file_timeout_case() -> Experiment3Case:
    initial = WorldState()
    goals = [{
        "type": "entity_field_equals",
        "entity_type": "file",
        "entity_id": "f",
        "field": "content",
        "expected": "ready",
    }]
    calls = [{"tool_name": "file.write", "args": {"file_id": "f", "content": "ready"}}]
    reference = ExperimentRunner().run(calls, initial_state=WorldState(), goals=goals).final_state
    return Experiment3Case(
        case_name="toolsandbox_noise::file::timeout",
        base_case_name="toolsandbox::file",
        scenario_name="file",
        source="toolsandbox_original_100",
        noise_type="timeout",
        noise_target_tool="file.write",
        noise_intensity=2,
        task_prompt="Write the requested file.",
        execution_goals=goals,
        state_goals=goals,
        semantic_observation_required=False,
        semantic_observation_tool=None,
        reference_observation=None,
        oracle_tool_calls=calls,
        noise_execution_calls=calls,
        fault_profile=FaultProfile(timeout_failures={"file.write": 2}),
        initial_state=initial,
        reference_final_state=reference,
        domain="file",
        difficulty_level="L1_SINGLE_TOOL",
        canonical_task_key="write file",
    )


def test_strategy_budgets_create_recovery_separation():
    case = _file_timeout_case()
    action = '{"tool_name":"file.write","args":{"file_id":"f","content":"ready"}}'
    direct = LLMExperiment3NoiseRobustnessRunner(
        FakeLLMClient([action]),
        config=LLMAgentConfig(max_steps=16, temperature=None),
        attempts_per_case=1,
    ).run_case(case, "direct")
    assert direct.success_at_1 is False
    assert direct.attempts[0].step_budget == 1
    assert direct.attempts[0].injected_fault_count == 1

    react = LLMExperiment3NoiseRobustnessRunner(
        FakeLLMClient([action, action, action, '{"tool_name":"final","args":{},"finish":true}']),
        config=LLMAgentConfig(max_steps=16, temperature=None),
        attempts_per_case=1,
    ).run_case(case, "react")
    assert react.success_at_1 is True
    assert react.attempts[0].step_budget == 4
    assert react.attempts[0].injected_fault_count == 2
    assert react.attempts[0].explicit_retry_detected is True
    assert react.attempts[0].recovery_detected is True


def test_semantic_observation_requires_clean_requery():
    initial = WorldState()
    ensure_toolsandbox_state(initial)
    calls = [{"tool_name": "get_cellular_service_status", "args": {}}]
    reference_run = ExperimentRunner().run(calls, initial_state=WorldState.from_dict(initial.to_dict()), goals=[])
    reference_observation = reference_run.trace[-1].observation
    case = Experiment3Case(
        case_name="toolsandbox_noise::cellular::vague_observation",
        base_case_name="toolsandbox::cellular",
        scenario_name="cellular",
        source="toolsandbox_original_100",
        noise_type="vague_observation",
        noise_target_tool="get_cellular_service_status",
        noise_intensity=2,
        task_prompt="Report cellular status.",
        execution_goals=[],
        state_goals=[],
        semantic_observation_required=True,
        semantic_observation_tool="get_cellular_service_status",
        reference_observation=reference_observation,
        oracle_tool_calls=calls,
        noise_execution_calls=calls,
        fault_profile=FaultProfile(vague_observation_failures={"get_cellular_service_status": 2}),
        initial_state=initial,
        reference_final_state=reference_run.final_state,
        domain="device_settings",
        difficulty_level="L1_SINGLE_TOOL",
        canonical_task_key="cellular status",
    )
    action = '{"tool_name":"get_cellular_service_status","args":{}}'
    direct = LLMExperiment3NoiseRobustnessRunner(
        FakeLLMClient([action]),
        config=LLMAgentConfig(max_steps=16),
        attempts_per_case=1,
    ).run_case(case, "direct")
    assert direct.attempts[0].semantic_observation_correct is False
    assert direct.success_at_1 is False

    react = LLMExperiment3NoiseRobustnessRunner(
        FakeLLMClient([action, action, action, '{"tool_name":"final","args":{},"finish":true}']),
        config=LLMAgentConfig(max_steps=16),
        attempts_per_case=1,
    ).run_case(case, "react")
    assert react.attempts[0].semantic_observation_correct is True
    assert react.success_at_1 is True


def test_prompt_does_not_leak_noise_or_evaluator_goals():
    case = _file_timeout_case()
    prompt = _task_prompt(case, "self_refine")
    assert "timeout" not in prompt.lower()
    assert "Target goals:" not in prompt
    assert "expected" not in prompt
    assert "Reasoning strategy: self_refine" in prompt

