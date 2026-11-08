"""Tests for LLM-backed Experiment 2 dependency curves without real API calls."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from toolsim.llm.types import LLMMessage, LLMResponse, LLMUsage
from toolsim.core.world_state import WorldState
from toolsim.runners.dependency_curve import (
    L1_SINGLE_TOOL,
    L2_EXPLICIT_DEP,
    DependencyCurveCase,
    build_synthetic_dependency_curve_cases,
)
from toolsim.runners.llm_agent_runner import LLMAgentConfig
from toolsim.runners.llm_experiment2_dependency_curve import (
    LLMExperiment2DependencyCurveRunner,
    dependency_case_tool_registry,
    render_llm_experiment2_markdown,
)


class FakeLLMClient:
    model = "fake-deepseek"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.messages_by_call = []

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        response_format=None,
        max_tokens=None,
        temperature=None,
    ) -> LLMResponse:
        del response_format, max_tokens, temperature
        self.messages_by_call.append([message.to_dict() for message in messages])
        content = self.responses[self.calls]
        self.calls += 1
        return LLMResponse(
            content=content,
            model=self.model,
            latency_ms=1.0,
            usage=LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )


def test_llm_experiment2_runner_executes_dependency_case():
    case = next(
        case for case in build_synthetic_dependency_curve_cases()
        if case.case_name == "l2_locked_file_release"
    )
    client = FakeLLMClient([
        '{"tool_name":"dependency.lookup","args":{"case_id":"l2_locked_file_release"}}',
        '{"tool_name":"dependency.prepare","args":{"case_id":"l2_locked_file_release","dependency_token":"tok_file_713"}}',
        '{"tool_name":"dependency.commit","args":{"case_id":"l2_locked_file_release","dependency_token":"tok_file_713","value":"release-note-approved"}}',
        '{"finish":true,"tool_name":"final","args":{}}',
    ])

    result = LLMExperiment2DependencyCurveRunner(
        client,
        config=LLMAgentConfig(max_steps=4),
    ).run([case], strategies=["cot"])

    assert result.model == "fake-deepseek"
    assert len(result.case_results) == 1
    case_result = result.case_results[0]
    assert case_result.final_state_correct is True
    assert case_result.dependency_correct is True
    assert case_result.dependency_completion_rate == 1.0
    assert case_result.difficulty_level == L2_EXPLICIT_DEP
    assert case_result.strategy == "cot"

    all_l2 = next(
        metric for metric in result.group_metrics
        if metric.source == "all" and metric.strategy == "cot" and metric.difficulty_level == L2_EXPLICIT_DEP
    )
    assert all_l2.total_cases == 1
    assert all_l2.final_state_correctness == 1.0
    assert all_l2.dependency_correctness_rate == 1.0
    assert all_l2.average_trajectory_length == 3.0
    assert all_l2.average_llm_steps == 4.0


def test_llm_experiment2_filters_visible_tools_by_dependency_case():
    case = next(
        case for case in build_synthetic_dependency_curve_cases()
        if case.case_name == "l2_locked_file_release"
    )
    registry = dependency_case_tool_registry(case)

    assert set(registry) == {"dependency.commit", "dependency.lookup", "dependency.prepare"}


def test_llm_experiment2_exposes_end_conversation_content_schema():
    case = DependencyCurveCase(
        case_name="toolsandbox::final_response",
        difficulty_level=L1_SINGLE_TOOL,
        dependency_type="none",
        description="Reply to the user.",
        goals=[{
            "type": "toolsandbox_record_exists",
            "entity_type": "sandbox",
            "fields": {
                "sender": "AGENT",
                "recipient": "USER",
                "content": "Done",
            },
        }],
        oracle_tool_calls=[{"tool_name": "end_conversation", "args": {"content": "Done"}}],
        strategy_tool_calls={},
        initial_state=WorldState(),
    )

    registry = dependency_case_tool_registry(case)
    schema = registry["end_conversation"].get_metadata().input_schema

    assert "content" in schema["properties"]
    assert "content" in schema["required"]


def test_llm_experiment2_report_renders_dependency_tables():
    case = next(
        case for case in build_synthetic_dependency_curve_cases()
        if case.case_name == "l2_locked_file_release"
    )
    client = FakeLLMClient([
        '{"tool_name":"dependency.lookup","args":{"case_id":"l2_locked_file_release"}}',
        '{"tool_name":"dependency.prepare","args":{"case_id":"l2_locked_file_release","dependency_token":"tok_file_713"}}',
        '{"tool_name":"dependency.commit","args":{"case_id":"l2_locked_file_release","dependency_token":"tok_file_713","value":"release-note-approved"}}',
        '{"finish":true,"tool_name":"final","args":{}}',
    ])
    result = LLMExperiment2DependencyCurveRunner(
        client,
        config=LLMAgentConfig(max_steps=4),
    ).run([case], strategies=["cot"])

    report = render_llm_experiment2_markdown(result)

    assert "# Experiment 2 LLM Cross-tool Dependency Curve Report" in report
    assert "## Dependency-aware Difficulty Summary" in report
    assert "Dependency-aware Correctness" in report
    assert "## Metrics By Difficulty" in report
    assert "l2_locked_file_release" in report


def test_llm_experiment2_prompt_contains_dependency_context():
    case = next(
        case for case in build_synthetic_dependency_curve_cases()
        if case.case_name == "l2_locked_file_release"
    )
    client = FakeLLMClient([
        '{"tool_name":"dependency.lookup","args":{"case_id":"l2_locked_file_release"}}',
        '{"tool_name":"dependency.prepare","args":{"case_id":"l2_locked_file_release","dependency_token":"tok_file_713"}}',
        '{"tool_name":"dependency.commit","args":{"case_id":"l2_locked_file_release","dependency_token":"tok_file_713","value":"release-note-approved"}}',
        '{"finish":true,"tool_name":"final","args":{}}',
    ])

    LLMExperiment2DependencyCurveRunner(client, config=LLMAgentConfig(max_steps=4)).run([case], strategies=["cot"])

    first_payload = json.loads(client.messages_by_call[0][1]["content"])
    assert "cross-tool dependency" in first_payload["task_description"]
    assert "L2_EXPLICIT_DEP" in first_payload["task_description"]


def test_llm_experiment2_prompt_surfaces_final_response_goal():
    case = DependencyCurveCase(
        case_name="toolsandbox::final_response",
        difficulty_level=L1_SINGLE_TOOL,
        dependency_type="none",
        description="Reply to the user.",
        goals=[{
            "type": "toolsandbox_record_exists",
            "entity_type": "sandbox",
            "fields": {
                "sender": "AGENT",
                "recipient": "USER",
                "content": "Done",
            },
        }],
        oracle_tool_calls=[{"tool_name": "end_conversation", "args": {"content": "Done"}}],
        strategy_tool_calls={},
        initial_state=WorldState(),
    )
    client = FakeLLMClient(['{"tool_name":"end_conversation","args":{"content":"Done"}}'])

    LLMExperiment2DependencyCurveRunner(client, config=LLMAgentConfig(max_steps=1)).run([case], strategies=["direct"])

    first_payload = json.loads(client.messages_by_call[0][1]["content"])
    assert "Final response requirements" in first_payload["task_description"]
    assert "'content': 'Done'" in first_payload["task_description"]


def test_llm_experiment2_rejects_early_finish_when_goals_unmet():
    case = DependencyCurveCase(
        case_name="toolsandbox::early_finish",
        difficulty_level=L1_SINGLE_TOOL,
        dependency_type="none",
        description="Reply to the user.",
        goals=[{
            "type": "toolsandbox_record_exists",
            "entity_type": "sandbox",
            "fields": {
                "sender": "AGENT",
                "recipient": "USER",
                "content": "Done",
            },
        }],
        oracle_tool_calls=[{"tool_name": "end_conversation", "args": {"content": "Done"}}],
        strategy_tool_calls={},
        initial_state=WorldState(),
    )
    client = FakeLLMClient([
        '{"finish":true,"tool_name":"final","args":{}}',
        '{"tool_name":"end_conversation","args":{"content":"Done","recipient":"USER"}}',
        '{"finish":true,"tool_name":"final","args":{}}',
    ])

    result = LLMExperiment2DependencyCurveRunner(client, config=LLMAgentConfig(max_steps=2)).run([case], strategies=["self_refine"])

    assert result.case_results[0].final_state_correct is True
    second_payload = json.loads(client.messages_by_call[1][1]["content"])
    assert second_payload["history"][0]["type"] == "finish_rejected"


def test_llm_experiment2_runner_keeps_strategy_as_metric_dimension():
    case = next(
        case for case in build_synthetic_dependency_curve_cases()
        if case.case_name == "l1_file_write"
    )
    client = FakeLLMClient([
        '{"tool_name":"file.write","args":{"file_id":"l1_file","content":"ready"}}',
        '{"finish":true,"tool_name":"final","args":{}}',
        '{"tool_name":"file.write","args":{"file_id":"l1_file","content":"ready"}}',
        '{"finish":true,"tool_name":"final","args":{}}',
    ])

    result = LLMExperiment2DependencyCurveRunner(
        client,
        config=LLMAgentConfig(max_steps=2),
    ).run([case], strategies=["direct", "self_refine"])

    assert [item.strategy for item in result.case_results] == ["direct", "self_refine"]
    summary_keys = {(summary.source, summary.strategy) for summary in result.difficulty_summaries}
    assert ("all", "direct") in summary_keys
    assert ("all", "self_refine") in summary_keys
