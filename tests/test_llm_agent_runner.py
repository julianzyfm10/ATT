"""Tests for LLM-backed tool-use runners without real API calls."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from toolsim.llm.types import LLMMessage, LLMResponse, LLMUsage
from toolsim.runners.comparison_runner import build_stateless_vs_stateful_cases
from toolsim.runners.llm_agent_runner import LLMAgentConfig, LLMAgentRunner, parse_llm_tool_action
from toolsim.runners.llm_experiment1 import LLMExperiment1Runner


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


def test_parse_llm_tool_action_accepts_json_object():
    action, error = parse_llm_tool_action('{"tool_name":"file.write","args":{"file_id":"f1","content":"hello"}}')

    assert error is None
    assert action["tool_name"] == "file.write"
    assert action["args"]["file_id"] == "f1"


def test_llm_agent_runner_executes_tool_loop_to_goal():
    client = FakeLLMClient([
        '{"tool_name":"file.write","args":{"file_id":"f1","content":"hello world"}}',
        '{"tool_name":"search.index","args":{"file_id":"f1"}}',
    ])
    case = build_stateless_vs_stateful_cases()[1]
    run = LLMAgentRunner(client, config=LLMAgentConfig(max_steps=4)).run(
        task_name=case.case_name,
        task_description=case.description,
        mode="stateful",
        initial_state=case.initial_state_stateful,
        goals=case.goals_stateful,
    )

    assert run.experiment_result.state_metrics is not None
    assert run.experiment_result.state_metrics.all_passed is True
    assert [record.tool_name for record in run.experiment_result.trace] == [
        "file.write",
        "search.index",
    ]
    assert run.total_tokens == 30


def test_llm_experiment1_runner_returns_stateful_and_stateless_metrics():
    client = FakeLLMClient([
        '{"tool_name":"file.write","args":{"file_id":"f1","content":"hello world"}}',
        '{"tool_name":"search.index","args":{"file_id":"f1"}}',
        '{"tool_name":"search.query","args":{"query":"hello"}}',
        '{"tool_name":"file.write","args":{"file_id":"f1","content":"hello world"}}',
        '{"tool_name":"search.query","args":{"query":"hello"}}',
    ])
    case = build_stateless_vs_stateful_cases()[1]
    result = LLMExperiment1Runner(client, config=LLMAgentConfig(max_steps=4)).run([case])

    assert result.model == "fake-deepseek"
    assert len(result.case_results) == 1
    assert {metric.mode for metric in result.metrics} == {"stateful", "stateless"}
    assert all(metric.final_state_correctness == 1.0 for metric in result.metrics)


def test_llm_experiment1_filters_visible_tools_by_case_domain():
    client = FakeLLMClient([
        '{"tool_name":"file.write","args":{"file_id":"f1","content":"hello world"}}',
        '{"tool_name":"search.index","args":{"file_id":"f1"}}',
        '{"tool_name":"file.write","args":{"file_id":"f1","content":"hello world"}}',
        '{"tool_name":"search.query","args":{"query":"hello"}}',
    ])
    case = build_stateless_vs_stateful_cases()[1]

    LLMExperiment1Runner(client, config=LLMAgentConfig(max_steps=4)).run([case])

    first_user_payload = json.loads(client.messages_by_call[0][1]["content"])
    first_tool_names = {tool["name"] for tool in first_user_payload["tools"]}
    assert first_tool_names == {"file.read", "file.write", "search.index", "search.query"}
    assert "GetUserToken" not in first_tool_names
    assert "add_contact" not in first_tool_names
