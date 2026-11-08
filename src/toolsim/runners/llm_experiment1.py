"""LLM-backed Experiment 1: stateful vs stateless comparison."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from toolsim.adapters.toolsandbox_adapter import ToolSandboxConvertedCase
from toolsim.core.world_state import WorldState
from toolsim.core.tool_spec import ToolSpec
from toolsim.llm.types import LLMClient
from toolsim.runners.comparison_runner import (
    ComparisonCase,
    build_toolsandbox_stateless_vs_stateful_cases,
)
from toolsim.runners.experiment2_cases import DEFAULT_CASES
from toolsim.runners.llm_agent_runner import LLMAgentConfig, LLMAgentRunResult, LLMAgentRunner, tool_registry_for_mode


@dataclass
class LLMExperiment1CaseResult:
    case_name: str
    source: str
    stateful: LLMAgentRunResult
    stateless: LLMAgentRunResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_name": self.case_name,
            "source": self.source,
            "stateful": self.stateful.to_dict(),
            "stateless": self.stateless.to_dict(),
        }


@dataclass
class LLMExperiment1Metrics:
    mode: str
    total_cases: int
    success_rate: float
    invalid_call_rate: float
    recovery_rate: float | None
    recovery_opportunities: int
    recovered_cases: int
    average_trajectory_length: float
    final_state_correctness: float
    parse_error_rate: float
    length_truncation_rate: float
    average_llm_steps: float
    average_tokens: float
    average_llm_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "total_cases": self.total_cases,
            "success_rate": self.success_rate,
            "invalid_call_rate": self.invalid_call_rate,
            "recovery_rate": self.recovery_rate,
            "recovery_opportunities": self.recovery_opportunities,
            "recovered_cases": self.recovered_cases,
            "average_trajectory_length": self.average_trajectory_length,
            "final_state_correctness": self.final_state_correctness,
            "parse_error_rate": self.parse_error_rate,
            "length_truncation_rate": self.length_truncation_rate,
            "average_llm_steps": self.average_llm_steps,
            "average_tokens": self.average_tokens,
            "average_llm_latency_ms": self.average_llm_latency_ms,
        }


@dataclass
class LLMExperiment1Result:
    model: str
    case_results: list[LLMExperiment1CaseResult]
    metrics: list[LLMExperiment1Metrics]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "case_results": [result.to_dict() for result in self.case_results],
            "metrics": [metric.to_dict() for metric in self.metrics],
        }


class LLMExperiment1Runner:
    """Run Experiment 1 with an LLM policy instead of oracle trajectories."""

    def __init__(self, client: LLMClient, *, config: LLMAgentConfig | None = None) -> None:
        self.client = client
        self.config = config or LLMAgentConfig()

    def run(self, cases: list[ComparisonCase]) -> LLMExperiment1Result:
        agent = LLMAgentRunner(self.client, config=self.config)
        results: list[LLMExperiment1CaseResult] = []
        for case in cases:
            source = "toolsandbox" if case.case_name.startswith("toolsandbox::") else "synthetic"
            stateful = agent.run(
                task_name=case.case_name,
                task_description=_task_prompt(case, "stateful"),
                mode="stateful",
                initial_state=case.initial_state_stateful,
                goals=case.goals_stateful,
                tool_registry=_case_tool_registry(case, "stateful"),
            )
            stateless = agent.run(
                task_name=case.case_name,
                task_description=_task_prompt(case, "stateless"),
                mode="stateless",
                initial_state=case.initial_state_stateless,
                goals=case.goals_stateless,
                tool_registry=_case_tool_registry(case, "stateless"),
            )
            results.append(LLMExperiment1CaseResult(
                case_name=case.case_name,
                source=source,
                stateful=stateful,
                stateless=stateless,
            ))
        return LLMExperiment1Result(
            model=getattr(self.client, "model", "unknown"),
            case_results=results,
            metrics=[
                _compute_mode_metrics(results, "stateful"),
                _compute_mode_metrics(results, "stateless"),
            ],
        )


def build_llm_experiment1_cases(
    *,
    cases_path: str | Path = DEFAULT_CASES,
    limit: int | None = None,
) -> list[ComparisonCase]:
    """Build Experiment 1 from the shared fixed ToolSandbox task manifest."""
    path = Path(cases_path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.resolve() == DEFAULT_CASES.resolve() and len(rows) != 100:
        raise ValueError(f"Expected 100 original ToolSandbox cases, found {len(rows)}")
    converted = [
        ToolSandboxConvertedCase(
            scenario_name=str(row["scenario_name"]),
            user_request=str(row["user_request"]),
            domain=str(row["domain"]),
            required_tools=[str(call["tool_name"]) for call in row["oracle_tool_calls"]],
            categories=list(row.get("categories") or []),
            initial_state=WorldState.from_dict(row["initial_state"]),
            oracle_tool_calls=list(row["oracle_tool_calls"]),
            goals=list(row["goals"]),
            minefield_goals=[],
        )
        for row in rows
        if not row.get("is_augmented_variant", False)
    ]
    cases = build_toolsandbox_stateless_vs_stateful_cases(converted)
    return cases[:limit] if limit is not None else cases


def render_llm_experiment1_markdown(result: LLMExperiment1Result) -> str:
    lines = [
        "# Experiment 1 LLM Stateful vs Stateless Report",
        "",
        f"- Model: `{result.model}`",
        f"- Cases: {len(result.case_results)}",
        "",
        "## Metrics",
        "",
        "| Mode | Cases | Success Rate | Invalid Call Rate | Recovery Rate | Avg Trajectory Length | Final State Correctness |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in result.metrics:
        lines.append(
            f"| {metric.mode} | {metric.total_cases} | {_pct(metric.success_rate)} | "
            f"{_pct(metric.invalid_call_rate)} | {_recovery_label(metric.recovery_rate)} | "
            f"{metric.average_trajectory_length:.2f} | {_pct(metric.final_state_correctness)} |"
        )

    lines.extend([
        "",
        "## LLM Diagnostics",
        "",
        "| Mode | Avg LLM Turns | Parse Errors | Length Stops | Avg Tokens | Avg LLM Latency ms |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for metric in result.metrics:
        lines.append(
            f"| {metric.mode} | {metric.average_llm_steps:.2f} | "
            f"{_pct(metric.parse_error_rate)} | {_pct(metric.length_truncation_rate)} | "
            f"{metric.average_tokens:.1f} | {metric.average_llm_latency_ms:.2f} |"
        )

    lines.extend(["", "## Case Outcomes", ""])
    lines.append("| Case | Source | Stateful Correct | Stateless Correct | Stateful Traj Len | Stateless Traj Len | Stateful LLM Turns | Stateless LLM Turns |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for item in result.case_results:
        stateful_correct = _final_correct(item.stateful)
        stateless_correct = _final_correct(item.stateless)
        lines.append(
            f"| {item.case_name} | {item.source} | {stateful_correct} | {stateless_correct} | "
            f"{len(item.stateful.experiment_result.trace)} | {len(item.stateless.experiment_result.trace)} | "
            f"{len(item.stateful.llm_steps)} | {len(item.stateless.llm_steps)} |"
        )
    return "\n".join(lines)


def _task_prompt(case: ComparisonCase, mode: str) -> str:
    goals = case.goals_stateful if mode == "stateful" else case.goals_stateless
    return (
        f"{case.description}\n\n"
        "Use the available tools to satisfy the target state. "
        "Do not explain in natural language; choose tools until the state satisfies the goals.\n\n"
        f"Target goals: {goals}"
    )


def _case_tool_registry(case: ComparisonCase, mode: str) -> dict[str, ToolSpec]:
    base_registry = tool_registry_for_mode(mode)
    planned_calls = case.stateful_tool_calls if mode == "stateful" else case.stateless_tool_calls
    goals = case.goals_stateful if mode == "stateful" else case.goals_stateless
    requested = _tool_names_from_calls(planned_calls)
    requested.update(_goal_implied_tool_names(goals or [], mode))
    requested.update(_domain_companion_tools(requested, mode))

    filtered = {
        tool_name: base_registry[tool_name]
        for tool_name in sorted(requested)
        if tool_name in base_registry
    }
    return filtered or base_registry


def _tool_names_from_calls(calls: list[dict[str, Any]]) -> set[str]:
    return {
        str(call.get("tool_name"))
        for call in calls
        if call.get("tool_name")
    }


def _goal_implied_tool_names(goals: list[dict[str, Any]], mode: str) -> set[str]:
    names: set[str] = set()
    for goal in goals:
        goal_type = str(goal.get("type", ""))
        entity_type = str(goal.get("entity_type", ""))
        if goal_type in {"query_hits_file", "indexed_contains"}:
            names.update({"file.write", "file.read", "search.query"})
            if mode == "stateful":
                names.add("search.index")
        if entity_type == "file":
            names.update({"file.write", "file.read"})
        if goal_type.startswith("issue_") or entity_type == "issue":
            names.update({"issue.create", "issue.assign", "issue.comment", "issue.close", "issue.reopen"})
        if goal_type.startswith("event_") or entity_type == "event":
            names.update({
                "calendar.create_event",
                "calendar.update_event",
                "calendar.search_events",
                "calendar.delete_event",
            })
        if goal_type.startswith("toolsandbox_"):
            names.update(_toolsandbox_goal_tools(goal))
    return names


def _toolsandbox_goal_tools(goal: dict[str, Any]) -> set[str]:
    entity_type = str(goal.get("entity_type", ""))
    if entity_type == "contact":
        return {"add_contact", "modify_contact", "remove_contact", "search_contacts"}
    if entity_type == "reminder":
        return {"add_reminder", "modify_reminder", "remove_reminder", "search_reminder"}
    if entity_type == "messaging":
        return {"send_message_with_phone_number", "search_messages"}
    if entity_type == "sandbox":
        return {"end_conversation"}
    if goal.get("type") == "toolsandbox_setting_equals":
        field = str(goal.get("field", ""))
        return {
            "wifi": {"set_wifi_status", "get_wifi_status"},
            "cellular": {"set_cellular_service_status", "get_cellular_service_status"},
            "location_service": {"set_location_service_status", "get_location_service_status", "get_current_location"},
            "low_battery_mode": {"set_low_battery_mode_status", "get_low_battery_mode_status"},
        }.get(field, set())
    return set()


def _domain_companion_tools(tool_names: set[str], mode: str) -> set[str]:
    companions: set[str] = set()
    if tool_names & {"file.write", "file.read", "search.index", "search.query"}:
        companions.update({"file.write", "file.read", "search.query"})
        if mode == "stateful":
            companions.add("search.index")
    if any(name.startswith("issue.") for name in tool_names):
        companions.update({"issue.create", "issue.assign", "issue.comment", "issue.close", "issue.reopen"})
    if any(name.startswith("calendar.") for name in tool_names):
        companions.update({
            "calendar.create_event",
            "calendar.update_event",
            "calendar.search_events",
            "calendar.delete_event",
        })
    if tool_names & {"add_contact", "modify_contact", "remove_contact", "search_contacts"}:
        companions.update({"add_contact", "modify_contact", "remove_contact", "search_contacts"})
    if tool_names & {"add_reminder", "modify_reminder", "remove_reminder", "search_reminder"}:
        companions.update({"add_reminder", "modify_reminder", "remove_reminder", "search_reminder"})
    if tool_names & {"send_message_with_phone_number", "search_messages"}:
        companions.update({"send_message_with_phone_number", "search_messages"})
    if tool_names & {
        "set_wifi_status",
        "get_wifi_status",
        "set_cellular_service_status",
        "get_cellular_service_status",
        "set_location_service_status",
        "get_location_service_status",
        "set_low_battery_mode_status",
        "get_low_battery_mode_status",
        "get_current_location",
    }:
        companions.update({
            "set_wifi_status",
            "get_wifi_status",
            "set_cellular_service_status",
            "get_cellular_service_status",
            "set_location_service_status",
            "get_location_service_status",
            "set_low_battery_mode_status",
            "get_low_battery_mode_status",
            "get_current_location",
        })
    if tool_names & {"end_conversation"}:
        companions.add("end_conversation")
    return companions


def _compute_mode_metrics(
    results: list[LLMExperiment1CaseResult],
    mode: str,
) -> LLMExperiment1Metrics:
    runs = [getattr(item, mode) for item in results]
    total = len(runs)
    total_calls = sum(run.experiment_result.call_metrics.total_calls for run in runs)
    invalid_calls = sum(run.experiment_result.call_metrics.invalid_calls for run in runs)
    final_correct_count = sum(1 for run in runs if _final_correct(run))
    recovery_opportunities = sum(1 for run in runs if run.experiment_result.call_metrics.failed_calls > 0)
    recovery_count = sum(
        1 for run in runs
        if run.experiment_result.call_metrics.failed_calls > 0 and _final_correct(run)
    )
    successful_calls = sum(run.experiment_result.call_metrics.successful_calls for run in runs)
    total_parse_errors = sum(run.parse_error_count for run in runs)
    total_llm_steps = sum(len(run.llm_steps) for run in runs)
    total_length_stops = sum(
        1
        for run in runs
        for step in run.llm_steps
        if _is_length_truncated_step(step.to_dict())
    )
    return LLMExperiment1Metrics(
        mode=mode,
        total_cases=total,
        success_rate=_rate(successful_calls, total_calls),
        invalid_call_rate=_rate(invalid_calls, total_calls),
        recovery_rate=_optional_rate(recovery_count, recovery_opportunities),
        recovery_opportunities=recovery_opportunities,
        recovered_cases=recovery_count,
        average_trajectory_length=_rate(sum(len(run.experiment_result.trace) for run in runs), total),
        final_state_correctness=_rate(final_correct_count, total),
        parse_error_rate=_rate(total_parse_errors, total_llm_steps),
        length_truncation_rate=_rate(total_length_stops, total_llm_steps),
        average_llm_steps=_rate(total_llm_steps, total),
        average_tokens=_rate(sum(run.total_tokens for run in runs), total),
        average_llm_latency_ms=_rate(sum(run.total_llm_latency_ms for run in runs), total),
    )


def _final_correct(run: LLMAgentRunResult) -> bool:
    metrics = run.experiment_result.state_metrics
    return metrics is not None and metrics.all_passed


def _rate(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _is_length_truncated_step(step: dict[str, Any]) -> bool:
    response = step.get("llm_response") or {}
    raw = response.get("raw") or {}
    choices = raw.get("choices") or []
    if not choices:
        return False
    return choices[0].get("finish_reason") == "length"


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _recovery_label(value: float | None) -> str:
    return "N/A" if value is None else _pct(value)


def _optional_rate(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


