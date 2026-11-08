"""LLM-backed Experiment 2: cross-tool dependency difficulty curves."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from toolsim.adapters.toolsandbox_adapter import convert_toolsandbox_file
from toolsim.core.tool_spec import ToolSpec
from toolsim.evaluators.trajectory_evaluator import TrajectoryLevelEvaluator
from toolsim.llm.types import LLMClient
from toolsim.runners.dependency_curve import (
    DEFAULT_STRATEGIES,
    L1_SINGLE_TOOL,
    L2_EXPLICIT_DEP,
    L3_IMPLICIT_SIDE_EFFECT,
    DependencyCurveCase,
    DifficultyLevel,
    StrategyName,
    build_synthetic_dependency_curve_cases,
    build_toolsandbox_dependency_curve_cases,
    evaluate_dependency_trace,
    select_toolsandbox_dependency_subset_cases,
)
from toolsim.runners.llm_agent_runner import LLMAgentConfig, LLMAgentRunResult, LLMAgentRunner, tool_registry_for_mode


@dataclass
class LLMExperiment2CaseResult:
    case_name: str
    source: str
    strategy: StrategyName
    difficulty_level: DifficultyLevel
    dependency_type: str
    run: LLMAgentRunResult
    final_state_correct: bool
    dependency_correct: bool
    dependency_completion_rate: float
    wrong_order_call_count: int
    missing_prerequisite_count: int
    side_effect_dependency_success: bool | None
    dependency_repair: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_name": self.case_name,
            "source": self.source,
            "strategy": self.strategy,
            "difficulty_level": self.difficulty_level,
            "dependency_type": self.dependency_type,
            "run": self.run.to_dict(),
            "final_state_correct": self.final_state_correct,
            "dependency_correct": self.dependency_correct,
            "dependency_completion_rate": self.dependency_completion_rate,
            "wrong_order_call_count": self.wrong_order_call_count,
            "missing_prerequisite_count": self.missing_prerequisite_count,
            "side_effect_dependency_success": self.side_effect_dependency_success,
            "dependency_repair": self.dependency_repair,
        }


@dataclass
class LLMExperiment2GroupMetrics:
    source: str
    strategy: StrategyName
    difficulty_level: DifficultyLevel
    total_cases: int
    success_count: int
    final_state_correctness: float
    dependency_correctness_rate: float
    success_rate: float
    invalid_call_rate: float
    recovery_rate: float | None
    recovery_opportunities: int
    recovered_cases: int
    average_trajectory_length: float
    dependency_completion_rate: float
    wrong_order_rate: float
    missing_prerequisite_rate: float
    side_effect_awareness_rate: float | None
    dependency_repair_rate: float
    parse_error_rate: float
    length_truncation_rate: float
    average_llm_steps: float
    average_tokens: float
    average_llm_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "strategy": self.strategy,
            "difficulty_level": self.difficulty_level,
            "total_cases": self.total_cases,
            "success_count": self.success_count,
            "final_state_correctness": self.final_state_correctness,
            "dependency_correctness_rate": self.dependency_correctness_rate,
            "success_rate": self.success_rate,
            "invalid_call_rate": self.invalid_call_rate,
            "recovery_rate": self.recovery_rate,
            "recovery_opportunities": self.recovery_opportunities,
            "recovered_cases": self.recovered_cases,
            "average_trajectory_length": self.average_trajectory_length,
            "dependency_completion_rate": self.dependency_completion_rate,
            "wrong_order_rate": self.wrong_order_rate,
            "missing_prerequisite_rate": self.missing_prerequisite_rate,
            "side_effect_awareness_rate": self.side_effect_awareness_rate,
            "dependency_repair_rate": self.dependency_repair_rate,
            "parse_error_rate": self.parse_error_rate,
            "length_truncation_rate": self.length_truncation_rate,
            "average_llm_steps": self.average_llm_steps,
            "average_tokens": self.average_tokens,
            "average_llm_latency_ms": self.average_llm_latency_ms,
        }


@dataclass
class LLMExperiment2DifficultySummary:
    source: str
    strategy: StrategyName
    l1_dependency_correctness: float
    l2_dependency_correctness: float
    l3_dependency_correctness: float
    dependency_degradation_l1_to_l3: float
    auc_dependency_correctness: float
    l1_final_state_correctness: float
    l2_final_state_correctness: float
    l3_final_state_correctness: float
    degradation_l1_to_l3: float
    auc_final_state_correctness: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "strategy": self.strategy,
            "l1_dependency_correctness": self.l1_dependency_correctness,
            "l2_dependency_correctness": self.l2_dependency_correctness,
            "l3_dependency_correctness": self.l3_dependency_correctness,
            "dependency_degradation_l1_to_l3": self.dependency_degradation_l1_to_l3,
            "auc_dependency_correctness": self.auc_dependency_correctness,
            "l1_final_state_correctness": self.l1_final_state_correctness,
            "l2_final_state_correctness": self.l2_final_state_correctness,
            "l3_final_state_correctness": self.l3_final_state_correctness,
            "degradation_l1_to_l3": self.degradation_l1_to_l3,
            "auc_final_state_correctness": self.auc_final_state_correctness,
        }


@dataclass
class LLMExperiment2Result:
    model: str
    case_results: list[LLMExperiment2CaseResult]
    group_metrics: list[LLMExperiment2GroupMetrics]
    difficulty_summaries: list[LLMExperiment2DifficultySummary]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "case_results": [result.to_dict() for result in self.case_results],
            "group_metrics": [metric.to_dict() for metric in self.group_metrics],
            "difficulty_summaries": [summary.to_dict() for summary in self.difficulty_summaries],
        }


class LLMExperiment2DependencyCurveRunner:
    """Run dependency-curve cases with a closed-loop LLM policy."""

    def __init__(self, client: LLMClient, *, config: LLMAgentConfig | None = None) -> None:
        self.client = client
        base_config = config or LLMAgentConfig()
        self.config = replace(base_config, stop_on_goals_passed=False)

    def run(
        self,
        cases: list[DependencyCurveCase],
        strategies: list[StrategyName] | None = None,
    ) -> LLMExperiment2Result:
        selected_strategies = strategies or DEFAULT_STRATEGIES
        agents = {
            strategy: LLMAgentRunner(self.client, config=_config_for_strategy(self.config, strategy))
            for strategy in selected_strategies
        }
        results = [
            self.run_case(agents[strategy], case, strategy)
            for case in cases
            for strategy in selected_strategies
        ]
        group_metrics = compute_llm_experiment2_group_metrics(results)
        return LLMExperiment2Result(
            model=getattr(self.client, "model", "unknown"),
            case_results=results,
            group_metrics=group_metrics,
            difficulty_summaries=compute_llm_experiment2_difficulty_summaries(group_metrics),
        )

    def run_case(
        self,
        agent: LLMAgentRunner,
        case: DependencyCurveCase,
        strategy: StrategyName,
    ) -> LLMExperiment2CaseResult:
        run = agent.run(
            task_name=case.case_name,
            task_description=_task_prompt(case, strategy),
            mode="stateful",
            initial_state=case.initial_state,
            goals=case.goals,
            tool_registry=dependency_case_tool_registry(case),
        )
        final_state_correct = _final_correct(run)
        dependency_metrics = evaluate_dependency_trace(
            [record.tool_name for record in run.experiment_result.trace],
            case.required_edges,
        )
        dependency_correct = (
            final_state_correct
            and dependency_metrics["dependency_completion_rate"] == 1.0
            and dependency_metrics["wrong_order_call_count"] == 0
            and dependency_metrics["missing_prerequisite_count"] == 0
        )
        trajectory = TrajectoryLevelEvaluator().evaluate(run.experiment_result.trace)
        dependency_repair = bool(final_state_correct and trajectory.failed_steps > 0)
        side_effect_success = None
        if case.difficulty_level == L3_IMPLICIT_SIDE_EFFECT:
            side_effect_success = final_state_correct and dependency_metrics["dependency_completion_rate"] == 1.0

        return LLMExperiment2CaseResult(
            case_name=case.case_name,
            source=case_source(case),
            strategy=strategy,
            difficulty_level=case.difficulty_level,
            dependency_type=case.dependency_type,
            run=run,
            final_state_correct=final_state_correct,
            dependency_correct=dependency_correct,
            dependency_completion_rate=dependency_metrics["dependency_completion_rate"],
            wrong_order_call_count=dependency_metrics["wrong_order_call_count"],
            missing_prerequisite_count=dependency_metrics["missing_prerequisite_count"],
            side_effect_dependency_success=side_effect_success,
            dependency_repair=dependency_repair,
        )


def build_llm_experiment2_cases(
    *,
    include_synthetic: bool = True,
    include_toolsandbox: bool = True,
    toolsandbox_path: str | Path | None = None,
    toolsandbox_per_level: int = 8,
    limit: int | None = None,
) -> list[DependencyCurveCase]:
    cases: list[DependencyCurveCase] = []
    if include_synthetic:
        cases.extend(build_synthetic_dependency_curve_cases())
    if include_toolsandbox:
        path = Path(toolsandbox_path) if toolsandbox_path is not None else _default_toolsandbox_path()
        converted = [case for case in convert_toolsandbox_file(path) if case.goals]
        subset = select_toolsandbox_dependency_subset_cases(converted, per_level=toolsandbox_per_level)
        cases.extend(build_toolsandbox_dependency_curve_cases(subset))
    return _balanced_limit_cases(cases, limit) if limit is not None else cases


def dependency_case_tool_registry(case: DependencyCurveCase) -> dict[str, ToolSpec]:
    base_registry = tool_registry_for_mode("stateful")
    requested = _tool_names_from_calls(case.oracle_tool_calls)
    requested.update(_goal_implied_tool_names(case.goals))
    requested.update(_domain_companion_tools(requested))
    filtered = {
        tool_name: base_registry[tool_name]
        for tool_name in sorted(requested)
        if tool_name in base_registry
    }
    return filtered or base_registry


def _balanced_limit_cases(cases: list[DependencyCurveCase], limit: int) -> list[DependencyCurveCase]:
    if limit <= 0:
        return []

    buckets: dict[tuple[str, str], list[DependencyCurveCase]] = {}
    for case in cases:
        buckets.setdefault((case_source(case), case.difficulty_level), []).append(case)

    source_order = [source for source in ["synthetic", "toolsandbox"] if any(key[0] == source for key in buckets)]
    level_order = [L1_SINGLE_TOOL, L2_EXPLICIT_DEP, L3_IMPLICIT_SIDE_EFFECT]
    selected: list[DependencyCurveCase] = []
    while len(selected) < limit and any(buckets.values()):
        progressed = False
        for source in source_order:
            for level in level_order:
                bucket = buckets.get((source, level), [])
                if not bucket:
                    continue
                selected.append(bucket.pop(0))
                progressed = True
                if len(selected) >= limit:
                    return selected
        if not progressed:
            break
    return selected


def compute_llm_experiment2_group_metrics(
    results: list[LLMExperiment2CaseResult],
) -> list[LLMExperiment2GroupMetrics]:
    metrics: list[LLMExperiment2GroupMetrics] = []
    sources = ["all", *sorted({result.source for result in results})]
    strategies = sorted({result.strategy for result in results})
    for source in sources:
        source_results = results if source == "all" else [result for result in results if result.source == source]
        for strategy in strategies:
            strategy_results = [result for result in source_results if result.strategy == strategy]
            for level in [L1_SINGLE_TOOL, L2_EXPLICIT_DEP, L3_IMPLICIT_SIDE_EFFECT]:
                group = [result for result in strategy_results if result.difficulty_level == level]
                metrics.append(_summarize_group(source, strategy, level, group))
    return metrics


def compute_llm_experiment2_difficulty_summaries(
    group_metrics: list[LLMExperiment2GroupMetrics],
) -> list[LLMExperiment2DifficultySummary]:
    summaries: list[LLMExperiment2DifficultySummary] = []
    by_key = {(metric.source, metric.strategy, metric.difficulty_level): metric for metric in group_metrics}
    keys = sorted({(metric.source, metric.strategy) for metric in group_metrics})
    for source, strategy in keys:
        l1_metric = by_key[(source, strategy, L1_SINGLE_TOOL)]
        l2_metric = by_key[(source, strategy, L2_EXPLICIT_DEP)]
        l3_metric = by_key[(source, strategy, L3_IMPLICIT_SIDE_EFFECT)]
        l1_dep = l1_metric.dependency_correctness_rate
        l2_dep = l2_metric.dependency_correctness_rate
        l3_dep = l3_metric.dependency_correctness_rate
        l1_final = l1_metric.final_state_correctness
        l2_final = l2_metric.final_state_correctness
        l3_final = l3_metric.final_state_correctness
        summaries.append(LLMExperiment2DifficultySummary(
            source=source,
            strategy=strategy,
            l1_dependency_correctness=l1_dep,
            l2_dependency_correctness=l2_dep,
            l3_dependency_correctness=l3_dep,
            dependency_degradation_l1_to_l3=l1_dep - l3_dep,
            auc_dependency_correctness=(l1_dep + l2_dep + l3_dep) / 3,
            l1_final_state_correctness=l1_final,
            l2_final_state_correctness=l2_final,
            l3_final_state_correctness=l3_final,
            degradation_l1_to_l3=l1_final - l3_final,
            auc_final_state_correctness=(l1_final + l2_final + l3_final) / 3,
        ))
    return summaries


def render_llm_experiment2_markdown(result: LLMExperiment2Result) -> str:
    return render_llm_experiment2_markdown_from_dict(result.to_dict())


def render_llm_experiment2_markdown_from_dict(result: dict[str, Any]) -> str:
    lines = [
        "# Experiment 2 LLM Cross-tool Dependency Curve Report",
        "",
        f"- Model: `{result['model']}`",
        f"- Completed cases: {len(result['case_results'])}",
        "- Mode: `stateful` closed-loop tool execution",
        "",
        "## Dependency-aware Difficulty Summary",
        "",
        "| Source | Strategy | L1 Dep Correct | L2 Dep Correct | L3 Dep Correct | L1→L3 Dep Degradation | Dep AUC | L3 Final Correct |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in result["difficulty_summaries"]:
        lines.append(
            f"| {summary['source']} | {summary['strategy']} | {_pct(summary['l1_dependency_correctness'])} | "
            f"{_pct(summary['l2_dependency_correctness'])} | {_pct(summary['l3_dependency_correctness'])} | "
            f"{_pct(summary['dependency_degradation_l1_to_l3'])} | {_pct(summary['auc_dependency_correctness'])} | "
            f"{_pct(summary['l3_final_state_correctness'])} |"
        )

    lines.extend([
        "",
        "## Metrics By Difficulty",
        "",
        "| Source | Strategy | Difficulty | Cases | Success Rate | Invalid Call Rate | Recovery Rate | Avg Trajectory Length | Final State Correctness | Dependency-aware Correctness | Dependency Completion | Wrong Order | Missing Prereq | Repair Rate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for metric in result["group_metrics"]:
        lines.append(
            f"| {metric['source']} | {metric['strategy']} | {metric['difficulty_level']} | {metric['total_cases']} | "
            f"{_pct(metric['success_rate'])} | {_pct(metric['invalid_call_rate'])} | "
            f"{_recovery_label(metric.get('recovery_rate'))} | {metric['average_trajectory_length']:.2f} | "
            f"{_pct(metric['final_state_correctness'])} | {_pct(metric['dependency_correctness_rate'])} | "
            f"{_pct(metric['dependency_completion_rate'])} | "
            f"{_pct(metric['wrong_order_rate'])} | {_pct(metric['missing_prerequisite_rate'])} | "
            f"{_pct(metric['dependency_repair_rate'])} |"
        )

    lines.extend([
        "",
        "## LLM Diagnostics",
        "",
        "| Source | Strategy | Difficulty | Avg LLM Turns | Parse Errors | Length Stops | Avg Tokens | Avg LLM Latency ms |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for metric in result["group_metrics"]:
        lines.append(
            f"| {metric['source']} | {metric['strategy']} | {metric['difficulty_level']} | "
            f"{metric['average_llm_steps']:.2f} | {_pct(metric['parse_error_rate'])} | "
            f"{_pct(metric['length_truncation_rate'])} | {metric['average_tokens']:.1f} | "
            f"{metric['average_llm_latency_ms']:.2f} |"
        )

    lines.extend(["", "## Case Outcomes", ""])
    lines.append("| Case | Source | Strategy | Difficulty | Final Correct | Dep Correct | Dependency Completion | Wrong Order | Missing Prereq | Traj Len | LLM Turns |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for item in result["case_results"]:
        run = item["run"]
        dependency_correct = _dependency_correct_from_dict(item)
        lines.append(
            f"| {item['case_name']} | {item['source']} | {item['strategy']} | {item['difficulty_level']} | "
            f"{item['final_state_correct']} | {dependency_correct} | {_pct(item['dependency_completion_rate'])} | "
            f"{item['wrong_order_call_count']} | {item['missing_prerequisite_count']} | "
            f"{len(run['experiment_result']['trace'])} | {len(run['llm_steps'])} |"
        )
    return "\n".join(lines)


def case_source(case: DependencyCurveCase) -> str:
    return "toolsandbox" if case.case_name.startswith("toolsandbox::") else "synthetic"


def _dependency_correct_from_dict(item: dict[str, Any]) -> bool:
    return bool(
        item.get("dependency_correct")
        or (
            item.get("final_state_correct")
            and item.get("dependency_completion_rate") == 1.0
            and item.get("wrong_order_call_count") == 0
            and item.get("missing_prerequisite_count") == 0
        )
    )


def _task_prompt(case: DependencyCurveCase, strategy: StrategyName) -> str:
    final_response_requirements = _final_response_requirements(case.goals)
    final_response_section = ""
    if final_response_requirements:
        final_response_section = (
            "\nFinal response requirements: before finishing, call end_conversation with "
            f"these exact user-visible response arguments: {final_response_requirements}\n"
        )
    return (
        f"{case.description}\n\n"
        "This task is part of a cross-tool dependency difficulty evaluation. "
        "Use the available tools to satisfy the target state. Some tasks require using one tool "
        "before another, repairing failed calls, or accounting for side effects that are only visible "
        "through the environment observations. Do not explain in natural language; choose tools until "
        "the state satisfies the goals.\n\n"
        f"Difficulty level: {case.difficulty_level}\n"
        f"Dependency type: {case.dependency_type}\n"
        f"Reasoning strategy: {strategy}\n"
        f"Strategy instruction: {_strategy_instruction(strategy)}\n"
        f"{final_response_section}"
        f"Target goals: {case.goals}"
    )


def _strategy_instruction(strategy: StrategyName) -> str:
    if strategy == "direct":
        return (
            "Use the shortest plausible sequence of tool calls. Do not add exploratory calls unless a required "
            "argument is unknown or a previous call failed."
        )
    if strategy == "cot":
        return (
            "Before each tool call, use the JSON thought field to explicitly plan prerequisite tools and dependency "
            "order. Prefer satisfying prerequisites before final actions."
        )
    if strategy == "react":
        return (
            "Act based on the latest observation. If a call fails or returns missing information, adapt the next "
            "tool call to repair the trajectory."
        )
    if strategy == "self_refine":
        return (
            "Before finishing, self-check every target goal and dependency. If any goal may be unsatisfied, call "
            "the missing or repair tool instead of finishing."
        )
    return "Use the named strategy consistently while choosing tools."


def _config_for_strategy(base_config: LLMAgentConfig, strategy: StrategyName) -> LLMAgentConfig:
    if strategy == "direct":
        return replace(
            base_config,
            max_steps=min(base_config.max_steps, 2),
            reject_finish_when_goals_unmet=False,
        )
    if strategy == "cot":
        return replace(
            base_config,
            max_steps=min(base_config.max_steps, 4),
            reject_finish_when_goals_unmet=False,
        )
    if strategy == "react":
        return replace(
            base_config,
            reject_finish_when_goals_unmet=False,
        )
    if strategy == "self_refine":
        return replace(
            base_config,
            max_steps=base_config.max_steps + 2,
            reject_finish_when_goals_unmet=True,
        )
    return replace(base_config, reject_finish_when_goals_unmet=False)


def _final_response_requirements(goals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    requirements: list[dict[str, Any]] = []
    for goal in goals:
        if goal.get("type") != "toolsandbox_record_exists" or goal.get("entity_type") != "sandbox":
            continue
        fields = goal.get("fields") or {}
        if fields.get("sender") != "AGENT" or "content" not in fields:
            continue
        requirements.append({
            "content": fields.get("content"),
            "recipient": fields.get("recipient", "USER"),
        })
    return requirements


def _summarize_group(
    source: str,
    strategy: StrategyName,
    difficulty_level: DifficultyLevel,
    group: list[LLMExperiment2CaseResult],
) -> LLMExperiment2GroupMetrics:
    total_cases = len(group)
    total_calls = sum(item.run.experiment_result.call_metrics.total_calls for item in group)
    successful_calls = sum(item.run.experiment_result.call_metrics.successful_calls for item in group)
    invalid_calls = sum(item.run.experiment_result.call_metrics.invalid_calls for item in group)
    recovery_opportunities = sum(1 for item in group if item.run.experiment_result.call_metrics.failed_calls > 0)
    recovered = sum(
        1 for item in group
        if item.run.experiment_result.call_metrics.failed_calls > 0 and item.final_state_correct
    )
    total_llm_steps = sum(len(item.run.llm_steps) for item in group)
    total_parse_errors = sum(item.run.parse_error_count for item in group)
    total_length_stops = sum(
        1
        for item in group
        for step in item.run.llm_steps
        if _is_length_truncated_step(step.to_dict())
    )
    side_effect_cases = [item for item in group if item.side_effect_dependency_success is not None]

    return LLMExperiment2GroupMetrics(
        source=source,
        strategy=strategy,
        difficulty_level=difficulty_level,
        total_cases=total_cases,
        success_count=sum(1 for item in group if item.final_state_correct),
        final_state_correctness=_rate(sum(1 for item in group if item.final_state_correct), total_cases),
        dependency_correctness_rate=_rate(sum(1 for item in group if item.dependency_correct), total_cases),
        success_rate=_rate(successful_calls, total_calls),
        invalid_call_rate=_rate(invalid_calls, total_calls),
        recovery_rate=_optional_rate(recovered, recovery_opportunities),
        recovery_opportunities=recovery_opportunities,
        recovered_cases=recovered,
        average_trajectory_length=_rate(total_calls, total_cases),
        dependency_completion_rate=_rate(sum(item.dependency_completion_rate for item in group), total_cases),
        wrong_order_rate=_rate(sum(1 for item in group if item.wrong_order_call_count > 0), total_cases),
        missing_prerequisite_rate=_rate(sum(1 for item in group if item.missing_prerequisite_count > 0), total_cases),
        side_effect_awareness_rate=(
            _rate(sum(1 for item in side_effect_cases if item.side_effect_dependency_success), len(side_effect_cases))
            if side_effect_cases
            else None
        ),
        dependency_repair_rate=_rate(sum(1 for item in group if item.dependency_repair), total_cases),
        parse_error_rate=_rate(total_parse_errors, total_llm_steps),
        length_truncation_rate=_rate(total_length_stops, total_llm_steps),
        average_llm_steps=_rate(total_llm_steps, total_cases),
        average_tokens=_rate(sum(item.run.total_tokens for item in group), total_cases),
        average_llm_latency_ms=_rate(sum(item.run.total_llm_latency_ms for item in group), total_cases),
    )


def _final_correct(run: LLMAgentRunResult) -> bool:
    metrics = run.experiment_result.state_metrics
    return metrics is not None and metrics.all_passed


def _tool_names_from_calls(calls: list[dict[str, Any]]) -> set[str]:
    return {
        str(call.get("tool_name"))
        for call in calls
        if call.get("tool_name")
    }


def _goal_implied_tool_names(goals: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for goal in goals:
        goal_type = str(goal.get("type", ""))
        entity_type = str(goal.get("entity_type", ""))
        if goal_type in {"query_hits_file", "indexed_contains"}:
            names.update({"file.write", "file.read", "search.index", "search.query"})
        if entity_type == "file":
            names.update({"file.write", "file.read", "search.index", "search.query"})
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


def _domain_companion_tools(tool_names: set[str]) -> set[str]:
    companions: set[str] = set()
    if tool_names & {"file.write", "file.read", "search.index", "search.query"}:
        companions.update({"file.write", "file.read", "search.index", "search.query"})
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


def _is_length_truncated_step(step: dict[str, Any]) -> bool:
    response = step.get("llm_response") or {}
    raw = response.get("raw") or {}
    choices = raw.get("choices") or []
    if not choices:
        return False
    return choices[0].get("finish_reason") == "length"


def _rate(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _optional_rate(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _recovery_label(value: float | None) -> str:
    return "N/A" if value is None else _pct(value)


def _default_toolsandbox_path() -> Path:
    return Path(__file__).resolve().parents[3] / "Toolsandbox" / "tool_sandbox_scenarios.json"
