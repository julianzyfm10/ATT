"""LLM batch runtime for the 30-case Experiment 4 Core Local benchmark."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from toolsim.llm.types import LLMClient
from toolsim.runners.core_local_dataset import CoreLocalCase, CoreLocalStateEvaluator, dependency_compliance
from toolsim.runners.core_local_oracle import backend_for_name
from toolsim.runners.llm_agent_runner import LLMAgentConfig, LLMAgentRunner
from toolsim.tools.core_local_mcp_tools import registry_for_servers

CORE_LOCAL_STRATEGIES = ["direct", "cot", "react", "self_refine"]
CORE_LOCAL_BACKENDS = ["mock", "sandbox", "local_mcp"]


def run_llm_core_local_case(
    client: LLMClient,
    case: CoreLocalCase,
    strategy: str,
    backend_name: str,
    *,
    config: LLMAgentConfig | None = None,
) -> dict[str, Any]:
    if strategy not in CORE_LOCAL_STRATEGIES:
        raise ValueError(f"Unsupported strategy: {strategy}")
    if backend_name not in CORE_LOCAL_BACKENDS:
        raise ValueError(f"Unsupported backend: {backend_name}")
    base_config = config or LLMAgentConfig(max_steps=8)
    agent = LLMAgentRunner(client, config=_config_for_case(base_config, strategy, len(case.oracle_plan)))
    backend = backend_for_name(backend_name, case.case_id)
    try:
        run = agent.run(
            task_name=case.case_id,
            task_description=_task_prompt(case, strategy),
            mode="stateful",
            initial_state=None,
            goals=case.goals,
            tool_registry=registry_for_servers(case.servers),
            state_evaluator=CoreLocalStateEvaluator(case),
            backend=backend,
        )
        final_correct = bool(run.experiment_result.state_metrics and run.experiment_result.state_metrics.all_passed)
        trace = run.experiment_result.trace
        call_metrics = run.experiment_result.call_metrics
        return {
            "case_id": case.case_id,
            "title": case.data.get("title", case.case_id),
            "difficulty": case.data.get("difficulty", ""),
            "topology": case.data.get("topology", ""),
            "strategy": strategy,
            "backend": backend_name,
            "run": run.to_dict(),
            "final_state_correct": final_correct,
            "dependency_compliant": dependency_compliance(case, run.experiment_result.final_state),
            "all_calls_succeeded": run.experiment_result.all_calls_succeeded,
            "invalid_call_rate": _rate(call_metrics.invalid_calls, call_metrics.total_calls),
            "recovery_detected": call_metrics.failed_calls > 0 and final_correct,
            "trajectory_length": len(trace),
            "tool_latency_ms": sum(item.duration_ms for item in trace),
        }
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()


def summarize_llm_core_local(
    results: list[dict[str, Any]],
    strategies: list[str],
    backends: list[str],
    *,
    model: str,
    expected_runs: int,
) -> dict[str, Any]:
    mock_success = {
        strategy: _rate(
            sum(item["final_state_correct"] for item in results if item["strategy"] == strategy and item["backend"] == "mock"),
            sum(1 for item in results if item["strategy"] == strategy and item["backend"] == "mock"),
        )
        for strategy in strategies
    }
    metrics: list[dict[str, Any]] = []
    for strategy in strategies:
        for backend in backends:
            group = [item for item in results if item["strategy"] == strategy and item["backend"] == backend]
            total = len(group)
            calls = sum(item["run"]["experiment_result"]["call_metrics"]["total_calls"] for item in group)
            successful = sum(item["run"]["experiment_result"]["call_metrics"]["successful_calls"] for item in group)
            invalid = sum(item["run"]["experiment_result"]["call_metrics"]["invalid_calls"] for item in group)
            opportunities = sum(item["run"]["experiment_result"]["call_metrics"]["failed_calls"] > 0 for item in group)
            recovered = sum(item["recovery_detected"] for item in group)
            turns = sum(len(item["run"]["llm_steps"]) for item in group)
            correct_rate = _rate(sum(item["final_state_correct"] for item in group), total)
            metrics.append({
                "strategy": strategy,
                "backend": backend,
                "completed": total,
                "final_state_correctness": correct_rate,
                "mock_to_backend_gap": max(0.0, mock_success.get(strategy, 0.0) - correct_rate),
                "dependency_compliance_rate": _rate(sum(item["dependency_compliant"] for item in group), total),
                "tool_success_rate": _rate(successful, calls),
                "invalid_call_rate": _rate(invalid, calls),
                "recovery_rate": _optional_rate(recovered, opportunities),
                "average_trajectory_length": _rate(calls, total),
                "average_llm_turns": _rate(turns, total),
                "parse_error_rate": _rate(sum(item["run"]["parse_error_count"] for item in group), turns),
                "average_tokens": _rate(sum(item["run"]["total_tokens"] for item in group), total),
                "average_llm_latency_ms": _rate(sum(item["run"]["total_llm_latency_ms"] for item in group), total),
                "average_tool_latency_ms": _rate(sum(item["tool_latency_ms"] for item in group), total),
            })
    return {
        "model": model,
        "expected_runs": expected_runs,
        "completed_runs": len(results),
        "complete": len(results) == expected_runs,
        "group_metrics": metrics,
        "case_results": results,
    }


def render_llm_core_local_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Experiment 4 Core Local LLM Report",
        "",
        f"- Model: `{summary['model']}`",
        f"- Progress: {summary['completed_runs']} / {summary['expected_runs']}",
        f"- Complete: {summary['complete']}",
        "- Backend comparison: `mock -> sandbox -> local_mcp`",
        "",
        "| Strategy | Backend | N | Correct | Mock gap | Dependency | Tool success | Invalid | Recovery | Avg calls | Avg turns | Avg tokens |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in summary["group_metrics"]:
        recovery = "N/A" if metric["recovery_rate"] is None else _pct(metric["recovery_rate"])
        lines.append(
            f"| {metric['strategy']} | {metric['backend']} | {metric['completed']} | "
            f"{_pct(metric['final_state_correctness'])} | {_pct(metric['mock_to_backend_gap'])} | "
            f"{_pct(metric['dependency_compliance_rate'])} | {_pct(metric['tool_success_rate'])} | "
            f"{_pct(metric['invalid_call_rate'])} | {recovery} | "
            f"{metric['average_trajectory_length']:.2f} | {metric['average_llm_turns']:.2f} | "
            f"{metric['average_tokens']:.1f} |"
        )
    failures = [item for item in summary["case_results"] if not item["final_state_correct"]]
    lines.extend(["", "## Failed runs", ""])
    if not failures:
        lines.append("None.")
    else:
        for item in failures:
            lines.append(f"- `{item['case_id']}::{item['strategy']}::{item['backend']}`")
    lines.append("")
    return "\n".join(lines)


def result_key(item: dict[str, Any]) -> str:
    return f"{item.get('case_id')}::{item.get('strategy')}::{item.get('backend')}"


def _config_for_case(base: LLMAgentConfig, strategy: str, oracle_steps: int) -> LLMAgentConfig:
    if strategy in {"direct", "cot"}:
        return replace(base, max_steps=max(1, oracle_steps), reject_finish_when_goals_unmet=False)
    minimum = max(base.max_steps, oracle_steps)
    if strategy == "react":
        return replace(base, max_steps=minimum + 2, reject_finish_when_goals_unmet=False)
    return replace(base, max_steps=minimum + 3, reject_finish_when_goals_unmet=True)


def _task_prompt(case: CoreLocalCase, strategy: str) -> str:
    instructions = {
        "direct": "Execute the required tool sequence directly. Do not probe or retry unless the task explicitly requires recovery.",
        "cot": "Plan the complete dependency chain internally, then execute one necessary tool call per turn without exploratory calls.",
        "react": "Use each observation to choose the next tool. On schema or validation failure, adapt the arguments and retry.",
        "self_refine": "Before each call, self-check the tool schema and prior outputs; repair failures and verify every requested result before finishing.",
    }
    return (
        f"{case.prompt}\n\n"
        "Use only the available MCP tools and continue until every requested computation is complete. "
        "Pass numeric outputs from earlier tools into later tools when the task has dependencies. "
        "Do not replace tool execution with a natural-language answer.\n"
        f"Reasoning policy: {strategy}. {instructions[strategy]}"
    )


def _rate(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _optional_rate(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"