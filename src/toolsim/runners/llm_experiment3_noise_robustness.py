"""Experiment 3: ToolSandbox noise robustness."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from toolsim.core.world_state import WorldState
from toolsim.evaluators.evaluator import StateLevelEvaluator
from toolsim.execution.stateful_executor import ExecutorConfig
from toolsim.faults import FaultProfile
from toolsim.llm.types import LLMClient
from toolsim.runners.llm_agent_runner import LLMAgentConfig, LLMAgentRunResult, LLMAgentRunner
from toolsim.runners.noise_tool_registry import policy_noise_tool_registry

def state_corruption(initial_state: dict[str, Any], reference_state: dict[str, Any], final_state: dict[str, Any]) -> tuple[bool, list[str]]:
    """Detect unintended persistent mutations independently of task success."""
    def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for key in sorted(value):
                out.update(flatten(value[key], f"{prefix}.{key}" if prefix else str(key)))
            return out
        if isinstance(value, list):
            out = {}
            for index, item in enumerate(value):
                out.update(flatten(item, f"{prefix}[{index}]"))
            return out
        return {prefix: value}

    def persistent(state: dict[str, Any]) -> dict[str, Any]:
        entities = json.loads(json.dumps(state.get("entities") or {}))
        entities.pop("sandbox", None)
        for bucket in entities.values():
            if isinstance(bucket, dict):
                for record in bucket.values():
                    if isinstance(record, dict):
                        record.pop("sandbox_message_index", None)
        return flatten({"entities": entities})

    initial, reference, final = (persistent(state) for state in (initial_state, reference_state, final_state))
    missing = object()
    paths = []
    for path in sorted(set(initial) | set(reference) | set(final)):
        before, expected, actual = initial.get(path, missing), reference.get(path, missing), final.get(path, missing)
        if actual != before and (before == expected or actual != expected):
            paths.append(path)
    return bool(paths), paths[:50]


STRATEGIES = ["direct", "cot", "react", "self_refine"]
STRATEGY_EXTRA_STEPS = {"direct": 0, "cot": 1, "react": 3, "self_refine": 4}


@dataclass
class Experiment3Case:
    case_name: str
    base_case_name: str
    scenario_name: str
    source: str
    noise_type: str
    noise_target_tool: str
    noise_intensity: int
    task_prompt: str
    execution_goals: list[dict[str, Any]]
    state_goals: list[dict[str, Any]]
    semantic_observation_required: bool
    semantic_observation_tool: str | None
    reference_observation: dict[str, Any] | None
    oracle_tool_calls: list[dict[str, Any]]
    noise_execution_calls: list[dict[str, Any]]
    fault_profile: FaultProfile
    initial_state: WorldState
    reference_final_state: WorldState
    domain: str
    difficulty_level: str
    canonical_task_key: str
    k: int = 1

    @property
    def payload(self) -> dict[str, Any]:
        return {
            "oracle_tool_calls": self.noise_execution_calls,
            "target_tool": self.noise_target_tool,
        }

    @property
    def goals(self) -> list[dict[str, Any]]:
        """Compatibility view used by the shared ToolSandbox registry filter."""
        return self.execution_goals

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Experiment3Case":
        return cls(
            case_name=str(row["case_name"]),
            base_case_name=str(row["base_case_name"]),
            scenario_name=str(row["scenario_name"]),
            source=str(row["source"]),
            noise_type=str(row["noise_type"]),
            noise_target_tool=str(row["noise_target_tool"]),
            noise_intensity=int(row.get("noise_intensity", 1)),
            task_prompt=str(row["task_prompt"]),
            execution_goals=list(row.get("execution_goals") or []),
            state_goals=list(row.get("state_goals") or []),
            semantic_observation_required=bool(row.get("semantic_observation_required")),
            semantic_observation_tool=(str(row["semantic_observation_tool"]) if row.get("semantic_observation_tool") else None),
            reference_observation=(dict(row["reference_observation"]) if row.get("reference_observation") is not None else None),
            oracle_tool_calls=list(row.get("oracle_tool_calls") or []),
            noise_execution_calls=list(row.get("noise_execution_calls") or []),
            fault_profile=FaultProfile.from_dict(dict(row.get("fault_profile") or {})),
            initial_state=WorldState.from_dict(dict(row.get("initial_state") or {})),
            reference_final_state=WorldState.from_dict(dict(row.get("reference_final_state") or {})),
            domain=str(row.get("domain") or "unknown"),
            difficulty_level=str(row.get("difficulty_level") or "unknown"),
            canonical_task_key=str(row.get("canonical_task_key") or row["scenario_name"]),
            k=int(row.get("k", 1)),
        )


@dataclass
class Experiment3AttemptResult:
    attempt_index: int
    run: LLMAgentRunResult
    final_correct: bool
    state_correct: bool
    semantic_observation_correct: bool
    fault_triggered: bool
    injected_fault_count: int
    fault_indices: list[int]
    explicit_retry_detected: bool
    recovery_detected: bool
    state_corrupted: bool
    corruption_paths: list[str]
    step_budget: int
    tool_call_overhead: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_index": self.attempt_index,
            "run": self.run.to_dict(),
            "final_correct": self.final_correct,
            "state_correct": self.state_correct,
            "semantic_observation_correct": self.semantic_observation_correct,
            "fault_triggered": self.fault_triggered,
            "injected_fault_count": self.injected_fault_count,
            "fault_indices": self.fault_indices,
            "explicit_retry_detected": self.explicit_retry_detected,
            "recovery_detected": self.recovery_detected,
            "state_corrupted": self.state_corrupted,
            "corruption_paths": self.corruption_paths,
            "step_budget": self.step_budget,
            "tool_call_overhead": self.tool_call_overhead,
        }


@dataclass
class Experiment3CaseResult:
    case_name: str
    base_case_name: str
    source: str
    domain: str
    difficulty_level: str
    canonical_task_key: str
    strategy: str
    noise_type: str
    noise_target_tool: str
    noise_intensity: int
    k: int
    attempts: list[Experiment3AttemptResult]
    success_at_1: bool
    pass_at_k: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_name": self.case_name,
            "base_case_name": self.base_case_name,
            "source": self.source,
            "domain": self.domain,
            "difficulty_level": self.difficulty_level,
            "canonical_task_key": self.canonical_task_key,
            "strategy": self.strategy,
            "noise_type": self.noise_type,
            "noise_target_tool": self.noise_target_tool,
            "noise_intensity": self.noise_intensity,
            "k": self.k,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "success_at_1": self.success_at_1,
            "pass_at_k": self.pass_at_k,
        }


class LLMExperiment3NoiseRobustnessRunner:
    def __init__(
        self,
        client: LLMClient,
        *,
        config: LLMAgentConfig | None = None,
        attempts_per_case: int = 1,
    ) -> None:
        if attempts_per_case < 1:
            raise ValueError("attempts_per_case must be positive")
        self.client = client
        self.config = config or LLMAgentConfig(max_steps=16, temperature=None)
        self.attempts_per_case = attempts_per_case

    def run_case(self, case: Experiment3Case, strategy: str) -> Experiment3CaseResult:
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}")
        attempts = [self._run_attempt(case, strategy, index) for index in range(self.attempts_per_case)]
        return Experiment3CaseResult(
            case_name=case.case_name,
            base_case_name=case.base_case_name,
            source=case.source,
            domain=case.domain,
            difficulty_level=case.difficulty_level,
            canonical_task_key=case.canonical_task_key,
            strategy=strategy,
            noise_type=case.noise_type,
            noise_target_tool=case.noise_target_tool,
            noise_intensity=case.noise_intensity,
            k=self.attempts_per_case,
            attempts=attempts,
            success_at_1=attempts[0].final_correct,
            pass_at_k=any(attempt.final_correct for attempt in attempts),
        )

    def _run_attempt(self, case: Experiment3Case, strategy: str, attempt_index: int) -> Experiment3AttemptResult:
        nominal = max(1, len(case.noise_execution_calls))
        step_budget = min(self.config.max_steps, nominal + STRATEGY_EXTRA_STEPS[strategy])
        config = replace(
            self.config,
            max_steps=step_budget,
            reject_finish_when_goals_unmet=strategy == "self_refine",
        )
        agent = LLMAgentRunner(
            self.client,
            config=config,
            executor_config=ExecutorConfig(fault_profile=case.fault_profile),
        )
        run = agent.run(
            task_name=case.case_name,
            task_description=_task_prompt(case, strategy),
            mode="stateful",
            initial_state=WorldState.from_dict(case.initial_state.to_dict()),
            goals=case.execution_goals,
            tool_registry=policy_noise_tool_registry(case),
        )
        trace = run.experiment_result.trace
        trace_dicts = [record.to_dict() for record in trace]
        fault_indices = _fault_indices(trace_dicts, case.noise_type)
        semantic_correct = _semantic_correct(case, trace_dicts)
        if case.state_goals:
            state_correct = StateLevelEvaluator().evaluate(
                run.experiment_result.final_state,
                case.state_goals,
            ).all_passed
        else:
            state_correct = True
        final_correct = bool(state_correct and semantic_correct)
        last_fault = max(fault_indices) if fault_indices else -1
        explicit_retry = any(
            index > last_fault and record.tool_name == case.noise_target_tool
            for index, record in enumerate(trace)
        ) if fault_indices else False
        corrupted, paths = state_corruption(
            case.initial_state.to_dict(),
            case.reference_final_state.to_dict(),
            run.experiment_result.final_state.to_dict(),
        )
        return Experiment3AttemptResult(
            attempt_index=attempt_index,
            run=run,
            final_correct=final_correct,
            state_correct=bool(state_correct),
            semantic_observation_correct=semantic_correct,
            fault_triggered=bool(fault_indices),
            injected_fault_count=len(fault_indices),
            fault_indices=fault_indices,
            explicit_retry_detected=explicit_retry,
            recovery_detected=bool(final_correct and explicit_retry),
            state_corrupted=corrupted,
            corruption_paths=paths,
            step_budget=step_budget,
            tool_call_overhead=max(0, len(trace) - len(case.oracle_tool_calls)),
        )


def load_experiment3_cases(path: str | Path) -> list[Experiment3Case]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = [Experiment3Case.from_dict(row) for row in rows]
    if len({case.case_name for case in cases}) != len(cases):
        raise ValueError("Duplicate case names")
    return cases


def _task_prompt(case: Experiment3Case, strategy: str) -> str:
    instruction = {
        "direct": "Use the shortest plausible path; do not add optional diagnostic calls.",
        "cot": "Plan prerequisites explicitly and use one focused repair when evidence requires it.",
        "react": "Adapt to observations and actively retry or re-query inconsistent results.",
        "self_refine": "Verify the result before finishing and repair incomplete or suspicious state.",
    }[strategy]
    return (
        f"{case.task_prompt}\n\n"
        "Respond only with one JSON tool action per turn. Do not output prose outside the JSON action.\n\n"
        f"Reasoning strategy: {strategy}\nStrategy instruction: {instruction}"
    )


def _semantic_correct(case: Experiment3Case, trace: list[dict[str, Any]]) -> bool:
    if not case.semantic_observation_required:
        return True
    actual = None
    for record in reversed(trace):
        if record.get("tool_name") == case.semantic_observation_tool and record.get("success"):
            actual = record.get("observation")
            break
    return _canonical_observation(actual) == _canonical_observation(case.reference_observation)


def _canonical_observation(value: Any) -> Any:
    if value is None:
        return None
    data = json.loads(json.dumps(value))
    if isinstance(data, dict):
        data.pop("input", None)
        data.pop("tool_name", None)
    return data


def _fault_indices(trace: list[dict[str, Any]], noise_type: str) -> list[int]:
    expected = {
        "stale_state": "stale_observation",
        "vague_observation": "vague_observation",
        "misleading_observation": "misleading_observation",
    }.get(noise_type, noise_type)
    indices: list[int] = []
    for index, record in enumerate(trace):
        if (record.get("metadata") or {}).get("fault") == expected:
            indices.append(index)
            continue
        error = str(record.get("error") or "").lower().replace("_", " ")
        if "injected" in error and expected.replace("_", " ") in error:
            indices.append(index)
    return indices


def compute_experiment3_group_metrics_from_dict(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    sources = ["all", *sorted({str(row["source"]) for row in rows})]
    strategies = sorted({str(row["strategy"]) for row in rows})
    noises = sorted({str(row["noise_type"]) for row in rows})
    for source in sources:
        source_rows = rows if source == "all" else [row for row in rows if row["source"] == source]
        for strategy in strategies:
            for noise in noises:
                group = [row for row in source_rows if row["strategy"] == strategy and row["noise_type"] == noise]
                if group:
                    output.append(_summarize(source, strategy, noise, group))
    return output


def _summarize(source: str, strategy: str, noise_type: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    attempts = [attempt for row in rows for attempt in row.get("attempts", [])]
    faulted = [attempt for attempt in attempts if attempt["fault_triggered"]]
    traces = [attempt["run"]["experiment_result"]["trace"] for attempt in attempts]
    calls = sum(len(trace) for trace in traces)
    invalid = sum(
        int(attempt["run"]["experiment_result"]["call_metrics"].get("invalid_calls", 0))
        for attempt in attempts
    )
    total_cases = len(rows)
    total_attempts = len(attempts)
    return {
        "source": source,
        "strategy": strategy,
        "noise_type": noise_type,
        "total_cases": total_cases,
        "total_attempts": total_attempts,
        "success_at_1": _rate(sum(bool(row["success_at_1"]) for row in rows), total_cases),
        "pass_at_k": _rate(sum(bool(row["pass_at_k"]) for row in rows), total_cases),
        "attempt_accuracy": _rate(sum(bool(a["final_correct"]) for a in attempts), total_attempts),
        "state_accuracy": _rate(sum(bool(a["state_correct"]) for a in attempts), total_attempts),
        "semantic_observation_accuracy": _rate(sum(bool(a["semantic_observation_correct"]) for a in attempts), total_attempts),
        "noise_trigger_rate": _rate(len(faulted), total_attempts),
        "full_intensity_trigger_rate": _rate(
            sum(int(a["injected_fault_count"]) >= int(row["noise_intensity"]) for row in rows for a in row["attempts"]),
            total_attempts,
        ),
        "recovery_rate": _optional_rate(sum(bool(a["recovery_detected"]) for a in faulted), len(faulted)),
        "explicit_retry_rate": _optional_rate(sum(bool(a["explicit_retry_detected"]) for a in faulted), len(faulted)),
        "state_corruption_rate": _rate(sum(bool(a["state_corrupted"]) for a in attempts), total_attempts),
        "invalid_call_rate": _rate(invalid, calls),
        "average_tool_calls": _rate(calls, total_attempts),
        "average_tokens": _rate(sum(int(a["run"].get("total_tokens", 0)) for a in attempts), total_attempts),
        "average_latency_seconds": _rate(sum(float(a["run"].get("total_llm_latency_ms", 0.0)) for a in attempts), total_attempts) / 1000.0,
    }


def render_experiment3_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Experiment 3 Calibrated Noise Robustness",
        "",
        f"- Model: `{result.get('model', 'unknown')}`",
        f"- Completed case-strategy groups: {len(result.get('case_results', []))}",
        f"- Independent faulted attempts per group: {result.get('attempts_per_case', 0)}",
        "- Clean branch: `disabled`",
        "",
        "| Strategy | Noise | Cases | Success@1 | Pass@k | Attempt Acc. | State Acc. | Semantic Acc. | Full Noise | Recovery | Corruption |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in result.get("group_metrics", []):
        if metric.get("source") != "all":
            continue
        lines.append(
            f"| {metric['strategy']} | {metric['noise_type']} | {metric['total_cases']} | "
            f"{_pct(metric['success_at_1'])} | {_pct(metric['pass_at_k'])} | {_pct(metric['attempt_accuracy'])} | "
            f"{_pct(metric['state_accuracy'])} | {_pct(metric['semantic_observation_accuracy'])} | "
            f"{_pct(metric['full_intensity_trigger_rate'])} | {_optional_pct(metric['recovery_rate'])} | "
            f"{_pct(metric['state_corruption_rate'])} |"
        )
    return "\n".join(lines) + "\n"


def _rate(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _optional_rate(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _optional_pct(value: float | None) -> str:
    return "n/a" if value is None else _pct(value)

