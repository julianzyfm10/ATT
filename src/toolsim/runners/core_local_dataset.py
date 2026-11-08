"""Loader and rule-based evaluator for the Experiment 4 Core Local dataset."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from toolsim.core.world_state import WorldState
from toolsim.evaluators.evaluator import StateEvaluationResult, StateGoalResult, StateLevelEvaluator


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = ROOT / "data" / "benchmark" / "experiment4_mcpbench_core_local_30.jsonl"


@dataclass(frozen=True)
class CoreLocalCase:
    data: dict[str, Any]

    @property
    def case_id(self) -> str:
        return str(self.data["case_id"])

    @property
    def servers(self) -> list[str]:
        return list(self.data["servers"])

    @property
    def prompt(self) -> str:
        return str(self.data["user_prompt"])

    @property
    def oracle_plan(self) -> list[dict[str, Any]]:
        return list(self.data["oracle_plan"])

    @property
    def goals(self) -> list[dict[str, Any]]:
        return list(self.data["goals"])


def load_core_local_cases(path: str | Path | None = None) -> list[CoreLocalCase]:
    dataset_path = Path(path) if path is not None else DEFAULT_DATASET
    return [
        CoreLocalCase(json.loads(line))
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def state_calls(state: WorldState) -> list[dict[str, Any]]:
    return sorted(state.entities.get("core_local_call", {}).values(), key=lambda item: int(item["sequence"]))


def match_oracle_steps(case: CoreLocalCase, state: WorldState) -> dict[str, dict[str, Any]]:
    calls = state_calls(state)
    matched: dict[str, dict[str, Any]] = {}
    matched_indices: dict[str, int] = {}
    used: set[int] = set()
    for oracle in case.oracle_plan:
        minimum_index = max((matched_indices.get(dep, -1) for dep in oracle.get("depends_on", [])), default=-1)
        expected_failure = bool(oracle.get("expected_failure"))
        for index, call in enumerate(calls):
            if index in used or index <= minimum_index:
                continue
            if call.get("server_id") != oracle.get("server_id") or call.get("tool_name") != oracle.get("tool_name"):
                continue
            if expected_failure == bool(call.get("success")):
                continue
            if not _args_compatible(oracle.get("args", {}), call.get("input", {})):
                continue
            matched[oracle["step_id"]] = call
            matched_indices[oracle["step_id"]] = index
            used.add(index)
            break
    return matched


def dependency_compliance(case: CoreLocalCase, state: WorldState) -> bool:
    matched = match_oracle_steps(case, state)
    required_steps = [item["step_id"] for item in case.oracle_plan if not item.get("expected_failure")]
    return all(step_id in matched for step_id in required_steps)


class CoreLocalStateEvaluator(StateLevelEvaluator):
    def __init__(self, case: CoreLocalCase) -> None:
        self.case = case

    def evaluate(self, state: WorldState, goals: Iterable[dict[str, Any]]) -> StateEvaluationResult:
        del goals
        matched = match_oracle_steps(self.case, state)
        details = [self._evaluate_case_goal(goal, matched) for goal in self.case.goals]
        passed = sum(detail.passed for detail in details)
        return StateEvaluationResult(
            goal_count=len(details),
            passed_count=passed,
            failed_count=len(details) - passed,
            all_passed=bool(details) and passed == len(details),
            details=details,
        )

    def _evaluate_case_goal(self, goal: dict[str, Any], matched: dict[str, dict[str, Any]]) -> StateGoalResult:
        source = str(goal["source"])
        if source not in matched:
            return StateGoalResult("core_local", False, f"Oracle step not observed: {source}")
        actual = _get_path(matched[source], str(goal.get("path", "output")))
        operation = str(goal["op"])
        expected = goal.get("expected")
        tolerance = float(goal.get("tolerance", 0.0))
        try:
            passed = _compare(operation, actual, expected, tolerance)
        except Exception as exc:
            return StateGoalResult("core_local", False, f"{source}: evaluator error: {exc}")
        message = f"{source}.{goal.get('path')} {operation} {expected!r}; actual={actual!r}"
        return StateGoalResult("core_local", passed, message)


def _args_compatible(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict) and "$ref" in expected:
        return True
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        actual_keys = {str(key).lower(): key for key in actual}
        for key, value in expected.items():
            actual_key = key if key in actual else actual_keys.get(str(key).lower())
            if actual_key is None or not _args_compatible(value, actual[actual_key]):
                return False
        return True
    if isinstance(expected, list):
        return isinstance(actual, list) and len(expected) == len(actual) and all(_args_compatible(left, right) for left, right in zip(expected, actual))
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(float(expected), float(actual), rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(expected, str) and isinstance(actual, str):
        aliases = {
            "feet": "foot", "meters": "meter", "miles": "mile", "kilometers": "kilometer",
            "inches": "inch", "pounds": "pound", "kilograms": "kilogram", "mph": "miles per hour",
            "m/s": "meters per second", "minute": "minutes", "hour": "hours", "°f": "fahrenheit",
            "f": "fahrenheit", "°c": "celsius", "c": "celsius",
        }
        left = aliases.get(expected.strip().lower(), expected.strip().lower())
        right = aliases.get(actual.strip().lower(), actual.strip().lower())
        return left == right
    return expected == actual


def _get_path(value: Any, path: str) -> Any:
    current = value
    for part in path.split(".") if path else []:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _numeric(value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError("boolean is not numeric")
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).strip())


def _flatten(value: Any) -> list[float]:
    if isinstance(value, (list, tuple)):
        result: list[float] = []
        for item in value:
            result.extend(_flatten(item))
        return result
    return [_numeric(value)]


def _compare(operation: str, actual: Any, expected: Any, tolerance: float) -> bool:
    if operation == "equals":
        return actual == expected
    if operation == "contains":
        return str(expected).lower() in str(actual).lower()
    if operation in {"numeric_equals", "approx"}:
        absolute = tolerance if operation == "approx" else 1e-9
        return math.isclose(_numeric(actual), _numeric(expected), rel_tol=1e-9, abs_tol=absolute)
    if operation == "array_approx":
        left, right = _flatten(actual), _flatten(expected)
        return len(left) == len(right) and all(math.isclose(a, b, rel_tol=1e-8, abs_tol=tolerance) for a, b in zip(left, right))
    if operation == "unordered_array_approx":
        left, right = sorted(_flatten(actual)), sorted(_flatten(expected))
        return len(left) == len(right) and all(math.isclose(a, b, rel_tol=1e-8, abs_tol=tolerance) for a, b in zip(left, right))
    if operation == "time_equals":
        return f"T{expected}" in str(actual) or str(actual).startswith(str(expected))
    raise ValueError(f"Unsupported Core Local evaluator operation: {operation}")

