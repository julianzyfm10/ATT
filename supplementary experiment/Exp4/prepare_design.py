"""Freeze the Exp4 task sample, configuration matrix, and reference-flow audit."""

from collections import Counter, defaultdict
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from toolsim.llm.types import LLMResponse
from toolsim.runners.core_local_dataset import DEFAULT_DATASET, load_core_local_cases
from toolsim.runners.core_local_oracle import run_oracle_case
from toolsim.runners.experiment2_cases import DEFAULT_CASES, load_frozen_cases
from toolsim.runners.llm_agent_runner import LLMAgentConfig, LLMAgentRunner
from toolsim.runners.llm_core_local import CORE_LOCAL_BACKENDS, CORE_LOCAL_STRATEGIES, _config_for_case
from toolsim.runners.llm_experiment2_dependency_curve import LLMExperiment2DependencyCurveRunner, _config_for_strategy

MODELS = ("Qwen/Qwen3.6-35B-A3B", "google/gemma-4-31B-it")
LEVEL_COUNTS = {"L1_SINGLE_TOOL": 8, "L2_EXPLICIT_DEP": 11, "L3_IMPLICIT_SIDE_EFFECT": 11}
CONDITIONS = ("A", "B", "C")
SEED = "ATT-Exp4-dependency-task-sample"
EQUAL_STEPS = 16
BASE = LLMAgentConfig(max_steps=8, temperature=0.0, max_tokens=1024)


class ReferenceClient:
    model = "scripted-reference"

    def __init__(self, actions):
        self.actions = iter(actions)

    def complete(self, messages, **kwargs):
        action = next(self.actions, {"finish": True})
        return LLMResponse(json.dumps(action), self.model, 0)


def select_dependency():
    groups = defaultdict(list)
    for line in DEFAULT_CASES.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            groups[row["difficulty_level"]].append(row)
    selected = []
    for level, count in LEVEL_COUNTS.items():
        ordered = sorted(groups[level], key=lambda row: sha256(f"{SEED}:{row['case_name']}".encode()).hexdigest())
        distinct, deferred, seen = [], [], set()
        for row in ordered:
            task = " ".join((row.get("canonical_task_key") or row["case_name"]).casefold().split())
            (deferred if task in seen else distinct).append(row)
            seen.add(task)
        selected.extend((distinct + deferred)[:count])
    if Counter(row["difficulty_level"] for row in selected) != LEVEL_COUNTS:
        raise ValueError("Dependency difficulty strata do not have the required sizes")
    return selected


def config_for(study, strategy, reference_steps, condition):
    if study == "dependency":
        original = _config_for_strategy(replace(BASE, stop_on_goals_passed=False), strategy)
    elif study == "backend":
        original = _config_for_case(BASE, strategy, reference_steps)
    else:
        raise ValueError(study)
    if condition == "A":
        return original
    if condition == "B":
        return replace(original, max_steps=EQUAL_STEPS)
    if condition == "C":
        return replace(original, max_steps=EQUAL_STEPS, stop_on_goals_passed=False,
                       reject_finish_when_goals_unmet=False)
    raise ValueError(condition)


def runtime_configs(study, reference_steps):
    result = {}
    for strategy in CORE_LOCAL_STRATEGIES:
        values = {condition: asdict(config_for(study, strategy, reference_steps, condition)) for condition in CONDITIONS}
        original, equal, autonomous = (values[condition] for condition in CONDITIONS)
        assert equal["max_steps"] == autonomous["max_steps"] == EQUAL_STEPS
        assert {key for key in original if original[key] != equal[key]} <= {"max_steps"}
        assert {key for key in equal if equal[key] != autonomous[key]} <= {
            "stop_on_goals_passed", "reject_finish_when_goals_unmet"}
        result[strategy] = values
    return result


def prepare():
    chosen = select_dependency()
    dependency = {case.case_name: case for case in load_frozen_cases(DEFAULT_CASES)}
    backend = load_core_local_cases(DEFAULT_DATASET)
    assert len(backend) == len({case.case_id for case in backend}) == 30
    cases = {"dependency": [], "backend": []}
    for row in chosen:
        case = dependency[row["case_name"]]
        steps = len(case.oracle_tool_calls)
        client = ReferenceClient(case.oracle_tool_calls)
        config = config_for("dependency", "direct", steps, "C")
        result = LLMExperiment2DependencyCurveRunner(client).run_case(LLMAgentRunner(client, config=config), case, "direct")
        cases["dependency"].append({"case_id": case.case_name, "level": case.difficulty_level,
                                    "canonical_task_key": row.get("canonical_task_key"), "reference_steps": steps,
                                    "reference_decisions": len(result.run.llm_steps),
                                    "reference_oracle_pass": result.dependency_correct,
                                    "reference_terminal_pass": result.final_state_correct,
                                    "reference_missing_prerequisites": result.missing_prerequisite_count,
                                    "reference_wrong_order": result.wrong_order_call_count,
                                    "configs": runtime_configs("dependency", steps)})
    for case in backend:
        result = run_oracle_case(case, "mock")
        steps = len(case.oracle_plan)
        cases["backend"].append({"case_id": case.case_id, "level": case.data["difficulty"], "reference_steps": steps,
                                 "reference_decisions": steps + 1, "reference_oracle_pass": result.final_correct and result.dependency_compliant,
                                 "reference_backend": "mock", "configs": runtime_configs("backend", steps)})
    max_decisions = {study: max(row["reference_decisions"] for row in records) for study, records in cases.items()}
    assert all(value <= EQUAL_STEPS for value in max_decisions.values())
    failures = {study: [row["case_id"] for row in records if not row["reference_oracle_pass"]]
                for study, records in cases.items()}
    backend_groups = {"dependency": ["mock"], "backend": list(CORE_LOCAL_BACKENDS)}
    counts = {study: len(records) * len(MODELS) * len(CONDITIONS) * len(CORE_LOCAL_STRATEGIES) * len(backend_groups[study])
              for study, records in cases.items()}
    manifest = {"sample_seed": SEED, "models": list(MODELS), "strategies": CORE_LOCAL_STRATEGIES,
                "conditions": list(CONDITIONS), "thinking": False, "equal_decision_budget": EQUAL_STEPS,
                "source_sha256": {str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
                                  for path in (DEFAULT_CASES, DEFAULT_DATASET)},
                "backends": backend_groups, "runs_by_study": counts, "total_runs": sum(counts.values()),
                "max_reference_decisions": max_decisions, "reference_oracle_failures": failures, "cases": cases}
    path = HERE / "design_manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Frozen manifest differs from the current design or reference replay")
    else:
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runs": counts, "max_reference_decisions": max_decisions, "reference_oracle_failures": failures},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    prepare()
