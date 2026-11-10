"""Offline, scripted stale-state reachability against the original runner and oracle."""

from collections import Counter
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from toolsim.core.world_state import WorldState
from toolsim.evaluators.evaluator import StateLevelEvaluator
from toolsim.execution.stateful_executor import ExecutorConfig
from toolsim.faults import FaultInjector
from toolsim.llm.types import LLMResponse
from toolsim.runners.llm_agent_runner import LLMAgentConfig, LLMAgentRunner, _goals_passed
from toolsim.runners.llm_experiment3_noise_robustness import (
    STRATEGY_EXTRA_STEPS, _canonical_observation, _fault_indices, _semantic_correct, _task_prompt,
    load_experiment3_cases,
)
from toolsim.runners.noise_tool_registry import policy_noise_tool_registry

CASES = ROOT / "outputs/cases/experiment3_llm_deepseek_cases_toolsandbox_original_100_noise_v3.jsonl"
CONDITIONS = {"A": (False, True), "B": (True, True), "C": (False, False),
              "D": (True, False), "D_negative": (True, False)}
ORIGINAL_CAP = 16


class ScriptedClient:
    model = "scripted"

    def __init__(self, actions):
        self.actions = iter(actions)

    def complete(self, messages, **kwargs):
        return LLMResponse(json.dumps(next(self.actions, {"finish": True})), self.model, 0)


class RecordingFaultInjector(FaultInjector):
    def after_call(self, tool_name, result):
        raw_observation = deepcopy(result.observation)
        adjusted = super().after_call(tool_name, result)
        adjusted.metadata = {**adjusted.metadata, "scripted_raw_observation": raw_observation}
        return adjusted


def scripted_actions(case):
    actions = case.noise_execution_calls
    target_indices = [index for index, action in enumerate(actions)
                      if action["tool_name"] == case.noise_target_tool]
    if (len(target_indices) != 2 or target_indices[-1] != len(actions) - 1
            or actions[target_indices[0]]["args"] != actions[-1]["args"]
            or case.fault_profile.stale_observation_replays != {case.noise_target_tool: 1}):
        raise ValueError(f"Unexpected source script or replay profile: {case.case_name}")
    return [*actions, actions[-1]]


def run_condition(case, strategy, budget, condition, actions):
    return execute_condition(case, strategy, budget, condition, ScriptedClient(actions))


def execute_condition(case, strategy, budget, condition, client, *, temperature=0.0, max_tokens=None):
    stale, early_stop = CONDITIONS[condition]
    config = LLMAgentConfig(max_steps=budget, stop_on_goals_passed=early_stop,
                            temperature=temperature, max_tokens=max_tokens,
                            reject_finish_when_goals_unmet=strategy == "self_refine")
    profile = case.fault_profile if stale else replace(
        case.fault_profile, stale_observation_replays={}, stale_observation_tools=set())
    runner = LLMAgentRunner(client, config=config,
                           executor_config=ExecutorConfig(fault_profile=profile))
    with patch("toolsim.execution.stateful_executor.FaultInjector", RecordingFaultInjector):
        result = runner.run(task_name=case.case_name, task_description=_task_prompt(case, strategy),
                            mode="stateful", initial_state=case.initial_state, goals=case.execution_goals,
                            tool_registry=policy_noise_tool_registry(case))
    return config, result


def describe(case, strategy, condition, regime, config, result, clean_trace=None):
    trace = [record.to_dict() for record in result.experiment_result.trace]
    if clean_trace is not None and len(clean_trace) < len(trace):
        raise AssertionError("Clean matched prefix is too short")
    evaluator = StateLevelEvaluator()
    faults = _fault_indices(trace, case.noise_type)
    last_fault = faults[-1] if faults else None
    timeline = []
    finish_rejections = 0
    state = case.initial_state
    state_correct = not case.state_goals or evaluator.evaluate(state, case.state_goals).all_passed
    terminal_reason = "budget_exhausted"
    last_fault_decision = None
    for decision in result.llm_steps:
        step = decision.execution_record
        if step is None:
            if decision.parse_error:
                if clean_trace is not None:
                    raise AssertionError("Script returned an invalid action")
                timeline.append({"decision": decision.step_index + 1, "event": "parse_error", "error": decision.parse_error})
                continue
            if not decision.parsed_action:
                raise AssertionError("Script returned an invalid action")
            if config.reject_finish_when_goals_unmet and not _goals_passed(state, case.execution_goals, evaluator):
                finish_rejections += 1
                event = "finish_rejected"
            else:
                terminal_reason = "policy_finish"
                event = "policy_finish"
            timeline.append({"decision": decision.step_index + 1, "event": event})
            continue
        index = sum("tool" in entry for entry in timeline)
        if clean_trace is not None:
            clean = clean_trace[index]
            for key in ("tool_name", "args", "success"):
                if step[key] != clean[key]:
                    raise AssertionError((case.case_name, condition, index, key))
        state = WorldState.from_dict(step["post_state_snapshot"])
        state_correct = not case.state_goals or evaluator.evaluate(state, case.state_goals).all_passed
        visible = step["observation"]
        actual = step["metadata"].get("scripted_raw_observation")
        if index not in faults and actual is not None and actual != visible:
            raise AssertionError((case.case_name, condition, index, "unexplained observation difference"))
        entry = {"decision": decision.step_index + 1, "tool": step["tool_name"], "args": step["args"],
                 "success": step["success"], "error": step["error"], "pre_state_hash": step["pre_state_hash"],
                 "post_state_hash": step["post_state_hash"], "raw_observation": actual,
                 "visible_observation": visible, "replay_triggered": index in faults,
                 "observation_actually_stale": index in faults and
                 _canonical_observation(actual) != _canonical_observation(visible),
                 "state_correct": state_correct, "observation_correct": _semantic_correct(case, trace[:index + 1])}
        if index in faults:
            previous = next(item for item in reversed(timeline) if item.get("tool") == case.noise_target_tool
                            and item.get("raw_observation") is not None)
            entry["cache_decision"] = previous["decision"]
            if previous["raw_observation"] != visible:
                raise AssertionError((case.case_name, "replay differs from cached response"))
        if index == last_fault:
            last_fault_decision = entry["decision"]
        if config.stop_on_goals_passed and _goals_passed(state, case.execution_goals, evaluator):
            terminal_reason = "early_goal_stop"
        timeline.append(entry)
    observation_correct = _semantic_correct(case, trace)
    retry = last_fault is not None and any(step["tool_name"] == case.noise_target_tool
                                           for step in trace[last_fault + 1:])
    fresh_retry = last_fault_decision is not None and any(
        entry["decision"] > last_fault_decision and entry.get("tool") == case.noise_target_tool
        and entry["success"] and entry["raw_observation"] is not None
        and entry["raw_observation"] == entry["visible_observation"] for entry in timeline)
    joint = state_correct and observation_correct
    return {"case": case.case_name, "condition": condition, "regime": regime, "strategy": strategy,
            "decision_budget": config.max_steps, "decisions": len(result.llm_steps), "stop_reason": terminal_reason,
            "finish_rejections": finish_rejections, "state_correct": state_correct,
            "observation_correct": observation_correct, "joint_correct": joint, "fault_count": len(faults),
            "genuine_stale": any(entry.get("observation_actually_stale") for entry in timeline),
            "decision_after_fault": last_fault_decision is not None and len(result.llm_steps) > last_fault_decision,
            "tool_after_fault": last_fault is not None and last_fault + 1 < len(trace),
            "explicit_retry": retry, "fresh_retry": fresh_retry, "recovery": joint and retry, "timeline": timeline}


def run_matrix(case, strategy, regime, budget):
    actions = scripted_actions(case)
    results = {}
    rows = []
    for condition in CONDITIONS:
        sequence = actions[:-1] if condition == "D_negative" else actions
        config, result = run_condition(case, strategy, budget, condition, sequence)
        results[condition] = [record.to_dict() for record in result.experiment_result.trace]
        clean = {"B": "A", "D": "C", "D_negative": "C"}.get(condition, condition)
        rows.append(describe(case, strategy, condition, regime, config, result, results[clean]))
    positive, negative = rows[-2:]
    positive_prefix = [step for step in positive["timeline"] if "tool" in step][:len(actions) - 1]
    negative_prefix = [step for step in negative["timeline"] if "tool" in step]
    if [(step["tool"], step["args"], step["replay_triggered"]) for step in positive_prefix] != [
            (step["tool"], step["args"], step["replay_triggered"]) for step in negative_prefix]:
        raise AssertionError((case.case_name, "negative control differs before recovery"))
    if negative["explicit_retry"] or (positive["recovery"] and not positive["fresh_retry"]):
        raise AssertionError((case.case_name, "unverified correction or retry in the negative control"))
    return rows


def tally(rows):
    flags = ("state_correct", "observation_correct", "joint_correct", "genuine_stale", "decision_after_fault",
             "tool_after_fault", "explicit_retry", "fresh_retry", "recovery")
    result = {key: sum(bool(row[key]) for row in rows) for key in flags}
    result.update(n=len(rows), faulted=sum(row["fault_count"] > 0 for row in rows),
                  fault_events=sum(row["fault_count"] for row in rows),
                  finish_rejections=sum(row["finish_rejections"] for row in rows),
                  stop_reasons=dict(Counter(row["stop_reason"] for row in rows)))
    result["recovery_rate"] = result["recovery"] / result["faulted"] if result["faulted"] else None
    return result


def main():
    cases = [case for case in load_experiment3_cases(CASES) if case.noise_type == "stale_state"]
    if len(cases) != 20:
        raise ValueError(f"Expected 20 stale-state cases, got {len(cases)}")
    rows = []
    for case in cases:
        nominal = len(case.noise_execution_calls)
        for strategy, extra in STRATEGY_EXTRA_STEPS.items():
            rows.extend(run_matrix(case, strategy, "original_budget", min(ORIGINAL_CAP, nominal + extra)))
        rows.extend(run_matrix(case, "direct", "expanded_budget_diagnostic", nominal + 2))
    output = HERE / "results"
    output.mkdir(exist_ok=True)
    with (output / "scripted_traces.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    grouped = {regime: {condition: tally([row for row in rows if row["regime"] == regime
                                         and row["condition"] == condition]) for condition in CONDITIONS}
               for regime in ("original_budget", "expanded_budget_diagnostic")}
    per_task = [{"case": case.case_name, "target": case.noise_target_tool, "reference_observation": case.reference_observation,
                 "runs": [{key: value for key, value in row.items() if key not in ("timeline", "case")}
                          for row in rows if row["case"] == case.case_name]} for case in cases]
    summary = {"source_sha256": sha256(CASES.read_bytes()).hexdigest(), "tasks": len(cases), "runs": len(rows),
               "groups": grouped, "by_strategy_original_budget": {
                   strategy: {condition: tally([row for row in rows if row["regime"] == "original_budget"
                                                and row["strategy"] == strategy and row["condition"] == condition])
                              for condition in CONDITIONS} for strategy in STRATEGY_EXTRA_STEPS},
               "per_task": per_task}
    (output / "scripted_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(grouped, indent=2))


if __name__ == "__main__":
    main()
