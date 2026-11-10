"""Run the frozen Exp4 matrix with one SQLite commit per completed episode."""

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import time

from prepare_design import HERE, ROOT, DEFAULT_CASES, DEFAULT_DATASET, load_frozen_cases, load_core_local_cases
from siliconflow_support import load_local_environment
from toolsim.core.world_state import WorldState
from toolsim.evaluators.evaluator import StateLevelEvaluator
from toolsim.llm.siliconflow_client import SiliconFlowClient
from toolsim.llm.types import LLMMessage
from toolsim.runners.core_local_dataset import CoreLocalStateEvaluator, dependency_compliance
from toolsim.runners.core_local_oracle import backend_for_name
from toolsim.runners.llm_agent_runner import LLMAgentConfig, LLMAgentRunner, _goals_passed, parse_llm_tool_action
from toolsim.runners.llm_core_local import _task_prompt as backend_prompt
from toolsim.runners.llm_experiment2_dependency_curve import LLMExperiment2DependencyCurveRunner
from toolsim.tools.core_local_mcp_tools import registry_for_servers
from openai import APIConnectionError, InternalServerError

DESIGN = HERE / "design_manifest.json"
QUALIFICATION = ROOT / "outputs/core_local/experiment4_core_local_oracle_summary_rebuttal_exp4_qualified_20261110.json"


class BackendTransportError(RuntimeError):
    pass


class PacedClient:
    def __init__(self, client):
        self.client = client
        self.model = client.model
        self.next_request = 0.0

    def complete(self, messages, **kwargs):
        for attempt in range(4):
            time.sleep(max(0, self.next_request - time.monotonic()))
            self.next_request = time.monotonic() + 1.5
            try:
                return self.client.complete(messages, **kwargs)
            except (APIConnectionError, InternalServerError) as exc:
                if attempt == 3:
                    raise
                delay = 10 * 2 ** attempt
                print(f"{type(exc).__name__}; retry in {delay}s", flush=True)
                time.sleep(delay)


def design():
    manifest = json.loads(DESIGN.read_text(encoding="utf-8"))
    for path, digest in manifest["source_sha256"].items():
        if sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen source changed: {path}")
    datasets = {"dependency": {case.case_name: case for case in load_frozen_cases(DEFAULT_CASES)},
                "backend": {case.case_id: case for case in load_core_local_cases(DEFAULT_DATASET)}}
    matrix = {}
    for study, records in manifest["cases"].items():
        for record in records:
            for model in manifest["models"]:
                for strategy in manifest["strategies"]:
                    for backend in manifest["backends"][study]:
                        block = study, model, record["case_id"], strategy, backend
                        order = sorted(manifest["conditions"], key=lambda name: sha256(repr((*block, name)).encode()).digest())
                        for condition in order:
                            matrix[(*block, condition)] = (datasets[study][record["case_id"]], record["level"],
                                                          record["configs"][strategy][condition])
    if len(matrix) != manifest["total_runs"] or any(manifest["reference_oracle_failures"].values()):
        raise ValueError("Frozen design failed its reference checks")
    return manifest, matrix


def termination(run, initial, goals, evaluator, config):
    state = initial
    rejected = explicit = 0
    reason = "budget_exhausted"
    for step in run.llm_steps:
        if step.execution_record is not None:
            state = WorldState.from_dict(step.execution_record["post_state_snapshot"])
            if config.stop_on_goals_passed and _goals_passed(state, goals, evaluator):
                reason = "early_goal_stop"
        elif step.parsed_action is not None and step.parse_error is None:
            explicit += step.parsed_action.get("finish") is True
            if config.reject_finish_when_goals_unmet and not _goals_passed(state, goals, evaluator):
                rejected += 1
            else:
                reason = "policy_finish"
    assert len(run.llm_steps) <= config.max_steps
    assert reason != "budget_exhausted" or len(run.llm_steps) == config.max_steps
    return {"stop_reason": reason, "finish_rejections": rejected, "explicit_finish_actions": explicit}


def execute(key, spec, client):
    study, model, case_id, strategy, backend_name, condition = key
    case, level, config_dict = spec
    config = LLMAgentConfig(**config_dict)
    agent = LLMAgentRunner(client, config=config)
    if study == "dependency":
        result = LLMExperiment2DependencyCurveRunner(client).run_case(agent, case, strategy)
        run = result.run
        state_correct = result.final_state_correct
        joint = result.dependency_correct
        primary = joint
        initial = case.initial_state
        evaluator = StateLevelEvaluator()
        components = {"dependency_completion_rate": result.dependency_completion_rate,
                      "wrong_order_call_count": result.wrong_order_call_count,
                      "missing_prerequisite_count": result.missing_prerequisite_count}
    else:
        backend = backend_for_name(backend_name, case_id)
        transport_failures = []
        if backend_name == "local_mcp":
            call_tool = backend.call_core_local_tool

            def observe_transport(*args, **kwargs):
                try:
                    return call_tool(*args, **kwargs)
                except Exception as exc:
                    transport_failures.append(type(exc).__name__)
                    raise

            backend.call_core_local_tool = observe_transport
        try:
            initial = backend.create_state()
            evaluator = CoreLocalStateEvaluator(case)
            run = agent.run(task_name=case_id, task_description=backend_prompt(case, strategy), mode="stateful",
                            initial_state=None, goals=case.goals, tool_registry=registry_for_servers(case.servers),
                            state_evaluator=evaluator, backend=backend)
            if transport_failures:
                raise BackendTransportError(", ".join(transport_failures))
            state_correct = bool(run.experiment_result.state_metrics and run.experiment_result.state_metrics.all_passed)
            compliant = dependency_compliance(case, run.experiment_result.final_state)
            joint = state_correct and compliant
            primary = state_correct
            components = {"dependency_compliant": compliant}
        finally:
            close = getattr(backend, "close", None)
            if callable(close):
                close()
    return {"study": study, "model": model, "case": case_id, "strategy": strategy, "backend": backend_name,
            "condition": condition, "level": level, "decision_budget": config.max_steps, "primary": primary,
            "state_correct": state_correct, "joint_correct": joint, "decisions": len(run.llm_steps),
            "tool_calls": len(run.experiment_result.trace), "parse_errors": run.parse_error_count,
            "input_tokens": run.total_prompt_tokens, "output_tokens": run.total_completion_tokens,
            **termination(run, initial, case.goals, evaluator, config), **components, "run": run.to_dict()}


def metrics(rows):
    return {"n": len(rows), **{key: sum(row[key] for row in rows) for key in (
        "primary", "state_correct", "joint_correct", "decisions", "tool_calls", "parse_errors", "input_tokens",
        "output_tokens", "finish_rejections", "explicit_finish_actions")},
        "stop_reasons": dict(Counter(row["stop_reason"] for row in rows))}


def report(db, matrix, output):
    rows = {tuple(json.loads(key)): json.loads(value) for key, value in db.execute("SELECT key,value FROM results")}
    if rows.keys() != matrix.keys():
        raise ValueError("Incomplete matrix")
    strata, pairs = defaultdict(list), defaultdict(list)
    for key, row in rows.items():
        study, model, case, strategy, backend, condition = key
        stratum = (study, model, strategy, row["level"] if study == "dependency" else backend)
        strata[(*stratum, condition)].append(row)
        if condition in ("A", "B"):
            other = "B" if condition == "A" else "C"
            pairs[(*stratum, f"{condition}_to_{other}")].append((row, rows[(*key[:5], other)]))
        if study == "backend" and backend == "mock":
            pairs[(study, model, strategy, condition, "mock_to_local_mcp")].append(
                (row, rows[(study, model, case, strategy, "local_mcp", condition)]))
    paired = []
    for key, values in sorted(pairs.items()):
        for field in ("primary", "joint_correct"):
            gain = sum(not before[field] and after[field] for before, after in values)
            loss = sum(before[field] and not after[field] for before, after in values)
            paired.append({"stratum": key, "metric": field, "n": len(values), "corrected": gain, "introduced": loss,
                           "unchanged": len(values) - gain - loss, "delta_pp": 100 * (gain - loss) / len(values)})
    gaps = []
    for study, model, strategy, stratum, condition in sorted(strata):
        if (study, stratum) == ("dependency", "L1_SINGLE_TOOL"):
            high = strata[study, model, strategy, "L3_IMPLICIT_SIDE_EFFECT", condition]
        elif (study, stratum) == ("backend", "mock"):
            high = strata[study, model, strategy, "local_mcp", condition]
        else:
            continue
        low = strata[study, model, strategy, stratum, condition]
        gaps.append({"study": study, "model": model, "strategy": strategy, "condition": condition,
                     "first_correct": sum(row["primary"] for row in low), "first_n": len(low),
                     "last_correct": sum(row["primary"] for row in high), "last_n": len(high),
                     "gap_pp": 100 * (sum(row["primary"] for row in low) / len(low) -
                                      sum(row["primary"] for row in high) / len(high))})
    summary = {"runs": len(rows), "strata": [{"stratum": key, **metrics(values)} for key, values in sorted(strata.items())],
               "paired": paired, "gaps": gaps}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Exp4 等预算与 Oracle 干预", "", "正差值表示 L1 或 mock 的正确率较高。", "",
             "| 实验线 | 模型 | 策略 | 条件 | L1 / mock | L3 / local MCP | 差距（百分点） |",
             "|---|---|---|---|---:|---:|---:|"]
    for row in gaps:
        lines.append(f"| {row['study']} | {row['model']} | {row['strategy']} | {row['condition']} | "
                     f"{row['first_correct']}/{row['first_n']} | {row['last_correct']}/{row['last_n']} | {row['gap_pp']:+.2f} |")
    lines += ["", "分层正确数、配对增减及运行时终止统计见 `summary.json`。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def smoke(clients, output):
    for model, client in clients.items():
        response = client.complete([
            LLMMessage("system", "Respond with one JSON tool action. Include tool_name, args, and finish."),
            LLMMessage("user", 'Call get_wifi_status with {} and set finish to false.')],
            response_format={"type": "json_object"}, temperature=0, max_tokens=1024)
        action, error = parse_llm_tool_action(response.content)
        if error or action.get("tool_name") != "get_wifi_status" or action.get("args") != {} or action.get("finish") is not False:
            raise ValueError(f"Model action format smoke test failed: {model}")
        print(f"Model action smoke passed: {model}", flush=True)
    (output / "model_smoke.json").write_text(json.dumps({model: "pass" for model in clients}, indent=2), encoding="utf-8")


def run(manifest, matrix, output):
    qualification = json.loads(QUALIFICATION.read_text(encoding="utf-8"))
    if {item["backend"]: (item["runs"], item["correct"], item["dependency_compliant"])
            for item in qualification["backend_metrics"]} != {name: (30, 30, 30) for name in ("sandbox", "local_mcp")}:
        raise ValueError("Fresh sandbox and local MCP qualification did not pass")
    load_local_environment()
    clients = {model: PacedClient(SiliconFlowClient(model=model, enable_thinking=manifest["thinking"], rate_limit_retries=6))
               for model in manifest["models"]}
    if not all(client.client.is_configured for client in clients.values()):
        raise SystemExit("SiliconFlow key is missing")
    inputs = (DESIGN, QUALIFICATION, Path(__file__), ROOT / "src/toolsim/runners/llm_agent_runner.py",
              ROOT / "src/toolsim/runners/llm_core_local.py", ROOT / "src/toolsim/runners/llm_experiment2_dependency_curve.py")
    config = {"hashes": {path.name: sha256(path.read_bytes()).hexdigest() for path in inputs},
              "base_url": next(iter(clients.values())).client.base_url}
    output.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(output / "runs.sqlite")) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS results (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        db.execute("INSERT OR IGNORE INTO metadata VALUES (1,?)", (json.dumps(config, sort_keys=True),))
        if json.loads(db.execute("SELECT value FROM metadata WHERE id=1").fetchone()[0]) != config:
            raise ValueError("Checkpoint config differs from frozen inputs")
        db.commit()
        completed = {tuple(json.loads(key)) for (key,) in db.execute("SELECT key FROM results")}
        if not completed <= matrix.keys():
            raise ValueError("Unexpected cells in checkpoint")
        if not (output / "model_smoke.json").exists():
            smoke(clients, output)
        count = len(completed)
        print(f"Exp4: {count}/{len(matrix)} committed", flush=True)
        for key, spec in matrix.items():
            if key in completed:
                continue
            for attempt in range(3):
                try:
                    result = execute(key, spec, clients[key[1]])
                    break
                except BackendTransportError as exc:
                    if attempt == 2:
                        raise
                    print(f"MCP transport failure ({exc}); retrying episode", flush=True)
                    time.sleep(10 * 2 ** attempt)
            db.execute("INSERT INTO results VALUES (?,?)", (json.dumps(key), json.dumps(result, ensure_ascii=False)))
            db.commit()
            count += 1
            print(f"Saved {count}/{len(matrix)} {' | '.join(key)}", flush=True)
        report(db, matrix, output)
        print("Exp4 complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    manifest, matrix = design()
    if args.preflight:
        print(f"Frozen matrix verified: {len(matrix)} episodes")
    else:
        run(manifest, matrix, args.output)
