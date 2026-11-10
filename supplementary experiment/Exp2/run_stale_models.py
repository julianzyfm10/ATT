"""Resume the 640-run paired stale-state model experiment."""

import argparse
from contextlib import closing
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys
import time

from run_stale_reachability import (CASES, HERE, ROOT, STRATEGY_EXTRA_STEPS, describe, execute_condition,
                                    load_experiment3_cases, scripted_actions, tally)

sys.path.insert(0, str(ROOT / "scripts"))
from openai import APIConnectionError, InternalServerError
from siliconflow_support import load_local_environment
from toolsim.llm.siliconflow_client import SiliconFlowClient

MODELS = ("Qwen/Qwen3.6-35B-A3B", "google/gemma-4-31B-it")
CONDITIONS = ("A", "B", "C", "D")
MAX_TOKENS = 512
REQUEST_INTERVAL = 1.5


class PacedClient:
    def __init__(self, client):
        self.client = client
        self.model = client.model
        self.next_request = 0.0

    def complete(self, messages, **kwargs):
        for attempt in range(4):
            time.sleep(max(0, self.next_request - time.monotonic()))
            self.next_request = time.monotonic() + REQUEST_INTERVAL
            try:
                return self.client.complete(messages, **kwargs)
            except (APIConnectionError, InternalServerError) as exc:
                if attempt == 3:
                    raise
                delay = 10 * 2 ** attempt
                print(f"{type(exc).__name__}; retrying decision in {delay}s", flush=True)
                time.sleep(delay)


def setup():
    cases = [case for case in load_experiment3_cases(CASES) if case.noise_type == "stale_state"]
    if len(cases) != 20:
        raise ValueError(f"Expected 20 stale-state cases, found {len(cases)}")
    budgets = {case.case_name: len(scripted_actions(case)) + 1 for case in cases}
    keys = [(model, case.case_name, strategy, condition) for case in cases for strategy in STRATEGY_EXTRA_STEPS
            for model in MODELS for condition in CONDITIONS]
    assert len(keys) == len(set(keys)) == 640
    manifest = {"cases_sha256": sha256(CASES.read_bytes()).hexdigest(), "models": list(MODELS),
                "strategies": list(STRATEGY_EXTRA_STEPS), "conditions": list(CONDITIONS), "budgets": budgets,
                "temperature": 0, "max_tokens": MAX_TOKENS, "thinking": False,
                "instrumentation_sha256": sha256((HERE / "run_stale_reachability.py").read_bytes()).hexdigest()}
    return {case.case_name: case for case in cases}, keys, manifest


def summarize(db, keys, manifest, output):
    data = {tuple(json.loads(key)): json.loads(value) for key, value in db.execute("SELECT key, value FROM results")}
    if data.keys() != set(keys):
        raise ValueError(f"Incomplete experiment: {len(data)}/{len(keys)}")
    summary = {"design": manifest, "n": len(data), "conditions": {}, "by_model_strategy": [], "paired": {}}
    for condition in CONDITIONS:
        summary["conditions"][condition] = tally([row for key, row in data.items() if key[3] == condition])
    for model in MODELS:
        for strategy in STRATEGY_EXTRA_STEPS:
            for condition in CONDITIONS:
                rows = [row for key, row in data.items() if (key[0], key[2], key[3]) == (model, strategy, condition)]
                summary["by_model_strategy"].append({"model": model, "strategy": strategy, "condition": condition,
                                                     **tally(rows)})
    for left, right in (("A", "B"), ("C", "D"), ("A", "C"), ("B", "D")):
        pairs = [(row, data[(*key[:3], right)]) for key, row in data.items() if key[3] == left]
        both_faulted = [(before, after) for before, after in pairs if before["fault_count"] and after["fault_count"]]
        summary["paired"][f"{left}_to_{right}"] = {
            "n": len(pairs), "both_triggered_n": len(both_faulted),
            "both_triggered_recovery": [sum(pair[index]["recovery"] for pair in both_faulted) for index in (0, 1)],
            "changes": {flag: {"gain": sum(not a[flag] and bool(b[flag]) for a, b in pairs),
                               "loss": sum(bool(a[flag]) and not b[flag] for a, b in pairs)}
                        for flag in ("state_correct", "observation_correct", "joint_correct", "fault_count",
                                     "decision_after_fault", "recovery")}}
    (output / "model_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Exp2 Stale-state 模型实验", "", "| 条件 | n | 终态正确 | 观察正确 | 联合正确 | 触发 | 有后续决策 | 显式重试 | Recovery |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for condition, row in summary["conditions"].items():
        recovery = f"{row['recovery']}/{row['faulted']}" if row["faulted"] else "N/A"
        values = [condition, *(str(row[key]) for key in ("n", "state_correct", "observation_correct", "joint_correct",
                   "faulted", "decision_after_fault", "explicit_retry")), recovery]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "分模型、策略、停止原因和配对变化见 `model_summary.json`。"]
    (output / "model_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output):
    cases, keys, manifest = setup()
    load_local_environment()
    clients = {model: PacedClient(SiliconFlowClient(model=model, enable_thinking=False, rate_limit_retries=6))
               for model in MODELS}
    if not all(client.client.is_configured for client in clients.values()):
        raise SystemExit("SILICONFLOW_API_KEY is missing")
    manifest["base_url"] = clients[MODELS[0]].client.base_url
    output.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(output / "model_runs.sqlite")) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS results (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        db.execute("INSERT OR IGNORE INTO metadata VALUES (1, ?)", (json.dumps(manifest, sort_keys=True),))
        if json.loads(db.execute("SELECT value FROM metadata WHERE id=1").fetchone()[0]) != manifest:
            raise ValueError("Checkpoint uses a different experiment configuration")
        db.commit()
        completed = {tuple(json.loads(key)) for (key,) in db.execute("SELECT key FROM results")}
        if not completed <= set(keys):
            raise ValueError("Checkpoint contains unexpected cells")
        print(f"Exp2 model run: {len(completed)}/{len(keys)} saved", flush=True)
        count = len(completed)
        for key in keys:
            if key in completed:
                continue
            model, case_name, strategy, condition = key
            case = cases[case_name]
            config, result = execute_condition(case, strategy, manifest["budgets"][case_name], condition,
                                               clients[model], temperature=0, max_tokens=MAX_TOKENS)
            row = describe(case, strategy, condition, "equal_decision_budget", config, result)
            row.update(model=model, tokens=result.total_tokens, parse_errors=result.parse_error_count)
            db.execute("INSERT INTO results VALUES (?, ?)", (json.dumps(key), json.dumps(row, ensure_ascii=False)))
            db.commit()
            count += 1
            print(f"Saved {count}/{len(keys)} {model} {strategy} {condition} {case.scenario_name}", flush=True)
        summarize(db, keys, manifest, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "model_results")
    args = parser.parse_args()
    if args.preflight:
        _, keys, manifest = setup()
        print(f"{len(keys)} paired cells; budgets {min(manifest['budgets'].values())}–{max(manifest['budgets'].values())}")
    else:
        run(args.output)
