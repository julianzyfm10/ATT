"""Run Experiment 2 dependency curves with a DeepSeek LLM policy.

This script requires ``DEEPSEEK_API_KEY`` for real API execution. Use
``--limit`` for a local check before running all 100 fixed ToolSandbox cases.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from output_paths import CASES, METRICS, REPORTS, ensure_output_dirs  # noqa: E402
from toolsim.runners.experiment2_cases import DEFAULT_CASES, load_frozen_cases  # noqa: E402
from toolsim.llm.deepseek_client import DEFAULT_DEEPSEEK_MODEL, DeepSeekClient  # noqa: E402
from toolsim.runners.dependency_curve import (  # noqa: E402
    DEFAULT_STRATEGIES,
    L1_SINGLE_TOOL,
    L2_EXPLICIT_DEP,
    L3_IMPLICIT_SIDE_EFFECT,
)
from toolsim.runners.llm_agent_runner import LLMAgentConfig  # noqa: E402
from toolsim.runners.llm_experiment2_dependency_curve import (  # noqa: E402
    LLMExperiment2DependencyCurveRunner,
    render_llm_experiment2_markdown_from_dict,
)


def main() -> None:
    args = _parse_args()
    ensure_output_dirs()

    suffix = args.output_suffix
    cases_path = CASES / f"experiment2_llm_deepseek_cases{suffix}.jsonl"
    metrics_path = METRICS / f"experiment2_llm_deepseek_summary{suffix}.json"
    report_path = REPORTS / f"experiment2_llm_deepseek_report{suffix}.md"
    traces_path = CASES / f"experiment2_llm_deepseek_traces{suffix}.jsonl"

    if args.summarize_only:
        completed = _load_completed_case_results(traces_path)
        if not completed:
            raise SystemExit(f"No completed traces found at {traces_path}")
        _write_checkpoint_outputs(
            model=_load_existing_model(metrics_path, default=args.model),
            completed=completed,
            metrics_path=metrics_path,
            report_path=report_path,
        )
        print(f"Wrote {metrics_path}")
        print(f"Wrote {report_path}")
        return

    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY is not set. Set it first, then rerun this script.")

    client = DeepSeekClient(
        model=args.model,
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    cases = load_frozen_cases(Path(args.cases_file))
    if args.limit is not None:
        cases = cases[:max(0, args.limit)]
    config = LLMAgentConfig(
        max_steps=args.max_steps,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    strategies = _parse_strategies(args.strategies)

    with cases_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps({
                "case_name": case.case_name,
                "difficulty_level": case.difficulty_level,
                "dependency_type": case.dependency_type,
                "description": case.description,
                "goals": case.goals,
                "required_edges": [edge.to_dict() for edge in case.required_edges],
                "initial_state": case.initial_state.to_dict(),
                "strategies": strategies,
            }, ensure_ascii=False) + "\n")

    completed = _load_completed_case_results(traces_path) if args.resume else []
    completed_keys = {_result_key(item) for item in completed}
    if not args.resume:
        traces_path.write_text("", encoding="utf-8")

    runner = LLMExperiment2DependencyCurveRunner(client, config=config)
    total_runs = len(cases) * len(strategies)
    if completed:
        print(f"Resuming from {len(completed)} completed run(s).", flush=True)
        _write_checkpoint_outputs(
            model=getattr(client, "model", "unknown"),
            completed=completed,
            metrics_path=metrics_path,
            report_path=report_path,
        )

    run_index = 0
    for case in cases:
        for strategy in strategies:
            run_index += 1
            key = f"{case.case_name}::{strategy}"
            if key in completed_keys:
                print(f"[{run_index}/{total_runs}] Skipping completed {key}", flush=True)
                continue
            print(f"[{run_index}/{total_runs}] Running {key}", flush=True)
            single_result = runner.run([case], strategies=[strategy])
            item = single_result.case_results[0].to_dict()
            with traces_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                handle.flush()
            completed.append(item)
            completed_keys.add(key)
            _write_checkpoint_outputs(
                model=getattr(client, "model", "unknown"),
                completed=completed,
                metrics_path=metrics_path,
                report_path=report_path,
            )

    print(f"Wrote {cases_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {traces_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL))
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional output token cap. Omit by default so the provider is not artificially capped.",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional case limit for a local check.")
    parser.add_argument("--cases-file", default=str(DEFAULT_CASES))
    parser.add_argument(
        "--strategies",
        default=",".join(DEFAULT_STRATEGIES),
        help="Comma-separated reasoning strategies to run, e.g. direct,cot,react,self_refine.",
    )
    parser.add_argument(
        "--output-suffix",
        default="_toolsandbox_original_100",
        help="Suffix for local output files.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from an existing traces JSONL file for the same output suffix.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Regenerate summary/report from an existing traces JSONL file without calling the LLM.",
    )
    return parser.parse_args()


def _load_completed_case_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    completed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        key = _result_key(item)
        if not key or key in seen:
            continue
        completed.append(item)
        seen.add(key)
    return completed


def _parse_strategies(value: str) -> list[str]:
    strategies = [item.strip() for item in value.split(",") if item.strip()]
    if not strategies:
        raise SystemExit("--strategies must contain at least one strategy")
    return strategies


def _result_key(item: dict[str, Any]) -> str:
    case_name = item.get("case_name")
    strategy = item.get("strategy", "direct")
    return f"{case_name}::{strategy}" if case_name else ""


def _load_existing_model(path: Path, *, default: str) -> str:
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
    return str(data.get("model") or default)


def _write_checkpoint_outputs(
    *,
    model: str,
    completed: list[dict[str, Any]],
    metrics_path: Path,
    report_path: Path,
) -> None:
    group_metrics = _compute_group_metrics(completed)
    result = {
        "model": model,
        "case_results": completed,
        "group_metrics": group_metrics,
        "difficulty_summaries": _compute_difficulty_summaries(group_metrics),
    }
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(render_llm_experiment2_markdown_from_dict(result), encoding="utf-8")


def _compute_group_metrics(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    sources = ["all", *sorted({result["source"] for result in results})]
    strategies = sorted({result.get("strategy", "direct") for result in results})
    for source in sources:
        source_results = results if source == "all" else [result for result in results if result["source"] == source]
        for strategy in strategies:
            strategy_results = [result for result in source_results if result.get("strategy", "direct") == strategy]
            for level in [L1_SINGLE_TOOL, L2_EXPLICIT_DEP, L3_IMPLICIT_SIDE_EFFECT]:
                group = [result for result in strategy_results if result["difficulty_level"] == level]
                metrics.append(_summarize_group(source, strategy, level, group))
    return metrics


def _summarize_group(source: str, strategy: str, difficulty_level: str, group: list[dict[str, Any]]) -> dict[str, Any]:
    total_cases = len(group)
    total_calls = sum(item["run"]["experiment_result"]["call_metrics"]["total_calls"] for item in group)
    successful_calls = sum(item["run"]["experiment_result"]["call_metrics"]["successful_calls"] for item in group)
    invalid_calls = sum(item["run"]["experiment_result"]["call_metrics"]["invalid_calls"] for item in group)
    recovery_opportunities = sum(
        1 for item in group
        if item["run"]["experiment_result"]["call_metrics"]["failed_calls"] > 0
    )
    recovered = sum(
        1 for item in group
        if item["run"]["experiment_result"]["call_metrics"]["failed_calls"] > 0 and item["final_state_correct"]
    )
    total_llm_steps = sum(len(item["run"]["llm_steps"]) for item in group)
    total_parse_errors = sum(item["run"]["parse_error_count"] for item in group)
    total_length_stops = sum(
        1
        for item in group
        for step in item["run"]["llm_steps"]
        if _is_length_truncated_step(step)
    )
    side_effect_cases = [item for item in group if item.get("side_effect_dependency_success") is not None]

    return {
        "source": source,
        "strategy": strategy,
        "difficulty_level": difficulty_level,
        "total_cases": total_cases,
        "success_count": sum(1 for item in group if item["final_state_correct"]),
        "final_state_correctness": _rate(sum(1 for item in group if item["final_state_correct"]), total_cases),
        "dependency_correctness_rate": _rate(sum(1 for item in group if _dependency_correct(item)), total_cases),
        "success_rate": _rate(successful_calls, total_calls),
        "invalid_call_rate": _rate(invalid_calls, total_calls),
        "recovery_rate": _optional_rate(recovered, recovery_opportunities),
        "recovery_opportunities": recovery_opportunities,
        "recovered_cases": recovered,
        "average_trajectory_length": _rate(total_calls, total_cases),
        "dependency_completion_rate": _rate(sum(item["dependency_completion_rate"] for item in group), total_cases),
        "wrong_order_rate": _rate(sum(1 for item in group if item["wrong_order_call_count"] > 0), total_cases),
        "missing_prerequisite_rate": _rate(sum(1 for item in group if item["missing_prerequisite_count"] > 0), total_cases),
        "side_effect_awareness_rate": (
            _rate(sum(1 for item in side_effect_cases if item["side_effect_dependency_success"]), len(side_effect_cases))
            if side_effect_cases
            else None
        ),
        "dependency_repair_rate": _rate(sum(1 for item in group if item["dependency_repair"]), total_cases),
        "parse_error_rate": _rate(total_parse_errors, total_llm_steps),
        "length_truncation_rate": _rate(total_length_stops, total_llm_steps),
        "average_llm_steps": _rate(total_llm_steps, total_cases),
        "average_tokens": _rate(sum(item["run"]["total_tokens"] for item in group), total_cases),
        "average_llm_latency_ms": _rate(sum(item["run"]["total_llm_latency_ms"] for item in group), total_cases),
    }


def _compute_difficulty_summaries(group_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    by_key = {(metric["source"], metric["strategy"], metric["difficulty_level"]): metric for metric in group_metrics}
    for source, strategy in sorted({(metric["source"], metric["strategy"]) for metric in group_metrics}):
        l1_metric = by_key[(source, strategy, L1_SINGLE_TOOL)]
        l2_metric = by_key[(source, strategy, L2_EXPLICIT_DEP)]
        l3_metric = by_key[(source, strategy, L3_IMPLICIT_SIDE_EFFECT)]
        l1_dep = l1_metric["dependency_correctness_rate"]
        l2_dep = l2_metric["dependency_correctness_rate"]
        l3_dep = l3_metric["dependency_correctness_rate"]
        l1_final = l1_metric["final_state_correctness"]
        l2_final = l2_metric["final_state_correctness"]
        l3_final = l3_metric["final_state_correctness"]
        summaries.append({
            "source": source,
            "strategy": strategy,
            "l1_dependency_correctness": l1_dep,
            "l2_dependency_correctness": l2_dep,
            "l3_dependency_correctness": l3_dep,
            "dependency_degradation_l1_to_l3": l1_dep - l3_dep,
            "auc_dependency_correctness": (l1_dep + l2_dep + l3_dep) / 3,
            "l1_final_state_correctness": l1_final,
            "l2_final_state_correctness": l2_final,
            "l3_final_state_correctness": l3_final,
            "degradation_l1_to_l3": l1_final - l3_final,
            "auc_final_state_correctness": (l1_final + l2_final + l3_final) / 3,
        })
    return summaries


def _dependency_correct(item: dict[str, Any]) -> bool:
    return bool(
        item.get("dependency_correct")
        or (
            item.get("final_state_correct")
            and item.get("dependency_completion_rate") == 1.0
            and item.get("wrong_order_call_count") == 0
            and item.get("missing_prerequisite_count") == 0
        )
    )


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


if __name__ == "__main__":
    main()
