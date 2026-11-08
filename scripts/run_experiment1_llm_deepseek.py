"""Run Experiment 1 with a DeepSeek LLM policy.

This script requires ``DEEPSEEK_API_KEY`` for real API execution. Use
``--limit`` for a local check before running all 100 ToolSandbox cases.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from output_paths import CASES, METRICS, REPORTS, ensure_output_dirs  # noqa: E402
from toolsim.llm.deepseek_client import DEFAULT_DEEPSEEK_MODEL, DeepSeekClient  # noqa: E402
from toolsim.runners.llm_agent_runner import LLMAgentConfig  # noqa: E402
from toolsim.runners.llm_experiment1 import (  # noqa: E402
    LLMExperiment1Runner,
    build_llm_experiment1_cases,
)


def main() -> None:
    args = _parse_args()
    ensure_output_dirs()

    suffix = args.output_suffix
    cases_path = CASES / f"experiment1_llm_deepseek_cases{suffix}.jsonl"
    metrics_path = METRICS / f"experiment1_llm_deepseek_summary{suffix}.json"
    report_path = REPORTS / f"experiment1_llm_deepseek_report{suffix}.md"
    traces_path = CASES / f"experiment1_llm_deepseek_traces{suffix}.jsonl"

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
        raise SystemExit(
            "DEEPSEEK_API_KEY is not set. Set it first, then rerun this script."
        )

    client = DeepSeekClient(
        model=args.model,
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    cases = build_llm_experiment1_cases(
        cases_path=args.cases_file,
        limit=args.limit,
    )
    config = LLMAgentConfig(
        max_steps=args.max_steps,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    with cases_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps({
                "case_name": case.case_name,
                "description": case.description,
                "goals_stateful": case.goals_stateful,
                "goals_stateless": case.goals_stateless,
                "initial_state_stateful": case.initial_state_stateful.to_dict() if case.initial_state_stateful else None,
                "initial_state_stateless": case.initial_state_stateless.to_dict() if case.initial_state_stateless else None,
            }, ensure_ascii=False) + "\n")

    completed = _load_completed_case_results(traces_path) if args.resume else []
    completed_names = {item["case_name"] for item in completed}
    if not args.resume:
        traces_path.write_text("", encoding="utf-8")

    runner = LLMExperiment1Runner(client, config=config)
    total_cases = len(cases)
    if completed:
        print(f"Resuming from {len(completed)} completed case(s).", flush=True)
        _write_checkpoint_outputs(
            model=getattr(client, "model", "unknown"),
            completed=completed,
            metrics_path=metrics_path,
            report_path=report_path,
        )

    for index, case in enumerate(cases, start=1):
        if case.case_name in completed_names:
            print(f"[{index}/{total_cases}] Skipping completed {case.case_name}", flush=True)
            continue
        print(f"[{index}/{total_cases}] Running {case.case_name}", flush=True)
        single_result = runner.run([case])
        item = single_result.case_results[0].to_dict()
        with traces_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            handle.flush()
        completed.append(item)
        completed_names.add(case.case_name)
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
    parser.add_argument("--cases-file", default=str(ROOT / "data" / "benchmark" / "experiment2_toolsandbox_original_100.jsonl"))
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


def _load_completed_case_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    completed: list[dict] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        case_name = item.get("case_name")
        if not case_name or case_name in seen:
            continue
        completed.append(item)
        seen.add(case_name)
    return completed


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
    completed: list[dict],
    metrics_path: Path,
    report_path: Path,
) -> None:
    result = {
        "model": model,
        "case_results": completed,
        "metrics": [
            _compute_mode_metrics(completed, "stateful"),
            _compute_mode_metrics(completed, "stateless"),
        ],
    }
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(_render_markdown_report(result), encoding="utf-8")


def _compute_mode_metrics(results: list[dict], mode: str) -> dict:
    runs = [item[mode] for item in results]
    total = len(runs)
    total_calls = sum(run["experiment_result"]["call_metrics"]["total_calls"] for run in runs)
    invalid_calls = sum(run["experiment_result"]["call_metrics"]["invalid_calls"] for run in runs)
    final_correct_count = sum(1 for run in runs if _final_correct(run))
    successful_calls = sum(run["experiment_result"]["call_metrics"]["successful_calls"] for run in runs)
    recovery_opportunities = sum(
        1 for run in runs
        if run["experiment_result"]["call_metrics"]["failed_calls"] > 0
    )
    recovery_count = sum(
        1 for run in runs
        if run["experiment_result"]["call_metrics"]["failed_calls"] > 0 and _final_correct(run)
    )
    total_parse_errors = sum(run["parse_error_count"] for run in runs)
    total_llm_steps = sum(len(run["llm_steps"]) for run in runs)
    total_length_stops = sum(
        1
        for run in runs
        for step in run["llm_steps"]
        if _is_length_truncated_step(step)
    )
    return {
        "mode": mode,
        "total_cases": total,
        "success_rate": _rate(successful_calls, total_calls),
        "invalid_call_rate": _rate(invalid_calls, total_calls),
        "recovery_rate": _optional_rate(recovery_count, recovery_opportunities),
        "recovery_opportunities": recovery_opportunities,
        "recovered_cases": recovery_count,
        "average_trajectory_length": _rate(
            sum(len(run["experiment_result"]["trace"]) for run in runs),
            total,
        ),
        "average_llm_steps": _rate(total_llm_steps, total),
        "final_state_correctness": _rate(final_correct_count, total),
        "parse_error_rate": _rate(total_parse_errors, total_llm_steps),
        "length_truncation_rate": _rate(total_length_stops, total_llm_steps),
        "average_tokens": _rate(sum(run["total_tokens"] for run in runs), total),
        "average_llm_latency_ms": _rate(sum(run["total_llm_latency_ms"] for run in runs), total),
    }


def _render_markdown_report(result: dict) -> str:
    lines = [
        "# Experiment 1 LLM Stateful vs Stateless Report",
        "",
        f"- Model: `{result['model']}`",
        f"- Completed cases: {len(result['case_results'])}",
        "",
        "## Metrics",
        "",
        "| Mode | Cases | Success Rate | Invalid Call Rate | Recovery Rate | Avg Trajectory Length | Final State Correctness |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in result["metrics"]:
        lines.append(
            f"| {metric['mode']} | {metric['total_cases']} | {_pct(metric['success_rate'])} | "
            f"{_pct(metric['invalid_call_rate'])} | {_recovery_label(metric.get('recovery_rate'))} | "
            f"{metric['average_trajectory_length']:.2f} | {_pct(metric['final_state_correctness'])} |"
        )

    lines.extend([
        "",
        "## LLM Diagnostics",
        "",
        "| Mode | Avg LLM Turns | Parse Errors | Length Stops | Avg Tokens | Avg LLM Latency ms |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for metric in result["metrics"]:
        lines.append(
            f"| {metric['mode']} | {metric.get('average_llm_steps', 0.0):.2f} | "
            f"{_pct(metric['parse_error_rate'])} | {_pct(metric.get('length_truncation_rate', 0.0))} | "
            f"{metric['average_tokens']:.1f} | {metric['average_llm_latency_ms']:.2f} |"
        )

    lines.extend(["", "## Case Outcomes", ""])
    lines.append("| Case | Source | Stateful Correct | Stateless Correct | Stateful Traj Len | Stateless Traj Len | Stateful LLM Turns | Stateless LLM Turns |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for item in result["case_results"]:
        lines.append(
            f"| {item['case_name']} | {item['source']} | {_final_correct(item['stateful'])} | "
            f"{_final_correct(item['stateless'])} | {len(item['stateful']['experiment_result']['trace'])} | "
            f"{len(item['stateless']['experiment_result']['trace'])} | {len(item['stateful']['llm_steps'])} | "
            f"{len(item['stateless']['llm_steps'])} |"
        )
    return "\n".join(lines)


def _final_correct(run: dict) -> bool:
    metrics = run["experiment_result"].get("state_metrics")
    return bool(metrics and metrics.get("all_passed"))


def _is_length_truncated_step(step: dict) -> bool:
    response = step.get("llm_response") or {}
    raw = response.get("raw") or {}
    choices = raw.get("choices") or []
    if not choices:
        return False
    return choices[0].get("finish_reason") == "length"


def _rate(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _recovery_label(value: float | None) -> str:
    return "N/A" if value is None else _pct(value)


def _optional_rate(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


if __name__ == "__main__":
    main()
