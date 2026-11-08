"""Run calibrated Experiment 3 with independent faulted attempts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from output_paths import CASES, METRICS, REPORTS, ensure_output_dirs  # noqa: E402
from toolsim.llm.deepseek_client import DEFAULT_DEEPSEEK_MODEL, DeepSeekClient  # noqa: E402
from toolsim.runners.llm_agent_runner import LLMAgentConfig  # noqa: E402
from toolsim.runners.llm_experiment3_noise_robustness import (  # noqa: E402
    LLMExperiment3NoiseRobustnessRunner,
    STRATEGIES,
    compute_experiment3_group_metrics_from_dict,
    load_experiment3_cases,
    render_experiment3_markdown,
)

DEFAULT_CASES = ROOT / "data" / "benchmark" / "experiment3_toolsandbox_original_100_noise.jsonl"
DEFAULT_SUFFIX = "_toolsandbox_original_100_noise_single_attempt"


def main() -> None:
    args = _parse_args()
    ensure_output_dirs()
    suffix = args.output_suffix
    cases_output = CASES / f"experiment3_llm_deepseek_cases{suffix}.jsonl"
    traces_path = CASES / f"experiment3_llm_deepseek_traces{suffix}.jsonl"
    metrics_path = METRICS / f"experiment3_llm_deepseek_summary{suffix}.json"
    report_path = REPORTS / f"experiment3_llm_deepseek_report{suffix}.md"

    if args.summarize_only:
        completed = _load_completed(traces_path)
        if not completed:
            raise SystemExit(f"No traces found at {traces_path}")
        _write_checkpoint(args.model, args.attempts, completed, metrics_path, report_path)
        return
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY is not set")

    raw_rows = _read_jsonl(Path(args.cases_file))
    _validate_rows(raw_rows)
    cases = load_experiment3_cases(args.cases_file)
    if args.pilot_per_noise is not None:
        cases = _diverse_pilot(cases, args.pilot_per_noise)
    strategies = _parse_strategies(args.strategies)
    selected = {case.case_name for case in cases}
    with cases_output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in raw_rows:
            if row["case_name"] in selected:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    client = DeepSeekClient(
        model=args.model,
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    runner = LLMExperiment3NoiseRobustnessRunner(
        client,
        config=LLMAgentConfig(
            max_steps=args.max_step_cap,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        ),
        attempts_per_case=args.attempts,
    )
    completed = _load_completed(traces_path) if args.resume else []
    completed_keys = {_key(row) for row in completed}
    if not args.resume:
        traces_path.write_text("", encoding="utf-8")
    if completed:
        print(f"Resuming from {len(completed)} completed case-strategy group(s).", flush=True)

    total_groups = len(cases) * len(strategies)
    total_trajectories = total_groups * args.attempts
    print(
        f"Experiment 3: {len(cases)} cases x {len(strategies)} strategies x {args.attempts} attempts "
        f"= {total_trajectories} faulted trajectories; clean=disabled",
        flush=True,
    )
    index = 0
    for case in cases:
        for strategy in strategies:
            index += 1
            key = f"{case.case_name}::{strategy}"
            if key in completed_keys:
                print(f"[{index}/{total_groups}] Skipping completed {key}", flush=True)
                continue
            print(f"[{index}/{total_groups}] Running {key} ({args.attempts} attempts)", flush=True)
            item = runner.run_case(case, strategy).to_dict()
            with traces_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                handle.flush()
            completed.append(item)
            completed_keys.add(key)
            _write_checkpoint(client.model, args.attempts, completed, metrics_path, report_path)

    print(f"Wrote {cases_output}")
    print(f"Wrote {traces_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {report_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-file", default=str(DEFAULT_CASES))
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL))
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-step-cap", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--pilot-per-noise", type=int, default=None)
    parser.add_argument("--strategies", default=",".join(STRATEGIES))
    parser.add_argument("--output-suffix", default=DEFAULT_SUFFIX)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def _diverse_pilot(cases: list[Any], per_noise: int) -> list[Any]:
    selected: list[Any] = []
    for noise_type in sorted({case.noise_type for case in cases}):
        available = [case for case in cases if case.noise_type == noise_type]
        chosen: list[Any] = []
        canonical: Counter[str] = Counter()
        domains: Counter[str] = Counter()
        levels: Counter[str] = Counter()
        while available and len(chosen) < per_noise:
            candidate = min(
                available,
                key=lambda case: (
                    canonical[case.canonical_task_key],
                    domains[case.domain],
                    levels[case.difficulty_level],
                    cases.index(case),
                ),
            )
            available.remove(candidate)
            chosen.append(candidate)
            canonical[candidate.canonical_task_key] += 1
            domains[candidate.domain] += 1
            levels[candidate.difficulty_level] += 1
        if len(chosen) != per_noise:
            raise SystemExit(f"Could not select {per_noise} diverse {noise_type} pilot cases")
        selected.extend(chosen)
    return selected


def _validate_rows(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 100:
        raise SystemExit(f"Expected 100 cases, found {len(rows)}")
    if any(row.get("official_is_augmented_variant") for row in rows):
        raise SystemExit("Dataset contains augmented scenarios")
    if any("Target goals:" in str(row.get("task_prompt") or "") for row in rows):
        raise SystemExit("Dataset prompt leaks full evaluator goals")
    if any(row.get("source") != "toolsandbox_original_100" for row in rows):
        raise SystemExit("Dataset contains non-ToolSandbox data")


def _parse_strategies(value: str) -> list[str]:
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(STRATEGIES))
    if not selected or unknown:
        raise SystemExit(f"Invalid strategies: {unknown or selected}")
    return selected


def _load_completed(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = _key(row)
        if key and key not in seen:
            output.append(row)
            seen.add(key)
    return output


def _key(row: dict[str, Any]) -> str:
    return f"{row.get('case_name')}::{row.get('strategy')}" if row.get("case_name") and row.get("strategy") else ""


def _write_checkpoint(
    model: str,
    attempts: int,
    completed: list[dict[str, Any]],
    metrics_path: Path,
    report_path: Path,
) -> None:
    result = {
        "model": model,
        "attempts_per_case": attempts,
        "clean_branch_enabled": False,
        "case_results": completed,
        "group_metrics": compute_experiment3_group_metrics_from_dict(completed),
    }
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(render_experiment3_markdown(result), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
