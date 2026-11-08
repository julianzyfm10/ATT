"""Run the 30-case Core Local benchmark with DeepSeek across three backends."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from toolsim.llm.deepseek_client import DEFAULT_DEEPSEEK_MODEL, DeepSeekClient  # noqa: E402
from toolsim.runners.core_local_dataset import load_core_local_cases  # noqa: E402
from toolsim.runners.llm_agent_runner import LLMAgentConfig  # noqa: E402
from toolsim.runners.llm_core_local import (  # noqa: E402
    CORE_LOCAL_BACKENDS,
    CORE_LOCAL_STRATEGIES,
    render_llm_core_local_markdown,
    result_key,
    run_llm_core_local_case,
    summarize_llm_core_local,
)


def main() -> None:
    args = _parse_args()
    strategies = _csv(args.strategies, CORE_LOCAL_STRATEGIES)
    backends = _csv(args.backends, CORE_LOCAL_BACKENDS)
    _validate_choices(strategies, backends)
    cases = load_core_local_cases(args.dataset)
    selected_ids = set(_csv(args.case_ids, []))
    if selected_ids:
        cases = [case for case in cases if case.case_id in selected_ids]
    if args.limit is not None:
        cases = cases[: max(0, args.limit)]

    expected_runs = len(cases) * len(strategies) * len(backends)
    out = ROOT / "outputs" / "core_local"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"experiment4_core_local_llm{args.output_suffix}"
    traces_path = out / f"{stem}_traces.jsonl"
    summary_path = out / f"{stem}_summary.json"
    report_path = out / f"{stem}_report.md"
    plan = {
        "dataset": str(Path(args.dataset).resolve()) if args.dataset else "default_core_local_30",
        "cases": len(cases),
        "strategies": strategies,
        "backends": backends,
        "expected_runs": expected_runs,
        "offline_tools": True,
        "llm_api_required": not args.validate_only and not args.summarize_only,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.validate_only:
        if len({case.case_id for case in cases}) != len(cases):
            raise SystemExit("Duplicate case_id detected")
        print("Core Local full-run plan validated.")
        return

    completed = _load_results(traces_path) if args.resume or args.summarize_only else []
    if args.summarize_only:
        if not completed:
            raise SystemExit(f"No completed traces found: {traces_path}")
        _write_outputs(completed, strategies, backends, args.model, expected_runs, summary_path, report_path)
        return
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY is not set. Use --validate-only to validate without API calls.")
    if not args.resume:
        traces_path.write_text("", encoding="utf-8")
        completed = []

    client = DeepSeekClient(
        model=args.model,
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    config = LLMAgentConfig(
        max_steps=args.max_steps,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    completed_keys = {result_key(item) for item in completed}
    index = 0
    for case in cases:
        for strategy in strategies:
            for backend in backends:
                index += 1
                key = f"{case.case_id}::{strategy}::{backend}"
                if key in completed_keys:
                    print(f"[{index}/{expected_runs}] skip {key}", flush=True)
                    continue
                print(f"[{index}/{expected_runs}] run {key}", flush=True)
                item = run_llm_core_local_case(client, case, strategy, backend, config=config)
                with traces_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                    handle.flush()
                completed.append(item)
                completed_keys.add(key)
                _write_outputs(
                    completed, strategies, backends, client.model, expected_runs, summary_path, report_path
                )

    print(f"Wrote {traces_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")


def _write_outputs(
    completed: list[dict[str, Any]],
    strategies: list[str],
    backends: list[str],
    model: str,
    expected_runs: int,
    summary_path: Path,
    report_path: Path,
) -> None:
    summary = summarize_llm_core_local(
        completed, strategies, backends, model=model, expected_runs=expected_runs
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_llm_core_local_markdown(summary), encoding="utf-8")


def _load_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    found: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid checkpoint JSON at line {line_number}: {exc}") from exc
        found.setdefault(result_key(item), item)
    return list(found.values())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset")
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL))
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--strategies", default=",".join(CORE_LOCAL_STRATEGIES))
    parser.add_argument("--backends", default=",".join(CORE_LOCAL_BACKENDS))
    parser.add_argument("--case-ids", default="")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-suffix", default="_full")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _csv(value: str, default: list[str]) -> list[str]:
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    return parsed or list(default)


def _validate_choices(strategies: list[str], backends: list[str]) -> None:
    invalid_strategies = sorted(set(strategies) - set(CORE_LOCAL_STRATEGIES))
    invalid_backends = sorted(set(backends) - set(CORE_LOCAL_BACKENDS))
    if invalid_strategies or invalid_backends:
        raise SystemExit(f"Invalid strategies={invalid_strategies}, backends={invalid_backends}")


if __name__ == "__main__":
    main()
