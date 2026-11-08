"""Run all Core Local oracle trajectories across mock/sandbox/local_mcp."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from toolsim.runners.core_local_dataset import load_core_local_cases  # noqa: E402
from toolsim.runners.core_local_oracle import run_oracle_case  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", default="mock,sandbox,local_mcp")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()

    backends = [item.strip() for item in args.backends.split(",") if item.strip()]
    cases = load_core_local_cases()
    if args.limit is not None:
        cases = cases[: args.limit]
    output_dir = ROOT / "outputs" / "core_local"
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_path = output_dir / f"experiment4_core_local_oracle_traces{args.output_suffix}.jsonl"
    summary_path = output_dir / f"experiment4_core_local_oracle_summary{args.output_suffix}.json"
    report_path = output_dir / f"experiment4_core_local_oracle_report{args.output_suffix}.md"

    results = []
    total = len(cases) * len(backends)
    index = 0
    with traces_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            for backend in backends:
                index += 1
                print(f"[{index}/{total}] {case.case_id}::{backend}", flush=True)
                result = run_oracle_case(case, backend).to_dict()
                results.append(result)
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()

    grouped = []
    for backend in backends:
        group = [item for item in results if item["backend"] == backend]
        grouped.append({
            "backend": backend,
            "runs": len(group),
            "correct": sum(item["final_correct"] for item in group),
            "dependency_compliant": sum(item["dependency_compliant"] for item in group),
            "failed_calls": sum(item["failed_calls"] for item in group),
        })
    summary = {
        "case_count": len(cases),
        "run_count": len(results),
        "backends": backends,
        "backend_metrics": grouped,
        "failure_counts": dict(Counter(item["backend"] for item in results if not item["final_correct"])),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(_markdown(summary, results), encoding="utf-8")
    print(f"Wrote {traces_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")
    if any(not item["final_correct"] or not item["dependency_compliant"] for item in results):
        raise SystemExit(1)


def _markdown(summary: dict, results: list[dict]) -> str:
    lines = [
        "# Experiment 4 Core Local Oracle Qualification",
        "",
        f"- Cases: {summary['case_count']}",
        f"- Runs: {summary['run_count']}",
        "",
        "| Backend | Runs | Correct | Dependency compliant | Failed calls |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in summary["backend_metrics"]:
        lines.append(f"| {item['backend']} | {item['runs']} | {item['correct']} | {item['dependency_compliant']} | {item['failed_calls']} |")
    failures = [item for item in results if not item["final_correct"] or not item["dependency_compliant"]]
    lines.extend(["", "## Failures", ""])
    if not failures:
        lines.append("None.")
    else:
        for item in failures:
            lines.append(f"- `{item['case_id']}::{item['backend']}` correct={item['final_correct']} dep={item['dependency_compliant']}")
            for detail in item["goal_details"]:
                if not detail["passed"]:
                    lines.append(f"  - {detail['message']}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

