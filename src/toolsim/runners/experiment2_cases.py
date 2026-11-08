"""Load the fixed 100 ToolSandbox cases used by Experiment 2."""

from __future__ import annotations

import json
from pathlib import Path
from toolsim.core.world_state import WorldState
from toolsim.runners.dependency_curve import DependencyCurveCase, DependencyEdge

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CASES = ROOT / "data" / "benchmark" / "experiment2_toolsandbox_original_100.jsonl"

def load_frozen_cases(path: Path) -> list[DependencyCurveCase]:
    if not path.exists():
        raise SystemExit(f"Frozen Experiment 2 dataset does not exist: {path}")
    cases: list[DependencyCurveCase] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        case_name = str(row.get("case_name") or "")
        if not case_name or case_name in seen:
            raise SystemExit(f"Invalid or duplicate case_name at {path}:{line_number}")
        seen.add(case_name)
        cases.append(DependencyCurveCase(
            case_name=case_name,
            difficulty_level=str(row["difficulty_level"]),
            dependency_type=str(row["dependency_type"]),
            description=str(row.get("description") or row.get("user_request") or ""),
            goals=list(row.get("goals") or []),
            oracle_tool_calls=list(row.get("oracle_tool_calls") or []),
            strategy_tool_calls=dict(row.get("strategy_tool_calls") or {}),
            initial_state=WorldState.from_dict(dict(row.get("initial_state") or {})),
            required_edges=[
                DependencyEdge(
                    before=str(edge["before"]),
                    after=str(edge["after"]),
                    description=str(edge.get("description") or ""),
                )
                for edge in row.get("required_edges") or []
            ],
        ))
    if len(cases) != 100 and path.resolve() == DEFAULT_CASES.resolve():
        raise SystemExit(f"Expected 100 frozen Experiment 2 cases, found {len(cases)}")
    print(f"Loaded {len(cases)} frozen Experiment 2 case(s) from {path}", flush=True)
    return cases
