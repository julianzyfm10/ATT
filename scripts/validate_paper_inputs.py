"""Check the fixed inputs and case counts without calling a model."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from toolsim.runners.experiment2_cases import DEFAULT_CASES, load_frozen_cases  # noqa: E402
from toolsim.runners.llm_experiment1 import build_llm_experiment1_cases  # noqa: E402
from toolsim.runners.llm_experiment3_noise_robustness import load_experiment3_cases  # noqa: E402
from toolsim.runners.core_local_dataset import load_core_local_cases  # noqa: E402


def main() -> None:
    experiment1 = build_llm_experiment1_cases()
    experiment2 = load_frozen_cases(DEFAULT_CASES)
    experiment3 = load_experiment3_cases(
        ROOT / "data" / "benchmark" / "experiment3_toolsandbox_original_100_noise.jsonl"
    )
    experiment4 = load_core_local_cases()
    counts = [len(experiment1), len(experiment2), len(experiment3), len(experiment4)]
    if counts != [100, 100, 100, 30]:
        raise SystemExit(f"Unexpected experiment case counts: {counts}")
    for number, cases in enumerate((experiment1, experiment2, experiment3, experiment4), start=1):
        names = [case.case_name if number < 4 else case.case_id for case in cases]
        if len(set(names)) != len(names):
            raise SystemExit(f"Experiment {number} has duplicate case identifiers")
    noise = Counter(case.noise_type for case in experiment3)
    if set(noise) != {"timeout", "schema_drift", "stale_state", "vague_observation", "misleading_observation"}:
        raise SystemExit(f"Unexpected Experiment 3 noise types: {noise}")
    print(json.dumps({"case_counts": counts, "experiment3_noise_counts": noise}, indent=2))


if __name__ == "__main__":
    main()
