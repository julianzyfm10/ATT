"""Run Experiment 3 single-attempt protocol via SiliconFlow.

This matches the latest DeepSeek run: 100 frozen ToolSandbox cases, four
strategies, one faulted trajectory per case-strategy, and no clean branch.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

import run_experiment3_llm_deepseek as experiment  # noqa: E402
from siliconflow_support import (  # noqa: E402
    append_default_argument,
    configure_legacy_runner,
    default_full_suffix,
)

DEFAULT_CASES = ROOT / "data" / "benchmark" / "experiment3_toolsandbox_original_100_noise.jsonl"


def main() -> None:
    model = configure_legacy_runner(experiment)
    append_default_argument("--cases-file", str(DEFAULT_CASES))
    append_default_argument("--attempts", "1")
    thinking_disabled = os.environ.get("SILICONFLOW_ENABLE_THINKING", "").strip().lower() in {
        "0", "false", "no", "off",
    }
    label = "toolsandbox_original_100_noise_single_attempt"
    if thinking_disabled:
        label += "_no_thinking"
    append_default_argument("--output-suffix", default_full_suffix(model, label))
    print(
        f"SiliconFlow thinking: {'disabled' if thinking_disabled else 'provider default'}",
        flush=True,
    )
    experiment.__doc__ = __doc__
    experiment.main()


if __name__ == "__main__":
    main()
