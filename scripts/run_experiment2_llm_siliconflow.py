"""Run Experiment 2 on the frozen ToolSandbox original-100 set via SiliconFlow."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

import run_experiment2_llm_deepseek as experiment  # noqa: E402
from siliconflow_support import (  # noqa: E402
    append_default_argument,
    configure_legacy_runner,
    default_full_suffix,
)


def main() -> None:
    model = configure_legacy_runner(experiment)
    append_default_argument("--output-suffix", default_full_suffix(model, "toolsandbox_original_100"))
    experiment.__doc__ = __doc__
    experiment.main()


if __name__ == "__main__":
    main()
