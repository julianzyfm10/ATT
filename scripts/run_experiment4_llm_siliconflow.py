"""Run Experiment 4 Core Local full matrix via SiliconFlow."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

import run_experiment4_llm_deepseek as experiment  # noqa: E402
from siliconflow_support import (  # noqa: E402
    append_default_argument,
    configure_legacy_runner,
    default_full_suffix,
)


def main() -> None:
    model = configure_legacy_runner(experiment, redirect_outputs=False)
    append_default_argument("--output-suffix", default_full_suffix(model, "full"))
    experiment.__doc__ = __doc__
    experiment.main()


if __name__ == "__main__":
    main()