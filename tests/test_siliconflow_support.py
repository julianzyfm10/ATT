"""Tests for SiliconFlow model registration and output naming."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from siliconflow_support import (  # noqa: E402
    SUPPORTED_SILICONFLOW_MODELS,
    model_slug,
    resolve_model,
)


def test_qwen_model_is_registered() -> None:
    model = "Qwen/Qwen3.6-35B-A3B"
    assert model in SUPPORTED_SILICONFLOW_MODELS
    assert resolve_model("qwen3.6-35b-a3b") == model
    assert resolve_model("qwen3_6_35b_a3b") == model
    assert resolve_model(model) == model


def test_qwen_model_has_stable_output_slug() -> None:
    assert model_slug("Qwen/Qwen3.6-35B-A3B") == "qwen3_6_35b_a3b"