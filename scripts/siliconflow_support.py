"""Shared SiliconFlow configuration for the experiment entry points."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

from output_paths import CASES, METRICS, REPORTS
from toolsim.llm.siliconflow_client import (
    DEFAULT_SILICONFLOW_BASE_URL,
    DEFAULT_SILICONFLOW_MODEL,
    SiliconFlowClient,
)

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_SILICONFLOW_MODELS = (
    "openai/gpt-oss-120b",
    "google/gemma-4-31B-it",
    "Qwen/Qwen3.6-35B-A3B",
)
MODEL_ALIASES = {
    "gpt-oss-120b": SUPPORTED_SILICONFLOW_MODELS[0],
    "gpt_oss_120b": SUPPORTED_SILICONFLOW_MODELS[0],
    "gemma-4-31b-it": SUPPORTED_SILICONFLOW_MODELS[1],
    "gemma_4_31b_it": SUPPORTED_SILICONFLOW_MODELS[1],
    "qwen3.6-35b-a3b": SUPPORTED_SILICONFLOW_MODELS[2],
    "qwen3_6_35b_a3b": SUPPORTED_SILICONFLOW_MODELS[2],
    "qwen/qwen3.6-35b-a3b": SUPPORTED_SILICONFLOW_MODELS[2],
}


class SiliconFlowOutputDir:
    """Redirect legacy DeepSeek filenames into a SiliconFlow namespace."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def __truediv__(self, filename: str) -> Path:
        return self.path / filename.replace("deepseek", "siliconflow")


def configure_legacy_runner(module: Any, *, redirect_outputs: bool = True) -> str:
    """Make a DeepSeek experiment module use SiliconFlow without changing its logic."""
    load_local_environment()
    model = resolve_model(option_value("--model") or os.environ.get("SILICONFLOW_MODEL"))
    base_url = os.environ.get("SILICONFLOW_BASE_URL", DEFAULT_SILICONFLOW_BASE_URL)
    append_default_argument("--model", model)
    append_default_argument("--base-url", base_url)

    api_key = os.environ.get("SILICONFLOW_API_KEY")
    needs_key = not any(
        flag_present(flag) for flag in ("--help", "-h", "--summarize-only", "--validate-only")
    )
    if needs_key and not api_key:
        raise SystemExit("SILICONFLOW_API_KEY is not set in .env or the process environment.")
    if api_key:
        # Legacy runners perform this presence check before constructing the patched client.
        os.environ["DEEPSEEK_API_KEY"] = api_key

    class ConfiguredSiliconFlowClient(SiliconFlowClient):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(api_key=os.environ.get("SILICONFLOW_API_KEY"), **kwargs)

    module.DeepSeekClient = ConfiguredSiliconFlowClient
    if redirect_outputs:
        module.CASES = SiliconFlowOutputDir(CASES)
        module.METRICS = SiliconFlowOutputDir(METRICS)
        module.REPORTS = SiliconFlowOutputDir(REPORTS)
    return model


def load_local_environment() -> dict[str, str]:
    """Load SiliconFlow settings, preferring real process variables over local files."""
    values: dict[str, str] = {}
    for path in (ROOT / ".env", ROOT / ".env.siliconflow"):
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key.startswith("SILICONFLOW_"):
                continue
            value = value.strip().strip('"').strip("'")
            values.setdefault(key, value)
            os.environ.setdefault(key, value)
    return values


def resolve_model(value: str | None) -> str:
    selected = (value or DEFAULT_SILICONFLOW_MODEL).strip()
    return MODEL_ALIASES.get(selected.lower(), selected)


def model_slug(model: str) -> str:
    tail = model.rsplit("/", 1)[-1].lower()
    return re.sub(r"[^a-z0-9]+", "_", tail).strip("_")


def default_full_suffix(model: str, label: str = "full") -> str:
    return f"_{model_slug(model)}_{label}"


def flag_present(flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv[1:])


def option_value(flag: str) -> str | None:
    for index, arg in enumerate(sys.argv[1:], start=1):
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
        if arg == flag and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
    return None


def pop_option(flag: str, default: str | None = None) -> str | None:
    for index, arg in enumerate(list(sys.argv)):
        if arg.startswith(f"{flag}="):
            del sys.argv[index]
            return arg.split("=", 1)[1]
        if arg == flag:
            if index + 1 >= len(sys.argv):
                raise SystemExit(f"{flag} requires a value")
            value = sys.argv[index + 1]
            del sys.argv[index:index + 2]
            return value
    return default


def append_default_argument(flag: str, value: str) -> None:
    if not flag_present(flag):
        sys.argv.extend([flag, value])