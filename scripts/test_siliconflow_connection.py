"""Validate SiliconFlow credentials, model availability, and JSON responses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from siliconflow_support import (  # noqa: E402
    SUPPORTED_SILICONFLOW_MODELS,
    load_local_environment,
    resolve_model,
)
from toolsim.llm.siliconflow_client import SiliconFlowClient  # noqa: E402
from toolsim.llm.types import LLMMessage  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", help="Model ID or supported short alias; repeatable.")
    parser.add_argument("--all-models", action="store_true", help="Test all project benchmark models.")
    parser.add_argument("--list-only", action="store_true", help="Only verify that model IDs are available.")
    args = parser.parse_args()

    load_local_environment()
    import os

    api_key = os.environ.get("SILICONFLOW_API_KEY", "")
    if not api_key:
        raise SystemExit("SILICONFLOW_API_KEY is missing from .env")
    base_url = os.environ.get("SILICONFLOW_BASE_URL") or None
    requested = (
        list(SUPPORTED_SILICONFLOW_MODELS)
        if args.all_models
        else [resolve_model(item) for item in (args.model or [os.environ.get("SILICONFLOW_MODEL", "")])]
    )

    probe = SiliconFlowClient(api_key=api_key, base_url=base_url, timeout=30, max_retries=0)
    try:
        available = {item.id for item in probe._get_client().models.list().data}
    except Exception as exc:
        raise SystemExit(_safe_error("Model-list check failed", exc, api_key)) from None

    missing = [model for model in requested if model not in available]
    if missing:
        raise SystemExit(f"Unavailable SiliconFlow model(s): {', '.join(missing)}")
    if args.list_only:
        print(json.dumps({"ok": True, "available_models": requested}, ensure_ascii=False))
        return

    results = []
    for model in requested:
        client = SiliconFlowClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=90,
            max_retries=1,
        )
        try:
            response = client.complete(
                [LLMMessage(role="user", content='Reply only with {"status":"ok"}.')],
                response_format={"type": "json_object"},
                max_tokens=64,
                temperature=0.0,
            )
        except Exception as exc:
            raise SystemExit(_safe_error(f"SiliconFlow test failed for {model}", exc, api_key)) from None
        results.append({
            "ok": bool(response.content),
            "requested_model": model,
            "response_model": response.model,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "content_length": len(response.content),
        })
    print(json.dumps({"ok": all(item["ok"] for item in results), "results": results}, ensure_ascii=False))


def _safe_error(prefix: str, exc: Exception, api_key: str) -> str:
    return f"{prefix}: {type(exc).__name__}: {str(exc).replace(api_key, '[REDACTED]')}"


if __name__ == "__main__":
    main()