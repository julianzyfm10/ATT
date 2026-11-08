"""DeepSeek API client for OpenAI-compatible chat completions.

DeepSeek's public API is OpenAI-compatible. The default values here follow the
current DeepSeek docs:

- ``DEEPSEEK_BASE_URL`` defaults to ``https://api.deepseek.com``.
- ``DEEPSEEK_MODEL`` defaults to ``deepseek-v4-flash``.
- ``DEEPSEEK_API_KEY`` must be provided before making real calls.
"""

from __future__ import annotations

import os
import time
from typing import Any

from toolsim.llm.types import LLMMessage, LLMResponse, LLMUsage


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"


class DeepSeekClient:
    """Small wrapper around the OpenAI SDK configured for DeepSeek."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL)
        self.model = model or os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = None

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set.")

        client = self._get_client()
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [message.to_dict() for message in messages],
            "stream": False,
        }
        if response_format is not None:
            request["response_format"] = response_format
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        if temperature is not None:
            request["temperature"] = temperature

        started = time.perf_counter()
        response = client.chat.completions.create(**request)
        latency_ms = (time.perf_counter() - started) * 1000.0
        message = response.choices[0].message
        usage = getattr(response, "usage", None)
        return LLMResponse(
            content=message.content or "",
            model=getattr(response, "model", self.model) or self.model,
            latency_ms=latency_ms,
            usage=LLMUsage(
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
            ),
            raw=_response_to_dict(response),
        )

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The openai package is required for DeepSeekClient.") from exc
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        return self._client


def _response_to_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    return {"repr": repr(response)}
