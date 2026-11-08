"""SiliconFlow client for OpenAI-compatible chat completions."""

from __future__ import annotations

import os
import sys
import time
from typing import Any

from toolsim.llm.types import LLMMessage, LLMResponse, LLMUsage


DEFAULT_SILICONFLOW_BASE_URL = "https://api.siliconflow.com/v1"
DEFAULT_SILICONFLOW_MODEL = "openai/gpt-oss-120b"
DEFAULT_RATE_LIMIT_RETRIES = 20
DEFAULT_RATE_LIMIT_BACKOFF = 30.0


class SiliconFlowClient:
    """OpenAI SDK wrapper configured for long-running SiliconFlow experiments."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        rate_limit_retries: int | None = None,
        rate_limit_backoff: float | None = None,
        enable_thinking: bool | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("SILICONFLOW_API_KEY")
        self.base_url = base_url or os.environ.get(
            "SILICONFLOW_BASE_URL", DEFAULT_SILICONFLOW_BASE_URL
        )
        self.model = model or os.environ.get("SILICONFLOW_MODEL", DEFAULT_SILICONFLOW_MODEL)
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limit_retries = (
            int(os.environ.get("SILICONFLOW_RATE_LIMIT_RETRIES", DEFAULT_RATE_LIMIT_RETRIES))
            if rate_limit_retries is None
            else rate_limit_retries
        )
        self.rate_limit_backoff = (
            float(os.environ.get("SILICONFLOW_RATE_LIMIT_BACKOFF", DEFAULT_RATE_LIMIT_BACKOFF))
            if rate_limit_backoff is None
            else rate_limit_backoff
        )
        self.enable_thinking = _optional_bool_environment(
            "SILICONFLOW_ENABLE_THINKING", enable_thinking
        )
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
            raise RuntimeError("SILICONFLOW_API_KEY is not set.")

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
        if self.enable_thinking is not None:
            request["extra_body"] = {"enable_thinking": self.enable_thinking}

        started = time.perf_counter()
        response = self._create_with_rate_limit_retry(request)
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

    def _create_with_rate_limit_retry(self, request: dict[str, Any]) -> Any:
        attempts = 0
        while True:
            try:
                return self._get_client().chat.completions.create(**request)
            except Exception as exc:
                if not _is_rate_limit_error(exc) or attempts >= self.rate_limit_retries:
                    raise
                attempts += 1
                delay = _retry_delay(exc, self.rate_limit_backoff, attempts)
                print(
                    f"SiliconFlow rate limit reached; retrying in {delay:.0f}s "
                    f"({attempts}/{self.rate_limit_retries}).",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The openai package is required for SiliconFlowClient.") from exc
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        return self._client


def _optional_bool_environment(name: str, explicit: bool | None) -> bool | None:
    if explicit is not None:
        return explicit
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")

def _is_rate_limit_error(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 429 or type(exc).__name__ == "RateLimitError"


def _retry_delay(exc: Exception, base: float, attempt: int) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        value = headers.get("retry-after") or headers.get("Retry-After")
        try:
            if value is not None:
                return max(1.0, min(float(value), 300.0))
        except (TypeError, ValueError):
            pass
    return max(1.0, min(base * (2 ** min(attempt - 1, 2)), 120.0))


def _response_to_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    return {"repr": repr(response)}