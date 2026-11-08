"""Unit tests for the SiliconFlow OpenAI-compatible client."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from toolsim.llm.siliconflow_client import SiliconFlowClient
from toolsim.llm.types import LLMMessage


class _FakeCompletions:
    def __init__(self) -> None:
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"action":"done"}'))],
            model="openai/gpt-oss-120b",
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=5, total_tokens=17),
        )


def test_siliconflow_client_uses_model_and_json_mode():
    completions = _FakeCompletions()
    client = SiliconFlowClient(api_key="test-key")
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    response = client.complete(
        [LLMMessage(role="user", content="test")],
        response_format={"type": "json_object"},
        max_tokens=128,
        temperature=0.0,
    )

    assert completions.request["model"] == "openai/gpt-oss-120b"
    assert completions.request["response_format"] == {"type": "json_object"}
    assert completions.request["max_tokens"] == 128
    assert response.usage.total_tokens == 17


def test_siliconflow_client_can_disable_thinking():
    completions = _FakeCompletions()
    client = SiliconFlowClient(api_key="test-key", enable_thinking=False)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    client.complete([LLMMessage(role="user", content="test")])

    assert completions.request["extra_body"] == {"enable_thinking": False}

def test_siliconflow_client_requires_api_key():
    client = SiliconFlowClient(api_key="")
    client.api_key = None
    try:
        client.complete([LLMMessage(role="user", content="test")])
    except RuntimeError as exc:
        assert "SILICONFLOW_API_KEY" in str(exc)
    else:
        raise AssertionError("Expected missing-key RuntimeError")


def test_siliconflow_client_retries_rate_limits(monkeypatch):
    class FakeRateLimitError(Exception):
        status_code = 429
        response = None

    class FlakyCompletions(_FakeCompletions):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise FakeRateLimitError("TPM limit reached")
            return super().create(**kwargs)

    completions = FlakyCompletions()
    client = SiliconFlowClient(
        api_key="test-key",
        rate_limit_retries=1,
        rate_limit_backoff=1,
    )
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr("toolsim.llm.siliconflow_client.time.sleep", lambda _: None)

    response = client.complete([LLMMessage(role="user", content="test")])

    assert completions.calls == 2
    assert response.content == '{"action":"done"}'