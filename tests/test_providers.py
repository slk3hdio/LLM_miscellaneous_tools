from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from universal_eval.providers import (
    LocalTransformersProvider,
    OpenAICompatibleProvider,
    VLLMProvider,
    create_provider,
)


ROOT = Path(__file__).resolve().parents[1]


class FakeChatTokenizer:
    chat_template = "template"

    def __init__(self, render_tools: bool = True, reject_openai_tools: bool = False) -> None:
        self.calls = []
        self.render_tools = render_tools
        self.reject_openai_tools = reject_openai_tools

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, tools=None):
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "tools": tools,
            }
        )
        if tools is not None:
            if self.reject_openai_tools and tools[0].get("type") == "function":
                raise TypeError("OpenAI tool wrapper is not supported")
            if self.render_tools:
                tool = tools[0].get("function", tools[0])
                return f"TOOLS: {tool['name']}"
        return "CHAT"


class FakePlainTokenizer:
    chat_template = None


def _fake_local_provider(tokenizer):
    provider = LocalTransformersProvider.__new__(LocalTransformersProvider)
    provider.tokenizer = tokenizer
    provider.logger = type("Logger", (), {"info": lambda *args: None, "warning": lambda *args: None, "debug": lambda *args: None})()
    provider._tool_template_format = None
    provider._supports_conversation_format = provider._detect_conversation_format_support()
    provider._supports_tool_calling = provider._detect_tool_calling_support()
    return provider


def test_create_provider_returns_openai_compatible_provider(monkeypatch):
    class FakeOpenAI:
        def __init__(self, api_key, base_url=None):
            self.api_key = api_key
            self.base_url = base_url

    fake_module = type("FakeModule", (), {"OpenAI": FakeOpenAI})
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    provider = create_provider(
        provider_config={
            "active": "openai",
            "openai": {
                "model": "demo-model",
                "base_url": "https://example.test/v1",
                "api_key": "test-key",
            },
        },
        max_new_tokens=32,
        temperature=0.0,
    )

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.model == "demo-model"
    assert provider.max_new_tokens == 32
    assert provider.client.api_key == "test-key"
    assert provider.client.base_url == "https://example.test/v1"


def test_create_provider_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported provider type"):
        create_provider(
            provider_config={"active": "unknown"},
            max_new_tokens=32,
            temperature=0.0,
        )


def test_local_provider_single_turn_uses_plain_completion_without_chat_template():
    tokenizer = FakeChatTokenizer()
    provider = _fake_local_provider(tokenizer)
    tokenizer.calls.clear()

    prompt = provider._render_prompt(
        [{"role": "user", "content": "plain prompt"}],
        tools=[{"type": "function", "function": {"name": "A"}}],
    )

    assert prompt == "plain prompt"
    assert tokenizer.calls == []


def test_local_provider_detects_conversation_support_from_chat_template_probe():
    assert _fake_local_provider(FakeChatTokenizer()).supports_conversation_format() is True
    assert _fake_local_provider(FakePlainTokenizer()).supports_conversation_format() is False


def test_local_provider_detects_tool_calling_only_when_tools_are_rendered():
    assert _fake_local_provider(FakeChatTokenizer(render_tools=True)).supports_tool_calling() is True
    assert _fake_local_provider(FakeChatTokenizer(render_tools=False)).supports_tool_calling() is False


def test_local_provider_uses_function_tool_schema_when_template_rejects_openai_wrapper():
    tokenizer = FakeChatTokenizer(reject_openai_tools=True)
    provider = _fake_local_provider(tokenizer)
    tokenizer.calls.clear()

    prompt = provider._render_prompt(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "call a tool"},
        ],
        tools=[{"type": "function", "function": {"name": "A", "parameters": {"type": "object"}}}],
    )

    assert prompt == "TOOLS: A"
    assert tokenizer.calls[-1]["tools"] == [{"name": "A", "parameters": {"type": "object"}}]


@pytest.mark.integration
def test_openai_provider_sends_real_request():
    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_TEST_MODEL")
    if not api_key or not model:
        pytest.skip("Set OPENAI_API_KEY and OPENAI_TEST_MODEL to run the real OpenAI integration test.")

    provider = create_provider(
        provider_config={
            "active": "openai",
            "openai": {
                "model": model,
                "api_key": api_key,
                "base_url": os.environ.get("OPENAI_BASE_URL"),
            },
        },
        max_new_tokens=32,
        temperature=0.0,
    )

    response = provider.generate(
        prompt="Reply with exactly PONG.",
        system_prompt="Return exactly the requested token and nothing else.",
    )
    print(f"Test OpenAI provider response(use model: {model}):")
    print(f"prompt: Reply with exactly PONG.")
    print(f"Received response: {response}")

    assert response
    assert "PONG" in response


@pytest.mark.integration
def test_local_provider_loads_real_weights_and_generates():
    model_path_env = os.environ.get("LOCAL_MODEL_PATH")
    if not model_path_env:
        pytest.skip("Set LOCAL_MODEL_PATH to run the real local model integration test.")
    model_path = Path(model_path_env)
    if not model_path.exists():
        pytest.skip(f"Local model path does not exist: {model_path}")

    provider = create_provider(
        provider_config={
            "active": "local",
            "local": {
                "model_path": str(model_path),
                "device": "auto",
            },
        },
        max_new_tokens=16,
        temperature=0.0,
    )

    assert isinstance(provider, LocalTransformersProvider)
    assert provider.tokenizer is not None
    assert provider.model_ref is not None

    response = provider.generate(
        prompt="Please answer with the single token OK.",
        system_prompt="Be concise.",
    )
    print(f"Test Local provider response(use model: {model_path}):")
    print(f"prompt: Please answer with the single token OK.")
    print(f"Received response: {response}")

    assert isinstance(response, str)
    assert response.strip() != ""
