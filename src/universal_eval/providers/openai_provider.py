from __future__ import annotations

import os
from typing import Any, Optional

from .model_provider import ModelProvider


class OpenAICompatibleProvider(ModelProvider):
    supports_tools = True

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        super().__init__(model=model, max_new_tokens=max_new_tokens, temperature=temperature)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai package is required for OpenAI/vLLM providers.") from exc

        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.client = OpenAI(api_key=resolved_api_key, base_url=base_url)

    def generate(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "extra_body": {"thinking": {"type": "disabled"}}
        }
        if tools:
            kwargs["tools"] = tools

        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        if message.tool_calls:
            from ..datasets.sample import EvalSample

            raw_calls: list[dict[str, Any]] = []
            for tc in message.tool_calls:
                if hasattr(tc, "model_dump"):
                    raw_calls.append(tc.model_dump())
                elif isinstance(tc, dict):
                    raw_calls.append(tc)
                else:
                    raw_calls.append({
                        "id": getattr(tc, "id", ""),
                        "type": "function",
                        "function": {
                            "name": getattr(tc.function, "name", ""),
                            "arguments": getattr(tc.function, "arguments", "{}"),
                        },
                    })
            return EvalSample.from_openai_tool_calls(raw_calls)
        if message.content:
            return message.content.strip()
        return ""

    @classmethod
    def from_config(cls):
        pass


class VLLMProvider(OpenAICompatibleProvider):
    pass
