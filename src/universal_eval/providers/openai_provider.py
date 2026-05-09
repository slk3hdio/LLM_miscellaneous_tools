from __future__ import annotations

import asyncio
import os
from typing import Any, Optional, Literal
from openai import AsyncOpenAI, OpenAI
from universal_eval.datasets import EvalSample
from .model_provider import ModelProvider
import logging


def _extract_raw_tool_calls(tool_calls: list[Any]) -> list[dict[str, Any]]:
    """从 OpenAI API 响应中提取 tool_call 字典列表。兼容 Pydantic 模型和纯字典。"""
    raw_calls: list[dict[str, Any]] = []
    for tc in tool_calls:
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
    return raw_calls


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

        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.client = OpenAI(api_key=resolved_api_key, base_url=base_url)
        self.async_client = AsyncOpenAI(api_key=resolved_api_key, base_url=base_url)
        self.logger = logging.getLogger(__name__)

    def generate(
        self,
        messages: list[EvalSample.Context],
        conversation_style: Literal['single', 'multi'],
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """调用 OpenAI 兼容 API 生成回复。若有 tool_calls 则转为标准字符串格式返回。"""
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
            if conversation_style == 'single':
                self.logger.warning(f"Received tool calls but conversation_style is single")
            return EvalSample.normalize_raw_tool_calls(_extract_raw_tool_calls(message.tool_calls))
        if tools and conversation_style == 'multi':
            self.logger.warning(f"Did not receive tool calls when using standard tool format")
        if message.content:
            self.logger.info(f"get fallback prediction: {message.content.strip()}")
            return message.content.strip()
        return ""

    async def _async_generate(
        self,
        messages: list[EvalSample.Context],
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        if tools:
            kwargs["tools"] = tools

        response = await self.async_client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        if message.tool_calls:
            raw_calls = _extract_raw_tool_calls(message.tool_calls)
            return EvalSample.normalize_raw_tool_calls(raw_calls)
        if message.content:
            return message.content.strip()
        return ""

    def generate_batch(
        self,
        batch_messages: list[list[EvalSample.Context]],
        conversation_style: Literal['single', 'multi'],
        batch_tools: list[list[dict[str, Any]] | None],
    ) -> list[str]:
        """异步并发调用 OpenAI API，同时发送 batch 中所有请求。"""

        async def _run() -> list[str]:
            tasks = [
                self._async_generate(msgs, tools)
                for msgs, tools in zip(batch_messages, batch_tools)
            ]
            return await asyncio.gather(*tasks)

        return asyncio.run(_run())

    def supports_conversation_format(self) -> bool:
        return True

    def supports_tool_calling(self) -> bool:
        return True

    # @classmethod
    # def from_config(cls):
    #     pass


class VLLMProvider(OpenAICompatibleProvider):
    pass
