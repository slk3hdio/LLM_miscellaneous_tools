from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Literal


from universal_eval.datasets import EvalSample


class ModelProvider(ABC):
    supports_tools: bool = False

    def __init__(
        self,
        model: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    @abstractmethod
    def generate(
        self,
        messages: list[EvalSample.Context],
        conversation_style: Literal['single', 'multi'],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        """生成模型回复。

        Args:
            messages: 对话上下文消息列表
            conversation_style: ``"single"`` 将所有消息压缩为单轮；``"multi"`` 保持多轮对话
            tools: OpenAI 格式的工具定义列表（可选）
        """
        raise NotImplementedError


    def supports_conversation_format(self) -> bool:
        """是否支持多轮对话格式（多消息序列而非单段文本）。"""
        return False

    def supports_tool_calling(self) -> bool:
        """是否支持 OpenAI 工具/函数调用 API。"""
        return False
