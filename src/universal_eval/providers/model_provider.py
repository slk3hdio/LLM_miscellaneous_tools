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
        # conversation_style: Literal['single', 'multi'],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        raise NotImplementedError


    def supports_conversation_format(self) -> bool:
        """Return whether this provider can consume multi-message conversations."""
        return False

    def supports_tool_calling(self) -> bool:
        """Return whether this provider supports the OpenAI tools/function-calling API."""
        return False
