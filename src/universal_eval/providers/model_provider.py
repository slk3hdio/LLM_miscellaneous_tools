from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional, Type, Dict, Any


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
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        raise NotImplementedError