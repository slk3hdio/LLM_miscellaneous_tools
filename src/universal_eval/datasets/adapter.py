from __future__ import annotations

import ast
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import logging

from .sample import EvalSample


class DatasetAdapter(ABC):
    name: str = "base"

    def __init__(self, path:Any, split:Any):
        self.path = Path(path)
        self.split = split

    @abstractmethod
    def load_samples(
        self,
        limit: int | None = None,
        conversation_style: Literal['single', 'multi'] = 'single',
        with_raw_data: bool = False,
        random_samples: bool = False,
        strip_tool_descriptions: bool = False,
    ) -> List[EvalSample]:
        raise NotImplementedError
