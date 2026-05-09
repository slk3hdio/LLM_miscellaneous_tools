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
    """数据集适配器抽象基类。

    所有数据集适配器必须实现 :meth:`load_samples`，将原始数据转换为统一的 :class:`EvalSample` 列表。
    """
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
        """加载并返回标准化评测样本列表。

        Args:
            limit: 最多加载的样本数，None 表示全部加载
            conversation_style: ``"single"`` 将所有历史压缩为一个 user 消息；``"multi"`` 保留多轮结构
            with_raw_data: 是否在 metadata 中保留原始数据
            random_samples: 是否随机打乱样本顺序
            strip_tool_descriptions: 是否去除工具描述（standard tool format 模式使用）
        """
        raise NotImplementedError
