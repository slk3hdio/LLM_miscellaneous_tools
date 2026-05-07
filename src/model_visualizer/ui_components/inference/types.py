from __future__ import annotations

"""推理步进器组件与嵌入投影组件之间的数据契约."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InferenceTraceView:
    """当前追踪切片，供嵌入投影等相邻组件消费.

    这是一个不可变快照，包含了渲染某一步嵌入投影所需的所有信息。
    """

    tokenizer: Any              # 分词器
    manager: Any                # TraceManager 实例（含 hidden_states 和 attention_weights）
    model: Any                  # 因果语言模型
    lm_head: Any                # LM 头
    generation_step: int        # 当前生成步号
    layer_index: int            # 当前层号
    top_n: int                  # Top-N 预测数
    trace_token_ids: list[int]  # 当前追踪中的 token ID 列表
    num_layers: int             # 模型总层数
    model_dir: str              # 模型目录路径
