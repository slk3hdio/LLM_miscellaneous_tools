from __future__ import annotations

"""逐步推理追踪的辅助函数 —— 解码、注意力头选择、top-k 预测."""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class DecodedToken:
    """供 UI 展示的已解码 token 元数据."""

    index: int        # 在序列中的位置
    token_id: int     # token ID
    text: str         # 解码后的文本


@dataclass(frozen=True)
class TopTokenPrediction:
    """单个 token 位置的 LM 头候选预测."""

    rank: int         # 概率排名（1=最高）
    token_id: int
    text: str
    probability: float


@dataclass(frozen=True)
class StepAdvance:
    """下一步动作的纯逻辑决策 —— 前进到哪一层/哪一步."""

    generation_step: int   # 目标生成步（从 0 开始）
    layer_index: int       # 目标层号
    should_generate: bool  # 是否需要在前进前执行一次 generate


def decode_single_token(tokenizer: Any, token_id: int) -> str:
    """解码单个 token，保留特殊 token 和空白字符."""

    return tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def decode_token_ids(tokenizer: Any, token_ids: list[int] | tuple[int, ...]) -> list[DecodedToken]:
    """批量解码 token ID 列表，保持原始顺序."""

    return [
        DecodedToken(
            index=index,
            token_id=int(token_id),
            text=decode_single_token(tokenizer, int(token_id)),
        )
        for index, token_id in enumerate(token_ids)
    ]


def select_attention_head(attention: torch.Tensor, head_index: int) -> torch.Tensor:
    """从注意力权重张量中提取指定 head 并转为 CPU float32.

    参数：
        attention: 形状为 [heads, seq, seq] 的注意力权重
        head_index: 要提取的 head 索引（0-based）

    返回：
        CPU float32 类型的 [seq, seq] 矩阵
    """

    if attention.ndim != 3:
        raise ValueError(
            f"Expected attention with shape [heads, seq, seq], got {tuple(attention.shape)}."
        )
    if not 0 <= head_index < attention.shape[0]:
        raise IndexError(
            f"Head index {head_index} is outside available heads 0..{attention.shape[0] - 1}."
        )
    return attention[head_index].detach().to(device="cpu", dtype=torch.float32)


def top_token_predictions(
    hidden_state: torch.Tensor,
    lm_head: Any,
    tokenizer: Any,
    *,
    top_n: int = 5,
    apply_final_norm: bool = True,
) -> list[list[TopTokenPrediction]]:
    """对序列每个位置的隐藏状态计算 LM 头的 top-N 预测.

    逐步流程：
    1. 通过 LM 头将隐藏状态解码为 logits
    2. 处理 batch 维度（squeeze batch=1）
    3. 对 logits 做 softmax 得到概率分布
    4. 使用 torch.topk 选取每个位置的前 N 个候选
    5. 解码为 TopTokenPrediction 列表

    参数：
        hidden_state: 形状为 [seq, hidden] 的隐藏状态张量
        lm_head: 语言模型头（将 hidden -> vocab）
        tokenizer: 用于将 token ID 解码为文本
        top_n: 每位置返回前 N 个预测
        apply_final_norm: 是否在 LM 头解码前应用 final layer norm

    返回：
        二维列表：predictions[position][rank] → TopTokenPrediction
    """

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    # 1. 通过 LM 头得到 logits
    logits = lm_head.decode(hidden_state, apply_final_norm=apply_final_norm)
    # 2. 去除 batch 维度
    if logits.ndim == 3 and logits.shape[0] == 1:
        logits = logits.squeeze(0)
    if logits.ndim != 2:
        raise ValueError(f"Expected logits with shape [seq, vocab], got {tuple(logits.shape)}.")

    # 3. Softmax 转换为概率
    probabilities = torch.softmax(
        logits.detach().to(device="cpu", dtype=torch.float32),
        dim=-1,
    )
    # 4. Top-k 选取
    k = min(top_n, probabilities.shape[-1])
    scores, token_ids = torch.topk(probabilities, k=k, dim=-1)

    # 5. 逐位置解码
    rows: list[list[TopTokenPrediction]] = []
    for position_scores, position_token_ids in zip(scores, token_ids):
        rows.append(
            [
                TopTokenPrediction(
                    rank=rank + 1,
                    token_id=int(token_id),
                    text=decode_single_token(tokenizer, int(token_id)),
                    probability=float(score),
                )
                for rank, (score, token_id) in enumerate(zip(position_scores, position_token_ids))
            ]
        )
    return rows


def next_inference_position(
    generation_step: int,
    layer_index: int,
    num_layers: int,
    *,
    has_trace: bool,
) -> StepAdvance:
    """计算"Next step"按钮点击后的目标位置.

    决策逻辑：
    1. 如果没有活跃追踪 → 进入第 1 个 generation step 的第 0 层（需 generate）
    2. 如果同一 generation step 还有下一层 → 层号 +1（不需 generate）
    3. 如果已是当前 step 的最后一层 → 进入下一个 generation step 的第 0 层（需 generate）

    参数：
        generation_step: 当前生成步号
        layer_index: 当前层号
        num_layers: 模型总层数
        has_trace: 当前是否有有效的追踪数据

    返回：
        StepAdvance 描述下一步应该去往哪个 generation_step / layer_index
    """

    if num_layers < 1:
        raise ValueError("num_layers must be at least 1.")
    # 情况 1：无追踪 → 前进到第一个生成步
    if not has_trace:
        return StepAdvance(generation_step=max(generation_step, -1) + 1, layer_index=0, should_generate=True)
    # 情况 2：同一生成步内前进一层
    if layer_index + 1 < num_layers:
        return StepAdvance(
            generation_step=generation_step,
            layer_index=layer_index + 1,
            should_generate=False,
        )
    # 情况 3：进入下一个生成步
    return StepAdvance(
        generation_step=generation_step + 1,
        layer_index=0,
        should_generate=True,
    )


def hidden_state_for_layer(
    hidden_states: list[torch.Tensor] | tuple[torch.Tensor, ...],
    layer_index: int,
) -> tuple[torch.Tensor, bool]:
    """获取指定 transformer 层的隐藏状态，并返回是否需要应用 final norm.

    约定：hidden_states[0] 是嵌入层输出，hidden_states[layer+1] 是第 layer 层的输出。
    layer_index=0 对应 hidden_states[1]。

    返回：
        (hidden_state, apply_final_norm)：
        - hidden_state: 该层的隐藏状态张量
        - apply_final_norm: 是否为非最终层（非最终层需要 final norm 来正确投影）
    """

    hidden_state_index = layer_index + 1
    if not 0 <= hidden_state_index < len(hidden_states):
        raise IndexError(
            f"Layer {layer_index} maps to hidden state {hidden_state_index}, "
            f"but only {len(hidden_states)} hidden states are available."
        )
    # 最后一个隐藏状态是 LM 头之前的最终输出，不需要额外 final norm
    is_final_hidden_state = hidden_state_index == len(hidden_states) - 1
    return hidden_states[hidden_state_index], not is_final_hidden_state
