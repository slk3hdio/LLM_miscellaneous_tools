from __future__ import annotations

"""层权重矩阵的直方图分析工具."""

import logging
from typing import Callable, Iterable

import numpy as np
import torch

from model_visualizer.analysis.files import DEFAULT_SAMPLE_SIZE, load_tensor, sample_tensor_values
from model_visualizer.analysis.types import LayerHistogram, TensorInfo

logger = logging.getLogger(__name__)


def available_layer_matrix_keys(infos: Iterable[TensorInfo]) -> list[str]:
    """从张量信息中提取可用的层局部 2D 权重矩阵键.

    筛选条件：
    - info.layer 不为 None（属于某个 transformer 层）
    - info.parameter == "weight"（权重矩阵而非 bias 等）
    - len(info.shape) == 2（2D 矩阵，排除 1D 向量如 layernorm 权重）

    返回格式如 ["self_attn.q_proj.weight", "mlp.down_proj.weight", ...]
    """

    keys = {
        f"{info.module}.{info.parameter}"
        for info in infos
        if info.layer is not None
        and info.parameter == "weight"
        and len(info.shape) == 2
    }
    return sorted(keys)


def matching_layer_matrix_infos(
    infos: Iterable[TensorInfo],
    matrix_key: str,
) -> list[TensorInfo]:
    """筛选匹配指定矩阵键（如 "self_attn.q_proj.weight"）的所有层张量."""

    matches = [
        info
        for info in infos
        if info.layer is not None
        and len(info.shape) == 2
        and f"{info.module}.{info.parameter}" == matrix_key
    ]
    return sorted(matches, key=lambda item: item.layer if item.layer is not None else -1)


def compute_layer_histogram_stack(
    infos: Iterable[TensorInfo],
    matrix_key: str,
    bins: int = 80,
    max_values_per_layer: int = DEFAULT_SAMPLE_SIZE,
    density: bool = True,
    *,
    tensor_loader: Callable[[str, str], torch.Tensor] = load_tensor,
) -> list[LayerHistogram]:
    """为指定矩阵键计算所有层的直方图，使用统一的箱边界以便横向比较.

    详细流程：
    1. 筛选匹配 matrix_key 的所有层张量
    2. 对每层张量采样（限制 max_values_per_layer）
    3. 将所有层的采样值拼接，确定全局最小/最大值
    4. 在全局 min/max 之间生成统一的 bins+1 个箱边界
    5. 用 numpy.histogram 为每层单独计算直方图

    这样做的好处：所有层共享相同的 X 轴范围，可以直接在 3D 堆叠图中比较分布变化。

    参数：
        infos: 所有张量的元数据列表
        matrix_key: 目标矩阵键
        bins: 箱数
        max_values_per_layer: 每层采样上限
        density: 是否归一化为密度
        tensor_loader: 张量加载函数（可注入以支持测试）

    返回：
        LayerHistogram 列表，每层一个
    """

    matches = matching_layer_matrix_infos(infos, matrix_key)
    if not matches:
        logger.warning("No layers matched matrix_key=%s", matrix_key)
        return []

    # 2. 采样每层的值
    samples: list[tuple[TensorInfo, np.ndarray]] = []
    for info in matches:
        sample = sample_tensor_values(
            tensor_loader(info.file, info.name),
            max_values=max_values_per_layer,
        )
        if sample.size:
            samples.append((info, sample))
    if not samples:
        logger.warning("No valid samples for matrix_key=%s", matrix_key)
        return []

    # 3. 确定全局范围（使用 10%/90% 百分位数以减少极端值影响）
    all_values = np.concatenate([sample for _info, sample in samples])
    # min_value, max_value = np.percentile(all_values, [3, 97]).astype(float)
    min_value = -0.1
    max_value = 0.1
    # 如果 percentile 相同，加一个微小的偏移避免零宽度 bin
    if min_value == max_value:
        min_value -= 0.5
        max_value += 0.5

    # 4. 生成统一箱边界
    bin_edges = np.linspace(min_value, max_value, bins + 1)
    bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).astype(float)

    # 5. 逐层计算直方图
    histograms: list[LayerHistogram] = []
    for info, sample in samples:
        values, _edges = np.histogram(sample, bins=bin_edges, density=density)
        histograms.append(
            LayerHistogram(
                layer=info.layer if info.layer is not None else -1,
                tensor_name=info.name,
                bin_centers=tuple(float(value) for value in bin_centers),
                values=tuple(float(value) for value in values),
            )
        )
    logger.info(
        "Computed histogram stack for %s: %d layers, range [%.4g, %.4g], bins=%d",
        matrix_key, len(histograms), min_value, max_value, bins,
    )
    return histograms
