from __future__ import annotations

"""嵌入投影可视化的内部数据类型."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ProjectionBasis:
    """PCA 投影基 —— 从全词汇嵌入矩阵计算得到.

    属性含义：
    - mean: 嵌入向量的均值（vocab 求平均），形状 [hidden_size]
    - components: 主成分矩阵，形状 [hidden_size, dimensions]
    - explained_variance: 各主成分的解释方差
    - explained_variance_ratio: 各主成分的解释方差比（0-1）
    """

    mean: NDArray[np.float32]
    components: NDArray[np.float32]
    explained_variance: NDArray[np.float32]
    explained_variance_ratio: NDArray[np.float32]
    model_name: str                     # 模型名称
    embedding_tensor_name: str          # 嵌入张量名称
    vocab_size: int                     # 词汇量
    hidden_size: int                    # 隐藏维度
    path: Path | None = None            # PCA 文件路径


@dataclass(frozen=True)
class ProjectedToken:
    """投影到 PCA 空间后的单个 token."""

    index: int      # 序列中的位置
    token_id: int   # token ID
    text: str       # 解码文本
    x: float        # PC1 坐标
    y: float        # PC2 坐标
    z: float        # PC3 坐标（2D 投影时为 0）
    current_token_id: int | None = None
    current_text: str | None = None


@dataclass(frozen=True)
class LocalProjection:
    initial_points: list[ProjectedToken]
    final_prediction_points: list[ProjectedToken]
    layers: list[LayerProjection]


@dataclass(frozen=True)
class LayerProjection:
    """某一层的完整投影数据."""

    layer_index: int                        # 层号（-1 表示嵌入层）
    points: list[ProjectedToken]            # 主投影点（当前序列的隐藏状态）
    top_prediction_points: list[ProjectedToken]  # top-k 预测 token 的嵌入投影
    best_prediction_points: list[ProjectedToken] = field(default_factory=list)
