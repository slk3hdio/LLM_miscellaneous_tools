from __future__ import annotations

"""局部 UMAP 投影 —— 为单个 generation trace 进行即时降维.

与 PCA 投影（需要预计算全词汇基）不同，UMAP 在每次推理追踪时
动态计算投影，将所有层的隐藏状态、top-k 预测嵌入和最终预测嵌入
统一拟合到一个投影空间中。
"""

import logging
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import torch

from model_visualizer.ui_components.inference.trace import decode_token_ids, hidden_state_for_layer
from model_visualizer.ui_components.embedding_projection.projection import (
    PROJECTION_DIMENSION_OPTIONS,
    normalize_hidden_for_projection,
    top_prediction_token_ids,
    top_prediction_tokens_by_position,
)
from model_visualizer.ui_components.embedding_projection.types import (
    LayerProjection,
    LocalProjection,
    ProjectedToken,
)

logger = logging.getLogger(__name__)

LocalUmapMetric = Literal[
    "cosine",
    "dot_product",
    "dot_product_tsne",
    "exp_dot_product",
    "exp_dot_product_tsne",
]


@dataclass(frozen=True)
class _LocalVector:
    """UMAP 拟合所需的中间向量表示.

    每个 _LocalVector 代表一个需要参与投影的点，可能是：
    - 某一层的隐藏状态向量（role="layer"）
    - top-k 预测的嵌入向量（role="top_prediction"）
    - 最终的 top-1 预测嵌入（role="final_prediction"）
    """

    role: str                    # 向量类别：layer / top_prediction / final_prediction
    layer_index: int | None      # 所属层号（嵌入/预测类为 None）
    index: int                   # 序列位置或预测排名
    token_id: int
    text: str
    vector: np.ndarray           # 实际的向量值（hidden 或 embedding）
    current_token_id: int | None = None   # 当前位置模型预测的最佳 token ID
    current_text: str | None = None       # 当前位置模型预测的最佳 token 文本


def _decode_token(tokenizer, token_id: int) -> str:
    """解码单个 token，保留特殊 token 和空白."""
    return tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _as_sequence_vectors(hidden_state: torch.Tensor) -> torch.Tensor:
    """将隐藏状态转为 [seq, hidden] 的 CPU float32 张量."""
    vectors = hidden_state.detach().to(device="cpu", dtype=torch.float32)
    if vectors.ndim == 3 and vectors.shape[0] == 1:
        vectors = vectors.squeeze(0)
    if vectors.ndim != 2:
        raise ValueError(f"Expected hidden states [seq, hidden], got {tuple(vectors.shape)}.")
    return vectors


def _embedding_rows(embedding_weight: torch.Tensor, token_ids: list[int]) -> torch.Tensor:
    """从嵌入权重矩阵中提取指定 token ID 的行向量."""
    if not token_ids:
        return torch.empty((0, 0), dtype=torch.float32)
    source_weight = embedding_weight.detach()
    if source_weight.device.type == "meta":
        raise ValueError("Input embedding weight is still on the meta device.")
    row_ids = torch.as_tensor([int(token_id) for token_id in token_ids], dtype=torch.long)
    return source_weight.index_select(0, row_ids.to(source_weight.device)).to(
        device="cpu",
        dtype=torch.float32,
    )


def final_prediction_token_ids(
    hidden_state: torch.Tensor,
    lm_head,
    *,
    top_k: int,
    apply_final_norm: bool,
) -> list[int]:
    """获取序列最后位置的 top-k 预测 token ID（仅最后一个位置）.

    与 projection.py 中的 top_prediction_token_ids（所有位置）不同，
    这个函数只关心最后一个位置的预测，用于显示"模型最终预测的下一个 token"。
    """

    if top_k < 1:
        return []
    logits = lm_head.decode(hidden_state, apply_final_norm=apply_final_norm)
    if logits.ndim == 3 and logits.shape[0] == 1:
        logits = logits.squeeze(0)
    if logits.ndim != 2:
        raise ValueError(f"Expected logits with shape [seq, vocab], got {tuple(logits.shape)}.")
    # 仅取最后一个序列位置的 logits
    _scores, token_ids = torch.topk(
        logits[-1].detach().to(device="cpu", dtype=torch.float32),
        k=min(top_k, logits.shape[-1]),
        dim=-1,
    )
    return [int(token_id) for token_id in token_ids.tolist()]


def _vectors_to_projected_tokens(vectors: list[_LocalVector], coords: np.ndarray) -> list[ProjectedToken]:
    """将 _LocalVector 列表和 UMAP 坐标合并为 ProjectedToken 列表."""
    return [
        ProjectedToken(
            index=item.index,
            token_id=item.token_id,
            text=item.text,
            x=float(coord[0]),
            y=float(coord[1]),
            z=float(coord[2]) if coords.shape[1] >= 3 else 0.0,
            current_token_id=item.current_token_id,
            current_text=item.current_text,
        )
        for item, coord in zip(vectors, coords)
    ]


def _projection_fit_key(item: _LocalVector) -> tuple[str, int | None, int]:
    """生成投影拟合的唯一键，用于去重.

    嵌入/预测类向量按 token_id 去重（同一个 token 的嵌入在所有层中相同），
    隐藏状态向量按 (role, layer_index, index) 去重。
    """

    if item.role in {"initial", "top_prediction", "best_prediction", "final_prediction"}:
        return ("embedding", None, item.token_id)
    return (item.role, item.layer_index, item.index)


def dot_product_distance_matrix(matrix: np.ndarray) -> np.ndarray:
    """将点积相似度矩阵转换为可用于 UMAP 的预计算距离矩阵.

    转换公式：distance = max_similarity - similarity
    对角线强制设为 0，NaN/inf 被替换为安全值。
    """

    vectors = np.asarray(matrix, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {vectors.shape}.")
    if vectors.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)

    similarities = vectors @ vectors.T
    max_similarity = float(np.nanmax(similarities))
    distances = max_similarity - similarities
    distances = np.nan_to_num(distances, nan=max_similarity, posinf=max_similarity, neginf=0.0)
    distances = np.maximum(distances, 0.0).astype(np.float32)
    np.fill_diagonal(distances, 0.0)
    return distances


def exp_negative_dot_product_distance_matrix(matrix: np.ndarray) -> np.ndarray:
    """Convert vectors to a precomputed distance matrix with exp(-dot(x, y))."""

    vectors = np.asarray(matrix, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {vectors.shape}.")
    if vectors.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)

    similarities = vectors @ vectors.T
    distances = np.exp(np.clip(-similarities, -80.0, 80.0))
    distances = np.nan_to_num(distances, nan=0.0, posinf=np.exp(80.0), neginf=0.0)
    distances = np.maximum(distances, 0.0).astype(np.float32)
    np.fill_diagonal(distances, 0.0)
    return distances


def _fit_local_projection(
    matrix: np.ndarray,
    *,
    dimensions: int,
    metric: LocalUmapMetric,
    reducer_factory: Callable[..., object] | None = None,
) -> np.ndarray:
    """用 UMAP 将矩阵拟合到低维空间.

    处理边缘情况：
    - 空矩阵：返回零矩阵
    - 样本数 <= dimensions + 1：退化为直接截取（无需降维）
    - 正常情况：调用 UMAP（可从外部注入 reducer_factory 以便测试）

    参数：
        matrix: [n_samples, n_features] 输入矩阵
        dimensions: 目标维度
        metric: "cosine" 或 "dot_product"（dot_product 会先转为距离矩阵）
        reducer_factory: UMAP 构造函数（默认从 umap-learn 导入）
    """

    if matrix.shape[0] == 0:
        return np.zeros((0, dimensions), dtype=np.float32)
    # 样本数太少，直接截取前 dimensions 列
    if matrix.shape[0] <= dimensions + 1:
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        coords = centered[:, :dimensions]
        if coords.shape[1] < dimensions:
            coords = np.pad(coords, ((0, 0), (0, dimensions - coords.shape[1])))
        return coords.astype(np.float32)

    fit_matrix = matrix
    reducer_metric = "cosine"
    if metric in {"dot_product", "dot_product_tsne"}:
        fit_matrix = dot_product_distance_matrix(matrix)
        reducer_metric = "precomputed"
    elif metric in {"exp_dot_product", "exp_dot_product_tsne"}:
        fit_matrix = exp_negative_dot_product_distance_matrix(matrix)
        reducer_metric = "precomputed"

    if metric in {"dot_product_tsne", "exp_dot_product_tsne"}:
        if reducer_factory is None:
            try:
                from sklearn.manifold import TSNE
            except ImportError as exc:
                raise ImportError("Install scikit-learn to use dot-product t-SNE projection.") from exc
            reducer_factory = TSNE
        perplexity = min(30.0, max(1.0, (fit_matrix.shape[0] - 1) / 3.0))
        reducer = reducer_factory(
            n_components=dimensions,
            metric=reducer_metric,
            perplexity=perplexity,
            init="random",
            learning_rate="auto",
            random_state=0,
        )
        coords = reducer.fit_transform(fit_matrix)
        return np.asarray(coords, dtype=np.float32)

    # 惰性导入 UMAP（umap-learn 是可选依赖）
    if reducer_factory is None:
        try:
            from umap import UMAP
        except ImportError as exc:
            raise ImportError("Install umap-learn to use local UMAP projection.") from exc
        reducer_factory = UMAP

    n_neighbors = min(15, max(2, matrix.shape[0] - 1))

    reducer = reducer_factory(
        n_components=dimensions,
        metric=reducer_metric,
        n_neighbors=n_neighbors,
        min_dist=0.05,
        random_state=0,  # 固定随机种子以保证可重复性
    )
    coords = reducer.fit_transform(fit_matrix)
    return np.asarray(coords, dtype=np.float32)


def _project_with_unique_fit_vectors(
    vectors: list[_LocalVector],
    *,
    dimensions: int,
    metric: LocalUmapMetric,
    reducer_factory: Callable[..., object] | None = None,
) -> np.ndarray:
    """对去重后的向量执行 UMAP 投影，再将坐标映射回原始列表.

    为什么需要去重：
    - 相同的嵌入向量（如相同的 top-k token）在多层中重复出现，
      直接全部参与 UMAP 拟合会很慢且浪费
    - 去重后用 _projection_fit_key 分组，只拟合唯一向量，
      然后将拟合结果通过位置映射扩展回原始顺序

    步骤：
    1. 遍历 vectors，用 _projection_fit_key 生成唯一键
    2. 相同键的向量只保留一份到 unique_vectors
    3. 记录每个原始位置的对应唯一向量索引（expanded_positions）
    4. 对唯一向量矩阵执行 UMAP
    5. 用 expanded_positions 索引还原坐标
    """

    if not vectors:
        return np.zeros((0, dimensions), dtype=np.float32)

    key_to_position: dict[tuple[str, int | None, int], int] = {}
    unique_vectors: list[np.ndarray] = []
    expanded_positions: list[int] = []
    for item in vectors:
        key = _projection_fit_key(item)
        position = key_to_position.get(key)
        if position is None:
            position = len(unique_vectors)
            key_to_position[key] = position
            unique_vectors.append(item.vector)
        expanded_positions.append(position)

    matrix = np.stack(unique_vectors).astype(np.float32)
    unique_coords = _fit_local_projection(
        matrix,
        dimensions=dimensions,
        metric=metric,
        reducer_factory=reducer_factory,
    )
    return unique_coords[np.asarray(expanded_positions, dtype=np.int64)]


def compute_local_umap_projection(
    hidden_states: list[torch.Tensor] | tuple[torch.Tensor, ...],
    token_ids: list[int],
    tokenizer,
    embedding_weight: torch.Tensor,
    lm_head,
    *,
    layer_count: int,
    top_k: int,
    dimensions: int,
    metric: LocalUmapMetric = "cosine",
    reducer_factory: Callable[..., object] | None = None,
) -> LocalProjection:
    """为单个推理追踪计算局部 UMAP 投影 —— 核心函数.

    与 PCA 投影的区别：
    - PCA：使用预计算的全词汇嵌入基，投影是线性的，每层独立投影
    - UMAP：将所有层的隐藏状态、top-k 预测嵌入、最终预测嵌入
            合并到一个统一的流形中进行降维，能更好地保留局部结构

    详细流程：
    1. 收集嵌入层（layer_index=-1）每个 token 的隐藏状态
    2. 对每个 transformer 层：
       a. 应用 final norm 后收集隐藏状态（含当前位置的预测信息）
       b. 收集该层的 top-k 预测嵌入向量（标记 role="top_prediction"）
    3. 收集最后一层的 top-1 预测嵌入（标记 role="final_prediction"）
    4. 对去重后的向量集执行 UMAP 降维
    5. 将坐标按 (role, layer_index) 分组为 LayerProjection 列表

    参数：
        hidden_states: 所有层的隐藏状态列表
        token_ids: 序列 token ID
        tokenizer: 分词器
        embedding_weight: 全词汇嵌入矩阵
        lm_head: LM 头
        layer_count: transformer 层数
        top_k: 每层收集的 top-k 预测嵌入数
        dimensions: 降维目标维度（2 或 3）
        metric: UMAP 距离度量（"cosine" 或 "dot_product"）
        reducer_factory: UMAP 构造函数（测试用）

    返回：
        LocalProjection 包含分层投影和最终预测点
    """

    if dimensions not in PROJECTION_DIMENSION_OPTIONS:
        raise ValueError(f"dimensions must be one of {PROJECTION_DIMENSION_OPTIONS}.")
    if metric not in (
        "cosine",
        "dot_product",
        "dot_product_tsne",
        "exp_dot_product",
        "exp_dot_product_tsne",
    ):
        raise ValueError(
            "metric must be 'cosine', 'dot_product', 'dot_product_tsne', "
            "'exp_dot_product', or 'exp_dot_product_tsne'."
        )
    if layer_count < 1:
        raise ValueError("layer_count must be at least 1.")
    if len(hidden_states) < layer_count + 1:
        raise ValueError(
            f"Need at least {layer_count + 1} hidden-state entries for {layer_count} layers, "
            f"got {len(hidden_states)}."
        )

    decoded_tokens = decode_token_ids(tokenizer, token_ids)
    vectors: list[_LocalVector] = []

    # ---- 1. 收集嵌入层的隐藏状态 ----
    embed_vectors = _as_sequence_vectors(hidden_states[0])
    embed_predictions = top_prediction_tokens_by_position(
        hidden_states[0],
        lm_head,
        tokenizer,
        apply_final_norm=True,
    )
    for token, vector in zip(decoded_tokens, embed_vectors):
        current_token_id, current_text = embed_predictions[token.index]
        vectors.append(
            _LocalVector(
                role="layer",
                layer_index=-1,
                index=token.index,
                token_id=token.token_id,
                text=token.text,
                vector=vector.numpy(),
                current_token_id=current_token_id,
                current_text=current_text,
            )
        )
    embed_prediction_ids = [token_id for token_id, _text in embed_predictions]
    embed_prediction_rows = _embedding_rows(embedding_weight, embed_prediction_ids)
    for token, (token_id, text), row in zip(decoded_tokens, embed_predictions, embed_prediction_rows):
        vectors.append(
            _LocalVector(
                role="best_prediction",
                layer_index=-1,
                index=token.index,
                token_id=token_id,
                text=text,
                vector=row.numpy(),
            )
        )

    # ---- 2. 逐层收集隐藏状态和 top-k 预测嵌入 ----
    for layer_index in range(layer_count):
        hidden_state, apply_final_norm = hidden_state_for_layer(hidden_states, layer_index)
        normalized_hidden = normalize_hidden_for_projection(
            hidden_state,
            lm_head,
            apply_final_norm=apply_final_norm,
        )
        # 收集该层每个位置的当前最佳预测
        current_predictions = top_prediction_tokens_by_position(
            hidden_state,
            lm_head,
            tokenizer,
            apply_final_norm=apply_final_norm,
        )
        for token, vector in zip(decoded_tokens, _as_sequence_vectors(normalized_hidden)):
            current_token_id, current_text = current_predictions[token.index]
            vectors.append(
                _LocalVector(
                    role="layer",
                    layer_index=layer_index,
                    index=token.index,
                    token_id=token.token_id,
                    text=token.text,
                    vector=vector.numpy(),
                    current_token_id=current_token_id,
                    current_text=current_text,
                )
            )

        # 收集该层每个位置的当前最佳预测嵌入向量，用于在图上醒目标出
        best_prediction_ids = [token_id for token_id, _text in current_predictions]
        best_prediction_rows = _embedding_rows(embedding_weight, best_prediction_ids)
        for token, (token_id, text), row in zip(decoded_tokens, current_predictions, best_prediction_rows):
            vectors.append(
                _LocalVector(
                    role="best_prediction",
                    layer_index=layer_index,
                    index=token.index,
                    token_id=token_id,
                    text=text,
                    vector=row.numpy(),
                )
            )

        # 收集该层的 top-k 预测嵌入向量
        prediction_ids = top_prediction_token_ids(
            hidden_state,
            lm_head,
            top_k=top_k,
            apply_final_norm=apply_final_norm,
        )
        prediction_rows = _embedding_rows(embedding_weight, prediction_ids)
        for index, (token_id, row) in enumerate(zip(prediction_ids, prediction_rows)):
            vectors.append(
                _LocalVector(
                    role="top_prediction",
                    layer_index=layer_index,
                    index=index,
                    token_id=token_id,
                    text=_decode_token(tokenizer, token_id),
                    vector=row.numpy(),
                )
            )

    # ---- 3. 收集最后一层的 top-1 最终预测 ----
    final_hidden_state, final_apply_norm = hidden_state_for_layer(hidden_states, layer_count - 1)
    final_prediction_ids = final_prediction_token_ids(
        final_hidden_state,
        lm_head,
        top_k=1,
        apply_final_norm=final_apply_norm,
    )
    final_rows = _embedding_rows(embedding_weight, final_prediction_ids)
    for index, (token_id, row) in enumerate(zip(final_prediction_ids, final_rows)):
        vectors.append(
            _LocalVector(
                role="final_prediction",
                layer_index=None,
                index=index,
                token_id=token_id,
                text=_decode_token(tokenizer, token_id),
                vector=row.numpy(),
            )
        )

    # ---- 4. UMAP 降维（自动去重） ----
    logger.info(
        "Computing local UMAP: %d vectors, metric=%s, dims=%d",
        len(vectors), metric, dimensions,
    )
    coords = _project_with_unique_fit_vectors(
        vectors,
        dimensions=dimensions,
        metric=metric,
        reducer_factory=reducer_factory,
    )

    # ---- 5. 按 (role, layer_index) 分组为 LayerProjection ----
    grouped: dict[tuple[str, int | None], list[ProjectedToken]] = {}
    for item, point in zip(vectors, _vectors_to_projected_tokens(vectors, coords)):
        grouped.setdefault((item.role, item.layer_index), []).append(point)

    layers = [
        LayerProjection(
            layer_index=-1,
            points=grouped.get(("layer", -1), []),
            top_prediction_points=[],
            best_prediction_points=grouped.get(("best_prediction", -1), []),
        )
    ]
    for layer_index in range(layer_count):
        layers.append(
            LayerProjection(
                layer_index=layer_index,
                points=grouped.get(("layer", layer_index), []),
                top_prediction_points=grouped.get(("top_prediction", layer_index), []),
                best_prediction_points=grouped.get(("best_prediction", layer_index), []),
            )
        )

    return LocalProjection(
        initial_points=[],
        final_prediction_points=grouped.get(("final_prediction", None), []),
        layers=layers,
    )
