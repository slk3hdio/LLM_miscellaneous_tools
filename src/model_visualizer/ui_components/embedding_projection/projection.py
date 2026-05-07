from __future__ import annotations

"""PCA 投影辅助工具 —— 计算投影基、投影隐藏状态和嵌入向量."""

import logging
from pathlib import Path

import numpy as np
import torch

from model_visualizer.ui_components.inference.trace import decode_token_ids, hidden_state_for_layer
from model_visualizer.ui_components.embedding_projection.types import (
    LayerProjection,
    ProjectedToken,
    ProjectionBasis,
)

logger = logging.getLogger(__name__)


DEFAULT_PROJECTION_DIR = Path("outputs/model_visualizer/embedding_projection")
DEFAULT_EMBEDDING_TENSOR_NAME = "model.embed_tokens.weight"
PROJECTION_DIMENSION_OPTIONS = (2, 3)
DEFAULT_PROJECTION_DIMENSIONS = 2


def projection_output_path(
    model_dir: str | Path,
    *,
    output_dir: str | Path = DEFAULT_PROJECTION_DIR,
) -> Path:
    """返回 PCA 投影文件的默认存储路径."""
    return Path(output_dir) / f"{Path(model_dir).name}_pca.npz"


def stabilize_component_signs(components: torch.Tensor) -> torch.Tensor:
    """稳定 PCA 分量的符号，使其具有确定性.

    问题：PCA 的特征向量符号可以任意翻转（v 和 -v 都是有效解）。
    解决：对每一列，找到绝对值最大的元素的位置（pivot），如果它是负的就将整列取反。
    这样保证同一个矩阵在不同环境下计算的 PCA 符号一致。
    """

    stable = components.clone()
    for column in range(stable.shape[1]):
        component = stable[:, column]
        pivot = int(torch.argmax(torch.abs(component)).item())
        if component[pivot] < 0:
            stable[:, column] = -component
    return stable


def compute_embedding_projection_basis(
    embedding: torch.Tensor,
    *,
    model_name: str,
    embedding_tensor_name: str = DEFAULT_EMBEDDING_TENSOR_NAME,
    dimensions: int = DEFAULT_PROJECTION_DIMENSIONS,
) -> ProjectionBasis:
    """从全词汇嵌入矩阵计算 PCA 投影基.

    计算流程：
    1. 将嵌入矩阵转为 CPU float32
    2. 计算均值并中心化
    3. 计算协方差矩阵 C = X^T·X / (n-1)（样本协方差）
    4. 使用 torch.linalg.eigh 做特征值分解（对称矩阵，更稳定）
    5. 选取 top-dimensions 个特征向量
    6. 不足 dimensions 时补零
    7. 调用 stabilize_component_signs 统一符号方向
    8. 计算解释方差比

    参数：
        embedding: [vocab_size, hidden_size] 全词汇嵌入矩阵
        model_name: 模型名称（存入元数据）
        embedding_tensor_name: 嵌入张量名称
        dimensions: 目标降维维度（2 或 3）

    返回：
        ProjectionBasis 包含均值、主成分、方差信息
    """

    if dimensions not in PROJECTION_DIMENSION_OPTIONS:
        raise ValueError(f"dimensions must be one of {PROJECTION_DIMENSION_OPTIONS}.")

    # 1. 准备数据
    matrix = embedding.detach().to(device="cpu", dtype=torch.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected embedding matrix [vocab, hidden], got {tuple(matrix.shape)}.")

    vocab_size, hidden_size = matrix.shape

    # 2. 中心化
    mean = matrix.mean(dim=0)
    centered = matrix - mean

    # 3. 协方差矩阵（样本协方差，除以 n-1）
    if vocab_size <= 1:
        cov = torch.zeros((hidden_size, hidden_size), dtype=torch.float32)
    else:
        cov = centered.T @ centered / (vocab_size - 1)

    # 4. 特征值分解
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    # eigh 返回升序排列，取最大的几个
    order = torch.argsort(eigenvalues, descending=True)
    top_order = order[: min(dimensions, hidden_size)]
    components = eigenvectors[:, top_order]
    explained_variance = eigenvalues[top_order].clamp_min(0)  # 防止数值误差导致负值

    # 5. 不足 dimensions 时补零
    if components.shape[1] < dimensions:
        components = torch.nn.functional.pad(components, (0, dimensions - components.shape[1]))
        explained_variance = torch.nn.functional.pad(
            explained_variance,
            (0, dimensions - explained_variance.shape[0]),
        )

    # 6. 稳定符号
    components = stabilize_component_signs(components[:, :dimensions])
    explained_variance = explained_variance[:dimensions]

    # 7. 解释方差比
    total_variance = eigenvalues.clamp_min(0).sum()
    explained_variance_ratio = (
        explained_variance / total_variance
        if float(total_variance) > 0
        else torch.zeros_like(explained_variance)
    )

    logger.info(
        "Computed PCA basis for %s: vocab=%d, hidden=%d, dims=%d, "
        "explained variance ratio: %s",
        model_name, vocab_size, hidden_size, dimensions,
        ", ".join(f"{float(explained_variance_ratio[i]) * 100:.1f}%" for i in range(dimensions)),
    )
    return ProjectionBasis(
        mean=mean.numpy(),
        components=components.numpy(),
        explained_variance=explained_variance.numpy(),
        explained_variance_ratio=explained_variance_ratio.numpy(),
        model_name=model_name,
        embedding_tensor_name=embedding_tensor_name,
        vocab_size=int(vocab_size),
        hidden_size=int(hidden_size),
    )


def save_projection_basis(basis: ProjectionBasis, output_path: str | Path) -> Path:
    """将 PCA 基保存为 .npz 文件."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving projection basis to %s", path)
    np.savez(
        path,
        mean=np.asarray(basis.mean, dtype=np.float32),
        components=np.asarray(basis.components, dtype=np.float32),
        explained_variance=np.asarray(basis.explained_variance, dtype=np.float32),
        explained_variance_ratio=np.asarray(basis.explained_variance_ratio, dtype=np.float32),
        embedding_tensor_name=np.asarray(basis.embedding_tensor_name),
        vocab_size=np.asarray(basis.vocab_size),
        hidden_size=np.asarray(basis.hidden_size),
        model_name=np.asarray(basis.model_name),
    )
    return path


def load_projection_basis(path: str | Path) -> ProjectionBasis:
    """从 .npz 文件加载 PCA 基."""
    projection_path = Path(path)
    logger.debug("Loading projection basis from %s", projection_path)
    with np.load(projection_path, allow_pickle=False) as data:
        return ProjectionBasis(
            mean=data["mean"].astype(np.float32),
            components=data["components"].astype(np.float32),
            explained_variance=data["explained_variance"].astype(np.float32),
            explained_variance_ratio=data["explained_variance_ratio"].astype(np.float32),
            embedding_tensor_name=str(data["embedding_tensor_name"].item()),
            vocab_size=int(data["vocab_size"].item()),
            hidden_size=int(data["hidden_size"].item()),
            model_name=str(data["model_name"].item()),
            path=projection_path,
        )


def project_hidden_states(
    hidden_state: torch.Tensor,
    token_ids: list[int],
    tokenizer,
    basis: ProjectionBasis,
    current_predictions: list[tuple[int, str]] | None = None,
) -> list[ProjectedToken]:
    """将序列隐藏状态投影到预计算的嵌入 PCA 空间中.

    步骤：
    1. 将隐藏状态转为 CPU float32，处理可能的 batch 维度
    2. 验证维度匹配（seq 长度 vs token_ids、hidden 维度 vs 基）
    3. coords = (vectors - mean) @ components（中心化后投影）
    4. 为每个 token 创建 ProjectedToken（含 PC1/PC2/PC3 坐标）

    参数：
        hidden_state: [seq, hidden] 或 [1, seq, hidden] 隐藏状态
        token_ids: 序列的 token ID 列表
        tokenizer: 用于解码文本
        basis: 预计算的 PCA 投影基

    返回：
        ProjectedToken 列表（按序列位置排列）
    """

    vectors = hidden_state.detach().to(device="cpu", dtype=torch.float32)
    # 处理 batch 维度
    if vectors.ndim == 3 and vectors.shape[0] == 1:
        vectors = vectors.squeeze(0)
    if vectors.ndim != 2:
        raise ValueError(f"Expected hidden states [seq, hidden], got {tuple(vectors.shape)}.")
    if vectors.shape[0] != len(token_ids):
        raise ValueError(
            f"Hidden-state sequence length {vectors.shape[0]} does not match {len(token_ids)} token ids."
        )

    # 加载基为 torch tensor
    mean = torch.as_tensor(basis.mean, dtype=torch.float32)
    components = torch.as_tensor(basis.components, dtype=torch.float32)
    if vectors.shape[1] != mean.shape[0]:
        raise ValueError(
            f"Hidden size {vectors.shape[1]} does not match projection basis size {mean.shape[0]}."
        )
    if components.ndim != 2 or components.shape[0] != mean.shape[0]:
        raise ValueError(
            f"Expected projection components [hidden, dimensions], got {tuple(components.shape)}."
        )
    if components.shape[1] not in PROJECTION_DIMENSION_OPTIONS:
        raise ValueError(
            f"Projection dimensions must be one of {PROJECTION_DIMENSION_OPTIONS}, "
            f"got {components.shape[1]}."
        )

    # 投影：中心化 × 主成分矩阵 → PCA 坐标
    coords = (vectors - mean) @ components  # [seq, dimensions]
    decoded_tokens = decode_token_ids(tokenizer, token_ids)
    if current_predictions is not None and len(current_predictions) != len(decoded_tokens):
        raise ValueError(
            f"Current prediction count {len(current_predictions)} does not match "
            f"{len(decoded_tokens)} token ids."
        )
    return [
        ProjectedToken(
            index=token.index,
            token_id=token.token_id,
            text=token.text,
            x=float(coord[0]),
            y=float(coord[1]),
            z=float(coord[2]) if components.shape[1] >= 3 else 0.0,
            current_token_id=(
                int(current_predictions[token.index][0])
                if current_predictions is not None
                else None
            ),
            current_text=(
                current_predictions[token.index][1]
                if current_predictions is not None
                else None
            ),
        )
        for token, coord in zip(decoded_tokens, coords)
    ]


def _project_vectors(
    vectors: torch.Tensor,
    token_ids: list[int],
    tokenizer,
    basis: ProjectionBasis,
) -> list[ProjectedToken]:
    """内部辅助：project_hidden_states 的包装."""
    return project_hidden_states(vectors, token_ids, tokenizer, basis)


def model_embedding_weight(model) -> torch.Tensor:
    """从模型中提取输入嵌入权重矩阵.

    尝试三种常见路径（按优先级）：
    1. model.get_input_embeddings().weight  （HF 标准 API）
    2. model.model.embed_tokens.weight        （某些包装模型）
    3. model.embed_tokens.weight              （直接属性）
    """

    if hasattr(model, "get_input_embeddings"):
        embedding = model.get_input_embeddings()
        if embedding is not None and hasattr(embedding, "weight"):
            logger.debug("Extracted embedding weight via get_input_embeddings()")
            return embedding.weight
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        logger.debug("Extracted embedding weight via model.model.embed_tokens")
        return model.model.embed_tokens.weight
    if hasattr(model, "embed_tokens"):
        logger.debug("Extracted embedding weight via model.embed_tokens")
        return model.embed_tokens.weight
    logger.error("Model does not expose input embedding weights")
    raise ValueError("Model does not expose input embedding weights.")


def project_token_embeddings(
    embedding_weight: torch.Tensor,
    token_ids: list[int],
    tokenizer,
    basis: ProjectionBasis,
) -> list[ProjectedToken]:
    """将指定 token 的嵌入向量投影到 PCA 空间中.

    用于显示初始嵌入点（未经 transformer 处理的词向量位置）。

    步骤：
    1. 将 token_ids 转为 long tensor
    2. 使用 index_select 从嵌入矩阵中选取对应行
    3. 调用 _project_vectors 进行 PCA 投影
    """

    if not token_ids:
        return []
    row_ids = torch.as_tensor([int(token_id) for token_id in token_ids], dtype=torch.long)
    source_weight = embedding_weight.detach()
    if source_weight.device.type == "meta":
        raise ValueError("Input embedding weight is still on the meta device.")
    rows = source_weight.index_select(0, row_ids.to(source_weight.device)).to(
        device="cpu",
        dtype=torch.float32,
    )
    return _project_vectors(rows, [int(token_id) for token_id in token_ids], tokenizer, basis)


def top_prediction_token_ids(
    hidden_state: torch.Tensor,
    lm_head,
    *,
    top_k: int,
    apply_final_norm: bool,
) -> list[int]:
    """获取所有序列位置 top-k 预测的唯一 token ID 集合.

    用于嵌入投影中显示"模型预测的下一个 token 在 PCA 空间中的位置"。

    步骤：
    1. 通过 LM 头解码 logits
    2. 使用 topk 选取所有位置的 top-k token ID
    3. 展平并去重（保持出现顺序）
    """

    if top_k < 1:
        return []
    logits = lm_head.decode(hidden_state, apply_final_norm=apply_final_norm)
    if logits.ndim == 3 and logits.shape[0] == 1:
        logits = logits.squeeze(0)
    if logits.ndim != 2:
        raise ValueError(f"Expected logits with shape [seq, vocab], got {tuple(logits.shape)}.")
    _scores, token_ids = torch.topk(
        logits.detach().to(device="cpu", dtype=torch.float32),
        k=min(top_k, logits.shape[-1]),
        dim=-1,
    )
    # 去重但保持顺序（dict.fromkeys 保留插入顺序）
    ordered_unique_ids = dict.fromkeys(int(token_id) for token_id in token_ids.reshape(-1).tolist())
    return list(ordered_unique_ids)


def top_prediction_tokens_by_position(
    hidden_state: torch.Tensor,
    lm_head,
    tokenizer,
    *,
    apply_final_norm: bool,
) -> list[tuple[int, str]]:
    logits = lm_head.decode(hidden_state, apply_final_norm=apply_final_norm)
    if logits.ndim == 3 and logits.shape[0] == 1:
        logits = logits.squeeze(0)
    if logits.ndim != 2:
        raise ValueError(f"Expected logits with shape [seq, vocab], got {tuple(logits.shape)}.")
    token_ids = torch.argmax(logits.detach().to(device="cpu", dtype=torch.float32), dim=-1)
    return [
        (
            int(token_id),
            tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
        )
        for token_id in token_ids.tolist()
    ]


def normalize_hidden_for_projection(
    hidden_state: torch.Tensor,
    lm_head,
    *,
    apply_final_norm: bool,
) -> torch.Tensor:
    """在 PCA 投影前应用 LM 头的 final layer norm.

    中间层的隐藏状态需要通过 final norm 才能正确映射到词汇空间，
    因此投影前先做归一化，使投影更接近"模型看到的"表示。

    步骤：
    1. 检查 lm_head 是否有 final_norm 属性
    2. 如果不需要应用 norm，直接返回
    3. 否则处理设备匹配：如果 lm_head 有 _move_for_module 方法则委托，
       否则手动将 hidden_state 移到 norm 参数所在的设备
    4. 应用 final_norm 并转到 CPU float32
    """

    final_norm = getattr(lm_head, "final_norm", None) if lm_head is not None else None
    if final_norm is None or not apply_final_norm:
        return hidden_state

    with torch.no_grad():
        if hasattr(lm_head, "_move_for_module"):
            # 使用 LM head 的内部设备管理
            normalized = lm_head._move_for_module(hidden_state, final_norm)
        else:
            normalized = hidden_state
            parameter = next(final_norm.parameters(), None)
            if parameter is not None and parameter.device.type != "meta":
                normalized = normalized.to(device=parameter.device, dtype=parameter.dtype)
        normalized = final_norm(normalized)
    return normalized.detach().to(device="cpu", dtype=torch.float32)


def project_hidden_state_layers(
    hidden_states: list[torch.Tensor] | tuple[torch.Tensor, ...],
    token_ids: list[int],
    tokenizer,
    basis: ProjectionBasis,
    *,
    layer_count: int,
    embedding_weight: torch.Tensor | None = None,
    lm_head=None,
    top_k: int = 0,
    include_embedding_layer: bool = False,
) -> list[LayerProjection]:
    """为所有 transformer 层计算 PCA 投影 —— 这是投影组件的核心.

    详细流程：
    1. 可选：投影嵌入层（hidden_states[0]），标记 layer_index=-1
    2. 对每个 transformer 层（0 到 layer_count-1）：
       a. 通过 hidden_state_for_layer 获取该层的隐藏状态
       b. 调用 normalize_hidden_for_projection 应用 final norm
       c. 如果提供了 embedding_weight 和 lm_head：
          - 计算该层的 top-k 预测 token ID
          - 将这些 token 的嵌入向量投影到 PCA 空间（作为参考点）
       d. 投影该层的隐藏状态到 PCA 空间
    3. 收集所有 LayerProjection 并返回

    参数：
        hidden_states: 包含所有层输出的列表
        token_ids: 序列 token ID
        tokenizer: 分词器
        basis: PCA 投影基
        layer_count: 要投影的层数
        embedding_weight: 嵌入权重矩阵（可选，用于投影 top-k 预测）
        lm_head: LM 头（可选，用于获取 top-k 预测）
        top_k: top-k 预测数
        include_embedding_layer: 是否包含嵌入层（layer_index=-1）

    返回：
        LayerProjection 列表（每层一个）
    """

    if layer_count < 1:
        raise ValueError("layer_count must be at least 1.")
    if len(hidden_states) < layer_count + 1:
        raise ValueError(
            f"Need at least {layer_count + 1} hidden-state entries for {layer_count} layers, "
            f"got {len(hidden_states)}."
        )

    logger.info(
        "Projecting %d layers (include_embedding=%s) for %d tokens",
        layer_count, include_embedding_layer, len(token_ids),
    )
    layer_projections: list[LayerProjection] = []

    # 嵌入层（可选）
    if include_embedding_layer:
        embedding_hidden_state = hidden_states[0]
        embedding_predictions = (
            top_prediction_tokens_by_position(
                embedding_hidden_state,
                lm_head,
                tokenizer,
                apply_final_norm=True,
            )
            if lm_head is not None and hasattr(lm_head, "decode")
            else None
        )
        embedding_best_prediction_points = (
            project_token_embeddings(
                embedding_weight,
                [token_id for token_id, _text in embedding_predictions],
                tokenizer,
                basis,
            )
            if embedding_weight is not None and embedding_predictions is not None
            else []
        )
        # if lm_head is not None:
        #     embedding_hidden_state = normalize_hidden_for_projection(
        #         embedding_hidden_state,
        #         lm_head,
        #         apply_final_norm=True,
        #     )
        layer_projections.append(
            LayerProjection(
                layer_index=-1,  # -1 标记为嵌入层
                points=project_hidden_states(
                    embedding_hidden_state,
                    token_ids,
                    tokenizer,
                    basis,
                    current_predictions=embedding_predictions,
                ),
                top_prediction_points=[],
                best_prediction_points=embedding_best_prediction_points,
            )
        )

    # 逐层投影
    for layer_index in range(layer_count):
        # a. 获取该层隐藏状态
        hidden_state, apply_final_norm = hidden_state_for_layer(hidden_states, layer_index)
        # b. 应用 final norm
        projection_hidden_state = normalize_hidden_for_projection(
            hidden_state,
            lm_head,
            apply_final_norm=apply_final_norm,
        )
        current_predictions = (
            top_prediction_tokens_by_position(
                hidden_state,
                lm_head,
                tokenizer,
                apply_final_norm=apply_final_norm,
            )
            if lm_head is not None and hasattr(lm_head, "decode")
            else None
        )

        # c. 投影 top-k 预测 token（可选）
        top_prediction_points: list[ProjectedToken] = []
        best_prediction_points: list[ProjectedToken] = []
        if embedding_weight is not None and lm_head is not None and top_k > 0:
            prediction_token_ids = top_prediction_token_ids(
                hidden_state,
                lm_head,
                top_k=top_k,
                apply_final_norm=apply_final_norm,
            )
            top_prediction_points = project_token_embeddings(
                embedding_weight,
                prediction_token_ids,
                tokenizer,
                basis,
            )
        if embedding_weight is not None and current_predictions is not None:
            best_prediction_points = project_token_embeddings(
                embedding_weight,
                [token_id for token_id, _text in current_predictions],
                tokenizer,
                basis,
            )

        # d. 投影该层隐藏状态
        layer_projections.append(
            LayerProjection(
                layer_index=layer_index,
                points=project_hidden_states(
                    projection_hidden_state,
                    token_ids,
                    tokenizer,
                    basis,
                    current_predictions=current_predictions,
                ),
                top_prediction_points=top_prediction_points,
                best_prediction_points=best_prediction_points,
            )
        )
    return layer_projections
