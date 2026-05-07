from __future__ import annotations

"""嵌入投影 Streamlit 组件 —— 支持 PCA / UMAP 两种降维方式."""

import logging
from pathlib import Path

import streamlit as st

from model_visualizer.ui_components.embedding_projection.figures import animated_token_projection_figure
from model_visualizer.ui_components.embedding_projection.local_umap import compute_local_umap_projection
from model_visualizer.ui_components.embedding_projection.projection import (
    DEFAULT_PROJECTION_DIMENSIONS,
    PROJECTION_DIMENSION_OPTIONS,
    load_projection_basis,
    model_embedding_weight,
    project_hidden_state_layers,
    project_token_embeddings,
    projection_output_path,
)
from model_visualizer.ui_components.embedding_projection.types import (
    LayerProjection,
    LocalProjection,
    ProjectionBasis,
)
from model_visualizer.ui_components.inference.types import InferenceTraceView


logger = logging.getLogger(__name__)
# 三种投影模式：UMAP（余弦距离）、UMAP（点积距离）、PCA（预计算基）
PROJECTION_MODES = (
    "UMAP",
    "Dot-product UMAP",
    "Exp-dot UMAP",
    "Dot-product t-SNE",
    "Exp-dot t-SNE",
    "PCA",
)


@st.cache_data(show_spinner=False)
def _cached_projection_basis(path: str, modified_time: float) -> ProjectionBasis:
    """缓存 PCA 投影基（文件修改时间变化时自动失效）."""
    del modified_time
    return load_projection_basis(path)


def _trace_layer_count(trace_view: InferenceTraceView) -> int:
    """计算当前追踪中可用的 transformer 层数."""
    hidden_states = trace_view.manager.hidden_states[trace_view.generation_step]
    return max(0, min(trace_view.num_layers, len(hidden_states) - 1))


def _clamp_layer_index(layer_index: int, layer_count: int) -> int:
    """将层索引限制在有效范围内."""
    if layer_count < 1:
        raise ValueError("Projection needs at least one layer of hidden states.")
    return max(0, min(int(layer_index), layer_count - 1))


class EmbeddingProjectionComponent:
    """将追踪隐藏状态投影到低维空间的 Streamlit 组件.

    支持三种投影模式：
    - **PCA**: 使用预计算的全词汇嵌入 PCA 基，投影是线性的，速度最快
    - **UMAP**: 使用余弦距离的局部 UMAP，保留流形结构
    - **Dot-product UMAP**: 使用点积距离的 UMAP，强调语义相似性

    通过动画帧展示每层隐藏状态在投影空间中的移动轨迹。
    """

    def __init__(
        self,
        *,
        model_dir: str,
        trace_view: InferenceTraceView | None,
        state_prefix: str = "embedding_projection",
    ):
        self.model_dir = model_dir
        self.trace_view = trace_view
        self.state_prefix = state_prefix

    def key(self, name: str) -> str:
        """生成带前缀的会话键."""
        return f"{self.state_prefix}_{name}"

    def render(self) -> None:
        """渲染组件."""
        self._render_fragment()

    @st.fragment
    def _render_fragment(self) -> None:
        """主渲染片段.

        渲染流程：
        1. 检查追踪数据是否就绪
        2. 渲染投影模式选择器（UMAP / Dot-product UMAP / PCA）
        3. 渲染维度选择器（2D / 3D）
        4. 渲染 top-k 预测数选择器
        5. 提取模型嵌入权重
        6. 根据模式分叉：
           - UMAP 模式：动态计算局部 UMAP 投影
           - PCA 模式：加载预计算 PCA 基 → 投影隐藏状态
        7. 渲染动画图
        8. 异常处理：日志记录 + UI 错误提示
        """

        st.subheader("Token State Projection")

        # 1. 检查追踪数据
        if self.trace_view is None:
            st.info("Click Next step to generate trace vectors.")
            return

        try:
            # 2-4. 渲染控制面板
            mode = self._render_projection_mode_control()
            dimensions = self._render_dimension_control()
            projection_top_k = self._render_projection_top_k_control()

            layer_count = _trace_layer_count(self.trace_view)
            embedding_weight = model_embedding_weight(self.trace_view.model)

            # 5-6. 根据模式计算投影
            if mode in {
                "UMAP",
                "Dot-product UMAP",
                "Exp-dot UMAP",
                "Dot-product t-SNE",
                "Exp-dot t-SNE",
            }:
                # UMAP 模式：动态拟合，不需要预计算文件
                basis = None
                local_projection = self._project_umap(
                    layer_count,
                    embedding_weight,
                    projection_top_k,
                    dimensions,
                    metric={
                        "UMAP": "cosine",
                        "Dot-product UMAP": "dot_product",
                        "Exp-dot UMAP": "exp_dot_product",
                        "Dot-product t-SNE": "dot_product_tsne",
                        "Exp-dot t-SNE": "exp_dot_product_tsne",
                    }[mode],
                )
                initial_points = local_projection.initial_points
                final_prediction_points = local_projection.final_prediction_points
                layers = local_projection.layers
            else:
                # PCA 模式：需要预计算的 .npz 文件
                path = projection_output_path(self.model_dir)
                if not path.exists():
                    self._render_missing_projection(path)
                    return
                basis = _cached_projection_basis(str(path), path.stat().st_mtime)
                if basis.components.shape[1] < dimensions:
                    raise ValueError(
                        f"Projection file is {basis.components.shape[1]}D. "
                        f"Recompute it for {dimensions}D with: "
                        f"{self._precompute_command(dimensions)}"
                    )
                initial_points = self._project_initial_points(basis, path, embedding_weight)
                layers = self._project_pca_layers(
                    basis,
                    path,
                    layer_count,
                    embedding_weight,
                    projection_top_k,
                )
                final_prediction_points = []

            initial_layer_index = -1  # 初始显示嵌入层
        except Exception as exc:
            logger.error("Token projection failed: %s", exc, exc_info=True)
            st.error(f"Token projection failed: {exc}")
            return

        # 7. 元数据标题
        st.caption(
            f"{mode} | generation step {self.trace_view.generation_step} | "
            f"layers 0-{layer_count - 1} | projection top-k {projection_top_k}"
        )
        # 8. 渲染动画图
        st.plotly_chart(
            animated_token_projection_figure(
                layers,
                basis,
                initial_layer_index=initial_layer_index,
                dimensions=dimensions,
                initial_points=initial_points,
                final_prediction_points=final_prediction_points,
                axis_title_prefix="t-SNE" if "t-SNE" in mode else "UMAP",
            ),
            width="stretch",
            key=self.key("scatter"),
        )

    def _render_missing_projection(self, path: Path) -> None:
        """PCA 文件不存在时的提示和预计算命令."""
        logger.warning("PCA projection file not found: %s", path)
        st.warning(f"Projection file not found: {path}")
        st.code(self._precompute_command(DEFAULT_PROJECTION_DIMENSIONS), language="powershell")

    def _precompute_command(self, dimensions: int) -> str:
        """生成预计算 PCA 投影的 CLI 命令."""
        quoted_model_dir = self.model_dir.replace("'", "''")
        return (
            "uv run python scripts/precompute_embedding_projection.py "
            f"--model-dir '{quoted_model_dir}' --dimensions {dimensions} --overwrite"
        )

    def _render_projection_mode_control(self) -> str:
        """渲染投影模式选择器（UMAP / Dot-product UMAP / PCA 单选按钮）.

        默认值从会话读取，切换模式后自动触发重新渲染。
        """

        stored_mode = st.session_state.get(self.key("mode"), "Dot-product UMAP")
        if stored_mode not in PROJECTION_MODES:
            stored_mode = "UMAP"
        return st.radio(
            "Projection mode",
            options=list(PROJECTION_MODES),
            index=list(PROJECTION_MODES).index(stored_mode),
            horizontal=True,
            key=self.key("mode"),
        )

    def _render_dimension_control(self) -> int:
        """渲染维度选择器（2D / 3D 单选按钮）."""

        stored_dimensions = int(st.session_state.get(self.key("dimensions"), DEFAULT_PROJECTION_DIMENSIONS))
        if stored_dimensions not in PROJECTION_DIMENSION_OPTIONS:
            stored_dimensions = DEFAULT_PROJECTION_DIMENSIONS
        return int(
            st.radio(
                "Projection dimensions",
                options=list(PROJECTION_DIMENSION_OPTIONS),
                index=list(PROJECTION_DIMENSION_OPTIONS).index(stored_dimensions),
                horizontal=True,
                format_func=lambda value: f"{value}D",
                key=self.key("dimensions"),
            )
        )

    def _render_projection_top_k_control(self) -> int:
        """渲染投影 top-k 预测数选择器（1-20 数字输入）."""

        return int(
            st.number_input(
                "Projection Top K",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                key=self.key("top_k"),
            )
        )

    def _project_pca_layers(
        self,
        basis: ProjectionBasis,
        path: Path,
        layer_count: int,
        embedding_weight,
        top_k: int,
    ) -> list[LayerProjection]:
        """使用 PCA 基投影所有层的隐藏状态（带会话级缓存）.

        缓存键包含：文件路径、修改时间、manager 身份、generation step、
        隐藏状态身份、token IDs、层数、top_k、嵌入权重身份。
        任意缓存键变化触发重新投影。
        """

        hidden_states = self.trace_view.manager.hidden_states[self.trace_view.generation_step]
        cache_key = (
            str(path),
            path.stat().st_mtime,
            id(self.trace_view.manager),
            self.trace_view.generation_step,
            id(hidden_states),
            tuple(self.trace_view.trace_token_ids),
            layer_count,
            top_k,
            id(embedding_weight),
            "pca_normalized_trajectory_v2",
        )
        if st.session_state.get(self.key("layers_cache_key")) == cache_key:
            cached_layers = st.session_state.get(self.key("layers_cache"))
            if cached_layers is not None:
                logger.debug("Using cached PCA layer projections (%d layers)", len(cached_layers))
                return cached_layers

        logger.debug("Computing PCA layer projections (cache miss) for %d layers", layer_count)
        layers = project_hidden_state_layers(
            hidden_states,
            self.trace_view.trace_token_ids,
            self.trace_view.tokenizer,
            basis,
            layer_count=layer_count,
            embedding_weight=embedding_weight,
            lm_head=self.trace_view.lm_head,
            top_k=top_k,
            include_embedding_layer=True,
        )
        st.session_state[self.key("layers_cache_key")] = cache_key
        st.session_state[self.key("layers_cache")] = layers
        return layers

    def _project_initial_points(
        self,
        basis: ProjectionBasis,
        path: Path,
        embedding_weight,
    ):
        """使用 PCA 基投影初始 token 嵌入点（带会话级缓存）.

        缓存键：文件路径 + 修改时间 + token IDs + 嵌入权重身份。
        """

        cache_key = (
            str(path),
            path.stat().st_mtime,
            tuple(self.trace_view.trace_token_ids),
            id(embedding_weight),
        )
        if st.session_state.get(self.key("initial_cache_key")) == cache_key:
            cached_points = st.session_state.get(self.key("initial_cache"))
            if cached_points is not None:
                return cached_points

        points = project_token_embeddings(
            embedding_weight,
            self.trace_view.trace_token_ids,
            self.trace_view.tokenizer,
            basis,
        )
        st.session_state[self.key("initial_cache_key")] = cache_key
        st.session_state[self.key("initial_cache")] = points
        return points

    def _project_umap(
        self,
        layer_count: int,
        embedding_weight,
        top_k: int,
        dimensions: int,
        metric: str,
    ) -> LocalProjection:
        """动态计算局部 UMAP 投影（带会话级缓存）.

        UMAP 不需要预计算文件，但计算成本高于 PCA。
        缓存键包含：manager 身份、generation step、隐藏状态身份、
        token IDs、层数、嵌入权重、top_k、维度、metric。

        缓存策略与 PCA 相同：键匹配时直接返回缓存结果。
        """

        hidden_states = self.trace_view.manager.hidden_states[self.trace_view.generation_step]
        cache_key = (
            id(self.trace_view.manager),
            self.trace_view.generation_step,
            id(hidden_states),
            tuple(self.trace_view.trace_token_ids),
            layer_count,
            id(embedding_weight),
            top_k,
            dimensions,
            metric,
            "local_umap_v4",
        )
        if st.session_state.get(self.key("umap_cache_key")) == cache_key:
            cached_projection = st.session_state.get(self.key("umap_cache"))
            if cached_projection is not None:
                logger.debug("Using cached UMAP projection (%d layers)", len(cached_projection.layers))
                return cached_projection

        logger.info(
            "Computing local UMAP projection: %d layers, metric=%s, dims=%d, top_k=%d",
            layer_count, metric, dimensions, top_k,
        )
        projection = compute_local_umap_projection(
            hidden_states,
            self.trace_view.trace_token_ids,
            self.trace_view.tokenizer,
            embedding_weight,
            self.trace_view.lm_head,
            layer_count=layer_count,
            top_k=top_k,
            dimensions=dimensions,
            metric=metric,
        )
        st.session_state[self.key("umap_cache_key")] = cache_key
        st.session_state[self.key("umap_cache")] = projection
        return projection
