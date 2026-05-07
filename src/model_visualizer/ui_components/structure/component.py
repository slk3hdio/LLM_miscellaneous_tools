from __future__ import annotations

"""模型元数据和 3D 张量结构的 Streamlit 组件."""

import streamlit as st

from model_visualizer.ui_components.structure.figures import format_parameters, tensor_rectangle_figure
from model_visualizer.ui_components.structure.cache import (
    cached_model_dirs,
    cached_structure_metadata,
)
from model_visualizer.ui_components.structure.types import ModelStructure, TensorRectangle


class ModelStructureComponent:
    """渲染模型元数据摘要和 3D 张量结构视图.

    负责：
    - 显示模型参数数量、张量数、层数、数据类型等摘要信息
    - 渲染 3D 立方体结构图展示各层权重矩阵的布局
    """

    def __init__(self, selected_model_dirs: tuple[str, ...]):
        self.selected_model_dirs = selected_model_dirs

    @staticmethod
    def model_dirs(root: str) -> list[str]:
        """（静态方法）返回根目录下的模型目录列表."""
        return cached_model_dirs(root)

    def load(self) -> tuple[tuple[ModelStructure, ...], tuple[TensorRectangle, ...]]:
        """加载缓存的模型结构和 3D 矩形布局."""
        return cached_structure_metadata(self.selected_model_dirs)

    def render(self) -> tuple[ModelStructure, ...]:
        """渲染模型摘要表格和 3D 结构图."""
        structures, rectangles = self.load()
        self.render_summary(structures)
        self.render_3d_structure(rectangles)
        return structures

    @staticmethod
    def render_summary(structures: tuple[ModelStructure, ...]) -> None:
        """渲染模型参数摘要 DataFrame."""
        rows = []
        for structure in structures:
            dtypes = ", ".join(sorted({info.dtype for info in structure.infos}))
            rows.append(
                {
                    "model": structure.model_name,
                    "parameters": format_parameters(structure.total_params),
                    "tensors": len(structure.infos),
                    "layers": structure.num_layers,
                    "dtype": dtypes,
                }
            )
        st.dataframe(rows, width='stretch', hide_index=True)

    @staticmethod
    def render_3d_structure(rectangles: tuple[TensorRectangle, ...]) -> None:
        """渲染 3D 张量立方体结构的 Plotly 图表."""
        st.subheader("3D Tensor Structure")
        st.plotly_chart(tensor_rectangle_figure(list(rectangles)), width='stretch')
