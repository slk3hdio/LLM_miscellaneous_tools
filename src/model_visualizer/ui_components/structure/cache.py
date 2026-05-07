from __future__ import annotations

"""模型结构组件的 Streamlit 缓存层."""

from pathlib import Path

import streamlit as st

from model_visualizer.analysis.files import (
    find_model_dirs,
    inspect_safetensors,
    list_safetensors_files,
)
from model_visualizer.ui_components.structure.layout import build_structure_rectangles_for_models
from model_visualizer.ui_components.structure.types import ModelStructure, TensorRectangle


@st.cache_data(show_spinner=False)
def cached_model_dirs(root: str) -> list[str]:
    """缓存模型目录列表，供侧边栏多选使用."""

    return [str(path) for path in find_model_dirs(root)]


@st.cache_data(show_spinner=True)
def cached_structure_metadata(
    model_dirs: tuple[str, ...],
) -> tuple[tuple[ModelStructure, ...], tuple[TensorRectangle, ...]]:
    """缓存模型元数据和全局缩放的 3D 结构矩形.

    步骤：
    1. 对每个模型目录：列出 safetensors 文件、扫描元数据
    2. 调用 build_structure_rectangles_for_models 计算 3D 布局
    3. 展开所有矩形用于渲染
    """

    model_payload = []
    for model_dir in model_dirs:
        files = list_safetensors_files(model_dir)
        infos = inspect_safetensors(files)
        model_payload.append((Path(model_dir).name, model_dir, tuple(infos)))

    structures = tuple(build_structure_rectangles_for_models(model_payload))
    rectangles = tuple(
        rectangle
        for structure in structures
        for rectangle in structure.rectangles
    )
    return structures, rectangles
