from __future__ import annotations

"""Streamlit 入口脚本 —— 启动模型参数可视化工具."""

import sys
from pathlib import Path
from transformers.utils import logging as hf_logging
import logging
import streamlit as st

# Streamlit 将此文件作为脚本运行，需要手动将 src 目录加入 sys.path
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_visualizer.ui_components.embedding_projection.component import EmbeddingProjectionComponent
from model_visualizer.ui_components.export.component import DemoExportComponent
from model_visualizer.ui_components.inference.component import InferenceStepperComponent
from model_visualizer.ui_components.structure.component import ModelStructureComponent

logging.basicConfig(
    format="[%(asctime)s]%(name)s %(levelname)s: %(message)s",
    level = logging.INFO
)
logger = logging.getLogger(__name__)

def render_app() -> None:
    """渲染完整的 Streamlit 应用.

    负责整个应用的生命周期：
    1. 侧边栏：选择模型目录（需包含 config.json 和 .safetensors 文件）
    2. 主体：依次渲染模型结构组件、推理步进组件和嵌入投影组件
    """

    logger.info("Starting Model Parameter Visualizer")
    st.title("Model Parameter Visualizer")

    # ---- 侧边栏：模型目录选择 ----
    model_root = st.sidebar.text_input("Model root", value="models")
    model_dirs = ModelStructureComponent.model_dirs(model_root)
    if not model_dirs:
        logger.warning("No model directories found under root=%s", model_root)
        st.warning("No model directory with config.json and .safetensors was found.")
        return
    logger.info("Found %d model directories under %s", len(model_dirs), model_root)

    # 默认选中包含 "qwen" 的目录，便于快速加载
    default_dirs = [model_dir for model_dir in model_dirs if "qwen" in model_dir.lower()][:1]
    if not default_dirs:
        default_dirs = model_dirs[:1]
    selected_model_dirs = st.sidebar.multiselect(
        "Model directories",
        model_dirs,
        default=default_dirs,
    )
    if not selected_model_dirs:
        st.warning("Select at least one model directory.")
        return

    # ---- 组件 1：模型结构（元数据 + 3D 张量布局） ----
    logger.info("Loading model structures for: %s", selected_model_dirs)
    structure_component = ModelStructureComponent(tuple(selected_model_dirs))
    structures = structure_component.render()
    if not structures:
        logger.warning("No model structures could be loaded")
        st.warning("No model structure could be loaded.")
        return

    # ---- 组件 2：推理步进器（逐步查看各层注意力、预测） ----
    trace_view = InferenceStepperComponent(structures[0]).render()
    # ---- 组件 3：嵌入投影（PCA 降维可视化隐藏状态轨迹） ----
    EmbeddingProjectionComponent(model_dir=structures[0].model_dir, trace_view=trace_view).render()
    # ---- 组件 4：导出演示静态文件 ----
    DemoExportComponent(structure=structures[0], trace_view=trace_view).render()


def main() -> None:
    """Streamlit 脚本入口."""

    st.set_page_config(page_title="Model Parameter Visualizer", layout="wide")
    hf_logging.set_verbosity_error()
    render_app()


if __name__ == "__main__":
    main()
