from __future__ import annotations

"""模型元数据分析子包 —— 扫描 safetensors 文件、直方图统计."""

from model_visualizer.analysis.files import (
    DEFAULT_SAMPLE_SIZE,
    find_model_dirs,
    inspect_safetensors,
    list_safetensors_files,
    load_tensor,
    parse_tensor_name,
    sample_tensor_values,
)
from model_visualizer.analysis.histograms import (
    available_layer_matrix_keys,
    compute_layer_histogram_stack,
    matching_layer_matrix_infos,
)
from model_visualizer.analysis.types import (
    LayerHistogram,
    TensorInfo,
)

__all__ = [
    "DEFAULT_SAMPLE_SIZE",
    "LayerHistogram",
    "TensorInfo",
    "available_layer_matrix_keys",
    "compute_layer_histogram_stack",
    "find_model_dirs",
    "inspect_safetensors",
    "list_safetensors_files",
    "load_tensor",
    "matching_layer_matrix_infos",
    "parse_tensor_name",
    "sample_tensor_values",
]
