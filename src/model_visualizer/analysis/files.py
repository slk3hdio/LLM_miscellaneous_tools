from __future__ import annotations

"""文件系统和 safetensors 辅助工具 —— 扫描、解析、采样模型权重."""

import logging
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from safetensors import safe_open

from model_visualizer.analysis.types import TensorInfo

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_SIZE = 100_000
# 匹配 transformer 层号的 regex，如 "model.layers.0.self_attn.q_proj.weight" → 0
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def find_model_dirs(root: str | Path = "models") -> list[Path]:
    """在指定根目录下查找 Hugging Face 格式的模型目录.

    识别条件：目录内同时存在 config.json 和至少一个 .safetensors 文件。
    """

    root_path = Path(root)
    if not root_path.exists():
        logger.debug("Model root does not exist: %s", root_path)
        return []
    result = sorted(
        path
        for path in root_path.iterdir()
        if path.is_dir()
        and (path / "config.json").exists()
        and any(path.glob("*.safetensors"))
    )
    logger.debug("Found %d model dirs under %s", len(result), root_path)
    return result


def list_safetensors_files(model_dir: str | Path) -> list[Path]:
    """返回模型目录下的 safetensors 文件列表，优先使用合并后的 model.safetensors."""

    model_path = Path(model_dir)
    # 优先返回单个合并文件，加载更快
    preferred = model_path / "model.safetensors"
    if preferred.exists():
        logger.debug("Using consolidated safetensors file: %s", preferred)
        return [preferred]
    files = sorted(model_path.glob("*.safetensors"))
    # 分片文件按 -of- 排序
    shard_files = [path for path in files if "-of-" in path.name]
    result = shard_files or files
    logger.debug("Found %d safetensors files in %s", len(result), model_path)
    return result


def parse_tensor_name(name: str) -> tuple[int | None, str, str]:
    """解析 safetensors 中的张量名称，提取层号、模块路径和参数名.

    示例：
        "model.layers.3.self_attn.q_proj.weight"
        → layer=3, module="self_attn.q_proj", parameter="weight"

    步骤：
    1. 使用 LAYER_RE 正则从名称中提取层号
    2. 参数名取最后一个点分隔的片段（如 "weight"）
    3. 模块路径取 layer 之后、parameter 之前的部分
    """

    parts = name.split(".")
    match = LAYER_RE.search(name)
    layer = int(match.group(1)) if match else None

    # 参数名：最后一个 "." 之后的片段
    parameter = parts[-1] if parts else name
    if layer is None:
        module = ".".join(parts[:-1]) if len(parts) > 1 else name
        return layer, module, parameter

    # 模块名：layer 之后、parameter 之前的部分
    layer_token = str(layer)
    try:
        layer_index = parts.index(layer_token)
    except ValueError:
        module = ".".join(parts[:-1]) if len(parts) > 1 else name
    else:
        module_parts = parts[layer_index + 1 : -1]
        module = ".".join(module_parts) if module_parts else "layer"
    return layer, module, parameter


def inspect_safetensors(files: Iterable[str | Path]) -> list[TensorInfo]:
    """扫描所有 safetensors 文件，提取每个张量的元数据.

    流程：
    1. 逐个文件用 safe_open 打开（只读元数据 + tensor 属性，不加载全部数据）
    2. 对每个张量名调用 parse_tensor_name 提取层号/模块/参数
    3. 收集到 TensorInfo 列表并按名称排序
    """

    infos: list[TensorInfo] = []
    for file_path in files:
        path = Path(file_path)
        logger.debug("Inspecting safetensors file: %s", path.name)
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                layer, module, parameter = parse_tensor_name(name)
                infos.append(
                    TensorInfo(
                        name=name,
                        shape=tuple(int(dim) for dim in tensor.shape),
                        dtype=str(tensor.dtype).replace("torch.", ""),
                        numel=int(tensor.numel()),
                        layer=layer,
                        module=module,
                        parameter=parameter,
                        file=str(path),
                    )
                )
    logger.info("Inspected %d tensors across %d safetensors file(s)", len(infos), len(list(files)))
    return sorted(infos, key=lambda item: item.name)


def _to_float_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """将张量转为 CPU float32 并展平，确保 bf16/fp16 等精度统计行为一致."""

    return tensor.detach().to(device="cpu", dtype=torch.float32).reshape(-1)


def sample_tensor_values(
    tensor: torch.Tensor,
    max_values: int = DEFAULT_SAMPLE_SIZE,
    *,
    seed: int = 0,
) -> np.ndarray:
    """从张量中确定性采样（最多 max_values 个值）用于绘图.

    步骤：
    1. 将张量转到 CPU float32 并展平
    2. 若元素数 ≤ max_values，直接返回全部
    3. 否则使用固定种子的 torch.Generator 随机排列后取前 max_values 个
       （固定种子保证同一张量每次采样结果一致）
    """

    flat = _to_float_tensor(tensor)
    if flat.numel() <= max_values:
        return flat.numpy()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randperm(flat.numel(), generator=generator)[:max_values]
    return flat[indices].numpy()


def load_tensor(file_path: str | Path, tensor_name: str) -> torch.Tensor:
    """按需从 safetensors 文件中加载单个张量."""

    with safe_open(file_path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(tensor_name)
