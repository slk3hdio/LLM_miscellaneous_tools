from __future__ import annotations

"""模型分析和可视化的共享数据类型."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TensorInfo:
    """safetensors 文件中单个张量的静态元数据."""

    name: str               # 张量全名，如 "model.layers.0.self_attn.q_proj.weight"
    shape: tuple[int, ...]  # 形状
    dtype: str              # 数据类型，如 "float32"
    numel: int              # 元素总数
    layer: int | None       # 所属 transformer 层号，None 表示非层参数
    module: str             # 模块名，如 "self_attn.q_proj"
    parameter: str          # 参数名，通常为 "weight"
    file: str               # 所在的 safetensors 文件路径

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["shape"] = "x".join(str(dim) for dim in self.shape)
        return data


@dataclass(frozen=True)
class LayerHistogram:
    """某一层指定权重矩阵的直方图数据."""

    layer: int                     # 层号
    tensor_name: str               # 张量全名
    bin_centers: tuple[float, ...] # 箱中心值（X 轴坐标）
    values: tuple[float, ...]      # 每个箱的密度/计数（Y 轴值）

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
