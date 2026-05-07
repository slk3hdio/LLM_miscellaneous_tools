from __future__ import annotations

"""模型结构组件的内部数据类型."""

from dataclasses import asdict, dataclass

from model_visualizer.analysis.types import TensorInfo


@dataclass(frozen=True)
class TensorRectangle:
    """3D 结构视图中单个张量矩形的布局元数据.

    每个属性含义：
    - name/shape/rows/cols: 张量标识和尺寸
    - module/flow: 所属模块和数据流分组（qkv、o_proj、mlp_up_gate、down_proj）
    - x/y/z + width/height: 在 3D 空间中的位置和尺寸
    - model_name/model_index/model_lane: 所属模型及在 Y 轴上的"车道"
    """

    name: str
    model_name: str
    model_index: int        # 模型在列表中的序号
    model_lane: float       # 模型在 Y 轴上的偏移（不同模型不同车道）
    layer: int
    module: str
    shape: tuple[int, ...]
    rows: int               # 矩阵行数（缩放后为高度）
    cols: int               # 矩阵列数（缩放后为宽度）
    flow: str               # 数据流分组名
    x: float                # 3D 空间中的 X 坐标（层 + 流偏移）
    y: float                # 3D 空间中的 Y 坐标（模型车道 + 组内偏移）
    z: float                # 3D 空间中的 Z 坐标（行方向，始终为 0）
    width: float            # 立方体宽度（列数 × 缩放比例）
    height: float           # 立方体高度（行数 × 缩放比例）

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["shape"] = "x".join(str(dim) for dim in self.shape)
        return data


@dataclass(frozen=True)
class ModelStructure:
    """单个模型在结构比较视图中的完整元数据和布局."""

    model_name: str                         # 模型名称
    model_dir: str                          # 模型目录路径
    infos: tuple[TensorInfo, ...]           # 所有张量的元数据
    rectangles: tuple[TensorRectangle, ...] # 3D 矩形布局
    total_params: int                       # 总参数量
    num_layers: int                         # transformer 层数

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "model_dir": self.model_dir,
            "parameters": self.total_params,
            "layers": self.num_layers,
            "tensors": len(self.infos),
        }
