from __future__ import annotations

"""模型结构组件的 3D 布局计算.

核心函数 build_structure_rectangles_for_models 将模型权重矩阵
映射到 3D 空间中的矩形布局，按层和模块类型组织。
"""

from pathlib import Path
from typing import Iterable

from model_visualizer.analysis.types import TensorInfo
from model_visualizer.ui_components.structure.types import ModelStructure, TensorRectangle


def tensor_matrix_shape(shape: tuple[int, ...]) -> tuple[int, int] | None:
    """将张量形状映射为 (行数, 列数)，仅支持 2D 矩阵."""
    if len(shape) == 2:
        return max(1, shape[0]), max(1, shape[1])
    return None


def structure_flow_slot(module: str) -> tuple[str, int] | None:
    """将模块名映射到数据流槽位.

    transformer 的每个层有 4 个主要数据流阶段：
    0. qkv（Q/K/V 投影，并行计算）
    1. o_proj（注意力输出投影）
    2. mlp_up_gate（MLP 的上投影和门控投影）
    3. down_proj（MLP 的下投影）

    返回 (流名称, 流序号)，不匹配则返回 None。
    """

    flow_order = {
        "qkv": 0,
        "o_proj": 1,
        "mlp_up_gate": 2,
        "down_proj": 3,
    }
    if module.endswith("q_proj") or ".q_proj" in module:
        return "qkv", flow_order["qkv"]
    if module.endswith("k_proj") or ".k_proj" in module:
        return "qkv", flow_order["qkv"]
    if module.endswith("v_proj") or ".v_proj" in module:
        return "qkv", flow_order["qkv"]
    if module.endswith("o_proj") or ".o_proj" in module:
        return "o_proj", flow_order["o_proj"]
    if module.endswith("gate_proj") or ".gate_proj" in module:
        return "mlp_up_gate", flow_order["mlp_up_gate"]
    if module.endswith("up_proj") or ".up_proj" in module:
        return "mlp_up_gate", flow_order["mlp_up_gate"]
    if module.endswith("down_proj") or ".down_proj" in module:
        return "down_proj", flow_order["down_proj"]
    return None


def parallel_rank(module: str) -> int:
    """并行组内的顺序号.

    Q/K/V 在 qkv 组中按 0/1/2 排序，Gate/Up 在 mlp_up_gate 组中按 0/1 排序。
    """

    if module.endswith("q_proj") or ".q_proj" in module:
        return 0
    if module.endswith("k_proj") or ".k_proj" in module:
        return 1
    if module.endswith("v_proj") or ".v_proj" in module:
        return 2
    if module.endswith("gate_proj") or ".gate_proj" in module:
        return 0
    if module.endswith("up_proj") or ".up_proj" in module:
        return 1
    return 0


# 候选结构： (TensorInfo, flow_index, flow, rank, rows, cols)
StructureCandidate = tuple[TensorInfo, int, str, int, int, int]


def structure_candidates(infos: Iterable[TensorInfo]) -> list[StructureCandidate]:
    """从张量信息中筛选可渲染为 3D 结构的候选项.

    条件：
    - 属于某个 transformer 层（layer 不为 None）
    - parameter 为 "weight"
    - 是 2D 矩阵
    - 能匹配到数据流槽位
    """

    candidates = []
    for info in infos:
        if info.layer is None or info.parameter != "weight":
            continue
        matrix_shape = tensor_matrix_shape(info.shape)
        flow_slot = structure_flow_slot(info.module)
        if matrix_shape is None or flow_slot is None:
            continue
        rows, cols = matrix_shape
        flow, flow_index = flow_slot
        candidates.append((info, flow_index, flow, parallel_rank(info.module), rows, cols))
    return candidates


def layout_structure_rectangles(
    candidates: list[StructureCandidate],
    *,
    model_name: str,
    model_index: int,
    model_lane: float,
    scale: float,
) -> list[TensorRectangle]:
    """为单个模型计算所有张量矩形在 3D 空间中的位置.

    布局策略：
    - X 轴：layer × 46 + flow_index × 11（层推进 + 流阶段偏移）
    - Y 轴：model_lane（模型车道）+ 组内偏移（并行分支水平排列）
    - Z 轴：始终为 0（所有矩阵在同一水平面上）
    - 宽度/高度 = (cols/rows) × 统一缩放比例

    参数：
        candidates: 结构候选项列表
        model_name: 模型名称
        model_index: 模型在比较列表中的序号
        model_lane: 模型在 Y 轴上的偏移（不同模型不同车道）
        scale: 全局缩放因子（8.0 / 全局最大维度）

    返回：
        TensorRectangle 列表，按 (层, X, Y, 名称) 排序
    """

    layer_stride = 46.0    # 层间 X 轴间距
    flow_stride = 11.0     # 流阶段间 X 轴间距
    group_gap = 0.35       # 组内并行分支在 Y 轴上的间距

    rectangles: list[TensorRectangle] = []

    # 按 (层号, 流索引) 分组，确定每个分支的 Y 偏移
    grouped: dict[tuple[int, int], list[tuple[TensorInfo, int, str, int, int]]] = {}
    for info, flow_index, flow, rank, rows, cols in candidates:
        grouped.setdefault((info.layer or 0, flow_index), []).append((info, rank, flow, rows, cols))

    y_offsets: dict[str, float] = {}
    for items in grouped.values():
        # 按 rank 排序，保证 Q/K/V 顺序一致
        ordered = sorted(items, key=lambda item: item[1])
        widths = [item[4] * scale for item in ordered]
        total_width = sum(widths) + group_gap * max(0, len(widths) - 1)
        # 从组中心向两侧展开
        cursor = -total_width / 2
        for (info, _rank, _flow, _rows, _cols), width in zip(ordered, widths):
            y_offsets[info.name] = cursor + width / 2
            cursor += width + group_gap

    # 为每个候选项创建 TensorRectangle
    for info, flow_index, flow, _rank, rows, cols in candidates:
        width = cols * scale
        height = rows * scale
        if info.layer is None:
            continue
        rectangles.append(
            TensorRectangle(
                name=info.name,
                model_name=model_name,
                model_index=model_index,
                model_lane=model_lane,
                layer=info.layer,
                module=info.module,
                shape=info.shape,
                rows=rows,
                cols=cols,
                flow=flow,
                # X: 每层 46，每流阶段 11
                x=info.layer * layer_stride + flow_index * flow_stride,
                # Y: 模型车道 + 组内偏移
                y=model_lane + y_offsets[info.name],
                z=0.0,
                width=width,
                height=height,
            )
        )

    return sorted(rectangles, key=lambda item: (item.layer, item.x, item.y, item.name))


def build_structure_rectangles_for_models(
    model_infos: Iterable[tuple[str, str | Path, Iterable[TensorInfo]]],
) -> list[ModelStructure]:
    """主布局构建器：为多个模型计算全局统一的 3D 布局.

    流程：
    1. 遍历所有模型的张量，计算全局最大行数/列数作为缩放基准
    2. 计算缩放因子 = 8.0 / 全局最大维度（保证所有立方体不超出可视范围）
    3. 每个模型分配一个 Y 轴"车道"（间距 12，居中对齐）
    4. 调用 layout_structure_rectangles 计算每个模型的矩形位置
    5. 收集为 ModelStructure 列表

    参数：
        model_infos: 元组列表，每项为 (模型名, 模型目录, 张量信息可迭代对象)

    返回：
        ModelStructure 列表
    """

    # 1. 第一遍扫描：收集所有候选，计算全局最大维度
    prepared: list[tuple[str, str, tuple[TensorInfo, ...], list[StructureCandidate]]] = []
    global_max_dim = 1
    for model_name, model_dir, infos_iterable in model_infos:
        infos = tuple(infos_iterable)
        candidates = structure_candidates(infos)
        for _info, _flow_index, _flow, _rank, rows, cols in candidates:
            global_max_dim = max(global_max_dim, rows, cols)
        prepared.append((model_name, str(model_dir), infos, candidates))

    # 2. 计算缩放因子
    scale = 8.0 / global_max_dim

    # 3. 第二遍扫描：为每模型分配车道并计算布局
    lane_stride = 12.0
    lane_start = -lane_stride * (len(prepared) - 1) / 2  # 居中对齐
    structures: list[ModelStructure] = []
    for index, (model_name, model_dir, infos, candidates) in enumerate(prepared):
        lane = lane_start + index * lane_stride
        rectangles = layout_structure_rectangles(
            candidates,
            model_name=model_name,
            model_index=index,
            model_lane=lane,
            scale=scale,
        )
        structures.append(
            ModelStructure(
                model_name=model_name,
                model_dir=model_dir,
                infos=infos,
                rectangles=tuple(rectangles),
                total_params=sum(info.numel for info in infos),
                num_layers=len({info.layer for info in infos if info.layer is not None}),
            )
        )
    return structures


def build_structure_rectangles(infos: Iterable[TensorInfo]) -> list[TensorRectangle]:
    """便捷函数：从张量信息直接构建单个模型的矩形布局."""
    structures = build_structure_rectangles_for_models([("model", "", tuple(infos))])
    return list(structures[0].rectangles) if structures else []
