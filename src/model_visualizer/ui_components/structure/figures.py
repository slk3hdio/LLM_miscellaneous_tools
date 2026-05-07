from __future__ import annotations

"""模型结构组件的 Plotly 3D 图形构建器.

核心函数 tensor_rectangle_figure 将模型中的权重矩阵渲染为 3D 立方体，
按层和模块类型排列，附带数据流贝塞尔曲线和方向箭头。
"""

import plotly.graph_objects as go

from model_visualizer.ui_components.structure.types import TensorRectangle

# 同一层内四个数据流阶段在 X 轴上的偏移位置
# Q/K/V 在偏移 0，O 投影在偏移 11，Gate/Up 在偏移 22，Down 在偏移 33
FLOW_TICKS = {
    0.0: "Q/K/V",
    11.0: "O",
    22.0: "Gate/Up",
    33.0: "Down",
}
LAYER_STRIDE = 46.0       # 每层在 X 轴上的跨度
MATRIX_THICKNESS = 4       # 立方体的厚度（沿 X 轴）
# 立方体 12 个三角面的顶点索引定义（每面由 i/j/k 中的两个三角形组成）
CUBOID_FACES = {
    "i": [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3],
    "j": [1, 2, 6, 5, 4, 7, 5, 2, 6, 3, 7, 0],
    "k": [2, 3, 5, 7, 7, 1, 6, 6, 3, 7, 4, 4],
}


def format_parameters(value: int) -> str:
    """人类可读的参数数量格式化：B（十亿）/ M（百万）/ K（千）."""

    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return str(value)


def _is_attention_matrix(rectangle: TensorRectangle) -> bool:
    """判断张量是否属于注意力模块（q_proj/k_proj/v_proj/o_proj）."""
    return any(
        marker in rectangle.module
        for marker in ("q_proj", "k_proj", "v_proj", "o_proj")
    )


def _cuboid_vertices(rectangle: TensorRectangle) -> tuple[list[float], list[float], list[float]]:
    """计算矩形在 3D 空间中 8 个顶点的坐标.

    以 (x, y, z) 为中心，沿 X 轴厚度为 MATRIX_THICKNESS，
    沿 Y 轴宽度为 rectangle.width，沿 Z 轴高度为 rectangle.height。
    返回 (xs, ys, zs) 三个 8 元素列表。
    """

    half_depth = MATRIX_THICKNESS / 2
    half_width = rectangle.width / 2
    half_height = rectangle.height / 2
    return (
        # X：前后各 4 个顶点
        [
            rectangle.x - half_depth,
            rectangle.x - half_depth,
            rectangle.x - half_depth,
            rectangle.x - half_depth,
            rectangle.x + half_depth,
            rectangle.x + half_depth,
            rectangle.x + half_depth,
            rectangle.x + half_depth,
        ],
        # Y：左右各 4 个顶点
        [
            rectangle.y - half_width,
            rectangle.y + half_width,
            rectangle.y + half_width,
            rectangle.y - half_width,
            rectangle.y - half_width,
            rectangle.y + half_width,
            rectangle.y + half_width,
            rectangle.y - half_width,
        ],
        # Z：上下各 4 个顶点
        [
            rectangle.z - half_height,
            rectangle.z - half_height,
            rectangle.z + half_height,
            rectangle.z + half_height,
            rectangle.z - half_height,
            rectangle.z - half_height,
            rectangle.z + half_height,
            rectangle.z + half_height,
        ],
    )


def _rectangle_color(rectangle: TensorRectangle) -> str:
    """根据模块类型返回对应的颜色.

    颜色映射：
    - q_proj → 蓝色, k_proj → 绿色, v_proj → 紫色, o_proj → 青色
    - gate_proj → 橙色, up_proj → 红色, down_proj → 黄色
    - layernorm → 灰色, 其他 → 深灰
    """

    if rectangle.module.endswith("q_proj") or ".q_proj" in rectangle.module:
        return "#2f80ed"
    if rectangle.module.endswith("k_proj") or ".k_proj" in rectangle.module:
        return "#27ae60"
    if rectangle.module.endswith("v_proj") or ".v_proj" in rectangle.module:
        return "#9b51e0"
    if rectangle.module.endswith("o_proj") or ".o_proj" in rectangle.module:
        return "#56ccf2"
    if rectangle.module.endswith("gate_proj") or ".gate_proj" in rectangle.module:
        return "#f2994a"
    if rectangle.module.endswith("up_proj") or ".up_proj" in rectangle.module:
        return "#eb5757"
    if rectangle.module.endswith("down_proj") or ".down_proj" in rectangle.module:
        return "#f2c94c"
    if "layernorm" in rectangle.module:
        return "#828282"
    return "#4f4f4f"


def _hover_text(rectangle: TensorRectangle) -> str:
    """生成悬停提示文本，显示张量的详细元数据."""

    return "<br>".join(
        [
            f"model: {rectangle.model_name}",
            rectangle.name,
            f"layer: {rectangle.layer}",
            f"flow: {rectangle.flow}",
            f"module: {rectangle.module}",
            f"shape: {'x'.join(str(dim) for dim in rectangle.shape)}",
            f"rows x cols: {rectangle.rows} x {rectangle.cols}",
        ]
    )


def _quadratic_points(
    start: tuple[float, float, float],
    control: tuple[float, float, float],
    end: tuple[float, float, float],
    steps: int = 36,
) -> tuple[list[float], list[float], list[float]]:
    """生成二次贝塞尔曲线的采样点序列.

    用于绘制数据流的分叉和汇合曲线：
    B(t) = (1-t)²·P0 + 2(1-t)t·P1 + t²·P2

    参数：
        start: 起点 P0
        control: 控制点 P1（决定弯曲方向）
        end: 终点 P2
        steps: 采样点数，默认 36
    """

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for index in range(steps + 1):
        t = index / steps
        inv = 1.0 - t
        xs.append(inv * inv * start[0] + 2 * inv * t * control[0] + t * t * end[0])
        ys.append(inv * inv * start[1] + 2 * inv * t * control[1] + t * t * end[1])
        zs.append(inv * inv * start[2] + 2 * inv * t * control[2] + t * t * end[2])
    return xs, ys, zs


def _add_mesh_traces(fig: go.Figure, rectangles: list[TensorRectangle]) -> None:
    """向图形中添加 3D 立方体网格轨迹.

    实现方式：
    1. 按颜色分组所有矩形
    2. 每组生成一个 Mesh3d 轨迹（同一颜色的立方体合并为单个 mesh，减少 GPU 绘制调用）
    3. 每组的每个矩形贡献 8 个顶点和 12 个三角面（面索引累加偏移量）

    注意力相关矩阵的透明度设为 0.8，非注意力矩阵为 0.4。
    """

    # 1. 按颜色分组
    by_color: dict[str, list[TensorRectangle]] = {}
    for rectangle in rectangles:
        by_color.setdefault(_rectangle_color(rectangle), []).append(rectangle)

    for color, color_rectangles in by_color.items():
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        i_values: list[int] = []
        j_values: list[int] = []
        k_values: list[int] = []
        for rectangle in color_rectangles:
            x, y, z = _cuboid_vertices(rectangle)
            offset = len(xs)  # 当前矩形在顶点数组中的起始索引
            xs.extend(x)
            ys.extend(y)
            zs.extend(z)
            # 面索引加偏移
            i_values.extend(offset + index for index in CUBOID_FACES["i"])
            j_values.extend(offset + index for index in CUBOID_FACES["j"])
            k_values.extend(offset + index for index in CUBOID_FACES["k"])

        fig.add_trace(
            go.Mesh3d(
                x=xs,
                y=ys,
                z=zs,
                i=i_values,
                j=j_values,
                k=k_values,
                color=color,
                opacity=0.8 if _is_attention_matrix(color_rectangles[0]) else 0.4,
                flatshading=True,          # 平面着色，棱角分明
                hoverinfo="skip",           # 不显示悬停（由 marker trace 处理）
                name="matrix group",
                showscale=False,
            )
        )


def _add_hover_marker_trace(fig: go.Figure, rectangles: list[TensorRectangle]) -> None:
    """添加半透明标记点以承载悬停提示.

    每个矩形的中心放置一个几乎透明的标记点，用户鼠标悬停时显示详细信息。
    注意力矩阵的标记稍大（16px vs 12px），方便点击。
    """

    fig.add_trace(
        go.Scatter3d(
            x=[rectangle.x for rectangle in rectangles],
            y=[rectangle.y for rectangle in rectangles],
            z=[rectangle.z for rectangle in rectangles],
            mode="markers",
            marker={
                "size": [16 if _is_attention_matrix(rectangle) else 12 for rectangle in rectangles],
                "color": [_rectangle_color(rectangle) for rectangle in rectangles],
                "opacity": 0.08,  # 几乎完全透明，仅用于响应悬停事件
            },
            hovertemplate=[
                _hover_text(rectangle) + "<extra></extra>" for rectangle in rectangles
            ],
            name="matrix hover targets",
            showlegend=False,
        )
    )


def _add_data_flow_traces(fig: go.Figure, rectangles: list[TensorRectangle]) -> None:
    """添加数据流路径的视觉效果 —— 连接线、分叉贝塞尔曲线和方向锥体.

    分三个层次：
    1. 主干线（水平线）：连接同一模型内相邻层的数据流路径（沿 X 轴）
    2. 分叉线（贝塞尔曲线）：同一组内 Q/K/V 或 Gate/Up 等并行分支的分叉与汇合
    3. 方向锥体：在最后一层末端显示数据流方向

    实现细节：
    - 先将所有矩形按 (层号, 流分组, X坐标, 模型名) 分组
    - 主干线画在 model_lane 的 Z=0 平面上
    - 分叉线从主干线的节点出发，用二次贝塞尔曲线偏移到各子模块的中心，再回到汇合点
    """

    # 按层+流+X+模型分组
    grouped: dict[tuple[int, str, float, str], list[TensorRectangle]] = {}
    for rectangle in rectangles:
        grouped.setdefault(
            (rectangle.layer, rectangle.flow, rectangle.x, rectangle.model_name),
            [],
        ).append(rectangle)

    groups = sorted(
        grouped.values(),
        key=lambda items: (items[0].model_index, items[0].layer, items[0].x),
    )
    groups_by_model: dict[str, list[list[TensorRectangle]]] = {}
    for items in groups:
        groups_by_model.setdefault(items[0].model_name, []).append(items)

    # 1. 主干线：每个模型画一条沿 X 轴的线
    for model_name, model_groups in groups_by_model.items():
        fig.add_trace(
            go.Scatter3d(
                x=[items[0].x for items in model_groups],
                y=[items[0].model_lane for items in model_groups],
                z=[0.0 for _items in model_groups],
                mode="lines",
                line={"color": "#FFF700", "width": 6},
                hoverinfo="skip",
                name=f"{model_name} data flow",
                showlegend=False,
            )
        )

    # 2. 分叉线：为每组内多个并行分支绘制贝塞尔曲线
    split_x: list[float | None] = []
    split_y: list[float | None] = []
    split_z: list[float | None] = []
    for items in groups:
        ordered = sorted(items, key=lambda rectangle: rectangle.y)
        if len(ordered) <= 1:
            continue  # 单分支不需分叉线

        group_x = ordered[0].x
        lane = ordered[0].model_lane
        # 分叉起点和汇合终点在主干线上，位于组中心的左右两侧
        split = (group_x - 5.2, lane, 0.0)
        merge = (group_x + 5.2, lane, 0.0)
        for rectangle in ordered:
            center = (rectangle.x, rectangle.y, rectangle.z)
            # 左侧：从分叉点到矩形中心的贝塞尔曲线
            left_x, left_y, left_z = _quadratic_points(
                split,
                ((split[0] + center[0]) / 2, center[1], center[2]),
                center,
            )
            # 右侧：从矩形中心到汇合点的贝塞尔曲线
            right_x, right_y, right_z = _quadratic_points(
                center,
                ((center[0] + merge[0]) / 2, center[1], center[2]),
                merge,
            )
            # 拼接左右两段，中间用 None 断开形成独立线段
            split_x.extend(left_x + right_x[1:] + [None])
            split_y.extend(left_y + right_y[1:] + [None])
            split_z.extend(left_z + right_z[1:] + [None])

    if split_x:
        fig.add_trace(
            go.Scatter3d(
                x=split_x,
                y=split_y,
                z=split_z,
                mode="lines",
                line={"color": "#FFF700", "width": 5},
                hoverinfo="skip",
                name="split flow",
                showlegend=False,
            )
        )

    # 3. 方向锥体：在每模型最后放置
    for model_name, model_groups in groups_by_model.items():
        final = model_groups[-1][0]
        fig.add_trace(
            go.Cone(
                x=[final.x + 4.0],
                y=[final.model_lane],
                z=[0.0],
                u=[1.0],   # X 轴正方向
                v=[0.0],
                w=[0.0],
                sizemode="absolute",
                sizeref=1.6,
                anchor="tail",
                colorscale=[[0, "#111111"], [1, "#111111"]],
                showscale=False,
                hoverinfo="skip",
                name=f"{model_name} flow direction",
            )
        )


def tensor_rectangle_figure(rectangles: list[TensorRectangle]) -> go.Figure:
    """主入口：将张量矩形列表渲染为完整的 3D 结构图.

    返回一个包含以下内容的 Plotly Figure：
    - 3D 立方体网格（按模块类型着色）
    - 透明的悬停标记点（鼠标悬停显示张量详情）
    - 数据流路径（主干线 + 分叉贝塞尔曲线 + 方向箭头）
    - X 轴刻度标注每层的 QKV / O / GateUp / Down 位置
    - Y 轴刻度标注模型名称

    布局参数：
    - 每层 X 跨度 46 单位，4 个流阶段各占 11 单位
    - 相机默认视角 (2.45, 1.15, 0.95)
    - 宽高比 x:y:z = 10:1:1
    """

    fig = go.Figure()
    if not rectangles:
        return fig

    # 依次添加三类轨迹
    _add_mesh_traces(fig, rectangles)
    _add_hover_marker_trace(fig, rectangles)
    _add_data_flow_traces(fig, rectangles)

    # 构建 X 轴刻度：每层 4 个流阶段标签
    max_layer = max(rectangle.layer for rectangle in rectangles)
    model_lanes = {
        rectangle.model_lane: rectangle.model_name
        for rectangle in rectangles
    }
    tickvals = []
    ticktext = []
    for layer in range(max_layer + 1):
        offset = layer * LAYER_STRIDE
        for flow_x, label in FLOW_TICKS.items():
            tickvals.append(offset + flow_x)
            # 第一列为 "L0 Q/K/V" 格式，后续只显示模块名
            ticktext.append(f"L{layer} {label}" if flow_x == 0.0 else label)

    fig.update_layout(
        height=680,
        showlegend=False,
        margin={"l": 0, "r": 0, "b": 0, "t": 36},
        title="Model structure schematic",
        scene={
            "xaxis": {
                "title": "Data flow",
                "tickmode": "array",
                "tickvals": tickvals,
                "ticktext": ticktext,
            },
            "yaxis": {
                "title": "Model / Columns",
                "tickmode": "array",
                "tickvals": list(model_lanes.keys()),
                "ticktext": list(model_lanes.values()),
            },
            "zaxis": {"title": "Rows"},
            "aspectmode": "manual",
            "aspectratio": {"x": 10.0, "y": 1.0, "z": 1.0},
            "camera": {"eye": {"x": 2.45, "y": 1.15, "z": 0.95}},
        },
    )
    return fig
