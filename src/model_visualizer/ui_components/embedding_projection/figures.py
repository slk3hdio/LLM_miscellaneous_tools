from __future__ import annotations

"""嵌入投影的 Plotly 图形构建器 —— PCA/UMAP 轨迹动画.

支持两种视图模式：
- 静态单帧 (token_projection_figure)：适合导出或单层查看
- 动画多帧 (animated_token_projection_figure)：浏览器端逐层播放
"""

import html

import plotly.graph_objects as go

from model_visualizer.ui_components.embedding_projection.types import (
    LayerProjection,
    ProjectedToken,
    ProjectionBasis,
)

# 每个 token 索引对应的固定颜色（12 色调色板，循环使用）
_TOKEN_COLORS = [
    "#2563eb",  # 蓝
    "#dc2626",  # 红
    "#16a34a",  # 绿
    "#9333ea",  # 紫
    "#ea580c",  # 橙
    "#0891b2",  # 青
    "#be123c",  # 深红
    "#4f46e5",  # 靛蓝
    "#65a30d",  # 黄绿
    "#c026d3",  # 品红
    "#0f766e",  # 蓝绿
    "#a16207",  # 金棕
]


# ── 标签与文本辅助 ────────────────────────────────────────────


def _layer_label(layer_index: int) -> str:
    """层标签：嵌入层显示 "Embed"，普通层显示数字."""
    return "Embed" if layer_index < 0 else str(layer_index)


def _display_token_text(text: str) -> str:
    """将 token 文本转为可显示的简短形式.

    - 换行 → \\n，制表符 → \\t
    - 空文本 → "<empty>"
    """
    visible = text.replace("\n", "\\n").replace("\t", "\\t")
    return visible if visible else "<empty>"


def _point_label(point: ProjectedToken) -> str:
    """生成投影点的显示标签.

    如果有点击前的"当前最佳预测"信息，显示 "索引(原文本->**新预测**)"。
    否则显示 "文本 (token_id)"。
    """

    if point.current_token_id is not None and point.current_text is not None:
        return (
            f"{point.index}"
            f"({html.escape(_display_token_text(point.text))}"
            f"->{html.escape(_display_token_text(point.current_text))})"
        )
    return f"{html.escape(_display_token_text(point.text))} ({point.token_id})"


def _token_color(index: int) -> str:
    """根据 token 索引返回对应的固定颜色."""
    return _TOKEN_COLORS[index % len(_TOKEN_COLORS)]


# ── 轨迹构建辅助 ──────────────────────────────────────────────


def _scatter_from_points(
    points: list[ProjectedToken],
    *,
    dimensions: int,
    name: str = "Hidden states",
    color: str | list[str] | None = None,
    size: int | None = None,
    opacity: float = 1.0,
    show_text: bool = True,
    connect_points: bool = True,
    line_color: str | None = None,
    textfont: dict | None = None,
) -> go.Scatter | go.Scatter3d:
    """从 ProjectedToken 列表构建 Scatter 或 Scatter3d 轨迹.

    参数：
        points: 投影后的 token 点列表
        dimensions: 2 或 3（决定返回 Scatter 还是 Scatter3d）
        name: 轨迹名称
        color: 单色字符串 / 颜色列表 / None（按 index 使用 Viridis 色阶）
        size: 标记大小（None 则自动：3D=5, 2D=10）
        opacity: 透明度
        show_text: 是否显示文本标签
        connect_points: 是否用线连接相邻点（隐藏状态要连线，嵌入点不需要）
        line_color: 连接线颜色
        textfont: 文本标签的字体设置

    返回：
        go.Scatter（2D）或 go.Scatter3d（3D）
    """

    x_values = [point.x for point in points]
    y_values = [point.y for point in points]
    z_values = [point.z for point in points]
    labels = [_point_label(point) for point in points] if show_text else None
    hover_text = [
        (
            f"#{point.index}<br>"
            f"token id: {point.token_id}<br>"
            f"text: {html.escape(_display_token_text(point.text))}<br>"
            + (
                f"current best id: {point.current_token_id}<br>"
                f"current best text: {html.escape(_display_token_text(point.current_text))}<br>"
                if point.current_token_id is not None and point.current_text is not None
                else ""
            )
            + f"x: {point.x:.4f}<br>"
            f"y: {point.y:.4f}<br>"
            f"z: {point.z:.4f}"
        )
        for point in points
    ]

    marker = {
        "size": size or (5 if dimensions == 3 else 10),
        "line": {"width": 1, "color": "#ffffff"},  # 白色边框区分重叠点
        "opacity": opacity,
    }
    # 颜色：未指定时按 index 使用 Viridis 色阶（序列越前越紫，越后越黄）
    if color is None:
        marker.update(
            {
                "color": [point.index for point in points],
                "colorscale": "Viridis",
                "showscale": False,
            }
        )
    else:
        marker["color"] = color

    mode = "lines+markers" if connect_points else "markers"
    if show_text:
        mode = f"{mode}+text"
    line = {"color": line_color or "#9aa7b7", "width": 1} if connect_points else None

    if dimensions == 2:
        return go.Scatter(
            name=name,
            x=x_values,
            y=y_values,
            mode=mode,
            text=labels,
            textposition="top center",
            marker=marker,
            line=line,
            textfont=textfont,
            hovertext=hover_text,
            hoverinfo="text",
        )
    return go.Scatter3d(
        name=name,
        x=x_values,
        y=y_values,
        z=z_values,
        mode=mode,
        text=labels,
        textposition="top center",
        marker=marker,
        line=line,
        textfont=textfont,
        hovertext=hover_text,
        hoverinfo="text",
    )


def _hidden_scatter(points: list[ProjectedToken], *, dimensions: int) -> go.Scatter | go.Scatter3d:
    """构建"当前 token 状态"散点轨迹 —— 大号标记，按 token 索引着色，不连线.

    用于动画帧中显示当前层的隐藏状态位置。
    """

    return _scatter_from_points(
        points,
        dimensions=dimensions,
        name="Current token states",
        color=[_token_color(point.index) for point in points],
        size=6 if dimensions == 3 else 10,
        connect_points=False,
        textfont={
            "size": 11 if dimensions == 3 else 14,
            "color": "#0f172a",
            "family": "Arial Black, Arial, sans-serif",
        },
    )


def _best_prediction_label(point: ProjectedToken) -> str:
    """生成当前最佳解码 token 嵌入点的标签."""
    return (
        f"{point.index} best: "
        f"<b>{html.escape(_display_token_text(point.text))}</b> "
        f"({point.token_id})"
    )


def _best_prediction_scatter(
    points: list[ProjectedToken],
    *,
    dimensions: int,
    active: bool = True,
    name: str = "Current best token embeddings",
) -> go.Scatter | go.Scatter3d:
    """构建当前层每个位置最佳解码 token 的嵌入点.

    这些点和隐藏状态点同色，但使用更深边框和菱形标记，放在所有静态参考点之上，
    避免被浅灰 top-k 参考点淹没。
    """

    labels = [_best_prediction_label(point) for point in points]
    hover_text = [
        (
            f"{_best_prediction_label(point)}<br>"
            f"token position: {point.index}<br>"
            f"best token id: {point.token_id}<br>"
            f"best token text: {html.escape(_display_token_text(point.text))}<br>"
            f"x: {point.x:.4f}<br>"
            f"y: {point.y:.4f}<br>"
            f"z: {point.z:.4f}"
        )
        for point in points
    ]
    marker = {
        "size": 5 if dimensions == 3 else 8,
        "color": [_token_color(point.index) for point in points],
        "symbol": "diamond",
        # "line": {"width": 3, "color": "#111827"},
        "opacity": 1.0,
    }
    if dimensions == 2:
        return go.Scatter(
            name=name,
            x=[point.x for point in points],
            y=[point.y for point in points],
            mode="markers+text",
            text=labels,
            textposition="bottom center",
            marker=marker,
            textfont={
                "size": 13,
                "color": "#111827",
                "family": "Arial Black, Arial, sans-serif",
            },
            hovertext=hover_text,
            hoverinfo="text",
            opacity=1.0 if active else 0.0,
        )
    return go.Scatter3d(
        name=name,
        x=[point.x for point in points],
        y=[point.y for point in points],
        z=[point.z for point in points],
        mode="markers+text",
        text=labels,
        textposition="bottom center",
        marker=marker,
        textfont={
            "size": 11,
            "color": "#111827",
            "family": "Arial Black, Arial, sans-serif",
        },
        hovertext=hover_text,
        hoverinfo="text",
        opacity=1.0 if active else 0.0,
    )


def _best_prediction_opacity_trace(
    *,
    dimensions: int,
    active: bool,
) -> go.Scatter | go.Scatter3d:
    """只更新透明度的占位帧，避免 best prediction 标签在帧间插值移动."""
    opacity = 1.0 if active else 0.0
    if dimensions == 2:
        return go.Scatter(opacity=opacity)
    return go.Scatter3d(opacity=opacity)


def _empty_trace(*, dimensions: int, name: str) -> go.Scatter | go.Scatter3d:
    """创建一个空的占位轨迹（用于没有轨迹数据的 token）."""
    if dimensions == 2:
        return go.Scatter(name=name, x=[], y=[], mode="lines")
    return go.Scatter3d(name=name, x=[], y=[], z=[], mode="lines")


def _trajectory_points(
    layers: list[LayerProjection],
    *,
    token_index: int,
    layer_position: int,
) -> list[ProjectedToken]:
    """收集指定 token 从嵌入层到当前层位置的轨迹点.

    遍历 layers[0..layer_position]，收集每一层中 index==token_index 的点。
    """

    points: list[ProjectedToken] = []
    for layer in layers[: layer_position + 1]:
        point = next((item for item in layer.points if item.index == token_index), None)
        if point is not None:
            points.append(point)
    return points


def _trajectory_trace(
    layers: list[LayerProjection],
    *,
    token_index: int,
    layer_position: int,
    dimensions: int,
) -> go.Scatter | go.Scatter3d:
    """构建单个 token 的轨迹线 —— 展示它在各层投影空间中的移动路径.

    轨迹线从嵌入层到当前层，用该 token 的颜色绘制。
    如果 token 没有轨迹点则返回空占位轨迹。
    """

    points = _trajectory_points(layers, token_index=token_index, layer_position=layer_position)
    if not points:
        return _empty_trace(dimensions=dimensions, name=f"Token {token_index} path")
    labels = [_point_label(point) for point in points]
    hover_text = [
        (
            f"token position: {point.index}<br>"
            f"token id: {point.token_id}<br>"
            f"text: {html.escape(_display_token_text(point.text))}<br>"
            + (
                f"current best id: {point.current_token_id}<br>"
                f"current best text: {html.escape(_display_token_text(point.current_text))}<br>"
                if point.current_token_id is not None and point.current_text is not None
                else ""
            )
            + f"frame: {_layer_label(layers[index].layer_index)}"
        )
        for index, point in enumerate(points)
    ]
    color = _token_color(token_index)
    if dimensions == 2:
        return go.Scatter(
            name=f"Token {token_index} path",
            x=[point.x for point in points],
            y=[point.y for point in points],
            mode="lines+markers",
            marker={"size": 5, "color": color, "opacity": 0.9},
            line={"color": color, "width": 2},
            text=labels,
            hovertext=hover_text,
            hoverinfo="text",
        )
    return go.Scatter3d(
        name=f"Token {token_index} path",
        x=[point.x for point in points],
        y=[point.y for point in points],
        z=[point.z for point in points],
        mode="lines+markers",
        marker={"size": 4, "color": color, "opacity": 0.5},
        line={"color": color, "width": 2},
        text=labels,
        hovertext=hover_text,
        hoverinfo="text",
    )


def _unique_points_by_token_id(
    points: list[ProjectedToken],
    *,
    exclude_token_ids: set[int] | None = None,
) -> list[ProjectedToken]:
    """按 token_id 去重，保留首次出现的点.

    用于避免初始点、top-k 预测点、最终预测点之间的 token 重复显示。
    """

    seen = set(exclude_token_ids or set())
    unique: list[ProjectedToken] = []
    for point in points:
        if point.token_id in seen:
            continue
        seen.add(point.token_id)
        unique.append(point)
    return unique


def _all_top_prediction_points(
    layers: list[LayerProjection],
    *,
    exclude_token_ids: set[int] | None = None,
) -> list[ProjectedToken]:
    """收集所有层的 top-k 预测点，按 token_id 去重.

    排除已在 exclude_token_ids 中的 token。
    """

    return _unique_points_by_token_id(
        [point for layer in layers for point in layer.top_prediction_points],
        exclude_token_ids=exclude_token_ids,
    )


# ── 坐标轴范围计算 ────────────────────────────────────────────


def _axis_range(values: list[float], *, padding_ratio: float = 0.02) -> list[float]:
    """计算坐标轴范围（含 padding）.

    处理边缘情况：
    - 空列表：返回 [-1, 1]
    - 所有值相同：添加绝对或比例 padding
    """

    if not values:
        return [-1.0, 1.0]
    low = min(values)
    high = max(values)
    if low == high:
        padding = max(1e-3, abs(low) * 0.02)
    else:
        padding = (high - low) * padding_ratio
    return [low - padding, high + padding]


def _cube_ranges(
    x_values: list[float],
    y_values: list[float],
    z_values: list[float],
    *,
    padding_ratio: float = 0.02,
) -> tuple[list[float], list[float], list[float]]:
    """为 3D 图计算各轴范围，强制保持立方体长宽比.

    策略：
    1. 计算 X/Y/Z 各自的原始范围
    2. 取最大跨度（span）
    3. 所有轴使用相同的半跨度，以中心对齐
    4. 添加 padding_ratio 的额外空间
    """

    ranges = [
        _axis_range(x_values, padding_ratio=0.0),
        _axis_range(y_values, padding_ratio=0.0),
        _axis_range(z_values, padding_ratio=0.0),
    ]
    centers = [(axis_range[0] + axis_range[1]) / 2.0 for axis_range in ranges]
    spans = [axis_range[1] - axis_range[0] for axis_range in ranges]
    span = max(max(spans), 1e-3)
    half_span = span * (0.5 + padding_ratio)
    return tuple(
        [center - half_span, center + half_span]
        for center in centers
    )


def _ranges_for_points(
    points: list[ProjectedToken],
    *,
    dimensions: int,
    exclude_first_token: bool = False,
) -> dict:
    """根据投影点计算坐标轴范围.

    参数：
        points: 投影点列表
        dimensions: 2 或 3
        exclude_first_token: 是否排除第一个 token（index=0）。
                             首 token 通常是特殊标记（BOS），位置可能离群。

    返回：
        {"x_range": [...], "y_range": [...], "z_range": [...] 或 None}
    """

    range_points = [
        point
        for point in points
        if not exclude_first_token or point.index != 0
    ]
    if not range_points:
        range_points = points
    x_values = [point.x for point in range_points]
    y_values = [point.y for point in range_points]
    z_values = [point.z for point in range_points]
    if dimensions == 3:
        x_range, y_range, z_range = _cube_ranges(x_values, y_values, z_values)
        return {
            "x_range": x_range,
            "y_range": y_range,
            "z_range": z_range,
        }
    return {
        "x_range": _axis_range(x_values),
        "y_range": _axis_range(y_values),
        "z_range": None,
    }


# ── 布局构建 ──────────────────────────────────────────────────


def _axis_titles(
    basis: ProjectionBasis | None,
    *,
    dimensions: int,
    axis_title_prefix: str = "UMAP",
) -> tuple[str, ...]:
    """生成坐标轴标题.

    - 有 PCA 基：显示 PC1/PC2/PC3 及其解释方差百分比
    - 无基（UMAP 模式）：显示 "UMAP 1" / "UMAP 2" / "UMAP 3"
    """

    if basis is None:
        titles = [f"{axis_title_prefix} 1", f"{axis_title_prefix} 2"]
        if dimensions == 3:
            titles.append(f"{axis_title_prefix} 3")
        return tuple(titles)
    variance = basis.explained_variance_ratio
    titles = [
        f"PC1 ({float(variance[0]) * 100:.1f}%)",
        f"PC2 ({float(variance[1]) * 100:.1f}%)",
    ]
    if dimensions == 3:
        if len(variance) >= 3:
            titles.append(f"PC3 ({float(variance[2]) * 100:.1f}%)")
        else:
            titles.append("PC3")
    return tuple(titles)


def _base_layout(
    basis: ProjectionBasis | None,
    *,
    height: int,
    x_range: list[float] | None = None,
    y_range: list[float] | None = None,
    z_range: list[float] | None = None,
    dimensions: int,
    axis_title_prefix: str = "UMAP",
) -> dict:
    """构建 Plotly 基础布局（用于静态图和动画图的初始状态）.

    2D 图特性：等比例坐标轴（scaleanchor + scaleratio），零线显示
    3D 图特性：cube 模式（等比例 3D 轴），固定相机角度
    """

    titles = _axis_titles(basis, dimensions=dimensions, axis_title_prefix=axis_title_prefix)
    x_title, y_title = titles[:2]

    if dimensions == 2:
        layout = {
            "height": height,
            "margin": {"l": 42, "r": 24, "t": 28, "b": 42},
            "xaxis_title": x_title,
            "yaxis_title": y_title,
            "showlegend": False,
            "template": "plotly_white",
            "xaxis": {"zeroline": True, "zerolinecolor": "#d0d5dd"},
            "yaxis": {
                "zeroline": True,
                "zerolinecolor": "#d0d5dd",
                "scaleanchor": "x",
                "scaleratio": 1,
            },
        }
        if x_range is not None:
            layout["xaxis"]["range"] = x_range
        if y_range is not None:
            layout["yaxis"]["range"] = y_range
        return layout

    # 3D 布局
    z_title = titles[2]
    x_axis = {"title": x_title, "zeroline": True, "zerolinecolor": "#d0d5dd"}
    y_axis = {"title": y_title, "zeroline": True, "zerolinecolor": "#d0d5dd"}
    z_axis = {"title": z_title, "zeroline": True, "zerolinecolor": "#d0d5dd"}
    if x_range is not None:
        x_axis["range"] = x_range
    if y_range is not None:
        y_axis["range"] = y_range
    if z_range is not None:
        z_axis["range"] = z_range

    layout = {
        "height": height,
        "margin": {"l": 42, "r": 24, "t": 28, "b": 42},
        "showlegend": False,
        "template": "plotly_white",
        "scene": {
            "xaxis": x_axis,
            "yaxis": y_axis,
            "zaxis": z_axis,
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 1.1}},
        },
    }
    return layout


# ── 公开 API ──────────────────────────────────────────────────


def token_projection_figure(
    points: list[ProjectedToken],
    basis: ProjectionBasis | None,
    *,
    height: int = 1280,
    dimensions: int = 3,
    initial_points: list[ProjectedToken] | None = None,
    top_prediction_points: list[ProjectedToken] | None = None,
    best_prediction_points: list[ProjectedToken] | None = None,
    final_prediction_points: list[ProjectedToken] | None = None,
    axis_title_prefix: str = "UMAP",
) -> go.Figure:
    """构建单帧的投影散点图（无动画，适合导出静态图片）.

    轨迹层次（按添加顺序）：
    1. 隐藏状态投影点（带连线，按 index Viridis 着色）
    2. 初始嵌入投影点（深灰 #3f3f46，较大标记）
    3. Top-k 预测嵌入点（浅灰 #d1d5db，较小标记）
    4. 最终预测嵌入点（深黑 #111827，最大的标记）

    token 去重策略：final_prediction > initial > top_prediction。
    后添加的类别排除前面已出现的 token_id。
    """

    initial_points = initial_points or []
    top_prediction_points = top_prediction_points or []
    best_prediction_points = best_prediction_points or []
    final_prediction_points = final_prediction_points or []

    # 去重：优先级 final > initial > top_prediction
    final_prediction_points = _unique_points_by_token_id(final_prediction_points)
    final_token_ids = {point.token_id for point in final_prediction_points}
    initial_points = _unique_points_by_token_id(initial_points, exclude_token_ids=final_token_ids)
    top_prediction_points = _unique_points_by_token_id(
        top_prediction_points,
        exclude_token_ids=final_token_ids | {point.token_id for point in initial_points},
    )

    range_points = (
        points
        + initial_points
        + top_prediction_points
        + final_prediction_points
        + best_prediction_points
    )
    fig = go.Figure()

    # 轨迹 1：隐藏状态（主轨迹）
    fig.add_trace(_scatter_from_points(points, dimensions=dimensions))

    # 轨迹 2：初始嵌入点
    fig.add_trace(
        _scatter_from_points(
            initial_points,
            dimensions=dimensions,
            name="Initial token embeddings",
            color="#3f3f46",
            size=7 if dimensions == 3 else 11,
            opacity=0.95,
            connect_points=False,
        )
    )

    # 轨迹 3：Top-k 预测点
    fig.add_trace(
        _scatter_from_points(
            top_prediction_points,
            dimensions=dimensions,
            name="Top-k token embeddings",
            color="#d1d5db",
            size=4 if dimensions == 3 else 7,
            opacity=0.65,
            show_text=True,
            connect_points=False,
        )
    )

    # 轨迹 4：最终预测点
    fig.add_trace(
        _scatter_from_points(
            final_prediction_points,
            dimensions=dimensions,
            name="Final prediction embedding",
            color="#111827",
            size=9 if dimensions == 3 else 14,
            opacity=1.0,
            connect_points=False,
        )
    )

    # 轨迹 5：当前层每个位置的最佳解码 token 嵌入点
    fig.add_trace(_best_prediction_scatter(best_prediction_points, dimensions=dimensions))

    ranges = _ranges_for_points(range_points, dimensions=dimensions)
    fig.update_layout(
        **_base_layout(
            basis,
            height=height,
            x_range=ranges["x_range"],
            y_range=ranges["y_range"],
            z_range=ranges["z_range"],
            dimensions=dimensions,
            axis_title_prefix=axis_title_prefix,
        )
    )
    return fig


def animated_token_projection_figure(
    layers: list[LayerProjection],
    basis: ProjectionBasis | None,
    *,
    initial_layer_index: int = 0,
    height: int = 1280,
    dimensions: int = 3,
    initial_points: list[ProjectedToken] | None = None,
    final_prediction_points: list[ProjectedToken] | None = None,
    axis_title_prefix: str = "UMAP",
) -> go.Figure:
    """构建带浏览器端动画的投影图 —— 核心可视化函数.

    与旧版不同，动画帧不切换整个散点图，而是：
    - 固定轨迹 0：当前层的 token 状态（大标记散点，随帧更新位置）
    - 固定轨迹 1..N：每个 token 从嵌入层到当前帧的轨迹线（随帧增长）
    - 固定轨迹（静止）：初始嵌入点、所有 top-k 预测点、最终预测点

    详细构建过程：
    1. 验证输入、确定初始层
    2. 去重各类嵌入点（final > initial > top_prediction）
    3. 收集所有活跃 token 的索引
    4. 计算全局固定坐标轴范围
    5. 构建 Frame 列表：每帧 = hidden_scatter + 每条轨迹线的当前状态
       （使用 traces 参数指定需要更新的轨迹索引）
    6. 构建基础 Figure：初始状态 + 所有帧 + 静止轨迹
    7. 添加 Play/Pause update menu
    8. 添加层选择 slider

    参数：
        layers: 所有层的投影数据
        basis: PCA 基或 None（None=UMAP 模式，轴标题显示 "UMAP 1/2/3"）
        initial_layer_index: 初始显示哪一层
        height: 图高度
        dimensions: 2 或 3
        initial_points: 初始嵌入投影点（所有帧共享，静止）
        final_prediction_points: 最终预测嵌入点（所有帧共享，静止）

    返回：
        带有完整动画控制（滑块 + 播放按钮）的 Plotly Figure
    """

    if not layers:
        raise ValueError("At least one layer projection is required.")

    # 1. 确定初始层
    layer_by_index = {layer.layer_index: layer for layer in layers}
    if initial_layer_index not in layer_by_index:
        initial_layer_index = layers[0].layer_index
    initial_layer = layer_by_index[initial_layer_index]

    # 2. 去重：优先级 final > initial > top_prediction
    initial_points = initial_points or []
    final_prediction_points = final_prediction_points or []
    final_prediction_points = _unique_points_by_token_id(final_prediction_points)
    final_token_ids = {point.token_id for point in final_prediction_points}
    initial_points = _unique_points_by_token_id(initial_points, exclude_token_ids=final_token_ids)
    top_prediction_points = _all_top_prediction_points(
        layers,
        exclude_token_ids=final_token_ids | {point.token_id for point in initial_points},
    )
    best_prediction_points = [point for layer in layers for point in layer.best_prediction_points]

    # 3. 收集所有 token 索引（用于生成轨迹线）
    moving_points = [point for layer in layers for point in layer.points]
    token_indices = sorted({point.index for point in moving_points})
    initial_layer_position = layers.index(initial_layer)

    # 4. 计算全局固定坐标轴范围
    fixed_ranges = _ranges_for_points(
        (
            moving_points
            + initial_points
            + top_prediction_points
            + final_prediction_points
            + best_prediction_points
        ),
        dimensions=dimensions,
    )
    initial_ranges = fixed_ranges
    best_trace_start = 1 + len(token_indices) + 3
    best_trace_indices = list(range(best_trace_start, best_trace_start + len(layers)))
    # 动画帧更新隐藏状态和轨迹线；best token 参考点只切换可见性，避免插值移动
    animated_trace_indices = list(range(1 + len(token_indices))) + best_trace_indices

    # 5. 构建 Frame 列表
    frames = [
        go.Frame(
            name=_layer_label(layer.layer_index),
            data=[
                # 轨迹 0：当前层的 token 状态散点
                _hidden_scatter(layer.points, dimensions=dimensions),
                # 轨迹 1..N：每个 token 从嵌入层到当前帧的轨迹线
                *[
                    _trajectory_trace(
                        layers,
                        token_index=token_index,
                        layer_position=layer_position,
                        dimensions=dimensions,
                    )
                    for token_index in token_indices
                ],
                # best token 参考点是每层固定的一组 trace；帧切换时只切换可见性
                *[
                    _best_prediction_opacity_trace(
                        dimensions=dimensions,
                        active=target_layer.layer_index == layer.layer_index,
                    )
                    for target_layer in layers
                ],
            ],
            traces=animated_trace_indices,  # 只有这些轨迹在帧间更新
        )
        for layer_position, layer in enumerate(layers)
    ]

    # 6. 构建 slider 步骤
    steps = [
        {
            "label": _layer_label(layer.layer_index),
            "method": "animate",
            "args": [
                [_layer_label(layer.layer_index)],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 360},
                },
            ],
        }
        for layer in layers
    ]

    # 7. 构建基础 Figure（初始状态 + 所有帧 + 静止轨迹）
    fig = go.Figure(
        data=[
            # 动画轨迹（帧间更新）
            _hidden_scatter(initial_layer.points, dimensions=dimensions),
            *[
                _trajectory_trace(
                    layers,
                    token_index=token_index,
                    layer_position=initial_layer_position,
                    dimensions=dimensions,
                )
                for token_index in token_indices
            ],
            # 静止轨迹（不随帧变化）
            _scatter_from_points(
                initial_points,
                dimensions=dimensions,
                name="Initial token embeddings",
                color="#3f3f46",
                size=7 if dimensions == 3 else 11,
                opacity=0.95,
                connect_points=False,
            ),
            _scatter_from_points(
                top_prediction_points,
                dimensions=dimensions,
                name="All top-k token embeddings",
                color="#d1d5db",
                size=4 if dimensions == 3 else 7,
                opacity=0.65,
                show_text=True,
                connect_points=False,
            ),
            _scatter_from_points(
                final_prediction_points,
                dimensions=dimensions,
                name="Final prediction embedding",
                color="#111827",
                size=9 if dimensions == 3 else 14,
                opacity=1.0,
                connect_points=False,
            ),
            # 放在所有静态参考点之后；每层一组固定点，切帧时消失/出现而不是移动
            *[
                _best_prediction_scatter(
                    layer.best_prediction_points,
                    dimensions=dimensions,
                    active=layer.layer_index == initial_layer.layer_index,
                )
                for layer in layers
            ],
        ],
        frames=frames,
    )

    # 8. 基础布局 + Play/Pause 按钮 + 层选择 slider
    fig.update_layout(
        **_base_layout(
            basis,
            height=height,
            x_range=initial_ranges["x_range"],
            y_range=initial_ranges["y_range"],
            z_range=initial_ranges["z_range"],
            dimensions=dimensions,
            axis_title_prefix=axis_title_prefix,
        ),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "direction": "left",
                "x": 0,
                "y": 1.08,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "frame": {"duration": 840, "redraw": True},
                                "transition": {"duration": 520},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": layers.index(initial_layer),
                "currentvalue": {"prefix": "Frame ", "font": {"size": 12}},
                "pad": {"t": 48, "b": 8},
                "steps": steps,
            }
        ],
    )
    return fig
