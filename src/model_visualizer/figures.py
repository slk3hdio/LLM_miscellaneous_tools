from __future__ import annotations

"""为直方图堆叠等通用可视化提供 Plotly 图形构建器."""

import plotly.graph_objects as go

from model_visualizer.analysis.types import LayerHistogram


_LAYER_BASE_COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4f46e5",
]


def _histogram_bin_width(histogram: LayerHistogram) -> float:
    """从 bin 中心估计柱形宽度."""

    centers = list(histogram.bin_centers)
    if len(centers) < 2:
        return 1.0
    widths = [
        abs(right - left)
        for left, right in zip(centers, centers[1:])
        if right != left
    ]
    return min(widths) if widths else 1.0


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    """将 #RRGGBB 转成 RGB 三元组."""

    value = color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _blend_rgb(
    color: tuple[int, int, int],
    other: tuple[int, int, int],
    amount: float,
) -> tuple[int, int, int]:
    """按 amount 将 color 混向 other."""

    return tuple(
        round(channel + (target - channel) * amount)
        for channel, target in zip(color, other)
    )


def _rgb_to_css(color: tuple[int, int, int]) -> str:
    """将 RGB 三元组转成 Plotly 可用的 CSS 颜色."""

    return f"rgb({color[0]}, {color[1]}, {color[2]})"


def _bar_vertex_colors(
    *,
    layer_position: int,
    height: float,
    max_height: float,
) -> list[str]:
    """生成一个柱体的 8 个顶点颜色.

    层号决定色系，柱高决定深浅。底部略浅、顶部略深，增强立体感。
    """

    base = _hex_to_rgb(_LAYER_BASE_COLORS[layer_position % len(_LAYER_BASE_COLORS)])
    normalized_height = 0.0 if max_height <= 0 else max(0.0, min(height / max_height, 1.0))
    shade = 0.32 + normalized_height * 0.68
    white = (255, 255, 255)
    bottom = _rgb_to_css(_blend_rgb(white, base, shade * 0.72))
    top = _rgb_to_css(_blend_rgb(white, base, shade))
    return [bottom, bottom, top, top, bottom, bottom, top, top]


def _add_bar_box(
    *,
    x_vertices: list[float],
    y_vertices: list[float],
    z_vertices: list[float],
    vertex_colors: list[str],
    custom_data: list[list[float]],
    face_i: list[int],
    face_j: list[int],
    face_k: list[int],
    center: float,
    height: float,
    z_center: float,
    width: float,
    layer: int,
    colors: list[str],
) -> None:
    """向 Mesh3d 顶点/面列表追加一个长方体柱."""

    base = len(x_vertices)
    x0 = center - width / 2.0
    x1 = center + width / 2.0
    z0 = z_center - width / 2.0
    z1 = z_center + width / 2.0
    vertices = [
        (x0, 0.0, z0),
        (x1, 0.0, z0),
        (x1, height, z0),
        (x0, height, z0),
        (x0, 0.0, z1),
        (x1, 0.0, z1),
        (x1, height, z1),
        (x0, height, z1),
    ]
    faces = [
        (0, 1, 2), (0, 2, 3),  # front
        (4, 6, 5), (4, 7, 6),  # back
        (0, 4, 5), (0, 5, 1),  # bottom
        (3, 2, 6), (3, 6, 7),  # top
        (1, 5, 6), (1, 6, 2),  # right
        (0, 3, 7), (0, 7, 4),  # left
    ]

    for (x, y, z), color in zip(vertices, colors):
        x_vertices.append(float(x))
        y_vertices.append(float(y))
        z_vertices.append(float(z))
        vertex_colors.append(color)
        custom_data.append([float(center), float(height), float(layer)])
    for i, j, k in faces:
        face_i.append(base + i)
        face_j.append(base + j)
        face_k.append(base + k)


def layer_histogram_stack_figure(
    histograms: list[LayerHistogram],
    *,
    density: bool = True,
) -> go.Figure:
    """构建每层权重直方图的 3D 柱形堆叠图.

    每个层对应一组竖向柱，沿 Z 轴堆叠，便于横向比较各层权重分布形态。

    参数：
        histograms: 按层排列的直方图数据列表
        density: True 表示使用密度（归一化），False 表示使用原始计数
    """

    fig = go.Figure()
    bin_width = min(_histogram_bin_width(histogram) for histogram in histograms)
    layer_gap = bin_width
    layer_tick_values: list[float] = []
    layer_tick_labels: list[str] = []

    for layer_position, histogram in enumerate(histograms):
        z_center = layer_position * layer_gap
        layer_tick_values.append(z_center)
        layer_tick_labels.append(str(histogram.layer))
        x_vertices: list[float] = []
        y_vertices: list[float] = []
        z_vertices: list[float] = []
        vertex_colors: list[str] = []
        custom_data: list[list[float]] = []
        face_i: list[int] = []
        face_j: list[int] = []
        face_k: list[int] = []
        max_height = max((float(value) for value in histogram.values), default=0.0)

        for center, value in zip(histogram.bin_centers, histogram.values):
            _add_bar_box(
                x_vertices=x_vertices,
                y_vertices=y_vertices,
                z_vertices=z_vertices,
                vertex_colors=vertex_colors,
                custom_data=custom_data,
                face_i=face_i,
                face_j=face_j,
                face_k=face_k,
                center=float(center),
                height=float(value),
                z_center=z_center,
                width=bin_width,
                layer=histogram.layer,
                colors=_bar_vertex_colors(
                    layer_position=layer_position,
                    height=float(value),
                    max_height=max_height,
                ),
            )

        fig.add_trace(
            go.Mesh3d(
                x=x_vertices,
                y=y_vertices,
                z=z_vertices,
                i=face_i,
                j=face_j,
                k=face_k,
                vertexcolor=vertex_colors,
                opacity=1.0,
                flatshading=True,
                customdata=custom_data,
                lighting={
                    "ambient": 0.55,
                    "diffuse": 0.75,
                    "roughness": 0.8,
                    "specular": 0.2,
                },
                hovertemplate=(
                    f"{histogram.tensor_name}<br>"
                    "layer: %{customdata[2]:.0f}<br>"
                    "value: %{customdata[0]:.6g}<br>"
                    f"{'density' if density else 'count'}: "
                    "%{customdata[1]:.6g}<extra></extra>"
                ),
                name=f"layer {histogram.layer}",
                showlegend=False,
            )
        )

    fig.update_layout(
        height=720,
        title="Layer histogram bar stack",
        margin={"l": 0, "r": 0, "b": 0, "t": 42},
        scene={
            "xaxis": {"title": "Parameter value"},                    # X 轴：参数值
            "yaxis": {"title": "Density" if density else "Count"},    # Y 轴：密度/计数
            "zaxis": {
                "title": "Layer",
                "tickmode": "array",
                "tickvals": layer_tick_values,
                "ticktext": layer_tick_labels,
            },
            "camera": {
                "eye": {"x": 1.8, "y": 1.15, "z": 1.8},
                "up": {"x": 0, "y": 1, "z": 0},
            },
        },
    )
    return fig
