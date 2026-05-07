from __future__ import annotations

"""推理步进器组件的 Plotly 图形 —— 注意力热图."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from model_visualizer.ui_components.inference.trace import DecodedToken


def attention_heatmap_figure(
    attention_matrix,
    tokens: list[DecodedToken],
    *,
    head_index: int,
    layer_index: int,
    height: int = 320,
) -> go.Figure:
    """构建单个注意力头的 Query-Key 热图.

    使用 Plotly Heatmap：
    - X 轴：Key token（被关注的 token）
    - Y 轴：Query token（发起关注的 token，倒序使左上角为序列起始）
    - 颜色：Viridis 色阶，固定范围 [0, 1]
    - 悬停信息：query token、key token、注意力权重值

    参数：
        attention_matrix: [seq, seq] 注意力权重（已转为 CPU float）
        tokens: 解码后的 token 列表（用于轴标签）
        head_index: head 序号
        layer_index: 层序号
        height: 图高度（px）
    """

    # 轴标签：token 索引和文本
    labels = [
        f"{token.index}: {token.text if token.text else '<empty>'}"
        for token in tokens
    ]
    z_values = attention_matrix.numpy() if hasattr(attention_matrix, "numpy") else attention_matrix
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z_values,
                x=labels,       # Key token
                y=labels,       # Query token
                colorscale="Viridis",
                zmin=0,
                zmax=1,         # 注意力权重在 [0,1] 内
                colorbar={"title": "weight"},
                hovertemplate=(
                    "query: %{y}<br>"
                    "key: %{x}<br>"
                    "attention: %{z:.4f}<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        height=height,
        title=f"Layer {layer_index} attention head {head_index}",
        margin={"l": 24, "r": 24, "b": 96, "t": 48},
        xaxis={"title": "Key token", "tickangle": -45, "automargin": True},
        yaxis={"title": "Query token", "autorange": "reversed", "automargin": True},
    )
    return fig


def attention_layer_heatmap_figure(
    attention,
    tokens: list[DecodedToken],
    *,
    layer_index: int,
    height_per_row: int = 300,
    columns_per_row: int = 3,
) -> go.Figure:
    """构建某一层所有 attention heads 的合并热图."""

    head_count = int(attention.shape[0])
    if head_count < 1:
        raise ValueError("attention layer must contain at least one head.")
    labels = [
        f"{token.index}: {token.text if token.text else '<empty>'}"
        for token in tokens
    ]
    rows = (head_count + columns_per_row - 1) // columns_per_row
    fig = make_subplots(
        rows=rows,
        cols=columns_per_row,
        subplot_titles=[f"Head {head_index}" for head_index in range(head_count)],
        horizontal_spacing=0.04,
        vertical_spacing=0.08,
    )

    for head_index in range(head_count):
        row = head_index // columns_per_row + 1
        col = head_index % columns_per_row + 1
        matrix = attention[head_index]
        z_values = matrix.numpy() if hasattr(matrix, "numpy") else matrix
        fig.add_trace(
            go.Heatmap(
                z=z_values,
                x=labels,
                y=labels,
                colorscale="Viridis",
                zmin=0,
                zmax=1,
                colorbar={"title": "weight"} if head_index == head_count - 1 else None,
                showscale=head_index == head_count - 1,
                hovertemplate=(
                    f"layer: {layer_index}<br>"
                    f"head: {head_index}<br>"
                    "query: %{y}<br>"
                    "key: %{x}<br>"
                    "attention: %{z:.4f}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(tickangle=-45, automargin=True, row=row, col=col)
        fig.update_yaxes(autorange="reversed", automargin=True, row=row, col=col)

    fig.update_layout(
        height=max(height_per_row * rows, 360),
        title=f"Layer {layer_index} attention heads",
        margin={"l": 32, "r": 32, "b": 96, "t": 72},
        template="plotly_white",
    )
    return fig
