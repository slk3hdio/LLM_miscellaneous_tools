from __future__ import annotations

"""推理追踪的 HTML/CSS 渲染辅助 —— token 条和 top-k 预测网格."""

import html

import streamlit as st

from model_visualizer.ui_components.inference.trace import DecodedToken, TopTokenPrediction


def display_token_text(text: str) -> str:
    """将 token 文本转为安全的 HTML 显示字符串.

    - 空文本 → "&lt;empty&gt;"
    - 空格 → "&middot;"（中间点，可视化空格）
    - 换行符 → "\\n"，制表符 → "\\t"
    - 其余进行 HTML 转义
    """

    if not text:
        return "&lt;empty&gt;"
    visible = text.replace("\n", "\\n").replace("\t", "\\t")
    return html.escape(visible).replace(" ", "&middot;")


def render_trace_css() -> None:
    """注入统一样式块.

    使用 st.markdown 和 unsafe_allow_html=True 在页面中嵌入 CSS。
    样式覆盖：token 卡片网格、预测溢出条、概率条等。
    """

    st.markdown(
        """
        <style>
        .trace-section-label {
            font-weight: 600;
            margin: 0.9rem 0 0.35rem 0;
        }
        .trace-token-scroll {
            overflow-x: auto;           /* token 条超出时水平滚动 */
            padding-bottom: 0.25rem;
        }
        .trace-token-grid,
        .trace-top-grid {
            display: grid;
            gap: 0.35rem;
            align-items: stretch;
        }
        .trace-token {
            min-height: 76px;
            border: 1px solid #d9dee8;
            border-radius: 6px;
            padding: 0.35rem;
            background: #ffffff;
            font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
        }
        .trace-token.generated {        /* 已生成的 token 用黄色高亮 */
            border-color: #b88a00;
            background: #fff8df;
        }
        .trace-token-index,
        .trace-token-id,
        .trace-top-header,
        .trace-prob-value {
            color: #667085;
            font-size: 0.72rem;
        }
        .trace-token-text {
            min-height: 1.45rem;
            margin: 0.18rem 0;
            overflow-wrap: anywhere;
            font-size: 0.86rem;
        }
        .trace-top-cell {
            border: 1px solid #d9dee8;
            border-radius: 6px;
            padding: 0.4rem;
            background: #ffffff;
            font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
        }
        .trace-top-row {
            font-size: 0.78rem;
            line-height: 1.25rem;
            margin-top: 0.25rem;
        }
        .trace-top-line {
            display: grid;
            grid-template-columns: 1.2rem minmax(0, 1fr) 3rem;
            gap: 0.18rem;
            align-items: baseline;
        }
        .trace-candidate {
            overflow-wrap: anywhere;
        }
        .trace-prob-bar {               /* 概率条容器 */
            height: 0.35rem;
            border-radius: 999px;
            background: #e8edf5;
            overflow: hidden;
            margin: 0.1rem 0 0.15rem 1.38rem;
        }
        .trace-prob-fill {              /* 概率条填充 */
            height: 100%;
            border-radius: inherit;
            background: #2f80ed;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_token_strip(
    title: str,
    tokens: list[DecodedToken],
    *,
    generated_from: int | None = None,
) -> None:
    """渲染水平滚动的 token 卡片条.

    每个 token 显示为一张卡片：索引号、文本、token ID。
    已生成的 token 用黄色边框和背景区分。

    参数：
        title: 条标题
        tokens: 解码后的 token 列表
        generated_from: 从第几个 token 开始标记为"已生成"，默认为全部是输入 token
    """

    if not tokens:
        st.caption(f"{title}: no tokens")
        return

    cells = []
    # 确定生成起点的默认值：全部标记为输入 token
    generated_from = len(tokens) if generated_from is None else generated_from
    for token in tokens:
        # 索引 >= generated_from 的标记为"已生成"
        classes = "trace-token generated" if token.index >= generated_from else "trace-token"
        cells.append(
            "<div class='{classes}'>"
            "<div class='trace-token-index'>#{index}</div>"
            "<div class='trace-token-text'>{text}</div>"
            "<div class='trace-token-id'>{token_id}</div>"
            "</div>".format(
                classes=classes,
                index=token.index,
                text=display_token_text(token.text),
                token_id=token.token_id,
            )
        )

    # 使用 CSS Grid 水平排列，每列固定 180px
    st.markdown(
        f"""
        <div class="trace-section-label">{html.escape(title)}</div>
        <div class="trace-token-scroll">
          <div class="trace-token-grid" style="grid-template-columns: repeat({len(tokens)}, 180px);">
            {''.join(cells)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_predictions(
    tokens: list[DecodedToken],
    predictions: list[list[TopTokenPrediction]],
) -> None:
    """渲染每个序列位置的 next-token 预测网格.

    每个位置显示为一个卡片列：
    - 标题：位置的 token 文本
    - 每行：排名、候选 token 文本、概率百分比
    - 概率条：蓝色填充宽度 = probability × 100%

    参数：
        tokens: 当前序列的已解码 token
        predictions: predictions[position][rank] → TopTokenPrediction
    """

    if not tokens or not predictions:
        return

    columns = []
    for token, token_predictions in zip(tokens, predictions):
        rows = []
        for prediction in token_predictions:
            probability_percent = max(0.0, min(100.0, prediction.probability * 100.0))
            rows.append(
                "<div class='trace-top-row'>"
                "<div class='trace-top-line'>"
                "<span class='trace-rank'>{rank}</span>"
                "<span class='trace-candidate'>{text}</span>"
                "<span class='trace-prob-value'>{probability:.1%}</span>"
                "</div>"
                "<div class='trace-prob-bar'>"
                "<div class='trace-prob-fill' style='width: {probability_percent:.2f}%;'></div>"
                "</div>"
                "</div>".format(
                    rank=prediction.rank,
                    text=display_token_text(prediction.text),
                    probability=prediction.probability,
                    probability_percent=probability_percent,
                )
            )
        columns.append(
            "<div class='trace-top-cell'>"
            "<div class='trace-top-header'>#{index} {text}</div>"
            "{rows}"
            "</div>".format(
                index=token.index,
                text=display_token_text(token.text),
                rows="".join(rows),
            )
        )

    st.markdown(
        f"""
        <div class="trace-section-label">Next token predictions</div>
        <div class="trace-token-scroll">
          <div class="trace-top-grid" style="grid-template-columns: repeat({len(columns)}, 180px);">
            {''.join(columns)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
