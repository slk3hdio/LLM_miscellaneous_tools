from __future__ import annotations

"""推理步进器 Streamlit 组件 —— 逐步回放每层的推理过程."""

from typing import Any

import streamlit as st

from model_visualizer.ui_components.inference.trace import (
    decode_token_ids,
    hidden_state_for_layer,
    select_attention_head,
    top_token_predictions,
)
from model_visualizer.ui_components.inference.figures import attention_heatmap_figure
from model_visualizer.ui_components.inference.runtime import CACHE_VERSION, cached_runtime, cached_tokenizer
from model_visualizer.ui_components.inference.state import InferenceState
from model_visualizer.ui_components.inference.types import InferenceTraceView
from model_visualizer.ui_components.inference.widgets import (
    render_token_strip,
    render_top_predictions,
    render_trace_css,
)
from model_visualizer.ui_components.structure.types import ModelStructure


class InferenceStepperComponent:
    """逐步推理回放组件.

    用户每次点击 "Next step" 前进一层（或一个生成步），组件展示：
    - 当前序列的 token 条
    - 每个位置的 top-k LM 头预测
    - 每层所有注意力头的热图

    到达最后一层后，执行一次实际的 token 生成，然后从新序列的第 0 层继续。
    """

    def __init__(self, structure: ModelStructure, *, state_prefix: str = "inference"):
        self.structure = structure
        self.model_dir = structure.model_dir
        self.state = InferenceState(structure, prefix=state_prefix)

    def _tokenize_prompt(self, prompt: str) -> tuple[Any, list[int]]:
        """对提示进行分词，返回 (tokenizer, token_id_list)."""
        tokenizer = cached_tokenizer(self.model_dir)
        token_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].squeeze(0).tolist()
        return tokenizer, [int(token_id) for token_id in token_ids]

    def runtime(self):
        """获取缓存的运行时（model + tokenizer + lm_head）."""
        return cached_runtime(self.model_dir, CACHE_VERSION)

    def advance(self) -> bool:
        """前进一步（层内或跨生成步）."""
        manager = self.state.ensure_trace_manager(self.runtime())
        return self.state.advance(manager)

    def reset(self, prompt: str, token_ids: list[int]) -> None:
        """重置追踪状态."""
        self.state.reset(prompt, token_ids)

    def render(self) -> InferenceTraceView | None:
        """渲染组件."""
        return self._render_fragment()

    @st.fragment
    def _render_fragment(self) -> InferenceTraceView | None:
        """主渲染片段（@st.fragment 允许局部更新而不刷新整个页面）.

        渲染流程：
        1. 显示标题和 CSS 样式
        2. 输入区：提示文本框 + Top-N 数选择
        3. 提示变化检测 → 必要时重置状态并 rerun
        4. 控制按钮：Next step / Reset
           - Next step：调用 advance()，若生成了新 token 则 rerun
           - Reset：清除状态并 rerun
        5. 渲染当前追踪视图（token 条、预测、注意力热图）
        """

        st.subheader("Inference Stepper")
        render_trace_css()

        # ---- 输入区 ----
        default_prompt = st.session_state.get(self.state.key("original_prompt"), "Once upon a time")
        prompt = st.text_area("Prompt", value=default_prompt, height=88, key=self.state.key("prompt_input"))
        top_n = int(
            st.number_input(
                "Top N",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                key=self.state.key("top_n"),
            )
        )

        # ---- 提示/模型切换检测 ----
        tokenizer, prompt_token_ids = self._tokenize_prompt(prompt)
        had_active_trace = self.state.manager is not None or self.state.generation_step >= 0
        reset_trace = self.state.ensure(prompt, prompt_token_ids)
        if reset_trace and had_active_trace:
            st.rerun(scope="app")  # 需要完整页面刷新以清除旧追踪

        # ---- 控制按钮 ----
        control_cols = st.columns([1, 1, 6])
        next_clicked = control_cols[0].button("Next step", type="primary", key=self.state.key("next"))
        reset_clicked = control_cols[1].button("Reset", key=self.state.key("reset"))

        if reset_clicked:
            self.reset(prompt, prompt_token_ids)
            st.rerun()

        if next_clicked:
            try:
                generated_new_token = self.advance()
            except Exception as exc:
                st.error(f"Inference step failed: {exc}")
            else:
                if generated_new_token:
                    st.rerun(scope="app")  # 新 token 生成后刷新以更新 token 条

        # ---- 渲染当前追踪 ----
        return self._render_current_trace(tokenizer, top_n)

    def _render_current_trace(self, tokenizer: Any, top_n: int) -> InferenceTraceView | None:
        """渲染当前步/层的追踪视图.

        依次渲染：
        1. Token 条：当前序列的所有 token（含生成 token 高亮）
        2. Top-k 预测：每个位置 LM 头的前 k 个候选
        3. 注意力热图：当前层所有 head 的 Query-Key 热图（3 列网格）
        4. 步进标题：generation step / layer 信息 + 本步生成的 token

        返回 InferenceTraceView 供嵌入投影组件使用。
        """

        manager = self.state.manager
        generation_step = self.state.generation_step
        layer_index = self.state.layer_index

        # 边界检查
        if manager is None or generation_step < 0 or generation_step >= len(manager.attention_weights):
            st.info("Click Next step to run the first generation step.")
            return None

        # 1. Token 条
        trace_token_ids = self.state.trace_input_token_ids[generation_step]
        trace_tokens = decode_token_ids(tokenizer, trace_token_ids)
        render_token_strip("Active trace input tokens", trace_tokens)

        # 2. Top-k 预测：先获取隐藏状态，再解码为预测
        hidden_state, apply_final_norm = hidden_state_for_layer(
            manager.hidden_states[generation_step],
            layer_index,
        )
        runtime = self.runtime()
        predictions = top_token_predictions(
            hidden_state,
            runtime.lm_head,
            runtime.tokenizer,
            top_n=top_n,
            apply_final_norm=apply_final_norm,
        )
        render_top_predictions(trace_tokens, predictions)

        # 3. 注意力热图
        self._render_attention_heads(
            manager.attention_weights[generation_step][layer_index],
            trace_tokens,
            layer_index,
        )

        # 4. 步进标题
        self._render_step_caption(generation_step, layer_index)

        # 构建并返回追踪视图快照
        return InferenceTraceView(
            tokenizer=tokenizer,
            manager=manager,
            model=runtime.model,
            lm_head=runtime.lm_head,
            generation_step=generation_step,
            layer_index=layer_index,
            top_n=top_n,
            trace_token_ids=list(trace_token_ids),
            num_layers=self.structure.num_layers,
            model_dir=self.model_dir,
        )

    def _render_attention_heads(self, attention, trace_tokens, layer_index: int) -> None:
        """渲染当前层的所有注意力头热图（3 列网格布局）."""

        show_attention = st.toggle(
            "Attention heads",
            value=False,
            key=self.state.key("show_attention_heads"),
            help="Render attention heatmaps for the current layer.",
        )
        if not show_attention:
            st.caption("Attention heads are collapsed. Enable them to render heatmaps.")
            return

        st.markdown("<div class='trace-section-label'>Attention heads</div>", unsafe_allow_html=True)
        head_count = int(attention.shape[0])
        columns_per_row = 3
        for start in range(0, head_count, columns_per_row):
            columns = st.columns(columns_per_row)
            for offset, column in enumerate(columns):
                head_index = start + offset
                if head_index >= head_count:
                    continue
                attention_matrix = select_attention_head(attention, head_index)
                column.plotly_chart(
                    attention_heatmap_figure(
                        attention_matrix,
                        trace_tokens,
                        head_index=head_index,
                        layer_index=layer_index,
                        height=320,
                    ),
                    width="stretch"
                )

    def _render_step_caption(self, generation_step: int, layer_index: int) -> None:
        """显示当前步进标题：generation step、层号、生成的 token."""

        generated_label = ""
        generated_tokens = self.state.generated_tokens
        if 0 <= generation_step < len(generated_tokens):
            token = generated_tokens[generation_step]
            generated_label = f" | generated: {token['token']} ({token['token_id']})"
        st.caption(
            f"Generation step {generation_step} | "
            f"layer {layer_index}/{self.structure.num_layers - 1}{generated_label}"
        )
