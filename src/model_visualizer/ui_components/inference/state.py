from __future__ import annotations

"""推理步进器的 Streamlit 会话状态管理器.

InferenceState 封装了所有会话键的读写，确保状态在 Streamlit 的
rerun 之间持久化且不会跨组件污染。
"""

import logging

import streamlit as st

from model_runtime.saver import TraceManager
from model_visualizer.ui_components.inference.trace import next_inference_position
from model_visualizer.ui_components.inference.runtime import CACHE_VERSION, RuntimeBundle
from model_visualizer.ui_components.structure.types import ModelStructure

logger = logging.getLogger(__name__)


class InferenceState:
    """管理一个推理组件实例的所有 Streamlit 会话键.

    每个 InferenceStepperComponent 持有自己的 InferenceState，
    通过前缀（prefix）区分不同实例的状态键。
    """

    def __init__(self, structure: ModelStructure, *, prefix: str = "inference"):
        self.structure = structure
        self.model_dir = structure.model_dir
        self.prefix = prefix

    def key(self, name: str) -> str:
        """生成带前缀的会话键名."""
        return f"{self.prefix}_{name}"

    def cleanup_trace_manager(self) -> None:
        """清理旧的 TraceManager 钩子（防止内存泄漏）."""
        manager = st.session_state.get(self.key("trace_manager"))
        if manager is not None:
            manager.clean_all_hooks()

    def reset(self, prompt: str, token_ids: list[int]) -> None:
        """重置所有状态到初始值.

        步骤：
        1. 清理旧 TraceManager
        2. 写入模型目录、提示文本、token ID
        3. 重置 generation_step 为 -1（表示未开始）、layer_index 为 0
        4. 清空已生成 token 列表、追踪输入列表
        5. TraceManager 置为 None（下次 advance 时惰性创建）
        """

        logger.info("Resetting inference state for model=%s, prompt_len=%d", self.model_dir, len(token_ids))
        self.cleanup_trace_manager()
        st.session_state[self.key("model_dir")] = self.model_dir
        st.session_state[self.key("original_prompt")] = prompt
        st.session_state[self.key("current_prompt")] = prompt
        st.session_state[self.key("prompt_token_ids")] = list(token_ids)
        st.session_state[self.key("generation_step")] = -1
        st.session_state[self.key("layer_index")] = 0
        st.session_state[self.key("generated_tokens")] = []
        st.session_state[self.key("trace_input_token_ids")] = []
        st.session_state[self.key("trace_manager")] = None
        st.session_state[self.key("runtime_cache_version")] = CACHE_VERSION

    def ensure(self, prompt: str, token_ids: list[int]) -> bool:
        """确保会话状态与当前组件配置一致.

        检测模型目录、提示文本或缓存版本是否变化。
        若不一致则触发 reset 并返回 True（表示需要 rerun）。
        """

        stored_model_dir = st.session_state.get(self.key("model_dir"))
        stored_prompt = st.session_state.get(self.key("original_prompt"))
        stored_cache_version = st.session_state.get(self.key("runtime_cache_version"))
        if (
            stored_model_dir != self.model_dir
            or stored_prompt != prompt
            or stored_cache_version != CACHE_VERSION
        ):
            self.reset(prompt, token_ids)
            return True
        return False

    def ensure_trace_manager(self, runtime: RuntimeBundle) -> TraceManager:
        """惰性创建 TraceManager.

        仅在首次调用或 reset 后创建，之后复用缓存。
        """

        manager = st.session_state.get(self.key("trace_manager"))
        if manager is None:
            logger.info("Creating new TraceManager for %s", self.model_dir)
            manager = TraceManager(runtime.model, runtime.tokenizer)
            manager.set_prompt(st.session_state[self.key("original_prompt")])
            st.session_state[self.key("trace_manager")] = manager
        return manager

    def advance(self, manager: TraceManager) -> bool:
        """执行一步推理前进.

        详细步骤：
        1. 从会话中读取当前 generation_step 和 layer_index
        2. 判断当前是否有有效的追踪数据
        3. 调用 next_inference_position 计算下一步目标
        4. 如果需要生成（should_generate），调用 manager.step() 执行一次解码
           - 保存当前 token_ids 快照
           - 追加生成的 token
           - 更新提示 token 列表
        5. 写入新的 generation_step 和 layer_index 到会话
        6. 返回是否生成了新 token（用于决定是否需要 st.rerun）
        """

        generation_step = int(st.session_state[self.key("generation_step")])
        layer_index = int(st.session_state[self.key("layer_index")])
        # 判断当前步是否有追踪数据
        has_current_trace = 0 <= generation_step < len(manager.attention_weights)
        num_layers = (
            len(manager.attention_weights[generation_step])
            if has_current_trace
            else self.structure.num_layers
        )

        # 计算下一步目标
        advance = next_inference_position(
            generation_step,
            layer_index,
            num_layers,
            has_trace=has_current_trace,
        )

        # 如果需要生成：执行一步解码
        if advance.should_generate:
            logger.debug(
                "Generating token at step=%d, current token count=%d",
                advance.generation_step, len(manager.token_ids),
            )
            trace_input_token_ids = list(manager.token_ids)
            step_info = manager.step()
            st.session_state[self.key("trace_input_token_ids")].append(trace_input_token_ids)
            st.session_state[self.key("generated_tokens")].append(
                {
                    "token_id": int(step_info["token_id"].item()),
                    "token": step_info["token"],
                }
            )
            # 更新提示 token 列表（追加了生成 token）
            st.session_state[self.key("prompt_token_ids")] = list(manager.token_ids)
            st.session_state[self.key("current_prompt")] = manager.current_prompt
            logger.info(
                "Generated token '%s' (id=%d) at generation_step=%d",
                step_info["token"], int(step_info["token_id"].item()), advance.generation_step,
            )

        # 更新步进位置
        st.session_state[self.key("generation_step")] = advance.generation_step
        st.session_state[self.key("layer_index")] = advance.layer_index
        return advance.should_generate

    # ---- 便捷属性（从会话读取） ----

    @property
    def generation_step(self) -> int:
        return int(st.session_state.get(self.key("generation_step"), -1))

    @property
    def layer_index(self) -> int:
        return int(st.session_state.get(self.key("layer_index"), 0))

    @property
    def manager(self):
        return st.session_state.get(self.key("trace_manager"))

    @property
    def trace_input_token_ids(self):
        return st.session_state.get(self.key("trace_input_token_ids"), [])

    @property
    def generated_tokens(self):
        return st.session_state.get(self.key("generated_tokens"), [])

    def current_token_ids(self, fallback: list[int]) -> list[int]:
        return st.session_state.get(self.key("prompt_token_ids"), fallback)
