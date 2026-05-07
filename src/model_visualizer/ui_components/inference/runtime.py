from __future__ import annotations

"""推理步进器的模型运行时缓存."""

import logging
from dataclasses import dataclass
from typing import Any

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_runtime.lm_head import LMHead

logger = logging.getLogger(__name__)


CACHE_VERSION = 3  # 缓存版本号，修改后强制重建缓存
INFERENCE_RUNTIME_CACHE_VERSION = CACHE_VERSION


@dataclass(frozen=True)
class RuntimeBundle:
    """推理运行时所需的三个核心对象."""
    tokenizer: Any      # 分词器
    model: Any          # 因果语言模型
    lm_head: LMHead     # LM 头（将隐藏状态映射到词表）


@st.cache_resource(show_spinner=True)
def cached_tokenizer(model_dir: str):
    """缓存分词器（资源缓存，跨会话共享）."""
    logger.info("Loading tokenizer from %s", model_dir)
    return AutoTokenizer.from_pretrained(model_dir)


@st.cache_resource(show_spinner=True)
def cached_runtime(model_dir: str, cache_version: int) -> RuntimeBundle:
    """加载模型并缓存为 Streamlit 资源.

    使用 eager 注意力实现以获得每层注意力权重。
    如果模型不支持 eager 模式（TypeError），回退到默认实现。

    参数：
        model_dir: 模型目录
        cache_version: 缓存版本，变化时强制重新加载
    """

    del cache_version  # 仅用于缓存失效，不实际使用
    tokenizer = cached_tokenizer(model_dir)
    model_kwargs = {
        "torch_dtype": "auto",
        "device_map": "auto",
        "attn_implementation": "eager",  # 获取实际注意力权重，不是 flash attention
    }
    logger.info("Loading model from %s (attn_implementation=%s)", model_dir, model_kwargs["attn_implementation"])
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs)
    except TypeError:
        # 某些模型不支持 attn_implementation 参数
        logger.info("Model does not support eager attn_implementation, falling back to default")
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs)
    model.eval()
    logger.info("Model loaded successfully from %s", model_dir)
    return RuntimeBundle(
        tokenizer=tokenizer,
        model=model,
        lm_head=LMHead(model, apply_final_norm=True),
    )
