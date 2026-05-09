from __future__ import annotations

import json
import re
from typing import Any, Dict, Literal
from pathlib import Path
from universal_eval.datasets import EvalSample
from .model_provider import ModelProvider
import logging


def _load_model(model_path:str, device):
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError("transformers and torch are required for the local provider.") from exc
    model_ref = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
    return model_ref

def _ensure_tool_args_are_dicts(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """将消息中 tool_calls 的 JSON 字符串 arguments 转为 dict。

    某些 tokenizer 的 chat_template（如 qwen 3.5）会遍历 arguments|items，
    需要 arguments 是 dict 而非 JSON 字符串。
    """
    result: list[dict[str, Any]] = []
    for m in messages:
        tool_calls = m.get("tool_calls")
        if not tool_calls:
            result.append(m)
            continue
        fixed_calls: list[dict[str, Any]] = []
        for tc in tool_calls:
            tc = dict(tc)
            func = dict(tc.get("function", {}))
            args = func.get("arguments")
            if isinstance(args, str):
                try:
                    func["arguments"] = json.loads(args)
                except json.JSONDecodeError:
                    pass
            tc["function"] = func
            fixed_calls.append(tc)
        result.append({**m, "tool_calls": fixed_calls})
    return result


def _load_tokenizer(model_path:str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers and torch are required for the local provider.") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return tokenizer
class LocalTransformersProvider(ModelProvider):
    """基于本地 HuggingFace Transformers 模型的推理提供器。

    支持两种模式：
    - single: 将所有消息拼接为一段文本输入
    - multi:  使用 tokenizer.apply_chat_template 构建多轮对话格式
    """
    def __init__(
        self,
        model: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        device: str = "auto",
    ) -> None:
        super().__init__(model=model, max_new_tokens=max_new_tokens, temperature=temperature)
        self.model_device = device
        self.model_path = model
        self.logger = logging.getLogger(__name__)

        self._tokenizer = None
        self._model = None 
        self._supports_conversation_format = None
        self._supports_tool_calling = None

    def get_model(self):
        if self._model is None:
            self._model = _load_model(self.model_path, self.model_device)
        self.model_device = self._model.device
        return self._model
    
    def get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = _load_tokenizer(self.model_path)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer
    
    def format_output(self, output:str)->str:
        """将本地模型的原始输出解析为标准工具调用格式。

        按优先级尝试：``<tool_call>`` XML 标签 → 顶层 JSON → 纯文本回退。
        """
        text = output.strip()
        if not text:
            return ""

        match = re.search(r"<tool_call(?:s)?>\s*(.*?)\s*</tool_call(?:s)?>", text, flags=re.DOTALL)
        if match:
            payload = re.sub(r"^```(?:json)?\s*|\s*```$", "", match.group(1).strip(), flags=re.DOTALL)
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                parsed = None
            if parsed is not None:
                if isinstance(parsed, dict):
                    parsed = [parsed]
                plain_calls = EvalSample.normalize_raw_tool_calls(parsed)
                if plain_calls:
                    return plain_calls

        # 匹配 qwen 3.5 的 XML 参数格式: <function=Name><parameter=key>value</parameter></function>
        func_blocks = re.findall(
            r"<function=([^>]+)>\s*(.*?)</function>", text, flags=re.DOTALL
        )
        if func_blocks:
            calls: list[dict[str, Any]] = []
            for func_name, params_block in func_blocks:
                args: dict[str, str] = {}
                for pk, pv in re.findall(
                    r"<parameter=([^>]+)>\s*(.*?)</parameter>", params_block, flags=re.DOTALL
                ):
                    args[pk.strip()] = pv.strip()
                calls.append({"name": func_name.strip(), "arguments": args})
            from ..evaluator.parser_tools import format_call_string
            return format_call_string(calls)

        if text.startswith("{") or text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                plain_calls = EvalSample.normalize_raw_tool_calls(parsed.get("tool_calls") or parsed.get("function") or parsed)
                if plain_calls:
                    return plain_calls
                content = parsed.get("content")
                if isinstance(content, str):
                    return content.strip()
            elif isinstance(parsed, list):
                plain_calls = EvalSample.normalize_raw_tool_calls(parsed)
                if plain_calls:
                    return plain_calls

        return re.sub(r"</?s>|<\|[^>]+\|>", "", text).strip()

    def generate(
        self,
        messages: list[EvalSample.Context],
        conversation_style: Literal['single', 'multi'],
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """使用本地 Transformers 模型生成回复。

        single 模式将消息拼接为单段文本后直接 tokenize；
        multi 模式通过 apply_chat_template 构建多轮对话输入。
        """
        model = self.get_model()  # ensures model is loaded and self.model_device is set

        if conversation_style == 'multi':
            # qwen 3.5 模板会遍历 tool_calls.arguments|items，需要 dict 而非 JSON 字符串
            msgs = _ensure_tool_args_are_dicts(messages)
            try:
                encoded = self.get_tokenizer().apply_chat_template(msgs, tokenize=True, return_tensors='pt', tools=tools, padding=True)
            except TypeError:
                self.logger.warning("apply_chat_template failed with structured tool_calls, falling back to plain mode")
                text = "\n".join(msg.get("content", "") or "" for msg in messages)
                encoded = self.get_tokenizer()(text, return_tensors="pt", padding=True)
        else:
            text = "\n".join(msg.get("content", "") or "" for msg in messages)
            encoded = self.get_tokenizer()(text, return_tensors="pt", padding=True)
        encoded = {k: v.to(self.model_device) for k, v in encoded.items()}

        do_sample = self.temperature > 0
        outputs = model.generate( #type:ignore
            **encoded,
            max_new_tokens=self.max_new_tokens,
            temperature=max(self.temperature, 1e-5),
            do_sample=do_sample,
            pad_token_id=self.get_tokenizer().pad_token_id,
        )
        new_tokens = outputs[0][encoded["input_ids"].shape[1] :]
        if conversation_style == 'single':
            return self.get_tokenizer().decode(new_tokens, skip_special_tokens=True).strip()
        else:
            raw_output = self.get_tokenizer().decode(new_tokens, skip_special_tokens=False).strip()
            return self.format_output(raw_output)

    def supports_conversation_format(self) -> bool:
        if self._supports_conversation_format is None:
            test_messages = [
                {"role": "system", "content": "a"},
                {"role": "user", "content": "b"},
                {"role": "assistant", "content": "c"},
                {"role": "user", "content": "d"},
            ]
            try:
                prompt = self.get_tokenizer().apply_chat_template(
                    test_messages,
                    tokenize=False
                )
                self._supports_conversation_format = True
            except Exception as e:
                self.logger.info(f'{self.model_path} doesn\'t support conversation_format because:\n{e}')
                self._supports_conversation_format = False
        return self._supports_conversation_format

    def supports_tool_calling(self) -> bool:
        if self._supports_tool_calling is None:
            test_tool = {
                "type": "function",
                "function": {
                    "name": "a",
                    "description": "a",
                    "parameters": {
                        "type": "object",
                        "properties": {"p": {"type": "string"}},
                        "required": ["p"],
                    },
                },
            }
            test_message = [{"role": "user", "content": "a"}]
            try:
                prompt = self.get_tokenizer().apply_chat_template(
                    test_message,
                    tools = [test_tool],
                    tokenize=False
                )
                self._supports_tool_calling = True
            except Exception as e:
                self._supports_tool_calling = False
                self.logger.info(f'{self.model_path} doesn\'t support tool calling because:\n{e}')
        return self._supports_tool_calling
    
