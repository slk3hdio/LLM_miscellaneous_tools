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

def _load_tokenizer(model_path:str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers and torch are required for the local provider.") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return tokenizer
class LocalTransformersProvider(ModelProvider):
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
        text = output.strip()
        if not text:
            return ""

        def _to_plain_tool_calls(raw_calls: Any) -> str:
            items = raw_calls if isinstance(raw_calls, list) else [raw_calls]
            tool_calls: list[dict[str, Any]] = []
            for i, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                function = item.get("function") if isinstance(item.get("function"), dict) else item
                name = function.get("name")
                if not name:
                    continue
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                elif not isinstance(arguments, dict):
                    arguments = {}
                tool_calls.append({
                    "id": item.get("id", f"call_{i}"),
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                })
            return EvalSample.from_openai_tool_calls(tool_calls) if tool_calls else ""

        match = re.search(r"<tool_call(?:s)?>\s*(.*?)\s*</tool_call(?:s)?>", text, flags=re.DOTALL)
        if match:
            payload = re.sub(r"^```(?:json)?\s*|\s*```$", "", match.group(1).strip(), flags=re.DOTALL)
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                parsed = None
            if parsed is not None:
                plain_calls = _to_plain_tool_calls(parsed)
                if plain_calls:
                    return plain_calls

        if text.startswith("{") or text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                plain_calls = _to_plain_tool_calls(parsed.get("tool_calls") or parsed.get("function") or parsed)
                if plain_calls:
                    return plain_calls
                content = parsed.get("content")
                if isinstance(content, str):
                    return content.strip()
            elif isinstance(parsed, list):
                plain_calls = _to_plain_tool_calls(parsed)
                if plain_calls:
                    return plain_calls

        return re.sub(r"</?s>|<\|[^>]+\|>", "", text).strip()

    def generate(
        self,
        messages: list[EvalSample.Context],
        conversation_style: Literal['single', 'multi'],
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        model = self.get_model()  # ensures model is loaded and self.model_device is set

        if conversation_style == 'multi':
            encoded = self.get_tokenizer().apply_chat_template(messages, tokenize=True, return_tensors='pt', tools=tools, padding=True)
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
                    tools = test_tool,
                    tokenize=False
                )
                self._supports_tool_calling = True
            except Exception as e:
                self._supports_tool_calling = False
                self.logger.info(f'{self.model_path} doesn\'t support tool calling because:\n{e}')
        return self._supports_tool_calling
    
