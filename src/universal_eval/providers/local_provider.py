from __future__ import annotations

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

def _load_tokenizer(model_path:str, device):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers and torch are required for the local provider.") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_path, device_map=device, trust_remote_code=True)
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
        if self._model_ref is None:
            self._model_ref = _load_model(self.model_path, self.model_device)
        self.model_device = self._model_ref
        return self._model_ref
    
    def get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = _load_tokenizer(self.model_path, self.model_device)
        return self._tokenizer
    
    def format_output(self, output:str)->str:
        #TODO: parse response and tool call to plain text
        return ""

    def generate(
        self,
        messages: list[EvalSample.Context],
        # conversation_style: Literal['single', 'multi'],
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        # rendered_prompt = self._render_prompt(messages, tools=tools)
        rendered_prompt = self.get_tokenizer().apply_chat_complete(messages, tokenize=True, return_tensors='pt')

        encoded = self.get_tokenizer()(rendered_prompt, return_tensors="pt")
        encoded = {k: v.to(self.model_device) for k, v in encoded.items()}

        do_sample = self.temperature > 0
        outputs = self.get_model().generate( #type:ignore
            **encoded,
            max_new_tokens=self.max_new_tokens,
            temperature=max(self.temperature, 1e-5),
            do_sample=do_sample,
            pad_token_id=self.get_tokenizer().pad_token_id,
        )
        new_tokens = outputs[0][encoded["input_ids"].shape[1] :]
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
    
