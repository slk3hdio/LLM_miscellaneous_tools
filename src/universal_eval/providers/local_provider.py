from __future__ import annotations

from typing import Dict, Any
from .model_provider import ModelProvider
import logging



class LocalTransformersProvider(ModelProvider):
    def __init__(
        self,
        model: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        device: str = "auto",
    ) -> None:
        super().__init__(model=model, max_new_tokens=max_new_tokens, temperature=temperature)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("transformers and torch are required for the local provider.") from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs:Dict[str, Any] = {"trust_remote_code": True}
        if device == "auto":
            model_kwargs["device_map"] = "auto"
        self.model_ref = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
        self.device = device
        self.logger = logging.getLogger(__name__)

    def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        rendered_prompt = self._render_prompt(messages)

        encoded = self.tokenizer(rendered_prompt, return_tensors="pt")
        if self.device == "auto":
            encoded = {k: v.to(self.model_ref.device) for k, v in encoded.items()}

        do_sample = self.temperature > 0
        outputs = self.model_ref.generate( #type:ignore
            **encoded,
            max_new_tokens=self.max_new_tokens,
            temperature=max(self.temperature, 1e-5),
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_tokens = outputs[0][encoded["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _render_prompt(self, messages: list[dict[str, str]]) -> str:
        if len(messages) == 1:
            return messages[0]["content"]
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            self.logger.info('Use tokenizer.apply_chat_template to format chat.')
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        lines = []
        self.logger.warning('cannot formate prompt')
        for message in messages:
            lines.append(f"{message['role'].upper()}: {message['content'].strip()}")
        lines.append("ASSISTANT:")
        return "\n".join(lines)
    
    @classmethod
    def from_config(cls):
        pass




