import torch
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


class LMHead(torch.nn.Module):
    def __init__(self, model, apply_final_norm=True, device=None, dtype=None):
        super().__init__()
        self.apply_final_norm = apply_final_norm

        lm_head = self._get_lm_head(model)
        self.lm_head = lm_head
        self.lm_head.eval()
        self.device = device
        self.dtype = dtype

        self.final_norm = None
        if apply_final_norm:
            final_norm = self._get_final_norm(model)
            self.final_norm = final_norm
            self.final_norm.eval()

    @staticmethod
    def _get_lm_head(model):
        if hasattr(model, "get_output_embeddings"):
            lm_head = model.get_output_embeddings()
            if lm_head is not None:
                return lm_head
        if hasattr(model, "lm_head"):
            return model.lm_head
        raise ValueError("Model does not expose lm_head parameters.")

    @staticmethod
    def _get_final_norm(model):
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            return model.model.norm
        if hasattr(model, "norm"):
            return model.norm
        raise ValueError("Model does not expose final norm parameters.")

    @classmethod
    def from_pretrained(cls, model_path, apply_final_norm=True, **model_kwargs):
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        model.eval()
        return cls(model, apply_final_norm=apply_final_norm)

    @staticmethod
    def _module_device_dtype(module):
        dtype = None
        for tensor in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
            dtype = dtype or tensor.dtype
            if tensor.device.type != "meta":
                return tensor.device, tensor.dtype

        hook = getattr(module, "_hf_hook", None)
        execution_device = getattr(hook, "execution_device", None)
        if isinstance(execution_device, dict):
            execution_device = next(iter(execution_device.values()), None)
        if execution_device is not None:
            return torch.device(execution_device), dtype
        return None, dtype

    @staticmethod
    def _move_for_module(hidden_state, module, fallback_device=None, fallback_dtype=None):
        device, dtype = LMHead._module_device_dtype(module)
        device = device or fallback_device
        dtype = dtype or fallback_dtype
        if device is not None and getattr(device, "type", None) == "meta":
            device = None
        if device is None and dtype is None:
            return hidden_state
        move_kwargs = {}
        if device is not None:
            move_kwargs["device"] = device
        if dtype is not None:
            move_kwargs["dtype"] = dtype
        return hidden_state.to(**move_kwargs)

    def forward(self, hidden_state, apply_final_norm=None):
        if apply_final_norm is None:
            apply_final_norm = self.apply_final_norm
        if self.final_norm is not None and apply_final_norm:
            hidden_state = self._move_for_module(
                hidden_state,
                self.final_norm,
                fallback_device=self.device,
                fallback_dtype=self.dtype,
            )
            hidden_state = self.final_norm(hidden_state)
        hidden_state = self._move_for_module(
            hidden_state,
            self.lm_head,
            fallback_device=self.device,
            fallback_dtype=self.dtype,
        )
        return self.lm_head(hidden_state)

    @torch.no_grad()
    def decode(self, hidden_state, apply_final_norm=None):
        return self.forward(hidden_state, apply_final_norm=apply_final_norm)
