from __future__ import annotations

from types import SimpleNamespace

import torch

from model_runtime.lm_head import LMHead


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(norm=torch.nn.LayerNorm(2))
        self.lm_head = torch.nn.Linear(2, 3, bias=False)

    def get_output_embeddings(self):
        return self.lm_head


class MetaModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(norm=torch.nn.LayerNorm(2, device="meta"))
        self.lm_head = torch.nn.Linear(2, 3, bias=False, device="meta")

    def get_output_embeddings(self):
        return self.lm_head


def test_lm_head_decodes_with_model_modules():
    model = ToyModel()
    with torch.no_grad():
        model.model.norm.weight.fill_(1.0)
        model.model.norm.bias.zero_()
        model.lm_head.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ]
            )
        )

    head = LMHead(model, apply_final_norm=True)
    logits = head.decode(torch.tensor([[1.0, 3.0]]))

    assert logits.shape == (1, 3)


def test_lm_head_does_not_copy_meta_modules_on_init():
    model = MetaModel()

    head = LMHead(model, apply_final_norm=True)

    assert head.lm_head is model.lm_head
    assert head.final_norm is model.model.norm
