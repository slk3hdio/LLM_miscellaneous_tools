from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import torch

from model_visualizer.ui_components.export.bundle import (
    build_demo_export_bundle,
    default_export_filename,
)
from model_visualizer.ui_components.export.types import ExportOptions
from model_visualizer.ui_components.inference.types import InferenceTraceView
from model_visualizer.ui_components.structure.types import ModelStructure


class FakeTokenizer:
    def decode(self, token_ids, **_kwargs):
        return "".join(f"tok{int(token_id)}" for token_id in token_ids)


class FakeLMHead:
    def decode(self, _hidden_state, apply_final_norm=None):
        del apply_final_norm
        return torch.tensor(
            [
                [0.1, 2.0, 1.0, 0.0],
                [0.1, 0.0, 3.0, 1.0],
            ]
        )


class FakeManager:
    attention_weights = [[
        torch.stack([torch.eye(2), torch.ones(2, 2) * 0.5]),
        torch.stack([torch.tril(torch.ones(2, 2)), torch.ones(2, 2) * 0.25]),
    ]]
    hidden_states = [[torch.zeros(2, 3), torch.ones(2, 3), torch.ones(2, 3) * 2]]


def _trace_view():
    return InferenceTraceView(
        tokenizer=FakeTokenizer(),
        manager=FakeManager(),
        model=object(),
        lm_head=FakeLMHead(),
        generation_step=0,
        layer_index=0,
        top_n=2,
        trace_token_ids=[1, 2],
        num_layers=2,
        model_dir="models/toy",
    )


def _structure():
    return ModelStructure(
        model_name="toy",
        model_dir="models/toy",
        infos=(),
        rectangles=(),
        total_params=0,
        num_layers=2,
    )


def test_demo_export_bundle_contains_static_trace_files():
    bundle = build_demo_export_bundle(
        structure=_structure(),
        trace_view=_trace_view(),
        options=ExportOptions(include_attention=True, include_projection=False),
    )

    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        names = set(archive.namelist())
        assert "manifest.json" in names
        assert "tokens.csv" in names
        assert "current_layer_top_predictions.csv" in names
        assert "arrays/current_layer_hidden_state.npy" in names
        assert "arrays/attention/layer_00_attention.npy" in names
        assert "arrays/attention/layer_01_attention.npy" in names
        assert "attention/layer_00_all_heads.html" in names
        assert "attention/layer_01_all_heads.html" in names
        assert "attention/layer_00_head_00.html" not in names
        assert "projection/token_state_projection.html" not in names

        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        assert manifest["model_name"] == "toy"
        assert manifest["generation_step"] == 0
        assert manifest["layer_index"] == 0
        assert manifest["attention_layers"] == 2
        assert manifest["attention_heads"] == 2

        hidden = np.load(io.BytesIO(archive.read("arrays/current_layer_hidden_state.npy")))
        assert hidden.shape == (2, 3)
        attention = np.load(io.BytesIO(archive.read("arrays/attention/layer_01_attention.npy")))
        assert attention.shape == (2, 2, 2)
        assert "tok1" in archive.read("tokens.csv").decode("utf-8")


def test_default_export_filename_uses_trace_position():
    assert default_export_filename(_trace_view()) == "toy_step_0_layer_0_demo_export.zip"
