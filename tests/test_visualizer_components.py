from __future__ import annotations

from model_visualizer.app import render_app
from model_visualizer.ui_components.embedding_projection.component import (
    EmbeddingProjectionComponent,
    _clamp_layer_index,
    _trace_layer_count,
)
from model_visualizer.ui_components.inference import InferenceStepperComponent, InferenceTraceView
from model_visualizer.ui_components.structure import ModelStructureComponent
from model_visualizer.ui_components.structure.types import ModelStructure


def test_app_render_app_is_importable():
    assert callable(render_app)


def test_structure_component_loads_through_own_cache(monkeypatch):
    structure = ModelStructure(
        model_name="toy",
        model_dir="toy",
        infos=(),
        rectangles=(),
        total_params=0,
        num_layers=1,
    )

    monkeypatch.setattr(
        "model_visualizer.ui_components.structure.component.cached_structure_metadata",
        lambda model_dirs: ((structure,), ()),
    )

    component = ModelStructureComponent(("toy",))

    assert component.load() == ((structure,), ())


def test_inference_component_state_keys_are_prefixed():
    structure = ModelStructure(
        model_name="toy",
        model_dir="toy",
        infos=(),
        rectangles=(),
        total_params=0,
        num_layers=1,
    )

    component = InferenceStepperComponent(structure, state_prefix="custom")

    assert component.state.key("prompt_token_ids") == "custom_prompt_token_ids"


def test_embedding_projection_component_state_keys_are_prefixed():
    component = EmbeddingProjectionComponent(
        model_dir="toy",
        trace_view=None,
        state_prefix="projection",
    )

    assert component.key("scatter") == "projection_scatter"


def test_inference_trace_view_carries_current_trace_slice():
    view = InferenceTraceView(
        tokenizer="tokenizer",
        manager="manager",
        model="model",
        lm_head="lm_head",
        generation_step=2,
        layer_index=3,
        top_n=5,
        trace_token_ids=[1, 2],
        num_layers=8,
        model_dir="toy",
    )

    assert view.trace_token_ids == [1, 2]
    assert view.layer_index == 3


def test_embedding_projection_layer_helpers_use_trace_layers():
    class Manager:
        hidden_states = {0: [object(), object(), object(), object()]}

    view = InferenceTraceView(
        tokenizer="tokenizer",
        manager=Manager(),
        model="model",
        lm_head="lm_head",
        generation_step=0,
        layer_index=1,
        top_n=5,
        trace_token_ids=[1, 2],
        num_layers=8,
        model_dir="toy",
    )

    assert _trace_layer_count(view) == 3
    assert _clamp_layer_index(-4, 3) == 0
    assert _clamp_layer_index(9, 3) == 2
