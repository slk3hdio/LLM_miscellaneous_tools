from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import model_visualizer.histogram_stack as histogram_stack_module
from model_visualizer.analysis.files import (
    inspect_safetensors,
    parse_tensor_name,
    sample_tensor_values,
)
from model_visualizer.analysis.histograms import (
    available_layer_matrix_keys,
    compute_layer_histogram_stack,
)
from model_visualizer.ui_components.structure.layout import (
    build_structure_rectangles,
    build_structure_rectangles_for_models,
)
from model_visualizer.analysis.types import LayerHistogram, TensorInfo
from model_visualizer.histogram_stack import (
    default_grid_output_path,
    render_layer_histogram_stack_html,
    render_layer_histogram_stack_grid_html,
    rotation_post_script,
    safe_output_stem,
)
from model_visualizer.figures import layer_histogram_stack_figure
from model_visualizer.ui_components.structure.figures import tensor_rectangle_figure


def test_parse_tensor_name_for_qwen_layer_parameter():
    layer, module, parameter = parse_tensor_name(
        "model.layers.0.self_attn.q_proj.weight"
    )

    assert layer == 0
    assert module == "self_attn.q_proj"
    assert parameter == "weight"


def test_parse_tensor_name_for_embedding_parameter():
    layer, module, parameter = parse_tensor_name("model.embed_tokens.weight")

    assert layer is None
    assert module == "model.embed_tokens"
    assert parameter == "weight"


def test_sampling_is_bounded_and_does_not_mutate_tensor():
    tensor = torch.arange(10_000, dtype=torch.float32)
    before = tensor.clone()

    sample = sample_tensor_values(tensor, max_values=128)

    assert sample.shape == (128,)
    assert torch.equal(tensor, before)


def test_build_structure_rectangles_uses_schematic_data_flow_layout():
    infos = [
        TensorInfo("model.embed_tokens.weight", (100, 8), "float32", 800, None, "model.embed_tokens", "weight", "x"),
        TensorInfo("model.layers.0.self_attn.q_proj.bias", (8,), "float32", 8, 0, "self_attn.q_proj", "bias", "x"),
        TensorInfo("model.layers.0.input_layernorm.weight", (8,), "float32", 8, 0, "input_layernorm", "weight", "x"),
        TensorInfo("model.layers.0.self_attn.q_proj.weight", (8, 8), "float32", 64, 0, "self_attn.q_proj", "weight", "x"),
        TensorInfo("model.layers.0.self_attn.k_proj.weight", (4, 8), "float32", 32, 0, "self_attn.k_proj", "weight", "x"),
        TensorInfo("model.layers.0.self_attn.v_proj.weight", (4, 8), "float32", 32, 0, "self_attn.v_proj", "weight", "x"),
        TensorInfo("model.layers.0.self_attn.o_proj.weight", (8, 8), "float32", 64, 0, "self_attn.o_proj", "weight", "x"),
        TensorInfo("model.layers.0.mlp.gate_proj.weight", (16, 8), "float32", 128, 0, "mlp.gate_proj", "weight", "x"),
        TensorInfo("model.layers.0.mlp.up_proj.weight", (16, 8), "float32", 128, 0, "mlp.up_proj", "weight", "x"),
        TensorInfo("model.layers.0.mlp.down_proj.weight", (8, 16), "float32", 128, 0, "mlp.down_proj", "weight", "x"),
    ]

    rectangles = build_structure_rectangles(infos)
    by_name = {rectangle.name: rectangle for rectangle in rectangles}

    assert "model.embed_tokens.weight" not in by_name
    assert "model.layers.0.self_attn.q_proj.bias" not in by_name
    assert "model.layers.0.input_layernorm.weight" not in by_name
    assert by_name["model.layers.0.self_attn.q_proj.weight"].flow == "qkv"
    assert by_name["model.layers.0.self_attn.q_proj.weight"].x == by_name["model.layers.0.self_attn.k_proj.weight"].x
    assert by_name["model.layers.0.self_attn.k_proj.weight"].x == by_name["model.layers.0.self_attn.v_proj.weight"].x
    qkv_center = sum(
        by_name[name].y
        for name in [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
        ]
    ) / 3
    assert np.isclose(qkv_center, 0.0)
    assert by_name["model.layers.0.mlp.gate_proj.weight"].flow == "mlp_up_gate"
    assert by_name["model.layers.0.mlp.gate_proj.weight"].x == by_name["model.layers.0.mlp.up_proj.weight"].x
    assert by_name["model.layers.0.mlp.gate_proj.weight"].y < by_name["model.layers.0.mlp.up_proj.weight"].y
    assert by_name["model.layers.0.self_attn.o_proj.weight"].x < by_name["model.layers.0.mlp.down_proj.weight"].x
    assert by_name["model.layers.0.mlp.down_proj.weight"].cols == 16
    assert np.isclose(
        by_name["model.layers.0.mlp.gate_proj.weight"].y
        + by_name["model.layers.0.mlp.up_proj.weight"].y,
        0.0,
    )
    assert np.isclose(
        by_name["model.layers.0.mlp.gate_proj.weight"].height
        / by_name["model.layers.0.mlp.gate_proj.weight"].width,
        2.0,
    )


def test_tensor_rectangles_continue_next_layer_in_same_direction():
    infos = [
        TensorInfo("model.layers.0.mlp.down_proj.weight", (8, 16), "float32", 128, 0, "mlp.down_proj", "weight", "x"),
        TensorInfo("model.layers.1.self_attn.q_proj.weight", (8, 8), "float32", 64, 1, "self_attn.q_proj", "weight", "x"),
    ]
    rectangles = build_structure_rectangles(infos)
    by_name = {rectangle.name: rectangle for rectangle in rectangles}

    assert by_name["model.layers.0.mlp.down_proj.weight"].x < by_name["model.layers.1.self_attn.q_proj.weight"].x
    assert by_name["model.layers.1.self_attn.q_proj.weight"].x - by_name["model.layers.0.mlp.down_proj.weight"].x >= 10.0


def test_multi_model_structure_aligns_low_layers_and_uses_global_scale():
    small_infos = [
        TensorInfo("model.layers.0.self_attn.q_proj.weight", (4, 4), "float32", 16, 0, "self_attn.q_proj", "weight", "small"),
        TensorInfo("model.layers.0.self_attn.k_proj.weight", (4, 4), "float32", 16, 0, "self_attn.k_proj", "weight", "small"),
        TensorInfo("model.layers.0.self_attn.v_proj.weight", (4, 4), "float32", 16, 0, "self_attn.v_proj", "weight", "small"),
    ]
    large_infos = [
        TensorInfo("model.layers.0.self_attn.q_proj.weight", (8, 8), "float32", 64, 0, "self_attn.q_proj", "weight", "large"),
        TensorInfo("model.layers.0.self_attn.k_proj.weight", (8, 8), "float32", 64, 0, "self_attn.k_proj", "weight", "large"),
        TensorInfo("model.layers.0.self_attn.v_proj.weight", (8, 8), "float32", 64, 0, "self_attn.v_proj", "weight", "large"),
        TensorInfo("model.layers.1.self_attn.q_proj.weight", (8, 8), "float32", 64, 1, "self_attn.q_proj", "weight", "large"),
    ]

    structures = build_structure_rectangles_for_models(
        [
            ("small", "small", small_infos),
            ("large", "large", large_infos),
        ]
    )
    small_rectangles = {rectangle.name: rectangle for rectangle in structures[0].rectangles}
    large_rectangles = {rectangle.name: rectangle for rectangle in structures[1].rectangles}

    assert small_rectangles["model.layers.0.self_attn.q_proj.weight"].x == large_rectangles["model.layers.0.self_attn.q_proj.weight"].x
    assert small_rectangles["model.layers.0.self_attn.q_proj.weight"].model_lane != large_rectangles["model.layers.0.self_attn.q_proj.weight"].model_lane
    assert large_rectangles["model.layers.0.self_attn.q_proj.weight"].width > small_rectangles["model.layers.0.self_attn.q_proj.weight"].width
    small_qkv_center = sum(rectangle.y for rectangle in structures[0].rectangles[:3]) / 3
    large_qkv_center = sum(rectangle.y for rectangle in list(structures[1].rectangles)[:3]) / 3
    assert np.isclose(small_qkv_center, structures[0].rectangles[0].model_lane)
    assert np.isclose(large_qkv_center, structures[1].rectangles[0].model_lane)
    assert max(rectangle.layer for rectangle in structures[1].rectangles) == 1


def test_available_layer_matrix_keys_filters_layer_2d_weights():
    infos = [
        TensorInfo("model.layers.0.self_attn.q_proj.weight", (2, 2), "float32", 4, 0, "self_attn.q_proj", "weight", "x"),
        TensorInfo("model.layers.0.self_attn.q_proj.bias", (2,), "float32", 2, 0, "self_attn.q_proj", "bias", "x"),
        TensorInfo("model.embed_tokens.weight", (10, 2), "float32", 20, None, "model.embed_tokens", "weight", "x"),
    ]

    assert available_layer_matrix_keys(infos) == ["self_attn.q_proj.weight"]


def test_compute_layer_histogram_stack_uses_shared_bins(monkeypatch):
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
        "model.layers.1.self_attn.q_proj.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "model.layers.0.mlp.down_proj.weight": torch.ones(2, 2),
    }
    infos = [
        TensorInfo(name, tuple(tensor.shape), "float32", tensor.numel(), layer, module, "weight", "x")
        for name, tensor, layer, module in [
            ("model.layers.0.self_attn.q_proj.weight", tensors["model.layers.0.self_attn.q_proj.weight"], 0, "self_attn.q_proj"),
            ("model.layers.1.self_attn.q_proj.weight", tensors["model.layers.1.self_attn.q_proj.weight"], 1, "self_attn.q_proj"),
            ("model.layers.0.mlp.down_proj.weight", tensors["model.layers.0.mlp.down_proj.weight"], 0, "mlp.down_proj"),
        ]
    ]
    histograms = compute_layer_histogram_stack(
        infos,
        "self_attn.q_proj.weight",
        bins=4,
        max_values_per_layer=100,
        density=True,
        tensor_loader=lambda _file_path, tensor_name: tensors[tensor_name],
    )

    assert [histogram.layer for histogram in histograms] == [0, 1]
    assert histograms[0].bin_centers == histograms[1].bin_centers
    bin_width = histograms[0].bin_centers[1] - histograms[0].bin_centers[0]
    assert np.isclose(sum(histograms[0].values) * bin_width, 1.0)


def test_histogram_stack_safe_output_stem():
    assert (
        safe_output_stem("models/qwen_2_5_1_5b", "self_attn.q_proj.weight")
        == "qwen_2_5_1_5b_self_attn_q_proj_weight"
    )


def test_histogram_stack_default_grid_output_path():
    path = default_grid_output_path(
        "models/qwen_2_5_1_5b",
        ["self_attn.q_proj.weight", "mlp.up_proj.weight"],
    )

    assert path.name == "qwen_2_5_1_5b_self_attn_q_proj_weight_mlp_up_proj_weight.html"


def test_histogram_stack_rotation_script_targets_plotly_scene():
    script = rotation_post_script()

    assert "document.getElementById('{plot_id}')" in script
    assert "scene.camera.eye" in script
    assert "Math.hypot(initialEye.x, initialEye.z)" in script
    assert "z: radius * Math.sin(angle)" in script
    assert "setInterval" in script


def test_layer_histogram_stack_figure_renders_bar_columns():
    figure = layer_histogram_stack_figure(
        [
            LayerHistogram(
                0,
                "model.layers.0.self_attn.q_proj.weight",
                (0.0, 1.0),
                (0.25, 0.75),
            ),
            LayerHistogram(
                1,
                "model.layers.1.self_attn.q_proj.weight",
                (0.0, 1.0),
                (0.4, 0.6),
            )
        ]
    )

    trace = figure.data[0]
    second_trace = figure.data[1]

    assert figure.layout.title.text == "Layer histogram bar stack"
    assert trace.type == "mesh3d"
    assert len(trace.x) == 16
    assert len(trace.i) == 24
    assert list(trace.x[:8]) == [-0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5]
    assert list(trace.y[:8]) == [0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.25, 0.25]
    assert list(trace.z[:8]) == [-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5]
    assert list(trace.customdata[0]) == [0.0, 0.25, 0.0]
    assert trace.vertexcolor[0] != trace.vertexcolor[2]
    assert second_trace.vertexcolor[0] != trace.vertexcolor[0]
    assert list(second_trace.z[:8]) == [0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5]
    assert figure.layout.scene.zaxis.tickvals == (0.0, 1.0)
    assert figure.layout.scene.zaxis.ticktext == ("0", "1")
    assert figure.layout.scene.camera.up.y == 1
    assert figure.layout.scene.camera.eye.x == 1.8
    assert figure.layout.scene.camera.eye.y == 1.15
    assert figure.layout.scene.camera.eye.z == 1.8


def test_render_layer_histogram_stack_html_writes_file(monkeypatch):
    infos = [
        TensorInfo("model.layers.0.self_attn.q_proj.weight", (2, 2), "float32", 4, 0, "self_attn.q_proj", "weight", "x"),
        TensorInfo("model.layers.1.self_attn.q_proj.weight", (2, 2), "float32", 4, 1, "self_attn.q_proj", "weight", "x"),
    ]
    histograms = [
        LayerHistogram(0, infos[0].name, (0.0, 1.0), (0.5, 0.5)),
        LayerHistogram(1, infos[1].name, (0.0, 1.0), (0.25, 0.75)),
    ]
    monkeypatch.setattr(histogram_stack_module, "list_safetensors_files", lambda _model_dir: [Path("x")])
    monkeypatch.setattr(histogram_stack_module, "inspect_safetensors", lambda _files: infos)
    monkeypatch.setattr(histogram_stack_module, "compute_layer_histogram_stack", lambda *_args, **_kwargs: histograms)
    output_path = Path("outputs/model_visualizer/test_histogram_stack.html")
    output_path.unlink(missing_ok=True)

    rendered = render_layer_histogram_stack_html(
        "toy_model",
        "self_attn.q_proj.weight",
        bins=4,
        output=output_path,
    )

    assert rendered == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert "scene.camera.eye" in output_path.read_text(encoding="utf-8")
    output_path.unlink(missing_ok=True)


def test_render_layer_histogram_stack_grid_html_writes_multiple_figures(monkeypatch):
    infos = [
        TensorInfo("model.layers.0.self_attn.q_proj.weight", (2, 2), "float32", 4, 0, "self_attn.q_proj", "weight", "x"),
        TensorInfo("model.layers.0.mlp.up_proj.weight", (2, 2), "float32", 4, 0, "mlp.up_proj", "weight", "x"),
    ]
    histograms_by_key = {
        "self_attn.q_proj.weight": [
            LayerHistogram(0, infos[0].name, (0.0, 1.0), (0.5, 0.5))
        ],
        "mlp.up_proj.weight": [
            LayerHistogram(0, infos[1].name, (0.0, 1.0), (0.25, 0.75))
        ],
    }
    calls = []
    monkeypatch.setattr(histogram_stack_module, "list_safetensors_files", lambda _model_dir: [Path("x")])
    monkeypatch.setattr(histogram_stack_module, "inspect_safetensors", lambda _files: infos)

    def fake_compute(_infos, matrix_key, **_kwargs):
        calls.append(matrix_key)
        return histograms_by_key[matrix_key]

    monkeypatch.setattr(histogram_stack_module, "compute_layer_histogram_stack", fake_compute)
    output_path = Path("outputs/model_visualizer/test_histogram_stack_grid.html")
    output_path.unlink(missing_ok=True)

    rendered = render_layer_histogram_stack_grid_html(
        "toy_model",
        ["self_attn.q_proj.weight", "mlp.up_proj.weight"],
        bins=4,
        output=output_path,
    )
    html = output_path.read_text(encoding="utf-8")

    assert rendered == output_path
    assert calls == ["self_attn.q_proj.weight", "mlp.up_proj.weight"]
    assert "self_attn.q_proj.weight" in html
    assert "mlp.up_proj.weight" in html
    assert html.count("scene.camera.eye") >= 2
    assert html.count('<section class="plot-card">') == 2
    output_path.unlink(missing_ok=True)


def test_tensor_rectangle_figure_adds_centerline_and_split_branches():
    infos = [
        TensorInfo("model.layers.0.self_attn.q_proj.weight", (8, 8), "float32", 64, 0, "self_attn.q_proj", "weight", "x"),
        TensorInfo("model.layers.0.self_attn.k_proj.weight", (8, 8), "float32", 64, 0, "self_attn.k_proj", "weight", "x"),
        TensorInfo("model.layers.0.self_attn.v_proj.weight", (8, 8), "float32", 64, 0, "self_attn.v_proj", "weight", "x"),
        TensorInfo("model.layers.0.self_attn.o_proj.weight", (8, 8), "float32", 64, 0, "self_attn.o_proj", "weight", "x"),
    ]

    figure = tensor_rectangle_figure(build_structure_rectangles(infos))
    trace_names = [trace.name for trace in figure.data]

    assert "model data flow" in trace_names
    assert trace_names.count("split flow") == 1
    assert "model flow direction" in trace_names
    assert trace_names.count("matrix hover targets") == 1


def test_inspect_local_qwen_safetensors_when_available():
    file_path = Path("models/qwen_2_5_1_5b/model.safetensors")
    if not file_path.exists():
        pytest.skip("local Qwen safetensors file is not available")

    infos = inspect_safetensors([file_path])

    assert len(infos) == 338
    assert len({info.layer for info in infos if info.layer is not None}) == 28
