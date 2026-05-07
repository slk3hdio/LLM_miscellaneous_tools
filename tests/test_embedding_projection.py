from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import save_file

from model_visualizer.ui_components.embedding_projection.projection import (
    compute_embedding_projection_basis,
    load_projection_basis,
    normalize_hidden_for_projection,
    project_token_embeddings,
    project_hidden_state_layers,
    project_hidden_states,
    save_projection_basis,
    stabilize_component_signs,
    top_prediction_token_ids,
)
from model_visualizer.ui_components.embedding_projection.local_umap import (
    compute_local_umap_projection,
    dot_product_distance_matrix,
    exp_negative_dot_product_distance_matrix,
    final_prediction_token_ids,
)
from model_visualizer.ui_components.embedding_projection.figures import (
    _axis_range,
    _cube_ranges,
    _layer_label,
    _ranges_for_points,
    animated_token_projection_figure,
)
from model_visualizer.ui_components.embedding_projection.types import LayerProjection, ProjectedToken


class FakeTokenizer:
    def decode(self, token_ids, **_kwargs):
        return "".join(f"tok{int(token_id)}" for token_id in token_ids)


class FakeLMHead:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits
        self.apply_final_norm_values = []

    def decode(self, _hidden_state, apply_final_norm=None):
        self.apply_final_norm_values.append(apply_final_norm)
        return self.logits


class AddOneNorm(torch.nn.Module):
    def forward(self, hidden_state):
        return hidden_state + 1.0


def load_precompute_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "precompute_embedding_projection.py"
    spec = importlib.util.spec_from_file_location("precompute_embedding_projection", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_embedding_projection_basis_shapes_and_orthogonal_components():
    embedding = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 1.0],
        ]
    )

    basis = compute_embedding_projection_basis(embedding, model_name="toy", dimensions=3)

    assert basis.mean.shape == (3,)
    assert basis.components.shape == (3, 3)
    assert basis.explained_variance.shape == (3,)
    assert basis.explained_variance_ratio.shape == (3,)
    product = basis.components.T @ basis.components
    assert np.allclose(product, np.eye(3), atol=1e-5)


def test_embedding_projection_basis_can_be_2d():
    basis = compute_embedding_projection_basis(torch.eye(4), model_name="toy", dimensions=2)

    assert basis.components.shape == (4, 2)
    assert basis.explained_variance_ratio.shape == (2,)


def test_component_sign_stabilization_makes_largest_dimension_positive():
    components = torch.tensor(
        [
            [0.1, 0.2],
            [-0.9, 0.3],
            [0.4, -0.8],
        ]
    )

    stable = stabilize_component_signs(components)

    assert stable[1, 0] > 0
    assert stable[2, 1] > 0


def test_projection_basis_round_trips_npz(tmp_path):
    basis = compute_embedding_projection_basis(torch.eye(4), model_name="toy")
    path = save_projection_basis(basis, tmp_path / "toy_pca.npz")

    loaded = load_projection_basis(path)

    assert loaded.path == path
    assert loaded.model_name == "toy"
    assert loaded.vocab_size == 4
    assert loaded.hidden_size == 4
    assert np.allclose(loaded.mean, basis.mean)
    assert np.allclose(loaded.components, basis.components)


def test_project_hidden_states_preserves_token_order_and_count():
    basis = compute_embedding_projection_basis(torch.eye(4), model_name="toy")

    points = project_hidden_states(
        torch.eye(4)[:3],
        [3, 1, 3],
        FakeTokenizer(),
        basis,
    )

    assert [point.index for point in points] == [0, 1, 2]
    assert [point.token_id for point in points] == [3, 1, 3]
    assert [point.text for point in points] == ["tok3", "tok1", "tok3"]
    assert all(isinstance(point.z, float) for point in points)


def test_project_hidden_states_uses_zero_z_for_2d_basis():
    basis = compute_embedding_projection_basis(torch.eye(4), model_name="toy", dimensions=2)

    points = project_hidden_states(torch.eye(4)[:2], [0, 1], FakeTokenizer(), basis)

    assert [point.z for point in points] == [0.0, 0.0]


def test_project_token_embeddings_projects_selected_vocab_rows():
    basis = compute_embedding_projection_basis(torch.eye(4), model_name="toy")

    points = project_token_embeddings(torch.eye(4), [3, 1], FakeTokenizer(), basis)

    assert [point.token_id for point in points] == [3, 1]
    assert [point.text for point in points] == ["tok3", "tok1"]


def test_top_prediction_token_ids_returns_ordered_unique_ids():
    logits = torch.tensor(
        [
            [0.1, 5.0, 4.0, 0.0],
            [0.1, 3.0, 6.0, 4.0],
        ]
    )
    lm_head = FakeLMHead(logits)

    token_ids = top_prediction_token_ids(
        torch.zeros(2, 3),
        lm_head,
        top_k=2,
        apply_final_norm=False,
    )

    assert token_ids == [1, 2, 3]
    assert lm_head.apply_final_norm_values == [False]


def test_normalize_hidden_for_projection_applies_lm_head_final_norm():
    hidden_state = torch.zeros(2, 3)
    lm_head = SimpleNamespace(final_norm=AddOneNorm())

    normalized = normalize_hidden_for_projection(
        hidden_state,
        lm_head,
        apply_final_norm=True,
    )

    assert torch.equal(normalized, torch.ones(2, 3))


def test_project_hidden_state_layers_returns_one_projection_per_layer():
    basis = compute_embedding_projection_basis(torch.eye(3), model_name="toy")
    hidden_states = [
        torch.zeros(2, 3),
        torch.eye(3)[:2],
        torch.ones(2, 3),
    ]

    layers = project_hidden_state_layers(
        hidden_states,
        [0, 1],
        FakeTokenizer(),
        basis,
        layer_count=2,
    )

    assert [layer.layer_index for layer in layers] == [0, 1]
    assert [len(layer.points) for layer in layers] == [2, 2]
    assert layers[0].points[1].token_id == 1
    assert layers[0].top_prediction_points == []


def test_project_hidden_state_layers_can_include_top_prediction_embedding_points():
    basis = compute_embedding_projection_basis(torch.eye(4), model_name="toy")
    hidden_states = [
        torch.zeros(2, 4),
        torch.eye(4)[:2],
        torch.ones(2, 4),
    ]
    lm_head = FakeLMHead(
        torch.tensor(
            [
                [0.0, 4.0, 3.0, 2.0],
                [0.0, 1.0, 5.0, 4.0],
            ]
        )
    )

    layers = project_hidden_state_layers(
        hidden_states,
        [0, 1],
        FakeTokenizer(),
        basis,
        layer_count=2,
        embedding_weight=torch.eye(4),
        lm_head=lm_head,
        top_k=2,
    )

    assert [point.token_id for point in layers[0].top_prediction_points] == [1, 2, 3]
    assert [point.current_token_id for point in layers[0].points] == [1, 2]
    assert [point.token_id for point in layers[0].best_prediction_points] == [1, 2]
    assert [point.index for point in layers[0].best_prediction_points] == [0, 1]
    assert len(lm_head.apply_final_norm_values) == 4


def test_project_hidden_state_layers_can_include_embedding_frame():
    basis = compute_embedding_projection_basis(torch.eye(3), model_name="toy")

    layers = project_hidden_state_layers(
        [torch.zeros(2, 3), torch.eye(3)[:2]],
        [0, 1],
        FakeTokenizer(),
        basis,
        layer_count=1,
        include_embedding_layer=True,
    )

    assert [layer.layer_index for layer in layers] == [-1, 0]
    assert layers[0].top_prediction_points == []
    assert layers[0].best_prediction_points == []
    assert [point.token_id for point in layers[0].points] == [0, 1]


def test_project_hidden_state_layers_projects_normalized_hidden_states():
    basis = compute_embedding_projection_basis(torch.eye(3), model_name="toy")
    hidden_states = [
        torch.zeros(2, 3),
        torch.zeros(2, 3),
        torch.zeros(2, 3),
    ]
    lm_head = SimpleNamespace(final_norm=AddOneNorm())

    layers = project_hidden_state_layers(
        hidden_states,
        [0, 1],
        FakeTokenizer(),
        basis,
        layer_count=1,
        lm_head=lm_head,
    )
    expected_points = project_hidden_states(
        torch.ones(2, 3),
        [0, 1],
        FakeTokenizer(),
        basis,
    )

    assert [(point.x, point.y, point.z) for point in layers[0].points] == [
        (point.x, point.y, point.z) for point in expected_points
    ]


def test_animated_projection_figure_contains_browser_frames():
    basis = compute_embedding_projection_basis(torch.eye(3), model_name="toy")
    layers = project_hidden_state_layers(
        [torch.zeros(2, 3), torch.eye(3)[:2], torch.ones(2, 3)],
        [0, 1],
        FakeTokenizer(),
        basis,
        layer_count=2,
    )

    figure = animated_token_projection_figure(layers, basis, initial_layer_index=1)

    assert len(figure.frames) == 2
    assert len(figure.data) == 8
    assert len(figure.frames[0].data) == 5
    assert figure.layout.sliders[0].active == 1
    assert figure.layout.updatemenus[0].buttons[0].label == "Play"
    assert figure.data[0].type == "scatter3d"
    assert figure.data[0].mode == "markers+text"
    assert figure.data[1].mode == "lines+markers"
    assert figure.data[2].mode == "lines+markers"
    assert all(trace.mode == "markers+text" for trace in figure.data[3:])
    assert figure.data[-1].name == "Current best token embeddings"
    assert len(figure.frames[0].traces) == 5


def test_animated_projection_figure_labels_embedding_frame():
    basis = compute_embedding_projection_basis(torch.eye(3), model_name="toy")
    layers = project_hidden_state_layers(
        [torch.zeros(2, 3), torch.eye(3)[:2]],
        [0, 1],
        FakeTokenizer(),
        basis,
        layer_count=1,
        include_embedding_layer=True,
    )

    figure = animated_token_projection_figure(layers, basis, initial_layer_index=-1)

    assert _layer_label(-1) == "Embed"
    assert figure.frames[0].name == "Embed"
    assert figure.layout.sliders[0].steps[0].label == "Embed"


def test_animated_projection_figure_uses_fixed_range_for_all_reference_points():
    far_reference = ProjectedToken(index=0, token_id=99, text="far", x=100.0, y=0.0, z=0.0)
    layers = [
        LayerProjection(
            layer_index=-1,
            points=[
                ProjectedToken(index=0, token_id=1, text="a", x=0.0, y=0.0, z=0.0),
                ProjectedToken(index=1, token_id=2, text="b", x=1.0, y=0.0, z=0.0),
            ],
            top_prediction_points=[],
        ),
        LayerProjection(
            layer_index=0,
            points=[
                ProjectedToken(index=0, token_id=1, text="a", x=2.0, y=0.0, z=0.0),
                ProjectedToken(index=1, token_id=2, text="b", x=3.0, y=0.0, z=0.0),
            ],
            top_prediction_points=[far_reference],
        ),
    ]

    figure = animated_token_projection_figure(layers, None, dimensions=2, initial_layer_index=-1)

    assert figure.layout.xaxis.range[1] > 100.0
    assert figure.data[-4].name == "All top-k token embeddings"
    assert list(figure.frames[0].traces) == [0, 1, 2, 6, 7]


def test_animated_projection_labels_show_position_original_and_current_token():
    layers = [
        LayerProjection(
            layer_index=0,
            points=[
                ProjectedToken(
                    index=0,
                    token_id=1,
                    text="orig",
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    current_token_id=2,
                    current_text="best",
                )
            ],
            top_prediction_points=[],
        )
    ]

    figure = animated_token_projection_figure(layers, None, dimensions=2, initial_layer_index=0)

    assert list(figure.data[0].text) == ["0(orig->best)"]


def test_animated_projection_highlights_current_best_token_embeddings():
    layers = [
        LayerProjection(
            layer_index=0,
            points=[ProjectedToken(index=0, token_id=1, text="orig", x=0.0, y=0.0, z=0.0)],
            top_prediction_points=[],
            best_prediction_points=[
                ProjectedToken(index=0, token_id=2, text="best0", x=1.0, y=1.0, z=1.0)
            ],
        ),
        LayerProjection(
            layer_index=1,
            points=[ProjectedToken(index=0, token_id=1, text="orig", x=0.2, y=0.2, z=0.2)],
            top_prediction_points=[],
            best_prediction_points=[
                ProjectedToken(index=0, token_id=3, text="best1", x=2.0, y=2.0, z=2.0)
            ],
        ),
    ]

    figure = animated_token_projection_figure(layers, None, dimensions=2, initial_layer_index=0)

    assert figure.data[-2].name == "Current best token embeddings"
    assert figure.data[-1].name == "Current best token embeddings"
    assert figure.data[-2].mode == "markers+text"
    assert figure.data[-1].mode == "markers+text"
    assert list(figure.data[-2].text) == ["0 best: <b>best0</b> (2)"]
    assert list(figure.data[-1].text) == ["0 best: <b>best1</b> (3)"]
    assert list(figure.data[-2].hovertext) == [
        "0 best: <b>best0</b> (2)<br>"
        "token position: 0<br>"
        "best token id: 2<br>"
        "best token text: best0<br>"
        "x: 1.0000<br>"
        "y: 1.0000<br>"
        "z: 1.0000"
    ]
    assert figure.data[-2].opacity == 1.0
    assert figure.data[-1].opacity == 0.0
    assert figure.data[-2].marker.size == 8
    assert list(figure.frames[1].data[-2].to_plotly_json().keys()) == ["opacity", "type"]
    assert figure.frames[1].data[-2].opacity == 0.0
    assert figure.frames[1].data[-1].opacity == 1.0
    assert list(figure.frames[1].traces) == [0, 1, 5, 6]


def test_animated_projection_reference_points_deduplicate_final_prediction_tokens():
    duplicate = ProjectedToken(index=0, token_id=9, text="same", x=1.0, y=0.0, z=0.0)
    layers = [
        LayerProjection(
            layer_index=0,
            points=[ProjectedToken(index=0, token_id=1, text="a", x=0.0, y=0.0, z=0.0)],
            top_prediction_points=[duplicate],
        )
    ]

    figure = animated_token_projection_figure(
        layers,
        None,
        dimensions=2,
        final_prediction_points=[duplicate],
    )

    assert len(figure.data[-3].x) == 0
    assert len(figure.data[-2].x) == 1


class FakeReducer:
    last_matrix_shape = None
    last_matrix = None
    last_metric = None

    def __init__(self, n_components, metric=None, **_kwargs):
        self.n_components = n_components
        type(self).last_metric = metric

    def fit_transform(self, matrix):
        type(self).last_matrix_shape = matrix.shape
        type(self).last_matrix = np.asarray(matrix)
        coords = np.zeros((matrix.shape[0], self.n_components), dtype=np.float32)
        coords[:, 0] = np.arange(matrix.shape[0], dtype=np.float32)
        if self.n_components > 1:
            coords[:, 1] = -coords[:, 0]
        return coords


def test_dot_product_distance_matrix_is_nonnegative_with_zero_diagonal():
    distances = dot_product_distance_matrix(
        np.array(
            [
                [2.0, 0.0],
                [1.0, 0.0],
                [-1.0, 0.0],
            ],
            dtype=np.float32,
        )
    )

    assert distances.shape == (3, 3)
    assert np.all(distances >= 0.0)
    assert np.allclose(np.diag(distances), 0.0)
    assert distances[0, 1] < distances[0, 2]


def test_exp_negative_dot_product_distance_matrix_uses_exponential_similarity():
    distances = exp_negative_dot_product_distance_matrix(
        np.array(
            [
                [2.0, 0.0],
                [1.0, 0.0],
                [-1.0, 0.0],
            ],
            dtype=np.float32,
        )
    )

    assert distances.shape == (3, 3)
    assert np.all(distances >= 0.0)
    assert np.allclose(np.diag(distances), 0.0)
    assert distances[0, 1] < distances[0, 2]


def test_local_umap_projection_groups_generation_points():
    FakeReducer.last_matrix_shape = None
    FakeReducer.last_matrix = None
    FakeReducer.last_metric = None
    hidden_states = [
        torch.zeros(2, 4),
        torch.eye(4)[:2],
        torch.ones(2, 4),
    ]
    lm_head = FakeLMHead(
        torch.tensor(
            [
                [0.0, 4.0, 3.0, 2.0],
                [0.0, 1.0, 5.0, 4.0],
            ]
        )
    )

    projection = compute_local_umap_projection(
        hidden_states,
        [0, 1],
        FakeTokenizer(),
        torch.eye(4),
        lm_head,
        layer_count=2,
        top_k=2,
        dimensions=2,
        reducer_factory=FakeReducer,
    )

    assert projection.initial_points == []
    assert len(projection.final_prediction_points) == 1
    assert [layer.layer_index for layer in projection.layers] == [-1, 0, 1]
    assert [point.token_id for point in projection.layers[0].points] == [0, 1]
    assert [point.token_id for point in projection.layers[0].best_prediction_points] == [1, 2]
    assert [point.token_id for point in projection.layers[1].top_prediction_points] == [1, 2, 3]
    assert [point.token_id for point in projection.layers[1].best_prediction_points] == [1, 2]
    assert [point.index for point in projection.layers[1].best_prediction_points] == [0, 1]
    assert FakeReducer.last_matrix_shape == (9, 4)

    layer_0_token_2 = next(
        point for point in projection.layers[1].top_prediction_points if point.token_id == 2
    )
    layer_1_token_2 = next(
        point for point in projection.layers[2].top_prediction_points if point.token_id == 2
    )
    final_token_2 = projection.final_prediction_points[0]
    assert (layer_0_token_2.x, layer_0_token_2.y) == (layer_1_token_2.x, layer_1_token_2.y)
    assert (layer_0_token_2.x, layer_0_token_2.y) == (final_token_2.x, final_token_2.y)
    assert FakeReducer.last_metric == "cosine"


def test_local_umap_can_fit_with_dot_product_distances():
    FakeReducer.last_matrix_shape = None
    FakeReducer.last_matrix = None
    FakeReducer.last_metric = None
    hidden_states = [
        torch.zeros(2, 4),
        torch.eye(4)[:2],
        torch.ones(2, 4),
    ]
    lm_head = FakeLMHead(
        torch.tensor(
            [
                [0.0, 4.0, 3.0, 2.0],
                [0.0, 1.0, 5.0, 4.0],
            ]
        )
    )

    projection = compute_local_umap_projection(
        hidden_states,
        [0, 1],
        FakeTokenizer(),
        torch.eye(4),
        lm_head,
        layer_count=2,
        top_k=2,
        dimensions=2,
        metric="dot_product",
        reducer_factory=FakeReducer,
    )

    assert [layer.layer_index for layer in projection.layers] == [-1, 0, 1]
    assert FakeReducer.last_metric == "precomputed"
    assert FakeReducer.last_matrix_shape == (9, 9)
    assert np.all(FakeReducer.last_matrix >= 0.0)
    assert np.allclose(np.diag(FakeReducer.last_matrix), 0.0)


def test_local_umap_can_fit_with_exp_dot_product_distances():
    FakeReducer.last_matrix_shape = None
    FakeReducer.last_matrix = None
    FakeReducer.last_metric = None
    hidden_states = [
        torch.zeros(2, 4),
        torch.eye(4)[:2],
        torch.ones(2, 4),
    ]
    lm_head = FakeLMHead(
        torch.tensor(
            [
                [0.0, 4.0, 3.0, 2.0],
                [0.0, 1.0, 5.0, 4.0],
            ]
        )
    )

    projection = compute_local_umap_projection(
        hidden_states,
        [0, 1],
        FakeTokenizer(),
        torch.eye(4),
        lm_head,
        layer_count=2,
        top_k=2,
        dimensions=2,
        metric="exp_dot_product",
        reducer_factory=FakeReducer,
    )

    assert [layer.layer_index for layer in projection.layers] == [-1, 0, 1]
    assert FakeReducer.last_metric == "precomputed"
    assert FakeReducer.last_matrix_shape == (9, 9)
    assert np.all(FakeReducer.last_matrix >= 0.0)
    assert np.allclose(np.diag(FakeReducer.last_matrix), 0.0)


def test_local_tsne_can_fit_with_dot_product_distances():
    FakeReducer.last_matrix_shape = None
    FakeReducer.last_matrix = None
    FakeReducer.last_metric = None
    hidden_states = [
        torch.zeros(2, 4),
        torch.eye(4)[:2],
        torch.ones(2, 4),
    ]
    lm_head = FakeLMHead(
        torch.tensor(
            [
                [0.0, 4.0, 3.0, 2.0],
                [0.0, 1.0, 5.0, 4.0],
            ]
        )
    )

    projection = compute_local_umap_projection(
        hidden_states,
        [0, 1],
        FakeTokenizer(),
        torch.eye(4),
        lm_head,
        layer_count=2,
        top_k=2,
        dimensions=2,
        metric="dot_product_tsne",
        reducer_factory=FakeReducer,
    )

    assert [layer.layer_index for layer in projection.layers] == [-1, 0, 1]
    assert FakeReducer.last_metric == "precomputed"
    assert FakeReducer.last_matrix_shape == (9, 9)
    assert np.all(FakeReducer.last_matrix >= 0.0)
    assert np.allclose(np.diag(FakeReducer.last_matrix), 0.0)


def test_local_tsne_can_fit_with_exp_dot_product_distances():
    FakeReducer.last_matrix_shape = None
    FakeReducer.last_matrix = None
    FakeReducer.last_metric = None
    hidden_states = [
        torch.zeros(2, 4),
        torch.eye(4)[:2],
        torch.ones(2, 4),
    ]
    lm_head = FakeLMHead(
        torch.tensor(
            [
                [0.0, 4.0, 3.0, 2.0],
                [0.0, 1.0, 5.0, 4.0],
            ]
        )
    )

    projection = compute_local_umap_projection(
        hidden_states,
        [0, 1],
        FakeTokenizer(),
        torch.eye(4),
        lm_head,
        layer_count=2,
        top_k=2,
        dimensions=2,
        metric="exp_dot_product_tsne",
        reducer_factory=FakeReducer,
    )

    assert [layer.layer_index for layer in projection.layers] == [-1, 0, 1]
    assert FakeReducer.last_metric == "precomputed"
    assert FakeReducer.last_matrix_shape == (9, 9)
    assert np.all(FakeReducer.last_matrix >= 0.0)
    assert np.allclose(np.diag(FakeReducer.last_matrix), 0.0)


def test_final_prediction_token_ids_uses_last_position():
    lm_head = FakeLMHead(
        torch.tensor(
            [
                [0.0, 9.0, 1.0],
                [0.0, 1.0, 8.0],
            ]
        )
    )

    token_ids = final_prediction_token_ids(
        torch.zeros(2, 3),
        lm_head,
        top_k=1,
        apply_final_norm=False,
    )

    assert token_ids == [2]


def test_animated_projection_figure_can_render_2d():
    basis = compute_embedding_projection_basis(torch.eye(3), model_name="toy")
    layers = project_hidden_state_layers(
        [torch.zeros(2, 3), torch.eye(3)[:2]],
        [0, 1],
        FakeTokenizer(),
        basis,
        layer_count=1,
    )

    figure = animated_token_projection_figure(layers, basis, dimensions=2)

    assert figure.data[0].type == "scatter"
    assert figure.layout.xaxis.title.text.startswith("PC1")


def test_projection_axis_ranges_are_tight():
    assert _axis_range([0.0, 10.0]) == [-0.2, 10.2]
    x_range, y_range, z_range = _cube_ranges([0.0, 10.0], [4.0, 5.0], [-1.0, 1.0])

    assert np.allclose(x_range, [-0.2, 10.2])
    assert np.allclose(y_range, [-0.7, 9.7])
    assert np.allclose(z_range, [-5.2, 5.2])


def test_projection_ranges_can_be_computed_per_layer():
    points = [
        ProjectedToken(index=0, token_id=1, text="a", x=0.0, y=4.0, z=-1.0),
        ProjectedToken(index=1, token_id=2, text="b", x=10.0, y=5.0, z=1.0),
    ]

    ranges_2d = _ranges_for_points(points, dimensions=2)
    ranges_3d = _ranges_for_points(points, dimensions=3)

    assert np.allclose(ranges_2d["x_range"], [-0.2, 10.2])
    assert ranges_2d["z_range"] is None
    assert np.allclose(ranges_3d["z_range"], [-5.2, 5.2])


def test_projection_ranges_can_ignore_first_token_outlier():
    points = [
        ProjectedToken(index=0, token_id=1, text="a", x=1000.0, y=1000.0, z=1000.0),
        ProjectedToken(index=1, token_id=2, text="b", x=0.0, y=0.0, z=0.0),
        ProjectedToken(index=2, token_id=3, text="c", x=10.0, y=5.0, z=1.0),
    ]

    ranges = _ranges_for_points(points, dimensions=2, exclude_first_token=True)

    assert np.allclose(ranges["x_range"], [-0.2, 10.2])
    assert np.allclose(ranges["y_range"], [-0.1, 5.1])


def test_zero_variance_embedding_does_not_crash():
    basis = compute_embedding_projection_basis(torch.ones(3, 2), model_name="flat", dimensions=3)
    points = project_hidden_states(torch.ones(1, 2), [7], FakeTokenizer(), basis)

    assert basis.components.shape == (2, 3)
    assert np.allclose(basis.explained_variance_ratio, np.zeros(3))
    assert len(points) == 1


def test_precompute_finds_embedding_tensor_file(tmp_path):
    save_file({"other.weight": torch.ones(1)}, tmp_path / "a.safetensors")
    target_path = tmp_path / "b.safetensors"
    save_file({"model.embed_tokens.weight": torch.eye(2)}, target_path)

    assert load_precompute_script().find_tensor_file(tmp_path, "model.embed_tokens.weight") == target_path


def test_precompute_auto_detects_common_embedding_tensor(tmp_path):
    target_path = tmp_path / "model.safetensors"
    save_file({"transformer.wte.weight": torch.eye(2)}, target_path)

    tensor_file, tensor_name = load_precompute_script().find_embedding_tensor(
        tmp_path,
        "model.embed_tokens.weight",
    )

    assert tensor_file == target_path
    assert tensor_name == "transformer.wte.weight"
