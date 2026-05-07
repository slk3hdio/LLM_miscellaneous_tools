from __future__ import annotations

"""将当前演示状态打包为可下载静态文件."""

import csv
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from model_visualizer.ui_components.embedding_projection.figures import (
    animated_token_projection_figure,
)
from model_visualizer.ui_components.embedding_projection.local_umap import compute_local_umap_projection
from model_visualizer.ui_components.embedding_projection.projection import (
    load_projection_basis,
    model_embedding_weight,
    project_hidden_state_layers,
    project_token_embeddings,
    projection_output_path,
)
from model_visualizer.ui_components.export.types import ExportOptions
from model_visualizer.ui_components.inference.figures import attention_layer_heatmap_figure
from model_visualizer.ui_components.inference.trace import (
    decode_token_ids,
    hidden_state_for_layer,
    top_token_predictions,
)
from model_visualizer.ui_components.inference.types import InferenceTraceView
from model_visualizer.ui_components.structure.types import ModelStructure


def _csv_bytes(rows: Iterable[dict]) -> bytes:
    rows = list(rows)
    buffer = io.StringIO(newline="")
    if not rows:
        return b""
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _npy_bytes(array) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array))
    return buffer.getvalue()


def _figure_html_bytes(figure) -> bytes:
    return figure.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")


def _token_rows(trace_view: InferenceTraceView) -> list[dict]:
    tokens = decode_token_ids(trace_view.tokenizer, trace_view.trace_token_ids)
    return [
        {
            "position": token.index,
            "token_id": token.token_id,
            "text": token.text,
        }
        for token in tokens
    ]


def _top_prediction_rows(trace_view: InferenceTraceView) -> list[dict]:
    hidden_state, apply_final_norm = hidden_state_for_layer(
        trace_view.manager.hidden_states[trace_view.generation_step],
        trace_view.layer_index,
    )
    tokens = decode_token_ids(trace_view.tokenizer, trace_view.trace_token_ids)
    predictions = top_token_predictions(
        hidden_state,
        trace_view.lm_head,
        trace_view.tokenizer,
        top_n=trace_view.top_n,
        apply_final_norm=apply_final_norm,
    )
    rows: list[dict] = []
    for token, row in zip(tokens, predictions):
        for prediction in row:
            rows.append(
                {
                    "position": token.index,
                    "source_token_id": token.token_id,
                    "source_text": token.text,
                    "rank": prediction.rank,
                    "prediction_token_id": prediction.token_id,
                    "prediction_text": prediction.text,
                    "probability": prediction.probability,
                }
            )
    return rows


def _projection_figure(trace_view: InferenceTraceView, options: ExportOptions):
    layer_count = max(
        0,
        min(
            trace_view.num_layers,
            len(trace_view.manager.hidden_states[trace_view.generation_step]) - 1,
        ),
    )
    embedding_weight = model_embedding_weight(trace_view.model)
    if options.projection_mode in {"UMAP", "Dot-product UMAP"}:
        projection = compute_local_umap_projection(
            trace_view.manager.hidden_states[trace_view.generation_step],
            trace_view.trace_token_ids,
            trace_view.tokenizer,
            embedding_weight,
            trace_view.lm_head,
            layer_count=layer_count,
            top_k=options.projection_top_k,
            dimensions=options.projection_dimensions,
            metric="dot_product" if options.projection_mode == "Dot-product UMAP" else "cosine",
        )
        return animated_token_projection_figure(
            projection.layers,
            None,
            initial_layer_index=-1,
            dimensions=options.projection_dimensions,
            initial_points=projection.initial_points,
            final_prediction_points=projection.final_prediction_points,
        )

    path = projection_output_path(trace_view.model_dir)
    if not path.exists():
        raise FileNotFoundError(f"PCA projection file not found: {path}")
    basis = load_projection_basis(path)
    if basis.components.shape[1] < options.projection_dimensions:
        raise ValueError(
            f"PCA projection file has {basis.components.shape[1]} dimensions; "
            f"requested {options.projection_dimensions}."
        )
    layers = project_hidden_state_layers(
        trace_view.manager.hidden_states[trace_view.generation_step],
        trace_view.trace_token_ids,
        trace_view.tokenizer,
        basis,
        layer_count=layer_count,
        embedding_weight=embedding_weight,
        lm_head=trace_view.lm_head,
        top_k=options.projection_top_k,
        include_embedding_layer=True,
    )
    initial_points = project_token_embeddings(
        embedding_weight,
        trace_view.trace_token_ids,
        trace_view.tokenizer,
        basis,
    )
    return animated_token_projection_figure(
        layers,
        basis,
        initial_layer_index=-1,
        dimensions=options.projection_dimensions,
        initial_points=initial_points,
        final_prediction_points=[],
    )


def build_demo_export_bundle(
    *,
    structure: ModelStructure,
    trace_view: InferenceTraceView,
    options: ExportOptions,
) -> bytes:
    """构建当前演示状态的静态文件 zip 包."""

    tokens = decode_token_ids(trace_view.tokenizer, trace_view.trace_token_ids)
    attention_layers = trace_view.manager.attention_weights[trace_view.generation_step]
    attention = attention_layers[trace_view.layer_index]
    hidden_state, _apply_final_norm = hidden_state_for_layer(
        trace_view.manager.hidden_states[trace_view.generation_step],
        trace_view.layer_index,
    )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": structure.model_name,
        "model_dir": structure.model_dir,
        "generation_step": trace_view.generation_step,
        "layer_index": trace_view.layer_index,
        "num_layers": trace_view.num_layers,
        "top_n": trace_view.top_n,
        "sequence_length": len(trace_view.trace_token_ids),
        "attention_layers": len(attention_layers),
        "attention_heads": int(attention.shape[0]),
        "projection": {
            "included": options.include_projection,
            "mode": options.projection_mode,
            "dimensions": options.projection_dimensions,
            "top_k": options.projection_top_k,
        },
    }

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        archive.writestr(
            "README.md",
            "\n".join(
                [
                    "# Model Visualizer Demo Export",
                    "",
                    "This bundle contains static files for the current inference step.",
                    "",
                    "- `manifest.json`: export metadata",
                    "- `tokens.csv`: active trace input tokens",
                    "- `current_layer_top_predictions.csv`: top-N decoded candidates",
                    "- `arrays/current_layer_hidden_state.npy`: current layer hidden state",
                    "- `arrays/attention/layer_XX_attention.npy`: optional attention weights for each layer",
                    "- `attention/layer_XX_all_heads.html`: optional static attention heatmaps grouped by layer",
                    "- `projection/token_state_projection.html`: optional projection figure",
                    "",
                ]
            ),
        )
        archive.writestr("tokens.csv", _csv_bytes(_token_rows(trace_view)))
        archive.writestr(
            "current_layer_top_predictions.csv",
            _csv_bytes(_top_prediction_rows(trace_view)),
        )
        archive.writestr(
            "arrays/current_layer_hidden_state.npy",
            _npy_bytes(hidden_state.detach().cpu().float().numpy()),
        )

        if options.include_attention:
            for layer_index, layer_attention in enumerate(attention_layers):
                archive.writestr(
                    f"arrays/attention/layer_{layer_index:02d}_attention.npy",
                    _npy_bytes(layer_attention.detach().cpu().float().numpy()),
                )
                figure = attention_layer_heatmap_figure(
                    layer_attention.detach().cpu().float(),
                    tokens,
                    layer_index=layer_index,
                )
                archive.writestr(
                    f"attention/layer_{layer_index:02d}_all_heads.html",
                    _figure_html_bytes(figure),
                )

        if options.include_projection:
            figure = _projection_figure(trace_view, options)
            archive.writestr(
                "projection/token_state_projection.html",
                _figure_html_bytes(figure),
            )

    return buffer.getvalue()


def default_export_filename(trace_view: InferenceTraceView) -> str:
    model_name = Path(trace_view.model_dir).name or "model"
    return (
        f"{model_name}_step_{trace_view.generation_step}_"
        f"layer_{trace_view.layer_index}_demo_export.zip"
    )
