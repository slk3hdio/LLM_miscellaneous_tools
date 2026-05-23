# LLM Miscellaneous Tools

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white)](https://plotly.com)

Utilities for inspecting local language-model weights, visualizing model internals with Streamlit, and evaluating tool-use datasets such as API-Bank and ToolACE.

> *[中文版本](README.md)*

---

## Model Visualizer

An interactive **Streamlit** web application for exploring, inspecting, and debugging Hugging Face transformer models. It provides four integrated visualization components covering everything from static structure to dynamic inference tracing.

### Architecture

| Module | Role |
|--------|------|
| **`model_runtime`** | Inference engine — loads models and captures per-layer activations via PyTorch forward hooks during autoregressive generation. |
| **`model_visualizer`** | Streamlit frontend — consumes trace data and renders 3D structures, attention heatmaps, prediction grids, and embedding trajectory animations. |

### Four Components

1. **3D Model Structure** — Scans `config.json` and `.safetensors` files, classifies weight matrices by function (QKV / attention output / MLP gate & up / MLP down), and renders colored 3D cuboids with data-flow Bezier curves.

2. **Step-by-Step Inference** — Loads a model and lets you single-step through generation, inspecting per-layer Top-K predictions and multi-head attention heatmaps at each layer.

3. **Embedding Trajectory Projection** — Projects hidden states into 2D/3D using PCA or UMAP with configurable distance metrics (cosine, dot product, exp-dot). Animated frames show token-level representation drift across layers.

4. **Demo Export** — Bundles the current inference state into a downloadable ZIP (CSV, NPY, standalone HTML figures).

### Quick Start

```powershell
# Place models under models/ (directories with config.json + .safetensors)
python scripts/run_model_visualizer.py
```

For PCA-based embedding projection, precompute the basis first:

```powershell
python scripts/precompute_embedding_projection.py
```

### Screenshots

#### 1. Model Structure — 3D Weight Matrix Layout

Qwen2.5-1.5B: each layer is split into Q/K/V (blue), attention output (green), MLP gate/up (orange/red), and MLP down (purple) stages.

![Model Structure](assets/qwen_1.5b_structure.png)

#### 2. Cross-model Structure Comparison

Qwen2.5-1.5B vs Qwen3-0.6B vs Qwen2.5-7B vs Llama-3.1-8B.

![Structure Comparison](assets/structure_comp_all.png)

#### 3. Parameter Distribution Histograms

Per-layer normalized histograms for all six core matrix types.

| Qwen2.5-1.5B | Llama-3.1-8B |
|--------------|--------------|
| ![Hist Qwen](assets/histogram_qwen1.5b_all.png) | ![Hist Llama](assets/histogram_llama8b_all.png) |

Qwen2.5-7B `v_proj.weight` — higher peaks near zero in shallow layers, slight widening in deeper layers.

![Hist v_proj](assets/histogram_qwen7b_attn_v.png)

#### 4. Step-by-Step Inference — Top-K Predictions

Layer 15 of Qwen2.5-1.5B processing *"Once upon a time"* — the final position already ranks a comma as top-1, demonstrating intermediate-layer early convergence.

![Top-K](assets/Top-K_1-15.png)

#### 5. Attention Heatmaps (Layer 15, 12 heads)

Most heads exhibit first-token anchoring; a few show diagonal self-attention or mixed-context patterns.

![Attention](assets/atten_headers_layer15.png)

#### 6. Embedding Trajectory — PCA vs UMAP

PCA compresses tokens into a wide horizontal band — trajectories are recognizable but lack clear separability.

![PCA](assets/PCA_step0.png)

Dot-product UMAP at Layer 15 — four distinct trajectories with continuous paths and prediction-consistent endpoints.

![UMAP dot](assets/UMAP+dot_step15.png)

#### 7. Distance Metric Comparison (Layer 15)

| Default UMAP (L2) | Exp-Dot UMAP |
|-------------------|--------------|
| ![UMAP L2](assets/UMAP+L2_step15.png) | ![UMAP exp-dot](assets/UMAP+Exp-dot_step15.png) |

Dot-product UMAP achieves the best balance of trajectory separability, endpoint consistency, and path continuity.

### Key Findings

1. **Parameter distributions** differ primarily by matrix family (Q/K/V sharper, MLP smoother), with a consistent "shallow-concentrated, deep-dispersed" trend across all model series.
2. **Intermediate-layer decoding** already reveals decision direction — in *"Once upon a time"*, Layer 15's final position strongly favors punctuation completion.
3. **Dot-product UMAP** outperforms PCA, L2, and cosine-based projections in capturing token-level trajectory separability and prediction-consistent endpoints.

---

## Evaluation

Copy the example config and provide local paths or API credentials:

```powershell
Copy-Item configs/example.yaml configs/eval.yaml
python scripts/test_model.py --config configs/eval.yaml
```

`configs/eval.yaml`, `data/`, `models/`, `outputs/`, and log files are ignored by Git.

---

## Tests

```powershell
pytest
```

Tests marked `integration` require API credentials or local model weights and are automatically skipped when unavailable.
