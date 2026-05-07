# Paper

Utilities for inspecting local language-model weights, visualizing model internals with Streamlit, and evaluating tool-use datasets such as API-Bank and ToolACE.

## Setup

```powershell
uv sync
```

If you do not use `uv`, install the project dependencies from `pyproject.toml` with your preferred Python 3.11+ environment manager.

## Model Visualizer

```powershell
python scripts/run_model_visualizer.py
```

Place local model snapshots under `models/`. The app expects model directories that contain `config.json` and `.safetensors` files.

## Evaluation

Copy the example config and fill in local paths or API credentials through environment variables:

```powershell
Copy-Item configs/example.yaml configs/eval.yaml
python scripts/test_model.py --config configs/eval.yaml
```

`configs/eval.yaml`, `data/`, `models/`, `outputs/`, and logs are intentionally ignored by Git.

## Tests

```powershell
pytest
```

Tests marked `integration` require external API credentials or local model weights and skip automatically when those resources are not configured.
