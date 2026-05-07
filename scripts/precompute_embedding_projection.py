from __future__ import annotations

import argparse
import sys
from pathlib import Path

from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_visualizer.analysis.files import list_safetensors_files, load_tensor
from model_visualizer.ui_components.embedding_projection.projection import (
    DEFAULT_PROJECTION_DIMENSIONS,
    DEFAULT_EMBEDDING_TENSOR_NAME,
    DEFAULT_PROJECTION_DIR,
    PROJECTION_DIMENSION_OPTIONS,
    compute_embedding_projection_basis,
    projection_output_path,
    save_projection_basis,
)


EMBEDDING_TENSOR_CANDIDATES = (
    DEFAULT_EMBEDDING_TENSOR_NAME,
    "transformer.wte.weight",
    "gpt_neox.embed_in.weight",
    "model.tok_embeddings.weight",
    "tok_embeddings.weight",
    "bert.embeddings.word_embeddings.weight",
    "roberta.embeddings.word_embeddings.weight",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute a PCA projection basis from a model vocabulary embedding."
    )
    parser.add_argument("--model-dir", required=True, help="Local Hugging Face model directory.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_PROJECTION_DIR),
        help="Directory for the saved .npz projection file.",
    )
    parser.add_argument(
        "--embedding-tensor-name",
        default=DEFAULT_EMBEDDING_TENSOR_NAME,
        help="Embedding tensor key inside the safetensors file.",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        choices=PROJECTION_DIMENSION_OPTIONS,
        default=DEFAULT_PROJECTION_DIMENSIONS,
        help="Number of PCA dimensions to save.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing projection file.")
    return parser.parse_args()


def _matching_embedding_keys(keys: list[str]) -> list[str]:
    exact_candidates = [key for key in EMBEDDING_TENSOR_CANDIDATES if key in keys]
    fuzzy_candidates = [
        key
        for key in keys
        if key.endswith(".embed_tokens.weight")
        or key.endswith(".word_embeddings.weight")
        or key.endswith(".wte.weight")
        or key.endswith(".embed_in.weight")
        or key.endswith(".tok_embeddings.weight")
    ]
    return sorted(set(exact_candidates + fuzzy_candidates))


def _safetensors_files(model_dir: str | Path) -> list[Path]:
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_path.resolve()}")
    files = list_safetensors_files(model_path)
    if not files:
        raise FileNotFoundError(f"No .safetensors files were found in {model_path}.")
    return files


def find_tensor_file(model_dir: str | Path, tensor_name: str) -> Path:
    file_path, _resolved_tensor_name = find_embedding_tensor(model_dir, tensor_name)
    return file_path


def find_embedding_tensor(model_dir: str | Path, tensor_name: str) -> tuple[Path, str]:
    available_embedding_keys: list[str] = []
    for file_path in _safetensors_files(model_dir):
        with safe_open(file_path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            if tensor_name in keys:
                return file_path, tensor_name
            for key in _matching_embedding_keys(keys):
                available_embedding_keys.append(key)

    unique_keys = sorted(set(available_embedding_keys))
    if len(unique_keys) == 1:
        resolved_name = unique_keys[0]
        for file_path in _safetensors_files(model_dir):
            with safe_open(file_path, framework="pt", device="cpu") as handle:
                if resolved_name in handle.keys():
                    return file_path, resolved_name

    hint = ""
    if unique_keys:
        hint = " Candidate embedding tensors: " + ", ".join(unique_keys)
    raise FileNotFoundError(f"Tensor {tensor_name!r} was not found in {model_dir}.{hint}")


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_path = projection_output_path(model_dir, output_dir=args.output_dir)
    if output_path.exists() and not args.overwrite:
        print(f"Projection already exists: {output_path}")
        print("Use --overwrite to recompute it.")
        return 0

    tensor_file, tensor_name = find_embedding_tensor(model_dir, args.embedding_tensor_name)
    embedding = load_tensor(tensor_file, tensor_name)
    basis = compute_embedding_projection_basis(
        embedding,
        model_name=model_dir.name,
        embedding_tensor_name=tensor_name,
        dimensions=args.dimensions,
    )
    save_projection_basis(basis, output_path)

    print(f"Saved projection: {output_path}")
    print(f"source tensor: {tensor_name}")
    print(f"source file: {tensor_file}")
    print(f"vocab_size: {basis.vocab_size}")
    print(f"hidden_size: {basis.hidden_size}")
    print(
        "explained_variance_ratio: " + ", ".join(
            f"{float(value):.6f}" for value in basis.explained_variance_ratio
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
