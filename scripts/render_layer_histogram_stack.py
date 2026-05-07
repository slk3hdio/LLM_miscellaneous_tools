from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_visualizer.histogram_stack import (
    render_layer_histogram_stack_grid_html,
    render_layer_histogram_stack_html,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a 3D layer histogram stack to HTML.")
    parser.add_argument("--model-dir", default="models/llama_3_1_8b")
    parser.add_argument("--matrix-key", nargs="+", default=["mlp.up_proj.weight"])
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--max-values-per-layer", type=int, default=100_0000)
    parser.add_argument("--count", action="store_true", help="Render raw counts instead of density.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--open", action="store_true", help="Open the rendered HTML file after writing it.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.matrix_key) == 1:
        output_path = render_layer_histogram_stack_html(
            args.model_dir,
            args.matrix_key[0],
            bins=args.bins,
            max_values_per_layer=args.max_values_per_layer,
            density=not args.count,
            output=args.output,
        )
    else:
        output_path = render_layer_histogram_stack_grid_html(
            args.model_dir,
            args.matrix_key,
            bins=args.bins,
            max_values_per_layer=args.max_values_per_layer,
            density=not args.count,
            output=args.output,
        )
    print(output_path)
    if args.open:
        subprocess.Popen(["cmd", "/c", "start", "", str(output_path)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
