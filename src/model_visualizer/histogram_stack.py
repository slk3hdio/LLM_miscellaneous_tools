from __future__ import annotations

"""离线渲染工具 —— 将层直方图柱形堆叠图表输出为独立 HTML 文件."""

import logging
import re
from html import escape
from pathlib import Path

logger = logging.getLogger(__name__)

from model_visualizer.analysis.files import (
    inspect_safetensors,
    list_safetensors_files,
)
from model_visualizer.analysis.histograms import (
    available_layer_matrix_keys,
    compute_layer_histogram_stack,
)
from model_visualizer.figures import layer_histogram_stack_figure


DEFAULT_OUTPUT_DIR = Path("outputs/model_visualizer/histogram_stacks")


def rotation_post_script() -> str:
    """返回让 Plotly 3D 图缓慢自动旋转的 HTML 后置脚本."""

    return """
const graph = document.getElementById('{plot_id}');
if (graph && window.Plotly) {
  const initialEye = graph.layout.scene?.camera?.eye || {x: 1.8, y: 1.15, z: 1.8};
  const radius = Math.hypot(initialEye.x, initialEye.z) || 2.5;
  const y = initialEye.y || 1.15;
  let angle = Math.atan2(initialEye.z, initialEye.x);
  let userInteracting = false;

  graph.addEventListener('pointerdown', () => { userInteracting = true; });
  graph.addEventListener('pointerup', () => { userInteracting = false; });
  graph.addEventListener('pointerleave', () => { userInteracting = false; });

  window.setInterval(() => {
    if (userInteracting) {
      return;
    }
    angle += 0.006;
    Plotly.relayout(graph, {
      'scene.camera.eye': {
        x: radius * Math.cos(angle),
        y: y,
        z: radius * Math.sin(angle)
      }
    });
  }, 80);
}
"""


def safe_output_stem(model_dir: str | Path, matrix_key: str) -> str:
    """根据模型目录名和矩阵键生成安全的文件名主干.

    移除非法字符，仅保留字母、数字、下划线和短横线。
    """

    model_name = Path(model_dir).name
    raw = f"{model_name}_{matrix_key}"
    return re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_")


def default_output_path(model_dir: str | Path, matrix_key: str) -> Path:
    """返回直方图柱形堆叠 HTML 的默认输出路径."""

    return DEFAULT_OUTPUT_DIR / f"{safe_output_stem(model_dir, matrix_key)}.html"


def default_grid_output_path(model_dir: str | Path, matrix_keys: list[str]) -> Path:
    """返回多图柱形堆叠 HTML 的默认输出路径."""

    key_text = "_".join(matrix_keys[:3])
    if len(matrix_keys) > 3:
        key_text = f"{key_text}_and_{len(matrix_keys) - 3}_more"
    return DEFAULT_OUTPUT_DIR / f"{safe_output_stem(model_dir, key_text)}.html"


def _load_available_tensor_infos(model_dir: str | Path):
    """加载模型 safetensors 元数据并返回可用矩阵键."""

    files = list_safetensors_files(model_dir)
    if not files:
        raise FileNotFoundError(f"No safetensors files found under {model_dir}")

    infos = inspect_safetensors(files)
    available = available_layer_matrix_keys(infos)
    return infos, available


def _validate_matrix_key(matrix_key: str, available: list[str]) -> None:
    """确认矩阵键存在."""

    if matrix_key not in available:
        available_text = ", ".join(available) if available else "(none)"
        raise ValueError(f"Matrix key {matrix_key!r} is not available. Available: {available_text}")


def _render_grid_html(
    *,
    model_dir: str | Path,
    matrix_keys: list[str],
    figure_html_parts: list[str],
) -> str:
    """把多个 Plotly 图组织成一个响应式 HTML 页面."""

    cards = "\n".join(
        (
            '<section class="plot-card">'
            f"<h2>{escape(matrix_key)}</h2>"
            f"{figure_html}"
            "</section>"
        )
        for matrix_key, figure_html in zip(matrix_keys, figure_html_parts)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(Path(model_dir).name)} histogram stacks</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      background: #f8fafc;
      color: #0f172a;
    }}
    body {{
      margin: 0;
      padding: 28px;
      background: #f8fafc;
    }}
    h1 {{
      margin: 0 0 22px;
      font-size: 24px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .plot-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(680px, 1fr));
      gap: 22px;
      align-items: start;
    }}
    .plot-card {{
      min-width: 0;
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 14px;
      box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
    }}
    .plot-card h2 {{
      margin: 0 0 8px;
      font-size: 15px;
      font-weight: 650;
      color: #334155;
      letter-spacing: 0;
    }}
    @media (max-width: 760px) {{
      body {{
        padding: 12px;
      }}
      .plot-grid {{
        grid-template-columns: minmax(0, 1fr);
      }}
    }}
  </style>
</head>
<body>
  <h1>{escape(Path(model_dir).name)} histogram stacks</h1>
  <main class="plot-grid">
    {cards}
  </main>
</body>
</html>
"""


def render_layer_histogram_stack_html(
    model_dir: str | Path,
    matrix_key: str = "self_attn.q_proj.weight",
    *,
    bins: int = 80,
    max_values_per_layer: int = 100_000,
    density: bool = True,
    output: str | Path | None = None,
) -> Path:
    """将某一矩阵键的每层直方图柱形图渲染为独立的 HTML 文件.

    完整流程：
    1. 扫描模型目录下的 safetensors 文件
    2. 读取所有张量元数据
    3. 找到可用的层矩阵键
    4. 计算每层的共享分箱直方图
    5. 构建 3D 柱形堆叠图并通过 Plotly 的 write_html 写入文件

    参数：
        model_dir: 模型目录路径
        matrix_key: 目标矩阵键，如 "self_attn.q_proj.weight"
        bins: 直方图箱数，默认 80
        max_values_per_layer: 每层最多采样多少值，默认 100k
        density: 是否归一化为密度
        output: 自定义输出路径，默认自动生成
    """

    infos, available = _load_available_tensor_infos(model_dir)
    _validate_matrix_key(matrix_key, available)

    # 4. 计算直方图堆叠
    histograms = compute_layer_histogram_stack(
        infos,
        matrix_key,
        bins=bins,
        max_values_per_layer=max_values_per_layer,
        density=density,
    )
    if not histograms:
        raise ValueError(f"No histograms were generated for {matrix_key!r}")

    # 5. 构建图形并写入 HTML
    logger.info("Rendering histogram stack for %s/%s", model_dir, matrix_key)
    figure = layer_histogram_stack_figure(histograms, density=density)
    output_path = Path(output) if output is not None else default_output_path(model_dir, matrix_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        output_path,
        include_plotlyjs="cdn",
        full_html=True,
        post_script=rotation_post_script(),
    )
    logger.info("Histogram stack written to %s", output_path)
    return output_path


def render_layer_histogram_stack_grid_html(
    model_dir: str | Path,
    matrix_keys: list[str],
    *,
    bins: int = 80,
    max_values_per_layer: int = 100_000,
    density: bool = True,
    output: str | Path | None = None,
) -> Path:
    """将多个矩阵键的层直方图柱形图排列到同一个 HTML 文件中."""

    if not matrix_keys:
        raise ValueError("At least one matrix key is required.")

    infos, available = _load_available_tensor_infos(model_dir)
    figure_html_parts: list[str] = []
    for index, matrix_key in enumerate(matrix_keys):
        _validate_matrix_key(matrix_key, available)
        histograms = compute_layer_histogram_stack(
            infos,
            matrix_key,
            bins=bins,
            max_values_per_layer=max_values_per_layer,
            density=density,
        )
        if not histograms:
            raise ValueError(f"No histograms were generated for {matrix_key!r}")
        figure = layer_histogram_stack_figure(histograms, density=density)
        figure_html_parts.append(
            figure.to_html(
                include_plotlyjs="cdn" if index == 0 else False,
                full_html=False,
                post_script=rotation_post_script(),
            )
        )

    output_path = (
        Path(output)
        if output is not None
        else default_grid_output_path(model_dir, matrix_keys)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _render_grid_html(
            model_dir=model_dir,
            matrix_keys=matrix_keys,
            figure_html_parts=figure_html_parts,
        ),
        encoding="utf-8",
    )
    logger.info("Histogram stack grid written to %s", output_path)
    return output_path
