from __future__ import annotations

"""导出演示静态文件的内部配置类型."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExportOptions:
    """导出包选项."""

    include_attention: bool = True
    include_projection: bool = True
    projection_mode: str = "Dot-product UMAP"
    projection_dimensions: int = 2
    projection_top_k: int = 5

