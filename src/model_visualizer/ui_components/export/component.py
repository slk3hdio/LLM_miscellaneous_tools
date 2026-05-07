from __future__ import annotations

"""Streamlit 导出组件：导出演示所需的静态文件包."""

import logging

import streamlit as st

from model_visualizer.ui_components.embedding_projection.component import PROJECTION_MODES
from model_visualizer.ui_components.embedding_projection.projection import (
    DEFAULT_PROJECTION_DIMENSIONS,
    PROJECTION_DIMENSION_OPTIONS,
)
from model_visualizer.ui_components.export.bundle import (
    build_demo_export_bundle,
    default_export_filename,
)
from model_visualizer.ui_components.export.types import ExportOptions
from model_visualizer.ui_components.inference.types import InferenceTraceView
from model_visualizer.ui_components.structure.types import ModelStructure


logger = logging.getLogger(__name__)


class DemoExportComponent:
    """导出当前演示状态为 zip 文件."""

    def __init__(
        self,
        *,
        structure: ModelStructure,
        trace_view: InferenceTraceView | None,
        state_prefix: str = "demo_export",
    ):
        self.structure = structure
        self.trace_view = trace_view
        self.state_prefix = state_prefix

    def key(self, name: str) -> str:
        return f"{self.state_prefix}_{name}"

    def render(self) -> None:
        st.subheader("Demo Export")
        if self.trace_view is None:
            st.info("Run an inference step before exporting static demo files.")
            return

        self._clear_stale_bundle()
        options = self._render_options()
        if st.button("Prepare export bundle", key=self.key("prepare")):
            try:
                with st.spinner("Preparing static files..."):
                    data = build_demo_export_bundle(
                        structure=self.structure,
                        trace_view=self.trace_view,
                        options=options,
                    )
                st.session_state[self.key("bundle")] = data
                st.session_state[self.key("filename")] = default_export_filename(self.trace_view)
                st.success("Export bundle is ready.")
            except Exception as exc:
                logger.error("Demo export failed: %s", exc, exc_info=True)
                st.error(f"Demo export failed: {exc}")

        data = st.session_state.get(self.key("bundle"))
        filename = st.session_state.get(self.key("filename"), default_export_filename(self.trace_view))
        if data:
            st.download_button(
                "Download static demo bundle",
                data=data,
                file_name=filename,
                mime="application/zip",
                key=self.key("download"),
            )

    def _render_options(self) -> ExportOptions:
        include_attention = st.checkbox(
            "Include attention heatmap HTML",
            value=True,
            key=self.key("include_attention"),
        )
        include_projection = st.checkbox(
            "Include token projection HTML",
            value=True,
            key=self.key("include_projection"),
        )
        projection_mode = st.selectbox(
            "Export projection mode",
            options=list(PROJECTION_MODES),
            index=list(PROJECTION_MODES).index("Dot-product UMAP"),
            key=self.key("projection_mode"),
            disabled=not include_projection,
        )
        dimensions = int(
            st.radio(
                "Export projection dimensions",
                options=list(PROJECTION_DIMENSION_OPTIONS),
                index=list(PROJECTION_DIMENSION_OPTIONS).index(DEFAULT_PROJECTION_DIMENSIONS),
                horizontal=True,
                format_func=lambda value: f"{value}D",
                key=self.key("projection_dimensions"),
                disabled=not include_projection,
            )
        )
        top_k = int(
            st.number_input(
                "Export projection Top K",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                key=self.key("projection_top_k"),
                disabled=not include_projection,
            )
        )
        return ExportOptions(
            include_attention=include_attention,
            include_projection=include_projection,
            projection_mode=projection_mode,
            projection_dimensions=dimensions,
            projection_top_k=top_k,
        )

    def _clear_stale_bundle(self) -> None:
        current_key = (
            self.trace_view.model_dir,
            self.trace_view.generation_step,
            self.trace_view.layer_index,
            tuple(self.trace_view.trace_token_ids),
        )
        previous_key = st.session_state.get(self.key("trace_key"))
        if previous_key != current_key:
            st.session_state.pop(self.key("bundle"), None)
            st.session_state.pop(self.key("filename"), None)
            st.session_state[self.key("trace_key")] = current_key
