from __future__ import annotations

import logging

import pytest

from universal_eval.runner import _resolve_runtime_options


class DummyRuntimeProvider:
    def __init__(self, supports_conversation: bool = True, supports_tools: bool = True) -> None:
        self._supports_conversation = supports_conversation
        self._supports_tools = supports_tools

    def supports_conversation_format(self) -> bool:
        return self._supports_conversation

    def supports_tool_calling(self) -> bool:
        return self._supports_tools


def test_resolve_runtime_options_keeps_supported_standard_tools():
    config = {"conversation_style": "multi", "tool_format": "standard"}

    assert _resolve_runtime_options(config, DummyRuntimeProvider(), logging.getLogger(__name__)) == (
        "multi",
        "standard",
    )
    assert config["conversation_style"] == "multi"
    assert config["tool_format"] == "standard"


def test_resolve_runtime_options_downgrades_standard_tools_for_single_turn():
    config = {"conversation_style": "single", "tool_format": "standard"}

    assert _resolve_runtime_options(config, DummyRuntimeProvider(), logging.getLogger(__name__)) == (
        "single",
        "plain",
    )
    assert config["tool_format"] == "plain"


def test_resolve_runtime_options_downgrades_multi_when_provider_does_not_support_chat():
    config = {"conversation_style": "multi", "tool_format": "standard"}

    assert _resolve_runtime_options(
        config,
        DummyRuntimeProvider(supports_conversation=False),
        logging.getLogger(__name__),
    ) == ("single", "plain")
    assert config["conversation_style"] == "single"
    assert config["tool_format"] == "plain"


def test_resolve_runtime_options_rejects_unknown_values():
    with pytest.raises(ValueError, match="Unsupported conversation_style"):
        _resolve_runtime_options(
            {"conversation_style": "flat", "tool_format": "plain"},
            DummyRuntimeProvider(),
            logging.getLogger(__name__),
        )
