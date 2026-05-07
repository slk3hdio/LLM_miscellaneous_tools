from __future__ import annotations

import ast
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import logging

from .sample import EvalSample


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _extract_call_block(text: str) -> Optional[str]:
    start = text.find("[")
    if start < 0:
        return None

    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _split_top_level_items(text: str) -> List[str]:
    items: List[str] = []
    current: List[str] = []
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    quote_char: Optional[str] = None
    escape = False

    for char in text:
        if quote_char is not None:
            current.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote_char:
                quote_char = None
            continue

        if char in {"'", '"'}:
            quote_char = char
            current.append(char)
        elif char == "(":
            paren_depth += 1
            current.append(char)
        elif char == ")":
            paren_depth -= 1
            current.append(char)
        elif char == "[":
            bracket_depth += 1
            current.append(char)
        elif char == "]":
            bracket_depth -= 1
            current.append(char)
        elif char == "{":
            brace_depth += 1
            current.append(char)
        elif char == "}":
            brace_depth -= 1
            current.append(char)
        elif char == "," and paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
            item = "".join(current).strip()
            if item:
                items.append(item)
            current = []
        else:
            current.append(char)

    tail = "".join(current).strip()
    if tail:
        items.append(tail)
    return items


def _normalize_argument_value(value: str) -> str:
    stripped = value.strip()
    # Try JSON first (handles true/false/null)
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(stripped)
        except Exception:
            return _normalize_text(stripped)

    if isinstance(parsed, bool):
        return str(parsed).lower()
    if isinstance(parsed, float) and parsed == int(parsed):
        return str(int(parsed))
    if isinstance(parsed, (dict, list, tuple)):
        return json.dumps(parsed, ensure_ascii=False, sort_keys=True)
    return str(parsed)


def _parse_call_string(text: str) -> List[Dict[str, Any]]:
    block = _extract_call_block(text)
    if not block:
        return []

    inner = block[1:-1].strip()
    if not inner:
        return []

    calls: List[Dict[str, Any]] = []
    for raw_call in _split_top_level_items(inner):
        match = re.match(r"^\s*(?P<name>[^()]+?)\s*\((?P<args>.*)\)\s*$", raw_call, re.DOTALL)
        if not match:
            continue

        method_name = _normalize_text(match.group("name"))
        arguments: Dict[str, str] = {}
        args_text = match.group("args").strip()
        if args_text:
            for raw_argument in _split_top_level_items(args_text):
                if "=" not in raw_argument:
                    continue
                key, value = raw_argument.split("=", 1)
                arguments[_normalize_text(key)] = _normalize_argument_value(value)

        calls.append({"name": method_name, "arguments": arguments})

    return calls


class DatasetAdapter(ABC):
    name: str = "base"

    @classmethod
    def can_load(cls, dataset_path: Path) -> bool:
        return False

    @abstractmethod
    def load_samples(
        self,
        dataset_config: dict[str, Any],
        limit: int | None = None,
        conversation_style: Literal['single', 'multi'] = 'single',
        with_raw_data: bool = False,
        random_samples: bool = False,
        strip_tool_descriptions: bool = False,
    ) -> List[EvalSample]:
        raise NotImplementedError
