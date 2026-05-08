from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional
from ..evaluator.parser_tools import parse_call_string
from typing import TypedDict
from logging import getLogger
from typing_extensions import NotRequired, Required
import re

logger = getLogger(__name__)

@dataclass
class EvalSample:
    """Standardised evaluation sample.

    All adapters MUST produce data in the formats described below so that
    conversion helpers work on a single canonical representation.

    Standard formats
    ----------------
    **api_set entry**::

        {
            "name": "func_name",
            "description": "...",
            "parameters": {
                "type": "object",
                "properties": {"p": {"type": "string", "description": "..."}},
                "required": ["p"]
            }
        }

    **assistant tool-call** (context ``content``)::

        [func_name(key1="val1", key2="val2")]

    **tool result** (context ``content``)::

        [{"name": "func_name", "result": {...}}]
    """

    class Context(TypedDict):
        role: Required[Literal["user", "assistant", "system", "tool"]]
        content: Required[Optional[str]]
        tool_calls: NotRequired[List[Any]]
        tool_call_id: NotRequired[str]
        reasoning_content: NotRequired[Optional[str]]

    sample_id: str
    context: List[Context]
    target: str
    api_set: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # OpenAI ChatCompletion helpers
    # ------------------------------------------------------------------

    def to_openai_messages(self, format_tools: bool, save_to_meta = True) -> list["EvalSample.Context"]:
        """Convert context to OpenAI ChatCompletion format."""
        converted: list['EvalSample.Context'] = []
        pending_call_ids: list[str] = []

        for msg in self.context:
            role = msg["role"]
            content = msg["content"] or ""

            if role in ("system", "user"):
                converted.append({"role": role, "content": content})
                pending_call_ids = []

            elif role == "assistant":
                if _is_tool_call(content) and format_tools:
                    tool_calls = _build_tool_calls(content, offset=len(converted))
                    converted.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls,
                        "reasoning_content": None,
                    })
                    pending_call_ids = [tc["id"] for tc in tool_calls]
                else:
                    converted.append({
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": None,
                    })
                    pending_call_ids = []

            elif role == "tool":
                if pending_call_ids and _is_tool_result(content) and format_tools:
                    tool_msgs = _build_tool_messages(content, pending_call_ids)
                    converted.extend(tool_msgs)
                else:
                    logger.warning('')
                    converted.append({
                        'role': 'user',
                        'content': f'Tool result: {content}'
                    })
                pending_call_ids = []

        if save_to_meta:
            if self.metadata:
                self.metadata['openai_format'] = converted
            else:
                self.metadata = {'openai_format':converted}
        # self.context = converted
        return converted
            

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """Convert *already-normalised* api_set to OpenAI ``tools`` parameter.

        Function names are sanitised to match ``^[a-zA-Z0-9_-]+$``.
        """
        import re

        tools: list[dict[str, Any]] = []
        for api in self.api_set:
            safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", api["name"]).strip("_")
            tools.append({
                "type": "function",
                "function": {
                    "name": safe_name,
                    "description": api.get("description", ""),
                    "parameters": api.get("parameters", {"type": "object", "properties": {}}),
                },
            })
        return tools

    @staticmethod
    def from_openai_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
        """Convert OpenAI ``tool_calls`` back to standard call format.

        ``[func1(key="val"), func2(k="v")]``
        """
        parts: list[str] = []
        for tc in tool_calls:
            func = tc.get("function", tc)
            name = func["name"]
            try:
                args: dict[str, Any] = json.loads(func.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                args = {}
            arg_parts = [
                f'{k}={json.dumps(v, ensure_ascii=False)}'
                for k, v in args.items()
            ]
            parts.append(f"{name}({', '.join(arg_parts)})")
        return "[" + ", ".join(parts) + "]"


# ------------------------------------------------------------------
# conversion helpers
# ------------------------------------------------------------------

def _is_tool_call(content: str | None) -> bool:
    if content is None:
        return False
    return content.strip().startswith("[")


def _is_tool_result(content: str|None) -> bool:
    """True when *content* is the standard ``[{"name": ..., "result": ...}]`` format."""
    if content is None:
        return False
    stripped = content.strip()
    if not stripped.startswith("["):
        return False
    try:
        parsed = json.loads(stripped)
        return isinstance(parsed, list) and all(isinstance(r, dict) for r in parsed)
    except json.JSONDecodeError:
        return False


# def _convert_messages(messages: list[EvalSample.Context]) -> list[dict[str, Any]]:
#     converted: list[dict[str, Any]] = []
#     pending_call_ids: list[str] = []

#     for msg in messages:
#         role = msg["role"]
#         content = msg["content"].strip()

#         if role in ("system", "user"):
#             converted.append({"role": role, "content": content})
#             pending_call_ids = []

#         elif role == "assistant":
#             if _is_tool_call(content):
#                 tool_calls = _build_tool_calls(content, offset=len(converted))
#                 converted.append({
#                     "role": "assistant",
#                     "content": None,
#                     "tool_calls": tool_calls,
#                     "reasoning_content": None,
#                 })
#                 pending_call_ids = [tc["id"] for tc in tool_calls]
#             else:
#                 converted.append({
#                     "role": "assistant",
#                     "content": content,
#                     "reasoning_content": None,
#                 })
#                 pending_call_ids = []

#         elif role == "tool":
#             if pending_call_ids and _is_tool_result(content):
#                 tool_msgs = _build_tool_messages(content, pending_call_ids)
#                 converted.extend(tool_msgs)
#             else:
#                 fallback = f"Tool result: {content}"
#                 if converted and converted[-1].get("role") == "assistant":
#                     existing = converted[-1].get("content") or ""
#                     converted[-1]["content"] = f"{existing}\n{fallback}".strip()
#                 else:
#                     converted.append({"role": "user", "content": fallback})
#             pending_call_ids = []

#     return converted


def _build_tool_calls(content: str, offset: int = 0) -> list[dict[str, Any]]:
    calls = parse_call_string(content)
    tool_calls: list[dict[str, Any]] = []
    for i, call in enumerate(calls):
        tool_calls.append({
            "id": f"call_{offset + i}",
            "type": "function",
            "function": {
                "name": call["name"],
                "arguments": json.dumps(call["arguments"], ensure_ascii=False),
            },
        })
    return tool_calls


def _build_tool_messages(content: str, call_ids: list[str]) -> list['EvalSample.Context']:
    results = json.loads(content)  # guaranteed valid by _is_tool_result
    items: list[Any] = results if isinstance(results, list) else [results]

    if not call_ids:
        logger.warning('')
        return [{"role": "user", "content": f"Tool result: {json.dumps(results, ensure_ascii=False)}"}]

    msgs:list['EvalSample.Context'] = []
    for i, result in enumerate(items):
        if i < len(call_ids):
            msgs.append({
                "role": "tool",
                "tool_call_id": call_ids[i],
                "content": json.dumps(result, ensure_ascii=False),
            })
        elif msgs and msgs[-1]["content"]:
            prev = json.loads(msgs[-1]["content"])
            merged = [prev, result] if not isinstance(prev, list) else prev + [result]
            msgs[-1]["content"] = json.dumps(merged, ensure_ascii=False)

    # Ensure every pending call_id has at least one tool response
    for i in range(len(items), len(call_ids)):
        msgs.append({
            "role": "tool",
            "tool_call_id": call_ids[i],
            "content": json.dumps(None),
        })

    return msgs
