from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional
from ..evaluator.parser_tools import parse_call_string, sanitize_name, format_call_string
from typing import TypedDict
from logging import getLogger
from typing_extensions import NotRequired, Required
import re

logger = getLogger(__name__)

@dataclass
class EvalSample:
    """标准化评测样本。

    所有适配器必须按以下规范格式产出数据，确保转换工具能基于统一表示工作。

    标准格式
    --------
    **API 定义条目** (``api_set``)::

        {
            "name": "func_name",
            "description": "...",
            "parameters": {
                "type": "object",
                "properties": {"p": {"type": "string", "description": "..."}},
                "required": ["p"]
            }
        }

    **助手工具调用** (context ``content``)::

        [func_name(key1="val1", key2="val2")]

    **工具返回结果** (context ``content``)::

        [{"name": "func_name", "result": {...}}]
    """

    class Context(TypedDict):
        role: Required[Literal["user", "assistant", "system", "tool"]]
        content: Required[Optional[str]]
        tool_calls: NotRequired[List[Any]]
        tool_call_id: NotRequired[str]
        reasoning_content: NotRequired[Optional[str]]
    sample_id: str
    api_set: List[Dict[str, Any]]
    context: List[Context]
    target: str
    prediction: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # OpenAI ChatCompletion helpers
    # ------------------------------------------------------------------

    def to_openai_messages(self, format_tools: bool, save_to_meta = True) -> list["EvalSample.Context"]:
        """将标准 context 转换为 OpenAI ChatCompletion 格式。

        ``format_tools=True`` 时将 ``[func(args)]`` 文本转为结构化 tool_calls，
        将 ``[{"name":..., "result":...}]`` 转为 tool 角色消息。
        """
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
                    })
                    pending_call_ids = [tc["id"] for tc in tool_calls]
                else:
                    converted.append({
                        "role": "assistant",
                        "content": content,
                    })
                    pending_call_ids = []

            elif role == "tool":
                if pending_call_ids and _is_tool_result(content) and format_tools:
                    tool_msgs = _build_tool_messages(content, pending_call_ids)
                    converted.extend(tool_msgs)
                else:
                    if format_tools:
                        logger.warning("Tool result doesn't match expected format in standard mode: %s", content[:120])
                    else:
                        logger.debug("Tool result in plain mode (expected): %s", content[:80])
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
        """将已规范化的 api_set 转换为 OpenAI ``tools`` 参数。

        函数名会被净化以匹配 ``^[a-zA-Z0-9_-]+$`` 规范。
        """
        tools: list[dict[str, Any]] = []
        for api in self.api_set:
            safe_name = sanitize_name(api["name"])
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
        """将 OpenAI ``tool_calls`` 转换回标准调用格式 ``[func1(key="val"), func2(k="v")]``。"""
        calls: list[dict[str, Any]] = []
        for tc in tool_calls:
            func = tc.get("function", tc)
            try:
                args: dict[str, Any] = json.loads(func.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                args = {}
            calls.append({"name": func["name"], "arguments": args})
        return format_call_string(calls)

    @staticmethod
    def normalize_raw_tool_calls(raw_calls: list[dict[str, Any]]) -> str:
        """将 LLM 输出的原始 tool_call 字典（可能格式不规范）规范化为标准 ``[func(key="val")]`` 格式。

        兼容多种输入格式：OpenAI Pydantic 模型、纯字典、缺少 function 包装层的扁平格式等。
        """
        tool_calls: list[dict[str, Any]] = []
        for i, item in enumerate(raw_calls):
            if not isinstance(item, dict):
                continue
            func = item.get("function", item)
            name = func.get("name")
            if not name:
                continue
            arguments = func.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            elif not isinstance(arguments, dict):
                arguments = {}
            tool_calls.append({
                "id": item.get("id", f"call_{i}"),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            })
        return EvalSample.from_openai_tool_calls(tool_calls) if tool_calls else ""
# ------------------------------------------------------------------
# conversion helpers
# ------------------------------------------------------------------

def _is_tool_call(content: str | None) -> bool:
    """判断内容是否为工具调用字符串（以 ``[`` 开头）。"""
    if content is None:
        return False
    return content.strip().startswith("[")


def _is_tool_result(content: str|None) -> bool:
    """判断内容是否为标准工具结果格式 ``[{"name": ..., "result": ...}]``。"""
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


def _build_tool_calls(content: str, offset: int = 0) -> list[dict[str, Any]]:
    """将标准工具调用字符串解析为 OpenAI tool_calls 格式。"""
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
    """将标准工具结果字符串转换为 OpenAI tool 角色消息列表。"""
    results = json.loads(content)  # guaranteed valid by _is_tool_result
    items: list[Any] = results if isinstance(results, list) else [results]

    if not call_ids:
        logger.warning("No tool call IDs provided for tool results.")
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
