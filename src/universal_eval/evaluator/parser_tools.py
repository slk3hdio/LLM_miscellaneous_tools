
from typing import Optional, List, Dict, Any, Tuple
import json
import re
import ast



def normalize_text(text: str) -> str:
    """规范化文本：合并连续空白字符为单个空格。"""
    return " ".join(text.strip().split())


def _extract_call_block(text: str) -> Optional[str]:
    """从文本中提取最外层的 ``[...]`` 工具调用块（跟踪括号深度）。"""
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


def split_top_level_items(text: str) -> List[str]:
    """按顶层逗号分割字符串，忽略括号、引号内的逗号。"""
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
    """规范化参数值：尝试 JSON/ast 解析，统一布尔、浮点、容器类型的输出。"""
    stripped = value.strip()
    # Try JSON first (handles true/false/null)
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(stripped)
        except Exception:
            return normalize_text(stripped)

    if isinstance(parsed, bool):
        return str(parsed).lower()
    if isinstance(parsed, float) and parsed == int(parsed):
        return str(int(parsed))
    if isinstance(parsed, (dict, list, tuple)):
        return json.dumps(parsed, ensure_ascii=False, sort_keys=True)
    return str(parsed)


def parse_call_string(text: str) -> List[Dict[str, Any]]:
    """将 ``[func(key="val"), ...]`` 格式的工具调用字符串解析为结构化字典列表。"""
    block = _extract_call_block(text)
    if not block:
        return []

    inner = block[1:-1].strip()
    if not inner:
        return []

    calls: List[Dict[str, Any]] = []
    for raw_call in split_top_level_items(inner):
        match = re.match(r"^\s*(?P<name>[^()]+?)\s*\((?P<args>.*)\)\s*$", raw_call, re.DOTALL)
        if not match:
            continue

        method_name = normalize_text(match.group("name"))
        arguments: Dict[str, str] = {}
        args_text = match.group("args").strip()
        if args_text:
            for raw_argument in split_top_level_items(args_text):
                if "=" not in raw_argument:
                    continue
                key, value = raw_argument.split("=", 1)
                arguments[normalize_text(key)] = _normalize_argument_value(value)

        calls.append({"name": method_name, "arguments": arguments})

    return calls


def sanitize_name(name: str) -> str:
    """净化函数名：将非字母数字字符替换为下划线，确保符合 OpenAI 命名规范 ``^[a-zA-Z0-9_-]+$``。"""
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")


def format_call_string(calls: List[Dict[str, Any]]) -> str:
    """将结构化调用列表反序列化为 ``[func(key="val"), ...]`` 格式的标准字符串。

    与 :func:`parse_call_string` 互为逆操作。
    """
    parts: list[str] = []
    for call in calls:
        arg_parts = [
            f'{k}={json.dumps(v, ensure_ascii=False)}'
            for k, v in call["arguments"].items()
        ]
        parts.append(f'{call["name"]}({", ".join(arg_parts)})')
    return "[" + ", ".join(parts) + "]"