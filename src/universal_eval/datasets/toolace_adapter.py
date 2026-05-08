import random
from .adapter import DatasetAdapter, EvalSample
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal
import json
from dataclasses import asdict
import logging
import re


def _sanitize_call_string(content: str) -> str:
    """Sanitise function names inside a call string ``[func(args)]``."""
    import re
    return re.sub(
        r'(\[|, )([a-zA-Z][a-zA-Z0-9 ._-]*?)\s*\(',
        lambda m: m.group(1) + _sanitize_name(m.group(2)) + '(',
        content,
    )


def _sanitize_name(name: str) -> str:
    """Replace non-alphanumeric characters with underscores for OpenAI compatibility."""
    import re
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")


def _normalize_api_set(raw_apis: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalise ToolACE API definitions to standard format.

    Standard format: ``{name, description, parameters: {type: "object", properties, required}}``
    """
    type_map = {"list": "array", "float": "number", "int": "integer", "bool": "boolean", "dict": "object"}

    def _fix_types(obj: Any) -> Any:
        if isinstance(obj, dict):
            fixed: Dict[str, Any] = {}
            for k, v in obj.items():
                if k == "type" and isinstance(v, str):
                    fixed[k] = type_map.get(v, v)
                elif k == "items" and isinstance(v, dict):
                    fixed[k] = _fix_types(v)
                else:
                    fixed[k] = _fix_types(v) if isinstance(v, (dict, list)) else v
            return fixed
        if isinstance(obj, list):
            return [_fix_types(v) for v in obj]
        return obj

    normalized: List[Dict[str, Any]] = []
    for api in raw_apis:
        name = _sanitize_name(api.get("name", api.get("tool_name", "")))
        desc = api.get("description", api.get("definition", ""))
        params = api.get("parameters", {})

        if isinstance(params, dict) and "properties" in params:
            properties = _fix_types(params.get("properties", {}))
            required = params.get("required", [])
        elif isinstance(params, dict) and params:
            # Flat params format: {param: {type, description}}
            properties = {}
            required = []
            for pname, pdef in params.items():
                if not isinstance(pdef, dict):
                    continue
                raw_type = pdef.get("type", "string")
                prop: Dict[str, Any] = {"type": type_map.get(raw_type, raw_type)}
                if "description" in pdef:
                    prop["description"] = pdef["description"]
                properties[pname] = prop
                if pdef.get("required"):
                    required.append(pname)
        else:
            properties = {}
            required = []

        normalized.append({
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        })
    return normalized


def _normalize_tool_result(content: str) -> str:
    """Normalise ToolACE tool-result key ``results`` → ``result``."""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return content
    if isinstance(parsed, list):
        for obj in parsed:
            if isinstance(obj, dict) and "results" in obj and "result" not in obj:
                obj["result"] = obj.pop("results")
        return json.dumps(parsed, ensure_ascii=False)
    if isinstance(parsed, dict) and "results" in parsed and "result" not in parsed:
        parsed["result"] = parsed.pop("results")
        return json.dumps(parsed, ensure_ascii=False)
    return content


class ToolACEDatasetAdapter(DatasetAdapter):
    name = "toolace"

    def __init__(self, path, split):
        super().__init__(path, split)
        self.logger = logging.getLogger(__name__)

    def load_samples(
        self,
        limit: int | None = None,
        conversation_style: Literal['single', 'multi'] = 'single',
        with_raw_data: bool = False,
        random_samples: bool = False,
        strip_tool_descriptions: bool = False,
    ) -> List[EvalSample]:
        file_name = 'data.json' if self.split == 'all' else 'data_smoke_20.json'
        file_path = self.path / file_name
        data = json.loads(file_path.read_text(encoding="utf-8"))

        if random_samples:
            random.shuffle(data)

        samples: List[EvalSample] = []
        for index, item in enumerate(data):
            if limit is not None and index >= limit:
                break
            sample = self._parse_item(item, index, file_name, conversation_style, strip_tool_descriptions)
            if sample is None:
                continue
            if with_raw_data:
                sample.metadata = {**(sample.metadata or {}), "raw_data": item}
            samples.append(sample)

        self.logger.info(f"Loaded {len(samples)} samples from {len(data)} items")
        return samples

    def _parse_item(
        self,
        item: Dict[str, Any],
        index: int,
        file_name: str,
        conversation_style: Literal['single', 'multi'] = 'single',
        strip_tool_descriptions: bool = False,
    ) -> Optional[EvalSample]:
        system_prompt = item.get('system', '')
        conversations = item.get('conversations', [])
        if not conversations:
            self.logger.warning(f"Item {index} has no conversations, skipping")
            return None

        api_set = _normalize_api_set(self._extract_apis(system_prompt))

        if strip_tool_descriptions and conversation_style == 'multi':
            system_prompt = self._strip_apis(system_prompt)

        # Find the last assistant turn that contains a tool call
        tool_call_idx = None
        for i in range(len(conversations) - 1, -1, -1):
            turn = conversations[i]
            if turn['from'] == 'assistant' and turn['value'].strip().startswith('['):
                tool_call_idx = i
                break

        if tool_call_idx is None:
            return None

        target = _sanitize_call_string(conversations[tool_call_idx]['value'].strip())
        context = self._build_context(system_prompt, conversations, tool_call_idx, conversation_style, sanitize_calls=True)

        sample_id = f"{file_name}_{index}"
        return EvalSample(
            sample_id=sample_id,
            context=context,
            target=target,
            api_set=api_set,
        )

    @staticmethod
    def _extract_apis(system_prompt: str) -> List[Dict[str, Any]]:
        """Extract function definitions from the system prompt, handling multiple formats."""
        # Format 1: JSON array — "Here is a list of functions in JSON format that you can invoke:\n[...]"
        marker = 'Here is a list of functions in JSON format that you can invoke:'
        if marker in system_prompt:
            rest = system_prompt[system_prompt.find(marker) + len(marker):]
            arr_start = rest.find('[')
            if arr_start >= 0:
                depth = 0
                arr_end = -1
                for i in range(arr_start, len(rest)):
                    if rest[i] == '[':
                        depth += 1
                    elif rest[i] == ']':
                        depth -= 1
                        if depth == 0:
                            arr_end = i + 1
                            break
                if arr_end > 0:
                    try:
                        return json.loads(rest[arr_start:arr_end])
                    except json.JSONDecodeError:
                        pass
            return []

        # Format 2: Various "Here are the tools you can use:" variants
        marker2 = 'Here are the tools you can use:'
        if marker2 not in system_prompt:
            return []

        rest = system_prompt[system_prompt.find(marker2) + len(marker2):].strip()

        # Format 2a: Single JSON object — {"tool_name": "...", ...}
        if rest.startswith('{'):
            depth = 0
            obj_end = -1
            for i, ch in enumerate(rest):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        obj_end = i + 1
                        break
            if obj_end > 0:
                try:
                    obj = json.loads(rest[:obj_end])
                    return [obj]
                except json.JSONDecodeError:
                    pass
            return []

        # Format 2b: HTML table
        if rest.startswith('<table'):
            return ToolACEDatasetAdapter._extract_from_html(rest)

        # Format 2c: XML tags — <tool_name>...</tool_name><definition>...</definition>...
        if rest.startswith('<tool_name>'):
            return ToolACEDatasetAdapter._extract_from_xml(rest)

        # Format 2d: Markdown-style — "- **tool_name**: ..."
        if '- **tool_name**' in rest:
            return ToolACEDatasetAdapter._extract_from_markdown(rest)

        # Format 2e: Fallback — try to find any tool name in the text
        return ToolACEDatasetAdapter._extract_tool_names_from_text(rest)

    @staticmethod
    def _extract_from_html(text: str) -> List[Dict[str, Any]]:
        """Extract tool definitions from HTML table format."""
        apis = []
        # Find all <tr> rows after the header
        rows = re.findall(r'<tr>(.*?)</tr>', text, re.DOTALL)
        for row in rows[1:]:  # Skip header row
            cols = re.findall(r'<td>(.*?)</td>', row, re.DOTALL)
            if len(cols) >= 1:
                name = re.sub(r'<[^>]+>', '', cols[0]).strip()
                definition = re.sub(r'<[^>]+>', '', cols[1]).strip() if len(cols) > 1 else ''
                apis.append({'name': name, 'description': definition, 'parameters': {}})
        return apis

    @staticmethod
    def _extract_from_xml(text: str) -> List[Dict[str, Any]]:
        """Extract tool definitions from XML-style format."""
        apis = []
        names = re.findall(r'<tool_name>(.*?)</tool_name>', text)
        defs = re.findall(r'<definition>(.*?)</definition>', text, re.DOTALL)
        for i, name in enumerate(names):
            desc = defs[i].strip() if i < len(defs) else ''
            apis.append({'name': name.strip(), 'description': desc, 'parameters': {}})
        return apis

    @staticmethod
    def _extract_from_markdown(text: str) -> List[Dict[str, Any]]:
        """Extract tool definitions from markdown-style format."""
        apis = []
        # Split by tool_name entries
        blocks = re.split(r'- \*\*tool_name\*\*:', text)
        for block in blocks[1:]:  # Skip content before first tool
            lines = block.strip().split('\n')
            name = lines[0].strip() if lines else 'unknown'
            desc = ''
            for line in lines[1:]:
                if line.strip().startswith('- **definition**'):
                    desc = line.split('**:**', 1)[-1].strip() if '**:**' in line else line.split('**:', 1)[-1].strip()
                    break
            apis.append({'name': name, 'description': desc, 'parameters': {}})
        return apis

    @staticmethod
    def _extract_tool_names_from_text(text: str) -> List[Dict[str, Any]]:
        """Fallback: extract tool names from free-text format descriptions."""
        apis = []
        # Look for patterns like: tool_name: SomeName, "ToolName", etc.
        names = re.findall(r'(?:tool_name|name)[:\s]+["\']?(\w+)', text, re.IGNORECASE)
        for name in set(names):
            apis.append({'name': name, 'description': '', 'parameters': {}})
        return apis

    @staticmethod
    def _strip_apis(system_prompt: str) -> str:
        """Remove API definitions from the system prompt, keeping instructions."""
        import re

        # Format 1: "Here is a list of functions in JSON format that you can invoke:\n[...]"
        marker = "Here is a list of functions in JSON format that you can invoke:"
        if marker in system_prompt:
            before = system_prompt[:system_prompt.find(marker) + len(marker)]
            # Find the end of the JSON array
            rest = system_prompt[system_prompt.find(marker) + len(marker):]
            arr_start = rest.find("[")
            if arr_start >= 0:
                depth = 0
                arr_end = -1
                for i in range(arr_start, len(rest)):
                    if rest[i] == "[":
                        depth += 1
                    elif rest[i] == "]":
                        depth -= 1
                        if depth == 0:
                            arr_end = i + 1
                            break
                if arr_end > 0:
                    after = rest[arr_end:]
                    return before + "\n[Tools provided via the API tools parameter]\n" + after
            return system_prompt

        # Format 2: "Here are the tools you can use:\n..." (HTML/XML/markdown/json)
        marker2 = "Here are the tools you can use:"
        if marker2 in system_prompt:
            before = system_prompt[:system_prompt.find(marker2) + len(marker2)]
            rest = system_prompt[system_prompt.find(marker2) + len(marker2):]
            # Find the usage instructions after the tool definitions
            after_marker = re.search(
                r"(Please use the following format|Should you decide to return)",
                rest,
            )
            if after_marker:
                after = rest[after_marker.start():]
            else:
                after = ""
            return before + "\n[Tools provided via the API tools parameter]\n" + after

        return system_prompt

    @staticmethod
    def _build_context(
        system_prompt: str,
        conversations: List[Dict[str, str]],
        tool_call_idx: int,
        conversation_style: Literal['single', 'multi'],
        sanitize_calls: bool = False,
    ) -> List[EvalSample.Context]:
        context_turns = conversations[:tool_call_idx]

        if conversation_style == 'single':
            parts = [system_prompt]
            for turn in context_turns:
                role_label = 'User' if turn['from'] == 'user' else 'Assistant' if turn['from'] == 'assistant' else 'Tool'
                value = turn['value']
                if sanitize_calls and turn['from'] == 'assistant':
                    value = _sanitize_call_string(value)
                parts.append(f"{role_label}: {value}")
            return [{"role": "user", "content": '\n\n'.join(parts)}]

        # Multi-turn style
        context: List[EvalSample.Context] = []
        context.append({"role": "system", "content": system_prompt})
        for turn in context_turns:
            role = turn['from'] if turn['from'] in ('user', 'assistant', 'tool') else 'user'
            value = turn['value']
            if sanitize_calls and role == 'assistant':
                value = _sanitize_call_string(value)
            if role == 'tool':
                value = _normalize_tool_result(value)
            context.append({"role": role, "content": value})
        return context


# if __name__ == "__main__":
#     root = Path(__file__).resolve().parents[3]
#     config = {
#         "path": str(root / "data" / "ToolACE"),
#         "split": {
#             "file": "data.json",
#         }
#     }
#     adapter = ToolACEDatasetAdapter(config)
#     logging.basicConfig(
#         level=logging.INFO,
#         format='[%(asctime)s]%(name)s %(levelname)s: %(message)s'
#     )
#     samples = adapter.load_samples(conversation_style='multi', random_samples=False, limit=10)
#     with open("test.json", "w", encoding="utf-8") as f:
#         json.dump([asdict(sample) for sample in samples], f, ensure_ascii=False, indent=4)

        
