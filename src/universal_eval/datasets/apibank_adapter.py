from dataclasses import dataclass, asdict
import random
from .adapter import DatasetAdapter, EvalSample
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal
import json
import logging
import ast
import re

"""
api description:
{
  "apiCode": "Create_New_Session",
  "description": "Create a new virtual yoga or meditation session.",
  "parameters": {
    "session_name": {
      "type": "string",
      "description": "Name of the session.",
      "required": true
    },
    "session_date": {
      "type": "string",
      "description": "Date of the session, in the format yyyy-MM-dd.",
      "format": "formatted",
      "required": true
    },
    "session_time": {
      "type": "string",
      "description": "Time of the session, in the format HH:mm:ss.",
      "format": "formatted",
      "required": true
    },
    "session_instructor": {
      "type": "string",
      "description": "Name of the session instructor.",
      "required": true
    },
    "session_description": {
      "type": "string",
      "description": "Description of the session.",
      "required": true
    }
  },
  "response": {
    "data": {
      "description": "Confirmation of new session creation.",
      "type": "object",
      "properties": {
        "session_id": {
          "type": "integer",
          "description": "ID of the newly created session."
        },
        "status": {
          "type": "string",
          "description": "Status of the creation request."
        }
      }
    }
  }
}
"""

def _walk_and_fix(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively convert non-JSON-serializable Python objects (sets -> lists)."""
    if isinstance(obj, dict):
        return {k: _walk_and_fix(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_walk_and_fix(v) for v in obj]
    return obj


def _split_top_level_items(text: str) -> List[str]:
    items: List[str] = []
    current: List[str] = []
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    quote_char: str | None = None
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


def _format_answer_value(value: str) -> str:
    stripped = value.strip()
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(stripped)
        except Exception:
            return json.dumps(stripped, ensure_ascii=False)

    if isinstance(parsed, str):
        return json.dumps(parsed, ensure_ascii=False)
    if isinstance(parsed, bool):
        return str(parsed).lower()
    if isinstance(parsed, int):
        return str(parsed)
    if isinstance(parsed, float):
        return str(int(parsed)) if parsed.is_integer() else str(parsed)
    if isinstance(parsed, (dict, list, tuple)):
        return json.dumps(parsed, ensure_ascii=False, sort_keys=True)
    if parsed is None:
        return "null"
    return json.dumps(str(parsed), ensure_ascii=False)


def _normalize_answer_calls(target: str) -> str:
    text = target.strip()
    if not text.startswith("[") or not text.endswith("]"):
        return text

    inner = text[1:-1].strip()
    if not inner:
        return "[]"

    normalized_calls: List[str] = []
    for raw_call in _split_top_level_items(inner):
        match = re.match(r"^\s*(?P<name>[^()]+?)\s*\((?P<args>.*)\)\s*$", raw_call, re.DOTALL)
        if not match:
            normalized_calls.append(raw_call.strip())
            continue

        method_name = match.group("name").strip()
        args_text = match.group("args").strip()
        if not args_text:
            normalized_calls.append(f"{method_name}()")
            continue

        formatted_args: List[str] = []
        for raw_argument in _split_top_level_items(args_text):
            if "=" not in raw_argument:
                formatted_args.append(raw_argument.strip())
                continue
            key, value = raw_argument.split("=", 1)
            formatted_args.append(f"{key.strip()}={_format_answer_value(value)}")
        normalized_calls.append(f"{method_name}({', '.join(formatted_args)})")

    return "[" + ", ".join(normalized_calls) + "]"


def _normalize_api_set(raw_apis: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalise APIBank API definitions to standard format.

    Standard format: ``{name, description, parameters: {type: "object", properties, required}}``
    """
    type_map = {"list": "array", "float": "number", "int": "integer", "bool": "boolean", "str": "string"}
    normalized: List[Dict[str, Any]] = []
    for api in raw_apis:
        props: Dict[str, Any] = {}
        req: list[str] = []
        for pname, pdef in api.get("parameters", {}).items():
            if not isinstance(pdef, dict):
                continue
            raw_type = pdef.get("type", "string")
            prop: Dict[str, Any] = {"type": type_map.get(raw_type, raw_type)}
            if "description" in pdef:
                prop["description"] = pdef["description"]
            if "enum" in pdef:
                prop["enum"] = pdef["enum"]
            if "format" in pdef:
                prop["format"] = pdef["format"]
            if raw_type == "list" and "items" in pdef:
                items = pdef["items"]
                if isinstance(items, dict):
                    it = {"type": type_map.get(items.get("type", "string"), items.get("type", "string"))}
                    if "description" in items:
                        it["description"] = items["description"]
                    prop["items"] = it
            props[pname] = prop
            if pdef.get("required"):
                req.append(pname)
        normalized.append({
            "name": api.get("apiCode", api.get("name", "")),
            "description": api.get("description", ""),
            "parameters": {
                "type": "object",
                "properties": props,
                "required": req,
            },
        })
    return normalized


class APIBankDatasetAdapter(DatasetAdapter):
    name = "apibank"

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
        # dataset_path = Path(dataset_config['path'])
        # split = dataset_config['split']

        file_path = self._resolve_file(self.path, self.split)
        data = json.loads(file_path.read_text(encoding="utf-8"))
        # for index, item in enumerate(data):
        #     item['global_id'] = f"{file_path.name}_{index}"
        if random_samples:
            random.shuffle(data)

        samples: List[EvalSample] = []

        context_length = {}
        for index, item in enumerate(data):
            if limit is not None and index >= limit:
                break
            sample = self._parse_item(item, self.split['type'], conversation_style, strip_tool_descriptions)
            if not sample:
                continue

            metadata = {'context_length': len(sample.context), 'api_num': len(sample.api_set)}
            if with_raw_data:
                metadata['raw_data'] = item
            sample.metadata = metadata

            context_length[metadata['context_length']] = context_length.get(metadata['context_length'], 0) + 1
            samples.append(sample)
            
        self.logger.info(f"Loaded {len(samples)} samples from {len(data)} items")
        for context_len, count in context_length.items():
            self.logger.info(f"Context length {context_len}: {count} samples")
        return samples

    def _parse_item(
        self,
        item: Dict[str, Any],
        data_type: str,
        conversation_style: Literal['single', 'multi'] = 'single',
        strip_tool_descriptions: bool = False,
    ) -> EvalSample:
        # -- 提取api -- #
        api_set, api_lines = self._extract_api(item, data_type)
        if not api_set:
            self.logger.warning(f"Item {item['global_id']} has empty API set")

        # -- 提取答案 -- #
        target = item.get('output', item.get('expected_output', None))
        if target is None:
            raise ValueError(f"Item {item['global_id']} is missing output or expected_output")
        # Strip API-Request: prefix so the target uses the standard call format
        if target.startswith("API-Request:"):
            target = target.split("API-Request:", 1)[1].strip()
        elif target.startswith("API Request:"):
            target = target.split("API Request:", 1)[1].strip()
        target = _normalize_answer_calls(target)

        # -- 组装conversation-- #
        instruction = str(item['instruction'])
        if conversation_style == 'single':
            conversation = self._extract_single_round_conversation(item)
        else:
            conversation = self._extract_conversation_history(item)
            system_prompt = self._get_system_prompt(instruction, api_lines, strip_tool_descriptions)
            conversation.insert(0, {"role": "system", "content": system_prompt})
        
        return EvalSample(
            sample_id=item['global_id'],
            context=conversation,
            target=target,
            api_set=api_set
        )

    @staticmethod
    def _resolve_file(dataset_path: Path, split) -> Path:
        """Resolve the dataset file path from config parameters."""
        data_type = split['type']
        level = split['level']
        if data_type == 'test' and level == 3:
            raise NotImplementedError("Level 3 test data is not available yet.")
        subset = split.get('subset', '')
        if data_type == "test":
            if subset:
                file_name = f"level-{level}-{subset}.json"
            else:
                file_name = f"level-{level}.json"
            return dataset_path / "test-data" / file_name
        else:
            if subset:
                file_name = f"lv{level}-{subset}-train.json"
            else:
                file_name = f"lv{level}-train.json"
            return dataset_path / "training-data" / file_name

    def _extract_api(self, item: Dict[str, Any], data_type: str) -> tuple[List[Dict[str, Any]], List[str]]:
        """Extract JSON from lines and normalise to standard format."""
        if data_type == 'train':
            lines = item['input'].splitlines()
        elif data_type == 'test':
            lines = item['instruction'].splitlines()
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        raw_apis = []
        api_lines = []
        for line in lines:
            if line.startswith("{"):
                try:
                    raw_apis.append(json.loads(line))
                    api_lines.append(line)
                except json.JSONDecodeError:
                    self.logger.warning(f"Item [{item.get('global_id', 'unknown')}] has invalid JSON in input when extracting API description: {line}")
                    continue
            else:
                break
        return _normalize_api_set(raw_apis), api_lines

    """
    tool call response:
    API-Request: [tool_name(args)]->response_obj
    """

    def _extract_conversation_history(self, item: Dict[str, Any]) -> List[EvalSample.Context]:
        """Extract JSON from lines."""
        lines = item['input'].splitlines()
        conversation_json_objs: List[EvalSample.Context] = []

        for line in lines:
            if line.startswith("{") or line.startswith("Generate AI Response:"):
                continue
            elif line.startswith("User:"):
                conversation_json_objs.append({"role": "user", "content": line[5:].strip()})
            elif line.startswith("API-Request:") or line.startswith("API Request:"):
                try:
                    request, response = line[12:].strip().split("->", 1)
                    if not request.startswith('['):
                        request = '[' + request + ']'
                        self.logger.warning(f"Item [{item.get('global_id', 'unknown')}] has invalid API request when extracting conversation: {line}, auto repaired.")
                    api_name = request.split('(')[0].strip(' []')
                    if response == 'success':
                        result_data = {'status': 'success'}
                    else:
                        result_data = self._parse_response(response, item.get('global_id', 'unknown'))
                except (json.JSONDecodeError, ValueError) as e:
                    self.logger.debug(f"Item [{item.get('global_id', 'unknown')}] has invalid API response format when extracting conversation: {line} -- ({e.__class__.__name__}: {e})")
                    self.logger.info('One sample skipped due to invalid API-Request format.')
                    return []
                conversation_json_objs.append({"role": "assistant", "content": request.strip()})
                conversation_json_objs.append({
                    "role": "tool",
                    "content": json.dumps([{"name": api_name, "result": result_data}], ensure_ascii=False, sort_keys=True),
                })
            elif line.startswith("AI:"):
                conversation_json_objs.append({"role": "assistant", "content": line[3:].strip()})
            elif line.startswith('Generate API Request:'):
                break
            else:
                self.logger.warning(f"Item [{item.get('global_id', 'unknown')}] has unknown format line when extracting conversation: {line}") 
        return conversation_json_objs

    @staticmethod
    def _parse_response(response: str, global_id: str = "unknown") -> Dict[str, Any]:
        """Parse a response string, handling both Python dict and JSON formats."""
        response = response.strip()
        if not response:
            raise ValueError("Empty response")

        # Try JSON first if it looks like JSON (starts with { or [)
        if response[0] in '{[':
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                pass

        # Try Python literal (handles single-quoted dicts like {'name':'Tom'})
        try:
            obj = ast.literal_eval(response)
        except (ValueError, SyntaxError):
            raise ValueError(f"Failed to parse response: {response[:100]}")

        # Convert non-JSON-serializable types (e.g. sets -> lists)
        return _walk_and_fix(obj)

    def _extract_single_round_conversation(self, item: Dict[str, Any]) -> List[EvalSample.Context]:
        prompt = item['instruction'] + item['input']
        return [{"role": "user", "content": prompt}]

    def _get_current_time(self, s:str):
        curr_year, curr_time = None, None
        for line in s.splitlines():
            if 'The current time is' in line:
                curr_time = line.split('The current time is ')[1].strip()
            elif 'The current year is' in line:
                curr_year = line.split('The current year is ')[1].strip()
        return curr_year, curr_time

    def _get_system_prompt(self, instruction: str, api_lines: List[str], strip_tools: bool = False) -> str:
        curr_year, curr_time = self._get_current_time(instruction)
        if strip_tools:
            return f"""
You are a helpful assistant. Use the provided function tools to answer the user's request.
{f'the current year is {curr_year}' if curr_year else ''}
{f'the current time is {curr_time}' if curr_time else ''}
            """
        return f"""
You are a helpful assistant, you can use the following APIs when needed:
{','.join(api_lines)}
When using APIs, the expected output is:
API-Request: [ApiName(key1='value1', key2='value2', ...)]
{f'the current year is {curr_year}' if curr_year else ''}
{f'the current time is {curr_time}' if curr_time else ''}
            """

# path: ./data/API-Bank
#     split: 
#       type: test # test/train
#       level: 1 # 1/2/3
#       subset: api # api/response

# if __name__ == "__main__":
#     root = Path(__file__).resolve().parents[3]
#     adapter = APIBankDatasetAdapter()
#     config = {
#         "path": str(root / "data" / "API-Bank"),
#         "split": {
#             "type": "train",
#             "level": 2,
#             "subset": "api",
#         }
#     }
#     logging.basicConfig(
#         level = logging.DEBUG,
#         format='[%(asctime)s]%(name)s %(levelname)s: %(message)s'
#     )
#     samples = adapter.load_samples(config, with_raw_data=True, conversation_style='single', random_samples=True)
#     with open("test.json", "w", encoding="utf-8") as f:
#         json.dump([asdict(sample) for sample in samples], f, ensure_ascii=False, indent=4)
