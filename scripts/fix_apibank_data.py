"""Fix Python dict response strings in API-Bank dataset files.

Issues fixed:
1. Bare Python type names (str, int, float, etc.) used as dict values → quoted
2. Apostrophes in single-quoted strings → escaped for ast.literal_eval
3. Python sets (e.g. {'none'}) → converted to lists for JSON compatibility
4. Truncated data → best-effort structural repair
"""

import json
import re
import ast
from pathlib import Path
from typing import Any


def _walk_and_fix(obj: Any) -> Any:
    """Recursively convert non-JSON-serializable Python objects."""
    if isinstance(obj, dict):
        return {k: _walk_and_fix(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_fix(v) for v in obj]
    if isinstance(obj, tuple):
        return [_walk_and_fix(v) for v in obj]
    if isinstance(obj, set):
        return [_walk_and_fix(v) for v in obj]
    if isinstance(obj, frozenset):
        return [_walk_and_fix(v) for v in obj]
    return obj


def fix_python_dict_response(response_str: str) -> str:
    """Convert a Python dict string to valid JSON, handling common quirks."""
    s = response_str.strip()

    # Step 1: Fix known truncated data patterns (specific to this dataset)
    s = s.replace(
        "'location': {'type...test appoinment.', 'time_slot'",
        "'location': {'type': 'str', 'description': 'The city location for the lab test appoinment.'}, 'time_slot'"
    )

    # Step 2: Quote bare Python type names used as dict values
    bare_types = ['str', 'int', 'float', 'list', 'dict', 'bool', 'datetime', 'Any', 'Union']
    for t in bare_types:
        s = re.sub(r'(?<=:\s)(' + t + r')(?=\s*[,}\]])', r'"\1"', s)

    # Step 3: Escape apostrophes inside single-quoted strings
    s = re.sub(r"(?<=[a-zA-Z])'(?=[a-zA-Z])", r"\\'", s)

    # Step 4: Parse as Python literal
    try:
        obj = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        # If still failing, return original unchanged
        return response_str

    # Step 5: Convert non-JSON-compatible types
    obj = _walk_and_fix(obj)

    return json.dumps(obj, ensure_ascii=False)


def fix_item_input(input_str: str) -> str:
    """Fix all API-Request response strings in an item's input field."""
    lines = input_str.splitlines()
    fixed_lines = []
    for line in lines:
        if 'API-Request:' in line or 'API Request:' in line:
            try:
                prefix, response = line.split('->', 1)
            except ValueError:
                fixed_lines.append(line)
                continue
            fixed_response = fix_python_dict_response(response)
            fixed_lines.append(prefix + '->' + fixed_response)
        else:
            fixed_lines.append(line)
    return '\n'.join(fixed_lines)


def fix_file(filepath: Path) -> int:
    """Fix a single JSON data file. Returns number of lines fixed."""
    data = json.loads(filepath.read_text(encoding='utf-8'))
    fixed_count = 0

    for item in data:
        if not isinstance(item, dict):
            continue
        inp = item.get('input', '')
        if not inp:
            continue
        fixed_inp = fix_item_input(inp)
        if fixed_inp != inp:
            item['input'] = fixed_inp
            fixed_count += 1

    if fixed_count > 0:
        filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

    return fixed_count


def main():
    base = Path(__file__).resolve().parents[1] / "data" / "API-Bank"
    total_fixed = 0

    for f in sorted(base.rglob('*.json')):
        n = fix_file(f)
        if n > 0:
            print(f'  Fixed {n} items in {f.relative_to(base)}')
            total_fixed += n

    print(f'\nTotal items fixed: {total_fixed}')


if __name__ == '__main__':
    main()
