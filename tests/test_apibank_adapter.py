from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "src" / "universal_eval"


def _load_apibank_module():
    package_names = [
        ("universal_eval", ROOT),
        ("universal_eval.evaluator", ROOT / "evaluator"),
        ("universal_eval.datasets", ROOT / "datasets"),
    ]
    for name, path in package_names:
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module

    for name, file_path in [
        ("universal_eval.evaluator.parser_tools", ROOT / "evaluator" / "parser_tools.py"),
        ("universal_eval.datasets.sample", ROOT / "datasets" / "sample.py"),
        ("universal_eval.datasets.adapter", ROOT / "datasets" / "adapter.py"),
        ("universal_eval.datasets.apibank_adapter", ROOT / "datasets" / "apibank_adapter.py"),
    ]:
        if name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)

    return sys.modules["universal_eval.datasets.apibank_adapter"]


def test_apibank_answer_values_use_double_quotes():
    mod = _load_apibank_module()

    normalized = mod._normalize_answer_calls(
        "[Weather(city='Beijing', count=3, ok=True, tags=['a', 'b'])]"
    )

    assert normalized == '[Weather(city="Beijing", count=3, ok=true, tags=["a", "b"])]'
