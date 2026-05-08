from __future__ import annotations

import json

from universal_eval.datasets import EvalSample
from universal_eval.evaluator.evaluator import evaluate_dataset
from universal_eval.providers import ModelProvider


class DummyProvider(ModelProvider):
    def __init__(self, responses):
        super().__init__(model="dummy")
        self._responses = list(responses)
        self.calls = []

    def generate(self, messages, tools=None) -> str:
        self.calls.append({"messages": messages, "tools": tools})
        return self._responses.pop(0)


def test_evaluate_dataset_computes_summary_and_writes_files(tmp_path):
    samples = [
        EvalSample(
            sample_id="a",
            context=[{"role": "user", "content": "prompt-a"}],
            target="[A()]",
            api_set=[{"name": "A", "description": "", "parameters": {"type": "object", "properties": {}}}],
        ),
        EvalSample(
            sample_id="b",
            context=[{"role": "user", "content": "prompt-b"}],
            target="[B()]",
            api_set=[{"name": "B", "description": "", "parameters": {"type": "object", "properties": {}}}],
        ),
    ]
    provider = DummyProvider(["[A()]", "wrong"])

    summary_data, records = evaluate_dataset(
        provider=provider,
        samples=samples,
        output_dir=tmp_path,
    )

    assert summary_data == {
        "total": 2,
        "exact_match_count": 1,
        "exact_match_rate": 0.5,
    }
    assert len(records) == 2
    assert provider.calls == [
        {"messages": [{"role": "user", "content": "prompt-a"}], "tools": None},
        {"messages": [{"role": "user", "content": "prompt-b"}], "tools": None},
    ]

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary == summary_data

    lines = (tmp_path / "predictions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first_record = json.loads(lines[0])
    assert first_record["sample"]["sample_id"] == "a"
    assert first_record["score"]["exact_match"] is True


def test_evaluate_dataset_handles_empty_samples(tmp_path):
    provider = DummyProvider([])

    summary_data, records = evaluate_dataset(
        provider=provider,
        samples=[],
        output_dir=tmp_path,
    )

    assert summary_data == {
        "total": 0,
        "exact_match_count": 0,
        "exact_match_rate": 0.0,
    }
    assert records == []


def test_evaluate_dataset_uses_standard_tool_format_when_enabled(tmp_path):
    sample = EvalSample(
        sample_id="tool",
        context=[{"role": "user", "content": "call A"}],
        target="[A()]",
        api_set=[{"name": "A", "description": "demo", "parameters": {"type": "object", "properties": {}}}],
    )
    provider = DummyProvider(["[A()]"])

    evaluate_dataset(
        provider=provider,
        samples=[sample],
        use_standard_tool_format=True,
        output_dir=tmp_path,
    )

    assert provider.calls == [
        {
            "messages": [{"role": "user", "content": "call A"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "A",
                        "description": "demo",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        }
    ]
