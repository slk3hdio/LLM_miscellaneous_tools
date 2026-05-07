from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import tqdm

from .datasets import DatasetAdapter, EvalSample
from .datasets.scoring import score_prediction
from .providers import ModelProvider


@dataclass
class EvalRecord:
    sample: EvalSample
    prediction: str
    score: Dict[str, Any]

    @classmethod
    def save(cls, records: List[EvalRecord], file_path: Path) -> None:
        with file_path.open("w", encoding="utf-8") as fp:
            for record in records:
                fp.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, file_path: Path) -> List[EvalRecord]:
        records: List[EvalRecord] = []
        with file_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                records.append(cls(**json.loads(line)))
        return records


def evaluate_dataset(
    provider: ModelProvider,
    samples: List[EvalSample],
    output_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Any], List[EvalRecord]]:
    records: List[EvalRecord] = []
    correct = 0
    use_tools = getattr(provider, "supports_tools", False)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for index, sample in enumerate(tqdm.tqdm(samples, desc="Evaluating"), start=1):
        if use_tools:
            messages = sample.to_openai_messages()
            tools = sample.to_openai_tools() or None
        else:
            messages = sample.context
            tools = None

        prediction = provider.generate(messages, tools=tools)
        score = score_prediction(sample, prediction)
        exact_match = bool(score.get("exact_match"))
        correct += int(exact_match)

        record = EvalRecord(sample=sample, prediction=prediction, score=score)
        records.append(record)

    total = len(samples)
    summary = {
        "total": total,
        "exact_match_count": correct,
        "exact_match_rate": (correct / total) if total else 0.0,
    }

    if output_dir is not None:
        with (output_dir / "summary.json").open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, ensure_ascii=False, indent=2)
        EvalRecord.save(records, output_dir / "predictions.jsonl")

    return summary, records
