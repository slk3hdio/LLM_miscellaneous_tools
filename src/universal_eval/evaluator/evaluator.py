from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
import tqdm
import logging

from ..datasets.sample import EvalSample
from .scoring import score_prediction
from ..providers import ModelProvider

logger = logging.getLogger(__name__)




@dataclass
class EvalRecord:
    """单条评测记录：包含样本、预测结果和评分。"""

    sample: EvalSample
    prediction: str
    score: Dict[str, Any]

    @classmethod
    def save(cls, records: List[EvalRecord], file_path: Path) -> None:
        """将评测记录保存为 JSONL 文件。"""
        with file_path.open("w", encoding="utf-8") as fp:
            for record in records:
                fp.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, file_path: Path) -> List[EvalRecord]:
        """从 JSONL 文件加载评测记录。"""
        records: List[EvalRecord] = []
        with file_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                records.append(cls(**json.loads(line)))
        return records


def evaluate_dataset(
    provider: ModelProvider,
    samples: List[EvalSample],
    conversation_style: Literal['single', 'multi'],
    use_standard_tool_format: bool = False,
    # output_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Any], List[EvalRecord]]:
    """对数据集逐样本进行评测。

    每个样本经过：上下文转换 → 模型生成 → 评分 → 记录。
    返回汇总指标和全部记录列表。
    """
    records: List[EvalRecord] = []
    correct = 0
    name_correct = 0

    sample_iter = tqdm.tqdm(samples, desc="Evaluating") 
    for index, sample in enumerate(sample_iter, start=1):
        logger.debug(f"Processing sample {sample.sample_id}")
        messages = sample.to_openai_messages(format_tools = use_standard_tool_format)
        tools = sample.to_openai_tools() or None
        if tools is None and use_standard_tool_format:
            logger.warning(f"Did not find tool set for sample {sample.sample_id} when using standard tool format")

        prediction = provider.generate(messages, conversation_style=conversation_style, tools=tools)
        score = score_prediction(sample, prediction)
        exact_match = bool(score.get("exact_match"))
        correct += int(exact_match)
        name_correct += int(score.get("method_name_match", False))

        record = EvalRecord(sample=sample, prediction=prediction, score=score)
        records.append(record)

    total = len(samples)
    summary = {
        "total": total,
        "exact_match_count": correct,
        "exact_match_rate": (correct / total) if total else 0.0,
        "method_name_match_count": name_correct,
        "method_name_match_rate": (name_correct / total) if total else 0.0,
    }
    return summary, records
