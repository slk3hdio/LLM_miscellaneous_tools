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
        import os as _os
        logger.debug("Saving %d records to %s", len(records), file_path)
        with file_path.open("w", encoding="utf-8") as fp:
            for idx, record in enumerate(records):
                fp.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            fp.flush()
            _os.fsync(fp.fileno())
        actual = sum(1 for _ in file_path.open("r", encoding="utf-8"))
        logger.debug("Wrote %d records, verified %d lines on disk", len(records), actual)
        if actual != len(records):
            logger.error(
                "MISMATCH: expected %d records but only %d written to %s",
                len(records), actual, file_path,
            )

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
    batch_size: int = 1,
    # output_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Any], List[EvalRecord]]:
    """对数据集逐批评测。

    每个样本经过：上下文转换 → 批量模型生成 → 评分 → 记录。
    返回汇总指标和全部记录列表。
    """
    records: List[EvalRecord] = []
    correct = 0
    name_correct = 0

    total_batches = (len(samples) + batch_size - 1) // batch_size
    batch_iter = tqdm.tqdm(range(0, len(samples), batch_size), desc="Evaluating", total=total_batches)
    for i in batch_iter:
        batch = samples[i:i + batch_size]

        # 准备 batch 输入
        batch_messages: list[list[EvalSample.Context]] = []
        batch_tools: list[list[dict[str, Any]] | None] = []
        for sample in batch:
            batch_messages.append(sample.to_openai_messages(format_tools=use_standard_tool_format))
            if use_standard_tool_format:
                tools = sample.to_openai_tools() or None
                if tools is None:
                    logger.warning("Did not find tool set for sample %s when using standard tool format", sample.sample_id)
            else:
                tools = None
            batch_tools.append(tools)

        # 批量生成
        predictions = provider.generate_batch(batch_messages, conversation_style, batch_tools)

        # 逐条评分
        for sample, prediction in zip(batch, predictions):
            sample.prediction = prediction
            score = score_prediction(sample, prediction)
            exact_match = bool(score.get("exact_match"))
            correct += int(exact_match)
            name_correct += int(score.get("method_name_match", False))
            records.append(EvalRecord(sample=sample, prediction=prediction, score=score))

    total = len(samples)
    summary = {
        "total": total,
        "exact_match_count": correct,
        "exact_match_rate": (correct / total) if total else 0.0,
        "method_name_match_count": name_correct,
        "method_name_match_rate": (name_correct / total) if total else 0.0,
    }
    return summary, records
