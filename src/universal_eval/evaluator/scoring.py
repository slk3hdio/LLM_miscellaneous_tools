from __future__ import annotations

from typing import Any, Dict, List, Optional

from .parser_tools import parse_call_string, normalize_text, extract_call_block, format_call_string
from ..datasets.sample import EvalSample


def _build_argument_entries(calls: List[Dict[str, Any]]) -> set[str]:
    entries: set[str] = set()
    for index, call in enumerate(calls):
        for key, value in sorted(call["arguments"].items()):
            entries.add(f"{index}|{call['name']}|{key}={value}")
    return entries


def _compute_iou(left: set[str], right: set[str]) -> Optional[float]:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def score_prediction(sample: EvalSample, prediction: str) -> Dict[str, Any]:
    """对模型预测进行评分。

    将预测与目标字符串均按 ``[func(key="val"), ...]`` 格式解析，
    比较函数名（精确匹配）和参数（IoU 交并比）。
    """
    # normalized_prediction = normalize_text(prediction)
    # normalized_target = normalize_text(sample.target)
    prediction_calls = parse_call_string(prediction)
    normalized_prediction = format_call_string(prediction_calls)
    target_calls = parse_call_string(sample.target)
    normalized_target = format_call_string(target_calls)

    if target_calls:
        predicted_method_names = [call["name"] for call in prediction_calls]
        target_method_names = [call["name"] for call in target_calls]
        method_name_match: Optional[bool] = predicted_method_names == target_method_names
        argument_iou: Optional[float] = _compute_iou(
            _build_argument_entries(prediction_calls),
            _build_argument_entries(target_calls),
        )
    else:
        predicted_method_names = []
        target_method_names = []
        method_name_match = None
        argument_iou = None

    return {
        "exact_match": normalized_prediction == normalized_target,
        "normalized_prediction": normalized_prediction,
        "normalized_target": normalized_target,
        "predicted_method_names": predicted_method_names,
        "target_method_names": target_method_names,
        "method_name_match": method_name_match,
        "argument_iou": argument_iou,
    }
