from typing import Dict, Any
from .adapter import DatasetAdapter, EvalSample
from ..evaluator.scoring import score_prediction
from .toolace_adapter import ToolACEDatasetAdapter
from .apibank_adapter import APIBankDatasetAdapter


def create_dataset_adapter(dataset_config:Dict[str, Any]) -> DatasetAdapter:
    name = dataset_config['active']
    active_config = dataset_config[name]
    if name == 'toolace':
        return ToolACEDatasetAdapter(active_config['path'], active_config['split'])
    if name == 'apibank':
        return APIBankDatasetAdapter(active_config['path'], active_config['split'])
    raise ValueError(f"Unsupported dataset type: {name}")

__all__ = [
    "EvalSample",
    "score_prediction",
    "ToolACEDatasetAdapter",
    "APIBankDatasetAdapter",
    "create_dataset_adapter",
]
