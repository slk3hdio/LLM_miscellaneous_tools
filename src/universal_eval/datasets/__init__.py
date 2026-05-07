from .adapter import DatasetAdapter, EvalSample
from .scoring import score_prediction
from .toolace_adapter import ToolACEDatasetAdapter
from .apibank_adapter import APIBankDatasetAdapter


def create_dataset_adapter(name: str) -> DatasetAdapter:
    if name == 'toolace':
        return ToolACEDatasetAdapter()
    if name == 'apibank':
        return APIBankDatasetAdapter()
    raise ValueError(f"Unsupported dataset type: {name}")

__all__ = [
    "EvalSample",
    "score_prediction",
    "ToolACEDatasetAdapter",
    "APIBankDatasetAdapter",
    "create_dataset_adapter",
]
