from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import yaml

from .datasets import create_dataset_adapter
from .evaluator import EvalRecord, evaluate_dataset
from .providers import create_provider

ROOT = Path(__file__).parent.parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "eval.yaml"


def _setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level={
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
        }[log_level],
        format="[%(asctime)s]%(name)s %(levelname)s: %(message)s",
    )
    for noisy in ("openai._base_client", "httpcore", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _build_output_dir(config: dict[str, Any]) -> Path:
    active = config["provider"]["active"]
    if active == "local":
        model_name = config["provider"]["local"]["model_path"].split("/")[-1]
    else:
        model_name = config["provider"]["openai"]["model"]

    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    output_dir = ROOT / "outputs" / "evaluation" / model_name / config["dataset"]["active"] / ts
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("config.yaml").write_text(
        yaml.dump(config, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output_dir


def run(config_path: Path | None = None) -> dict[str, Any]:
    """Load config, prepare data, run evaluation, save results.

    Returns the summary dict.
    """
    path = config_path or DEFAULT_CONFIG
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    _setup_logging(config["log_level"])
    logger = logging.getLogger(__name__)

    provider_cfg = config["provider"]
    dataset_cfg = config["dataset"]

    # --- dataset ---
    adapter = create_dataset_adapter(dataset_cfg["active"])
    # use_tools: 将工具信息从提示词中去除
    use_tools = provider_cfg["active"] in ("openai", "vllm") and dataset_cfg["conversation_style"] == 'multi'
    samples = adapter.load_samples(
        dataset_config=dataset_cfg[dataset_cfg["active"]],
        limit=dataset_cfg["limit"],
        conversation_style=dataset_cfg["conversation_style"],
        with_raw_data=dataset_cfg["with_raw_data"],
        random_samples=dataset_cfg["random_samples"],
        strip_tool_descriptions=use_tools,
    )
    if not samples:
        raise ValueError(
            f"No samples loaded from {dataset_cfg[dataset_cfg['active']]['path']}"
        )
    logger.info("Loaded %d samples", len(samples))

    # --- provider ---
    provider = create_provider(
        provider_cfg,
        max_new_tokens=provider_cfg["max_new_token"],
        temperature=provider_cfg["temperature"],
    )

    # --- evaluate ---
    output_dir = _build_output_dir(config)
    summary, records = evaluate_dataset(provider, adapter, samples, output_dir=output_dir)

    logger.info(json.dumps({"summary": summary, "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))

    records_path = output_dir / "records.jsonl"
    EvalRecord.save(records, records_path)
    logger.info("Records saved to %s", records_path)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Summary saved to %s", summary_path)

    return summary
