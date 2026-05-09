from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any,Literal

import yaml

from .datasets import create_dataset_adapter
from .evaluator.evaluator import EvalRecord, evaluate_dataset
from .providers import create_provider

ROOT = Path(__file__).parent.parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "eval.yaml"
CONVERSATION_STYLES = {"single", "multi"}
TOOL_FORMATS = {"plain", "standard"}


_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")


def _resolve_env_vars(obj: Any) -> Any:
    """递归解析配置中的 ``${VAR_NAME}`` 环境变量占位符。"""
    if isinstance(obj, str):
        def _replace(m: re.Match[str]) -> str:
            return os.environ.get(m.group(1), m.group(0))
        return _ENV_VAR_RE.sub(_replace, obj)
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_vars(v) for v in obj]
    return obj


def _setup_logging(log_level: str, output_dir: Path) -> None:
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
    # 将日志也写入输出目录
    file_handler = logging.FileHandler(
        output_dir / "run.log", encoding="utf-8"
    )
    file_handler.setFormatter(logging.getLogger().handlers[0].formatter)
    logging.getLogger().addHandler(file_handler)


def _build_output_dir(config: dict[str, Any]) -> Path:
    active = config["provider"]["active"]
    if active in {"local", "vllm"}:
        model_name = config["provider"]["local"]["model_path"].split("/")[-1]
    else:
        model_name = config["provider"]["openai"]["model"]

    marker = f"{config['conversation_style']}_conv-{config['tool_format']}_tool-{config['sample_limit']}"
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    dir_name = ts + '-' + marker
    output_dir = ROOT / "outputs" / "evaluation" / model_name / config["dataset"]["active"] / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    # output_dir.joinpath("config.yaml").write_text(
    #     yaml.dump(config, default_flow_style=False, allow_unicode=True),
    #     encoding="utf-8",
    # )
    return output_dir


def _resolve_runtime_options(
    config: dict[str, Any],
    provider: Any,
    logger: logging.Logger,
) -> tuple[Literal['single', 'multi'], str]:
    """根据 provider 能力解析并降级 conversation_style 和 tool_format。"""
    conversation_style = config.get("conversation_style", "single")
    tool_format = config.get("tool_format", "plain")

    if conversation_style not in CONVERSATION_STYLES:
        raise ValueError(
            f"Unsupported conversation_style: {conversation_style!r}. "
            f"Expected one of {sorted(CONVERSATION_STYLES)}."
        )
    if tool_format not in TOOL_FORMATS:
        raise ValueError(
            f"Unsupported tool_format: {tool_format!r}. "
            f"Expected one of {sorted(TOOL_FORMATS)}."
        )

    if conversation_style == "multi" and not provider.supports_conversation_format():
        logger.warning(
            "Provider %s does not support multi-turn conversation format; "
            "falling back to conversation_style=single and tool_format=plain.",
            provider.__class__.__name__,
        )
        conversation_style = "single"
        tool_format = "plain"

    if tool_format == "standard" and conversation_style != "multi":
        logger.warning(
            "tool_format=standard requires conversation_style=multi; "
            "falling back to tool_format=plain."
        )
        tool_format = "plain"

    if tool_format == "standard" and not provider.supports_tool_calling():
        logger.warning(
            "Provider %s does not support standard tool calling; "
            "falling back to tool_format=plain.",
            provider.__class__.__name__,
        )
        tool_format = "plain"

    config["conversation_style"] = conversation_style
    config["tool_format"] = tool_format
    return conversation_style, tool_format


def run(config_path: Path | None = None) -> dict[str, Any]:
    """主入口：加载配置 → 准备数据 → 运行评测 → 保存结果。

    Returns:
        评测汇总字典，包含 total / exact_match_count / exact_match_rate。
    """
    path = config_path or DEFAULT_CONFIG
    # --- config ---
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = _resolve_env_vars(config)

    # --- output dir ---
    output_dir = _build_output_dir(config)
    _setup_logging(config["log_level"], output_dir)
    logger = logging.getLogger(__name__)

    # --- provider ---
    logger.info("="*50)
    logger.info(f"Loading provider {config['provider']['active']}...")
    provider = create_provider(config["provider"])
    conversation_style, tool_format = _resolve_runtime_options(config, provider, logger)

    # --- dataset ---
    logger.info("="*50)
    logger.info("Loading samples...")
    dataset_cfg = config["dataset"]
    adapter = create_dataset_adapter(dataset_cfg)
    samples = adapter.load_samples(
        limit=dataset_cfg.get("limit", config.get("sample_limit")),
        conversation_style=conversation_style,
        with_raw_data=dataset_cfg["with_raw_data"],
        random_samples=dataset_cfg["random_samples"],
        strip_tool_descriptions=tool_format == "standard",
    )
    if not samples:
        raise ValueError(
            f"No samples loaded from {adapter.name}"
        )
    logger.info("Loaded %d samples", len(samples))

    # --- evaluate ---
    logger.info("="*50)
    logger.info("Evaluating...")
    summary, records = evaluate_dataset(
        provider,
        samples,
        conversation_style=conversation_style,
        use_standard_tool_format=tool_format == "standard",
        batch_size=config.get("batch_size", 1),
    )
    logger.info(json.dumps({"summary": summary, "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))

    # --- save records ---
    logger.info("="*50)
    logger.info("Saving records...")
    records_path = output_dir / "records.jsonl"
    EvalRecord.save(records, records_path)
    logger.info("Records saved to %s", records_path)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Summary saved to %s", summary_path)

    if config.get('openai', {}).get('api_key', None):
        config['openai']['api_key'] = "<hidden>"
    output_dir.joinpath("config.yaml").write_text(
        yaml.dump(config, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )

    return summary
