"""API-Bank evaluation error pattern analysis.

Usage:
  uv run python src/universal_eval/analysis/error_analysis.py
  uv run python src/universal_eval/analysis/error_analysis.py --output-dir outputs/evaluation
"""

from __future__ import annotations

import json
import re
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
import tqdm


# ---------------------------------------------------------------------------
# Error categories
# ---------------------------------------------------------------------------

class ErrorCategory:
    CORRECT = "correct"
    PARAM_HALLUCINATION = "param_hallucination"   # A: right function, wrong param values
    TOOLSEARCHER_PROXY = "toolsearcher_proxy"      # B: ToolSearcher proxy
    NATURAL_LANGUAGE = "natural_language"           # C: natural language output
    FORMAT_ERROR = "format_error"                   # D: quoting / type formatting
    EMPTY_PREDICTION = "empty_prediction"           # E: empty prediction
    EXTRA_MISSING_PARAMS = "extra_missing_params"   # F: extra or missing param names
    WRONG_FUNCTION = "wrong_function"               # wrong function name (not ToolSearcher)
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class RunMeta:
    model: str
    provider: str
    conversation_style: str
    tool_format: str
    level: int
    sample_limit: int
    run_timestamp: str
    output_dir: Path


@dataclass
class ErrorStats:
    meta: RunMeta
    total: int = 0
    correct: int = 0
    categories: Counter = field(default_factory=Counter)
    examples: dict[str, list[dict[str, str]]] = field(
        default_factory=lambda: defaultdict(list)
    )


# ---------------------------------------------------------------------------
# Classification logic — uses score dict fields directly
# ---------------------------------------------------------------------------

_FUNC_RE = re.compile(r"(\w+)\([^)]*\)")


def _extract_func_name(expr: str) -> str | None:
    m = _FUNC_RE.search(expr)
    return m.group(1) if m else None


def _is_natural_language(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    if "[" in text or text.startswith("{"):
        return False
    return not re.search(r"\w+\s*\(.*\)", text)


def classify_error(score: dict[str, Any]) -> str:
    """Classify a single evaluation record."""
    if score["exact_match"]:
        return ErrorCategory.CORRECT

    pred = score["normalized_prediction"]
    target = score["normalized_target"]
    iou = score.get("argument_iou")

    # E: empty prediction
    if not pred.strip() or pred.strip() == "[]":
        return ErrorCategory.EMPTY_PREDICTION

    # C: natural language output
    if _is_natural_language(pred):
        return ErrorCategory.NATURAL_LANGUAGE

    target_func = _extract_func_name(target)
    pred_func = _extract_func_name(pred)

    if pred_func is None:
        return ErrorCategory.NATURAL_LANGUAGE

    # B: ToolSearcher proxy pattern
    if target_func != "ToolSearcher" and pred_func == "ToolSearcher":
        return ErrorCategory.TOOLSEARCHER_PROXY

    # Wrong function name
    if pred_func != target_func:
        return ErrorCategory.WRONG_FUNCTION

    # Function name matches — check argument issues
    method_names_match = score.get("method_name_match")

    if method_names_match:
        if iou is not None and iou >= 1.0:
            # Args are semantically identical but serialization differs
            return ErrorCategory.FORMAT_ERROR

        # Check for extra/missing param names vs just wrong values
        pred_param_names = set(score.get("predicted_method_names", []))
        target_param_names = set(score.get("target_method_names", []))
        # Extract actual arg keys
        t_keys = set(_extract_arg_keys(target))
        p_keys = set(_extract_arg_keys(pred))
        extra = p_keys - t_keys
        missing = t_keys - p_keys

        if extra or missing:
            return ErrorCategory.EXTRA_MISSING_PARAMS

        return ErrorCategory.PARAM_HALLUCINATION

    return ErrorCategory.UNKNOWN


_ARG_KEY_RE = re.compile(r"(\w+)\s*=")


def _extract_arg_keys(expr: str) -> list[str]:
    """Extract parameter names from a normalized call string."""
    inner = expr.strip("[]")
    return [m.group(1) for m in _ARG_KEY_RE.finditer(inner)]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_meta(output_dir: Path) -> RunMeta:
    config_path = output_dir / "config.yaml"
    cfg: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    provider_cfg = cfg.get("provider", {})
    provider_key = provider_cfg.get("active", "local")
    if provider_key == "local":
        local = provider_cfg.get("local", {})
        model_name = Path(local.get("model_path", "")).name
    else:
        model_name = provider_cfg.get("openai", {}).get("model", provider_key)

    dataset_cfg = cfg.get("dataset", {}).get("apibank", {})
    level = dataset_cfg.get("split", {}).get("level", 1)

    return RunMeta(
        model=model_name,
        provider=provider_key,
        conversation_style=cfg.get("conversation_style", "single"),
        tool_format=cfg.get("tool_format", "plain"),
        level=int(level),
        sample_limit=cfg.get("sample_limit", 0),
        run_timestamp=output_dir.name.split("-")[0],
        output_dir=output_dir,
    )


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def discover_runs(base_dir: Path) -> list[tuple[RunMeta, list[dict[str, Any]]]]:
    runs: list[tuple[RunMeta, list[dict[str, Any]]]] = []
    for record_path in sorted(base_dir.glob("**/apibank/*/records.jsonl")):
        output_dir = record_path.parent
        meta = load_meta(output_dir)
        records = load_records(record_path)
        runs.append((meta, records))
    return runs


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_run(meta: RunMeta, records: list[dict[str, Any]]) -> ErrorStats:
    stats = ErrorStats(meta=meta, total=len(records))
    for r in records:
        cat = classify_error(r["score"])
        stats.categories[cat] += 1
        if cat == ErrorCategory.CORRECT:
            stats.correct += 1
        elif len(stats.examples[cat]) < 3:
            stats.examples[cat].append({
                "sample_id": r["sample"]["sample_id"],
                "target": r["score"]["normalized_target"][:130],
                "prediction": r["score"]["normalized_prediction"][:130],
            })
    return stats


CATEGORY_LABELS: dict[str, str] = {
    ErrorCategory.CORRECT:              "Correct",
    ErrorCategory.PARAM_HALLUCINATION:  "A-Param hallucination",
    ErrorCategory.TOOLSEARCHER_PROXY:   "B-ToolSearcher proxy",
    ErrorCategory.WRONG_FUNCTION:       "Wrong function",
    ErrorCategory.NATURAL_LANGUAGE:     "C-Natural language",
    ErrorCategory.FORMAT_ERROR:         "D-Format error",
    ErrorCategory.EMPTY_PREDICTION:     "E-Empty prediction",
    ErrorCategory.EXTRA_MISSING_PARAMS: "F-Extra/missing params",
    ErrorCategory.UNKNOWN:              "Unknown",
}

CAT_ORDER = [
    ErrorCategory.CORRECT,
    ErrorCategory.PARAM_HALLUCINATION,
    ErrorCategory.TOOLSEARCHER_PROXY,
    ErrorCategory.WRONG_FUNCTION,
    ErrorCategory.NATURAL_LANGUAGE,
    ErrorCategory.FORMAT_ERROR,
    ErrorCategory.EMPTY_PREDICTION,
    ErrorCategory.EXTRA_MISSING_PARAMS,
    ErrorCategory.UNKNOWN,
]


def build_label(meta: RunMeta) -> str:
    return f"{meta.model} ({meta.conversation_style}, {meta.tool_format}, L{meta.level})"


def _pad_row(cols: list[str], widths: list[int]) -> str:
    padded = [c.ljust(w) for c, w in zip(cols, widths)]
    return " | ".join(padded)


def print_table(stats_list: list[ErrorStats]) -> None:
    meaningful = [s for s in stats_list if s.total >= 10]

    error_cols = [c for c in CAT_ORDER if c != ErrorCategory.CORRECT]
    header = ["Model (style, tool, level)", "N", "Acc"]
    for c in error_cols:
        header.append(CATEGORY_LABELS.get(c, c))

    # Build rows
    rows: list[list[str]] = []
    for s in sorted(meaningful, key=lambda x: (-x.correct / max(x.total, 1), x.meta.model)):
        label = build_label(s.meta)
        acc = f"{s.correct / s.total:.1%}" if s.total else "-"
        row = [label, str(s.total), acc]
        for c in error_cols:
            cnt = s.categories.get(c, 0)
            row.append(f"{cnt} ({cnt / max(s.total, 1):.1%})")
        rows.append(row)

    # Compute column widths
    col_widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]

    # Print header
    print("\n## API-Bank Error Pattern Analysis\n")

    print("\n=== Summary Table ===\n")
    print(_pad_row(header, col_widths))
    print(_pad_row(["-" * w for w in col_widths], col_widths))
    for row in rows:
        print(_pad_row(row, col_widths))


def print_examples(stats_list: list[ErrorStats]) -> None:
    all_examples: dict[str, list[dict[str, str]]] = defaultdict(list)
    for s in stats_list:
        for cat, examples in s.examples.items():
            all_examples[cat].extend(examples)

    print("\n## Typical Error Examples\n")
    for cat in CAT_ORDER:
        if cat == ErrorCategory.CORRECT or cat not in all_examples:
            continue
        examples = all_examples[cat][:3]
        if not examples:
            continue
        print(f"### {CATEGORY_LABELS.get(cat, cat)}\n")
        for i, ex in enumerate(examples, 1):
            print(f"**Ex {i}** `{ex['sample_id']}`")
            print(f"  target: `{ex['target']}`")
            print(f"  pred:   `{ex['prediction']}`")
            print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze API-Bank evaluation errors")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/evaluation"),
        help="Evaluation outputs directory",
    )
    args = parser.parse_args()

    base = Path(args.output_dir)
    if not base.exists():
        print(f"Directory not found: {base}")
        return

    runs = discover_runs(base)
    if not runs:
        print(f"No runs found under {base}")
        return

    print(f"Found {len(runs)} runs")

    stats_list: list[ErrorStats] = []
    for meta, records in tqdm.tqdm(runs, desc="Analyzing"):
        stats = analyze_run(meta, records)
        stats_list.append(stats)

    for s in sorted(stats_list, key=lambda x: -x.correct):
        label = build_label(s.meta)
        acc = s.correct / s.total if s.total else 0
        top = s.categories.most_common(3)
        top_str = ", ".join(
            f"{CATEGORY_LABELS.get(c, c)}:{n}" for c, n in top if c != ErrorCategory.CORRECT
        )
        print(f"  {label}: {s.correct}/{s.total} ({acc:.1%})  top: {top_str}")

    print_table(stats_list)
    print_examples(stats_list)


if __name__ == "__main__":
    main()
