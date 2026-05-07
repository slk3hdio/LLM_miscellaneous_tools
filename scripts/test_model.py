import argparse
import sys
from pathlib import Path

SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from universal_eval.runner import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run universal evaluation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to a YAML configuration file.",
    )
    args = parser.parse_args()

    run(config_path=args.config)


if __name__ == "__main__":
    main()
