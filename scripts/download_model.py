import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except Exception as e:
    print("[ERROR] huggingface_hub is not installed:", e)
    print("[INFO] Install it with: pip install huggingface_hub")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Hugging Face model snapshot.")
    parser.add_argument("--model-id", default="openbmb/MiniCPM-V-4.6")
    parser.add_argument("--target-dir", type=Path, default=ROOT / "models" / "minicpm_v_4_6")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    # endpoint = os.environ.get("HF_ENDPOINT")
    endpoint = "https://huggingface.co"
    print(f"[INFO] HF_TOKEN: {token}")
    print(f"[INFO] HF_ENDPOINT: {endpoint}")

    print(f"[INFO] Downloading {args.model_id} to {args.target_dir}")
    snapshot_download(
        repo_id=args.model_id,
        local_dir=args.target_dir,
        token=token,
        # local_dir_use_symlinks=False,
        # resume_download=True,
        endpoint=endpoint,
    )
    print("[INFO] Download complete")


if __name__ == "__main__":
    main()
