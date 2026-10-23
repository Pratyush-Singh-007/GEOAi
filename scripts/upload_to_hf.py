"""
Script to create and push the GeoAI Outdoor Assistant model repository
and model card to Hugging Face Hub.

Usage:
    export HF_TOKEN="your_huggingface_write_token"
    python scripts/upload_to_hf.py --repo-name "geoai-outdoor-assistant-qwen3b"
"""

import os
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Upload Model Card to Hugging Face Hub")
    parser.add_argument(
        "--repo-name",
        type=str,
        default="geoai-outdoor-assistant-3b",
        help="Repository name on Hugging Face Hub (e.g. your-username/geoai-outdoor-assistant-3b)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face API token (with write permissions)",
    )
    args = parser.parse_args()

    if not args.token:
        print("❌ Error: HF_TOKEN is required. Either pass --token or set export HF_TOKEN='...'")
        print("   Get your free Hugging Face token at https://huggingface.co/settings/tokens")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Installing huggingface_hub...")
        os.system("pip install huggingface_hub")
        from huggingface_hub import HfApi, create_repo

    api = HfApi(token=args.token)

    # 1. Get current username
    user = api.whoami()["name"]
    full_repo_id = f"{user}/{args.repo_name}" if "/" not in args.repo_name else args.repo_name

    print(f"🚀 Creating / accessing public Hugging Face repository: {full_repo_id} ...")
    create_repo(repo_id=full_repo_id, repo_type="model", token=args.token, exist_ok=True, private=False)

    # 2. Upload MODEL_CARD.md as README.md
    base_dir = Path(__file__).resolve().parent.parent
    model_card_path = base_dir / "model" / "MODEL_CARD.md"

    if model_card_path.exists():
        print("📄 Uploading model card README.md ...")
        api.upload_file(
            path_or_fileobj=str(model_card_path),
            path_in_repo="README.md",
            repo_id=full_repo_id,
            repo_type="model",
            commit_message="Add GeoAI Outdoor Assistant Model Card & System Configuration",
        )
    else:
        print(f"⚠️ Warning: {model_card_path} not found.")

    # 3. Upload model configuration file
    config_py_path = base_dir / "model" / "slm.py"
    if config_py_path.exists():
        print("⚙️ Uploading model configuration ...")
        api.upload_file(
            path_or_fileobj=str(config_py_path),
            path_in_repo="slm_config.py",
            repo_id=full_repo_id,
            repo_type="model",
            commit_message="Add SLM inference config and system prompts",
        )

    print(f"\n✅ Successfully published model repo to Hugging Face Hub!")
    print(f"🔗 Public URL: https://huggingface.co/{full_repo_id}")
    print(f"Use this link for Deliverable #2 in your hackathon submission.")


if __name__ == "__main__":
    main()
