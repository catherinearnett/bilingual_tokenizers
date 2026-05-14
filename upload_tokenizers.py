#!/usr/bin/env python3
"""
Upload a bilingual tokenizer model and its checkpoints to HuggingFace Hub.

Usage:
    python upload_to_hf.py --model afr_Latn_als_Latn_50_50_bpe_nowhitespace_16384
    python upload_to_hf.py --model afr_Latn_als_Latn_50_50_bpe_nowhitespace_16384 --skip-checkpoints
    python upload_to_hf.py --model afr_Latn_als_Latn_50_50_bpe_nowhitespace_16384 --checkpoint checkpoint-27156
    python upload_to_hf.py --model afr_Latn_als_Latn_50_50_bpe_nowhitespace_16384 --skip-main
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

HF_USERNAME = "catherinearnett"
BASE_DIR = "/mnt/ssd-3/catherine/bilingual_tokenizers/matched_compression/custom_models"

MAIN_MODEL_FILES = [
    "all_results.json",
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "train_results.json",
    "trainer_state.json",
    "training_args.bin",
]


def get_token() -> str:
    token = os.environ.get("HF_TOKEN_WRITE_bilingual_tokenizers")
    if not token:
        print("ERROR: Environment variable HF_TOKEN_WRITE_bilingual_tokenizers is not set.")
        sys.exit(1)
    return token


def get_checkpoints(model_dir: Path) -> list:
    return sorted(
        [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[1]),
    )


def ensure_repo(repo_id: str, private: bool):
    """Create the repo if it doesn't exist yet."""
    from huggingface_hub import create_repo
    try:
        url = create_repo(repo_id=repo_id, token=get_token(), private=private, exist_ok=True)
        print(f"Repository ready: {url}")
    except Exception as e:
        print(f"ERROR creating repo '{repo_id}': {e}")
        sys.exit(1)


def upload_main(api, model_dir: Path, repo_id: str):
    print(f"\n{'='*60}")
    print("Uploading main model files → branch: main")
    print(f"{'='*60}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        copied = 0
        for fname in MAIN_MODEL_FILES:
            src = model_dir / fname
            if src.exists():
                shutil.copy2(src, tmp / fname)
                print(f"  Queued: {fname}")
                copied += 1
            else:
                print(f"  [SKIP] {fname} not found")

        if copied == 0:
            print("  No main model files found, skipping.")
            return

        api.upload_folder(
            folder_path=str(tmp),
            repo_id=repo_id,
            revision="main",
            token=get_token(),
            commit_message="Add main model files",
            repo_type="model"
        )
    print("  ✓ Main model files uploaded.")


def upload_checkpoint(api, checkpoint_dir: Path, repo_id: str):
    branch = checkpoint_dir.name
    print(f"\n{'='*60}")
    print(f"Uploading checkpoint: {branch}")
    print(f"{'='*60}")

    files = [f for f in checkpoint_dir.iterdir() if f.is_file()]
    if not files:
        print(f"  [SKIP] {branch} has no files.")
        return

    # Create branch off main
    try:
        api.create_branch(repo_id=repo_id, branch=branch, token=get_token(), exist_ok=True)
    except Exception as e:
        print(f"  Warning: could not create branch '{branch}': {e}")

    api.upload_folder(
        folder_path=str(checkpoint_dir),
        repo_id=repo_id,
        revision=branch,
        token=get_token(),
        commit_message=f"Add {branch}",
        repo_type="model"
    )
    print(f"  ✓ {branch} uploaded.")


def main():
    parser = argparse.ArgumentParser(
        description="Upload a bilingual tokenizer model + checkpoints to HuggingFace Hub."
    )
    parser.add_argument("--model", "-m", required=True,
        help="Model directory name, e.g. afr_Latn_als_Latn_50_50_bpe_nowhitespace_16384")
    parser.add_argument("--base-dir", "-d", default=BASE_DIR,
        help=f"Base directory (default: {BASE_DIR})")
    parser.add_argument("--username", "-u", default=HF_USERNAME,
        help=f"HuggingFace username or org (default: {HF_USERNAME})")
    parser.add_argument("--private", action="store_true", default=False,
        help="Make the repo private")
    parser.add_argument("--skip-main", action="store_true", default=False,
        help="Skip uploading main model files")
    parser.add_argument("--skip-checkpoints", action="store_true", default=False,
        help="Skip uploading checkpoints")
    parser.add_argument("--checkpoint", "-c", default=None,
        help="Upload only one specific checkpoint, e.g. checkpoint-27156")
    args = parser.parse_args()

    model_dir = Path(args.base_dir) / args.model
    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}")
        sys.exit(1)

    repo_id = f"{args.username}/{args.model}"

    print(f"Model dir : {model_dir}")
    print(f"Repo ID   : {repo_id}")
    print(f"Private   : {args.private}")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi()

    # Always create repo first — this is what was missing before
    ensure_repo(repo_id, private=args.private)

    # Single checkpoint mode
    if args.checkpoint:
        cp_dir = model_dir / args.checkpoint
        if not cp_dir.exists():
            print(f"ERROR: Checkpoint not found: {cp_dir}")
            sys.exit(1)
        upload_checkpoint(api, cp_dir, repo_id)
        print("\nDone.")
        return

    # Full upload
    if not args.skip_main:
        upload_main(api, model_dir, repo_id)

    if not args.skip_checkpoints:
        checkpoints = get_checkpoints(model_dir)
        if not checkpoints:
            print("\nNo checkpoints found.")
        else:
            print(f"\nFound {len(checkpoints)} checkpoints.")
            for cp in checkpoints:
                upload_checkpoint(api, cp, repo_id)

    print("\nAll done!")


if __name__ == "__main__":
    main()
