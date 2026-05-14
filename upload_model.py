#!/usr/bin/env python3
"""
Upload a bilingual tokenizer model and its checkpoints to HuggingFace Hub.

Usage:
    python upload_to_hf.py --model afr_Latn_als_Latn_50_50_bpe_nowhitespace_16384
    python upload_to_hf.py --model afr_Latn_als_Latn_50_50_bpe_nowhitespace_16384 --base-dir /mnt/ssd-3/catherine/bilingual_tokenizers/matched_compression/custom_models
    python upload_to_hf.py --model afr_Latn_als_Latn_50_50_bpe_nowhitespace_16384 --org my-org
    python upload_to_hf.py --model afr_Latn_als_Latn_50_50_bpe_nowhitespace_16384 --skip-checkpoints
"""

import argparse
import os
import sys
from pathlib import Path

# Files to upload as the "main" revision (non-checkpoint model files)
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


def get_checkpoints(model_dir: Path) -> list[Path]:
    """Return sorted checkpoint directories."""
    checkpoints = sorted(
        [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[1]),
    )
    return checkpoints


def create_or_get_repo(api, repo_id: str, private: bool = False):
    from huggingface_hub import create_repo
    try:
        url = create_repo(repo_id=repo_id, token=get_token(), private=private, exist_ok=True)
        print(f"Repository ready: {url}")
    except Exception as e:
        print(f"ERROR creating/accessing repo: {e}")
        sys.exit(1)


def upload_main(api, model_dir: Path, repo_id: str):
    """Upload main model files to the 'main' branch."""
    print(f"\n{'='*60}")
    print(f"Uploading main model files → branch: main")
    print(f"{'='*60}")

    files_to_upload = []
    for fname in MAIN_MODEL_FILES:
        fpath = model_dir / fname
        if fpath.exists():
            files_to_upload.append(fpath)
        else:
            print(f"  [SKIP] {fname} not found")

    if not files_to_upload:
        print("  No main model files found, skipping.")
        return

    for fpath in files_to_upload:
        print(f"  Uploading {fpath.name} ...")
        api.upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=fpath.name,
            repo_id=repo_id,
            revision="main",
            token=get_token(),
            commit_message=f"Add {fpath.name}",
        )
    print("  ✓ Main model files uploaded.")


def upload_checkpoint(api, checkpoint_dir: Path, repo_id: str):
    """Upload a single checkpoint directory as a revision named after the checkpoint."""
    revision = checkpoint_dir.name  # e.g. "checkpoint-27156"
    print(f"\n{'='*60}")
    print(f"Uploading checkpoint: {revision}")
    print(f"{'='*60}")

    files = list(checkpoint_dir.iterdir())
    if not files:
        print(f"  [SKIP] {revision} is empty.")
        return

    # Create the branch/revision if it doesn't exist
    try:
        api.create_branch(
            repo_id=repo_id,
            branch=revision,
            token=get_token(),
            exist_ok=True,
        )
    except Exception as e:
        print(f"  Warning: could not create branch '{revision}': {e}")

    for fpath in sorted(files):
        if fpath.is_file():
            print(f"  Uploading {fpath.name} ...")
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=fpath.name,
                repo_id=repo_id,
                revision=revision,
                token=get_token(),
                commit_message=f"Add {fpath.name} for {revision}",
            )

    print(f"  ✓ {revision} uploaded.")


def main():
    parser = argparse.ArgumentParser(description="Upload a model + checkpoints to HuggingFace Hub.")
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Model directory name, e.g. afr_Latn_als_Latn_50_50_bpe_nowhitespace_16384",
    )
    parser.add_argument(
        "--base-dir", "-d",
        default="/mnt/ssd-3/catherine/bilingual_tokenizers/matched_compression/custom_models",
        help="Base directory containing model folders (default: %(default)s)",
    )
    parser.add_argument(
        "--org", "-o",
        default=None,
        help="HuggingFace org or username to upload under. If omitted, uploads to your personal account.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Make the HuggingFace repo private.",
    )
    parser.add_argument(
        "--skip-main",
        action="store_true",
        default=False,
        help="Skip uploading the main model files.",
    )
    parser.add_argument(
        "--skip-checkpoints",
        action="store_true",
        default=False,
        help="Skip uploading checkpoints.",
    )
    parser.add_argument(
        "--checkpoint", "-c",
        default=None,
        help="Upload only a specific checkpoint, e.g. checkpoint-27156. Implies --skip-main.",
    )
    args = parser.parse_args()

    # Resolve paths
    model_dir = Path(args.base_dir) / args.model
    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}")
        sys.exit(1)

    # Build repo_id
    repo_id = f"{args.org}/{args.model}" if args.org else args.model

    print(f"Model dir : {model_dir}")
    print(f"Repo ID   : {repo_id}")
    print(f"Private   : {args.private}")

    # Import here so missing package gives a clear error
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub is not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi()

    # Create/verify repo
    create_or_get_repo(api, repo_id, private=args.private)

    # -- Single checkpoint mode --
    if args.checkpoint:
        cp_dir = model_dir / args.checkpoint
        if not cp_dir.exists():
            print(f"ERROR: Checkpoint directory not found: {cp_dir}")
            sys.exit(1)
        upload_checkpoint(api, cp_dir, repo_id)
        print("\nDone.")
        return

    # -- Full upload mode --
    if not args.skip_main:
        upload_main(api, model_dir, repo_id)

    if not args.skip_checkpoints:
        checkpoints = get_checkpoints(model_dir)
        if not checkpoints:
            print("\nNo checkpoints found.")
        else:
            print(f"\nFound {len(checkpoints)} checkpoints: {[c.name for c in checkpoints]}")
            for cp in checkpoints:
                upload_checkpoint(api, cp, repo_id)

    print("\nAll done!")


if __name__ == "__main__":
    main()
