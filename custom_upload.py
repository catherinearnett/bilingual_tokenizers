"""
Convert custom-mix SPM tokenizers → HF format and upload to HuggingFace.

Input:  custom_spm_tokenizers/{lang1}_{lang2}_{p1}_{p2}_{tok_type}_{whitespace}_{vocab_size}.model
Output: catherinearnett/bilingual_tokenizers2 as custom_mix/{stem}/tokenizer.json

Usage:
    export HF_TOKEN_WRITE_bilingual_tokenizers=hf_...
    python upload_custom_tokenizers.py
    python upload_custom_tokenizers.py --dry_run
    python upload_custom_tokenizers.py --workers 8 --batch_size 20
"""

import os
import sys
import glob
import json
import shutil
import argparse
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

from transformers import LlamaTokenizer, PreTrainedTokenizerFast
from transformers.convert_slow_tokenizer import SpmConverter
from tokenizers.normalizers import Replace as NormalizerReplace
from tokenizers.decoders import (
    ByteFallback,
    Metaspace,
    Sequence as DecoderSequence,
)
from huggingface_hub import HfApi, CommitOperationAdd

# ── Config ────────────────────────────────────────────────────────────────────
BI_REPO    = "catherinearnett/bilingual_tokenizers2"
SPM_DIR    = "custom_spm_tokenizers"
TMP_ROOT   = "custom_hf_tmp"
BATCH_SIZE = 20
HF_SUBDIR  = "custom_mix"   # uploaded as custom_mix/{stem}/tokenizer.json

BILINGUAL_PROPORTIONS = {str(i) for i in range(101)}  # any integer proportion


# ── Filename parsing ──────────────────────────────────────────────────────────

def parse_stem(stem):
    """
    Custom mix format:
      {lang1}_{lang2}_{p1}_{p2}_{tok_type}_{whitespace}_{vocab_size}
      e.g. afr_Latn_deu_Latn_8_92_bpe_nowhitespace_16384
      → 9 underscore-parts, parts[6] is tok_type
    """
    parts = stem.split("_")
    if len(parts) == 9 and parts[6] in ("bpe", "unigram"):
        return {
            "kind":       "bi",
            "lang1":      "_".join(parts[0:2]),
            "lang2":      "_".join(parts[2:4]),
            "p1":         parts[4],
            "p2":         parts[5],
            "tok_type":   parts[6],
            "whitespace": parts[7],
            "vocab_size": parts[8],
        }
    return None


# ── Conversion ────────────────────────────────────────────────────────────────

def _spm_bpe_to_hf(model_path):
    import re
    import sentencepiece as spm_lib
    from tokenizers import Tokenizer, AddedToken
    from tokenizers.models import BPE

    sp = spm_lib.SentencePieceProcessor()
    sp.Load(model_path)
    n = sp.get_piece_size()

    vocab = {sp.id_to_piece(i): i for i in range(n)}
    special_ids = {sp.id_to_piece(i) for i in range(n)
                   if sp.IsControl(i) or sp.IsUnknown(i) or sp.IsByte(i)}

    merges = []
    for i in range(n):
        piece = sp.id_to_piece(i)
        if piece in special_ids or len(piece) <= 1:
            continue
        best = None
        best_max_id = n + 1
        for pos in range(1, len(piece)):
            left, right = piece[:pos], piece[pos:]
            if left in vocab and right in vocab:
                max_id = max(vocab[left], vocab[right])
                if max_id < best_max_id:
                    best_max_id = max_id
                    best = (left, right)
        if best:
            merges.append(best)

    special_tokens = [
        AddedToken("<pad>", special=True),
        AddedToken("<unk>", special=True),
        AddedToken("<s>",   special=True),
        AddedToken("</s>",  special=True),
    ]
    tokenizer = Tokenizer(BPE(
        vocab=vocab,
        merges=merges,
        unk_token="<unk>",
        fuse_unk=True,
        byte_fallback=True,
    ))
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer


def convert_spm(model_path, tok_type, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    if tok_type == "bpe":
        tokenizer_obj = _spm_bpe_to_hf(model_path)
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_obj,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<s>",
            eos_token="</s>",
            add_bos_token=True,
            add_eos_token=False,
        )
    else:
        slow_tokenizer = LlamaTokenizer(vocab_file=model_path)
        converter = SpmConverter(slow_tokenizer)
        converted = converter.converted()
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=converted,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<s>",
            eos_token="</s>",
            add_bos_token=True,
            add_eos_token=False,
        )

    hf_tokenizer.backend_tokenizer.normalizer    = NormalizerReplace(" ", "▁")
    hf_tokenizer.backend_tokenizer.pre_tokenizer = None
    hf_tokenizer.backend_tokenizer.decoder = DecoderSequence([
        ByteFallback(),
        Metaspace(replacement="▁", prepend_scheme="never", split=False),
    ])

    hf_tokenizer.save_pretrained(out_dir)
    tok_json_path = os.path.join(out_dir, "tokenizer.json")
    with open(tok_json_path) as f:
        tok_data = json.load(f)
    tok_data["model"]["byte_fallback"] = True
    with open(tok_json_path, "w") as f:
        json.dump(tok_data, f, ensure_ascii=False, indent=2)

    return tok_json_path


def convert_one(args):
    stem, model_path, out_dir = args
    info = parse_stem(stem)
    if info is None:
        return stem, None, "unparseable filename"
    tok_json = os.path.join(out_dir, "tokenizer.json")
    if os.path.exists(tok_json):
        return stem, tok_json, None
    try:
        tok_json = convert_spm(model_path, info["tok_type"], out_dir)
        return stem, tok_json, None
    except Exception as e:
        return stem, None, f"{e}\n{traceback.format_exc()}"


# ── Upload ────────────────────────────────────────────────────────────────────

def get_uploaded_stems(api, repo_id, token, subdir):
    stems = set()
    try:
        for item in api.list_repo_tree(
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=subdir,
            recursive=False,
            token=token,
        ):
            if not hasattr(item, "rfilename") and hasattr(item, "path"):
                # strip the subdir prefix to get just the stem
                stems.add(item.path.replace(f"{subdir}/", ""))
    except Exception as e:
        print(f"  Warning fetching {repo_id}/{subdir}: {e}")
    return stems


def upload_batch(api, repo_id, token, batch, subdir):
    operations = [
        CommitOperationAdd(
            path_in_repo=f"{subdir}/{stem}/tokenizer.json",
            path_or_fileobj=tok_json,
        )
        for stem, tok_json in batch
        if tok_json and os.path.exists(tok_json)
    ]
    if not operations:
        return
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=f"Add {len(operations)} custom-mix HF tokenizers",
        operations=operations,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spm_dir",    default=SPM_DIR)
    parser.add_argument("--tmp_dir",    default=TMP_ROOT)
    parser.add_argument("--workers",    type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--dry_run",    action="store_true")
    parser.add_argument("--error_log",  default="custom_convert_upload_errors.txt")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN_WRITE_bilingual_tokenizers")
    if not token and not args.dry_run:
        sys.exit("ERROR: set HF_TOKEN_WRITE_bilingual_tokenizers env var")

    api = HfApi()

    model_files = sorted(glob.glob(os.path.join(args.spm_dir, "*.model")))
    print(f"Found {len(model_files)} .model files in {args.spm_dir}/\n")

    if not args.dry_run:
        print(f"Checking what's already uploaded under {HF_SUBDIR}/...")
        already_done = get_uploaded_stems(api, BI_REPO, token, HF_SUBDIR)
        print(f"  {len(already_done)} already uploaded\n")
    else:
        already_done = set()

    jobs = []
    for model_path in model_files:
        stem = os.path.splitext(os.path.basename(model_path))[0]
        info = parse_stem(stem)
        if info is None:
            print(f"  SKIP (unparseable): {stem}")
            continue
        if stem in already_done:
            print(f"  SKIP (already uploaded): {stem}")
            continue
        out_dir = os.path.join(args.tmp_dir, stem)
        jobs.append((stem, model_path, out_dir))

    print(f"\nJobs to convert + upload: {len(jobs)}\n")
    if not jobs:
        print("Nothing to do!")
        return

    os.makedirs(args.tmp_dir, exist_ok=True)
    total   = len(jobs)
    pending = []
    ok = err = 0

    with open(args.error_log, "w") as error_log:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(convert_one, job): job for job in jobs}
            for i, future in enumerate(as_completed(futures), 1):
                stem, tok_json, error = future.result()

                if error:
                    print(f"  [ERROR] ({i}/{total}) {stem}: {error.splitlines()[0]}")
                    error_log.write(f"\n{'='*60}\n{stem}\n{error}\n")
                    err += 1
                    continue

                ok += 1
                pending.append((stem, tok_json))
                print(f"  [OK] ({i}/{total}) {stem}")

                if not args.dry_run and len(pending) >= args.batch_size:
                    print(f"  → Uploading batch of {len(pending)}...")
                    try:
                        upload_batch(api, BI_REPO, token, pending, HF_SUBDIR)
                        for s, _ in pending:
                            shutil.rmtree(os.path.join(args.tmp_dir, s), ignore_errors=True)
                        pending.clear()
                        print(f"    ✓ Uploaded")
                    except Exception as e:
                        print(f"    ✗ Upload failed: {e}")
                        error_log.write(f"\nUPLOAD ERROR: {e}\n")

        if not args.dry_run and pending:
            print(f"  → Uploading final batch of {len(pending)}...")
            try:
                upload_batch(api, BI_REPO, token, pending, HF_SUBDIR)
                for s, _ in pending:
                    shutil.rmtree(os.path.join(args.tmp_dir, s), ignore_errors=True)
                print(f"    ✓ Uploaded")
            except Exception as e:
                print(f"    ✗ Final upload failed: {e}")
                error_log.write(f"\nFINAL UPLOAD ERROR: {e}\n")

    print(f"\nDone. Converted: {ok}  Errors: {err}")
    print(f"Errors logged to {args.error_log}")


if __name__ == "__main__":
    main()
