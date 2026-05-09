"""
Convert SPM tokenizers → HF format and upload to HuggingFace.

Conversion logic (matched to tok_type + whitespace from filename):
  bpe    + nowhitespace → SpmConverter, Replace normalizer, byte_fallback on model object
  bpe    + whitespace   → same as above (split_by_whitespace is baked into SPM weights)
  unigram + nowhitespace → SpmConverter, Replace normalizer, patch byte_fallback in JSON
  unigram + whitespace   → same as above

Upload destinations:
  Bilingual   → catherinearnett/bilingual_tokenizers2   as {stem}/tokenizer.json
  Monolingual → catherinearnett/monolingual_tokenizers  as {stem}/tokenizer.json

Loadable after upload with:
  AutoTokenizer.from_pretrained("catherinearnett/monolingual_tokenizers/{stem}")

Usage:
    export HF_TOKEN_WRITE_bilingual_tokenizers=hf_...
    python upload_tokenizers.py
    python upload_tokenizers.py --spm_dir spm_tokenizers --workers 8 --batch_size 20
    python upload_tokenizers.py --dry_run    # convert only, no upload

Requirements:
    pip install transformers tokenizers sentencepiece huggingface_hub
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
 
# ── config ────────────────────────────────────────────────────────────────────
 
BI_REPO    = "catherinearnett/bilingual_tokenizers2"
MONO_REPO  = "catherinearnett/monolingual_tokenizers"
SPM_DIR    = "spm_tokenizers"
TMP_ROOT   = "hf_tmp"
BATCH_SIZE = 20
 
 
# ── filename parsing ──────────────────────────────────────────────────────────
 
def parse_stem(stem):
    parts = stem.split("_")
    if len(parts) == 5 and parts[2] in ("bpe", "unigram"):
        return {"kind": "mono", "tok_type": parts[2], "whitespace": parts[3]}
    if len(parts) == 9 and parts[4] in {"10", "25", "50", "75", "90"}:
        return {"kind": "bi",   "tok_type": parts[6], "whitespace": parts[7]}
    return None
 
 
# ── conversion ────────────────────────────────────────────────────────────────
 
def convert_spm(model_path, tok_type, out_dir):
    """
    Convert one SPM .model → HF tokenizer saved in out_dir/.
    tok_type: "bpe" or "unigram"
    whitespace behaviour is baked into the SPM model — no change needed here.
    """
    os.makedirs(out_dir, exist_ok=True)
 
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
 
    # Same normalizer + decoder for all four variants
    hf_tokenizer.backend_tokenizer.normalizer   = NormalizerReplace(" ", "▁")
    hf_tokenizer.backend_tokenizer.pre_tokenizer = None
    hf_tokenizer.backend_tokenizer.decoder = DecoderSequence([
        ByteFallback(),
        Metaspace(replacement="▁", prepend_scheme="never", split=False),
    ])
 
    if tok_type == "bpe":
        # byte_fallback settable directly on BPE model object
        hf_tokenizer.backend_tokenizer.model.byte_fallback = True
        hf_tokenizer.save_pretrained(out_dir)
    else:
        # unigram: save first, then patch byte_fallback into tokenizer.json
        hf_tokenizer.save_pretrained(out_dir)
        tok_json_path = os.path.join(out_dir, "tokenizer.json")
        with open(tok_json_path) as f:
            tok_data = json.load(f)
        tok_data["model"]["byte_fallback"] = True
        with open(tok_json_path, "w") as f:
            json.dump(tok_data, f, ensure_ascii=False, indent=2)
 
    return os.path.join(out_dir, "tokenizer.json")
 
 
def convert_one(args):
    """Worker: convert one SPM model. Returns (stem, tok_json_path_or_None, error_or_None)."""
    stem, model_path, out_dir = args
    info = parse_stem(stem)
    if info is None:
        return stem, None, "unparseable filename"
    # Skip if already converted locally
    tok_json = os.path.join(out_dir, "tokenizer.json")
    if os.path.exists(tok_json):
        return stem, tok_json, None
    try:
        tok_json = convert_spm(model_path, info["tok_type"], out_dir)
        return stem, tok_json, None
    except Exception as e:
        return stem, None, f"{e}\n{traceback.format_exc()}"
 
 
# ── upload ────────────────────────────────────────────────────────────────────
 
def get_uploaded_stems(api, repo_id, token):
    """
    Return set of stems already uploaded.
    Each tokenizer is a top-level subdirectory containing tokenizer.json,
    so listing root non-recursively gives us all stems as RepoFolder entries —
    no deep pagination, no 500 errors even with thousands of tokenizers.
    """
    stems = set()
    try:
        for item in api.list_repo_tree(
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo="",   # root
            recursive=False,   # top-level only — avoids HF pagination 500
            token=token,
        ):
            # RepoFolder has 'path' but not 'rfilename'
            # RepoFile has 'rfilename'
            # We want the folders (each is a tokenizer stem)
            if not hasattr(item, "rfilename") and hasattr(item, "path"):
                stems.add(item.path)
    except Exception as e:
        print(f"  Warning fetching {repo_id}: {e}")
    return stems
 
 
def upload_batch(api, repo_id, token, batch):
    """batch: list of (stem, tok_json_path)"""
    operations = [
        CommitOperationAdd(
            path_in_repo=f"{stem}/tokenizer.json",
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
        commit_message=f"Add {len(operations)} HF tokenizers",
        operations=operations,
    )
 
 
# ── group runner ──────────────────────────────────────────────────────────────
 
def run_group(jobs, repo_id, label, api, token, workers, batch_size,
              dry_run, tmp_dir, error_log):
    if not jobs:
        print(f"Nothing to do for {label}.\n")
        return
 
    total = len(jobs)
    pending = []   # (stem, tok_json_path) ready to upload
    ok = err = 0
 
    with ProcessPoolExecutor(max_workers=workers) as pool:
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
 
            if not dry_run and len(pending) >= batch_size:
                print(f"  → Uploading batch of {len(pending)} to {repo_id}...")
                try:
                    upload_batch(api, repo_id, token, pending)
                    for s, _ in pending:
                        shutil.rmtree(os.path.join(tmp_dir, s), ignore_errors=True)
                    pending.clear()
                    print(f"    ✓ Uploaded")
                except Exception as e:
                    print(f"    ✗ Upload failed: {e}")
                    error_log.write(f"\nUPLOAD ERROR ({repo_id}): {e}\n")
 
    # Upload remainder
    if not dry_run and pending:
        print(f"  → Uploading final batch of {len(pending)} to {repo_id}...")
        try:
            upload_batch(api, repo_id, token, pending)
            for s, _ in pending:
                shutil.rmtree(os.path.join(tmp_dir, s), ignore_errors=True)
            print(f"    ✓ Uploaded")
        except Exception as e:
            print(f"    ✗ Final upload failed: {e}")
            error_log.write(f"\nFINAL UPLOAD ERROR ({repo_id}): {e}\n")
 
    print(f"{label} done.  Converted: {ok}  Errors: {err}\n")
 
 
# ── main ──────────────────────────────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spm_dir",    default=SPM_DIR)
    parser.add_argument("--tmp_dir",    default=TMP_ROOT)
    parser.add_argument("--workers",    type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--dry_run",    action="store_true",
                        help="Convert only, no upload")
    parser.add_argument("--error_log",  default="convert_upload_errors.txt")
    args = parser.parse_args()
 
    token = os.environ.get("HF_TOKEN_WRITE_bilingual_tokenizers")
    if not token and not args.dry_run:
        sys.exit("ERROR: set HF_TOKEN_WRITE_bilingual_tokenizers env var")
 
    api = HfApi()
 
    # ── find local .model files ───────────────────────────────────────────────
    model_files = sorted(glob.glob(os.path.join(args.spm_dir, "*.model")))
    print(f"Found {len(model_files)} .model files in {args.spm_dir}/\n")
 
    # ── fetch already-uploaded stems ─────────────────────────────────────────
    if not args.dry_run:
        print("Checking what's already on HF...")
        bi_done   = get_uploaded_stems(api, BI_REPO,   token)
        mono_done = get_uploaded_stems(api, MONO_REPO, token)
        print(f"  {BI_REPO}:   {len(bi_done)} already uploaded")
        print(f"  {MONO_REPO}: {len(mono_done)} already uploaded\n")
    else:
        bi_done = mono_done = set()
 
    # ── split into bi / mono job lists ───────────────────────────────────────
    bi_jobs, mono_jobs = [], []
    for model_path in model_files:
        stem = os.path.splitext(os.path.basename(model_path))[0]
        info = parse_stem(stem)
        if info is None:
            print(f"  SKIP (unparseable): {stem}")
            continue
        out_dir = os.path.join(args.tmp_dir, stem)
        if info["kind"] == "bi":
            if stem not in bi_done:
                bi_jobs.append((stem, model_path, out_dir))
        else:
            if stem not in mono_done:
                mono_jobs.append((stem, model_path, out_dir))
 
    print(f"Jobs to convert + upload:")
    print(f"  Bilingual   : {len(bi_jobs)}")
    print(f"  Monolingual : {len(mono_jobs)}")
    print(f"  Total       : {len(bi_jobs) + len(mono_jobs)}\n")
 
    os.makedirs(args.tmp_dir, exist_ok=True)
 
    with open(args.error_log, "w") as error_log:
        run_group(bi_jobs,   BI_REPO,   "Bilingual",
                  api, token, args.workers, args.batch_size,
                  args.dry_run, args.tmp_dir, error_log)
        run_group(mono_jobs, MONO_REPO, "Monolingual",
                  api, token, args.workers, args.batch_size,
                  args.dry_run, args.tmp_dir, error_log)
 
    print(f"Errors (if any) logged to {args.error_log}")
 
 
if __name__ == "__main__":
    main()
