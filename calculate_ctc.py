"""
Calculate CTC (token count) for all HF tokenizers against FLORES+.

Tokenizers are loaded directly from HuggingFace repos:
  Bilingual   → catherinearnett/bilingual_tokenizers2
  Monolingual → catherinearnett/monolingual_tokenizers

Each tokenizer is a subdirectory containing tokenizer.json, loadable with:
  AutoTokenizer.from_pretrained("{repo_id}/{stem}", token=...)

Rules:
  Bilingual tokenizers  → CTC for lang1 and lang2 only
  Monolingual tokenizers → CTC for ALL available FLORES+ languages

CTC = total non-special tokens across all sentences (dev + devtest).

Output:
  ctc_results.csv — columns:
    tokenizer, kind, lang1, lang2, proportion, tok_type, whitespace,
    vocab_size, flores_lang, ctc

Usage:
    export HF_TOKEN_READ=hf_...
    python calculate_ctc.py
    python calculate_ctc.py --out ctc_results.csv --resume

Requirements:
    pip install transformers datasets pandas huggingface_hub
"""

import os
import sys
import argparse
import traceback
import pandas as pd
from huggingface_hub import HfApi
from transformers import AutoTokenizer
from datasets import load_dataset, get_dataset_config_names
import warnings
warnings.filterwarnings("ignore")

BI_REPO    = "catherinearnett/bilingual_tokenizers2"
MONO_REPO  = "catherinearnett/monolingual_tokenizers"
FLORES_REPO   = "openlanguagedata/flores_plus"
FLORES_SPLITS = ["dev", "devtest"]
TEXT_COLUMN   = "sentence"

OUT_COLS = ["tokenizer", "kind", "lang1", "lang2", "proportion",
            "tok_type", "whitespace", "vocab_size", "flores_lang", "ctc"]


# ── filename parsing ──────────────────────────────────────────────────────────

def parse_stem(stem):
    parts = stem.split("_")
    if len(parts) == 5 and parts[2] in ("bpe", "unigram"):
        return {
            "kind":       "mono",
            "lang1":      "_".join(parts[0:2]),
            "lang2":      None,
            "proportion": None,
            "tok_type":   parts[2],
            "whitespace": parts[3],
            "vocab_size": parts[4],
        }
    if len(parts) == 9 and parts[4] in {"10", "25", "50", "75", "90"}:
        return {
            "kind":       "bi",
            "lang1":      "_".join(parts[0:2]),
            "lang2":      "_".join(parts[2:4]),
            "proportion": "_".join(parts[4:6]),
            "tok_type":   parts[6],
            "whitespace": parts[7],
            "vocab_size": parts[8],
        }
    return None


# ── HF repo helpers ───────────────────────────────────────────────────────────

def get_repo_stems(api, repo_id, token):
    """
    List all tokenizer stems in a repo (each is a top-level subdirectory).
    Uses non-recursive listing to avoid HF pagination 500 errors.
    """
    stems = []
    try:
        for item in api.list_repo_tree(
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo="",
            recursive=False,
            token=token,
        ):
            # RepoFolder: has 'path', no 'rfilename'
            if not hasattr(item, "rfilename") and hasattr(item, "path"):
                stems.append(item.path)
    except Exception as e:
        print(f"  WARNING fetching stems from {repo_id}: {e}")
    return stems


# ── FLORES helpers ────────────────────────────────────────────────────────────

def get_flores_configs(token):
    try:
        return set(get_dataset_config_names(FLORES_REPO, token=token))
    except Exception as e:
        print(f"WARNING: could not fetch FLORES+ config list: {e}")
        return set()


# Cache sentences in memory within a process to avoid re-downloading
_flores_cache = {}

def load_flores_sentences(lang, token, flores_configs):
    if lang not in flores_configs:
        return None
    if lang in _flores_cache:
        return _flores_cache[lang]
    sentences = []
    for split in FLORES_SPLITS:
        try:
            ds = load_dataset(FLORES_REPO, lang, split=split,
                              token=token, trust_remote_code=False)
            sentences.extend(ds[TEXT_COLUMN])
        except Exception as e:
            print(f"  WARNING: could not load {lang}/{split}: {e}")
    result = sentences if sentences else None
    _flores_cache[lang] = result
    return result


# ── CTC ───────────────────────────────────────────────────────────────────────

def compute_ctc(sentences, tokenizer):
    special_ids = set(tokenizer.all_special_ids)
    total = 0
    for sent in sentences:
        ids = tokenizer(sent)["input_ids"]
        total += sum(1 for t in ids if t not in special_ids)
    return total


def process_one(stem, repo_id, info, flores_configs, token, all_flores_langs):
    """Load tokenizer from HF and compute CTC for all target languages."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            f"{repo_id}/{stem}", token=token, trust_remote_code=False
        )
    except Exception as e:
        return [], f"ERROR loading tokenizer {stem}: {e}"

    target_langs = (
        [info["lang1"], info["lang2"]] if info["kind"] == "bi"
        else sorted(all_flores_langs)
    )

    rows = []
    for flores_lang in target_langs:
        sentences = load_flores_sentences(flores_lang, token, flores_configs)
        if sentences is None:
            continue
        try:
            ctc = compute_ctc(sentences, tokenizer)
        except Exception as e:
            ctc = None
        rows.append({
            "tokenizer":   stem,
            "kind":        info["kind"],
            "lang1":       info["lang1"],
            "lang2":       info["lang2"],
            "proportion":  info["proportion"],
            "tok_type":    info["tok_type"],
            "whitespace":  info["whitespace"],
            "vocab_size":  info["vocab_size"],
            "flores_lang": flores_lang,
            "ctc":         ctc,
        })

    return rows, None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",    default="ctc_results.csv")
    parser.add_argument("--resume", action="store_true",
                        help="Skip tokenizers already in output CSV")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN_READ")
    if not token:
        sys.exit("ERROR: set HF_TOKEN_READ environment variable")

    api = HfApi()

    # ── collect stems from both repos ─────────────────────────────────────────
    print(f"Fetching tokenizer list from HF repos...")
    bi_stems   = get_repo_stems(api, BI_REPO,   token)
    mono_stems = get_repo_stems(api, MONO_REPO, token)
    print(f"  {BI_REPO}:   {len(bi_stems)} tokenizers")
    print(f"  {MONO_REPO}: {len(mono_stems)} tokenizers")

    # Build unified job list: (stem, repo_id, info)
    jobs = []
    for stem in bi_stems:
        info = parse_stem(stem)
        if info:
            jobs.append((stem, BI_REPO, info))
        else:
            print(f"  SKIP (unparseable): {stem}")
    for stem in mono_stems:
        info = parse_stem(stem)
        if info:
            jobs.append((stem, MONO_REPO, info))
        else:
            print(f"  SKIP (unparseable): {stem}")

    print(f"  Total jobs: {len(jobs)}\n")

    # ── resume: skip already-done tokenizers ──────────────────────────────────
    if args.resume and os.path.exists(args.out):
        done = set(pd.read_csv(args.out)["tokenizer"].unique())
        before = len(jobs)
        jobs = [(s, r, i) for s, r, i in jobs if s not in done]
        print(f"Resuming: skipped {before - len(jobs)} done, {len(jobs)} remaining\n")

    # ── fetch FLORES+ configs ─────────────────────────────────────────────────
    print("Fetching FLORES+ language configs...")
    flores_configs = get_flores_configs(token)
    print(f"  {len(flores_configs)} configs available\n")
    all_flores_langs = flores_configs

    # ── write CSV header ──────────────────────────────────────────────────────
    if not (args.resume and os.path.exists(args.out)):
        pd.DataFrame(columns=OUT_COLS).to_csv(args.out, index=False)

    # ── process sequentially ──────────────────────────────────────────────────
    # Sequential is intentional: FLORES cache is per-process and HF datasets
    # caches to disk. Parallel workers can't share the in-memory cache and
    # may hit rate limits. Monolingual tokenizers re-use cached FLORES data
    # for each new tokenizer, so the bottleneck quickly becomes tokenization
    # rather than downloads.
    total = len(jobs)
    done_count = err_count = 0

    for i, (stem, repo_id, info) in enumerate(jobs, 1):
        print(f"[{i}/{total}] {stem}  ({info['kind']})")

        rows, err = process_one(
            stem, repo_id, info, flores_configs, token, all_flores_langs
        )

        if err:
            print(f"  {err}")
            err_count += 1
        else:
            if rows:
                pd.DataFrame(rows).to_csv(args.out, mode="a", header=False, index=False)
            done_count += 1
            print(f"  → {len(rows)} CTC values written")

    print(f"\nDone. Processed: {done_count}  Errors: {err_count}")
    print(f"Results: {args.out}  ({pd.read_csv(args.out).shape[0]} rows)")


if __name__ == "__main__":
    main()
