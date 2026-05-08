"""
Calculate CTC (token count) for all HF tokenizers against FLORES+.

Rules:
  Bilingual tokenizers  → CTC for lang1 and lang2 only
  Monolingual tokenizers → CTC for ALL available FLORES+ languages

CTC = total non-special tokens across all sentences (dev + devtest) for a language.

Filename convention (stem of .json in hf_tokenizers/):
  Monolingual : {lang}_{tok_type}_{whitespace}_{vocab_size}          (5 parts)
  Bilingual   : {lang1}_{lang2}_{p1}_{p2}_{tok_type}_{whitespace}_{vocab_size}  (9 parts)

FLORES+ is loaded via HuggingFace datasets with your read token.
Languages are loaded one at a time and cached to avoid re-downloading.
If a language config is not in FLORES+, it is skipped with a warning.

Output:
  ctc_results.csv — columns:
    tokenizer, kind, lang1, lang2, proportion, tok_type, whitespace,
    vocab_size, flores_lang, ctc

Usage:
    python calculate_ctc.py
    python calculate_ctc.py --hf_dir hf_tokenizers --out ctc_results.csv --workers 4

Requirements:
    pip install transformers datasets pandas
"""

import os
import glob
import argparse
import traceback
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset, get_dataset_config_names
import warnings
warnings.filterwarnings("ignore")

FLORES_REPO = "openlanguagedata/flores_plus"
FLORES_SPLITS = ["dev", "devtest"]
TEXT_COLUMN = "sentence"


# ── filename parsing ──────────────────────────────────────────────────────────

def parse_stem(stem):
    parts = stem.split("_")
    if len(parts) == 5 and parts[2] in ("bpe", "unigram"):
        return {
            "kind":       "mono",
            "lang":       "_".join(parts[0:2]),
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
            "lang":       None,
            "lang1":      "_".join(parts[0:2]),
            "lang2":      "_".join(parts[2:4]),
            "proportion": "_".join(parts[4:6]),
            "tok_type":   parts[6],
            "whitespace": parts[7],
            "vocab_size": parts[8],
        }
    return None


# ── FLORES helpers ────────────────────────────────────────────────────────────

def get_flores_configs(token):
    """Fetch all available language configs from FLORES+."""
    try:
        configs = get_dataset_config_names(FLORES_REPO, token=token)
        return set(configs)
    except Exception as e:
        print(f"WARNING: could not fetch FLORES+ config list: {e}")
        return set()


def load_flores_sentences(lang, token, flores_configs):
    """
    Load all sentences for a language from FLORES+ (dev + devtest).
    Returns list of strings, or None if lang not available.
    """
    if lang not in flores_configs:
        return None
    sentences = []
    for split in FLORES_SPLITS:
        try:
            ds = load_dataset(FLORES_REPO, lang, split=split, token=token,
                              trust_remote_code=False)
            sentences.extend(ds[TEXT_COLUMN])
        except Exception as e:
            print(f"  WARNING: could not load {lang}/{split}: {e}")
    return sentences if sentences else None


def compute_ctc(sentences, tokenizer):
    """Sum of non-special tokens across all sentences."""
    special_ids = set(tokenizer.all_special_ids)
    total = 0
    for sent in sentences:
        ids = tokenizer(sent)["input_ids"]
        total += sum(1 for t in ids if t not in special_ids)
    return total


# ── per-tokenizer worker ──────────────────────────────────────────────────────

def process_tokenizer(json_path, flores_configs, token, all_flores_langs):
    """
    Compute CTC for one tokenizer file.
    Returns list of result dicts (one per flores_lang evaluated).
    """
    stem = os.path.splitext(os.path.basename(json_path))[0]
    info = parse_stem(stem)
    if info is None:
        return [], f"SKIP (unparseable): {stem}"

    try:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=json_path)
    except Exception as e:
        return [], f"ERROR loading tokenizer {stem}: {e}"

    # Determine which FLORES languages to evaluate
    if info["kind"] == "bi":
        target_langs = [info["lang1"], info["lang2"]]
    else:
        # monolingual: all FLORES languages
        target_langs = sorted(all_flores_langs)

    rows = []
    for flores_lang in target_langs:
        if flores_lang not in flores_configs:
            continue
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
    parser.add_argument("--hf_dir",  default="hf_tokenizers",
                        help="Directory of .json HF fast tokenizer files")
    parser.add_argument("--out",     default="ctc_results.csv",
                        help="Output CSV path")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers (keep low — each loads from HF)")
    parser.add_argument("--resume",  action="store_true",
                        help="Skip tokenizers already present in output CSV")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN_READ")
    if not token:
        raise EnvironmentError("HF_TOKEN_READ environment variable not set.")

    # ── discover tokenizer files ──────────────────────────────────────────────
    json_files = sorted(glob.glob(os.path.join(args.hf_dir, "*.json")))
    print(f"Found {len(json_files)} tokenizer .json files in {args.hf_dir}/")

    if args.resume and os.path.exists(args.out):
        done = set(pd.read_csv(args.out)["tokenizer"].unique())
        json_files = [f for f in json_files
                      if os.path.splitext(os.path.basename(f))[0] not in done]
        print(f"  Resuming: {len(json_files)} tokenizers remaining")

    # ── fetch available FLORES+ configs ──────────────────────────────────────
    print(f"\nFetching available FLORES+ language configs...")
    flores_configs = get_flores_configs(token)
    print(f"  {len(flores_configs)} language configs available\n")

    # Languages used by bilingual tokenizers (for mono → evaluate all of these)
    # We evaluate mono tokenizers on all FLORES+ configs found.
    all_flores_langs = flores_configs  # full set

    # ── write CSV header if needed ────────────────────────────────────────────
    out_cols = ["tokenizer", "kind", "lang1", "lang2", "proportion",
                "tok_type", "whitespace", "vocab_size", "flores_lang", "ctc"]
    if not (args.resume and os.path.exists(args.out)):
        pd.DataFrame(columns=out_cols).to_csv(args.out, index=False)

    # ── process tokenizers ────────────────────────────────────────────────────
    # NOTE: we use a modest worker count because each worker downloads from HF.
    # For monolingual tokenizers (many FLORES langs), consider --workers 1-2
    # and let the HF datasets cache do the heavy lifting.
    total = len(json_files)
    done_count = 0
    error_count = 0

    # Sequential processing is safer here because datasets caches per-process
    # and parallel HF requests can hit rate limits. Use workers for bilingual
    # (only 2 langs each) but be careful with monolingual (200+ langs each).
    for i, json_path in enumerate(json_files, 1):
        stem = os.path.splitext(os.path.basename(json_path))[0]
        info = parse_stem(stem)
        kind = info["kind"] if info else "?"
        print(f"[{i}/{total}] {stem}  ({kind})")

        rows, err = process_tokenizer(json_path, flores_configs, token, all_flores_langs)

        if err:
            print(f"  {err}")
            error_count += 1
        else:
            if rows:
                pd.DataFrame(rows).to_csv(args.out, mode="a", header=False, index=False)
            done_count += 1
            print(f"  → {len(rows)} CTC values written")

    print(f"\nDone. Processed: {done_count}  Errors: {error_count}")
    print(f"Results written to {args.out}")
    total_rows = pd.read_csv(args.out).shape[0]
    print(f"Total rows in CSV: {total_rows}")


if __name__ == "__main__":
    main()
