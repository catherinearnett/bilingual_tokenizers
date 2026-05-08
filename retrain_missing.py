"""
Identifies and retrains missing tokenizers from the bilingual_tokenizers dataset.

Monolingual format: {lang}_{tok_type}_{whitespace}_{vocab_size}
  → 5 underscore-parts, parts[4] is the vocab size (a digit)
  → BUT parts[2] is tok_type (alpha), so we distinguish by checking
     whether parts[2] is alpha (mono) vs parts[4] being a proportion (bi)

Bilingual format: {lang1}_{lang2}_{p1}_{p2}_{tok_type}_{whitespace}_{vocab_size}
  → 9 underscore-parts, parts[4] is proportion like "10", "25", "50", "75", "90"

Key distinction: bilingual proportions are always one of {10,25,50,75,90}
                 monolingual vocab sizes are always one of {16384,32768,65536}
"""

import os
import glob
import traceback
import sentencepiece as spm
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from huggingface_hub import list_repo_files
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
REPO_ID   = "catherinearnett/bilingual_tokenizers"
REPO_TYPE = "dataset"
SUBFOLDER = "spm_tokenizers"
SPM_DIR   = "spm_tokenizers"
LOG_FILE  = "retrain_errors.txt"
NUM_WORKERS = 16

BILINGUAL_PROPORTIONS = {"10", "25", "50", "75", "90"}
EXPECTED_PROPORTIONS  = {"10_90", "25_75", "50_50", "75_25", "90_10"}
EXPECTED_VOCAB_SIZES  = {"16384", "32768", "65536"}

# Monolingual: all 8 combinations
EXPECTED_MONO_KEYS = {
    (tt, ws, vs)
    for tt in ("bpe", "unigram")
    for ws in ("nowhitespace", "whitespace")
    for vs in ("16384", "32768", "65536")
}

# ── Filename parsers ──────────────────────────────────────────────────────────

def parse_filename(stem):
    """
    Returns dict with 'kind' = 'mono' or 'bi', or None if unparseable.

    Disambiguation:
      - Split on '_' gives parts
      - Monolingual: 5 parts where parts[2] is a known tok_type (bpe/unigram)
      - Bilingual:   9 parts where parts[4] is a bilingual proportion digit
    """
    parts = stem.split("_")

    # Monolingual: exactly 5 parts, parts[2] is tok_type
    if len(parts) == 5 and parts[2] in ("bpe", "unigram"):
        return {
            "kind":       "mono",
            "lang":       "_".join(parts[0:2]),
            "tok_type":   parts[2],
            "whitespace": parts[3],
            "vocab_size": parts[4],
        }

    # Bilingual: exactly 9 parts, parts[4] is a proportion value
    if len(parts) == 9 and parts[4] in BILINGUAL_PROPORTIONS:
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

# ── Fetch existing tokenizers from HuggingFace ────────────────────────────────

def fetch_existing():
    print(f"Fetching file list from {REPO_ID}...")
    all_files = list(list_repo_files(REPO_ID, repo_type=REPO_TYPE))
    vocab_files = [
        f for f in all_files
        if f.startswith(SUBFOLDER + "/") and f.endswith(".vocab")
    ]
    print(f"Found {len(vocab_files)} .vocab files on HuggingFace\n")

    mono_parsed = []
    bi_parsed   = []
    unparseable = []

    for path in vocab_files:
        stem   = path.split("/")[-1].replace(".vocab", "")
        result = parse_filename(stem)
        if result is None:
            unparseable.append(stem)
        elif result["kind"] == "mono":
            mono_parsed.append(result)
        else:
            bi_parsed.append(result)

    if unparseable:
        print(f"WARNING: Still could not parse {len(unparseable)} filenames:")
        for u in unparseable[:5]:
            print(f"  {u}")
        if len(unparseable) > 5:
            print(f"  ... and {len(unparseable)-5} more")
        print()

    print(f"Monolingual tokenizers found : {len(mono_parsed)}")
    print(f"Bilingual  tokenizers found  : {len(bi_parsed)}\n")
    return mono_parsed, bi_parsed

# ── Identify missing jobs ─────────────────────────────────────────────────────

def find_missing_mono(mono_parsed):
    """
    For each language found in the dataset, check which of the 12 conditions
    (2 tok_types × 2 whitespace × 3 vocab_sizes) are missing.
    """
    # What exists on HuggingFace
    existing = defaultdict(set)
    for p in mono_parsed:
        existing[p["lang"]].add((p["tok_type"], p["whitespace"], p["vocab_size"]))

    missing_jobs = []
    for lang, found in existing.items():
        for key in EXPECTED_MONO_KEYS - found:
            tok_type, whitespace, vocab_size = key
            missing_jobs.append({
                "kind":       "mono",
                "lang":       lang,
                "tok_type":   tok_type,
                "whitespace": whitespace,
                "vocab_size": vocab_size,
            })

    return missing_jobs


def find_missing_bi(bi_parsed):
    """
    For each (lang1, lang2, tok_type, whitespace, vocab_size) group,
    check which of the 5 proportions are missing.
    """
    existing = defaultdict(set)
    for p in bi_parsed:
        key = (p["lang1"], p["lang2"], p["tok_type"], p["whitespace"], p["vocab_size"])
        existing[key].add(p["proportion"])

    missing_jobs = []
    for key, found_props in existing.items():
        lang1, lang2, tok_type, whitespace, vocab_size = key
        for prop in EXPECTED_PROPORTIONS - found_props:
            p1, p2 = prop.split("_")
            missing_jobs.append({
                "kind":       "bi",
                "lang1":      lang1,
                "lang2":      lang2,
                "proportion": prop,
                "p1":         p1,
                "p2":         p2,
                "tok_type":   tok_type,
                "whitespace": whitespace,
                "vocab_size": vocab_size,
            })

    return missing_jobs

# ── Map missing jobs → input files ───────────────────────────────────────────

def map_jobs_to_files(missing_mono, missing_bi, data_dir="mixed_data"):
    """
    Match each missing job to its input .txt file in mixed_data/.
    Files are named like:
      Monolingual: {lang}.txt  or  {lang}_nfc.txt  or  {lang}_500mb_subset_1_nfc.txt
      Bilingual:   {lang1}_{lang2}_{p1}_{p2}.txt  (or with _nfc suffix)
    Returns (ready_jobs, unmatched_jobs).
    """
    # Build a lookup: normalized stem → actual path
    file_map = {}
    for path in glob.glob(os.path.join(data_dir, "*.txt")):
        base = os.path.splitext(os.path.basename(path))[0]
        stem = base.replace("_500mb", "").replace("_subset_1_nfc", "").replace("_nfc", "").strip("_")
        file_map[stem] = path

    ready   = []
    missing = []

    for job in missing_mono:
        key = job["lang"]
        if key in file_map:
            job["input_file"]      = file_map[key]
            job["tokenizer_name"]  = f"{job['lang']}_{job['tok_type']}_{job['whitespace']}_{job['vocab_size']}"
            ready.append(job)
        else:
            job["missing_key"] = key
            missing.append(job)

    for job in missing_bi:
        # Bilingual data file key: {lang1}_{lang2}_{p1}_{p2}
        key = f"{job['lang1']}_{job['lang2']}_{job['p1']}_{job['p2']}"
        if key in file_map:
            job["input_file"]     = file_map[key]
            job["tokenizer_name"] = (
                f"{job['lang1']}_{job['lang2']}_{job['p1']}_{job['p2']}"
                f"_{job['tok_type']}_{job['whitespace']}_{job['vocab_size']}"
            )
            ready.append(job)
        else:
            job["missing_key"] = key
            missing.append(job)

    return ready, missing

# ── Training ──────────────────────────────────────────────────────────────────

def log_error(label, exc):
    with open(LOG_FILE, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"TIME:  {datetime.now().isoformat()}\n")
        f.write(f"JOB:   {label}\n")
        f.write(f"ERROR: {exc}\n")
        f.write(traceback.format_exc())


def run_job(job):
    tokenizer_name = job["tokenizer_name"]
    input_file     = job["input_file"]
    tok_type       = job["tok_type"]
    whitespace     = job["whitespace"]
    vocab_size     = int(job["vocab_size"])
    split_by_ws    = (whitespace == "whitespace")
    label          = tokenizer_name
    spm_path       = os.path.join(SPM_DIR, f"{tokenizer_name}.model")

    if os.path.exists(spm_path):
        return "skipped", label

    try:
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=os.path.join(SPM_DIR, tokenizer_name),
            vocab_size=vocab_size,
            model_type=tok_type,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            normalization_rule_name="identity",
            split_by_whitespace=split_by_ws,
            byte_fallback=True,
            num_threads=1,
        )
        return "trained", label
    except Exception as e:
        log_error(label, e)
        return "failed", label

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(SPM_DIR, exist_ok=True)
    with open(LOG_FILE, "w") as f:
        f.write(f"Retrain error log — started {datetime.now().isoformat()}\n")

    # 1. Fetch what exists
    mono_parsed, bi_parsed = fetch_existing()

    # 2. Find what's missing
    missing_mono = find_missing_mono(mono_parsed)
    missing_bi   = find_missing_bi(bi_parsed)
    print(f"Missing monolingual tokenizers : {len(missing_mono)}")
    print(f"Missing bilingual  tokenizers  : {len(missing_bi)}")
    print(f"Total missing                  : {len(missing_mono) + len(missing_bi)}\n")

    # 3. Map to input files
    ready_jobs, unmatched = map_jobs_to_files(missing_mono, missing_bi)
    print(f"Jobs matched to input files : {len(ready_jobs)}")
    if unmatched:
        print(f"Jobs with NO input file found ({len(unmatched)}) — skipping:")
        for j in unmatched:
            print(f"  [{j['kind']}] missing key: {j['missing_key']}")
    print()

    if not ready_jobs:
        print("Nothing to train!")
        return

    # 4. Train
    total = len(ready_jobs)
    print(f"Training {total} missing tokenizers with {NUM_WORKERS} workers...\n")
    counts = {"trained": 0, "skipped": 0, "failed": 0}

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(run_job, job): job for job in ready_jobs}
        for future in as_completed(futures):
            result, label = future.result()
            counts[result] += 1
            t, s, f = counts["trained"], counts["skipped"], counts["failed"]
            done = t + s + f
            print(f"[{result.upper():7s}] {label}")
            print(f"  Progress: {done}/{total}  (✓ {t}  ⏭ {s}  ✗ {f})")

    t, s, f = counts["trained"], counts["skipped"], counts["failed"]
    print(f"\nDone. Trained: {t} | Skipped: {f} | Failed: {f}")
    print(f"Errors logged to {LOG_FILE}")
    print(f"\nRemember to upload the new tokenizers to HuggingFace:")
    print(f"  huggingface-cli upload {REPO_ID} ./{SPM_DIR} {SUBFOLDER} --repo-type dataset")


if __name__ == "__main__":
    main()
