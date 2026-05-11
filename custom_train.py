"""
Train custom-mix tokenizers from extreme_pairs_with_crossing.csv.
Input files are in custom_mix_data/, output tokenizers go to custom_spm_tokenizers/.

File naming convention:
  Input:  {lang1}_{lang2}_{p1}_{p2}_{tok_type}_{whitespace}_{vocab_size}_nfc.txt
  Output: {lang1}_{lang2}_{p1}_{p2}_{tok_type}_{whitespace}_{vocab_size}.model/.vocab
"""

import os
import glob
import traceback
import sentencepiece as spm
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = "custom_mix_data"
SPM_DIR    = "custom_spm_tokenizers"
LOG_FILE   = "custom_tokenizer_errors.txt"
CSV_PATH   = "extreme_pairs_with_crossing.csv"
NUM_WORKERS = 16

# ── Build jobs from CSV ───────────────────────────────────────────────────────

def build_jobs(csv_path):
    df = pd.read_csv(csv_path).dropna(subset=['crossing_proportion'])

    ready   = []
    missing = []

    # build lookup of available input files
    file_map = {}
    for path in glob.glob(os.path.join(DATA_DIR, "*.txt")):
        base = os.path.basename(path)
        file_map[base] = path

    for _, row in df.iterrows():
        lang1      = row['lang1']
        lang2      = row['lang2']
        tok_type   = row['tok_type']
        whitespace = row['whitespace']
        vocab_size = int(row['vocab_size'])
        p1         = round(row['crossing_proportion'])
        p2         = 100 - p1

        tokenizer_name = f"{lang1}_{lang2}_{p1}_{p2}_{tok_type}_{whitespace}_{vocab_size}"
        input_filename = f"{tokenizer_name}_nfc.txt"
        input_path     = os.path.join(DATA_DIR, input_filename)

        if input_filename in file_map:
            ready.append({
                'tokenizer_name': tokenizer_name,
                'input_file':     file_map[input_filename],
                'tok_type':       tok_type,
                'whitespace':     whitespace,
                'vocab_size':     vocab_size,
            })
        else:
            missing.append(input_filename)

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
    tokenizer_name = job['tokenizer_name']
    input_file     = job['input_file']
    tok_type       = job['tok_type']
    whitespace     = job['whitespace']
    vocab_size     = job['vocab_size']
    split_by_ws    = (whitespace == 'whitespace')
    spm_path       = os.path.join(SPM_DIR, f"{tokenizer_name}.model")

    if os.path.exists(spm_path):
        return "skipped", tokenizer_name

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
        return "trained", tokenizer_name
    except Exception as e:
        log_error(tokenizer_name, e)
        return "failed", tokenizer_name

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(SPM_DIR, exist_ok=True)
    with open(LOG_FILE, "w") as f:
        f.write(f"Custom tokenizer error log — started {datetime.now().isoformat()}\n")

    ready_jobs, missing_files = build_jobs(CSV_PATH)

    print(f"Jobs ready to train : {len(ready_jobs)}")
    if missing_files:
        print(f"Input files not found ({len(missing_files)}) — skipping:")
        for f in missing_files:
            print(f"  {f}")
    print()

    if not ready_jobs:
        print("Nothing to train!")
        return

    total  = len(ready_jobs)
    counts = {"trained": 0, "skipped": 0, "failed": 0}

    print(f"Training {total} tokenizers with {NUM_WORKERS} workers...\n")
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
    print(f"\nDone. Trained: {t} | Skipped: {s} | Failed: {f}")
    print(f"Errors logged to {LOG_FILE}")
    print(f"\nUpload with:")
    print(f"  huggingface-cli upload catherinearnett/bilingual_tokenizers ./{SPM_DIR} custom_spm_tokenizers --repo-type dataset")


if __name__ == "__main__":
    main()
