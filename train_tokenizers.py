import sentencepiece as spm
import glob
import os
import traceback
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

LOG_FILE = "tokenizer_errors.txt"
SPM_DIR = "spm_tokenizers"


def log_error(label, exc):
    with open(LOG_FILE, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"TIME:  {datetime.now().isoformat()}\n")
        f.write(f"JOB:   {label}\n")
        f.write(f"ERROR: {exc}\n")
        f.write(traceback.format_exc())


def build_jobs():
    jobs = []
    for input_file in glob.glob("mixed_data/*.txt"):
        base = os.path.splitext(os.path.basename(input_file))[0]  # strip .txt
        # Normalize: remove trailing _nfc or _500mb__nfc suffixes
        stem = base.replace("_500mb", "").replace("_subset_1_nfc", "").strip("_")

        for model_type in ["unigram", "bpe"]:
            for split_by_whitespace in [True, False]:
                pretok = "whitespace" if split_by_whitespace else "nowhitespace"
                for vocab_size in [16384, 32768, 65536]:
                    tokenizer_name = f"{stem}_{model_type}_{pretok}_{vocab_size}"
                    jobs.append((input_file, tokenizer_name, model_type, split_by_whitespace, vocab_size))
    return jobs


def run_job(args):
    input_file, tokenizer_name, model_type, split_by_whitespace, vocab_size = args
    pretok = "whitespace" if split_by_whitespace else "nowhitespace"
    label = f"{tokenizer_name} | {model_type} | {pretok} | vocab={vocab_size}"
    spm_path = os.path.join(SPM_DIR, f"{tokenizer_name}.model")

    if os.path.exists(spm_path):
        return "skipped", label

    try:
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=os.path.join(SPM_DIR, tokenizer_name),
            vocab_size=vocab_size,
            model_type=model_type,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            normalization_rule_name="identity",
            split_by_whitespace=split_by_whitespace,
            byte_fallback=True,
            num_threads=1,  # one thread per worker; parallelism comes from the pool
        )
        return "trained", label
    except Exception as e:
        log_error(label, e)
        return "failed", label


if __name__ == "__main__":
    NUM_WORKERS = 4

    os.makedirs(SPM_DIR, exist_ok=True)  # once, before workers start

    with open(LOG_FILE, "w") as f:
        f.write(f"Tokenizer error log — started {datetime.now().isoformat()}\n")

    jobs = build_jobs()
    total = len(jobs)
    print(f"Total jobs: {total} | Workers: {NUM_WORKERS}")

    counts = {"trained": 0, "skipped": 0, "failed": 0}

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(run_job, job): job for job in jobs}
        for future in as_completed(futures):
            result, label = future.result()
            counts[result] += 1
            t, s, f = counts["trained"], counts["skipped"], counts["failed"]
            print(f"[{result.upper()}] {label}")
            print(f"  Progress: {t + s + f}/{total}  (✓ {t}  ⏭ {s}  ✗ {f})")

    t, s, f = counts["trained"], counts["skipped"], counts["failed"]
    print(f"\nDone. Trained: {t}, Skipped: {s}, Failed: {f}. Errors logged to {LOG_FILE}")
