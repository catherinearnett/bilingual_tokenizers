import sentencepiece as spm
import glob
import os
import sys
import time
import traceback
import multiprocessing
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from huggingface_hub import HfApi, CommitOperationAdd

LOG_FILE = "tokenizer_errors.txt"
SPM_DIR = "spm_tokenizers"
HF_TOKEN_WRITE = os.environ["HF_TOKEN_WRITE_bilingual_tokenizers"]
HF_REPO_ID = "catherinearnett/bilingual_tokenizers"

TOTAL_CORES = multiprocessing.cpu_count()
USABLE_CORES = max(1, int(TOTAL_CORES * 0.8))  # 80% of available cores
NUM_WORKERS = max(1, USABLE_CORES // 6)
SPM_THREADS = max(1, USABLE_CORES // NUM_WORKERS)


def log_error(label, exc):
    with open(LOG_FILE, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"TIME:  {datetime.now().isoformat()}\n")
        f.write(f"JOB:   {label}\n")
        f.write(f"ERROR: {exc}\n")
        f.write(traceback.format_exc())


def get_hf_uploaded_files():
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset", token=HF_TOKEN_WRITE)
        return {os.path.basename(f) for f in files}
    except Exception as e:
        print(f"Warning: could not fetch HF file list: {e}")
        return set()


def upload_batch(tokenizer_names):
    if not tokenizer_names:
        return
    api = HfApi()
    operations = []
    for name in tokenizer_names:
        for ext in [".model", ".vocab"]:
            local_path = os.path.join(SPM_DIR, f"{name}{ext}")
            if os.path.exists(local_path):
                operations.append(CommitOperationAdd(
                    path_in_repo=f"spm_tokenizers/{name}{ext}",
                    path_or_fileobj=local_path,
                ))
    if not operations:
        return
    try:
        api.create_commit(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN_WRITE,
            commit_message=f"Add {len(tokenizer_names)} tokenizers",
            operations=operations,
        )
        print(f"  ↑ Uploaded {tokenizer_names[0]} ({len(operations)} files)")
    except Exception as e:
        print(f"  ✗ Upload failed: {e}")
        log_error("upload", e)


def build_jobs(uploaded_files):
    jobs = []
    for input_file in glob.glob("mixed_data/*.txt"):
        base = os.path.splitext(os.path.basename(input_file))[0]
        stem = base.replace("_500mb", "").replace("_subset_1_nfc", "").strip("_")
        for model_type in ["unigram", "bpe"]:
            for split_by_whitespace in [True, False]:
                pretok = "whitespace" if split_by_whitespace else "nowhitespace"
                for vocab_size in [16384, 32768, 65536]:
                    tokenizer_name = f"{stem}_{model_type}_{pretok}_{vocab_size}"
                    if f"{tokenizer_name}.model" in uploaded_files:
                        continue
                    jobs.append((input_file, tokenizer_name, model_type, split_by_whitespace, vocab_size))
    jobs.sort(key=lambda j: j[4])  # sort by vocab_size ascending to manage memory pressure
    return jobs


def run_job(args):
    input_file, tokenizer_name, model_type, split_by_whitespace, vocab_size = args
    pretok = "whitespace" if split_by_whitespace else "nowhitespace"
    label = f"{tokenizer_name} | {model_type} | {pretok} | vocab={vocab_size}"
    spm_path = os.path.join(SPM_DIR, f"{tokenizer_name}.model")

    if os.path.exists(spm_path):
        return "skipped", label, tokenizer_name

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
            num_threads=SPM_THREADS,
        )
        return "trained", label, tokenizer_name
    except Exception as e:
        log_error(label, e)
        return "failed", label, None


if __name__ == "__main__":
    os.makedirs(SPM_DIR, exist_ok=True)

    print(f"Core config: {TOTAL_CORES} total → {USABLE_CORES} usable (80%) → {NUM_WORKERS} workers × {SPM_THREADS} threads each")

    with open(LOG_FILE, "a") as f:
        f.write(f"\nTokenizer run — started {datetime.now().isoformat()}\n")
        f.write(f"Core config: {TOTAL_CORES} total, {USABLE_CORES} usable, {NUM_WORKERS} workers, {SPM_THREADS} threads/job\n")

    print("Fetching already-uploaded files from HF...")
    uploaded_files = get_hf_uploaded_files()
    print(f"  {len(uploaded_files)} files already on HF, skipping those jobs.")

    jobs = build_jobs(uploaded_files)
    total = len(jobs)
    print(f"Total jobs: {total} | Workers: {NUM_WORKERS} | Threads/job: {SPM_THREADS}")

    if total == 0:
        print("All tokenizers already uploaded. Nothing to do.")
        sys.exit(0)

    counts = {"trained": 0, "skipped": 0, "failed": 0}

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(run_job, job): job for job in jobs}
        for future in as_completed(futures):
            result, label, tokenizer_name = future.result()
            counts[result] += 1
            t, s, f = counts["trained"], counts["skipped"], counts["failed"]
            print(f"[{result.upper()}] {label}")
            print(f"  Progress: {t + s + f}/{total}  (✓ {t}  ⏭ {s}  ✗ {f})")

            if result in ("trained", "skipped") and tokenizer_name:
                if f"{tokenizer_name}.model" not in uploaded_files:
                    upload_batch([tokenizer_name])

    t, s, f = counts["trained"], counts["skipped"], counts["failed"]
    print(f"\nDone. Trained: {t}, Skipped: {s}, Failed: {f}. Errors logged to {LOG_FILE}")
    sys.exit(0)
