import random
import unicodedata
import os
import pandas as pd
from datasets import load_dataset, get_dataset_config_names
from multiprocessing import Pool, cpu_count

TOTAL_SIZE = 500_000_000
NUM_WORKERS = cpu_count()
os.makedirs('mixed_data', exist_ok=True)


def normalize_string(text):
    return unicodedata.normalize('NFC', text)


def cache_language(lang_code):
    """Download and cache a language to local /tmp once."""
    out_path = f'/tmp/{lang_code}_subset_1.txt'
    if os.path.exists(out_path):
        return f"CACHED (already existed): {lang_code}"

    try:
        dataset = load_dataset(
            "catherinearnett/bilingual-tokenizer-training-data",
            name=f"{lang_code}_subset_1",
            split="train",
            streaming=True,
            trust_remote_code=True
        )
        with open(out_path, 'w', encoding='utf-8') as f:
            for example in dataset:
                for line in example['text'].splitlines():
                    f.write(line + '\n')
        return f"OK: {lang_code}"
    except Exception as e:
        return f"FAILED: {lang_code} — {e}"


def read_cached(lang_code, max_bytes):
    """Read up to max_bytes from a locally cached language file."""
    lines = []
    total_bytes = 0
    with open(f'/tmp/{lang_code}_subset_1.txt', encoding='utf-8') as f:
        for line in f:
            encoded = line.encode('utf-8')
            if total_bytes + len(encoded) > max_bytes:
                break
            lines.append(line)
            total_bytes += len(encoded)
    return lines


def process_language_pair(args):
    """Mix two languages at 5 ratios and write NFC-normalized output files."""
    l1_name, l2_name, total_size = args
    for mix in [10, 25, 50, 75, 90]:
        l1_proportion = mix
        l2_proportion = 100 - mix

        try:
            l1_data = read_cached(l1_name, total_size * (l1_proportion / 100))
            l2_data = read_cached(l2_name, total_size * (l2_proportion / 100))
        except Exception as e:
            return f"FAILED loading {l1_name}/{l2_name}: {e}"

        mixed_lines = l1_data + l2_data
        random.shuffle(mixed_lines)

        nfc_name = f'mixed_data/{l1_name}_{l2_name}_{l1_proportion}_{l2_proportion}_subset_1_nfc.txt'

        try:
            with open(nfc_name, 'w', encoding='utf-8') as f:
                for line in mixed_lines:
                    f.write(normalize_string(line))
        except Exception as e:
            return f"FAILED writing {nfc_name}: {e}"

    return f"OK: {l1_name}/{l2_name}"


def normalize_monolingual(args):
    """Write a NFC-normalized monolingual file from the local cache."""
    l_name, total_size = args
    dst = f'mixed_data/{l_name}_500mb_subset_1_nfc.txt'
    try:
        lines = read_cached(l_name, total_size)
        with open(dst, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(normalize_string(line))
        return f"OK: {l_name}"
    except Exception as e:
        return f"FAILED: {l_name} — {e}"


# --- Main ---
train = pd.read_csv('language_pairs_train.csv')
test  = pd.read_csv('language_pairs_test.csv')
all_pairs = pd.concat([train, test]).reset_index(drop=True)

pair_languages = set(all_pairs['l1'].tolist() + all_pairs['l2'].tolist())

subsets = get_dataset_config_names("catherinearnett/bilingual-tokenizer-training-data")
seen, mono_languages = set(), []
for s in subsets:
    l = s[:8]
    if l not in seen:
        seen.add(l)
        mono_languages.append(l)

all_languages = pair_languages | set(mono_languages)

# --- Step 1: Cache all languages ---
print(f"Caching {len(all_languages)} languages with {NUM_WORKERS} workers...")
with Pool(NUM_WORKERS) as pool:
    cache_results = pool.map(cache_language, list(all_languages))

cache_failures = [r for r in cache_results if r.startswith("FAILED")]
if cache_failures:
    print(f"\n{len(cache_failures)} cache failures:")
    for r in cache_failures:
        print(" ", r)
    print("Aborting — fix cache failures before proceeding.")
    exit(1)
else:
    print("All languages cached successfully.\n")

# --- Step 2 & 3: Mix bilingual pairs and normalize monolingual files ---
bilingual_args = [(row['l1'], row['l2'], TOTAL_SIZE) for _, row in all_pairs.iterrows()]
mono_args = [(l, TOTAL_SIZE) for l in mono_languages]

print(f"Submitting {len(bilingual_args)} bilingual and {len(mono_args)} monolingual jobs...")
with Pool(NUM_WORKERS) as pool:
    bilingual_results = pool.map(process_language_pair, bilingual_args)
    mono_results = pool.map(normalize_monolingual, mono_args)

all_results = bilingual_results + mono_results
failed = [r for r in all_results if r.startswith("FAILED")]
succeeded = [r for r in all_results if r.startswith("OK")]

print(f"\nDone. {len(succeeded)} succeeded, {len(failed)} failed.")
if failed:
    print("Failures:")
    for r in failed:
        print(" ", r)