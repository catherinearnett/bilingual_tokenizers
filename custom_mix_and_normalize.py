from datasets import load_dataset
import random
import unicodedata
import os
import pandas as pd

os.makedirs('custom_mix_data', exist_ok=True)
TOTAL_SIZE = 500_000_000

def normalize_string(text):
    return unicodedata.normalize('NFC', text)

def fetch_language(lang_code, max_bytes):
    """Stream directly from HuggingFace up to max_bytes."""
    dataset = load_dataset(
        "catherinearnett/bilingual-tokenizer-training-data",
        name=f"{lang_code}_subset_1",
        split="train",
        streaming=True,
        trust_remote_code=True
    )
    lines = []
    total_bytes = 0
    for example in dataset:
        for line in example['text'].splitlines():
            encoded = (line + '\n').encode('utf-8')
            if total_bytes + len(encoded) > max_bytes:
                return lines
            lines.append(line + '\n')
            total_bytes += len(encoded)
    return lines

extreme_pairs_with_crossing = pd.read_csv('extreme_pairs_with_crossing.csv')

for _, row in extreme_pairs_with_crossing.iterrows():
    lang1 = row['lang1']
    lang2 = row['lang2']
    crossing = row['crossing_proportion']
    tok_type = row['tok_type']
    whitespace = row['whitespace']
    vocab_size = int(row['vocab_size'])

    if pd.isna(crossing):
        print(f"Skipping {lang1}/{lang2} — no crossing proportion")
        continue

    l1_proportion = round(crossing)
    l2_proportion = 100 - l1_proportion

    out_name = (
        f'custom_mix_data/'
        f'{lang1}_{lang2}_{l1_proportion}_{l2_proportion}'
        f'_{tok_type}_{whitespace}_{vocab_size}_nfc.txt'
    )

    if os.path.exists(out_name):
        print(f"SKIP (exists): {out_name}")
        continue

    try:
        print(f"Fetching {lang1} ({l1_proportion}%)...")
        l1_data = fetch_language(lang1, TOTAL_SIZE * (l1_proportion / 100))
        print(f"Fetching {lang2} ({l2_proportion}%)...")
        l2_data = fetch_language(lang2, TOTAL_SIZE * (l2_proportion / 100))
    except Exception as e:
        print(f"FAILED loading {lang1}/{lang2}: {e}")
        continue

    mixed_lines = l1_data + l2_data
    random.shuffle(mixed_lines)

    try:
        with open(out_name, 'w', encoding='utf-8') as f:
            for line in mixed_lines:
                f.write(normalize_string(line))
        print(f"OK: {out_name}")
    except Exception as e:
        print(f"FAILED writing {out_name}: {e}")
