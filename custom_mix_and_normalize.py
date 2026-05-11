import random
import unicodedata
import os
import pandas as pd

os.makedirs('custom_mix_data', exist_ok=True)
TOTAL_SIZE = 500_000_000

extreme_pairs_with_crossing = pd.read_csv('extreme_pairs_with_crossing.csv')

def normalize_string(text):
    return unicodedata.normalize('NFC', text)

def read_cached(lang_code, max_bytes):
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

for _, row in extreme_pairs_with_crossing.iterrows():
    lang1 = row['lang1']
    lang2 = row['lang2']
    crossing = row['crossing_proportion']
    tok_type = row['tok_type']
    whitespace = row['whitespace']
    vocab_size = int(row['vocab_size'])

    if pd.isna(crossing):
        print(f"Skipping {lang1}/{lang2} ({tok_type}|{whitespace}|{vocab_size}) — no crossing proportion")
        continue

    l1_proportion = round(crossing)
    l2_proportion = 100 - l1_proportion

    try:
        l1_data = read_cached(lang1, TOTAL_SIZE * (l1_proportion / 100))
        l2_data = read_cached(lang2, TOTAL_SIZE * (l2_proportion / 100))
    except Exception as e:
        print(f"FAILED loading {lang1}/{lang2}: {e}")
        continue

    mixed_lines = l1_data + l2_data
    random.shuffle(mixed_lines)

    out_name = (
        f'custom_mix_data/'
        f'{lang1}_{lang2}_{l1_proportion}_{l2_proportion}'
        f'_{tok_type}_{whitespace}_{vocab_size}_nfc.txt'
    )

    try:
        with open(out_name, 'w', encoding='utf-8') as f:
            for line in mixed_lines:
                f.write(normalize_string(line))
        print(f"OK: {out_name}")
    except Exception as e:
        print(f"FAILED writing {out_name}: {e}")
