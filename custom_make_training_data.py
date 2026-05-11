"""
Create byte-premium-balanced bilingual datasets from monolingual 500MB files.

For each language pair in extreme_pairs_with_crossing.csv:
  1. Look up byte premium between lang1 and lang2
  2. Compute scaled effective size for each (500MB / byte_premium)
  3. Take the smaller effective size as the target
  4. Read that many raw bytes from each monolingual file
  5. Shuffle and write to balanced_data/{lang1}_{lang2}_bp_balanced.txt

Requires: byte-premium-tool cloned alongside this script (or on PYTHONPATH)
  git clone https://github.com/catherinearnett/byte-premium-tool.git
"""

import os
import sys
import random
import unicodedata
import pandas as pd

# -- adjust this path if byte-premium-tool is elsewhere
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'byte-premium-tool'))
from byte_premium_tool import get_pairwise_premium

MONO_DIR    = 'mixed_data'          # where {lang}_500mb_subset_1_nfc.txt files live
OUT_DIR     = 'custom_training_data'
TOTAL_BYTES = 500_000_000           # 500 MB raw size of each monolingual file

os.makedirs(OUT_DIR, exist_ok=True)


def find_mono_file(lang):
    """Locate the monolingual 500mb file for a language."""
    candidates = [
        os.path.join(MONO_DIR, f'{lang}_500mb_subset_1_nfc.txt'),
        os.path.join(MONO_DIR, f'{lang}_subset_1_nfc.txt'),
        os.path.join(MONO_DIR, f'{lang}.txt'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def read_bytes(path, max_bytes):
    """Read up to max_bytes from a text file, returning complete lines."""
    lines = []
    total = 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            encoded = line.encode('utf-8')
            if total + len(encoded) > max_bytes:
                break
            lines.append(line)
            total += len(encoded)
    return lines, total


def normalize(text):
    return unicodedata.normalize('NFC', text)


def flores_code(lang):
    """Convert flores lang code (e.g. hye_Armn) to byte-premium format (hye_armn)."""
    parts = lang.split('_')
    return f"{parts[0]}_{parts[1].lower()}"


# load unique pairs (deduplicate — same pair may appear across facets)
df = pd.read_csv('extreme_pairs_with_crossing.csv')
pairs = df[['lang1', 'lang2']].drop_duplicates().reset_index(drop=True)

print(f"Processing {len(pairs)} unique language pairs\n")

results = []

for _, row in pairs.iterrows():
    lang1, lang2 = row['lang1'], row['lang2']
    out_path = os.path.join(OUT_DIR, f'{lang1}_{lang2}_bp_balanced.txt')

    if os.path.exists(out_path):
        print(f"SKIP (exists): {out_path}")
        continue

    # -- locate monolingual files
    f1 = find_mono_file(lang1)
    f2 = find_mono_file(lang2)
    if not f1 or not f2:
        missing = [l for l, f in [(lang1, f1), (lang2, f2)] if not f]
        print(f"SKIP (no mono file): {missing}")
        continue

    # -- byte premium: how many bytes does lang1 take relative to lang2?
    # premium > 1 means lang1 is more expensive per unit of content
    try:
        bp = get_pairwise_premium(flores_code(lang1), flores_code(lang2), verbose=False)
    except Exception as e:
        print(f"SKIP (byte premium lookup failed for {lang1}/{lang2}): {e}")
        continue

    # -- effective content size if we use all 500MB of each
    # effective = raw_bytes / byte_premium  (relative to lang2 as baseline)
    # lang2 effective = 500MB / 1.0 = 500MB  (bp is lang1 relative to lang2)
    # lang1 effective = 500MB / bp
    effective_l1 = TOTAL_BYTES / bp      # in lang2-equivalent bytes
    effective_l2 = TOTAL_BYTES           # lang2 is the baseline

    target_effective = min(effective_l1, effective_l2)

    # -- raw bytes to read from each file to hit target_effective
    raw_l1 = target_effective * bp       # un-scale back to lang1 raw bytes
    raw_l2 = target_effective            # lang2 raw bytes == effective bytes

    print(f"{lang1} × {lang2}")
    print(f"  byte premium (l1/l2): {bp:.4f}")
    print(f"  effective @ 500MB:  l1={effective_l1/1e6:.1f}MB  l2={effective_l2/1e6:.1f}MB")
    print(f"  target effective:   {target_effective/1e6:.1f}MB")
    print(f"  raw bytes to read:  l1={raw_l1/1e6:.1f}MB  l2={raw_l2/1e6:.1f}MB")

    # -- read
    lines1, got1 = read_bytes(f1, int(raw_l1))
    lines2, got2 = read_bytes(f2, int(raw_l2))
    print(f"  actually read:      l1={got1/1e6:.1f}MB  l2={got2/1e6:.1f}MB")

    # -- shuffle and write
    mixed = lines1 + lines2
    random.shuffle(mixed)

    with open(out_path, 'w', encoding='utf-8') as f:
        for line in mixed:
            f.write(normalize(line))

    print(f"  → written: {out_path}  ({(got1+got2)/1e6:.1f}MB total)\n")

    results.append({
        'lang1': lang1, 'lang2': lang2,
        'byte_premium': bp,
        'effective_mb': target_effective / 1e6,
        'raw_mb_l1': got1 / 1e6,
        'raw_mb_l2': got2 / 1e6,
        'output': out_path,
    })

summary = pd.DataFrame(results)
summary.to_csv('balanced_data_summary.csv', index=False)
print(summary)
