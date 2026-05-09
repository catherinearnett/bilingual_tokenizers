"""
Check which HF tokenizers are present/missing from the two converted repos:
  Bilingual   → catherinearnett/bilingual_tokenizers2
  Monolingual → catherinearnett/monolingual_tokenizers

Each tokenizer is a subdirectory containing tokenizer.json.
Ground truth: 796 input files × 12 conditions = 9,552 expected total.

Monolingual format : {lang}_{tok_type}_{whitespace}_{vocab_size}          (5 parts)
Bilingual format   : {lang1}_{lang2}_{p1}_{p2}_{tok_type}_{whitespace}_{vocab_size}  (9 parts)
"""

import os
from huggingface_hub import HfApi
from collections import defaultdict

BI_REPO   = "catherinearnett/bilingual_tokenizers2"
MONO_REPO = "catherinearnett/monolingual_tokenizers"

EXPECTED_PROPORTIONS = {"10_90", "25_75", "50_50", "75_25", "90_10"}

EXPECTED_MONO_KEYS = {
    (tt, ws, vs)
    for tt in ("bpe", "unigram")
    for ws in ("nowhitespace", "whitespace")
    for vs in ("16384", "32768", "65536")
}

TOTAL_INPUT_FILES   = 796
CONDITIONS_PER_FILE = 12
EXPECTED_TOTAL      = TOTAL_INPUT_FILES * CONDITIONS_PER_FILE  # 9,552


# ── fetch stems from a repo (each tokenizer is a top-level subdir) ────────────

def fetch_stems(api, repo_id, token=None):
    """
    List all tokenizer stems from a repo.
    Each tokenizer is stored as {stem}/tokenizer.json — we list top-level
    folders non-recursively to avoid HF pagination 500 errors.
    """
    stems = []
    print(f"  Fetching stems from {repo_id}...")
    try:
        for item in api.list_repo_tree(
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo="",
            recursive=False,
            token=token,
        ):
            # RepoFolder has 'path' but not 'rfilename'
            if not hasattr(item, "rfilename") and hasattr(item, "path"):
                stems.append(item.path)
    except Exception as e:
        print(f"  WARNING: could not fetch {repo_id}: {e}")
    print(f"  Found {len(stems)} tokenizers\n")
    return stems


# ── parsers ───────────────────────────────────────────────────────────────────

def parse_stem(stem):
    parts = stem.split("_")
    if len(parts) == 5 and parts[2] in ("bpe", "unigram"):
        return {
            "kind":       "mono",
            "lang":       "_".join(parts[0:2]),
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


# ── duplicate helpers ─────────────────────────────────────────────────────────

def find_mono_duplicates(mono_parsed):
    slots = defaultdict(list)
    for p in mono_parsed:
        slots[(p["lang"], p["tok_type"], p["whitespace"], p["vocab_size"])].append(p["stem"])
    return {s: fns for s, fns in slots.items() if len(fns) > 1}, slots


def find_bi_duplicates(bi_parsed):
    slots = defaultdict(list)
    for p in bi_parsed:
        slots[(p["lang1"], p["lang2"], p["tok_type"], p["whitespace"],
               p["vocab_size"], p["proportion"])].append(p["stem"])
    return {s: fns for s, fns in slots.items() if len(fns) > 1}, slots


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    token = os.environ.get("HF_TOKEN_READ")
    api   = HfApi()

    print("Fetching tokenizer lists from HuggingFace...\n")
    bi_stems   = fetch_stems(api, BI_REPO,   token)
    mono_stems = fetch_stems(api, MONO_REPO, token)

    # Parse stems
    mono_parsed, bi_parsed, unparseable = [], [], []
    for stem in mono_stems:
        info = parse_stem(stem)
        if info:
            info["stem"] = stem
            mono_parsed.append(info)
        else:
            unparseable.append(f"[mono] {stem}")
    for stem in bi_stems:
        info = parse_stem(stem)
        if info:
            info["stem"] = stem
            bi_parsed.append(info)
        else:
            unparseable.append(f"[bi]   {stem}")

    if unparseable:
        print(f"WARNING: Could not parse {len(unparseable)} stems:")
        for u in unparseable:
            print(f"  {u}")
        print()

    print(f"Monolingual tokenizers : {len(mono_parsed)}")
    print(f"Bilingual  tokenizers  : {len(bi_parsed)}")
    print(f"Unparseable            : {len(unparseable)}\n")

    # ════════════════════════════════════════════════════════════════════════
    # MONOLINGUAL
    # ════════════════════════════════════════════════════════════════════════

    mono_dups, mono_slots = find_mono_duplicates(mono_parsed)

    print("=" * 65)
    print("MONOLINGUAL DUPLICATE CHECK")
    print("=" * 65)
    if mono_dups:
        extra_mono = sum(len(v) - 1 for v in mono_dups.values())
        print(f"  Duplicate slots      : {len(mono_dups)}")
        print(f"  Extra/redundant files: {extra_mono}")
        for slot, stems in sorted(mono_dups.items()):
            lang, tok_type, whitespace, vocab_size = slot
            print(f"\n  Language  : {lang}")
            print(f"  Condition : {tok_type} | {whitespace} | vocab {vocab_size}")
            for s in stems:
                print(f"    - {s}")
    else:
        extra_mono = 0
        print("  No monolingual duplicates found.")

    mono_groups = defaultdict(set)
    for slot in mono_slots:
        lang, tok_type, whitespace, vocab_size = slot
        mono_groups[lang].add((tok_type, whitespace, vocab_size))

    mono_complete   = {l: v for l, v in mono_groups.items() if v == EXPECTED_MONO_KEYS}
    mono_incomplete = {l: v for l, v in mono_groups.items() if v != EXPECTED_MONO_KEYS}

    print(f"\n{'='*65}")
    print(f"MONOLINGUAL LANGUAGES ({len(mono_groups)} unique languages)")
    print(f"{'='*65}")
    for lang, conditions in sorted(mono_groups.items()):
        n    = len(conditions)
        flag = " ✓" if lang in mono_complete else f"  {n}/12"
        print(f"  {lang}: {n} condition(s){flag}")

    print(f"\n{'='*65}")
    print("MONOLINGUAL COMPLETENESS CHECK")
    print(f"Expected 12 conditions per language: "
          "{bpe,unigram} x {nowhitespace,whitespace} x {16384,32768,65536}")
    print(f"{'='*65}")
    print(f"  Complete languages   : {len(mono_complete)}")
    print(f"  Incomplete languages : {len(mono_incomplete)}")

    if mono_incomplete:
        print(f"\nINCOMPLETE MONOLINGUAL LANGUAGES:")
        print("-" * 65)
        for lang, found in sorted(mono_incomplete.items()):
            missing = sorted(EXPECTED_MONO_KEYS - found)
            print(f"\n  Language : {lang}")
            print(f"  Found    : {sorted(found)}")
            print(f"  Missing  : {missing}")
    else:
        print("\n  All monolingual languages are complete!")

    # ════════════════════════════════════════════════════════════════════════
    # BILINGUAL
    # ════════════════════════════════════════════════════════════════════════

    bi_dups, bi_slots = find_bi_duplicates(bi_parsed)

    print(f"\n{'='*65}")
    print("BILINGUAL DUPLICATE CHECK")
    print(f"{'='*65}")
    if bi_dups:
        extra_bi = sum(len(v) - 1 for v in bi_dups.values())
        print(f"  Duplicate slots      : {len(bi_dups)}")
        print(f"  Extra/redundant files: {extra_bi}")
        for slot, stems in sorted(bi_dups.items()):
            lang1, lang2, tok_type, whitespace, vocab_size, proportion = slot
            print(f"\n  Pair      : {lang1}  <->  {lang2}")
            print(f"  Condition : {tok_type} | {whitespace} | vocab {vocab_size} | prop {proportion}")
            for s in stems:
                print(f"    - {s}")
    else:
        extra_bi = 0
        print("  No bilingual duplicates found.")

    bi_groups = defaultdict(set)
    for slot in bi_slots:
        lang1, lang2, tok_type, whitespace, vocab_size, proportion = slot
        bi_groups[(lang1, lang2, tok_type, whitespace, vocab_size)].add(proportion)

    pair_counts = defaultdict(int)
    for p in bi_parsed:
        pair_counts[(p["lang1"], p["lang2"])] += 1

    bi_complete   = {k: v for k, v in bi_groups.items() if v == EXPECTED_PROPORTIONS}
    bi_incomplete = {k: v for k, v in bi_groups.items() if v != EXPECTED_PROPORTIONS}

    print(f"\n{'='*65}")
    print(f"BILINGUAL LANGUAGE PAIRS ({len(pair_counts)} unique pairs)")
    print(f"{'='*65}")
    for pair, count in sorted(pair_counts.items()):
        print(f"  {pair[0]}  <->  {pair[1]}: {count} tokenizer(s)")

    print(f"\n{'='*65}")
    print("BILINGUAL COMPLETENESS CHECK")
    print(f"Expected proportions: {sorted(EXPECTED_PROPORTIONS)}")
    print(f"{'='*65}")
    print(f"  Condition groups total : {len(bi_groups)}")
    print(f"  Complete groups        : {len(bi_complete)}")
    print(f"  Incomplete groups      : {len(bi_incomplete)}")

    if bi_incomplete:
        print(f"\nINCOMPLETE BILINGUAL SETS:")
        print("-" * 65)
        for key, found_props in sorted(bi_incomplete.items()):
            lang1, lang2, tok_type, whitespace, vocab_size = key
            missing = sorted(EXPECTED_PROPORTIONS - found_props)
            found   = sorted(found_props)
            print(f"\n  Pair      : {lang1}  <->  {lang2}")
            print(f"  Condition : {tok_type} | {whitespace} | vocab {vocab_size}")
            print(f"  Found     : {found}")
            print(f"  Missing   : {missing}")
    else:
        print("\n  All bilingual condition groups are complete!")

    # ════════════════════════════════════════════════════════════════════════
    # OVERALL SUMMARY
    # ════════════════════════════════════════════════════════════════════════

    n_mono_langs  = len(mono_groups)
    expected_mono = n_mono_langs * CONDITIONS_PER_FILE
    unique_mono   = sum(len(v) for v in mono_groups.values())
    missing_mono  = sum(len(EXPECTED_MONO_KEYS - v) for v in mono_incomplete.values())

    n_bi_pairs    = len(pair_counts)
    expected_bi   = EXPECTED_TOTAL - expected_mono
    unique_bi     = len(bi_slots)
    missing_bi    = sum(len(EXPECTED_PROPORTIONS - v) for v in bi_incomplete.values())

    print(f"\n{'='*65}")
    print("OVERALL SUMMARY")
    print(f"{'='*65}")
    print(f"  Ground truth: {TOTAL_INPUT_FILES} input files × {CONDITIONS_PER_FILE}"
          f" conditions = {EXPECTED_TOTAL} expected total\n")
    print(f"  MONOLINGUAL ({n_mono_langs} languages, repo: {MONO_REPO})")
    print(f"    Expected  : {expected_mono}")
    print(f"    Found     : {len(mono_parsed)} raw  ({unique_mono} unique, {extra_mono} duplicates)")
    print(f"    Missing   : {missing_mono}")
    print()
    print(f"  BILINGUAL ({n_bi_pairs} pairs, repo: {BI_REPO})")
    print(f"    Expected  : {expected_bi}  (= {EXPECTED_TOTAL} - {expected_mono})")
    print(f"    Found     : {len(bi_parsed)} raw  ({unique_bi} unique, {extra_bi} duplicates)")
    print(f"    Missing   : {missing_bi}")
    print()
    print(f"  TOTAL")
    print(f"    Expected  : {EXPECTED_TOTAL}")
    print(f"    Found     : {len(mono_parsed) + len(bi_parsed)} raw  "
          f"({unique_mono + unique_bi} unique, {extra_mono + extra_bi} duplicates)")
    print(f"    Missing   : {missing_mono + missing_bi}")


if __name__ == "__main__":
    main()
