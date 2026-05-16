import os
import re
import pandas as pd
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from datasets import load_dataset


# --- Config ---
FLORES_SPLIT = "devtest"
CUSTOM_MODELS_DIR = "/mnt/ssd-3/catherine/bilingual_tokenizers/matched_compression/custom_models"
CSV_PATH = "flores_ctc_results.csv"

# FLORES language code mapping for languages that appear under a different code
# (same as original script — used if a lang code isn't found directly in flores_plus)
FLORES_LANG_MAPPING = {
    'ace_Arab': 'urd_Arab', 'acm_Arab': 'arb_Arab', 'acq_Arab': 'arb_Arab',
    'aeb_Arab': 'arb_Arab', 'ajp_Arab': 'arb_Arab', 'als_Latn': 'sqi_Latn',
    'arb_Latn': 'mlt_Latn', 'ars_Arab': 'arb_Arab', 'ary_Arab': 'arb_Arab',
    'awa_Deva': 'hin_Deva', 'ayr_Latn': 'aym_Latn', 'azb_Arab': 'aze_Arab',
    'azj_Latn': 'aze_Latn', 'bjn_Arab': 'urd_Arab', 'dik_Latn': 'din_Latn',
    'gaz_Latn': 'orm_Latn', 'kam_Latn': 'kik_Latn', 'kas_Arab': 'urd_Arab',
    'khk_Cyrl': 'mon_Cyrl', 'kmr_Latn': 'kur_Latn', 'lvs_Latn': 'lav_Latn',
    'min_Arab': 'urd_Arab', 'mni_Beng': 'ben_Beng', 'npi_Deva': 'nep_Deva',
    'nus_Latn': 'din_Latn', 'ory_Orya': 'ori_Orya', 'pbt_Arab': 'pus_Arab',
    'plt_Latn': 'mlg_Latn', 'quy_Latn': 'que_Latn', 'swh_Latn': 'swa_Latn',
    'taq_Latn': 'kab_Latn', 'taq_Tfng': None, 'tzm_Tfng': None,
    'uzn_Latn': 'uzb_Latn', 'ydd_Hebr': 'yid_Hebr', 'yue_Hant': 'zho_Hant',
    'zsm_Latn': 'msa_Latn',
}


def parse_langs_from_model_name(model_name):
    """
    Extract the two language codes from a model folder name.
    Pattern: {lang1}_{script1}_{lang2}_{script2}_...
    e.g. afr_Latn_als_Latn_50_50_bpe_nowhitespace_16384 -> ('afr_Latn', 'als_Latn')
    """
    # Match two consecutive LLL_SSSS pairs at the start
    m = re.match(r'^([a-z]{2,3}_[A-Z][a-z]{3})_([a-z]{2,3}_[A-Z][a-z]{3})_', model_name)
    if m:
        return m.group(1), m.group(2)
    return None, None


def load_flores_lines(flores_lang_code, split, token):
    """Load FLORES sentences, falling back via FLORES_LANG_MAPPING if needed."""
    # Try direct load first
    try:
        ds = load_dataset(
            "openlanguagedata/flores_plus",
            flores_lang_code,
            split=split,
            token=token,
        )
        return [row['text'] for row in ds]
    except Exception:
        pass

    # Try mapped code
    mapped = FLORES_LANG_MAPPING.get(flores_lang_code)
    if mapped is None:
        print(f"  No FLORES mapping for {flores_lang_code}, skipping.")
        return None
    try:
        ds = load_dataset(
            "openlanguagedata/flores_plus",
            mapped,
            split=split,
            token=token,
        )
        print(f"  Loaded {flores_lang_code} via mapping -> {mapped}")
        return [row['text'] for row in ds]
    except Exception as e:
        print(f"  Could not load FLORES for {flores_lang_code} (or {mapped}): {e}")
        return None


def load_tokenizer(model_dir):
    """Load tokenizer from a local model directory."""
    tokenizer_path = os.path.join(model_dir, "tokenizer.json")
    if not os.path.isfile(tokenizer_path):
        raise FileNotFoundError(f"No tokenizer.json found in {model_dir}")
    base_tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=base_tokenizer)
    return tokenizer


def compute_ctc(tokenizer, lines):
    """Count total tokens across all FLORES lines."""
    total_tokens = 0
    for line in tqdm(lines, leave=False):
        total_tokens += len(tokenizer.encode(line, add_special_tokens=False))
    return total_tokens


def already_done(existing_df, model_name, flores_code):
    if existing_df.empty:
        return False
    mask = (
        (existing_df['model'].str.strip() == model_name.strip()) &
        (existing_df['flores_lang'].str.strip() == flores_code.strip())
    )
    return mask.any()


# --- Main ---
token = os.environ.get("HF_TOKEN_READ")

# Load existing results
if os.path.isfile(CSV_PATH):
    existing_df = pd.read_csv(CSV_PATH)
    existing_df['model'] = existing_df['model'].astype(str)
    existing_df['flores_lang'] = existing_df['flores_lang'].astype(str)
    print(f"Loaded {len(existing_df)} existing rows from {CSV_PATH}")
else:
    existing_df = pd.DataFrame()
    print("No existing CSV found, starting fresh.")

# Discover all model directories
model_names = sorted([
    d for d in os.listdir(CUSTOM_MODELS_DIR)
    if os.path.isdir(os.path.join(CUSTOM_MODELS_DIR, d))
])
print(f"\nFound {len(model_names)} model directories.")

# Collect all unique language codes we'll need
all_lang_codes = set()
for model_name in model_names:
    lang1, lang2 = parse_langs_from_model_name(model_name)
    if lang1:
        all_lang_codes.add(lang1)
    if lang2:
        all_lang_codes.add(lang2)

print(f"Unique language codes across all models: {sorted(all_lang_codes)}")

# Pre-load all FLORES sentence sets
flores_sentences = {}
for code in sorted(all_lang_codes):
    if code in flores_sentences:
        continue
    print(f"Loading FLORES for {code} ...")
    lines = load_flores_lines(code, FLORES_SPLIT, token)
    if lines is not None:
        flores_sentences[code] = lines
        print(f"  -> {len(lines)} sentences")

# Main evaluation loop
new_rows = []

for model_name in model_names:
    model_dir = os.path.join(CUSTOM_MODELS_DIR, model_name)
    lang1, lang2 = parse_langs_from_model_name(model_name)

    if lang1 is None:
        print(f"\nSkipping {model_name} — could not parse language codes.")
        continue

    target_langs = [lang1, lang2]
    needed = [
        code for code in target_langs
        if code in flores_sentences and not already_done(existing_df, model_name, code)
    ]

    if not needed:
        print(f"\nSkipping {model_name} — all languages already computed.")
        continue

    print(f"\nModel: {model_name}  |  langs: {lang1}, {lang2}  |  to evaluate: {needed}")

    try:
        tokenizer = load_tokenizer(model_dir)
    except Exception as e:
        print(f"  Could not load tokenizer: {e}")
        continue

    for flores_code in needed:
        lines = flores_sentences[flores_code]
        print(f"  Computing CTC for {flores_code} ({len(lines)} sentences) ...")
        total_tokens = compute_ctc(tokenizer, lines)
        print(f"  {flores_code}: total_tokens={total_tokens}")
        new_rows.append({
            'model': model_name,
            'flores_lang': flores_code,
            'total_tokens': total_tokens,
        })

# Save
new_df = pd.DataFrame(new_rows)
combined_df = pd.concat([existing_df, new_df], ignore_index=True)
combined_df.to_csv(CSV_PATH, index=False)
print(f"\nSaved {len(combined_df)} total rows to {CSV_PATH}")
print(combined_df)
