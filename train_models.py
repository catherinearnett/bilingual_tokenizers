"""
Prepare and launch Goldfish-style GPT training for custom bilingual pairs.

Pipeline per model:
  1. Download HF tokenizer from catherinearnett/bilingual_tokenizers2
  2. Tokenize full training data (custom_training_data/{lang1}_{lang2}_bp_balanced.txt)
  3. Generate and run training shell script (goldfish hyperparameters)

Two models per row in extreme_pairs_with_crossing.csv:
  - custom-mix tokenizer: custom_mix/{lang1}_{lang2}_{p1}_{p2}_{tok_type}_{whitespace}_{vocab_size}
  - 50/50 tokenizer:      {lang1}_{lang2}_50_50_{tok_type}_{whitespace}_{vocab_size}  (root level)

Tokenizers are in catherinearnett/bilingual_tokenizers2:
  custom_mix/  → custom crossing-proportion tokenizers
  root level   → standard bilingual tokenizers (50/50 is one of these)
"""

import os
import codecs
import subprocess
import pandas as pd
from huggingface_hub import hf_hub_download
import shutil

# ── config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
GOLDFISH_DIR   = os.path.join(SCRIPT_DIR, 'goldfish')
WALM_DIR       = os.path.join(GOLDFISH_DIR, 'word-acquisition-language-models')
TRAINING_DATA  = os.path.join(SCRIPT_DIR, 'custom_training_data')
TOKENIZERS_DIR = os.path.join(SCRIPT_DIR, 'custom_tokenizers')
TOKENIZED_DIR  = os.path.join(SCRIPT_DIR, 'custom_tokenized_data')
MODELS_DIR     = os.path.join(SCRIPT_DIR, 'custom_models')
SCRIPTS_DIR    = os.path.join(SCRIPT_DIR, 'custom_training_scripts')
HF_REPO        = 'catherinearnett/bilingual_tokenizers2'
CONFIG_PATH    = os.path.join(GOLDFISH_DIR, 'training_code', 'gpt_base_config.json')

# Goldfish hyperparameters (1000mb scale)
WARMUP_PROPORTION = 0.10
EPOCHS            = 10
LEARNING_RATE     = 0.0001
BATCH_SIZE        = 64
MAX_BATCH_PER_DEV = 8
TOKENS_PER_SEQ    = 512

for d in [TOKENIZERS_DIR, TOKENIZED_DIR, MODELS_DIR, SCRIPTS_DIR]:
    os.makedirs(d, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def download_tokenizer(hf_path, local_dir):
    """Download tokenizer.json from HF repo to local_dir."""
    if os.path.exists(os.path.join(local_dir, 'tokenizer.json')):
        print(f'  SKIP (already downloaded): {local_dir}')
        return
    os.makedirs(local_dir, exist_ok=True)
    path = hf_hub_download(
        repo_id=HF_REPO,
        filename=f'{hf_path}/tokenizer.json',
        repo_type='dataset',
    )
    shutil.copy(path, os.path.join(local_dir, 'tokenizer.json'))
    print(f'  Downloaded: {hf_path} → {local_dir}')


def tokenize_dataset(tokenizer_dir, input_file, output_file):
    """Run tokenize_dataset.py on the full dataset (no example cap)."""
    if os.path.exists(output_file):
        print(f'  SKIP (already tokenized): {output_file}')
        return
    cmd = (
        f'python3 {WALM_DIR}/scripts/tokenize_dataset.py '
        f'--tokenizer={tokenizer_dir} '
        f'--input_file={input_file} '
        f'--output_file={output_file} '
        f'--max_segments=-1 --max_seq_len={TOKENS_PER_SEQ}'
    )
    print(f'  Tokenizing → {output_file}')
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'  ERROR: {result.stderr}')
    else:
        print(f'  Done.')


def count_tokens(tokenized_file):
    """Count lines in tokenized file — each line is one 512-token example."""
    with open(tokenized_file) as f:
        return sum(1 for _ in f)


def write_training_script(stem, tokenizer_dir, tokenized_train, n_examples, model_outdir):
    """Write a goldfish-style bash training script with steps scaled to full data."""
    batch_per_device = min(BATCH_SIZE, MAX_BATCH_PER_DEV)
    grad_accum       = BATCH_SIZE // batch_per_device

    epoch_steps  = n_examples / BATCH_SIZE
    max_steps    = int(EPOCHS * epoch_steps)
    warmup_steps = int(max_steps * WARMUP_PROPORTION)
    save_steps   = int(epoch_steps / 2)

    model_bin = os.path.join(model_outdir, 'pytorch_model.bin')

    script = f"""#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# {stem}
if test -f {model_bin}; then
echo "Model already found: {stem}."
else
rm -rf {model_outdir}
mkdir -p {model_outdir}
python3 {WALM_DIR}/lm_code/run_transformer_language_modeling.py \\
--tokenizer_name={tokenizer_dir} \\
--config_name={CONFIG_PATH} \\
--do_train --train_iterable \\
--per_device_train_batch_size={batch_per_device} \\
--gradient_accumulation_steps={grad_accum} \\
--eval_strategy=no --save_strategy=steps \\
--save_steps={save_steps} \\
--max_steps={max_steps} \\
--warmup_steps={warmup_steps} \\
--learning_rate={LEARNING_RATE} --adam_epsilon=1e-6 --weight_decay=0.01 \\
--train_data_file={tokenized_train} \\
--seed=43 \\
--override_n_examples={n_examples} \\
--output_dir={model_outdir}
cp {tokenizer_dir}/* {model_outdir}
fi
"""
    script_path = os.path.join(SCRIPTS_DIR, f'train_{stem}.sh')
    with codecs.open(script_path, 'w', encoding='utf-8') as f:
        f.write(script)
    os.chmod(script_path, 0o755)
    return script_path


# ── main ──────────────────────────────────────────────────────────────────────

df = pd.read_csv('extreme_pairs_with_crossing.csv').dropna(subset=['crossing_proportion'])

processed_tokenizations = set()

for _, row in df.iterrows():
    lang1      = row['lang1']
    lang2      = row['lang2']
    tok_type   = row['tok_type']
    whitespace = row['whitespace']
    vocab_size = int(row['vocab_size'])
    p1         = round(row['crossing_proportion'])
    p2         = 100 - p1

    train_file = os.path.join(TRAINING_DATA, f'{lang1}_{lang2}_bp_balanced.txt')
    if not os.path.exists(train_file):
        print(f'SKIP (no training file): {lang1}_{lang2}')
        continue

    for mix, p1_tok, p2_tok in [
        ('custom', p1,  p2),
        ('50_50',  50,  50),
    ]:
        stem           = f'{lang1}_{lang2}_{p1_tok}_{p2_tok}_{tok_type}_{whitespace}_{vocab_size}'
        hf_path        = f'custom_mix/{stem}' if mix == 'custom' else stem
        tokenizer_dir  = os.path.join(TOKENIZERS_DIR, stem)
        tokenized_file = os.path.join(TOKENIZED_DIR, f'{stem}.txt')
        model_outdir   = os.path.join(MODELS_DIR, stem)

        print(f'\n[{mix}] {stem}')

        # 1. download tokenizer
        try:
            download_tokenizer(hf_path, tokenizer_dir)
        except Exception as e:
            print(f'  FAILED download: {e}')
            continue

        # 2. tokenize full dataset
        tok_key = (stem, train_file)
        if tok_key not in processed_tokenizations:
            tokenize_dataset(tokenizer_dir, train_file, tokenized_file)
            processed_tokenizations.add(tok_key)

        if not os.path.exists(tokenized_file):
            print(f'  SKIP training (no tokenized file)')
            continue

        # 3. count examples and compute per-model steps
        n_examples = count_tokens(tokenized_file)
        print(f'  n_examples: {n_examples}')

        # 4. write training script
        os.makedirs(model_outdir, exist_ok=True)
        script_path = write_training_script(
            stem, tokenizer_dir, tokenized_file, n_examples, model_outdir
        )
        print(f'  Script written: {script_path}')

# write master script
all_scripts = sorted(f for f in os.listdir(SCRIPTS_DIR) if f.startswith('train_'))
master_path = os.path.join(SCRIPTS_DIR, 'run_all.sh')
with open(master_path, 'w') as f:
    f.write('#!/bin/bash\n')
    for s in all_scripts:
        f.write(f'bash {os.path.join(SCRIPTS_DIR, s)}\n')
os.chmod(master_path, 0o755)

print(f'\nDone. Run all with:\n  bash {master_path}')
