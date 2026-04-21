# bilingual_tokenizers
Bilingual tokenizers


## Server Setup

```
sudo apt update && sudo apt install vim && sudo apt install curl

sudo apt install tmux

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"
uv --version

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
/root/global_piqa/test/bin/python get-pip.py
```

Then create the environment

```
# PyTorch first (needs the special CUDA index URL)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Everything else
uv pip install -r requirements.txt
```
