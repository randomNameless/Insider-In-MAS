#!/usr/bin/env bash
# One-shot setup on a fresh Lambda Cloud instance (Ubuntu + CUDA preinstalled).
set -euo pipefail

cd "$(dirname "$0")/.."

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo
echo "Sanity: CPU-only unit tests (no GPU / no model download needed)"
python -m pytest -q

echo
echo "Next:"
echo "  1) huggingface-cli login          # if models are gated"
echo "  2) SFT the actor EXTERNALLY (e.g. LlamaFactory) -> a LoRA adapter dir"
echo "     data: attacker chat (malicious system + multiround user) -> assistant"
echo "           'Message: <text>\\nPosition: <int>', enable_thinking=False"
echo "  3) python scripts/preflight.py    --config configs/prototype.yaml --sft-adapter <sft_dir>"
echo "  4) python scripts/run_train.py    --config configs/prototype.yaml --sft-adapter <sft_dir>"
