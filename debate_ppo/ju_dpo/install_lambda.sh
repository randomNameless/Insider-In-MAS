#!/usr/bin/env bash
# Ju et al. DPO training venv (2026-10-22). Their requirements.txt pins trl==0.17.0, transformers==4.51.3 (kept);
# peft 0.9.0 -> 0.15.2 (trl 0.17 needs a newer peft API), torch 2.2.2 -> 2.6.0 (cu124 wheel for the H100 driver).
# Separate from .venv so the vLLM / transformers 5.8 environment is not touched.
set -euo pipefail
V=$HOME/venvs/ju-dpo
python3 -m venv $V
source $V/bin/activate
pip install -q --upgrade pip
pip install -q torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -q transformers==4.51.3 trl==0.17.0 peft==0.15.2 accelerate==1.6.0 datasets==3.5.0
python -c "import torch, transformers, trl, peft; print(torch.__version__, transformers.__version__, trl.__version__, peft.__version__, torch.cuda.is_available())"
