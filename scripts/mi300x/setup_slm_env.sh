#!/usr/bin/env bash
set -euo pipefail

echo "=== Installing SLM & Evaluation Dependencies inside rocm ==="
pip install --no-cache-dir --upgrade-strategy only-if-needed \
  transformers accelerate peft datasets sentencepiece protobuf huggingface_hub safetensors

python3 - << 'PYEOF'
import torch, transformers, peft, datasets, accelerate
print("PyTorch:", torch.__version__, "CUDA/HIP:", torch.cuda.is_available())
print("Transformers:", transformers.__version__)
print("PEFT:", peft.__version__)
print("Datasets:", datasets.__version__)
print("Accelerate:", accelerate.__version__)
print("ENV_SETUP_OK")
PYEOF
