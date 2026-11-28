#!/usr/bin/env bash
# Session C — System 2 (Qwen2.5-3B / 1.5B LoRA fine-tune + vLLM serving)
# Usage: bash session_c_system2.sh [3B|1.5B]
set -euo pipefail

SIZE="${1:-3B}"
export RUN="/scratch/runs/$(date +%Y%m%d)_sessionC_system2_${SIZE}"
mkdir -p "$RUN"/{data,out,logs}

echo "=== Session C: System 2 ($SIZE) ==="
pip freeze > "$RUN/out/env.txt"

if [ "$SIZE" = "1.5B" ]; then
  BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
else
  BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
fi

echo "Base model: $BASE_MODEL"
if [ -f "src/training/train_slm.py" ]; then
  python3 src/training/train_slm.py --base_model "$BASE_MODEL" --output_dir "$RUN/out/slm_lora"
else
  echo "[WARN] src/training/train_slm.py not found in CWD."
fi

if [ -f "src/training/verify_slm_cloud.py" ]; then
  python3 src/training/verify_slm_cloud.py --model_dir "$RUN/out/slm_lora" | tee "$RUN/logs/slm_eval.log"
fi

echo "Session C complete. To serve via vLLM ROCm:"
echo "  docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -p 8001:8000 rocm/vllm:latest --model $RUN/out/slm_lora"
