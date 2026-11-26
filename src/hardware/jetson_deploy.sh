#!/bin/bash
# ============================================================
#  Jetson Orin (Nano / NX) Edge Deployment Setup
#  Pulls trained weights from HuggingFace & compiles TensorRT FP16
# ============================================================
set -e

echo "=== Jetson Orin Edge Deployment Setup ==="
mkdir -p models

# 1. Download verified ONNX & PyTorch weights from HuggingFace
python3 - <<'PYEOF'
from huggingface_hub import hf_hub_download
for fn in ["system1_yolo/best.onnx", "system1_yolo/best.pt"]:
    p = hf_hub_download("vedantjadhav701/pomegranate-dual-brain-ai", fn, local_dir="models")
    print("Downloaded:", p)
PYEOF
cp -f models/system1_yolo/best.onnx models/best.onnx
cp -f models/system1_yolo/best.pt models/best.pt

# 2. Convert ONNX -> TensorRT FP16 engine on Jetson Orin
if command -v /usr/src/tensorrt/bin/trtexec &>/dev/null; then
    /usr/src/tensorrt/bin/trtexec \
        --onnx=models/best.onnx \
        --saveEngine=models/best_fp16.engine \
        --fp16 \
        --workspace=2048
    echo "[✓] TensorRT FP16 engine built: models/best_fp16.engine"
else
    echo "[!] trtexec not found — using models/best.pt or models/best.onnx"
fi

# 3. Ensure System 2 SLM is ready in Ollama
if command -v ollama &>/dev/null; then
    ollama pull qwen2.5:3b
fi

echo "=== Launch Edge Server ==="
echo "PYTHONPATH=. uvicorn src.api.mission_server:app --host 0.0.0.0 --port 8000"
