#!/bin/bash
# ============================================================
#  MI300X Cloud GPU Environment Setup
#  Target: AMD Instinct MI300X · 192 GB VRAM · ROCm
#  Run ONCE after SSH into the cloud instance
# ============================================================
set -e

echo "╔══════════════════════════════════════════════════════╗"
echo "║   Pomegranate Detection — MI300X Training Setup      ║"
echo "╚══════════════════════════════════════════════════════╝"

# ── 1. Detect GPU backend ──────────────────────────────────
if command -v rocm-smi &>/dev/null; then
    echo "[✓] ROCm detected"
    rocm-smi --showproductname 2>/dev/null || true
    PYTORCH_INDEX="https://download.pytorch.org/whl/rocm6.2"
    GPU_BACKEND="rocm"
elif command -v nvidia-smi &>/dev/null; then
    echo "[✓] NVIDIA CUDA detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    PYTORCH_INDEX="https://download.pytorch.org/whl/cu121"
    GPU_BACKEND="cuda"
else
    echo "[!] No GPU runtime detected — training will be slow"
    PYTORCH_INDEX="https://download.pytorch.org/whl/cpu"
    GPU_BACKEND="cpu"
fi

# ── 2. Create venv (or reuse) ──────────────────────────────
VENV_DIR="$HOME/pom_train_env"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "[✓] Created venv at $VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# ── 3. Install PyTorch + training stack ────────────────────
pip install --upgrade pip setuptools wheel -q
pip install torch torchvision torchaudio --index-url "$PYTORCH_INDEX" -q
pip install ultralytics>=8.4.0 -q
pip install roboflow -q
pip install albumentations>=1.4.0 -q
pip install onnx onnxruntime -q
pip install pydantic>=2.0 -q
pip install peft datasets accelerate==1.2.1 -q
# Re-pin PyTorch ROCm/CUDA after pip deps so PyPI does not overwrite it
pip install --force-reinstall torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url "$PYTORCH_INDEX" -q

# Optional: Weights & Biases for experiment tracking
# pip install wandb -q

# ── 4. Verify ──────────────────────────────────────────────
python3 - <<'PYEOF'
import torch, ultralytics, sys
print(f"Python  : {sys.version.split()[0]}")
print(f"PyTorch : {torch.__version__}")
print(f"YOLO    : {ultralytics.__version__}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}   : {props.name}  |  {props.total_memory/1024**3:.1f} GB VRAM")
else:
    print("GPU     : NONE (CPU mode)")
PYEOF

# ── 5. Create scratch workspace on NVMe ────────────────────
SCRATCH="/scratch/pomegranate_training"
mkdir -p "$SCRATCH"/{datasets,runs,exports}
echo "[✓] Scratch workspace: $SCRATCH"

echo ""
echo "=== SETUP COMPLETE ==="
echo "Activate with:  source $VENV_DIR/bin/activate"
echo "Next steps:"
echo "  1. cd /path/to/physical-ai-robotics-venture"
echo "  2. python src/training/download_dataset.py --api-key YOUR_ROBOFLOW_KEY"
echo "  3. python src/training/train_yolo.py --model yolov8s --epochs 300"
echo ""
