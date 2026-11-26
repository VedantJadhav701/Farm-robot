#!/bin/bash
# ============================================================
#  MI300X End-to-End Orchestrator (System 1 + System 2)
#  Run inside tmux on your MI300X cloud instance:
#    tmux new -s train
#    export ROBOFLOW_API_KEY="your_key"
#    bash src/training/cloud_run_all.sh
# ============================================================
set -e

echo "╔══════════════════════════════════════════════════════╗"
echo "║  MI300X FULL TRAINING: SYSTEM 1 (YOLO) + SYSTEM 2    ║"
echo "╚══════════════════════════════════════════════════════╝"

# 1. Setup environment if not already done
if [ ! -d "$HOME/pom_train_env" ]; then
    bash src/training/cloud_setup.sh
fi
source "$HOME/pom_train_env/bin/activate"

# Ensure SLM fine-tuning deps are present
pip install peft datasets -q

# 2. Download & curate pomegranate datasets
if [ -z "$ROBOFLOW_API_KEY" ]; then
    echo "[!] ROBOFLOW_API_KEY is not set."
    echo "    Export it first: export ROBOFLOW_API_KEY='your_key'"
    exit 1
fi
python src/training/download_dataset.py --presets all --api-key "$ROBOFLOW_API_KEY"

# 3. Train System 1 CNN (YOLOv8s, 300 epochs, MI300X auto-batch=128)
python src/training/train_yolo.py --model yolov8s --epochs 300 --imgsz 640

# 4. Evaluate & Export best System 1 weights to ONNX
BEST_PT=$(ls -t runs/pomegranate/*/weights/best.pt | head -n 1)
mkdir -p models
cp "$BEST_PT" models/best.pt
python src/training/evaluate_and_export.py --weights models/best.pt --benchmark --export onnx

# 5. Train System 2 SLM (Qwen2.5-3B-Instruct LoRA on Agricultural Safety & Dosing)
python src/training/train_slm.py --model Qwen/Qwen2.5-3B-Instruct --epochs 3 --batch-size 16 --output models/slm_lora

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  ALL TRAINING COMPLETE!                              ║"
echo "║  Artifacts ready in models/:                         ║"
echo "║   - models/best.pt  (System 1 YOLOv8s)               ║"
echo "║   - models/best.onnx                                 ║"
echo "║   - models/slm_lora/ (System 2 Qwen2.5-3B)           ║"
echo "╚══════════════════════════════════════════════════════╝"
