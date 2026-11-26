#!/bin/bash
set -e
cd ~/physical-ai-robotics-venture
source ~/pom_train_env/bin/activate

echo "[1/4] Ensuring accelerate==1.2.1 (preserving torch+rocm6.2)..."
pip install accelerate==1.2.1 --no-deps -q

echo "[2/4] Updating data.yaml for PG-YOLO-Dataset (13,840 images)..."
cat > data/pomegranate_dataset/data.yaml << 'EOF'
path: /root/physical-ai-robotics-venture/data/pomegranate_dataset
train: train/images
val: valid/images
nc: 1
names:
  0: pomegranate_fruit
EOF

echo "[3/4] Training System 1 YOLOv8s on MI300X (100 epochs, auto batch=128)..."
python src/training/train_yolo.py --model yolov8s --epochs 100 --imgsz 640 --workers 16

BEST_PT=$(ls -t runs/pomegranate/*/weights/best.pt | head -n 1)
mkdir -p models
cp "$BEST_PT" models/best.pt
python src/training/evaluate_and_export.py --weights models/best.pt --benchmark --export onnx

echo "[4/4] Training System 2 SLM (Qwen2.5-3B-Instruct LoRA)..."
python src/training/train_slm.py --model Qwen/Qwen2.5-3B-Instruct --epochs 3 --batch-size 16 --output models/slm_lora

echo "=== ALL DONE ==="
