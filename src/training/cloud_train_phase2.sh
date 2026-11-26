#!/bin/bash
# ============================================================
#  MI300X Phase 2 Cloud GPU Mission:
#  3-Class Pomegranate Fruit + Orchard Trunk Base Training
# ============================================================
set -e
cd ~/physical-ai-robotics-venture

if [ ! -d "$HOME/pom_train_env" ]; then
    bash src/training/cloud_setup.sh
fi
source "$HOME/pom_train_env/bin/activate"

mkdir -p data/raw_phase2 models/phase2

echo "[1/5] Downloading PG-YOLO-Dataset (13,840 pomegranate fruit images)..."
if [ ! -d "data/raw_phase2/PG-YOLO-Dataset" ]; then
    git clone --depth 1 https://github.com/LforikC/PG-YOLO-Dataset.git data/raw_phase2/PG-YOLO-Dataset
fi

echo "[2/5] Downloading Figshare Orchard Tree Trunk Dataset (1,756 images)..."
if [ ! -d "data/raw_phase2/trunks" ]; then
    curl -L -o data/raw_phase2/ovosad_trunks.zip "https://ndownloader.figshare.com/files/43722933"
    unzip -q data/raw_phase2/ovosad_trunks.zip -d data/raw_phase2/trunks
fi

echo "[3/5] Merging into Phase 2 multi-class dataset (data/pomegranate_phase2)..."
python src/training/prepare_phase2_trunk_weed.py --output data/pomegranate_phase2 --merge-folder data/raw_phase2/PG-YOLO-Dataset --target-class 0 --prefix pg
python src/training/prepare_phase2_trunk_weed.py --output data/pomegranate_phase2 --merge-folder data/raw_phase2/trunks --target-class 1 --prefix trunk

echo "[4/5] Training Phase 2 Multi-Class YOLOv8s on MI300X (80 epochs, batch=128)..."
python src/training/train_yolo.py --model yolov8s --data data/pomegranate_phase2/data_phase2.yaml --epochs 80 --imgsz 640 --workers 16 --project runs/phase2

BEST_P2=$(find runs -path "*phase2*" -name "best.pt" -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
cp "$BEST_P2" models/phase2/best_phase2.pt
python src/training/evaluate_and_export.py --weights models/phase2/best_phase2.pt --data data/pomegranate_phase2/data_phase2.yaml --benchmark --export onnx --output models/phase2 --report models/phase2/evaluation_report_phase2.json

echo "[5/5] Phase 2 Complete! Ready to push to HuggingFace."
