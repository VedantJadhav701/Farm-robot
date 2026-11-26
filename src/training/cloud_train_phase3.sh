#!/bin/bash
# ============================================================
#  MI300X Phase 3 Teacher-Student Mission:
#  Grounding-DINO-Base Teacher -> YOLOv8m 3-Class Student
# ============================================================
set -e
cd ~/physical-ai-robotics-venture

if [ ! -d "$HOME/pom_train_env" ]; then
    bash src/training/cloud_setup.sh
fi
source "$HOME/pom_train_env/bin/activate"
pip install transformers==4.47.1 yt-dlp -q

mkdir -p data/real_test_samples models/phase3

echo "[1/5] Downloading Turkey pomegranate orchard video (480p)..."
if [ ! -f "data/real_test_samples/orchard_turkey_480p.mp4" ]; then
    python -m yt_dlp -f 135 -o data/real_test_samples/orchard_turkey_480p.mp4 "https://youtu.be/Bq7UonY8_u4"
fi

echo "[2/5] Running Grounding-DINO Teacher Auto-Labeling (600 orchard frames)..."
python src/training/phase3_teacher_autolabel.py \
    --video data/real_test_samples/orchard_turkey_480p.mp4 \
    --output data/pomegranate_phase3 \
    --stride-sec 1.5 --max-frames 600

echo "[3/5] Merging PG-YOLO-Dataset (13,840 pomegranate fruit images)..."
if [ ! -d "data/raw_phase3/PG-YOLO-Dataset" ]; then
    git clone --depth 1 https://github.com/LforikC/PG-YOLO-Dataset.git data/raw_phase3/PG-YOLO-Dataset
fi
python src/training/prepare_phase2_trunk_weed.py \
    --output data/pomegranate_phase3 \
    --merge-folder data/raw_phase3/PG-YOLO-Dataset \
    --target-class 0 --prefix pg

echo "[4/5] Training YOLOv8m 3-Class Student on MI300X (100 epochs, imgsz=640)..."
python src/training/train_yolo.py --model yolov8m --data data/pomegranate_phase3/data_phase3.yaml --epochs 100 --imgsz 640 --workers 16 --project runs/phase3

BEST_P3=$(find runs -path "*phase3*" -name "best.pt" -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
cp "$BEST_P3" models/phase3/best_phase3.pt
python src/training/evaluate_and_export.py --weights models/phase3/best_phase3.pt --data data/pomegranate_phase3/data_phase3.yaml --benchmark --export onnx --output models/phase3 --report models/phase3/evaluation_report_phase3.json

echo "[5/5] Phase 3 Complete! Ready for video verification & HuggingFace push."
