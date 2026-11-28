#!/usr/bin/env bash
# Session B — Phase 4 Perception (YOLOv8m-Pose trunk_base keypoint + fruit)
# Usage: bash session_b_phase4.sh [/scratch/data/phase4/trunk_pose.yaml]
set -euo pipefail

DATA_YAML="${1:-/scratch/data/phase4/trunk_pose.yaml}"
export RUN="/scratch/runs/$(date +%Y%m%d)_sessionB_phase4"
mkdir -p "$RUN"/{data,out,logs}

echo "=== Session B: Phase 4 Perception ==="
pip freeze > "$RUN/out/env.txt"

# 1. Verify Phase 3 baseline class names first
if [ -f "models/phase3/best_phase3.pt" ]; then
  python3 - << 'PYEOF'
from ultralytics import YOLO
m = YOLO("models/phase3/best_phase3.pt")
print("Verified Phase 3 model.names:", m.names)
PYEOF
fi

# 2. Train YOLOv8m-Pose at 960px with field augmentations
if [ ! -f "$DATA_YAML" ]; then
  echo "[ERROR] Dataset YAML not found: $DATA_YAML"
  echo "Prepare contiguous time-split dataset (kpt_shape: [1, 3]) before launching training."
  exit 1
fi

yolo pose train \
  model=yolov8m-pose.pt \
  data="$DATA_YAML" \
  imgsz=960 \
  epochs=150 \
  batch=32 \
  device=0 \
  patience=30 \
  hsv_h=0.015 hsv_s=0.6 hsv_v=0.4 \
  degrees=8.0 translate=0.1 scale=0.35 \
  project="$RUN/out" \
  name=phase4

# 3. Export best weights to ONNX and record sha256
BEST_PT="$RUN/out/phase4/weights/best.pt"
yolo export model="$BEST_PT" format=onnx imgsz=960
sha256sum "$BEST_PT" "$RUN/out/phase4/weights/best.onnx" > "$RUN/out/weights.sha256"

echo "Session B complete. Model & checksums in $RUN/out/"
