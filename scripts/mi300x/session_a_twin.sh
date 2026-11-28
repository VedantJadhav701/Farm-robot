#!/usr/bin/env bash
# Session A — Digital Twin (VGGT -> COLMAP -> gsplat -> splat.ply)
# Usage: bash session_a_twin.sh /path/to/orchard.mp4 [start_s] [end_s] [max_steps]
set -euo pipefail

VIDEO_PATH="${1:-/workspace/orchard_turkey_480p.mp4}"
START_S="${2:-118}"
END_S="${3:-136}"
MAX_STEPS="${4:-30000}"

export RUN="/scratch/runs/$(date +%Y%m%d)_sessionA_twin"
mkdir -p "$RUN"/{data/frames_raw,data/scene/images,out/splat,logs}

echo "=== Session A: Digital Twin ==="
echo "Video: $VIDEO_PATH ($START_S..${END_S}s) | max_steps=$MAX_STEPS"
pip freeze > "$RUN/out/env.txt"

# 1. Extract frames and filter blurry frames via Laplacian variance
ffmpeg -y -ss "$START_S" -to "$END_S" -i "$VIDEO_PATH" -vf "fps=8" -q:v 2 "$RUN/data/frames_raw/%05d.jpg"

python3 - << 'PYEOF'
import os, glob, shutil, cv2

run_dir = os.environ["RUN"]
raw_files = sorted(glob.glob(f"{run_dir}/data/frames_raw/*.jpg"))
kept = []
for f in raw_files:
    img = cv2.imread(f)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var > 60.0:
        kept.append(f)

kept = kept[:300]
out_dir = f"{run_dir}/data/scene/images"
for idx, src in enumerate(kept):
    img = cv2.imread(src)
    h, w = img.shape[:2]
    scale = 1024.0 / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(f"{out_dir}/{idx:05d}.jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
print(f"Kept {len(kept)} / {len(raw_files)} sharp frames -> {out_dir}")
PYEOF

# 2. VGGT poses + COLMAP sparse reconstruction
if [ ! -d "/opt/vggt" ]; then
  git clone --depth 1 https://github.com/facebookresearch/vggt.git /opt/vggt
  pip install -r /opt/vggt/requirements.txt || true
fi
python3 /opt/vggt/demo_colmap.py --scene_dir="$RUN/data/scene"

# 3. Train 3D Gaussian Splat (gsplat ROCm)
if [ -d "/opt/gsplat/examples" ]; then
  cd /opt/gsplat/examples
  python3 simple_trainer.py default \
    --data_dir "$RUN/data/scene" \
    --data_factor 1 \
    --result_dir "$RUN/out/splat" \
    --max_steps "$MAX_STEPS"
else
  echo "[WARN] /opt/gsplat/examples not found. Install ROCm gsplat 1.5.3b2 first, then run simple_trainer.py."
fi

echo "Session A complete. Outputs in $RUN/out/"
