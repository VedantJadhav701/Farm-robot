# 🎯 Pomegranate Detection — Training Pipeline

## Architecture: System 1 CNN (YOLOv8)

```
Camera 30 FPS → [YOLOv8s] → Detections JSON → [System 2 SLM] → Dosing Command
                  ↑ This pipeline trains the YOLOv8 model
```

## Hardware

| Step | Hardware | Why |
|------|----------|-----|
| **Dataset download** | Any (cloud recommended) | Network speed |
| **Training** | **MI300X 192 GB** | Large batch, fast convergence |
| **Evaluation** | MI300X or local | Metrics + export |
| **Inference** | RTX 3050 / Jetson Orin | Edge deployment |

---

## Quick Start (MI300X Cloud)

```bash
# 1. SSH into cloud instance
ssh user@mi300x-instance

# 2. Upload project (from local machine)
rsync -avz --exclude='__pycache__' --exclude='.git' \
  physical-ai-robotics-venture/ user@mi300x:~/project/

# 3. Setup environment (run once)
cd ~/project
bash src/training/cloud_setup.sh
source ~/pom_train_env/bin/activate

# 4. Download dataset
export ROBOFLOW_API_KEY="your_key_here"
python src/training/download_dataset.py --presets all

# 5. Train (single model)
python src/training/train_yolo.py --model yolov8s --epochs 300

# 6. Train (compare n vs s vs m)
python src/training/train_yolo.py --compare-all --epochs 200

# 7. Evaluate best model
python src/training/evaluate_and_export.py \
  --weights runs/pomegranate/yolov8s_640_*/weights/best.pt \
  --benchmark --export onnx

# 8. Download trained weights back to local
# (from local machine)
scp user@mi300x:~/project/runs/pomegranate/*/weights/best.pt models/
```

---

## Pipeline Files

| File | Purpose |
|------|---------|
| `cloud_setup.sh` | One-time MI300X environment setup (ROCm + PyTorch + ultralytics) |
| `download_dataset.py` | Download from Roboflow, merge sources, remap classes, validate |
| `train_yolo.py` | Train YOLOv8 with agriculture-tuned hyperparams & auto batch sizing |
| `evaluate_and_export.py` | Validation metrics, speed benchmark, ONNX/TensorRT export |
| `../perception/pomegranate_detector.py` | Upgraded YOLO detector (replaces HSV fallback) |
| `../../data/pomegranate_dataset/data.yaml` | Dataset class config |

---

## Class Taxonomy

| ID | Class | Source | Use |
|----|-------|--------|-----|
| 0 | `pomegranate_fruit` | Roboflow datasets | Confirms pomegranate identity |
| 1 | `pomegranate_flower` | Roboflow datasets | Confirms identity + growth stage |
| 2 | `pomegranate_tree` | Custom annotation | Canopy bounding box |
| 3 | `trunk_base` | Custom annotation | Dosing pour target |
| 4 | `weed_cluster` | Custom annotation | Weed presence flag |
| 5 | `soil` | Custom annotation | Ground context |

> **Phase 1** (now): Train on classes 0-1 from Roboflow. Trunk base inferred geometrically.  
> **Phase 2** (later): Add custom annotations for classes 2-5 from field images.

---

## Training Parameters (MI300X Optimized)

```
Model     : yolov8s (small — best accuracy/speed tradeoff)
Epochs    : 300
Image size: 640 (or 1280 for small-object boost)
Batch     : 128 (auto-scaled for 192 GB VRAM)
Optimizer : AdamW (stable for large batches)
LR        : 0.002 → cosine → 0.00002
AMP       : Enabled (mixed precision)
Augment   : Heavy — mosaic, mixup, HSV jitter, erasing
Patience  : 50 epochs early stopping
Checkpoints: Every 25 epochs
```

---

## Expected Results

| Model | mAP@50 | mAP@50-95 | Latency (RTX 3050) | Size |
|-------|--------|-----------|---------------------|------|
| YOLOv8n | ~85% | ~55% | ~8 ms | 6 MB |
| YOLOv8s | ~90% | ~62% | ~15 ms | 22 MB |
| YOLOv8m | ~92% | ~66% | ~30 ms | 50 MB |

> These are estimates based on similar agricultural detection tasks.
> Actual numbers depend on dataset size and quality.

---

## Getting a Roboflow API Key (Free)

1. Go to [app.roboflow.com](https://app.roboflow.com/)
2. Sign up (free tier is sufficient)
3. Click your profile → **Settings** → **API Key** → Copy
4. Set as environment variable:
   ```bash
   export ROBOFLOW_API_KEY="your_key_here"
   ```

---

## Cost Estimate (MI300X @ $1.99/hr)

| Task | Duration | Cost |
|------|----------|------|
| Dataset download | ~5 min | ~$0.17 |
| Train YOLOv8s (300 epochs) | ~2-4 hrs | ~$4-8 |
| Train all 3 models | ~6-10 hrs | ~$12-20 |
| Evaluation + export | ~10 min | ~$0.33 |
| **Total (single model)** | **~3 hrs** | **~$6** |
| **Total (comparison)** | **~8 hrs** | **~$16** |

---

## After Training

1. Copy `best.pt` to local `models/` directory
2. The upgraded `pomegranate_detector.py` auto-discovers weights in:
   - `models/best.pt`
   - `runs/pomegranate/*/weights/best.pt`
3. Run the existing test: `python src/test_pomegranate_dosing.py`
4. Existing System 2 SLM integration works unchanged — the `CropDetectionResult` interface is backward-compatible
