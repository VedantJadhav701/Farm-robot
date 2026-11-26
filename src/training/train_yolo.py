"""
YOLOv8 Training Script — Optimized for MI300X (192 GB VRAM)
============================================================
Trains pomegranate detection models with agriculture-tuned
hyperparameters. Supports multi-model comparison and automatic
hardware-aware batch sizing.

Usage (on MI300X cloud):
  python train_yolo.py --model yolov8s --epochs 300
  python train_yolo.py --model yolov8n --epochs 200 --imgsz 1280
  python train_yolo.py --compare-all   # train n, s, m side-by-side

Usage (local RTX 3050 — small runs for testing):
  python train_yolo.py --model yolov8n --epochs 10 --batch 4 --imgsz 640
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
#  HARDWARE-AWARE BATCH SIZE PRESETS
#  Format: { vram_gb_floor: { model_variant: batch_size } }
# ═══════════════════════════════════════════════════════════════
BATCH_PRESETS = {
    # MI300X: 192 GB VRAM
    128: {"yolov8n": 256, "yolov8s": 128, "yolov8m": 64,
          "yolov8l": 32,  "yolov8x": 16},
    # A100 / H100: 40-80 GB
    32:  {"yolov8n": 64,  "yolov8s": 48,  "yolov8m": 32,
          "yolov8l": 16,  "yolov8x": 8},
    # RTX 3090 / 4090: 24 GB
    16:  {"yolov8n": 32,  "yolov8s": 24,  "yolov8m": 16,
          "yolov8l": 8,   "yolov8x": 4},
    # RTX 3060 / 3070: 8-12 GB
    6:   {"yolov8n": 16,  "yolov8s": 12,  "yolov8m": 8,
          "yolov8l": 4,   "yolov8x": 2},
    # RTX 3050 Laptop: 4 GB
    2:   {"yolov8n": 8,   "yolov8s": 4,   "yolov8m": 2,
          "yolov8l": 1,   "yolov8x": 1},
}

# ═══════════════════════════════════════════════════════════════
#  AGRICULTURE-OPTIMIZED HYPERPARAMETERS
#  Tuned for orchard / field crop detection conditions
# ═══════════════════════════════════════════════════════════════
AGRI_HYPERPARAMS = {
    # Data augmentation — heavy, to handle field variability
    "mosaic": 1.0,          # 4-image mosaic: critical for context learning
    "mixup": 0.15,          # Blend images to simulate lighting changes
    "copy_paste": 0.1,      # Paste fruit instances across images
    "degrees": 15.0,        # Rotation: fruit appears at all angles
    "translate": 0.15,      # Shift: handles camera jitter on rover
    "scale": 0.5,           # Zoom: distant vs close-up fruit
    "flipud": 0.5,          # Vertical flip: ground-mounted camera views
    "fliplr": 0.5,          # Horizontal flip: left-right invariance
    "hsv_h": 0.015,         # Hue: pomegranate color varies with ripeness
    "hsv_s": 0.7,           # Saturation: shadow vs direct sunlight
    "hsv_v": 0.4,           # Brightness: shade under canopy vs noon sun
    "erasing": 0.3,         # Random erasing: occlusion by leaves/branches

    # Optimizer & scheduler
    "optimizer": "AdamW",   # AdamW for large-batch stability on MI300X
    "lr0": 0.002,           # Initial LR (lower for large batch)
    "lrf": 0.01,            # Final LR fraction (cosine annealing)
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 5.0,
    "warmup_momentum": 0.8,

    # Training strategy
    "close_mosaic": 20,     # Disable mosaic for last 20 epochs (refine)
    "label_smoothing": 0.0,
    "nbs": 64,              # Nominal batch size for LR scaling
    "overlap_mask": True,
    "amp": True,            # Mixed precision — essential for MI300X throughput

    # Loss weights
    "box": 7.5,             # Box regression loss weight
    "cls": 0.5,             # Classification loss weight
    "dfl": 1.5,             # Distribution focal loss weight
}


def detect_hardware():
    """Detect GPU hardware and return (device, vram_gb, gpu_name)."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024 ** 3)
            return "0", vram_gb, props.name
    except Exception:
        pass
    return "cpu", 0, "CPU"


def get_batch_size(model_name: str, vram_gb: float, imgsz: int) -> int:
    """Calculate optimal batch size based on VRAM and model size."""
    # Find the matching VRAM tier
    batch = 8  # fallback
    for vram_floor in sorted(BATCH_PRESETS.keys(), reverse=True):
        if vram_gb >= vram_floor:
            batch = BATCH_PRESETS[vram_floor].get(model_name, 8)
            break

    # Scale down for larger image sizes
    if imgsz > 640:
        scale = (640 / imgsz) ** 2
        batch = max(1, int(batch * scale))

    return batch


def train_single_model(model_name: str, data_yaml: str, epochs: int,
                        imgsz: int, batch: int, device: str,
                        project_dir: str, run_name: str,
                        resume: bool = False, workers: int = 8):
    """Train a single YOLO model."""
    from ultralytics import YOLO

    print(f"\n{'═' * 60}")
    print(f"  TRAINING: {model_name}  |  {epochs} epochs  |  "
          f"batch={batch}  |  imgsz={imgsz}")
    print(f"  Output:   {project_dir}/{run_name}")
    print(f"{'═' * 60}\n")

    # Load model
    weights = f"{model_name}.pt"
    model = YOLO(weights)

    # Build training kwargs
    train_kwargs = {
        "data": data_yaml,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "workers": workers,
        "project": project_dir,
        "name": run_name,
        "exist_ok": True,
        "verbose": True,
        "save": True,
        "save_period": 25,       # Checkpoint every 25 epochs
        "val": True,
        "plots": True,           # Generate PR curves, confusion matrix
        "resume": resume,
        "seed": 42,
        "deterministic": True,
        "patience": 50,          # Early stopping patience
    }

    # Merge agriculture hyperparams
    train_kwargs.update(AGRI_HYPERPARAMS)

    # Adjust LR for large batches (linear scaling rule)
    if batch > 64:
        lr_scale = batch / 64
        train_kwargs["lr0"] = min(0.01, AGRI_HYPERPARAMS["lr0"] * lr_scale)
        train_kwargs["warmup_epochs"] = max(3, AGRI_HYPERPARAMS["warmup_epochs"])
        print(f"  [LR scaled] lr0={train_kwargs['lr0']:.5f} for batch={batch}")

    start_time = time.time()
    results = model.train(**train_kwargs)
    elapsed = time.time() - start_time

    print(f"\n[✓] Training complete in {elapsed/3600:.1f} hours")
    print(f"    Best weights: {project_dir}/{run_name}/weights/best.pt")

    return results


def run_comparison(data_yaml: str, epochs: int, imgsz: int, 
                   device: str, vram_gb: float, project_dir: str,
                   workers: int):
    """Train n, s, m models and produce a comparison report."""
    models = ["yolov8n", "yolov8s", "yolov8m"]
    results_summary = {}

    for model_name in models:
        batch = get_batch_size(model_name, vram_gb, imgsz)
        run_name = f"{model_name}_{imgsz}_{datetime.now():%Y%m%d}"

        try:
            results = train_single_model(
                model_name, data_yaml, epochs, imgsz, batch,
                device, project_dir, run_name, workers=workers
            )
            # Extract key metrics
            results_summary[model_name] = {
                "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
                "mAP50-95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
                "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
                "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
                "best_weights": f"{project_dir}/{run_name}/weights/best.pt",
                "batch_size": batch,
            }
        except Exception as e:
            print(f"[✗] {model_name} failed: {e}")
            results_summary[model_name] = {"error": str(e)}

    # Save comparison report
    report_path = Path(project_dir) / "model_comparison.json"
    report_path.write_text(json.dumps(results_summary, indent=2))
    print(f"\n{'═' * 60}")
    print("  MODEL COMPARISON RESULTS")
    print(f"{'═' * 60}")
    for name, metrics in results_summary.items():
        if "error" in metrics:
            print(f"  {name}: FAILED — {metrics['error']}")
        else:
            print(f"  {name}: mAP50={metrics['mAP50']:.3f}  "
                  f"mAP50-95={metrics['mAP50-95']:.3f}  "
                  f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}")
    print(f"\n  Full report: {report_path}")
    return results_summary


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 pomegranate detector (MI300X optimized)")
    parser.add_argument("--model", type=str, default="yolov8s",
                        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
                        help="Model variant (default: yolov8s)")
    parser.add_argument("--data", type=str, default="data/pomegranate_dataset/data.yaml",
                        help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Training epochs (default: 300)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size (default: 640; use 1280 for small objects)")
    parser.add_argument("--batch", type=int, default=0,
                        help="Batch size (0 = auto-detect from VRAM)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto', '0', 'cpu'")
    parser.add_argument("--workers", type=int, default=8,
                        help="DataLoader workers")
    parser.add_argument("--project", type=str, default="runs/pomegranate",
                        help="Project output directory")
    parser.add_argument("--name", type=str, default="",
                        help="Run name (auto-generated if empty)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--compare-all", action="store_true",
                        help="Train n/s/m models and compare metrics")
    args = parser.parse_args()

    # Detect hardware
    if args.device == "auto":
        device, vram_gb, gpu_name = detect_hardware()
    else:
        device = args.device
        vram_gb = 192 if "MI300" in str(args.device) else 4
        gpu_name = args.device

    print("╔══════════════════════════════════════════════════════╗")
    print("║   Pomegranate YOLOv8 Training                       ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  GPU    : {gpu_name}  ({vram_gb:.0f} GB VRAM)")
    print(f"  Data   : {args.data}")
    print(f"  Model  : {args.model}")
    print(f"  Epochs : {args.epochs}")
    print(f"  ImgSz  : {args.imgsz}")

    # Verify data.yaml exists
    if not Path(args.data).exists():
        print(f"\n[✗] Dataset config not found: {args.data}")
        print("    Run download_dataset.py first!")
        sys.exit(1)

    if args.compare_all:
        run_comparison(args.data, args.epochs, args.imgsz,
                       device, vram_gb, args.project, args.workers)
    else:
        # Auto batch size
        batch = args.batch if args.batch > 0 else get_batch_size(
            args.model, vram_gb, args.imgsz)

        run_name = args.name or f"{args.model}_{args.imgsz}_{datetime.now():%Y%m%d_%H%M}"

        train_single_model(
            args.model, args.data, args.epochs, args.imgsz, batch,
            device, args.project, run_name, args.resume, args.workers
        )

    print("\n[✓] Done! Next: python src/training/evaluate_and_export.py")


if __name__ == "__main__":
    main()
