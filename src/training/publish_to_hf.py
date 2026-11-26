"""
Publish Trained Pomegranate Dual-Brain Models to HuggingFace Hub
================================================================
Creates (or updates) a HuggingFace repository and uploads:
  - System 1 CNN: models/best.pt, models/best.onnx
  - System 2 SLM: models/slm_lora/
  - Metrics & plots: models/metrics/*

Usage:
  python src/training/publish_to_hf.py --token YOUR_HF_TOKEN --repo-name pomegranate-dual-brain-ai
"""

import argparse
import json
import os
from pathlib import Path


def build_model_card(repo_id: str, eval_report: dict) -> str:
    val = eval_report.get("validation", {})
    bench = eval_report.get("benchmark", {})
    map50 = val.get("mAP50", 0.0)
    map50_95 = val.get("mAP50-95", 0.0)
    prec = val.get("precision", 0.0)
    rec = val.get("recall", 0.0)
    fps = bench.get("fps", 0.0)
    avg_ms = bench.get("avg_ms", 0.0)

    return f"""---
license: apache-2.0
tags:
  - robotics
  - physical-ai
  - agriculture
  - pomegranate
  - yolov8
  - qwen2.5
  - lora
  - precision-agriculture
---

# Pomegranate Dual-Brain AI (`{repo_id}`)

Autonomous **System 1 (CNN) + System 2 (SLM)** Dual-Brain for pomegranate crop verification and precision liquid fertilizer dosing.

## Architecture

- **System 1 (Fast Reflex, 30+ Hz):** `YOLOv8s` trained on 13,840 orchard images (`PG-YOLO-Dataset`, 9,798 train / 4,042 val) on AMD Instinct MI300X (192 GB VRAM).
- **System 2 (Cognitive Supervisor, 1-2 Hz):** `Qwen2.5-3B-Instruct` fine-tuned with LoRA on agricultural dosing commands, confidence gating, and non-target refusal.

## Evaluation Metrics (Validation Set: 4,042 images / 13,815 instances)

| Metric | Value |
|---|---|
| **mAP@50** | `{map50:.4f}` |
| **mAP@50-95** | `{map50_95:.4f}` |
| **Precision** | `{prec:.4f}` |
| **Recall** | `{rec:.4f}` |
| **Inference Latency / FPS** | `{avg_ms:.1f} ms` / `{fps:.0f} FPS` |

## Literature Benchmark Comparison

| Model | Base | mAP@50 | mAP@50-95 | Notes |
|---|---|---|---|---|
| **PG-YOLO** (Wang et al., 2024) | YOLOv8 | 93.4% | - | Orchard fruit thinning |
| **YOLO-Punica** (2024) | YOLOv8n | 92.6% | - | Edge fruit monitoring (14.3 FPS) |
| **YOLO-MSNet** (2025) | YOLOv11n | 91.9% | - | Multi-stage orchard detection |
| **Ours (System 1 YOLOv8s)** | YOLOv8s | `{map50*100:.1f}%` | `{map50_95*100:.1f}%` | Paired with Qwen2.5-3B LoRA safety gate |

## Files Included

- `system1_yolo/best.pt` — Trained PyTorch YOLOv8s weights
- `system1_yolo/best.onnx` — Exported ONNX graph for edge deployment
- `system2_slm_lora/` — Merged Qwen2.5-3B-Instruct agricultural safety supervisor
- `metrics/` — `results.csv`, `evaluation_report.json`, confusion matrix & PR curves
"""


def main():
    parser = argparse.ArgumentParser(description="Push Dual-Brain models to HuggingFace Hub")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", ""), required=False)
    parser.add_argument("--repo-name", default="pomegranate-dual-brain-ai")
    parser.add_argument("--private", action="store_true", default=False)
    args = parser.parse_args()

    if not args.token:
        raise ValueError("Provide --token or set HF_TOKEN env var.")

    from huggingface_hub import HfApi

    api = HfApi(token=args.token)
    username = api.whoami()["name"]
    repo_id = f"{username}/{args.repo_name}"

    print(f"[→] Creating/checking HuggingFace repo: {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="model", private=args.private, exist_ok=True)

    eval_path = Path("models/evaluation_report.json")
    eval_report = json.loads(eval_path.read_text()) if eval_path.exists() else {}

    readme_path = Path("models/HF_README.md")
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(build_model_card(repo_id, eval_report), encoding="utf-8")

    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    print("[✓] Uploaded README.md")

    for local_file, remote_path in [
        ("models/best.pt", "system1_yolo/best.pt"),
        ("models/best.onnx", "system1_yolo/best.onnx"),
        ("models/evaluation_report.json", "metrics/evaluation_report.json"),
    ]:
        p = Path(local_file)
        if p.exists():
            print(f"[→] Uploading {local_file} → {remote_path}...")
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=remote_path,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"[✓] Uploaded {remote_path}")

    metrics_dir = Path("models/metrics")
    if metrics_dir.exists():
        print("[→] Uploading models/metrics/ → metrics/...")
        api.upload_folder(
            folder_path=str(metrics_dir),
            path_in_repo="metrics",
            repo_id=repo_id,
            repo_type="model",
        )
        print("[✓] Uploaded metrics/")

    slm_dir = Path("models/slm_lora")
    if slm_dir.exists():
        print("[→] Uploading models/slm_lora/ → system2_slm_lora/...")
        api.upload_folder(
            folder_path=str(slm_dir),
            path_in_repo="system2_slm_lora",
            repo_id=repo_id,
            repo_type="model",
        )
        print("[✓] Uploaded system2_slm_lora/")

    print(f"\n[✓] Published! URL: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
