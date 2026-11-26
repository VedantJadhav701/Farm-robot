"""
Model Evaluation, Benchmarking & Edge Export
=============================================
Runs validation metrics, generates visual reports, and exports
the best model to ONNX / TensorRT for Jetson Orin deployment.

Usage:
  python evaluate_and_export.py --weights runs/pomegranate/best.pt
  python evaluate_and_export.py --weights runs/pomegranate/best.pt --export onnx
  python evaluate_and_export.py --weights runs/pomegranate/best.pt --export engine --device 0
  python evaluate_and_export.py --weights runs/pomegranate/best.pt --benchmark
  python evaluate_and_export.py --weights runs/pomegranate/best.pt --test-image path/to/img.jpg
"""

import argparse
import json
import sys
import time
from pathlib import Path

TARGET_CLASSES = {
    0: "pomegranate_fruit",
    1: "pomegranate_flower",
    2: "pomegranate_tree",
    3: "trunk_base",
    4: "weed_cluster",
    5: "soil",
}


def run_validation(weights: str, data: str, imgsz: int, device: str):
    """Run full validation and return metrics."""
    from ultralytics import YOLO

    print(f"\n{'═' * 60}")
    print(f"  VALIDATION: {weights}")
    print(f"{'═' * 60}")

    model = YOLO(weights)
    metrics = model.val(data=data, imgsz=imgsz, device=device, 
                        plots=True, verbose=True)

    results = {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "per_class": {}
    }

    # Per-class metrics
    if hasattr(metrics.box, 'ap_class_index'):
        for i, cls_idx in enumerate(metrics.box.ap_class_index):
            cls_name = TARGET_CLASSES.get(int(cls_idx), f"class_{cls_idx}")
            results["per_class"][cls_name] = {
                "AP50": float(metrics.box.ap50[i]),
                "AP50-95": float(metrics.box.ap[i]),
            }

    print(f"\n  Overall mAP@50:    {results['mAP50']:.4f}")
    print(f"  Overall mAP@50-95: {results['mAP50-95']:.4f}")
    print(f"  Precision:          {results['precision']:.4f}")
    print(f"  Recall:             {results['recall']:.4f}")

    if results["per_class"]:
        print(f"\n  Per-Class AP@50:")
        for cls_name, ap in results["per_class"].items():
            print(f"    {cls_name:.<30s} {ap['AP50']:.4f}")

    return results


def run_benchmark(weights: str, imgsz: int, device: str):
    """Benchmark inference speed across formats."""
    from ultralytics import YOLO

    print(f"\n{'═' * 60}")
    print(f"  BENCHMARK: {weights}")
    print(f"{'═' * 60}")

    model = YOLO(weights)

    # Warm up
    import torch
    dummy = torch.randn(1, 3, imgsz, imgsz)
    if device != "cpu" and torch.cuda.is_available():
        dummy = dummy.cuda()

    # Benchmark PyTorch
    times = []
    for _ in range(100):
        start = time.perf_counter()
        model.predict(dummy, verbose=False, device=device)
        times.append((time.perf_counter() - start) * 1000)

    avg_ms = sum(times[10:]) / len(times[10:])  # skip warmup
    fps = 1000 / avg_ms

    print(f"\n  PyTorch Inference:")
    print(f"    Avg latency : {avg_ms:.1f} ms")
    print(f"    Throughput  : {fps:.0f} FPS")
    print(f"    Image size  : {imgsz}×{imgsz}")

    return {"avg_ms": avg_ms, "fps": fps, "format": "pytorch"}


def export_model(weights: str, format: str, imgsz: int, device: str,
                  output_dir: str, half: bool = True, dynamic: bool = True):
    """Export model to deployment format."""
    from ultralytics import YOLO

    print(f"\n{'═' * 60}")
    print(f"  EXPORT: {weights} → {format.upper()}")
    print(f"{'═' * 60}")

    model = YOLO(weights)

    export_kwargs = {
        "format": format,
        "imgsz": imgsz,
        "half": half and format != "onnx",  # FP16 for TensorRT
        "dynamic": dynamic and format == "onnx",
        "simplify": True,
    }

    if format == "engine":
        export_kwargs["device"] = device
        export_kwargs["workspace"] = 8  # GB for TensorRT builder

    exported_path = model.export(**export_kwargs)

    print(f"\n  [✓] Exported → {exported_path}")

    # Verify exported model
    if format == "onnx":
        try:
            import onnx
            onnx_model = onnx.load(str(exported_path))
            onnx.checker.check_model(onnx_model)
            print(f"  [✓] ONNX model validated successfully")
        except ImportError:
            print(f"  [!] Install onnx to validate: pip install onnx")

    return str(exported_path)


def test_on_image(weights: str, image_path: str, imgsz: int, 
                   device: str, output_dir: str, conf: float = 0.25):
    """Run inference on a test image and save annotated result."""
    from ultralytics import YOLO

    print(f"\n{'═' * 60}")
    print(f"  TEST INFERENCE: {image_path}")
    print(f"{'═' * 60}")

    model = YOLO(weights)
    results = model.predict(
        source=image_path, imgsz=imgsz, device=device,
        conf=conf, save=True, project=output_dir, name="test_inference",
        exist_ok=True
    )

    for r in results:
        print(f"\n  Detections ({len(r.boxes)} objects):")
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf_val = float(box.conf[0])
            cls_name = TARGET_CLASSES.get(cls_id, f"class_{cls_id}")
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            print(f"    [{cls_name}] conf={conf_val:.3f}  "
                  f"bbox=({x1},{y1})-({x2},{y2})")

    print(f"\n  [✓] Annotated image saved to {output_dir}/test_inference/")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate & export trained pomegranate detector")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to trained best.pt weights")
    parser.add_argument("--data", type=str, default="data/pomegranate_dataset/data.yaml",
                        help="Path to data.yaml for validation")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size")
    parser.add_argument("--device", type=str, default="0",
                        help="Device: '0' for GPU, 'cpu' for CPU")

    # Actions
    parser.add_argument("--validate", action="store_true", default=True,
                        help="Run validation metrics (default)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark inference speed")
    parser.add_argument("--export", type=str, default=None,
                        choices=["onnx", "engine", "torchscript", "openvino",
                                 "coreml", "tflite"],
                        help="Export format for edge deployment")
    parser.add_argument("--test-image", type=str, default=None,
                        help="Run inference on a test image")
    parser.add_argument("--output", type=str, default="models",
                        help="Output directory for exports")
    parser.add_argument("--report", type=str, default="models/evaluation_report.json",
                        help="Path to save evaluation report JSON")
    args = parser.parse_args()

    if not Path(args.weights).exists():
        print(f"[✗] Weights not found: {args.weights}")
        sys.exit(1)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    report = {"weights": args.weights, "imgsz": args.imgsz}

    # Validation
    if args.validate:
        val_results = run_validation(args.weights, args.data, args.imgsz, args.device)
        report["validation"] = val_results

    # Benchmark
    if args.benchmark:
        bench_results = run_benchmark(args.weights, args.imgsz, args.device)
        report["benchmark"] = bench_results

    # Export
    if args.export:
        export_path = export_model(args.weights, args.export, args.imgsz,
                                    args.device, args.output)
        report["export"] = {"format": args.export, "path": export_path}

    # Test image
    if args.test_image:
        test_on_image(args.weights, args.test_image, args.imgsz,
                      args.device, args.output)

    # Save report
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n[✓] Report saved → {report_path}")


if __name__ == "__main__":
    main()
