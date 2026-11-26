"""
Phase 3 Teacher-Student Auto-Labeler (Grounding-DINO Teacher -> YOLO Student)
=============================================================================
Extracts real orchard frames from pomegranate videos (e.g. orchard_turkey_480p.mp4)
and uses a large Vision-Language / Open-Vocabulary Teacher (Grounding-DINO-Base)
to generate in-domain YOLO labels for:
  0: pomegranate_fruit  (ripe red + green)
  1: trunk_base         (actual pomegranate trunks)
  2: weed_cluster       (ground weeds/grass around trunk)

Usage (on MI300X / big cloud GPU):
  python src/training/phase3_teacher_autolabel.py \
      --video data/real_test_samples/orchard_turkey_480p.mp4 \
      --output data/pomegranate_phase3_teacher \
      --stride-sec 1.5 --max-frames 600
"""

import argparse
import random
from pathlib import Path
import cv2
import torch


TEACHER_PROMPTS = {
    0: "pomegranate fruit.",
    1: "tree trunk.",
    2: "weed.",
}


def extract_frames(video_path: Path, raw_dir: Path, stride_sec: float, max_frames: int):
    raw_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps * stride_sec)))

    saved = []
    idx = 0
    while len(saved) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            out_p = raw_dir / f"vid_{idx:06d}.jpg"
            cv2.imwrite(str(out_p), frame)
            saved.append(out_p)
        idx += 1
    cap.release()
    print(f"[✓] Extracted {len(saved)} frames from {video_path.name}")
    return saved


def autolabel_with_grounding_dino(
    image_paths: list[Path],
    dataset_root: Path,
    model_id: str = "IDEA-Research/grounding-dino-base",
    box_thresh: float = 0.28,
    text_thresh: float = 0.25,
    val_ratio: float = 0.15,
):
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[→] Loading Teacher: {model_id} on {device}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()

    for split in ("train", "valid"):
        (dataset_root / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_root / split / "labels").mkdir(parents=True, exist_ok=True)

    random.seed(42)
    shuffled = list(image_paths)
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))

    prompt_str = " ".join(TEACHER_PROMPTS.values())
    total_boxes = {0: 0, 1: 0, 2: 0}

    for i, img_p in enumerate(shuffled):
        split = "valid" if i < n_val else "train"
        pil_img = Image.open(img_p).convert("RGB")
        w, h = pil_img.size

        inputs = processor(images=pil_img, text=prompt_str, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_thresh,
            text_threshold=text_thresh,
            target_sizes=[(h, w)],
        )[0]

        yolo_lines = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            lbl = str(label).lower()
            if "pomegranate" in lbl or "fruit" in lbl:
                cid = 0
            elif "trunk" in lbl:
                cid = 1
            elif "weed" in lbl:
                cid = 2
            else:
                continue

            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            bw = max(1.0, x2 - x1) / w
            bh = max(1.0, y2 - y1) / h
            # Reject huge full-frame false boxes
            if bw > 0.85 and bh > 0.85:
                continue
            xc = ((x1 + x2) / 2.0) / w
            yc = ((y1 + y2) / 2.0) / h
            yolo_lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            total_boxes[cid] += 1

        if yolo_lines:
            dst_img = dataset_root / split / "images" / img_p.name
            dst_lbl = dataset_root / split / "labels" / f"{img_p.stem}.txt"
            dst_img.write_bytes(img_p.read_bytes())
            dst_lbl.write_text("\n".join(yolo_lines) + "\n")

        if (i + 1) % 100 == 0:
            print(f"  Teacher labeled {i+1}/{len(shuffled)} frames | boxes={total_boxes}")

    yaml_text = (
        f"path: {dataset_root.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n\n"
        "nc: 3\n"
        "names:\n"
        "  0: pomegranate_fruit\n"
        "  1: trunk_base\n"
        "  2: weed_cluster\n"
    )
    (dataset_root / "data_phase3.yaml").write_text(yaml_text)
    print(f"[✓] Teacher auto-labeling complete! Boxes: {total_boxes}")


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Teacher Auto-Labeling")
    parser.add_argument("--video", default="data/real_test_samples/orchard_turkey_480p.mp4")
    parser.add_argument("--output", default="data/pomegranate_phase3_teacher")
    parser.add_argument("--stride-sec", type=float, default=1.5)
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.output)
    frames = extract_frames(Path(args.video), out_root / "_raw_frames", args.stride_sec, args.max_frames)
    if not args.extract_only and frames:
        autolabel_with_grounding_dino(frames, out_root)


if __name__ == "__main__":
    main()
