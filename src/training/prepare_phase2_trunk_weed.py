"""
Phase 2 Multi-Class Dataset Curator: Fruit + Trunk Base + Weed Cluster
=======================================================================
Prepares the 3-class orchard dataset (data/pomegranate_phase2/) combining:
  0: pomegranate_fruit  (from PG-YOLO-Dataset, 13,840 images)
  1: trunk_base         (from orchard trunk datasets / custom field images)
  2: weed_cluster       (from agricultural weed datasets / custom field images)

Also provides --merge-folder to import any YOLO-format trunk or weed dataset
with automatic class-ID remapping, and writes data_phase2.yaml ready for MI300X.

Usage:
  python src/training/prepare_phase2_trunk_weed.py --scaffold
  python src/training/prepare_phase2_trunk_weed.py --merge-folder path/to/trunks --target-class 1
  python src/training/prepare_phase2_trunk_weed.py --merge-folder path/to/weeds  --target-class 2
"""

import argparse
import shutil
from pathlib import Path
from collections import Counter

PHASE2_CLASSES = {
    0: "pomegranate_fruit",
    1: "trunk_base",
    2: "weed_cluster",
}


def scaffold_phase2(root: Path):
    for split in ("train", "valid", "test"):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)

    yaml_text = (
        f"path: {root.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        f"nc: {len(PHASE2_CLASSES)}\n"
        "names:\n"
        + "\n".join(f"  {k}: {v}" for k, v in PHASE2_CLASSES.items())
        + "\n"
    )
    yaml_path = root / "data_phase2.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    print(f"[✓] Phase 2 scaffold & config ready → {yaml_path}")
    return yaml_path


def merge_single_class_folder(src_root: Path, dst_root: Path, target_class_id: int, prefix: str):
    """Merge a YOLO dataset folder, mapping all annotations to target_class_id."""
    split_aliases = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    copied = 0

    for src_split, dst_split in split_aliases.items():
        img_dir = src_root / src_split / "images"
        lbl_dir = src_root / src_split / "labels"
        if not img_dir.exists():
            img_dir = src_root / "images" / src_split
            lbl_dir = src_root / "labels" / src_split
        if not img_dir.exists():
            continue

        dst_imgs = dst_root / dst_split / "images"
        dst_lbls = dst_root / dst_split / "labels"
        dst_imgs.mkdir(parents=True, exist_ok=True)
        dst_lbls.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in img_exts:
                continue
            new_stem = f"{prefix}_{img_path.stem}"
            shutil.copy2(img_path, dst_imgs / f"{new_stem}{img_path.suffix}")
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                lines = []
                for line in lbl_path.read_text().splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        parts[0] = str(target_class_id)
                        lines.append(" ".join(parts))
                if lines:
                    (dst_lbls / f"{new_stem}.txt").write_text("\n".join(lines) + "\n")
            copied += 1

    print(f"[✓] Merged {copied} images from {src_root} as class [{target_class_id}: {PHASE2_CLASSES[target_class_id]}]")


def summarize_dataset(root: Path):
    counts = Counter()
    total_imgs = 0
    for split in ("train", "valid", "test"):
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        if img_dir.exists():
            total_imgs += sum(1 for _ in img_dir.iterdir())
        if lbl_dir.exists():
            for lf in lbl_dir.glob("*.txt"):
                for line in lf.read_text().splitlines():
                    if line.strip():
                        counts[int(line.split()[0])] += 1
    print(f"\nPhase 2 Summary ({root}): {total_imgs} images")
    for cid, cname in PHASE2_CLASSES.items():
        print(f"  [{cid}] {cname:.<22s} {counts[cid]} boxes")


def main():
    parser = argparse.ArgumentParser(description="Prepare Phase 2 multi-class orchard dataset")
    parser.add_argument("--output", default="data/pomegranate_phase2")
    parser.add_argument("--scaffold", action="store_true")
    parser.add_argument("--merge-folder", type=str, default=None)
    parser.add_argument("--target-class", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--prefix", type=str, default="src")
    args = parser.parse_args()

    out_root = Path(args.output)
    scaffold_phase2(out_root)

    if args.merge_folder:
        merge_single_class_folder(Path(args.merge_folder), out_root, args.target_class, args.prefix)

    summarize_dataset(out_root)


if __name__ == "__main__":
    main()
