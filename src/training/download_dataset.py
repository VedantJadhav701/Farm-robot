"""
Pomegranate Detection Dataset Downloader & Curator
===================================================
Downloads annotated pomegranate datasets from Roboflow Universe,
remaps classes to our standard taxonomy, merges multiple sources,
validates YOLO-format structure, and reports class statistics.

Supports:
  • Roboflow API download (--api-key)
  • Manual folder import  (--import-dir)
  • Multiple dataset merge with class remapping

Usage:
  python download_dataset.py --api-key YOUR_KEY
  python download_dataset.py --import-dir /path/to/yolo_dataset
"""

import os
import sys
import json
import shutil
import argparse
import random
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════
#  OUR TARGET CLASS TAXONOMY  (System 1 CNN output classes)
# ═══════════════════════════════════════════════════════════════
TARGET_CLASSES = {
    0: "pomegranate_fruit",    # Fruits at any stage (bud → ripe)
    1: "pomegranate_flower",   # Flowers / blossoms
    2: "pomegranate_tree",     # Whole tree canopy bounding box
    3: "trunk_base",           # Lower trunk / root collar zone
    4: "weed_cluster",         # Weed patches near tree base
    5: "soil",                 # Bare soil patches
}

# ═══════════════════════════════════════════════════════════════
#  KNOWN ROBOFLOW DATASET PRESETS
#  Each entry: (workspace_slug, project_slug, version, source_class_names)
# ═══════════════════════════════════════════════════════════════
ROBOFLOW_PRESETS = {
    "pomegranate-growing-stage": {
        "workspace": "object-detection-yrzek",
        "project": "pomegranate-growing-stage",
        "version": 1,
        "source_classes": ["Flower", "Bud", "Early_fruits", "Mid_fruits", "Ripe"],
        "remap": {
            0: 1,   # Flower     → pomegranate_flower
            1: 1,   # Bud        → pomegranate_flower
            2: 0,   # Early_fruits → pomegranate_fruit
            3: 0,   # Mid_fruits → pomegranate_fruit
            4: 0,   # Ripe       → pomegranate_fruit
        },
        "description": "5-class fruit growth stages (Flower, Bud, Early, Mid, Ripe)"
    },
    "pomegranate-detection": {
        "workspace": "projects-tsmyb",
        "project": "pomegranate-nujlt",
        "version": 1,
        "source_classes": ["pomegranate"],
        "remap": {
            0: 0,   # pomegranate → pomegranate_fruit
        },
        "description": "Single-class pomegranate fruit detection"
    },
    "fresh-pomegranate": {
        "workspace": "catkin-tltqy",
        "project": "fresh_pomegranate",
        "version": 1,
        "source_classes": ["Fresh_Pomegranate"],
        "remap": {
            0: 0,   # Fresh_Pomegranate → pomegranate_fruit
        },
        "description": "Fresh pomegranate fruit bounding boxes"
    },
}


def download_roboflow(api_key: str, preset_name: str, 
                       download_dir: Path, format: str = "yolov8") -> Path:
    """Download a dataset from Roboflow Universe using a preset config."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[✗] Install roboflow first:  pip install roboflow")
        sys.exit(1)

    preset = ROBOFLOW_PRESETS[preset_name]
    print(f"\n[↓] Downloading '{preset_name}': {preset['description']}")
    print(f"    Workspace: {preset['workspace']}  Project: {preset['project']}  v{preset['version']}")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(preset["workspace"]).project(preset["project"])
    version = project.version(preset["version"])

    dest = download_dir / f"_raw_{preset_name}"
    dataset = version.download(format, location=str(dest))
    print(f"[✓] Downloaded → {dest}")
    return dest


def download_roboflow_custom(api_key: str, workspace: str, project: str,
                              version: int, download_dir: Path,
                              format: str = "yolov8") -> Path:
    """Download any Roboflow dataset with custom workspace/project/version."""
    from roboflow import Roboflow
    
    print(f"\n[↓] Downloading custom: {workspace}/{project} v{version}")
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    ver = proj.version(version)
    
    dest = download_dir / f"_raw_{project}_v{version}"
    ver.download(format, location=str(dest))
    print(f"[✓] Downloaded → {dest}")
    return dest


def remap_labels_in_dir(labels_dir: Path, remap: Dict[int, int]) -> Tuple[int, int]:
    """
    Remap class IDs in all YOLO .txt label files in a directory.
    Returns (remapped_count, dropped_count).
    """
    remapped_count = 0
    dropped_count = 0

    for label_file in sorted(labels_dir.glob("*.txt")):
        lines = label_file.read_text().strip().split('\n')
        new_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            old_id = int(parts[0])
            if old_id in remap:
                parts[0] = str(remap[old_id])
                new_lines.append(' '.join(parts))
                remapped_count += 1
            else:
                dropped_count += 1

        if new_lines:
            label_file.write_text('\n'.join(new_lines) + '\n')
        else:
            # No valid annotations left → remove label (and optionally image)
            label_file.unlink()

    return remapped_count, dropped_count


def merge_into_dataset(source_dir: Path, target_dir: Path, 
                        remap: Optional[Dict[int, int]] = None,
                        split_map: Optional[Dict[str, str]] = None):
    """
    Merge a downloaded YOLO dataset into our canonical directory structure.
    Handles different split folder names (train/valid/test vs train/val).
    """
    if split_map is None:
        split_map = {
            "train": "train", "valid": "valid", "val": "valid", 
            "test": "test",
        }

    total_images = 0
    for src_split, tgt_split in split_map.items():
        # Try both flat and nested structures
        for img_subdir in ["images", ""]:
            src_imgs = source_dir / src_split / img_subdir if img_subdir else source_dir / src_split
            if not src_imgs.exists():
                continue

            tgt_imgs = target_dir / tgt_split / "images"
            tgt_lbls = target_dir / tgt_split / "labels"
            tgt_imgs.mkdir(parents=True, exist_ok=True)
            tgt_lbls.mkdir(parents=True, exist_ok=True)

            # Find all images
            img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            for img_file in sorted(src_imgs.iterdir()):
                if img_file.suffix.lower() not in img_exts:
                    continue

                # Copy image
                dst_img = tgt_imgs / img_file.name
                if not dst_img.exists():
                    shutil.copy2(img_file, dst_img)
                    total_images += 1

                # Find and copy corresponding label
                lbl_name = img_file.stem + ".txt"
                for lbl_subdir in ["labels", ""]:
                    src_lbl = source_dir / src_split / lbl_subdir / lbl_name if lbl_subdir else source_dir / src_split / lbl_name
                    if src_lbl.exists():
                        dst_lbl = tgt_lbls / lbl_name
                        shutil.copy2(src_lbl, dst_lbl)
                        break

    # Apply class remapping if specified
    if remap:
        for split in ["train", "valid", "test"]:
            lbl_dir = target_dir / split / "labels"
            if lbl_dir.exists():
                remapped, dropped = remap_labels_in_dir(lbl_dir, remap)
                if remapped or dropped:
                    print(f"    {split}: {remapped} annotations remapped, {dropped} dropped")

    print(f"    Merged {total_images} images into {target_dir.name}")


def validate_dataset(dataset_dir: Path) -> Dict:
    """Validate YOLO dataset and return detailed statistics."""
    stats = {}
    for split in ['train', 'valid', 'test']:
        images_dir = dataset_dir / split / 'images'
        labels_dir = dataset_dir / split / 'labels'

        if not images_dir.exists():
            stats[split] = {'images': 0, 'labels': 0, 'class_counts': {},
                           'images_without_labels': 0, 'avg_objects_per_image': 0}
            continue

        img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        images = [f for f in images_dir.iterdir() if f.suffix.lower() in img_exts]
        labels = list(labels_dir.glob('*.txt')) if labels_dir.exists() else []

        # Match images to labels
        label_stems = {l.stem for l in labels}
        images_without_labels = sum(1 for img in images if img.stem not in label_stems)

        # Count class distribution
        class_counts = Counter()
        total_objects = 0
        for lf in labels:
            for line in lf.read_text().strip().split('\n'):
                if line.strip():
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
                    total_objects += 1

        stats[split] = {
            'images': len(images),
            'labels': len(labels),
            'class_counts': dict(class_counts),
            'images_without_labels': images_without_labels,
            'avg_objects_per_image': round(total_objects / max(1, len(labels)), 1)
        }

    return stats


def print_stats(stats: Dict, classes: Dict[int, str]):
    """Pretty-print dataset statistics."""
    print("\n" + "=" * 60)
    print("  DATASET STATISTICS")
    print("=" * 60)

    grand_total_imgs = 0
    grand_total_objs = 0

    for split in ['train', 'valid', 'test']:
        s = stats.get(split, {})
        n_imgs = s.get('images', 0)
        n_lbls = s.get('labels', 0)
        no_lbl = s.get('images_without_labels', 0)
        avg_obj = s.get('avg_objects_per_image', 0)
        grand_total_imgs += n_imgs

        print(f"\n  {split.upper():>5}:  {n_imgs} images  |  {n_lbls} labels  |  "
              f"avg {avg_obj} objects/img")
        if no_lbl > 0:
            print(f"         ⚠ {no_lbl} images have no label file (background negatives)")

        for cls_id in sorted(s.get('class_counts', {}).keys()):
            count = s['class_counts'][cls_id]
            cls_name = classes.get(cls_id, f"unknown_{cls_id}")
            grand_total_objs += count
            bar = "█" * min(40, count // 5)
            print(f"         [{cls_id}] {cls_name:.<25s} {count:>5}  {bar}")

    print(f"\n  TOTAL: {grand_total_imgs} images  |  {grand_total_objs} annotations")
    print("=" * 60)


def create_data_yaml(dataset_dir: Path, classes: Dict[int, str]) -> Path:
    """Generate the data.yaml config for YOLO training."""
    # Determine which classes actually exist in the dataset
    stats = validate_dataset(dataset_dir)
    present_classes = set()
    for split_stats in stats.values():
        present_classes.update(split_stats.get('class_counts', {}).keys())

    yaml_lines = [
        "# ═══════════════════════════════════════════════════════════",
        "# Pomegranate Agricultural Detection Dataset",
        "# Auto-generated by download_dataset.py",
        "# ═══════════════════════════════════════════════════════════",
        "",
        f"path: {dataset_dir.resolve().as_posix()}",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "",
        f"nc: {len(present_classes) if present_classes else len(classes)}",
        "names:",
    ]

    active_classes = present_classes if present_classes else classes.keys()
    for idx in sorted(active_classes):
        name = classes.get(idx, f"class_{idx}")
        yaml_lines.append(f"  {idx}: {name}")

    yaml_lines.append("")

    yaml_path = dataset_dir / "data.yaml"
    yaml_path.write_text('\n'.join(yaml_lines))
    print(f"[✓] data.yaml → {yaml_path}")
    return yaml_path


def ensure_splits(dataset_dir: Path, val_ratio: float = 0.15, test_ratio: float = 0.10):
    """
    If only train/ exists, create valid/ and test/ splits.
    Moves files (not copies) to avoid duplication.
    """
    train_imgs = dataset_dir / "train" / "images"
    valid_imgs = dataset_dir / "valid" / "images"
    test_imgs  = dataset_dir / "test"  / "images"

    if not train_imgs.exists():
        return

    # If valid already has images, skip
    if valid_imgs.exists() and any(valid_imgs.iterdir()):
        return

    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    all_images = sorted([f for f in train_imgs.iterdir() if f.suffix.lower() in img_exts])

    if len(all_images) < 10:
        print("[!] Too few images for splitting. Keep all in train/")
        return

    random.seed(42)
    random.shuffle(all_images)

    n = len(all_images)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))

    test_set = all_images[:n_test]
    val_set = all_images[n_test:n_test + n_val]
    # Rest stays in train

    for split_name, split_files in [("valid", val_set), ("test", test_set)]:
        tgt_imgs = dataset_dir / split_name / "images"
        tgt_lbls = dataset_dir / split_name / "labels"
        tgt_imgs.mkdir(parents=True, exist_ok=True)
        tgt_lbls.mkdir(parents=True, exist_ok=True)

        for img_file in split_files:
            shutil.move(str(img_file), str(tgt_imgs / img_file.name))
            lbl_file = dataset_dir / "train" / "labels" / (img_file.stem + ".txt")
            if lbl_file.exists():
                shutil.move(str(lbl_file), str(tgt_lbls / lbl_file.name))

    print(f"[✓] Split: {n - n_val - n_test} train / {n_val} valid / {n_test} test")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Download & curate pomegranate detection datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download preset datasets from Roboflow:
  python download_dataset.py --api-key RF_KEY --presets pomegranate-growing-stage

  # Download ALL preset datasets and merge:
  python download_dataset.py --api-key RF_KEY --presets all

  # Download a custom Roboflow dataset:
  python download_dataset.py --api-key RF_KEY \\
      --custom-workspace my_ws --custom-project my_proj --custom-version 2

  # Import a manually-downloaded YOLO dataset folder:
  python download_dataset.py --import-dir /path/to/dataset \\
      --remap '{"0":0, "1":1}'

  # Just create directory structure (no download):
  python download_dataset.py --scaffold-only
        """)

    parser.add_argument("--api-key", type=str,
                        default=os.environ.get("ROBOFLOW_API_KEY", ""),
                        help="Roboflow API key (or ROBOFLOW_API_KEY env var)")
    parser.add_argument("--output", type=str, default="data/pomegranate_dataset",
                        help="Output dataset directory")
    parser.add_argument("--presets", nargs="+",
                        default=["pomegranate-growing-stage"],
                        choices=list(ROBOFLOW_PRESETS.keys()) + ["all"],
                        help="Which preset datasets to download")
    parser.add_argument("--custom-workspace", type=str, help="Custom Roboflow workspace slug")
    parser.add_argument("--custom-project", type=str, help="Custom Roboflow project slug")
    parser.add_argument("--custom-version", type=int, default=1, help="Custom dataset version")
    parser.add_argument("--import-dir", type=str, help="Import from local YOLO dataset folder")
    parser.add_argument("--remap", type=str, default=None,
                        help='Class remap JSON for import, e.g. \'{"0":0, "1":1}\'')
    parser.add_argument("--scaffold-only", action="store_true",
                        help="Only create directory structure + data.yaml, no download")
    parser.add_argument("--scratch-dir", type=str, default=None,
                        help="Scratch dir for raw downloads (default: output/_raw_*)")

    args = parser.parse_args()
    output_dir = Path(args.output)
    scratch_dir = Path(args.scratch_dir) if args.scratch_dir else output_dir

    # Always ensure directories exist
    for split in ['train', 'valid', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    if args.scaffold_only:
        create_data_yaml(output_dir, TARGET_CLASSES)
        print("[✓] Scaffold created. Add images and labels manually.")
        return

    # ── Roboflow preset downloads ──────────────────────────
    if args.api_key and "all" not in args.presets:
        presets_to_download = args.presets
    elif args.api_key and "all" in args.presets:
        presets_to_download = list(ROBOFLOW_PRESETS.keys())
    else:
        presets_to_download = []

    if presets_to_download and not args.api_key:
        print("╔══════════════════════════════════════════════════════╗")
        print("║         ROBOFLOW API KEY REQUIRED                   ║")
        print("╚══════════════════════════════════════════════════════╝")
        print()
        print("  1. Go to  https://app.roboflow.com/  (free account)")
        print("  2. Settings → API Key → Copy")
        print("  3. Re-run:  python download_dataset.py --api-key YOUR_KEY")
        print("     OR set:  export ROBOFLOW_API_KEY=YOUR_KEY")
        print()
        print("  Available presets:")
        for name, cfg in ROBOFLOW_PRESETS.items():
            print(f"    • {name}: {cfg['description']}")
        print()
        create_data_yaml(output_dir, TARGET_CLASSES)
        return

    for preset_name in presets_to_download:
        try:
            raw_dir = download_roboflow(args.api_key, preset_name, scratch_dir)
            preset = ROBOFLOW_PRESETS[preset_name]
            merge_into_dataset(raw_dir, output_dir, remap=preset["remap"])
        except Exception as e:
            print(f"[✗] Failed to download '{preset_name}': {e}")
            print(f"    Try manually: https://universe.roboflow.com/search?q=pomegranate")
            continue

    # ── Custom Roboflow download ───────────────────────────
    if args.custom_workspace and args.custom_project and args.api_key:
        raw_dir = download_roboflow_custom(
            args.api_key, args.custom_workspace, args.custom_project,
            args.custom_version, scratch_dir
        )
        remap = json.loads(args.remap) if args.remap else None
        if remap:
            remap = {int(k): int(v) for k, v in remap.items()}
        merge_into_dataset(raw_dir, output_dir, remap=remap)

    # ── Local folder import ────────────────────────────────
    if args.import_dir:
        import_path = Path(args.import_dir)
        if not import_path.exists():
            print(f"[✗] Import directory not found: {import_path}")
            sys.exit(1)
        remap = None
        if args.remap:
            remap = {int(k): int(v) for k, v in json.loads(args.remap).items()}
        merge_into_dataset(import_path, output_dir, remap=remap)

    # ── Ensure proper splits ───────────────────────────────
    ensure_splits(output_dir)

    # ── Generate data.yaml ─────────────────────────────────
    create_data_yaml(output_dir, TARGET_CLASSES)

    # ── Report statistics ──────────────────────────────────
    stats = validate_dataset(output_dir)
    print_stats(stats, TARGET_CLASSES)

    # ── Cleanup raw downloads (optional) ───────────────────
    for raw in scratch_dir.glob("_raw_*"):
        if raw.is_dir():
            size_mb = sum(f.stat().st_size for f in raw.rglob('*') if f.is_file()) / 1024**2
            print(f"[i] Raw download kept: {raw} ({size_mb:.0f} MB) — delete manually if done")

    print("\n[✓] Dataset ready for training!")
    print(f"    Config: {output_dir / 'data.yaml'}")
    print(f"    Next:   python src/training/train_yolo.py --data {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
