"""
download_plantdoc.py
Download and verify the PlantDoc dataset.
"""

import argparse
import os
import sys
import subprocess
import json
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np


def create_smoke_dataset(output_dir: Path, num_samples_per_class: int = 4):
    """
    Creates a tiny synthetic PlantDoc dataset for CPU smoke testing (no download needed).
    Generates synthetic 224x224 RGB images with distinct spatial patterns.
    """
    mapping_path = Path("data/class_mapping.json")
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing {mapping_path}")

    with open(mapping_path, "r", encoding="utf-8") as f:
        class_mapping = json.load(f)

    print("=" * 70)
    print(f"[SMOKE MODE] Creating synthetic PlantDoc data in {output_dir}...")
    print("NOTE: Synthetic data is STRICTLY for local CPU pipeline verification.")
    print("=" * 70)

    for split in ["train", "test"]:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        split_offset = 50 if split == "test" else 0

        for cls in class_mapping["classes"]:
            pd_folder = split_dir / cls["plantdoc_name"]
            pd_folder.mkdir(parents=True, exist_ok=True)
            cid = cls["class_id"]

            for i in range(num_samples_per_class):
                # Generate distinct visual pattern
                img = Image.new("RGB", (224, 224), color=(60 + split_offset, 90 + cid * 8, 40 + i * 25))
                draw = ImageDraw.Draw(img)
                x = (cid * 20 + i * 40 + split_offset) % 170
                y = (cid * 15 + i * 35) % 170
                draw.rectangle([x, y, x + 50, y + 50], fill=(180 - cid * 5, 120 + i * 15, 60))
                draw.line([(0, 224 - i * 50), (224, i * 50)], fill=(80, 160 + cid * 6, 90), width=4)

                img_path = pd_folder / f"smoke_{split}_{i:03d}.jpg"
                img.save(img_path, "JPEG")
                total += 1
        print(f"[SMOKE MODE] Created {total} synthetic PlantDoc {split} images across {len(class_mapping['classes'])} classes.")

    return output_dir


def check_kaggle_credentials() -> bool:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return False


def download_plantdoc(output_dir: Path, source: str = "github"):
    output_dir.mkdir(parents=True, exist_ok=True)
    test_dir = output_dir / "test"
    if test_dir.exists() and any(test_dir.iterdir()):
        print(f"PlantDoc dataset already present at: {output_dir}")
        return output_dir

    if source == "github":
        repo_url = "https://github.com/pratikkayal/PlantDoc-Dataset.git"
        target_git_dir = output_dir / "repo"
        if not target_git_dir.exists():
            print(f"Cloning {repo_url} (depth 1)...")
            subprocess.run(["git", "clone", "--depth", "1", repo_url, str(target_git_dir)], check=True)

        for folder in ["train", "test"]:
            src = target_git_dir / folder
            dst = output_dir / folder
            if src.exists() and not dst.exists():
                import shutil
                shutil.copytree(str(src), str(dst))
                print(f"Copied {folder} to {dst}")

    elif source == "kaggle":
        if not check_kaggle_credentials():
            print("ERROR: Kaggle credentials not found in ~/.kaggle/kaggle.json")
            sys.exit(1)

        import kaggle
        print("Downloading daeyoung96/plant-doc-dataset via Kaggle API...")
        kaggle.api.dataset_download_files("daeyoung96/plant-doc-dataset", path=str(output_dir), unzip=True)
        print("Kaggle download completed.")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Download PlantDoc dataset or generate smoke data.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true", help="Generate minimal synthetic data for local CPU testing")
    group.add_argument("--real", action="store_true", help="Download real PlantDoc dataset (Colab/GPU)")

    parser.add_argument("--output_dir", type=str, default="data/plantdoc", help="Output directory")
    parser.add_argument("--source", type=str, choices=["github", "kaggle"], default="github", help="Download source")
    parser.add_argument("--samples_per_class", type=int, default=4, help="Samples per class in smoke mode")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    if args.smoke:
        create_smoke_dataset(out_path, args.samples_per_class)
    elif args.real:
        download_plantdoc(out_path, source=args.source)


if __name__ == "__main__":
    main()
