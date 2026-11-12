"""
download_plantvillage.py
Download and verify the PlantVillage dataset.
"""

import argparse
import os
import sys
import subprocess
import json
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np


def create_smoke_dataset(output_dir: Path, num_samples_per_class: int = 5):
    """
    Creates a tiny synthetic PlantVillage dataset for CPU smoke testing (no download needed).
    Generates synthetic 224x224 RGB images with distinct spatial patterns so each has a unique hash.
    """
    mapping_path = Path("data/class_mapping.json")
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing {mapping_path}")

    with open(mapping_path, "r", encoding="utf-8") as f:
        class_mapping = json.load(f)

    color_dir = output_dir / "raw" / "color"
    color_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"[SMOKE MODE] Creating synthetic PlantVillage data in {color_dir}...")
    print("NOTE: Synthetic data is STRICTLY for local CPU pipeline verification.")
    print("=" * 70)

    total_created = 0
    for cls in class_mapping["classes"]:
        pv_folder = color_dir / cls["plantvillage_name"]
        pv_folder.mkdir(parents=True, exist_ok=True)
        cid = cls["class_id"]

        for i in range(num_samples_per_class):
            # Create distinct pattern with gradient and shapes
            img = Image.new("RGB", (224, 224), color=(30 + cid * 12, 100 + i * 20, 40 + (cid * 5) % 80))
            draw = ImageDraw.Draw(img)
            # Draw distinct diagonal lines and ellipses based on cid and sample index
            x0 = (cid * 15 + i * 30) % 180
            y0 = (cid * 10 + i * 25) % 180
            draw.ellipse([x0, y0, x0 + 40, y0 + 40], fill=(200, 150 + cid * 5, 50))
            draw.line([(0, i * 40), (224, 224 - i * 40)], fill=(50, 200 - cid * 8, 100), width=3)

            img_path = pv_folder / f"smoke_img_{i:03d}.jpg"
            img.save(img_path, "JPEG")
            total_created += 1

    print(f"[SMOKE MODE] Created {total_created} synthetic PlantVillage images across {len(class_mapping['classes'])} classes.")
    return color_dir


def check_kaggle_credentials() -> bool:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return False


def download_plantvillage(output_dir: Path, source: str = "github"):
    output_dir.mkdir(parents=True, exist_ok=True)
    color_dir = output_dir / "raw" / "color"
    if color_dir.exists() and any(color_dir.iterdir()):
        print(f"PlantVillage color images already present at: {color_dir}")
        return color_dir

    if source == "github":
        repo_url = "https://github.com/spMohanty/PlantVillage-Dataset.git"
        target_git_dir = output_dir / "repo"
        print(f"Cloning {repo_url} (sparse checkout: raw/color only)...")
        if not target_git_dir.exists():
            subprocess.run([
                "git", "clone", "--filter=blob:none", "--no-checkout",
                repo_url, str(target_git_dir)
            ], check=True)
            subprocess.run(["git", "-C", str(target_git_dir), "sparse-checkout", "set", "raw/color"], check=True)
            subprocess.run(["git", "-C", str(target_git_dir), "checkout"], check=True)

        color_src = target_git_dir / "raw" / "color"
        if not color_dir.exists() and color_src.exists():
            import shutil
            shutil.copytree(str(color_src), str(color_dir))
            print(f"Successfully copied raw/color to {color_dir}")

    elif source == "kaggle":
        if not check_kaggle_credentials():
            print("ERROR: Kaggle credentials not found in ~/.kaggle/kaggle.json")
            sys.exit(1)

        import kaggle
        print("Downloading emmarex/plantdisease via Kaggle API...")
        kaggle.api.dataset_download_files("emmarex/plantdisease", path=str(output_dir), unzip=True)
        print("Kaggle download completed.")

    return color_dir


def main():
    parser = argparse.ArgumentParser(description="Download PlantVillage dataset or generate smoke data.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true", help="Generate minimal synthetic data for local CPU testing")
    group.add_argument("--real", action="store_true", help="Download real PlantVillage dataset (Colab/GPU)")

    parser.add_argument("--output_dir", type=str, default="data/plantvillage", help="Output directory")
    parser.add_argument("--source", type=str, choices=["github", "kaggle"], default="github", help="Download source")
    parser.add_argument("--samples_per_class", type=int, default=5, help="Samples per class in smoke mode")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    if args.smoke:
        create_smoke_dataset(out_path, args.samples_per_class)
    elif args.real:
        download_plantvillage(out_path, source=args.source)


if __name__ == "__main__":
    main()
