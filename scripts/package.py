"""
scripts/package.py
==================
Phase 7: Academic Submission Packaging Script.

Creates a clean, reproducible submission archive:
    GroupXX_code_Farmer_Advisory_Chatbot_Multilingual_NLP.zip

Includes:
  - All source code (*.py, *.yaml, *.json, *.md, *.sh, *.ps1)
  - Config files and mappings (configs/*)
  - Ingested PDF knowledge base and chunks (kb/*)
  - Evaluation scripts (eval/*)
  - Jupyter / Colab notebooks (notebooks/*)
  - Generated report assets (report_assets/*)
  - CSV results and metric logs (results/*, logs/*)
  - requirements.txt, README.md, .gitignore

Excludes:
  - Large binary model checkpoints (*.pth, torch weights)
  - Raw image datasets (PlantVillage, PlantDoc image files)
  - Virtual environments (venv/, .venv/)
  - Git version history (.git/)
  - Python byte code (__pycache__/, *.pyc)
  - Temporary zip files and OS artifacts

Usage:
  python scripts/package.py [GROUP_NUMBER] [TITLE]
"""

import argparse
import fnmatch
import os
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
    "venv",
    ".venv",
    "wandb",
    "plantvillage",
    "plantdoc",
    "raw",
    "pages",  # kb/pages rendered pngs (can be regenerated or included with --include-pages)
}

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".pth",
    ".bin",
    ".pt",
    ".zip",
    ".tar",
    ".tar.gz",
    ".tmp",
}


def should_exclude(rel_path: Path) -> bool:
    """Determines whether a given relative path should be excluded from the submission zip."""
    parts = rel_path.parts

    # Exclude directories
    for part in parts:
        if part in EXCLUDE_DIRS:
            return True

    # Exclude large file extensions
    if rel_path.suffix.lower() in EXCLUDE_EXTENSIONS:
        return True

    # Exclude raw image datasets under data/
    if len(parts) >= 2 and parts[0] == "data" and parts[1] in ("raw", "plantvillage", "plantdoc", "images"):
        return True

    # Exclude model weights under checkpoints (keep configs/dummy)
    if len(parts) >= 2 and parts[0] == "checkpoints":
        if rel_path.suffix.lower() in (".pth", ".bin", ".safetensors"):
            return True

    return False


def create_submission_zip(group: str, title: str) -> Path:
    zip_filename = f"{group}_code_{title}.zip"
    zip_path = PROJECT_ROOT / zip_filename

    print("=" * 65)
    print(f"[PACKAGE] Building Academic Submission Archive: {zip_filename}")
    print(f"Project Directory: {PROJECT_ROOT}")
    print("=" * 65)

    included_count = 0
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(PROJECT_ROOT):
            # Prune excluded directories in-place to avoid walking deep raw trees
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for file in files:
                abs_path = Path(root) / file
                rel_path = abs_path.relative_to(PROJECT_ROOT)

                if should_exclude(rel_path) or rel_path.name == zip_filename:
                    continue

                zf.write(abs_path, str(rel_path))
                included_count += 1
                total_uncompressed_bytes += abs_path.stat().st_size

    compressed_size_mb = zip_path.stat().st_size / (1024 * 1024)
    uncompressed_size_mb = total_uncompressed_bytes / (1024 * 1024)

    print(f"\n[PACKAGE] Successfully created: {zip_path.name}")
    print(f"Files Included       : {included_count}")
    print(f"Uncompressed Size    : {uncompressed_size_mb:.2f} MB")
    print(f"Compressed Archive   : {compressed_size_mb:.2f} MB")
    print("=" * 65)
    return zip_path


def main():
    parser = argparse.ArgumentParser(description="Package project for submission.")
    parser.add_argument("group", nargs="?", default="GroupXX", help="Student Group Identifier (e.g., Group12)")
    parser.add_argument("title", nargs="?", default="Farmer_Advisory_Chatbot_Multilingual_NLP", help="Submission Title")
    args = parser.parse_args()

    create_submission_zip(args.group, args.title)


if __name__ == "__main__":
    main()
