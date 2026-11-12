"""
eval_vision.py
Evaluation script for trained crop disease vision models.

Computes:
- Top-1 Accuracy & Macro-F1 on In-Domain test set (PlantVillage)
- Top-1 Accuracy & Macro-F1 on Out-of-Domain test set (PlantDoc)
- Domain Transfer Gap: In-Domain Acc - Out-of-Domain Acc
- Per-class Precision, Recall, and F1
- Confusion Matrix (saved to CSV for plotting in Phase 7)
- Model parameter count and CPU inference latency

Usage:
    python eval/eval_vision.py --config configs/vision_efficientnet_b0.yaml --checkpoint checkpoints/vision/efficientnet_b0_plantvillage_best.pth
    python eval/eval_vision.py --config configs/vision_efficientnet_b0.yaml --smoke
"""

import argparse
import os
import sys
import json
import csv
from pathlib import Path
from collections import defaultdict
from PIL import Image

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_vision import CropDiseaseDataset, get_transforms, build_model, profile_cpu_latency


def evaluate_split(model, dataloader, num_classes: int, device: torch.device):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    acc = float(accuracy_score(all_targets, all_preds))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    prec, rec, f1, support = precision_recall_fscore_support(all_targets, all_preds, labels=list(range(num_classes)), zero_division=0)
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))

    per_class = {}
    for i in range(num_classes):
        per_class[i] = {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i])
        }

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist()
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained vision classifier.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pth)")
    parser.add_argument("--smoke", action="store_true", help="Run in CPU smoke mode")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    is_smoke = args.smoke
    run_name = config.get("experiment_name", "vision_run")
    if is_smoke:
        run_name = f"smoke_{run_name}"
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_dir = Path(config["checkpoint"].get("save_dir", "checkpoints/vision"))
        ckpt_path = ckpt_dir / f"{run_name}_best.pth"

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    print("=" * 75)
    print(f"EVALUATING VISION MODEL: {run_name}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Device: {device}")
    print("=" * 75)

    num_classes = config["model"].get("num_classes", 14)
    backbone = config["model"].get("backbone", "efficientnet_b0")

    model = build_model(backbone, num_classes=num_classes, drop_rate=0.0, pretrained=False)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # Datasets
    val_transform = get_transforms(config, is_train=False)
    test_dataset = CropDiseaseDataset(config["data"]["test_csv"], transform=val_transform)
    pd_dataset = CropDiseaseDataset(config["data"]["plantdoc_test_csv"], transform=val_transform)

    batch_size = 16 if is_smoke else 32
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    pd_loader = DataLoader(pd_dataset, batch_size=batch_size, shuffle=False)

    # Profile latency and params
    profile = profile_cpu_latency(model, img_size=config["data"].get("image_size", 224), iterations=10 if is_smoke else 30)

    # In-Domain Evaluation (PlantVillage Test)
    id_results = evaluate_split(model, test_loader, num_classes, device)
    print(f"\n[IN-DOMAIN: PlantVillage Test] (N={len(test_dataset)})")
    print(f"  - Top-1 Accuracy: {id_results['accuracy']*100:.2f}%")
    print(f"  - Macro-F1:       {id_results['macro_f1']:.4f}")

    # Out-of-Domain Evaluation (PlantDoc Test)
    ood_results = evaluate_split(model, pd_loader, num_classes, device)
    print(f"\n[OUT-OF-DOMAIN: PlantDoc Test] (N={len(pd_dataset)})")
    print(f"  - Top-1 Accuracy: {ood_results['accuracy']*100:.2f}%")
    print(f"  - Macro-F1:       {ood_results['macro_f1']:.4f}")

    domain_gap = round(id_results["accuracy"] - ood_results["accuracy"], 4)
    print(f"\n[DOMAIN GAP] In-Domain Acc - Out-of-Domain Acc: {domain_gap*100:+.2f}%")

    # Export Evaluation Report
    log_dir = Path(config["logging"].get("log_dir", "logs/vision"))
    eval_report_path = log_dir / f"{run_name}_eval_results.json"
    report_data = {
        "run_name": run_name,
        "backbone": backbone,
        "checkpoint": str(ckpt_path),
        "is_smoke": is_smoke,
        "profile": profile,
        "in_domain_test": id_results,
        "out_of_domain_plantdoc": ood_results,
        "domain_transfer_gap_acc": domain_gap
    }

    with open(eval_report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # Save In-Domain Confusion Matrix CSV
    cm_path = log_dir / f"{run_name}_confusion_matrix.csv"
    with open(cm_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(id_results["confusion_matrix"])

    print(f"\nEvaluation completed! Full report saved to: {eval_report_path}")
    print(f"Confusion matrix CSV saved to: {cm_path}")


if __name__ == "__main__":
    main()
