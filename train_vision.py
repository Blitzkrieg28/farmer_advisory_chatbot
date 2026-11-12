"""
train_vision.py
Fine-tunes ImageNet-pretrained vision backbones for crop disease diagnosis.

Features:
- Backbones supported: EfficientNet-B0, ResNet-18, MobileNetV3 (via timm)
- Configurable class imbalance strategies: 'weighted_loss', 'weighted_sampler', 'none'
- Stratified evaluation across Validation, PlantVillage Test, and PlantDoc (out-of-domain)
- Per-epoch metrics logging: epoch, train_loss, train_loss_unweighted, val_loss, train_acc, val_acc, lr, epoch_time_sec
- Best checkpoint (by lowest val loss, storing best_val_loss) and last checkpoint saving
- Resumable training via --resume (validates path existence, restores best_val_loss, recreates CSV header if missing)
- Seed included in run_name and ALL output filenames (metrics, ckpts, preds, profile, snapshot)
- Latency profiler isolates a deepcopy CPU model with explicit torch.set_num_threads and hardware metadata
- CLI overrides: --epochs, --max_images_per_class (seeded, logged), --output_root (e.g. Google Drive)
- Strict CPU smoke test mode via --smoke (1 epoch, synthetic data, isolated smoke outputs)

Usage:
    # Full GPU training (Colab):
    python train_vision.py --config configs/vision_efficientnet_b0.yaml

    # Quick real-mode sanity check (1 epoch, 50 images per class):
    python train_vision.py --config configs/vision_efficientnet_b0.yaml --epochs 1 --max_images_per_class 50

    # CPU Smoke test (local laptop):
    python train_vision.py --config configs/vision_efficientnet_b0.yaml --smoke

    # Resume training:
    python train_vision.py --config configs/vision_efficientnet_b0.yaml --resume checkpoints/vision/efficientnet_b0_plantvillage_seed42_last.pth
"""

import argparse
import os
import sys
import time
import json
import random
import csv
import copy
import platform
import subprocess
import contextlib
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


# ---------------------------------------------------------------------------
# UTILITIES & REPRODUCIBILITY
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_git_commit() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return commit
    except Exception:
        return "not_a_git_repo_or_git_unavailable"


# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------
class CropDiseaseDataset(Dataset):
    """Dataset loading images from CSV file (filepath, label, class_name, crop)."""
    def __init__(self, csv_path: str, transform=None, max_images_per_class: int = None, seed: int = 42):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Missing dataset CSV: {self.csv_path}")

        raw_records = []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            valid_lines = [line for line in f if not line.strip().startswith("#")]
            reader = csv.DictReader(valid_lines)
            for row in reader:
                raw_records.append({
                    "filepath": row["filepath"],
                    "label": int(row["label"]),
                    "class_name": row["class_name"],
                    "crop": row["crop"]
                })

        # Seeded, logged sampling if max_images_per_class is specified
        if max_images_per_class is not None and max_images_per_class > 0:
            by_class = defaultdict(list)
            for r in raw_records:
                by_class[r["label"]].append(r)

            capped_records = []
            rng = random.Random(seed)
            sampled_counts = {}

            for lbl, items in sorted(by_class.items()):
                if len(items) > max_images_per_class:
                    sampled = rng.sample(items, max_images_per_class)
                    capped_records.extend(sampled)
                    sampled_counts[items[0]["class_name"]] = f"{len(sampled)}/{len(items)}"
                else:
                    capped_records.extend(items)
                    sampled_counts[items[0]["class_name"]] = f"{len(items)}/{len(items)}"

            print(f"[SAMPLING] Capped dataset to max {max_images_per_class} per class (seed={seed}): {len(capped_records)} total from {len(raw_records)}")
            self.records = capped_records
            self.sampling_log = {
                "max_images_per_class": max_images_per_class,
                "seed": seed,
                "original_total": len(raw_records),
                "capped_total": len(capped_records),
                "per_class_sampled": sampled_counts
            }
        else:
            self.records = raw_records
            self.sampling_log = None

        self.transform = transform
        self.labels = [r["label"] for r in self.records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = rec["filepath"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")

        if self.transform:
            image = self.transform(image)

        return image, rec["label"], img_path


# ---------------------------------------------------------------------------
# TRANSFORMS
# ---------------------------------------------------------------------------
def get_transforms(config: dict, is_train: bool = True):
    img_size = config["data"].get("image_size", 224)
    aug_cfg = config.get("augmentation", {})

    if is_train and aug_cfg.get("enabled", True):
        t_list = [
            transforms.RandomResizedCrop(
                img_size,
                scale=tuple(aug_cfg.get("random_resized_crop", {}).get("scale", [0.7, 1.0]))
            )
        ]
        if aug_cfg.get("horizontal_flip", True):
            t_list.append(transforms.RandomHorizontalFlip(p=0.5))

        cj = aug_cfg.get("color_jitter", {})
        if cj:
            t_list.append(transforms.ColorJitter(
                brightness=cj.get("brightness", 0.3),
                contrast=cj.get("contrast", 0.3),
                saturation=cj.get("saturation", 0.2),
                hue=cj.get("hue", 0.1)
            ))

        gb = aug_cfg.get("gaussian_blur", {})
        if gb and gb.get("probability", 0.0) > 0:
            t_list.append(transforms.RandomApply([
                transforms.GaussianBlur(
                    kernel_size=gb.get("kernel_size", 3),
                    sigma=tuple(gb.get("sigma", [0.1, 2.0]))
                )
            ], p=gb.get("probability", 0.3)))

        t_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transforms.Compose(t_list)
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


# ---------------------------------------------------------------------------
# MODEL BUILDER
# ---------------------------------------------------------------------------
def build_model(backbone: str, num_classes: int, drop_rate: float = 0.2, pretrained: bool = True, is_smoke: bool = False):
    """
    Creates classifier using timm.
    In real mode: fails loudly if timm model creation fails (no silent fallback).
    In smoke mode: allows lightweight fallback if timm is not installed or offline.
    """
    if HAS_TIMM:
        try:
            model = timm.create_model(
                backbone,
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=drop_rate
            )
            return model
        except Exception as e:
            if not is_smoke:
                raise RuntimeError(f"FATAL: timm failed to create model '{backbone}': {e}. Real training requires a valid timm backbone.")
            print(f"[SMOKE] timm creation error: {e}. Falling back to torchvision for smoke test.")

    # Smoke mode torchvision fallback
    import torchvision.models as tv_models
    if "resnet18" in backbone:
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        model = tv_models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif "mobilenet" in backbone:
        weights = tv_models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = tv_models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif "efficientnet" in backbone:
        weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = tv_models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=drop_rate, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    return model


# ---------------------------------------------------------------------------
# CLASS IMBALANCE HANDLING
# ---------------------------------------------------------------------------
def setup_imbalance_handling(labels: list, num_classes: int, strategy: str, device: torch.device):
    """
    Sets up class imbalance handling.
    strategy: 'weighted_loss', 'weighted_sampler', or 'none'.
    Returns: (criterion, unweighted_criterion, sampler)
    """
    label_counts = Counter(labels)
    total_samples = len(labels)
    class_weights = torch.zeros(num_classes, dtype=torch.float32)

    for cls_idx in range(num_classes):
        cnt = label_counts.get(cls_idx, 0)
        class_weights[cls_idx] = total_samples / (num_classes * max(cnt, 1))

    unweighted_criterion = nn.CrossEntropyLoss()
    sampler = None

    if strategy == "weighted_loss":
        print(f"[IMBALANCE] Strategy: 'weighted_loss'. Class weights min={class_weights.min():.2f}, max={class_weights.max():.2f}")
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    elif strategy == "weighted_sampler":
        print(f"[IMBALANCE] Strategy: 'weighted_sampler'.")
        sample_weights = [class_weights[lbl].item() for lbl in labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        criterion = nn.CrossEntropyLoss()
    else:
        print("[IMBALANCE] Strategy: 'none' (standard CrossEntropyLoss).")
        criterion = nn.CrossEntropyLoss()

    return criterion, unweighted_criterion, sampler


# ---------------------------------------------------------------------------
# LATENCY PROFILER
# ---------------------------------------------------------------------------
def profile_cpu_latency(model: nn.Module, img_size: int = 224, iterations: int = 30, num_threads: int = 4) -> dict:
    """
    Measures mean CPU inference latency per single image (ms) using an isolated CPU model copy.
    Sets explicit thread count and records thread count and hardware metadata.
    """
    # Profile a CPU copy so it functions seamlessly even when model is on CUDA
    cpu_model = copy.deepcopy(model).to("cpu")
    cpu_model.eval()

    orig_threads = torch.get_num_threads()
    eval_threads = min(num_threads, orig_threads) if num_threads else 1
    torch.set_num_threads(eval_threads)

    dummy_input = torch.randn(1, 3, img_size, img_size, device="cpu")

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = cpu_model(dummy_input)

    times = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.perf_counter()
            _ = cpu_model(dummy_input)
            end = time.perf_counter()
            times.append((end - start) * 1000.0)

    torch.set_num_threads(orig_threads)

    mean_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    total_params = sum(p.numel() for p in cpu_model.parameters())
    trainable_params = sum(p.numel() for p in cpu_model.parameters() if p.requires_grad)

    return {
        "mean_latency_ms": round(mean_ms, 2),
        "std_latency_ms": round(std_ms, 2),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "benchmark_iterations": iterations,
        "cpu_threads_used": eval_threads,
        "hardware_metadata": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "python_version": platform.python_version()
        }
    }


# ---------------------------------------------------------------------------
# SOFTMAX PROBABILITIES EXPORT (FOR CRITIC)
# ---------------------------------------------------------------------------
def export_predictions_and_probabilities(model: nn.Module, dataloader: DataLoader, output_csv: Path, num_classes: int, device: torch.device):
    """
    Evaluates dataset and exports true label, predicted label, all softmax probs, top-1, top-2, and margin.
    Required by Phase 5 Critic.
    """
    model.eval()
    records = []

    prob_headers = [f"prob_class_{i}" for i in range(num_classes)]
    headers = ["filepath", "true_label", "predicted_label"] + prob_headers + ["top1_prob", "top2_prob", "margin"]

    with torch.no_grad():
        for images, labels, paths in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=-1).cpu().numpy()
            labels = labels.numpy()

            for i in range(len(paths)):
                p = probs[i]
                sorted_p = np.sort(p)[::-1]
                top1 = float(sorted_p[0])
                top2 = float(sorted_p[1]) if len(sorted_p) > 1 else 0.0
                margin = float(top1 - top2)
                pred_label = int(np.argmax(p))

                row = {
                    "filepath": paths[i],
                    "true_label": int(labels[i]),
                    "predicted_label": pred_label,
                    "top1_prob": round(top1, 6),
                    "top2_prob": round(top2, 6),
                    "margin": round(margin, 6)
                }
                for c_idx in range(num_classes):
                    row[f"prob_class_{c_idx}"] = round(float(p[c_idx]), 6)

                records.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(records)

    print(f"Exported {len(records)} prediction probabilities -> {output_csv}")


# ---------------------------------------------------------------------------
# TRAINING & EVALUATION LOOPS
# ---------------------------------------------------------------------------
def train_epoch(model, dataloader, criterion, unweighted_criterion, optimizer, scaler, autocast_ctx, device, clip_norm: float = 1.0):
    model.train()
    running_loss = 0.0
    running_unweighted_loss = 0.0
    correct = 0
    total = 0

    for images, targets, _ in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None and device.type == "cuda":
            with autocast_ctx:
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            if clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()

        # Compute unweighted train loss for strict comparability with validation loss
        with torch.no_grad():
            unweighted_loss = unweighted_criterion(outputs.detach(), targets)

        running_loss += loss.item() * images.size(0)
        running_unweighted_loss += unweighted_loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == targets).item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_unweighted_loss = running_unweighted_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_unweighted_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets, _ in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == targets).item()
            total += targets.size(0)

    val_loss = running_loss / total if total > 0 else 0.0
    val_acc = correct / total if total > 0 else 0.0
    return val_loss, val_acc


# ---------------------------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train crop disease vision classifier.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--smoke", action="store_true", help="Run 1-epoch CPU smoke test on synthetic data")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from (must exist)")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed (e.g., 42, 43, 44)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--max_images_per_class", type=int, default=None, help="Cap training images per class (seeded, logged)")
    parser.add_argument("--output_root", type=str, default=None, help="Root directory for logs and checkpoints (e.g. Google Drive)")
    args = parser.parse_args()

    # Load YAML config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. Seed & run_name resolution (seed is included in run_name and ALL files)
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    set_seed(seed)

    base_exp_name = config.get("experiment_name", "vision_run")
    is_smoke = args.smoke

    if is_smoke:
        run_name = f"smoke_{base_exp_name}_seed{seed}"
        device = torch.device("cpu")
        print("\n" + "=" * 75)
        print(f"[SMOKE MODE] Running CPU smoke test on run: {run_name}")
        print("Enforcing: device=CPU, epochs=1, small batch size, smoke-labeled outputs.")
        print("=" * 75)
    else:
        run_name = f"{base_exp_name}_seed{seed}"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[TRAINING] Run: {run_name} | Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 2. Output directory resolution (--output_root support)
    if args.output_root:
        out_root = Path(args.output_root)
        log_dir = out_root / "logs" / "vision"
        ckpt_dir = out_root / "checkpoints" / "vision"
    else:
        log_dir = Path(config["logging"].get("log_dir", "logs/vision"))
        ckpt_dir = Path(config["checkpoint"].get("save_dir", "checkpoints/vision"))

    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv_path = log_dir / f"{run_name}_metrics.csv"
    best_ckpt_path = ckpt_dir / f"{run_name}_best.pth"
    last_ckpt_path = ckpt_dir / f"{run_name}_last.pth"
    snapshot_path = log_dir / f"{run_name}_config_snapshot.json"

    # 3. Datasets & Sampling
    data_cfg = config["data"]
    num_classes = config["model"].get("num_classes", 14)
    max_img_cap = args.max_images_per_class if args.max_images_per_class is not None else data_cfg.get("max_images_per_class")

    train_transform = get_transforms(config, is_train=True)
    val_transform = get_transforms(config, is_train=False)

    train_dataset = CropDiseaseDataset(
        data_cfg["train_csv"],
        transform=train_transform,
        max_images_per_class=max_img_cap,
        seed=seed
    )
    val_dataset = CropDiseaseDataset(data_cfg["val_csv"], transform=val_transform)
    test_dataset = CropDiseaseDataset(data_cfg["test_csv"], transform=val_transform)
    plantdoc_test_dataset = CropDiseaseDataset(data_cfg["plantdoc_test_csv"], transform=val_transform)

    print(f"Dataset Loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}, PlantDoc OOD Test={len(plantdoc_test_dataset)}")

    # 4. Class imbalance handling
    train_cfg = config["training"]
    imbalance_strategy = train_cfg.get("imbalance_strategy", "weighted_loss")
    criterion, unweighted_criterion, sampler = setup_imbalance_handling(train_dataset.labels, num_classes, imbalance_strategy, device)
    val_criterion = nn.CrossEntropyLoss()

    batch_size = 16 if is_smoke else train_cfg.get("batch_size", 32)
    num_workers = 0 if is_smoke else data_cfg.get("num_workers", 2)
    # drop_last=True for train DataLoader when dataset size >= batch_size
    drop_last = (len(train_dataset) >= batch_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    plantdoc_loader = DataLoader(plantdoc_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 5. Build model
    backbone = config["model"].get("backbone", "efficientnet_b0")
    backbone_variant = config["model"].get("backbone_variant", backbone)
    drop_rate = config["model"].get("drop_rate", 0.2)
    pretrained = False if is_smoke else config["model"].get("pretrained", True)

    model = build_model(backbone, num_classes=num_classes, drop_rate=drop_rate, pretrained=pretrained, is_smoke=is_smoke)
    model = model.to(device)

    # 6. Optimizer & Warmup Scheduler
    lr = float(train_cfg.get("lr", 3.0e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1.0e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_epochs = 1 if is_smoke else (args.epochs if args.epochs is not None else int(train_cfg.get("epochs", 20)))
    warmup_epochs = int(train_cfg.get("warmup_epochs", 1))

    # Scheduler with warmup
    if warmup_epochs > 0 and total_epochs > warmup_epochs:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(total_epochs - warmup_epochs), eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

    # 7. Modern AMP setup
    amp_enabled = train_cfg.get("amp", True) and (device.type == "cuda")
    if amp_enabled:
        try:
            autocast_ctx = torch.amp.autocast(device_type="cuda")
            scaler = torch.amp.GradScaler(device_type="cuda")
        except Exception:
            autocast_ctx = torch.cuda.amp.autocast()
            scaler = torch.cuda.amp.GradScaler()
    else:
        autocast_ctx = contextlib.nullcontext()
        scaler = None

    # 8. Resume validation & restoration
    start_epoch = 1
    best_val_loss = float("inf")
    resume_path = args.resume or config["checkpoint"].get("resume")

    if resume_path:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"FATAL: Specified --resume checkpoint path does not exist: {resume_path}")

        print(f"Resuming training from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", checkpoint.get("val_loss", float("inf")))
        print(f"Resumed successfully at epoch {start_epoch} (best_val_loss={best_val_loss:.4f})")

        # If metrics CSV is missing on resume, recreate the header
        if not os.path.exists(metrics_csv_path):
            with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
                if is_smoke:
                    f.write("# [SMOKE_MODE] Synthetic CPU data only. NOT REAL RESULTS.\n")
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "train_loss_unweighted", "val_loss", "train_acc", "val_acc", "lr", "epoch_time_sec"])
    else:
        # Initialize new metrics CSV with header
        with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
            if is_smoke:
                f.write("# [SMOKE_MODE] Synthetic CPU data only. NOT REAL RESULTS.\n")
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_loss_unweighted", "val_loss", "train_acc", "val_acc", "lr", "epoch_time_sec"])

    # 9. Config snapshot export
    snapshot_data = {
        "run_name": run_name,
        "base_experiment": base_exp_name,
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "is_smoke": is_smoke,
        "git_commit": get_git_commit(),
        "resolved_config": config,
        "environment": {
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__ if HAS_TORCHVISION else "not_installed",
            "timm_version": timm.__version__ if HAS_TIMM else "not_installed",
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version()
        },
        "model": {
            "backbone": backbone,
            "backbone_variant": backbone_variant,
            "num_classes": num_classes,
            "drop_rate": drop_rate
        },
        "training_params": {
            "epochs": total_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "warmup_epochs": warmup_epochs,
            "imbalance_strategy": imbalance_strategy,
            "drop_last": drop_last,
            "max_images_per_class": max_img_cap
        },
        "sampling_log": train_dataset.sampling_log
    }

    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(snapshot_data, f, indent=2)
    print(f"Config snapshot written -> {snapshot_path}")

    # 10. Profile CPU latency and parameters
    profile_info = profile_cpu_latency(model, img_size=data_cfg.get("image_size", 224), iterations=10 if is_smoke else 30)
    print(f"\nModel: {backbone} ({backbone_variant}) | Total Params: {profile_info['total_parameters']:,} | CPU Latency: {profile_info['mean_latency_ms']} ms/img ({profile_info['cpu_threads_used']} threads)")

    # -------------------------------------------------------------
    # TRAINING LOOP
    # -------------------------------------------------------------
    clip_norm = float(train_cfg.get("gradient_clip_max_norm", 1.0))

    for epoch in range(start_epoch, total_epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_loss_unweighted, train_acc = train_epoch(
            model, train_loader, criterion, unweighted_criterion, optimizer, scaler, autocast_ctx, device, clip_norm
        )
        val_loss, val_acc = evaluate(model, val_loader, val_criterion, device)
        scheduler.step()

        epoch_time = time.time() - t0

        print(f"Epoch {epoch:02d}/{total_epochs:02d} | Train Loss (w): {train_loss:.4f} | Train Loss (unw): {train_loss_unweighted:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% | LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")

        # Append to metrics CSV
        with open(metrics_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                round(train_loss, 5),
                round(train_loss_unweighted, 5),
                round(val_loss, 5),
                round(train_acc, 5),
                round(val_acc, 5),
                f"{current_lr:.6e}",
                round(epoch_time, 2)
            ])

        # Checkpoint dict
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_loss": min(best_val_loss, val_loss),
            "config": config,
            "seed": seed,
            "backbone_variant": backbone_variant
        }

        # Last checkpoint
        torch.save(ckpt_data, last_ckpt_path)

        # Best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_data["best_val_loss"] = best_val_loss
            torch.save(ckpt_data, best_ckpt_path)
            print(f"  -> Saved new best checkpoint to {best_ckpt_path} (val_loss: {val_loss:.4f})")

    # -------------------------------------------------------------
    # EXPORT PREDICTIONS & PROBABILITIES (FOR CRITIC EVALUATION)
    # -------------------------------------------------------------
    print("\n--- Evaluating Best Model and Exporting Prediction Probabilities ---")
    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    preds_val_csv = log_dir / f"{run_name}_preds_val.csv"
    preds_test_csv = log_dir / f"{run_name}_preds_test.csv"
    preds_plantdoc_csv = log_dir / f"{run_name}_preds_plantdoc_test.csv"

    export_predictions_and_probabilities(model, val_loader, preds_val_csv, num_classes, device)
    export_predictions_and_probabilities(model, test_loader, preds_test_csv, num_classes, device)
    export_predictions_and_probabilities(model, plantdoc_loader, preds_plantdoc_csv, num_classes, device)

    # Save profile JSON
    profile_path = log_dir / f"{run_name}_profile.json"
    profile_data = {
        "run_name": run_name,
        "backbone": backbone,
        "backbone_variant": backbone_variant,
        "seed": seed,
        "num_classes": num_classes,
        "is_smoke": is_smoke,
        "profile": profile_info,
        "best_val_loss": best_val_loss,
        "metrics_csv": str(metrics_csv_path),
        "best_checkpoint": str(best_ckpt_path)
    }
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile_data, f, indent=2)

    print(f"Training run completed! Profile saved to: {profile_path}")


if __name__ == "__main__":
    main()
