"""
train_retriever.py
==================
Phase 4-D: Fine-tune multilingual dense retriever (multilingual-e5-small)
using MultipleNegativesRankingLoss (MNRL) on query-passage pairs.

Features:
  - Base: intfloat/multilingual-e5-small (query_prefix="query: ", passage_prefix="passage: ")
  - Loss: MultipleNegativesRankingLoss (in-batch negatives)
  - Mixed precision (fp16) on CUDA / AMP
  - Strict resume logic: fails loudly if checkpoint path is missing
  - Per-epoch CSV logging: epoch, train_loss, val_loss, val_MRR@10, val_Recall@1, val_Recall@5, lr, epoch_time_sec
  - Checkpoint saving: best (by val_MRR@10) and last
  - Seed embedded in all output filenames
  - Config snapshot JSON with package versions & hardware metadata
  - Local CPU smoke mode (--smoke): uses tiny local dummy transformer for fast, reliable verification

Usage:
  python train_retriever.py --config configs/retriever.yaml --smoke
  python train_retriever.py --config configs/retriever.yaml --epochs 5 --seed 42
"""

import argparse
import csv
import json
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from sentence_transformers import SentenceTransformer, losses, util
    from sentence_transformers.readers import InputExample
except ImportError:
    SentenceTransformer = None
    losses = None
    util = None
    InputExample = None


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
class PairedRetrievalDataset(Dataset):
    """Dataset yielding (query_with_prefix, passage_with_prefix, chunk_id)."""

    def __init__(
        self,
        pairs: List[Dict],
        chunks_dict: Dict[str, str],
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
    ):
        self.samples = []
        for p in pairs:
            cid = p.get("chunk_id", "")
            q_text = p.get("query", "").strip()
            if cid in chunks_dict and q_text:
                self.samples.append({
                    "query": f"{query_prefix}{q_text}",
                    "passage": f"{passage_prefix}{chunks_dict[cid]}",
                    "chunk_id": cid,
                    "source_id": p.get("source_id", ""),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def load_chunks_dict(chunks_path: Path) -> Dict[str, str]:
    """Loads chunk_id -> verbatim text mapping."""
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
    chunks_dict = {}
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                item = json.loads(line)
                cid = item.get("chunk_id")
                txt = item.get("text", "").strip()
                if cid and txt:
                    chunks_dict[cid] = txt
    return chunks_dict


def load_pairs_csv(csv_path: Path) -> List[Dict]:
    """Loads queries and associated chunk_ids from CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Pairs CSV not found: {csv_path}")
    pairs = []
    with open(csv_path, "r", encoding="utf-8") as f:
        # Filter lines starting with '#'
        lines = [line for line in f if not line.strip().startswith("#")]
        reader = csv.DictReader(lines)
        for r in reader:
            pairs.append(r)
    return pairs


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Validation Evaluation
# ---------------------------------------------------------------------------
def evaluate_retrieval(
    model: SentenceTransformer,
    val_dataset: PairedRetrievalDataset,
    all_chunks_dict: Dict[str, str],
    passage_prefix: str,
    device: str,
    batch_size: int = 32,
) -> Tuple[float, float, float, float]:
    """
    Evaluates MRR@10, Recall@1, Recall@5, and Mean Cosine Sim Loss on val set.
    """
    if len(val_dataset) == 0:
        return 0.0, 0.0, 0.0, 0.0

    # Collect unique candidate passages from val set
    target_cids = list({s["chunk_id"] for s in val_dataset.samples})
    candidate_cids = target_cids
    candidate_texts = [f"{passage_prefix}{all_chunks_dict[cid]}" for cid in candidate_cids]

    model.eval()
    with torch.no_grad():
        doc_embeddings = model.encode(
            candidate_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
            device=device,
        )

        queries = [s["query"] for s in val_dataset.samples]
        ground_truth_cids = [s["chunk_id"] for s in val_dataset.samples]

        query_embeddings = model.encode(
            queries,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
            device=device,
        )

        sim_matrix = util.cos_sim(query_embeddings, doc_embeddings).cpu().numpy()

    rec1 = 0.0
    rec5 = 0.0
    mrr10 = 0.0
    val_loss = 0.0

    for i, true_cid in enumerate(ground_truth_cids):
        scores = sim_matrix[i]
        ranked_indices = np.argsort(scores)[::-1]
        ranked_cids = [candidate_cids[idx] for idx in ranked_indices]

        # In-batch loss surrogate: 1.0 - positive score
        true_pos_idx = candidate_cids.index(true_cid) if true_cid in candidate_cids else -1
        if true_pos_idx != -1:
            val_loss += max(0.0, float(1.0 - scores[true_pos_idx]))

        # Recall@1
        if ranked_cids[0] == true_cid:
            rec1 += 1.0

        # Recall@5
        if true_cid in ranked_cids[:5]:
            rec5 += 1.0

        # MRR@10
        if true_cid in ranked_cids[:10]:
            rank = ranked_cids[:10].index(true_cid) + 1
            mrr10 += 1.0 / rank

    n = max(1, len(ground_truth_cids))
    return val_loss / n, mrr10 / n, rec1 / n, rec5 / n


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def train_retriever(
    config: dict,
    smoke: bool = False,
    override_epochs: int = None,
    override_batch_size: int = None,
    override_lr: float = None,
    output_root: Path = None,
    resume_path: Path = None,
    seed: int = 42,
):
    set_seed(seed)
    mode_tag = "[SMOKE_MODE]" if smoke else "[REAL_MODE]"
    device = "cuda" if torch.cuda.is_available() and not smoke else "cpu"
    print(f"\n{'='*70}\n{mode_tag} Starting Retriever Training | Device: {device} | Seed: {seed}\n{'='*70}")

    # Output paths setup
    exp_name = config.get("experiment_name", "e5_small_finetuned")
    run_name = f"smoke_{exp_name}_seed{seed}" if smoke else f"{exp_name}_seed{seed}"

    base_dir = output_root or PROJECT_ROOT
    log_dir = base_dir / "logs" / "retriever"
    ckpt_dir = base_dir / "checkpoints" / "retriever"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv_path = log_dir / f"{run_name}_metrics.csv"
    best_ckpt_dir = ckpt_dir / f"{run_name}_best"
    last_ckpt_dir = ckpt_dir / f"{run_name}_last"
    snapshot_json_path = log_dir / f"{run_name}_config_snapshot.json"

    # Hyperparameters
    epochs = override_epochs or (1 if smoke else config["training"].get("epochs", 5))
    batch_size = override_batch_size or (4 if smoke else config["training"].get("batch_size", 32))
    lr = override_lr or (1e-4 if smoke else float(config["training"].get("lr", 2e-5)))
    query_prefix = config["model"].get("query_prefix", "query: ")
    passage_prefix = config["model"].get("passage_prefix", "passage: ")

    # Data paths
    if smoke:
        chunks_file = PROJECT_ROOT / "kb" / "smoke_chunks.jsonl"
        train_csv = PROJECT_ROOT / "data" / "smoke_retrieval_train.csv"
        val_csv = PROJECT_ROOT / "data" / "smoke_retrieval_val.csv"
    else:
        chunks_file = PROJECT_ROOT / config["data"].get("kb_chunks", "kb/chunks.jsonl")
        train_csv = PROJECT_ROOT / config["data"].get("train_pairs", "data/retrieval_train.csv")
        val_csv = PROJECT_ROOT / config["data"].get("val_pairs", "data/retrieval_val.csv")

    chunks_dict = load_chunks_dict(chunks_file)
    train_pairs = load_pairs_csv(train_csv)
    val_pairs = load_pairs_csv(val_csv)

    if smoke:
        # Cap data for ultra-fast CPU smoke test
        train_pairs = train_pairs[:12]
        val_pairs = val_pairs[:6]

    print(f"Loaded KB chunks: {len(chunks_dict)} | Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)}")

    train_dataset = PairedRetrievalDataset(train_pairs, chunks_dict, query_prefix, passage_prefix)
    val_dataset = PairedRetrievalDataset(val_pairs, chunks_dict, query_prefix, passage_prefix)

    # Model initialization
    if smoke:
        dummy_model_dir = PROJECT_ROOT / "checkpoints" / "smoke_dummy_transformer"
        if not dummy_model_dir.exists():
            print("[SMOKE] Generating tiny dummy transformer for local smoke test...")
            from transformers import BertConfig, BertModel, BertTokenizerFast
            dummy_model_dir.mkdir(parents=True, exist_ok=True)
            tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
            cfg = BertConfig(
                vocab_size=len(tok.vocab),
                hidden_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                max_position_embeddings=128,
            )
            m = BertModel(cfg)
            m.save_pretrained(str(dummy_model_dir))
            tok.save_pretrained(str(dummy_model_dir))
        model = SentenceTransformer(str(dummy_model_dir), device=device)
    else:
        model_name = config["model"].get("name", "intfloat/multilingual-e5-small")
        print(f"Loading dense model: {model_name}")
        model = SentenceTransformer(model_name, device=device)

    # Convert training data to InputExamples
    train_examples = [
        InputExample(texts=[s["query"], s["passage"]]) for s in train_dataset.samples
    ]
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=model.smart_batching_collate,
        drop_last=True if len(train_examples) >= batch_size else False,
    )

    train_loss_fn = losses.MultipleNegativesRankingLoss(model)

    # Resume handling
    start_epoch = 1
    best_val_mrr = -1.0

    if resume_path:
        if not resume_path.exists():
            raise FileNotFoundError(f"FATAL: Specified --resume checkpoint path does not exist: {resume_path}")
        print(f"Resuming training from checkpoint: {resume_path}")
        model = SentenceTransformer(str(resume_path), device=device)
        if metrics_csv_path.exists():
            with open(metrics_csv_path, "r", encoding="utf-8") as f:
                rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
                if len(rows) > 1:
                    start_epoch = int(rows[-1][0]) + 1
                    print(f"Resuming from epoch {start_epoch}")

    # Prepare metrics CSV
    csv_header = ["epoch", "train_loss", "val_loss", "val_MRR@10", "val_Recall@1", "val_Recall@5", "lr", "epoch_time_sec"]
    if not metrics_csv_path.exists():
        with open(metrics_csv_path, "w", encoding="utf-8", newline="") as f:
            if smoke:
                f.write("# [SMOKE_MODE]\n")
            writer = csv.writer(f)
            writer.writerow(csv_header)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_dataloader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))

    # Snapshot JSON
    snapshot_data = {
        "run_name": run_name,
        "seed": seed,
        "mode": "smoke" if smoke else "real",
        "device": device,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "config": config,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(snapshot_json_path, "w", encoding="utf-8") as f:
        json.dump(snapshot_data, f, indent=2)

    # --- Training Loop ---
    print(f"\nBeginning training for {epochs} epochs...")
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" and config["training"].get("amp", True) else None

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        model.train()
        running_train_loss = 0.0
        steps = 0

        for batch in train_dataloader:
            optimizer.zero_grad()
            features, labels = batch
            if scaler:
                with torch.amp.autocast("cuda"):
                    loss_value = train_loss_fn(features, labels)
                scaler.scale(loss_value).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_value = train_loss_fn(features, labels)
                loss_value.backward()
                optimizer.step()

            scheduler.step()
            running_train_loss += loss_value.item()
            steps += 1

        avg_train_loss = running_train_loss / max(1, steps)
        current_lr = scheduler.get_last_lr()[0]

        # Evaluate on validation set
        val_loss, val_mrr, val_rec1, val_rec5 = evaluate_retrieval(
            model, val_dataset, chunks_dict, passage_prefix, device, batch_size=batch_size
        )

        epoch_time = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | MRR@10: {val_mrr:.4f} | R@1: {val_rec1:.4f} | "
            f"R@5: {val_rec5:.4f} | LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
        )

        # Write to metrics CSV
        with open(metrics_csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{avg_train_loss:.5f}",
                f"{val_loss:.5f}",
                f"{val_mrr:.5f}",
                f"{val_rec1:.5f}",
                f"{val_rec5:.5f}",
                f"{current_lr:.6e}",
                f"{epoch_time:.2f}",
            ])

        # Checkpoint saving
        model.save(str(last_ckpt_dir))
        if val_mrr >= best_val_mrr:
            best_val_mrr = val_mrr
            model.save(str(best_ckpt_dir))
            print(f"  ★ Best model saved to {best_ckpt_dir} (MRR@10: {val_mrr:.4f})")

    print("\nTraining completed successfully.")
    print(f"Metrics: {metrics_csv_path}")
    print(f"Best Checkpoint: {best_ckpt_dir}")
    print(f"Last Checkpoint: {last_ckpt_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train multilingual dense retriever.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true", help="Run 1-epoch CPU smoke test")
    group.add_argument("--real", action="store_true", help="Run real training run")
    parser.add_argument("--config", type=str, default="configs/retriever.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint directory to resume from")
    parser.add_argument("--output-root", type=str, default=None, help="Root dir for logs and checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"FATAL: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_root = Path(args.output_root) if args.output_root else None
    res_path = Path(args.resume) if args.resume else None

    train_retriever(
        config=config,
        smoke=args.smoke,
        override_epochs=args.epochs,
        override_batch_size=args.batch_size,
        override_lr=args.lr,
        output_root=out_root,
        resume_path=res_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
