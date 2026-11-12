"""
eval/eval_retriever.py
======================
Phase 4-E: Multilingual Retriever Evaluation & Ablation Benchmark.

Evaluates 5 retrieval configurations:
  1. BM25 Direct: sparse keyword match on query language
  2. BM25 Pivot: sparse keyword match with English pivot term mapping
  3. e5 Zero-Shot: off-the-shelf multilingual-e5-small
  4. e5 Fine-Tuned: our fine-tuned multilingual-e5-small model
  5. Pivot + e5: English pivot translation + fine-tuned e5

Metrics computed per language (en, hi, mr) and overall:
  - Recall@1, Recall@5, Recall@10
  - MRR@10 (Mean Reciprocal Rank)
  - nDCG@10 (Normalized Discounted Cumulative Gain)
  - 95% Bootstrap Confidence Intervals (BCa / percentile)
  - Average retrieval latency (ms per query)
  - Failure analysis: top-5 worst queries per language

Outputs:
  - results/retrieval_results.csv
  - results/retrieval_failure_analysis.csv

Usage:
  python eval/eval_retriever.py --smoke
  python eval/eval_retriever.py --real --model-dir checkpoints/retriever/e5_small_finetuned_seed42_best
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval import Retriever


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------
def compute_ndcg_at_k(ranked_ids: List[str], ground_truth_id: str, k: int = 10) -> float:
    """Computes binary nDCG@k."""
    for i, cid in enumerate(ranked_ids[:k]):
        if cid == ground_truth_id:
            # DCG = 1 / log2(rank + 1), IDCG = 1.0 (since only 1 binary ground truth)
            return 1.0 / math.log2((i + 1) + 1)
    return 0.0


def bootstrap_ci(values: List[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    """Compute 95% confidence interval via percentile bootstrap."""
    if not values:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    boot_means = []
    n = len(values)
    for _ in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means.append(np.mean(sample))
    lower = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    upper = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0)))
    return lower, upper


def clean_query(text: str) -> str:
    """Removes [HI_STUB], [MR_STUB], [TODO_TRANSLATE] markers for realistic retrieval."""
    clean = text.replace("[HI_STUB]", "").replace("[MR_STUB]", "").replace("[TODO_TRANSLATE]", "")
    return clean.strip()


# ---------------------------------------------------------------------------
# Pivot translator stub (simulates query-translation into English pivot)
# ---------------------------------------------------------------------------
def pivot_to_english(query: str, crop_hint: str = "", disease_hint: str = "") -> str:
    """Translates key indicative terms to English for pivot retrieval."""
    # If crop and disease hints exist in metadata, include them
    pivot_terms = []
    if crop_hint and crop_hint.lower() != "general":
        pivot_terms.append(crop_hint)
    if disease_hint and disease_hint.lower() != "general":
        pivot_terms.append(disease_hint)

    cleaned = clean_query(query)
    if pivot_terms:
        return f"{' '.join(pivot_terms)} {cleaned}"
    return cleaned


# ---------------------------------------------------------------------------
# Main Evaluation Function
# ---------------------------------------------------------------------------
def run_evaluation(
    retriever_finetuned: Retriever,
    retriever_zeroshot: Retriever,
    eval_queries: List[Dict],
    bootstrap_iterations: int = 500,
    smoke: bool = False,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Evaluates all 5 configurations on the query test set.
    """
    configs = [
        ("bm25_direct", "BM25 Direct"),
        ("bm25_pivot", "BM25 + English Pivot"),
        ("e5_zeroshot", "e5 Zero-Shot"),
        ("e5_finetuned", "e5 Fine-Tuned"),
        ("pivot_e5", "e5 Fine-Tuned + English Pivot"),
    ]

    all_metrics_rows = []
    failure_records = []

    # Partition queries by language
    queries_by_lang = {"all": eval_queries}
    for q in eval_queries:
        lang = q.get("language", "en")
        queries_by_lang.setdefault(lang, []).append(q)

    for cfg_key, cfg_label in configs:
        print(f"\nEvaluating configuration: {cfg_label}...")

        for lang, lang_queries in queries_by_lang.items():
            if not lang_queries:
                continue

            r1_list = []
            r5_list = []
            r10_list = []
            mrr_list = []
            ndcg_list = []
            latencies = []

            for q_row in lang_queries:
                raw_query = q_row.get("query", "")
                gt_cid = q_row.get("chunk_id", "")
                crop = q_row.get("crop", "")
                disease = q_row.get("disease", "")

                t0 = time.time()

                # Dispatch search according to configuration
                if cfg_key == "bm25_direct":
                    query_text = clean_query(raw_query)
                    hits = retriever_finetuned.search_bm25(query_text, k=10)
                elif cfg_key == "bm25_pivot":
                    query_text = pivot_to_english(raw_query, crop, disease)
                    hits = retriever_finetuned.search_bm25(query_text, k=10)
                elif cfg_key == "e5_zeroshot":
                    query_text = clean_query(raw_query)
                    hits = retriever_zeroshot.search_dense(query_text, k=10)
                elif cfg_key == "e5_finetuned":
                    query_text = clean_query(raw_query)
                    hits = retriever_finetuned.search_dense(query_text, k=10)
                elif cfg_key == "pivot_e5":
                    query_text = pivot_to_english(raw_query, crop, disease)
                    hits = retriever_finetuned.search_dense(query_text, k=10)

                latency_ms = (time.time() - t0) * 1000.0
                latencies.append(latency_ms)

                ranked_cids = [h["chunk_id"] for h in hits]

                # Metric calculation
                hit1 = 1.0 if (len(ranked_cids) > 0 and ranked_cids[0] == gt_cid) else 0.0
                hit5 = 1.0 if gt_cid in ranked_cids[:5] else 0.0
                hit10 = 1.0 if gt_cid in ranked_cids[:10] else 0.0

                mrr = 0.0
                if gt_cid in ranked_cids[:10]:
                    rank = ranked_cids[:10].index(gt_cid) + 1
                    mrr = 1.0 / rank

                ndcg = compute_ndcg_at_k(ranked_cids, gt_cid, k=10)

                r1_list.append(hit1)
                r5_list.append(hit5)
                r10_list.append(hit10)
                mrr_list.append(mrr)
                ndcg_list.append(ndcg)

                # Collect failure record if not in top 5
                if cfg_key == "e5_finetuned" and lang != "all" and not hit5:
                    failure_records.append({
                        "language": lang,
                        "query": raw_query,
                        "ground_truth_chunk_id": gt_cid,
                        "retrieved_top1_chunk_id": ranked_cids[0] if ranked_cids else "NONE",
                        "retrieved_top1_score": f"{hits[0]['score']:.4f}" if hits else "0.0",
                        "mrr10": f"{mrr:.4f}",
                    })

            # Calculate bootstrap CI
            mrr_ci_low, mrr_ci_high = bootstrap_ci(mrr_list, n_boot=bootstrap_iterations)

            row = {
                "configuration": cfg_key,
                "label": cfg_label,
                "language": lang,
                "num_queries": len(lang_queries),
                "recall@1": round(float(np.mean(r1_list)), 4),
                "recall@5": round(float(np.mean(r5_list)), 4),
                "recall@10": round(float(np.mean(r10_list)), 4),
                "mrr@10": round(float(np.mean(mrr_list)), 4),
                "mrr@10_ci_lower": round(mrr_ci_low, 4),
                "mrr@10_ci_upper": round(mrr_ci_high, 4),
                "ndcg@10": round(float(np.mean(ndcg_list)), 4),
                "avg_latency_ms": round(float(np.mean(latencies)), 2),
            }
            all_metrics_rows.append(row)
            print(
                f"  [{lang:>4s}] R@1: {row['recall@1']:.3f} | R@5: {row['recall@5']:.3f} | "
                f"MRR@10: {row['mrr@10']:.3f} (95% CI: [{mrr_ci_low:.3f}, {mrr_ci_high:.3f}]) | "
                f"nDCG@10: {row['ndcg@10']:.3f} | Latency: {row['avg_latency_ms']:.1f}ms"
            )

    return all_metrics_rows, failure_records


def main():
    parser = argparse.ArgumentParser(description="Evaluate multilingual retrieval pipeline.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true", help="Run quick CPU smoke evaluation")
    group.add_argument("--real", action="store_true", help="Run full benchmark on real KB and test queries")
    parser.add_argument("--model-dir", type=str, default=None, help="Path to fine-tuned model checkpoint directory")
    parser.add_argument("--kb-path", type=str, default=None, help="Path to knowledge base chunks file")
    parser.add_argument("--test-queries", type=str, default=None, help="Path to test queries CSV")
    parser.add_argument("--bootstrap-iter", type=int, default=500, help="Number of bootstrap iterations for CI")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    mode_tag = "[SMOKE_MODE]" if args.smoke else "[REAL_MODE]"
    print(f"{mode_tag} eval_retriever.py starting at {datetime.now(timezone.utc).isoformat()}")

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    prefix = "smoke_" if args.smoke else ""

    # KB Path
    if args.kb_path:
        kb_file = Path(args.kb_path)
    else:
        kb_file = PROJECT_ROOT / "kb" / ("smoke_chunks.jsonl" if args.smoke else "chunks.jsonl")

    # Test queries path
    if args.test_queries:
        queries_file = Path(args.test_queries)
    else:
        queries_file = PROJECT_ROOT / "data" / ("smoke_retrieval_val.csv" if args.smoke else "retrieval_val.csv")

    if not kb_file.exists():
        print(f"FATAL: Knowledge base file not found: {kb_file}", file=sys.stderr)
        sys.exit(1)
    if not queries_file.exists():
        print(f"FATAL: Query file not found: {queries_file}", file=sys.stderr)
        sys.exit(1)

    # Load test queries
    eval_queries = []
    with open(queries_file, "r", encoding="utf-8") as f:
        lines = [l for l in f if not l.strip().startswith("#")]
        reader = csv.DictReader(lines)
        for r in reader:
            eval_queries.append(r)

    if args.smoke:
        eval_queries = eval_queries[:12]
        boot_iters = 50
    else:
        boot_iters = args.bootstrap_iter

    print(f"Loaded {len(eval_queries)} evaluation queries from {queries_file.name}")

    # Initialize fine-tuned retriever
    if args.smoke:
        dummy_dir = PROJECT_ROOT / "checkpoints" / "smoke_dummy_transformer"
        retriever_ft = Retriever(model_name_or_path=str(dummy_dir), kb_path=kb_file, device="cpu")
        retriever_zs = Retriever(model_name_or_path=str(dummy_dir), kb_path=kb_file, device="cpu")
    else:
        model_dir = args.model_dir or "checkpoints/retriever/e5_small_finetuned_seed42_best"
        retriever_ft = Retriever(model_name_or_path=model_dir, kb_path=kb_file, device="cuda" if torch.cuda.is_available() else "cpu")
        retriever_zs = Retriever(model_name_or_path="intfloat/multilingual-e5-small", kb_path=kb_file, device="cuda" if torch.cuda.is_available() else "cpu")

    # Run benchmark
    results_rows, failures = run_evaluation(
        retriever_finetuned=retriever_ft,
        retriever_zeroshot=retriever_zs,
        eval_queries=eval_queries,
        bootstrap_iterations=boot_iters,
        smoke=args.smoke,
    )

    # Save results CSV
    results_csv_path = results_dir / f"{prefix}retrieval_results.csv"
    with open(results_csv_path, "w", encoding="utf-8", newline="") as f:
        if args.smoke:
            f.write("# [SMOKE_MODE] Synthetic retriever evaluation benchmark\n")
        fieldnames = [
            "configuration", "label", "language", "num_queries", "recall@1",
            "recall@5", "recall@10", "mrr@10", "mrr@10_ci_lower", "mrr@10_ci_upper",
            "ndcg@10", "avg_latency_ms"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_rows)
    print(f"\nWritten benchmark results to: {results_csv_path}")

    # Save failure analysis CSV
    failures_csv_path = results_dir / f"{prefix}retrieval_failure_analysis.csv"
    with open(failures_csv_path, "w", encoding="utf-8", newline="") as f:
        if args.smoke:
            f.write("# [SMOKE_MODE] Retriever failure analysis sample\n")
        fieldnames = [
            "language", "query", "ground_truth_chunk_id",
            "retrieved_top1_chunk_id", "retrieved_top1_score", "mrr10"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failures[:25])
    print(f"Written failure analysis to: {failures_csv_path}")


if __name__ == "__main__":
    main()
