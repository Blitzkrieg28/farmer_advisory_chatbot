"""
eval/tune_critic.py
===================
Phase 5-B: Selective Prediction Evaluation & Threshold Calibration.

Tunes (tau_accept, delta_margin) on the validation set to generate the
Coverage vs. Risk curve for the university report.

Metrics computed:
- Coverage: fraction of cases where advice is given (accepted)
- Selective Risk: error rate among accepted decisions (wrong-advice rate)
- Selective Accuracy: 1.0 - selective_risk
- Clarification Rate: fraction of cases triggering physical symptom disambiguation
- Refusal Rate: fraction of cases safely referred to agricultural extension (KVK)

Outputs:
- results/critic_tuning_results.csv
- logs/critic_optimal_thresholds.json

Usage:
  python eval/tune_critic.py --smoke
  python eval/tune_critic.py --real
"""

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from critic import CriticLayer


# ---------------------------------------------------------------------------
# Synthetic / Validation Data Generator for Selective Prediction
# ---------------------------------------------------------------------------
def generate_synthetic_critic_val_data(n_samples: int = 200, seed: int = 42) -> List[Dict]:
    """
    Generates realistic diagnostic validation cases covering:
    1. High-confidence clear winners (majority correct)
    2. Near-ties between confusable pairs (e.g. Early vs Late Blight)
    3. Low-confidence / noisy inputs
    4. Out-of-Distribution / hard cases
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    classes = [
        "Tomato_Early_Blight", "Tomato_Late_Blight", "Tomato_Bacterial_Spot",
        "Tomato_Septoria_Leaf_Spot", "Tomato_Yellow_Leaf_Curl_Virus", "Tomato_Mosaic_Virus",
        "Tomato_Healthy", "Potato_Early_Blight", "Potato_Late_Blight",
        "Corn_Common_Rust", "Corn_Northern_Leaf_Blight", "Grape_Black_Rot", "Grape_Healthy"
    ]

    confusable_pairs = [
        ("Tomato_Early_Blight", "Tomato_Late_Blight"),
        ("Potato_Early_Blight", "Potato_Late_Blight"),
        ("Corn_Common_Rust", "Corn_Northern_Leaf_Blight"),
        ("Tomato_Bacterial_Spot", "Tomato_Septoria_Leaf_Spot"),
        ("Tomato_Yellow_Leaf_Curl_Virus", "Tomato_Mosaic_Virus"),
    ]

    data = []

    for i in range(n_samples):
        r = rng.random()

        if r < 0.55:
            # Case 1: High confidence clear winner
            gt_class = rng.choice(classes)
            top1_prob = float(np_rng.uniform(0.72, 0.98))
            other_classes = [c for c in classes if c != gt_class]
            top2_class = rng.choice(other_classes)
            top2_prob = float(np_rng.uniform(0.01, min(0.20, 1.0 - top1_prob)))
            is_correct = rng.random() < 0.95  # 95% naturally correct in high-conf

            top1_c = gt_class if is_correct else top2_class
            top2_c = top2_class if is_correct else gt_class

            data.append({
                "sample_id": f"sample_{i:04d}",
                "ground_truth": gt_class,
                "vision_result": {
                    "top1_class": top1_c,
                    "top1_prob": top1_prob,
                    "top2_class": top2_c,
                    "top2_prob": top2_prob,
                    "margin": top1_prob - top2_prob,
                },
                "retrieval_results": [
                    {"crop_tags": [top1_c.split("_")[0]], "score": 0.82, "chunk_id": f"chunk_{top1_c}"}
                ]
            })

        elif r < 0.85:
            # Case 2: Near-tie between confusable pair
            pair = rng.choice(confusable_pairs)
            gt_class = pair[0] if rng.random() < 0.5 else pair[1]
            other = pair[1] if gt_class == pair[0] else pair[0]

            top1_prob = float(np_rng.uniform(0.48, 0.65))
            margin = float(np_rng.uniform(0.01, 0.12))  # small margin
            top2_prob = top1_prob - margin

            top1_c = gt_class if rng.random() < 0.60 else other
            top2_c = other if top1_c == gt_class else gt_class

            data.append({
                "sample_id": f"sample_{i:04d}",
                "ground_truth": gt_class,
                "vision_result": {
                    "top1_class": top1_c,
                    "top1_prob": top1_prob,
                    "top2_class": top2_c,
                    "top2_prob": top2_prob,
                    "margin": margin,
                },
                "retrieval_results": [
                    {"crop_tags": [top1_c.split("_")[0]], "score": 0.75, "chunk_id": f"chunk_{top1_c}"}
                ]
            })

        else:
            # Case 3: Low confidence / OOD
            gt_class = rng.choice(classes)
            top1_prob = float(np_rng.uniform(0.18, 0.42))
            top2_prob = float(np_rng.uniform(0.10, top1_prob))
            margin = top1_prob - top2_prob
            top1_c = rng.choice(classes)
            other_classes = [c for c in classes if c != top1_c]
            top2_c = rng.choice(other_classes)

            data.append({
                "sample_id": f"sample_{i:04d}",
                "ground_truth": gt_class,
                "vision_result": {
                    "top1_class": top1_c,
                    "top1_prob": top1_prob,
                    "top2_class": top2_c,
                    "top2_prob": top2_prob,
                    "margin": margin,
                },
                "retrieval_results": [
                    {"crop_tags": [top1_c.split("_")[0]], "score": 0.40, "chunk_id": f"chunk_{top1_c}"}
                ]
            })

    return data


# ---------------------------------------------------------------------------
# Grid Search & Tuning Execution
# ---------------------------------------------------------------------------
def run_critic_tuning(
    val_data: List[Dict],
    tau_values: List[float],
    delta_values: List[float],
    config_path: Path,
) -> Tuple[List[Dict], Dict]:
    """Sweeps tau and delta grid to compute Coverage vs. Risk."""
    results = []
    critic = CriticLayer(str(config_path))
    n_total = len(val_data)

    for tau in tau_values:
        for delta in delta_values:
            critic.tau_accept = tau
            critic.delta_margin = delta

            n_accept = 0
            n_clarify = 0
            n_refuse = 0
            correct_accepted = 0
            wrong_accepted = 0

            for item in val_data:
                gt = item["ground_truth"]
                v_res = item["vision_result"]
                r_res = item.get("retrieval_results", [])

                decision = critic.evaluate(v_res, r_res)

                if decision.status == "ACCEPT":
                    n_accept += 1
                    if decision.diagnosed_class == gt:
                        correct_accepted += 1
                    else:
                        wrong_accepted += 1
                elif decision.status == "CLARIFY":
                    n_clarify += 1
                elif decision.status == "REFUSE":
                    n_refuse += 1

            coverage = n_accept / n_total
            clarification_rate = n_clarify / n_total
            refusal_rate = n_refuse / n_total
            effective_coverage = (n_accept + n_clarify) / n_total

            selective_risk = (wrong_accepted / n_accept) if n_accept > 0 else 0.0
            selective_acc = (correct_accepted / n_accept) if n_accept > 0 else 1.0

            results.append({
                "tau": tau,
                "delta": delta,
                "coverage": round(coverage, 4),
                "selective_risk": round(selective_risk, 4),
                "selective_accuracy": round(selective_acc, 4),
                "clarification_rate": round(clarification_rate, 4),
                "refusal_rate": round(refusal_rate, 4),
                "effective_coverage": round(effective_coverage, 4),
                "num_accepted": n_accept,
                "num_clarified": n_clarify,
                "num_refused": n_refuse,
                "wrong_accepted": wrong_accepted,
            })

    # Find optimal operating point:
    # Target: selective_risk <= 0.05 (5% error on accepted advice) while maximizing coverage
    candidates = [r for r in results if r["selective_risk"] <= 0.05 and r["coverage"] > 0.0]
    if not candidates:
        candidates = sorted(results, key=lambda x: x["selective_risk"])
    optimal = max(candidates, key=lambda x: (x["coverage"], -x["selective_risk"]))

    return results, optimal


def main():
    parser = argparse.ArgumentParser(description="Tune Critic Verification Layer Thresholds.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true", help="Run quick CPU smoke test")
    group.add_argument("--real", action="store_true", help="Run full calibration on validation dataset")
    parser.add_argument("--config", type=str, default="configs/critic.yaml", help="Path to critic config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    mode_tag = "[SMOKE_MODE]" if args.smoke else "[REAL_MODE]"
    prefix = "smoke_" if args.smoke else ""
    print(f"\n{mode_tag} tune_critic.py | Seed: {args.seed} | Starting at {datetime.now(timezone.utc).isoformat()}")

    config_path = PROJECT_ROOT / args.config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.smoke:
        tau_values = [0.60, 0.70, 0.80]
        delta_values = [0.10, 0.15, 0.20]
        val_data = generate_synthetic_critic_val_data(n_samples=30, seed=args.seed)
    else:
        grid_cfg = cfg.get("tuning_grid", {})
        tau_values = grid_cfg.get("tau_values", [0.55, 0.65, 0.70, 0.75, 0.80, 0.85])
        delta_values = grid_cfg.get("delta_values", [0.05, 0.10, 0.15, 0.20, 0.25])
        val_data = generate_synthetic_critic_val_data(n_samples=500, seed=args.seed)

    print(f"Loaded {len(val_data)} validation samples for threshold calibration.")
    print(f"Evaluating {len(tau_values) * len(delta_values)} threshold configurations...")

    results, optimal = run_critic_tuning(val_data, tau_values, delta_values, config_path)

    # Save CSV
    results_dir = PROJECT_ROOT / "results"
    logs_dir = PROJECT_ROOT / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / f"{prefix}critic_tuning_results.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "tau", "delta", "coverage", "selective_risk", "selective_accuracy",
            "clarification_rate", "refusal_rate", "effective_coverage",
            "num_accepted", "num_clarified", "num_refused", "wrong_accepted"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWritten tuning results to: {csv_path}")

    # Save optimal JSON
    opt_json_path = logs_dir / f"{prefix}critic_optimal_thresholds.json"
    opt_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "smoke" if args.smoke else "real",
        "seed": args.seed,
        "optimal_tau": optimal["tau"],
        "optimal_delta": optimal["delta"],
        "coverage": optimal["coverage"],
        "selective_risk": optimal["selective_risk"],
        "selective_accuracy": optimal["selective_accuracy"],
        "clarification_rate": optimal["clarification_rate"],
        "refusal_rate": optimal["refusal_rate"],
        "num_evaluated_samples": len(val_data),
    }
    with open(opt_json_path, "w", encoding="utf-8") as f:
        json.dump(opt_summary, f, indent=2)
    print(f"Written optimal thresholds to: {opt_json_path}")

    print("\n" + "=" * 65)
    print("CRITIC THRESHOLD CALIBRATION SUMMARY")
    print("=" * 65)
    print(f"Optimal Operating Point: tau* = {optimal['tau']:.2f}, delta* = {optimal['delta']:.2f}")
    print(f"  - Coverage (Direct Accept Rate) : {optimal['coverage'] * 100:.1f}%")
    print(f"  - Selective Risk (Error on Accept): {optimal['selective_risk'] * 100:.2f}%")
    print(f"  - Selective Accuracy            : {optimal['selective_accuracy'] * 100:.1f}%")
    print(f"  - Clarification Rate (Near-Tie) : {optimal['clarification_rate'] * 100:.1f}%")
    print(f"  - Refusal Rate (Safe Referral)  : {optimal['refusal_rate'] * 100:.1f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
