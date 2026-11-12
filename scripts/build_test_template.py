"""
scripts/build_test_template.py
==============================
Phase 4-C: Scaffold the human evaluation test set.

Samples ~50-60 chunks stratified across crops and diseases from kb/chunks.jsonl
(or smoke_chunks.jsonl in smoke mode), favoring chunks where has_dosage=True
and table_uncertain=False.

Outputs:
  data/test_queries_template.csv (or data/smoke_test_queries_template.csv)

Schema:
  test_id, query_en, query_hi, query_mr, ground_truth_chunk_id, doc_name,
  page, crop, disease, has_dosage, human_verified, verified_by, notes

Usage:
  python scripts/build_test_template.py --smoke
  python scripts/build_test_template.py --real
"""

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

KB_DIR = PROJECT_ROOT / "kb"
DATA_DIR = PROJECT_ROOT / "data"


def load_chunks(chunks_file: Path) -> list:
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file does not exist: {chunks_file}")
    chunks = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                chunks.append(json.loads(line))
    return chunks


def build_test_template(chunks: list, sample_size: int = 50, seed: int = 42, smoke: bool = False) -> list:
    rng = random.Random(seed)
    
    # Filter out empty or pure scan chunks
    usable = [c for c in chunks if not c.get("no_text_layer") and not c.get("garbled")]
    if not usable:
        raise ValueError("No usable chunks found to build test set.")

    # Group by (crop, disease)
    strata = {}
    for c in usable:
        crop = (c.get("crop_tags") or ["general"])[0]
        disease = (c.get("disease_tags") or ["general"])[0]
        key = (crop, disease)
        strata.setdefault(key, []).append(c)

    selected_chunks = []
    
    # Stratified pass: sort each pool by dosage/table quality first
    for key, pool in strata.items():
        # Priority: dosage=True, table_uncertain=False
        sorted_pool = sorted(
            pool,
            key=lambda x: (
                1 if x.get("has_dosage") else 0,
                0 if x.get("table_uncertain") else 1
            ),
            reverse=True
        )
        # Take 1-2 from each stratum initially
        take_n = min(len(sorted_pool), max(1, sample_size // max(len(strata), 1)))
        selected_chunks.extend(sorted_pool[:take_n])

    # If we need more to reach sample_size, draw from remaining
    remaining = [c for c in usable if c not in selected_chunks]
    rng.shuffle(remaining)
    needed = sample_size - len(selected_chunks)
    if needed > 0 and remaining:
        selected_chunks.extend(remaining[:needed])

    # If smoke mode and small KB, take all available up to sample_size
    if len(selected_chunks) < sample_size and smoke:
        # In smoke mode with tiny dummy pdf, fewer chunks is totally expected
        pass

    rows = []
    for i, c in enumerate(selected_chunks):
        cid = c.get("chunk_id", f"c_{i}")
        doc = c.get("doc_name", "")
        page = c.get("page", 0)
        crop = (c.get("crop_tags") or ["general"])[0]
        disease = (c.get("disease_tags") or ["general"])[0]
        has_dos = c.get("has_dosage", False)
        
        prefix = "smoke_" if smoke else ""
        test_id = f"{prefix}test_{i+1:03d}"
        
        # Starter placeholder query based on crop/disease
        if "Healthy" in disease:
            q_en = f"What is the recommended health maintenance practice for {crop.lower()}?"
            q_hi = f"[TODO_TRANSLATE] {crop} की स्वस्थ फसल के लिए क्या देखभाल आवश्यक है?"
            q_mr = f"[TODO_TRANSLATE] {crop} निरोगी ठेवण्यासाठी काय काळजी घ्यावी?"
        else:
            q_en = f"What is the chemical treatment and dosage for {disease.lower()} in {crop.lower()}?"
            q_hi = f"[TODO_TRANSLATE] {crop} में {disease} के नियंत्रण हेतु रासायनिक उपचार और खुराक क्या है?"
            q_mr = f"[TODO_TRANSLATE] {crop} वरील {disease} च्या नियंत्रणासाठी कोणते औषध व किती प्रमाण वापरावे?"

        rows.append({
            "test_id": test_id,
            "query_en": q_en,
            "query_hi": q_hi,
            "query_mr": q_mr,
            "ground_truth_chunk_id": cid,
            "doc_name": doc,
            "page": page,
            "crop": crop,
            "disease": disease,
            "has_dosage": has_dos,
            "human_verified": "false",
            "verified_by": "PENDING_REVIEW",
            "notes": "Verify extractive suitability and ground truth passage attribution before evaluating."
        })

    return rows


def main():
    parser = argparse.ArgumentParser(description="Build human evaluation test set scaffold.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true", help="Generate template from smoke chunks")
    group.add_argument("--real", action="store_true", help="Generate template from real chunks")
    parser.add_argument("--sample-size", type=int, default=50, help="Target number of test queries (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    mode_tag = "[SMOKE_MODE]" if args.smoke else "[REAL_MODE]"
    print(f"{mode_tag} build_test_template.py starting...")

    chunks_file = KB_DIR / ("smoke_chunks.jsonl" if args.smoke else "chunks.jsonl")
    if not chunks_file.exists():
        print(f"Error: {chunks_file} does not exist. Run ingest_pdfs.py first.", file=sys.stderr)
        sys.exit(1)

    chunks = load_chunks(chunks_file)
    print(f"Loaded {len(chunks)} chunks from {chunks_file.name}")

    rows = build_test_template(chunks, sample_size=args.sample_size, seed=args.seed, smoke=args.smoke)
    
    out_name = "smoke_test_queries_template.csv" if args.smoke else "test_queries_template.csv"
    out_path = DATA_DIR / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "test_id", "query_en", "query_hi", "query_mr", "ground_truth_chunk_id",
        "doc_name", "page", "crop", "disease", "has_dosage", "human_verified",
        "verified_by", "notes"
    ]

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write(f"# {mode_tag} Test set scaffold generated at {datetime.now(timezone.utc).isoformat()}\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSuccessfully wrote {len(rows)} test scaffold rows to {out_path}")
    print("=" * 70)
    print("MANUAL REVIEW INSTRUCTIONS:")
    print("1. Open the generated CSV.")
    print("2. Verify that each query accurately matches its ground_truth_chunk_id.")
    print("3. Check and refine the Hindi (query_hi) and Marathi (query_mr) questions.")
    print("4. Set 'human_verified' to 'true' and set 'verified_by' to your initials/name.")
    print("=" * 70)


if __name__ == "__main__":
    main()

