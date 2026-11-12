"""
generate_retrieval_pairs.py
Generate multilingual (query, passage) training and validation pairs with zero cross-lingual leakage.

Features:
- Unique `source_id` assigned to each English question template and inherited by all its translations (hi, mr).
- Grouped split by `source_id`: all translations of a given question stay strictly in the same split.
- Stratified within each passage: every knowledge base passage is represented in both train and val.
- Hard assertions:
  1. No `source_id` overlap between train and val.
  2. Every passage in val exists in train.
  3. No `pair_id` overlap.
- Logs full split statistics and verification status to logs/retrieval_split_report.json.

Usage:
    python data_prep/generate_retrieval_pairs.py
    python data_prep/generate_retrieval_pairs.py --smoke      # Tiny CPU smoke test
    python data_prep/generate_retrieval_pairs.py --use_nllb   # Full NLLB model on Colab
"""

import argparse
import os
import json
import random
import csv
import datetime
from pathlib import Path
from collections import defaultdict


OFFLINE_MULTILINGUAL_TEMPLATES = {
    "symptom_inquiry": [
        ("What are the visual symptoms of {disease} on {crop}?",
         "{crop} पर {disease} के दृश्य लक्षण क्या हैं?",
         "{crop} वरील {disease} ची दृश्य लक्षणे काय आहेत?"),
        ("How can I identify {disease} affecting my {crop} leaves?",
         "मैं अपनी {crop} की पत्तियों को प्रभावित करने वाले {disease} की पहचान कैसे करूँ?",
         "माझ्या {crop} च्या पानांवर परिणाम करणाऱ्या {disease} रोगाची ओळख कशी करावी?"),
        ("What do the early spots of {disease} on {crop} look like?",
         "{crop} पर {disease} के शुरुआती धब्बे कैसे दिखते हैं?",
         "{crop} वरील {disease} चे सुरुवातीचे डाग कसे दिसतात?")
    ],
    "distinguishing_feature": [
        ("What is the main distinguishing symptom of {disease} on {crop}?",
         "{crop} पर {disease} का मुख्य विशिष्ट लक्षण क्या है?",
         "{crop} वरील {disease} चे मुख्य वैशिष्ट्यपूर्ण लक्षण कोणते आहे?"),
        ("How to differentiate {disease} from other diseases on {crop}?",
         "{crop} पर अन्य रोगों से {disease} को कैसे अलग करें?",
         "{crop} वरील इतर रोगांपासून {disease} कसा वेगळा ओळखायचा?")
    ],
    "treatment_control": [
        ("What is the recommended treatment for {disease} in {crop}?",
         "{crop} में {disease} के लिए अनुशंसित उपचार क्या है?",
         "{crop} मधील {disease} साठी शिफारस केलेले उपचार काय आहेत?"),
        ("How should a farmer manage {disease} attacking {crop}?",
         "किसान को {crop} पर हमला करने वाले {disease} का प्रबंधन कैसे करना चाहिए?",
         "शेतकऱ्याने {crop} वरील {disease} चे व्यवस्थापन कसे करावे?")
    ],
    "prevention_spread": [
        ("How can I prevent {disease} from destroying my {crop} field?",
         "मैं {disease} को अपने {crop} के खेत को नष्ट करने से कैसे रोक सकता हूँ?",
         "मी {disease} ला माझ्या {crop} शेताचे नुकसान करण्यापासून कसे रोखू शकतो?"),
        ("What weather conditions cause {disease} to spread rapidly in {crop}?",
         "किन मौसमी परिस्थितियों में {crop} में {disease} तेजी से फैलता है?",
         "कोणत्या हवामानात {crop} मध्ये {disease} वेगाने पसरतो?")
    ]
}

CROP_TRANSLATIONS = {
    "Tomato": {"hi": "टमाटर", "mr": "टोमॅटो"},
    "Potato": {"hi": "आलू", "mr": "बटाटा"},
    "Grape": {"hi": "अंगूर", "mr": "द्राक्ष"},
    "Maize": {"hi": "मक्का", "mr": "मका"},
    "Corn": {"hi": "मक्का", "mr": "मका"}
}

DISEASE_TRANSLATIONS = {
    "Tomato Early Blight": {"hi": "टमाटर अगेती झुलसा", "mr": "टोमॅटो लवकर करपा"},
    "Tomato Late Blight": {"hi": "टमाटर पछेती झुलसा", "mr": "टोमॅटो उशीरा करपा"},
    "Potato Early Blight": {"hi": "आलू अगेती झुलसा", "mr": "बटाटा लवकर करपा"},
    "Potato Late Blight": {"hi": "आलू पछेती झुलसा", "mr": "बटाटा उशीरा करपा"},
    "Grape Black Rot": {"hi": "अंगूर काला सड़न रोग", "mr": "द्राक्ष काळा कुजवा"},
    "Maize Common Rust": {"hi": "मक्का सामान्य रतुआ", "mr": "मका तांबेरा"}
}


def translate_with_nllb(texts: list, target_lang: str, model_name: str = "facebook/nllb-200-distilled-600M") -> list:
    """Translates English texts using NLLB-200."""
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
        import torch
    except ImportError:
        raise ImportError("transformers and torch required for NLLB translation. Run: pip install transformers sentencepiece")

    lang_map = {
        "hi": "hin_Deva",
        "mr": "mar_Deva",
        "en": "eng_Latn"
    }

    tgt_code = lang_map.get(target_lang, "hin_Deva")
    device = 0 if torch.cuda.is_available() else -1

    print(f"Loading {model_name} for translation to {target_lang} ({tgt_code})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if device == 0:
        model = model.half().cuda()

    translator = pipeline(
        "translation",
        model=model,
        tokenizer=tokenizer,
        src_lang="eng_Latn",
        tgt_lang=tgt_code,
        device=device,
        max_length=128
    )

    results = []
    for t in texts:
        out = translator(t)
        results.append(out[0]["translation_text"])
    return results


def generate_source_questions(passage: dict, use_nllb: bool = False) -> list:
    """
    Generates structured question groups for a passage.
    Each group has a unique `source_id` containing translations in en, hi, mr.
    """
    disease_en = passage["disease"]
    crop_en = passage["crop"]
    passage_id = passage["passage_id"]
    passage_text = passage["text"]

    disease_hi = DISEASE_TRANSLATIONS.get(disease_en, {}).get("hi", disease_en)
    disease_mr = DISEASE_TRANSLATIONS.get(disease_en, {}).get("mr", disease_en)
    crop_hi = CROP_TRANSLATIONS.get(crop_en, {}).get("hi", crop_en)
    crop_mr = CROP_TRANSLATIONS.get(crop_en, {}).get("mr", crop_en)

    source_groups = []
    t_idx = 0

    for category, templates in OFFLINE_MULTILINGUAL_TEMPLATES.items():
        for t_en, t_hi, t_mr in templates:
            source_id = f"src_{passage_id}_{t_idx:03d}"
            en_q = t_en.format(disease=disease_en, crop=crop_en)

            if use_nllb:
                hi_q = translate_with_nllb([en_q], "hi")[0]
                mr_q = translate_with_nllb([en_q], "mr")[0]
            else:
                hi_q = t_hi.format(disease=disease_hi, crop=crop_hi)
                mr_q = t_mr.format(disease=disease_mr, crop=crop_mr)

            variants = [
                {
                    "source_id": source_id,
                    "pair_id": f"{source_id}_en",
                    "query": en_q,
                    "passage_id": passage_id,
                    "passage_text": passage_text,
                    "language": "en",
                    "query_type": category,
                    "disease": disease_en,
                    "crop": crop_en
                },
                {
                    "source_id": source_id,
                    "pair_id": f"{source_id}_hi",
                    "query": hi_q,
                    "passage_id": passage_id,
                    "passage_text": passage_text,
                    "language": "hi",
                    "query_type": category,
                    "disease": disease_en,
                    "crop": crop_en
                },
                {
                    "source_id": source_id,
                    "pair_id": f"{source_id}_mr",
                    "query": mr_q,
                    "passage_id": passage_id,
                    "passage_text": passage_text,
                    "language": "mr",
                    "query_type": category,
                    "disease": disease_en,
                    "crop": crop_en
                }
            ]

            source_groups.append({
                "source_id": source_id,
                "passage_id": passage_id,
                "category": category,
                "variants": variants
            })
            t_idx += 1

    return source_groups


def write_csv(filepath: Path, records: list):
    if not records:
        return
    headers = list(records[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description="Generate multilingual retrieval pairs with zero leakage.")
    parser.add_argument("--kb_path", type=str, default="kb/passages.json", help="Knowledge base JSON path")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for CSV files")
    parser.add_argument("--log_dir", type=str, default="logs", help="Log directory for reports")
    parser.add_argument("--use_nllb", action="store_true", help="Use NLLB-200 model for translation")
    parser.add_argument("--val_ratio", type=float, default=0.25, help="Ratio of question sources for validation")
    parser.add_argument("--smoke", action="store_true", help="Generate small set for fast CPU testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(args.kb_path, "r", encoding="utf-8") as f:
        kb_data = json.load(f)

    passages = [p for p in kb_data["passages"] if p["passage_id"] != "TEMPLATE_ENTRY"]
    print(f"Loaded {len(passages)} verified passages from knowledge base.")

    # Generate source groups per passage
    passages_source_groups = {}
    for passage in passages:
        groups = generate_source_questions(passage, use_nllb=args.use_nllb)
        passages_source_groups[passage["passage_id"]] = groups

    total_sources = sum(len(g) for g in passages_source_groups.values())
    total_pairs = total_sources * 3
    print(f"Generated {total_sources} distinct source questions ({total_pairs} multilingual pairs) across {len(passages)} passages.")

    train_records = []
    val_records = []
    train_source_ids = set()
    val_source_ids = set()
    train_passages = set()
    val_passages = set()

    # Stratify by passage: for each passage, split its source_ids
    for pid, groups in passages_source_groups.items():
        shuffled_groups = list(groups)
        random.shuffle(shuffled_groups)

        n_groups = len(shuffled_groups)
        # Ensure at least 1 group in val, and at least 1 group in train
        n_val = max(1, int(round(n_groups * args.val_ratio)))
        n_train = n_groups - n_val
        if n_train < 1:
            n_train = 1
            n_val = max(1, n_groups - 1)

        val_g = shuffled_groups[:n_val]
        train_g = shuffled_groups[n_val:]

        for g in train_g:
            train_source_ids.add(g["source_id"])
            train_passages.add(g["passage_id"])
            train_records.extend(g["variants"])

        for g in val_g:
            val_source_ids.add(g["source_id"])
            val_passages.add(g["passage_id"])
            val_records.extend(g["variants"])

    # -------------------------------------------------------------
    # STRICT LEAKAGE ASSERTIONS
    # -------------------------------------------------------------
    print("\n--- Running Retrieval Leakage Assertions ---")

    # 1. No source_id overlap (zero multilingual leakage)
    overlap = train_source_ids & val_source_ids
    assert len(overlap) == 0, f"MULTILINGUAL LEAKAGE: {len(overlap)} source_ids found in both train and val: {overlap}"
    print(f"✓ Passed: 0 overlapping source_ids between train and val.")

    # 2. Every val passage exists in train (in-domain verification)
    assert val_passages.issubset(train_passages), (
        f"ORPHAN PASSAGES IN VAL: {val_passages - train_passages} appear in val but have no training pairs!"
    )
    print(f"✓ Passed: All {len(val_passages)} validation passages have training representation.")

    # 3. No pair_id overlap
    train_pair_ids = {r["pair_id"] for r in train_records}
    val_pair_ids = {r["pair_id"] for r in val_records}
    assert len(train_pair_ids & val_pair_ids) == 0, "Duplicate pair_ids found!"
    print(f"✓ Passed: 0 overlapping pair_ids.")

    # Save to CSV
    train_path = output_dir / "retrieval_train.csv"
    val_path = output_dir / "retrieval_val.csv"
    write_csv(train_path, train_records)
    write_csv(val_path, val_records)

    # Count by language
    def count_lang(records):
        counts = defaultdict(int)
        for r in records:
            counts[r["language"]] += 1
        return dict(counts)

    lang_train = count_lang(train_records)
    lang_val = count_lang(val_records)

    print("\n" + "=" * 60)
    print(f"{'Language':<10} | {'Train Pairs':>12} | {'Val Pairs':>10} | {'Total':>8}")
    print("-" * 60)
    for lang in ["en", "hi", "mr"]:
        ntr = lang_train.get(lang, 0)
        nv = lang_val.get(lang, 0)
        print(f"{lang:<10} | {ntr:>12} | {nv:>10} | {ntr + nv:>8}")
    print("-" * 60)
    print(f"{'TOTAL':<10} | {len(train_records):>12} | {len(val_records):>10} | {len(train_records) + len(val_records):>8}")
    print("=" * 60)

    # Save report
    report = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "random_seed": args.seed,
        "split_strategy": "grouped_by_source_id_stratified_by_passage",
        "total_source_questions": total_sources,
        "train_sources": len(train_source_ids),
        "val_sources": len(val_source_ids),
        "train_pairs_total": len(train_records),
        "val_pairs_total": len(val_records),
        "languages": ["en", "hi", "mr"],
        "language_distribution_train": lang_train,
        "language_distribution_val": lang_val,
        "assertions": {
            "source_id_disjoint": len(overlap) == 0,
            "all_val_passages_in_train": val_passages.issubset(train_passages),
            "pair_id_disjoint": len(train_pair_ids & val_pair_ids) == 0
        }
    }

    report_path = log_dir / "retrieval_split_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nRetrieval split report written to: {report_path}")


if __name__ == "__main__":
    main()
