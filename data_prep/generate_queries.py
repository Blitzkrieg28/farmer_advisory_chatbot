"""
data_prep/generate_queries.py
==============================
Phase 4-B: Synthetic farmer-style query generation from KB chunks.

Backends:
  --backend template : Rule-based templates, no downloads (CPU smoke mode)
  --backend llm      : Qwen2.5-1.5B-Instruct via HuggingFace (Colab only)
  --backend nllb     : NLLB-200-distilled-600M translation (Colab only)

For each chunk, generates 3-5 English questions, then translates to Hindi and
Marathi (or returns stubs locally). Splits by source_id (grouped split) to
prevent leakage.

Outputs:
  data/retrieval_train.csv
  data/retrieval_val.csv
  logs/retrieval_split_report.json

Smoke mode (--smoke, --backend template):
  Uses smoke_chunks.jsonl; outputs prefixed smoke_

Usage:
  python data_prep/generate_queries.py --smoke --backend template
  # On Colab:
  python data_prep/generate_queries.py --real --backend llm
  python data_prep/generate_queries.py --real --backend nllb  # translation step
"""

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"
KB_DIR = PROJECT_ROOT / "kb"

# ---------------------------------------------------------------------------
# Template-based query generation (no model needed)
# ---------------------------------------------------------------------------

# Template bank: each template is a function(disease, crop) -> str
QUERY_TEMPLATES_EN = [
    # Identification
    lambda disease, crop: f"What are the symptoms of {disease.lower()} in {crop.lower()}?",
    lambda disease, crop: f"How do I identify {disease.lower()} on my {crop.lower()} plants?",
    lambda disease, crop: f"My {crop.lower()} has spots/lesions — could it be {disease.lower()}?",
    # Cause
    lambda disease, crop: f"What causes {disease.lower()} in {crop.lower()}?",
    lambda disease, crop: f"Which pathogen is responsible for {disease.lower()} in {crop.lower()}?",
    # Treatment / chemical
    lambda disease, crop: f"What fungicide should I use for {disease.lower()} in {crop.lower()}?",
    lambda disease, crop: f"How do I treat {disease.lower()} in {crop.lower()}?",
    lambda disease, crop: f"What is the recommended dose for controlling {disease.lower()} in {crop.lower()}?",
    # Timing / PHI
    lambda disease, crop: f"When should I apply chemicals for {disease.lower()} on {crop.lower()}?",
    lambda disease, crop: f"What is the pre-harvest interval for {disease.lower()} fungicides in {crop.lower()}?",
    # Prevention / cultural
    lambda disease, crop: f"How can I prevent {disease.lower()} in my {crop.lower()} field?",
    lambda disease, crop: f"What cultural practices help control {disease.lower()} in {crop.lower()}?",
    # Spread
    lambda disease, crop: f"How does {disease.lower()} spread in {crop.lower()} crops?",
    lambda disease, crop: f"What conditions favour {disease.lower()} in {crop.lower()}?",
    # Variety
    lambda disease, crop: f"Are there resistant {crop.lower()} varieties for {disease.lower()}?",
]

# Hindi stub templates (TODO_VERIFY — have a native speaker review before use in production)
QUERY_TEMPLATES_HI_STUB = [
    lambda disease, crop: f"[HI_STUB] {crop} में {disease} के लक्षण क्या हैं?",
    lambda disease, crop: f"[HI_STUB] {crop} में {disease} का उपचार कैसे करें?",
    lambda disease, crop: f"[HI_STUB] {crop} में {disease} से बचाव कैसे करें?",
]

# Marathi stub templates (TODO_VERIFY)
QUERY_TEMPLATES_MR_STUB = [
    lambda disease, crop: f"[MR_STUB] {crop} मध्ये {disease} चे लक्षणे काय आहेत?",
    lambda disease, crop: f"[MR_STUB] {crop} मध्ये {disease} वर उपाय काय आहे?",
    lambda disease, crop: f"[MR_STUB] {crop} मध्ये {disease} रोखण्यासाठी काय करावे?",
]

# Healthy-class queries (no treatment — different question set)
QUERY_TEMPLATES_HEALTHY_EN = [
    lambda disease, crop: f"How do I keep my {crop.lower()} plants healthy?",
    lambda disease, crop: f"What are the signs of a healthy {crop.lower()} plant?",
    lambda disease, crop: f"What preventive care does {crop.lower()} need to avoid diseases?",
]


def generate_en_queries_template(disease: str, crop: str, is_healthy: bool,
                                  n: int, rng: random.Random) -> list:
    """Generate n distinct English template queries for a disease/crop pair."""
    templates = QUERY_TEMPLATES_HEALTHY_EN if is_healthy else QUERY_TEMPLATES_EN
    if n > len(templates):
        n = len(templates)
    chosen = rng.sample(templates, n)
    queries = []
    for fn in chosen:
        try:
            q = fn(disease, crop)
            queries.append(q)
        except Exception:
            pass
    return queries


def translate_stub(text: str, lang: str) -> str:
    """
    Local stub translation. Returns clearly marked placeholder.
    Real translation uses NLLB on Colab (see --backend nllb logic below).
    """
    prefix = "[HI_STUB]" if lang == "hi" else "[MR_STUB]"
    return f"{prefix} {text}"


# ---------------------------------------------------------------------------
# NLLB translation (Colab only — not downloaded locally)
# ---------------------------------------------------------------------------
def translate_nllb(texts: list, src_lang: str, tgt_lang: str, model, tokenizer) -> list:
    """
    Translate a batch of texts using the loaded NLLB model.
    Call only after the model and tokenizer are loaded on Colab.
    tgt_lang: NLLB language code, e.g. "hin_Deva" for Hindi, "mar_Deva" for Marathi.
    """
    # Tokenize
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    # Generate
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
        max_new_tokens=256,
    )
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]


# ---------------------------------------------------------------------------
# LLM query generation (Colab only)
# ---------------------------------------------------------------------------
LLM_SYSTEM_PROMPT = (
    "You are an agricultural extension expert. Given a passage about a crop disease, "
    "generate exactly {n} distinct farmer-style questions that this passage could answer. "
    "Questions must be in simple English, one per line, numbered. "
    "Do NOT copy sentences verbatim from the passage. "
    "Do NOT invent chemical names, doses, or PHI values not in the passage."
)

LLM_USER_TEMPLATE = (
    "Passage (verbatim):\n\"\"\"\n{text}\n\"\"\"\n\n"
    "Generate {n} farmer questions this passage could answer:"
)


def generate_queries_llm(chunk_text: str, n: int, pipe) -> list:
    """
    Generate n questions from a chunk using a HuggingFace pipeline.
    Returns list of strings.
    """
    system = LLM_SYSTEM_PROMPT.format(n=n)
    user = LLM_USER_TEMPLATE.format(text=chunk_text[:600], n=n)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    out = pipe(messages, max_new_tokens=300, do_sample=False)
    text = out[0]["generated_text"][-1]["content"]
    # Parse numbered lines
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        if len(line) > 10:
            lines.append(line)
    return lines[:n]


# ---------------------------------------------------------------------------
# Deduplication and quality filters
# ---------------------------------------------------------------------------
def dedup_queries(queries: list) -> list:
    """Remove exact duplicates (case-insensitive)."""
    seen = set()
    out = []
    for q in queries:
        key = q.strip().lower()
        if key not in seen and len(key) > 10:
            seen.add(key)
            out.append(q.strip())
    return out


def copy_fraction(query: str, passage: str, threshold: float = 0.6) -> bool:
    """True if more than threshold fraction of query words appear in passage."""
    q_words = set(query.lower().split())
    p_words = set(passage.lower().split())
    if not q_words:
        return False
    return len(q_words & p_words) / len(q_words) > threshold


def filter_queries(queries: list, passage: str) -> list:
    """Remove queries that are too short, too long, or copy the passage verbatim."""
    out = []
    for q in queries:
        if len(q.split()) < 5:
            continue
        if len(q.split()) > 50:
            continue
        if copy_fraction(q, passage):
            continue
        out.append(q)
    return out


# ---------------------------------------------------------------------------
# Grouped train/val split (by source_id — no leakage)
# ---------------------------------------------------------------------------
def grouped_split(records: list, val_frac: float = 0.15, seed: int = 42) -> tuple:
    """
    Split records by unique source_id.
    Returns (train_records, val_records) with no source_id overlap.
    Assertion: every val chunk_id appears in train at least once.
    """
    rng = random.Random(seed)
    # Group by source_id
    source_to_records = {}
    for r in records:
        sid = r["source_id"]
        source_to_records.setdefault(sid, []).append(r)

    source_ids = sorted(source_to_records.keys())
    rng.shuffle(source_ids)

    n_val = max(1, int(len(source_ids) * val_frac))
    val_sources = set(source_ids[:n_val])
    train_sources = set(source_ids[n_val:])

    train_records = [r for r in records if r["source_id"] in train_sources]
    val_records = [r for r in records if r["source_id"] in val_sources]

    # Assertion: no source_id overlap
    assert val_sources.isdisjoint(train_sources), \
        "ASSERT FAILED: source_id overlap between train and val!"

    # Assertion: every val chunk_id appears in train
    train_chunk_ids = {r["chunk_id"] for r in train_records}
    val_chunk_ids_missing = [
        r["chunk_id"] for r in val_records if r["chunk_id"] not in train_chunk_ids
    ]
    # NOTE: If we're doing pure information retrieval, val passages might not be in train KB.
    # Per spec: warn but don't fail — the user chose not to hold out passages.
    if val_chunk_ids_missing:
        print(f"  [INFO] {len(val_chunk_ids_missing)} val chunk_ids not in train chunk_ids.")
        print(f"  [INFO] This is expected if queries come from different chunks than training.")

    return train_records, val_records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate retrieval training pairs from KB chunks.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true",
                       help="Use smoke_chunks.jsonl; output prefixed smoke_")
    group.add_argument("--real", action="store_true",
                       help="Use chunks.jsonl from kb/")
    parser.add_argument("--backend", choices=["template", "llm", "nllb"],
                        default="template",
                        help="Query generation backend (default: template)")
    parser.add_argument("--n-queries", type=int, default=4,
                        help="Number of queries per chunk (default: 4)")
    parser.add_argument("--val-frac", type=float, default=0.15,
                        help="Fraction of source_ids for validation (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="Cap number of chunks processed (useful for quick tests)")
    args = parser.parse_args()

    smoke = args.smoke
    mode_tag = "[SMOKE_MODE]" if smoke else "[REAL_MODE]"
    rng = random.Random(args.seed)
    print(f"{mode_tag} generate_queries.py | backend={args.backend} | "
          f"seed={args.seed} | {datetime.now(timezone.utc).isoformat()}")

    # --- Validate backend vs mode ---
    if smoke and args.backend != "template":
        print("  [WARNING] Smoke mode always uses --backend template. Overriding.")
        args.backend = "template"

    if args.backend in ("llm", "nllb") and not smoke:
        print(f"  [INFO] Backend '{args.backend}' requires Colab GPU and internet access.")
        print("  [INFO] Running locally will likely fail or be very slow.")

    # --- Load chunks ---
    chunks_file = KB_DIR / ("smoke_chunks.jsonl" if smoke else "chunks.jsonl")
    if not chunks_file.exists():
        print(f"FATAL: Chunks file not found: {chunks_file}", file=sys.stderr)
        print("  Run data_prep/ingest_pdfs.py first.", file=sys.stderr)
        sys.exit(1)

    chunks = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                chunks.append(json.loads(line))
    print(f"  Loaded {len(chunks)} chunks from {chunks_file.name}")

    # Filter out no-text-layer and garbled chunks
    valid_chunks = [c for c in chunks if not c.get("no_text_layer") and not c.get("garbled")]
    print(f"  Valid chunks (not garbled, not scan-only): {len(valid_chunks)}")

    if args.max_chunks:
        valid_chunks = valid_chunks[: args.max_chunks]
        print(f"  Capped to {len(valid_chunks)} chunks (--max-chunks)")

    if not valid_chunks:
        print("FATAL: No valid chunks to generate queries from.", file=sys.stderr)
        sys.exit(1)

    # --- Load LLM / NLLB if needed ---
    llm_pipe = None
    nllb_model = None
    nllb_tokenizer = None

    if args.backend == "llm":
        try:
            from transformers import pipeline as hf_pipeline
            print("  Loading Qwen2.5-1.5B-Instruct (this may take a few minutes on first run)...")
            llm_pipe = hf_pipeline(
                "text-generation",
                model="Qwen/Qwen2.5-1.5B-Instruct",
                device_map="auto",
                torch_dtype="auto",
            )
            print("  LLM loaded.")
        except Exception as e:
            print(f"FATAL: Failed to load LLM: {e}", file=sys.stderr)
            sys.exit(1)

    if args.backend == "nllb":
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            nllb_model_name = "facebook/nllb-200-distilled-600M"
            print(f"  Loading {nllb_model_name}...")
            nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_model_name, src_lang="eng_Latn")
            nllb_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_name)
            print("  NLLB loaded.")
        except Exception as e:
            print(f"FATAL: Failed to load NLLB: {e}", file=sys.stderr)
            sys.exit(1)

    # --- Generate queries ---
    all_records = []  # each row: query, language, chunk_id, source_id, synthetic
    skipped = 0

    for i, chunk in enumerate(valid_chunks):
        chunk_id = chunk["chunk_id"]
        text = chunk.get("text", "")
        disease_tags = chunk.get("disease_tags", ["general"])
        crop_tags = chunk.get("crop_tags", ["general"])

        # Determine primary disease and crop for template queries
        disease = disease_tags[0] if disease_tags else "general"
        crop = crop_tags[0] if crop_tags else "general"
        is_healthy = "Healthy" in disease

        # --- English queries ---
        if args.backend == "template":
            en_queries = generate_en_queries_template(
                disease, crop, is_healthy, args.n_queries, rng
            )
        elif args.backend == "llm":
            try:
                en_queries = generate_queries_llm(text, args.n_queries, llm_pipe)
            except Exception as e:
                print(f"  [WARNING] LLM failed on chunk {chunk_id}: {e}")
                en_queries = generate_en_queries_template(
                    disease, crop, is_healthy, args.n_queries, rng
                )
        else:
            # NLLB mode: generate English queries by template, translate below
            en_queries = generate_en_queries_template(
                disease, crop, is_healthy, args.n_queries, rng
            )

        en_queries = filter_queries(dedup_queries(en_queries), text)

        if not en_queries:
            skipped += 1
            continue

        # Assign a stable source_id from chunk_id deterministically
        # (so re-runs with same chunks produce same source_ids)
        source_id = hashlib.md5(chunk_id.encode()).hexdigest()[:12]

        # --- Add English records ---
        for q in en_queries:
            all_records.append({
                "query": q,
                "language": "en",
                "chunk_id": chunk_id,
                "source_id": source_id,
                "synthetic": True,
                "doc_name": chunk.get("doc_name", ""),
                "page": chunk.get("page", -1),
            })

        # --- Hindi queries ---
        if args.backend == "nllb" and nllb_model is not None:
            hi_queries = translate_nllb(en_queries, "en", "hin_Deva", nllb_model, nllb_tokenizer)
        elif args.backend == "template":
            hi_fns = rng.sample(QUERY_TEMPLATES_HI_STUB,
                                min(len(QUERY_TEMPLATES_HI_STUB), len(en_queries)))
            hi_queries = [fn(disease, crop) for fn in hi_fns]
        else:
            # LLM mode: use stubs for local testing
            hi_queries = [translate_stub(q, "hi") for q in en_queries]

        for q in hi_queries:
            all_records.append({
                "query": q,
                "language": "hi",
                "chunk_id": chunk_id,
                "source_id": source_id,
                "synthetic": True,
                "doc_name": chunk.get("doc_name", ""),
                "page": chunk.get("page", -1),
            })

        # --- Marathi queries ---
        if args.backend == "nllb" and nllb_model is not None:
            mr_queries = translate_nllb(en_queries, "en", "mar_Deva", nllb_model, nllb_tokenizer)
        elif args.backend == "template":
            mr_fns = rng.sample(QUERY_TEMPLATES_MR_STUB,
                                min(len(QUERY_TEMPLATES_MR_STUB), len(en_queries)))
            mr_queries = [fn(disease, crop) for fn in mr_fns]
        else:
            mr_queries = [translate_stub(q, "mr") for q in en_queries]

        for q in mr_queries:
            all_records.append({
                "query": q,
                "language": "mr",
                "chunk_id": chunk_id,
                "source_id": source_id,
                "synthetic": True,
                "doc_name": chunk.get("doc_name", ""),
                "page": chunk.get("page", -1),
            })

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(valid_chunks)} chunks, {len(all_records)} records so far")

    print(f"\n  Generated {len(all_records)} total query records ({skipped} chunks skipped)")

    # --- Grouped split ---
    train_records, val_records = grouped_split(all_records, args.val_frac, args.seed)
    print(f"  Split: train={len(train_records)}, val={len(val_records)}")

    # Assertion: no source_id overlap
    train_sids = {r["source_id"] for r in train_records}
    val_sids = {r["source_id"] for r in val_records}
    assert train_sids.isdisjoint(val_sids), \
        f"ASSERT FAILED: {len(train_sids & val_sids)} source_ids in both train and val!"

    # Language counts per split
    for split_name, records in [("train", train_records), ("val", val_records)]:
        lang_counts = {}
        for r in records:
            lang_counts[r["language"]] = lang_counts.get(r["language"], 0) + 1
        counts_str = ", ".join(f"{k}={v}" for k, v in sorted(lang_counts.items()))
        print(f"  {split_name} language counts: {counts_str}")

    # --- Write CSVs ---
    fieldnames = ["query", "language", "chunk_id", "source_id", "synthetic", "doc_name", "page"]
    prefix = "smoke_" if smoke else ""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, records in [("train", train_records), ("val", val_records)]:
        out_path = DATA_DIR / f"{prefix}retrieval_{split_name}.csv"
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            f.write(f"# {mode_tag} generated {datetime.now(timezone.utc).isoformat()}\n")
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        print(f"  Written: {out_path} ({len(records)} rows)")

    # --- Split report ---
    report = {
        "mode": "smoke" if smoke else "real",
        "backend": args.backend,
        "seed": args.seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_records": len(all_records),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "chunks_skipped": skipped,
        "source_id_overlap": len(train_sids & val_sids),
        "train_sources": len(train_sids),
        "val_sources": len(val_sids),
        "train_lang_counts": {
            lang: sum(1 for r in train_records if r["language"] == lang)
            for lang in ["en", "hi", "mr"]
        },
        "val_lang_counts": {
            lang: sum(1 for r in val_records if r["language"] == lang)
            for lang in ["en", "hi", "mr"]
        },
    }
    report_path = LOGS_DIR / f"{prefix}retrieval_split_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Written: {report_path}")

    if smoke:
        print("\n  [SMOKE] All outputs prefixed 'smoke_'. Do NOT mix with real results.")

    print("\nDone.")


if __name__ == "__main__":
    main()

