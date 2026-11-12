"""
retrieval.py
============
Core Retriever Module for Multilingual Agricultural Knowledge Retrieval.

Features:
- Dense retrieval powered by multilingual-e5-small (SentenceTransformer)
- Sparse BM25 retrieval fallback & baseline (BM25Okapi via rank_bm25)
- Crop-filtering: restricts or prioritizes passages matching detected crops
- Strict verbatim passage extraction with full document & page attribution
- Compatible with CPU smoke tests and GPU-accelerated production runs
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None
    util = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


def tokenize_for_bm25(text: str) -> List[str]:
    """Simple multilingual word/token segmentation for BM25."""
    # Matches alphanumeric words across latin and Indic unicode scripts
    return [t.lower() for t in re.findall(r"\w+", text) if len(t) > 1]


class Retriever:
    """
    Multilingual PDF-grounded knowledge retriever.
    Supports Dense (e5-small), Sparse (BM25), and filtered search.
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        kb_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
    ):
        self.device = device
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.model = None
        self.chunks: List[Dict] = []
        self.corpus_embeddings = None
        self.bm25_index = None

        if model_name_or_path:
            self.load_model(model_name_or_path)

        if kb_path:
            self.load_kb(kb_path)

    def load_model(self, model_name_or_path: str):
        """Loads a SentenceTransformer embedding model."""
        if SentenceTransformer is None:
            raise ImportError("sentence_transformers is not installed.")
        print(f"[Retriever] Loading model from: {model_name_or_path} (device: {self.device})")
        self.model = SentenceTransformer(model_name_or_path, device=self.device)

    def load_kb(self, kb_path: Union[str, Path], build_bm25: bool = True):
        """Loads chunked knowledge base from jsonl or json."""
        kb_path = Path(kb_path)
        if not kb_path.exists():
            raise FileNotFoundError(f"KB file not found: {kb_path}")

        self.chunks = []
        if kb_path.suffix == ".jsonl":
            with open(kb_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        self.chunks.append(json.loads(line))
        else:
            with open(kb_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.chunks = data
                elif isinstance(data, dict) and "passages" in data:
                    self.chunks = data["passages"]

        # Filter out purely empty or non-text chunks
        self.chunks = [
            c for c in self.chunks
            if not c.get("no_text_layer") and not c.get("garbled") and c.get("text", "").strip()
        ]
        print(f"[Retriever] Loaded {len(self.chunks)} valid chunks from {kb_path.name}")

        if build_bm25 and BM25Okapi is not None:
            self._build_bm25_index()

    def _build_bm25_index(self):
        """Builds an in-memory BM25 index over the chunk texts."""
        tokenized_corpus = [tokenize_for_bm25(c.get("text", "")) for c in self.chunks]
        self.bm25_index = BM25Okapi(tokenized_corpus)

    def encode_corpus(self, batch_size: int = 32, show_progress_bar: bool = True):
        """Encodes all loaded KB passages using the dense model."""
        if self.model is None:
            raise ValueError("Dense model is not loaded.")
        if not self.chunks:
            raise ValueError("No chunks loaded in KB.")

        texts = [f"{self.passage_prefix}{c.get('text', '')}" for c in self.chunks]
        self.corpus_embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=True,
            device=self.device,
        )

    def search_bm25(
        self,
        query: str,
        k: int = 5,
        crop_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Sparse search using BM25.
        """
        if self.bm25_index is None:
            if BM25Okapi is None:
                raise ImportError("rank_bm25 is not installed.")
            self._build_bm25_index()

        tokenized_query = tokenize_for_bm25(query)
        if not tokenized_query:
            return []

        doc_scores = self.bm25_index.get_scores(tokenized_query)
        ranked_indices = np.argsort(doc_scores)[::-1]

        results = []
        for idx in ranked_indices:
            chunk = self.chunks[idx]
            # Crop filtering
            if crop_filter:
                crop_tags = [c.lower() for c in chunk.get("crop_tags", ["general"])]
                if crop_filter.lower() not in crop_tags and "general" not in crop_tags:
                    continue

            results.append({
                "score": float(doc_scores[idx]),
                "text": chunk.get("text", ""),
                "doc_name": chunk.get("doc_name", ""),
                "page": chunk.get("page", 0),
                "crop_tags": chunk.get("crop_tags", []),
                "disease_tags": chunk.get("disease_tags", []),
                "has_dosage": chunk.get("has_dosage", False),
                "table_uncertain": chunk.get("table_uncertain", False),
                "chunk_id": chunk.get("chunk_id", ""),
            })
            if len(results) >= k:
                break

        return results

    def search_dense(
        self,
        query: str,
        k: int = 5,
        crop_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Dense vector semantic search using multilingual-e5.
        """
        if self.model is None:
            raise ValueError("Dense model is not loaded.")
        if self.corpus_embeddings is None:
            self.encode_corpus()

        query_text = f"{self.query_prefix}{query}"
        query_embedding = self.model.encode(
            query_text,
            convert_to_tensor=True,
            device=self.device,
        )

        hits = util.semantic_search(
            query_embedding,
            self.corpus_embeddings,
            top_k=min(k * 4, len(self.chunks)),  # over-fetch for filtering
        )[0]

        results = []
        for hit in hits:
            idx = hit["corpus_id"]
            score = float(hit["score"])
            chunk = self.chunks[idx]

            # Crop filtering
            if crop_filter:
                crop_tags = [c.lower() for c in chunk.get("crop_tags", ["general"])]
                if crop_filter.lower() not in crop_tags and "general" not in crop_tags:
                    continue

            results.append({
                "score": score,
                "text": chunk.get("text", ""),
                "doc_name": chunk.get("doc_name", ""),
                "page": chunk.get("page", 0),
                "crop_tags": chunk.get("crop_tags", []),
                "disease_tags": chunk.get("disease_tags", []),
                "has_dosage": chunk.get("has_dosage", False),
                "table_uncertain": chunk.get("table_uncertain", False),
                "chunk_id": chunk.get("chunk_id", ""),
            })
            if len(results) >= k:
                break

        return results

    def search(
        self,
        query: str,
        k: int = 5,
        crop_filter: Optional[str] = None,
        mode: str = "dense",
    ) -> List[Dict]:
        """
        Unified search interface.
        mode: 'dense' or 'bm25'
        """
        if mode == "bm25" or self.model is None:
            return self.search_bm25(query, k=k, crop_filter=crop_filter)
        return self.search_dense(query, k=k, crop_filter=crop_filter)

