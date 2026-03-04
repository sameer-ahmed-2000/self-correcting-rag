"""
Core vector store: FAISS (dense) + BM25 (sparse) with Reciprocal Rank Fusion.
Provides hybrid search combining semantic similarity and keyword matching.
"""
import os
import sys
import pickle
import logging
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("rag.vectorstore")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = Path("data/faiss.index")
DOCS_PATH  = Path("data/documents.pkl")
BM25_PATH  = Path("data/bm25.pkl")

# Module-level singletons
_model:     SentenceTransformer | None = None
_index:     faiss.IndexFlatIP | None   = None
_documents: List[dict] | None          = None
_bm25                                  = None   # rank_bm25.BM25Okapi or None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.astype(np.float32)


# ── Persistence helpers ────────────────────────────────────────────────────────

def _safe_pickle_dump(obj, path: Path) -> None:
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(10_000)
    try:
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        sys.setrecursionlimit(old_limit)


def _safe_pickle_load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _build_bm25(documents: List[dict]):
    """Build a BM25 index from the current document list."""
    try:
        from rank_bm25 import BM25Okapi
        tokenized = [doc["text"].lower().split() for doc in documents]
        return BM25Okapi(tokenized)
    except ImportError:
        logger.warning("rank-bm25 not installed — BM25 hybrid search disabled")
        return None


def load_index() -> Tuple[faiss.IndexFlatIP, List[dict]]:
    global _index, _documents, _bm25
    if _index is None:
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                "FAISS index not found. Please ingest documents first via /api/ingest/wikipedia."
            )
        _index = faiss.read_index(str(INDEX_PATH))
        try:
            _documents = _safe_pickle_load(DOCS_PATH)
        except (EOFError, pickle.UnpicklingError, Exception):
            logger.warning("documents.pkl is corrupted — clearing data store")
            _documents = []
            _index = None
            INDEX_PATH.unlink(missing_ok=True)
            DOCS_PATH.unlink(missing_ok=True)
            BM25_PATH.unlink(missing_ok=True)
            raise FileNotFoundError(
                "Data files were corrupted and cleared. Please re-ingest your documents."
            )
        # Load or rebuild BM25
        if BM25_PATH.exists():
            try:
                _bm25 = _safe_pickle_load(BM25_PATH)
            except Exception:
                _bm25 = _build_bm25(_documents)
        else:
            _bm25 = _build_bm25(_documents)
    return _index, _documents


def save_index(index: faiss.IndexFlatIP, documents: List[dict]) -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    _safe_pickle_dump(documents, DOCS_PATH)
    # Always rebuild + save BM25 after new documents
    bm25 = _build_bm25(documents)
    if bm25 is not None:
        _safe_pickle_dump(bm25, BM25_PATH)
    return bm25


def add_documents(chunks: List[dict]) -> None:
    global _index, _documents, _bm25

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    dim = embeddings.shape[1]

    if _index is None:
        if INDEX_PATH.exists():
            _index, _documents = load_index()
        else:
            _index = faiss.IndexFlatIP(dim)
            _documents = []

    _index.add(embeddings)
    _documents.extend(chunks)
    _bm25 = save_index(_index, _documents)
    logger.info(f"Added {len(chunks)} chunks | Total in store: {_index.ntotal}")


def similarity_search(query: str, k: int = 5) -> List[dict]:
    """
    Hybrid BM25 + FAISS search using Reciprocal Rank Fusion (RRF).
    Falls back to pure vector search if BM25 is not available.
    """
    index, documents = load_index()
    fetch_k = min(k * 3, len(documents))  # fetch more for fusion, then trim

    # ── 1. FAISS dense retrieval ──
    q_emb = embed_texts([query])
    scores, indices = index.search(q_emb, fetch_k)
    dense_ranked = [
        (int(idx), float(score))
        for score, idx in zip(scores[0], indices[0])
        if idx != -1
    ]

    # ── 2. BM25 sparse retrieval ──
    if _bm25 is not None:
        try:
            tokenized_query = query.lower().split()
            bm25_scores = _bm25.get_scores(tokenized_query)
            sparse_ranked = sorted(
                enumerate(bm25_scores), key=lambda x: x[1], reverse=True
            )[:fetch_k]
        except Exception as e:
            logger.warning(f"BM25 scoring failed: {e}")
            sparse_ranked = []
    else:
        sparse_ranked = []

    # ── 3. Reciprocal Rank Fusion ──
    RRF_K = 60
    rrf_scores: dict[int, float] = {}

    for rank, (doc_idx, _) in enumerate(dense_ranked):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (RRF_K + rank + 1)

    for rank, (doc_idx, _) in enumerate(sparse_ranked):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (RRF_K + rank + 1)

    # Sort by RRF score descending
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]

    results = []
    for doc_idx, rrf_score in fused:
        doc = documents[doc_idx].copy()
        doc["score"] = round(rrf_score, 4)
        results.append(doc)

    mode = "hybrid (FAISS + BM25)" if sparse_ranked else "dense only (FAISS)"
    logger.debug(f"similarity_search | mode={mode} | top_score={results[0]['score'] if results else 'n/a'}")
    return results
