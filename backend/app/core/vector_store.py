"""
Core vector store management using FAISS + HuggingFace sentence-transformers.
Embedding model: all-MiniLM-L6-v2 (fast, lightweight, no API key needed)
"""
import os
import sys
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = Path("data/faiss.index")
DOCS_PATH = Path("data/documents.pkl")

_model: SentenceTransformer | None = None
_index: faiss.IndexFlatIP | None = None
_documents: List[dict] | None = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBED_MODEL_NAME}")
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.astype(np.float32)


def load_index() -> Tuple[faiss.IndexFlatIP, List[dict]]:
    global _index, _documents
    if _index is None:
        if not INDEX_PATH.exists():
            raise FileNotFoundError("FAISS index not found. Please ingest documents first via /api/ingest/wikipedia.")
        _index = faiss.read_index(str(INDEX_PATH))
        try:
            with open(DOCS_PATH, "rb") as f:
                _documents = pickle.load(f)
        except (EOFError, pickle.UnpicklingError, Exception):
            # Corrupted docs file — reset documents to be consistent with index
            print("[WARN] documents.pkl is corrupted. Resetting document store.")
            _documents = []
            _index = None
            # Delete the broken files so next ingest starts fresh
            INDEX_PATH.unlink(missing_ok=True)
            DOCS_PATH.unlink(missing_ok=True)
            raise FileNotFoundError("Data files were corrupted and have been cleared. Please re-ingest your documents.")
    return _index, _documents


def save_index(index: faiss.IndexFlatIP, documents: List[dict]) -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    # Raise recursion limit temporarily for large document lists
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(10000)
    try:
        with open(DOCS_PATH, "wb") as f:
            pickle.dump(documents, f, protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        sys.setrecursionlimit(old_limit)


def add_documents(chunks: List[dict]) -> None:
    """
    chunks: list of {"text": str, "source": str, "title": str}
    """
    global _index, _documents

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
    save_index(_index, _documents)
    print(f"Added {len(chunks)} chunks. Total: {_index.ntotal}")


def similarity_search(query: str, k: int = 5) -> List[dict]:
    index, documents = load_index()
    q_emb = embed_texts([query])
    scores, indices = index.search(q_emb, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        doc = documents[idx].copy()
        doc["score"] = float(score)
        results.append(doc)
    return results
