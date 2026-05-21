from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List

import faiss
import numpy as np

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

from sentence_transformers import SentenceTransformer

from src.embedding_store import DEFAULT_MODEL, EmbeddingStore

_model = None


def get_model(model_name: str = DEFAULT_MODEL):
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model


def add_chunks(chunks, index, metadata):
    if not chunks:
        return
    model = get_model()
    texts = [c["chunk_text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    index.add(embeddings)
    metadata.extend(chunks)
    print(f"[VectorStore] Added {len(chunks)} chunks")


def search(query, index, metadata, k=5):
    if index is None or index.ntotal == 0:
        return []
    model = get_model()
    query_emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    k = min(k, index.ntotal)
    raw_scores, indices = index.search(query_emb, k)
    results = []
    for raw_score, idx in zip(raw_scores[0], indices[0]):
        if idx < 0:
            continue
        result = dict(metadata[int(idx)])
        score = float(raw_score)
        if isinstance(index, faiss.IndexFlatL2):
            similarity = 1.0 / (1.0 + max(score, 0.0))
            distance = score
        else:
            similarity = score
            distance = 1.0 - score
        result["score"] = similarity
        result["distance"] = float(distance)
        result["retrieval_confidence"] = max(0.0, min(1.0, similarity))
        results.append(result)
    return results


def save(index, metadata, path="index_store"):
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(target / "index.faiss"))
    import pickle

    with open(target / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)


def load(path="index_store"):
    target = Path(path)
    index = faiss.read_index(str(target / "index.faiss"))
    import pickle

    with open(target / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, metadata
