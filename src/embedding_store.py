from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import faiss
import numpy as np

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


@dataclass
class SearchResult:
    chunk_id: str
    chunk_text: str
    page_no: int
    source_file: str
    score: float
    distance: float
    bbox_list: List[Dict[str, Any]]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        payload = dict(self.metadata)
        payload.update(
            {
                "chunk_id": self.chunk_id,
                "chunk_text": self.chunk_text,
                "page_no": self.page_no,
                "source_file": self.source_file,
                "score": self.score,
                "distance": self.distance,
                "bbox_list": self.bbox_list,
            }
        )
        return payload


class EmbeddingStore:
    def __init__(self, model_name: str = DEFAULT_MODEL, index: Optional[faiss.Index] = None, metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        dim = self.model.get_sentence_embedding_dimension()
        self.index = index or faiss.IndexFlatIP(dim)
        self.metadata: List[Dict[str, Any]] = metadata or []

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        if not chunks:
            return
        texts = [c.get("chunk_text", "") for c in chunks]
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        self.index.add(embeddings)
        self.metadata.extend(chunks)
        print(f"[EmbeddingStore] Added {len(chunks)} chunks; total={len(self.metadata)}")

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        query_emb = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        scores, indices = self.index.search(query_emb, k)
        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item = dict(self.metadata[int(idx)])
            similarity = float(score)
            item["score"] = similarity
            item["distance"] = float(1.0 - similarity)
            item["retrieval_confidence"] = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
            results.append(item)
        return results

    def save(self, path: str | Path = "index_store") -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(target / "index.faiss"))
        (target / "metadata.json").write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        (target / "config.json").write_text(json.dumps({"model_name": self.model_name}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path = "index_store") -> "EmbeddingStore":
        target = Path(path)
        config_path = target / "config.json"
        model_name = DEFAULT_MODEL
        if config_path.exists():
            model_name = json.loads(config_path.read_text(encoding="utf-8")).get("model_name", DEFAULT_MODEL)
        index = faiss.read_index(str(target / "index.faiss"))
        metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
        return cls(model_name=model_name, index=index, metadata=metadata)
