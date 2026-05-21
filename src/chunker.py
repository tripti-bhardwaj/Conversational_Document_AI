from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Iterable, List


@dataclass
class Chunk:
    chunk_id: str
    chunk_text: str
    page_no: int
    bbox_list: List[Dict[str, Any]]
    source_file: str
    strategy: str = "fixed"
    token_start: int = 0
    token_end: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TextChunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 50, strategy: str = "fixed") -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be >= 0 and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy

    def chunk_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        for page in pages:
            text = (page.get("text") or "").strip()
            if not text:
                continue
            if self.strategy == "sentence":
                page_chunks = self._sentence_chunks(text)
            elif self.strategy == "semantic":
                page_chunks = self._semantic_chunks(text)
            else:
                page_chunks = self._fixed_chunks(text)
            for ordinal, (chunk_text, start, end) in enumerate(page_chunks):
                base_name = str(page.get("source_file", "document")).rsplit(".", 1)[0].replace(" ", "_")
                chunk = Chunk(
                    chunk_id=f"{base_name}_p{page.get('page_no', 1)}_c{ordinal}",
                    chunk_text=chunk_text,
                    page_no=int(page.get("page_no", 1)),
                    bbox_list=page.get("bbox_list", []),
                    source_file=page.get("source_file", "unknown"),
                    strategy=self.strategy,
                    token_start=start,
                    token_end=end,
                    metadata=page.get("metadata", {}) or {},
                )
                chunks.append(chunk.to_dict())
        print(f"[Chunker] {len(pages)} pages -> {len(chunks)} chunks ({self.strategy})")
        return chunks

    def _fixed_chunks(self, text: str) -> List[tuple[str, int, int]]:
        words = text.split()
        output: List[tuple[str, int, int]] = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            output.append((" ".join(words[start:end]), start, end))
            if end == len(words):
                break
            start += self.chunk_size - self.overlap
        return output

    def _sentence_chunks(self, text: str) -> List[tuple[str, int, int]]:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?।])\s+", text) if s.strip()]
        chunks: List[tuple[str, int, int]] = []
        current: List[str] = []
        start = 0
        cursor = 0
        for sentence in sentences:
            count = len(sentence.split())
            if current and len(" ".join(current).split()) + count > self.chunk_size:
                chunk_text = " ".join(current)
                chunks.append((chunk_text, start, cursor))
                overlap_words = chunk_text.split()[-self.overlap :] if self.overlap else []
                current = [" ".join(overlap_words), sentence] if overlap_words else [sentence]
                start = max(0, cursor - len(overlap_words))
            else:
                current.append(sentence)
            cursor += count
        if current:
            chunks.append((" ".join(current).strip(), start, cursor))
        return chunks

    def _semantic_chunks(self, text: str) -> List[tuple[str, int, int]]:
        sections = [s.strip() for s in re.split(r"\n\s*\n|(?=\n?[A-Z][A-Za-z /-]{2,}:)", text) if s.strip()]
        if len(sections) <= 1:
            return self._sentence_chunks(text)
        chunks: List[tuple[str, int, int]] = []
        cursor = 0
        buffer: List[str] = []
        start = 0
        for section in sections:
            count = len(section.split())
            pending_count = len(" ".join(buffer).split()) + count
            if buffer and pending_count > self.chunk_size:
                chunk_text = "\n".join(buffer)
                end = cursor
                chunks.append((chunk_text, start, end))
                buffer = [section]
                start = cursor
            else:
                buffer.append(section)
            cursor += count
        if buffer:
            chunks.append(("\n".join(buffer), start, cursor))
        return chunks


def chunk_pages(pages: list, chunk_size: int = 512, overlap: int = 50, strategy: str = "fixed") -> list:
    return TextChunker(chunk_size=chunk_size, overlap=overlap, strategy=strategy).chunk_pages(pages)
