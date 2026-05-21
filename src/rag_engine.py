from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langdetect import LangDetectException, detect
from llama_index.llms.ollama import Ollama

from src.confidence import combined_confidence
from src.invoice_extractor import answer_invoice_question
from src.vector_store import search


DEFAULT_SYSTEM_PROMPT = """You are a document QA assistant.
Answer only from the supplied context. If the answer is missing, say:
\"I could not find a reliable answer.\" Keep answers concise and cite source numbers when useful."""


@dataclass
class ChatMemory:
    turns: List[Dict[str, str]] = field(default_factory=list)
    max_turns: int = 6

    def add(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})
        if len(self.turns) > self.max_turns * 2:
            self.turns = self.turns[-self.max_turns * 2 :]

    def render(self) -> str:
        lines = []
        for turn in self.turns[-self.max_turns * 2 :]:
            role = "User" if turn.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {turn.get('content', '')}")
        return "\n".join(lines)


class RAGEngine:
    def __init__(self, model_name: str = "llama3", similarity_threshold: float = 0.25, request_timeout: float = 300.0) -> None:
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.request_timeout = request_timeout
        self.memory = ChatMemory()

    def build_prompt(self, query: str, context_chunks: List[Dict[str, Any]], history: Optional[List[Dict[str, str]]] = None) -> str:
        context = "\n\n".join(f"[Source {i + 1} | {c.get('source_file')} p.{c.get('page_no')}] {c.get('chunk_text', '')}" for i, c in enumerate(context_chunks))
        memory = ChatMemory(history or self.memory.turns).render()
        return f"""{DEFAULT_SYSTEM_PROMPT}

CONTEXT:
{context}

CHAT HISTORY:
{memory}

QUESTION: {query}

ANSWER:"""

    def answer(self, question: str, index, metadata, k: int = 5, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        language = safe_detect(question)
        invoice_answer = answer_invoice_question(question, metadata)
        if invoice_answer:
            answer = invoice_answer["answer"]
            sources = metadata[:1]
            confidence = {"retrieval_confidence": 1.0, "answer_confidence": 1.0, "combined_confidence": 1.0, "label": "high"}
            if chat_history is None:
                self.memory.add("user", question)
                self.memory.add("assistant", answer)
            return {"answer": answer, "language": language, "sources": sources, "fallback": False, "confidence": confidence}

        results = search(question, index, metadata, k)
        best_score = max((float(r.get("retrieval_confidence", r.get("score", 0.0))) for r in results), default=0.0)
        fallback = not results or best_score < self.similarity_threshold
        if fallback:
            answer = "I could not find a reliable answer."
            confidence = combined_confidence(results, answer, fallback=True).to_dict()
            return {"answer": answer, "language": language, "sources": results, "fallback": True, "confidence": confidence}

        prompt = self.build_prompt(question, results, history=chat_history)
        llm = Ollama(model=self.model_name, request_timeout=self.request_timeout, context_window=4096)
        response = llm.complete(prompt)
        answer = response.text.strip()

        if chat_history is None:
            self.memory.add("user", question)
            self.memory.add("assistant", answer)
        confidence = combined_confidence(results, answer, fallback=False).to_dict()
        return {"answer": answer, "language": language, "sources": results, "fallback": False, "confidence": confidence}


def safe_detect(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


_default_engine = RAGEngine()


def build_prompt(query, context_chunks, history):
    return _default_engine.build_prompt(query, context_chunks, history=history)


def query_engine(question, index, metadata, k=5, chat_history=None):
    return _default_engine.answer(question, index, metadata, k=k, chat_history=chat_history)
