from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List

import faiss
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from src.chunker import chunk_pages
from src.document_loader import load_document
from src.highlight import highlight_pdf
from src.rag_engine import query_engine
from src.vector_store import add_chunks, get_model

app = FastAPI(title="Conversational Document AI")

STATE: Dict[str, Any] = {"index": None, "metadata": [], "documents": {}, "last_sources": []}


class QueryRequest(BaseModel):
    question: str
    k: int = 5


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "document.pdf").suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        path = tmp.name

    pages = load_document(path)
    chunks = chunk_pages(pages)
    model_dim = get_model().get_sentence_embedding_dimension()
    if STATE["index"] is None:
        STATE["index"] = faiss.IndexFlatIP(model_dim)
    add_chunks(chunks, STATE["index"], STATE["metadata"])
    STATE["documents"][file.filename] = {"path": path, "chunks": len(chunks), "pages": len(pages)}
    return {"filename": file.filename, "pages": len(pages), "chunks": len(chunks)}


@app.post("/query")
def query(payload: QueryRequest):
    if STATE["index"] is None:
        raise HTTPException(status_code=400, detail="No documents indexed yet")
    result = query_engine(payload.question, STATE["index"], STATE["metadata"], k=payload.k)
    STATE["last_sources"] = result.get("sources", [])
    return result


@app.get("/documents")
def documents():
    return STATE["documents"]


@app.get("/highlight/{filename}")
def highlight(filename: str):
    doc = STATE["documents"].get(filename)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    pdf_path = doc["path"]
    if not str(pdf_path).lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Highlighting is only available for PDFs")
    data = highlight_pdf(pdf_path, STATE.get("last_sources", []))
    return Response(content=data, media_type="application/pdf")
