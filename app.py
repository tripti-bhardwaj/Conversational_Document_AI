import base64
import json
import tempfile

import faiss
import streamlit as st

from src.chunker import chunk_pages
from src.document_loader import load_document
from src.highlight import highlight_pdf
from src.rag_engine import query_engine
from src.vector_store import add_chunks, get_model

st.set_page_config(page_title="Conversational Document AI", page_icon="📄", layout="wide")

if "index" not in st.session_state:
    st.session_state.index = None
if "metadata" not in st.session_state:
    st.session_state.metadata = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None
if "source_path" not in st.session_state:
    st.session_state.source_path = None
if "highlighted" not in st.session_state:
    st.session_state.highlighted = None
if "documents" not in st.session_state:
    st.session_state.documents = []

with st.sidebar:
    st.title("Document AI")
    uploaded = st.file_uploader("Upload a document", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"])
    strategy = st.selectbox("Chunking strategy", ["fixed", "sentence", "semantic"], index=0)
    k = st.slider("Sources to retrieve", 1, 10, 5)

    if uploaded and st.button("Index Document"):
        with st.spinner("Extracting text and building the vector index..."):
            suffix = "." + uploaded.name.split(".")[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            pages = load_document(tmp_path)
            chunks = chunk_pages(pages, strategy=strategy)
            if not chunks:
                st.warning(
                    "No text could be extracted from this file. Install Tesseract or upload a clearer/digital PDF, then try again."
                )
                st.stop()
            model_dim = get_model().get_sentence_embedding_dimension()
            if st.session_state.index is None:
                st.session_state.index = faiss.IndexFlatIP(model_dim)
            add_chunks(chunks, st.session_state.index, st.session_state.metadata)

            st.session_state.pdf_path = tmp_path
            st.session_state.source_path = tmp_path
            st.session_state.highlighted = None
            st.session_state.documents.append({"name": uploaded.name, "pages": len(pages), "chunks": len(chunks)})
        st.success(f"Indexed {len(chunks)} chunks from {uploaded.name}")

    if st.session_state.documents:
        st.markdown("---")
        st.caption("Indexed documents")
        for doc in st.session_state.documents:
            st.write(f"{doc['name']} · {doc['pages']} pages · {doc['chunks']} chunks")

st.title("Conversational Document AI")

if st.session_state.index is None:
    st.info("Upload and index a document to begin.")
    st.stop()

question = st.chat_input("Ask in English, Hindi, or code-mixed language...")

col_chat, col_pdf = st.columns([1, 1], gap="large")

with col_chat:
    st.subheader("Ask a Question")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            meta = msg.get("meta")
            if meta:
                confidence = meta.get("confidence", {})
                label = confidence.get("label", "unknown")
                score = confidence.get("combined_confidence", 0)
                st.caption(f"Confidence: {label} ({score:.2f}) · Language: {meta.get('language', 'unknown')}")

    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = query_engine(question, st.session_state.index, st.session_state.metadata, k=k, chat_history=st.session_state.chat_history)
            answer = result["answer"]
            st.write(answer)
            confidence = result.get("confidence", {})
            label = confidence.get("label", "unknown")
            score = confidence.get("combined_confidence", 0)
            if label == "low":
                st.warning(f"Low confidence ({score:.2f})")
            elif label == "medium":
                st.info(f"Medium confidence ({score:.2f})")
            else:
                st.success(f"High confidence ({score:.2f})")

        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
                "meta": {"language": result.get("language"), "fallback": result.get("fallback"), "confidence": confidence},
            }
        )

        if not result.get("fallback") and st.session_state.pdf_path and st.session_state.pdf_path.endswith(".pdf"):
            st.session_state.highlighted = highlight_pdf(st.session_state.pdf_path, result.get("sources", []))

    if st.session_state.chat_history:
        col_a, col_b = st.columns(2)
        if col_a.button("Clear chat"):
            st.session_state.chat_history = []
            st.session_state.highlighted = None
            st.rerun()
        export_payload = json.dumps(st.session_state.chat_history, ensure_ascii=False, indent=2)
        col_b.download_button("Export JSON", export_payload, file_name="conversation.json", mime="application/json")

with col_pdf:
    st.subheader("Source Highlight")
    pdf_to_show = None
    source_path = st.session_state.source_path or st.session_state.pdf_path
    source_ext = source_path.rsplit(".", 1)[-1].lower() if source_path and "." in source_path else ""
    if st.session_state.highlighted:
        pdf_to_show = st.session_state.highlighted
        st.caption("Highlighted text shows the retrieved source chunks.")
        st.download_button("Download highlighted PDF", data=pdf_to_show, file_name="highlighted.pdf", mime="application/pdf")
    elif source_path and source_ext == "pdf":
        with open(source_path, "rb") as f:
            pdf_to_show = f.read()

    if pdf_to_show:
        b64 = base64.b64encode(pdf_to_show).decode()
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="720" style="border:1px solid #d0d7de;border-radius:6px;"></iframe>',
            unsafe_allow_html=True,
        )
    elif source_path and source_ext in {"png", "jpg", "jpeg", "tif", "tiff"}:
        st.image(source_path, width='stretch')
    else:
        st.info("Source preview appears here after indexing a document.")
