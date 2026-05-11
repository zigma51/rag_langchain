"""
app.py — Production RAG Chatbot UI powered by Streamlit.

Run with:
    streamlit run app.py
"""

import os
import tempfile
import logging

import streamlit as st
from dotenv import load_dotenv

from rag.ingestion import ingest
from rag.embeddings import build_vector_store, load_vector_store, index_exists
from rag.generator import build_chain, query_chain
from utils.helpers import setup_logging, format_sources_markdown, timestamp

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()
setup_logging(logging.WARNING)  # keep console quiet during UI use

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind — RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main chat area */
    .stChatMessage { border-radius: 12px; margin-bottom: 0.5rem; }

    /* Source expander */
    .streamlit-expanderHeader { font-size: 0.85rem; color: #6b7280; }

    /* Sidebar upload zone */
    section[data-testid="stSidebar"] .stFileUploader { border: 2px dashed #4f46e5; border-radius: 10px; }

    /* Status badges */
    .status-ready   { color: #16a34a; font-weight: 600; }
    .status-pending { color: #d97706; font-weight: 600; }
    .status-error   { color: #dc2626; font-weight: 600; }

    /* Answer container */
    .answer-box {
        background: #f8fafc;
        border-left: 4px solid #4f46e5;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-top: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State Initialisation ──────────────────────────────────────────────
def init_state():
    defaults = {
        "messages": [],          # [{role, content, sources, ts}]
        "vector_store": None,
        "chain": None,
        "doc_name": None,
        "api_key": "",
        "top_k": 5,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://em-content.zobj.net/source/apple/354/brain_1f9e0.png", width=60)
    st.title("DocMind")
    st.caption("Retrieval-Augmented Generation · Gemini + FAISS")
    st.divider()

    # --- API Key ---
    st.subheader("🔑 API Key")
    api_input = st.text_input(
        "Google AI API Key",
        value=st.session_state.api_key or os.getenv("GOOGLE_API_KEY", ""),
        type="password",
        placeholder="AIza…",
        help="Get yours at https://aistudio.google.com/apikey",
    )
    if api_input:
        st.session_state.api_key = api_input

    st.divider()

    # --- Document Upload ---
    st.subheader("📄 Document")
    uploaded_file = st.file_uploader(
        "Upload PDF / TXT / MD",
        type=["pdf", "txt", "md"],
        help="Your file is processed locally — nothing is sent to any third party except embeddings to Google.",
    )

    top_k = st.slider(
        "Chunks to retrieve (top-k)",
        min_value=2, max_value=10, value=5,
        help="More chunks = richer context but slower generation.",
    )
    st.session_state.top_k = top_k

    ingest_btn = st.button("⚡ Ingest & Index Document", use_container_width=True, type="primary")

    # --- Ingest Logic ---
    if ingest_btn:
        if not st.session_state.api_key:
            st.error("Please enter your Google API key first.")
        elif uploaded_file is None:
            st.warning("Please upload a document.")
        else:
            with st.spinner("Loading, chunking, embedding… this may take a moment."):
                try:
                    # Write upload to a temp file so LangChain loaders can read it
                    suffix = "." + uploaded_file.name.split(".")[-1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    store_name = uploaded_file.name.replace(" ", "_")
                    chunks = ingest(tmp_path)

                    vs = build_vector_store(
                        chunks,
                        api_key=st.session_state.api_key,
                        store_name=store_name,
                        persist=True,
                    )
                    st.session_state.vector_store = vs
                    st.session_state.chain = build_chain(
                        vs,
                        api_key=st.session_state.api_key,
                        top_k=top_k,
                    )
                    st.session_state.doc_name = uploaded_file.name
                    st.session_state.messages = []   # reset chat on new doc
                    # chain is now a bundle dict, not a plain chain object
                    st.success(f"✅ Indexed **{len(chunks)}** chunks from **{uploaded_file.name}**")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    st.divider()

    # --- Status ---
    st.subheader("📊 Status")
    if st.session_state.doc_name:
        st.markdown(f'<span class="status-ready">● Ready</span>  `{st.session_state.doc_name}`', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pending">● Awaiting document</span>', unsafe_allow_html=True)

    # --- Clear Chat ---
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        # Rebuild chain to reset memory too
        if st.session_state.vector_store and st.session_state.api_key:
            st.session_state.chain = build_chain(
                st.session_state.vector_store,
                api_key=st.session_state.api_key,
                top_k=st.session_state.top_k,
            )  # returns a fresh bundle dict, history is empty
        st.rerun()


# ── Main Chat Interface ────────────────────────────────────────────────────────
st.title("🧠 DocMind — Chat with your Document")

if not st.session_state.doc_name:
    st.info("👈 **Upload a document and click 'Ingest & Index'** in the sidebar to get started.")
    st.stop()

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 View {len(msg['sources'])} source chunk(s) · {msg.get('ts','')}"):
                st.markdown(format_sources_markdown(msg["sources"]))

# Chat input
if prompt := st.chat_input(f"Ask anything about {st.session_state.doc_name}…"):

    # Guard: need API key + chain
    if not st.session_state.api_key:
        st.warning("Please enter your API key in the sidebar.")
        st.stop()

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                answer, sources = query_chain(
                    st.session_state.chain,
                    prompt,
                    st.session_state.vector_store,
                )
            except Exception as e:
                answer = f"⚠️ Error: {e}"
                sources = []

        st.markdown(answer)
        ts = timestamp()
        if sources:
            with st.expander(f"📚 View {len(sources)} source chunk(s) · {ts}"):
                st.markdown(format_sources_markdown(sources))

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "ts": ts,
    })