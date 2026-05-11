"""
embeddings.py — FAISS vector store management using Gemini embeddings.
Handles building, persisting, loading, and adding to the index.
"""

import logging
import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

logger = logging.getLogger(__name__)

VECTOR_STORE_DIR = "vector_store"
EMBED_MODEL = "models/gemini-embedding-001"


def get_embedder(api_key: str) -> GoogleGenerativeAIEmbeddings:
    """Return a Gemini embedding model configured for document retrieval."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=api_key,
        task_type="retrieval_document",
    )


def build_vector_store(
    chunks: List[Document],
    api_key: str,
    store_name: str = "index",
    persist: bool = True,
) -> FAISS:
    """
    Embed all chunks and build a FAISS index. Optionally persist to disk
    so subsequent runs skip re-embedding (expensive + slow).
    """
    logger.info(f"Building FAISS index from {len(chunks)} chunks…")
    embedder = get_embedder(api_key)
    vs = FAISS.from_documents(chunks, embedder)

    if persist:
        save_dir = Path(VECTOR_STORE_DIR) / store_name
        save_dir.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(save_dir))
        logger.info(f"Index persisted → {save_dir}")

    return vs


def load_vector_store(api_key: str, store_name: str = "index") -> FAISS:
    """
    Load a previously persisted FAISS index from disk.
    Raises FileNotFoundError if the index doesn't exist yet.
    """
    save_dir = Path(VECTOR_STORE_DIR) / store_name
    if not save_dir.exists():
        raise FileNotFoundError(
            f"No saved index at '{save_dir}'. "
            "Please ingest a document first."
        )
    embedder = get_embedder(api_key)
    vs = FAISS.load_local(
        str(save_dir),
        embedder,
        allow_dangerous_deserialization=True,  # safe: we wrote this index ourselves
    )
    logger.info(f"Loaded index from {save_dir}")
    return vs


def index_exists(store_name: str = "index") -> bool:
    """Quick check: has this index been built and saved?"""
    return (Path(VECTOR_STORE_DIR) / store_name / "index.faiss").exists()