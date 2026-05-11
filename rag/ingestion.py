"""
ingestion.py — Document loading and semantic chunking pipeline.
Supports PDF, TXT, and MD files. Uses RecursiveCharacterTextSplitter
with sensible production defaults.
"""

import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# --- Defaults (easy to override) ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_documents(file_path: str) -> List[Document]:
    """
    Load a document from disk. Returns a list of LangChain Documents.
    Each Document carries page_content + metadata (source, page, etc.).
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Loading document: {path.name}")

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    else:
        loader = TextLoader(str(path), encoding="utf-8")

    docs = loader.load()

    # Stamp every doc with a clean source name
    for doc in docs:
        doc.metadata["source"] = path.name

    logger.info(f"Loaded {len(docs)} raw page(s) from {path.name}")
    return docs


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split documents into overlapping semantic chunks using a hierarchy of
    separators: paragraphs → lines → sentences → words → characters.
    Overlap preserves "connective tissue" across chunk boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    # Add chunk index to metadata for traceability
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    logger.info(f"Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def ingest(file_path: str) -> List[Document]:
    """Convenience: load + split in one call."""
    docs = load_documents(file_path)
    return split_documents(docs)