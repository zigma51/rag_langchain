"""
helpers.py — Logging setup and small UI utilities.
"""

import logging
import sys
from datetime import datetime


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with timestamp + level prefix."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def format_sources_markdown(sources: list) -> str:
    """
    Render retrieved source chunks as a collapsible markdown block
    for display in Streamlit's st.expander.
    """
    if not sources:
        return "_No sources retrieved._"

    lines = []
    for i, s in enumerate(sources, 1):
        score_pct = f"{s['score'] * 100:.1f}%" if s["score"] else "N/A"
        lines.append(
            f"**[{i}] {s['source']} · Page {s['page']} · "
            f"Chunk #{s['chunk_id']} · Relevance: {score_pct}**\n\n"
            f"> {s['content'][:400].strip()}{'…' if len(s['content']) > 400 else ''}"
        )
    return "\n\n---\n\n".join(lines)


def timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")