"""
generator.py — Conversational RAG chain with multi-turn memory.
Built with LCEL (LangChain Expression Language) for LangChain v1.x+.
ConversationalRetrievalChain and ConversationBufferWindowMemory were
removed in v1.0 — this uses the modern runnable pipeline instead.
"""

import logging
from typing import Dict, List, Tuple

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

from rag.retriever import get_retriever, retrieve_with_scores

logger = logging.getLogger(__name__)

GEN_MODEL = "gemini-2.5-flash"
TEMPERATURE = 0.2
MEMORY_WINDOW = 6  # max past turns to keep


# ── Prompts ────────────────────────────────────────────────────────────────────

# Step 1: condense follow-up questions into a standalone question
CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the conversation history and a follow-up question, "
     "rewrite the follow-up as a standalone question. "
     "Return ONLY the rewritten question, nothing else."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

# Step 2: answer using retrieved context
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are a strict Document Assistant.

RULES:
1. Answer ONLY from the CONTEXT provided below.
2. If the answer is not in the context, say exactly:
   "I'm sorry, that information is not available in the provided document."
3. Be concise but complete. Use bullet points when listing multiple items.
4. Cite the source page/chunk when possible.
5. Never hallucinate or use outside knowledge.

CONTEXT:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


# ── Chain builder ──────────────────────────────────────────────────────────────

def build_chain(
    vector_store: FAISS,
    api_key: str,
    top_k: int = 5,
) -> dict:
    """
    Returns a dict with:
      - 'chain'  : the LCEL runnable
      - 'history': list of messages (managed manually, replaces BufferWindowMemory)
      - 'retriever': for score lookups
    """
    llm = ChatGoogleGenerativeAI(
        model=GEN_MODEL,
        google_api_key=api_key,
        temperature=TEMPERATURE,
    )

    retriever = get_retriever(vector_store, top_k=top_k)

    def format_docs(docs):
        return "\n---\n".join(d.page_content for d in docs)

    # Condense question → retrieve → answer
    condense_chain = CONDENSE_PROMPT | llm | StrOutputParser()

    def get_standalone(inputs):
        """If no history, use question as-is. Otherwise condense."""
        if not inputs.get("chat_history"):
            return inputs["question"]
        return condense_chain.invoke(inputs)

    rag_chain = (
        RunnablePassthrough.assign(
            standalone=get_standalone,
        )
        | RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["standalone"])),
            question=lambda x: x["standalone"],
        )
        | RunnablePassthrough.assign(
            answer=QA_PROMPT | llm | StrOutputParser()
        )
    )

    logger.info("LCEL RAG chain ready.")
    return {
        "chain": rag_chain,
        "history": [],
        "retriever": retriever,
    }


# ── Query ──────────────────────────────────────────────────────────────────────

def query_chain(
    chain_bundle: dict,
    question: str,
    vector_store: FAISS,
) -> Tuple[str, List[Dict]]:
    """
    Run a question through the chain. Manages chat history externally.
    Returns (answer, sources).
    """
    chain = chain_bundle["chain"]
    history = chain_bundle["history"]

    result = chain.invoke({
        "question": question,
        "chat_history": history[-MEMORY_WINDOW * 2:],  # keep last N turns
    })

    answer = result.get("answer", "No answer generated.")

    # Update history
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))

    # Attach relevance scores
    scored = retrieve_with_scores(vector_store, question, top_k=5)
    score_map = {doc.page_content[:80]: score for doc, score in scored}

    sources = []
    seen = set()

    # Re-retrieve to get source docs (chain doesn't expose them directly)
    raw_docs = chain_bundle["retriever"].invoke(question)
    for doc in raw_docs:
        snippet = doc.page_content[:80]
        if snippet in seen:
            continue
        seen.add(snippet)
        sources.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "?"),
            "chunk_id": doc.metadata.get("chunk_id", "?"),
            "score": round(score_map.get(snippet, 0.0), 3),
        })

    return answer, sources