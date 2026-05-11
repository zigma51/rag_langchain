# ── .env ──────────────────────────────────────────────────────────────────────
GOOGLE_API_KEY=AIza_your_key_here


# ── README.md ─────────────────────────────────────────────────────────────────
# DocMind — Production RAG Chatbot
#
# Stack: LangChain · Google Gemini Flash · FAISS · Streamlit
#
# ── Quick Start ───────────────────────────────────────────────────────────────
#
# 1. Clone / copy this project, then:
#
#    python -m venv .venv
#    source .venv/bin/activate          # Windows: .venv\Scripts\activate
#    pip install -r requirements.txt
#
# 2. Add your API key to .env  (or paste it in the sidebar at runtime)
#
# 3. Run:
#    streamlit run app.py
#
# 4. Open http://localhost:8501
#    → Upload a PDF/TXT/MD in the sidebar
#    → Click "Ingest & Index Document"
#    → Chat!
#
# ── Architecture ──────────────────────────────────────────────────────────────
#
#  Upload → ingest() → RecursiveCharacterTextSplitter (800 tok, 150 overlap)
#         → GoogleGenerativeAIEmbeddings (gemini-embedding-001)
#         → FAISS IndexFlatL2  (persisted to ./vector_store/)
#
#  Query  → ConversationalRetrievalChain
#           ├─ Condense question with chat history (6-turn window)
#           ├─ MMR retrieval (top-5, λ=0.7)
#           └─ Grounded generation (gemini-2.5-flash, temp=0.2)
#
# ── Notes ─────────────────────────────────────────────────────────────────────
#  • FAISS index is saved under vector_store/<filename>/
#    Re-ingesting the same file reuses the cached index automatically.
#  • Swap GEN_MODEL in rag/generator.py to gemini-2.5-pro for harder tasks.
#  • Raise TOP_K in the sidebar slider for dense technical documents.
#collaborators 
#jahnavi_capstone_project
