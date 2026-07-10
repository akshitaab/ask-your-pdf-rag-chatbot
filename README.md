# 📄 Ask Your PDF — A RAG-based PDF Chatbot

Upload any PDF and ask questions about it in plain English. The app retrieves the
most relevant passages and uses an LLM to generate a grounded answer — with page-level
source citations so you can always verify where the answer came from.

Built with **LangChain**, **ChromaDB**, and **Google Gemini**, following a standard
Retrieval-Augmented Generation (RAG) architecture.

## How it works
PDF Upload
│
▼
Text Extraction (PyPDFLoader)
│
▼
Chunking (RecursiveCharacterTextSplitter, 800 chars, 120 overlap)
│
▼
Embedding (Gemini text-embedding-004)
│
▼
Vector Store (ChromaDB)
│
▼
User Question ──► Similarity Search (top-k chunks)
│
▼
Context-augmented Prompt ──► Gemini 1.5 Flash ──► Answer + Sources
## Features

- 🔍 Semantic search over PDF content using vector embeddings, not just keyword match
- 📚 Source-cited answers — every response links back to the exact page it came from
- 💬 Conversational UI built with Streamlit
- 🧠 Swappable embedding backend — works out of the box with a local TF-IDF fallback
  (zero API key, zero cost) and upgrades automatically to Gemini embeddings + LLM the moment
  a `GOOGLE_API_KEY` is set
- ⚙️ Clean separation between ingestion, retrieval, and generation (`rag_engine.py`)

## Demo mode vs. Gemini mode

| | **Demo mode** (default) | **Gemini mode** (with API key) |
|---|---|---|
| Embeddings | Local TF-IDF (scikit-learn) | Gemini `text-embedding-004` |
| Answer generation | Extractive (best-matching excerpt) | Generative (Gemini 1.5 Flash) |
| Cost | Free, fully offline | Free tier via Google AI Studio |

## Quickstart

```bash
git clone https://github.com/akshitaab/ask-your-pdf-rag-chatbot.git
cd ask-your-pdf-rag-chatbot
pip install -r requirements.txt

# Optional — enables Gemini-powered answers
export GOOGLE_API_KEY="your_key_here"

streamlit run app.py
```

## Project structure
ask-your-pdf-rag-chatbot/
├── app.py                  # Streamlit UI
├── rag_engine.py            # Core RAG pipeline
├── embeddings.py            # Offline TF-IDF embedding fallback
├── demo.py                  # CLI script
├── sample_data/
│   ├── make_sample_pdf.py
│   └── employee_handbook.pdf
├── requirements.txt
## Tech stack

`Python` · `LangChain` · `ChromaDB` · `Google Gemini` · `Streamlit` · `scikit-learn`

## License

MIT