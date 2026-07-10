import os
import tempfile
import streamlit as st

from rag_engine import PDFChatbot

st.set_page_config(page_title="Ask Your PDF", page_icon="📄", layout="wide")
st.title("📄 Ask Your PDF — RAG Chatbot")
st.caption("LangChain • ChromaDB • Google Gemini (with offline demo fallback)")

if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("1. Upload a PDF")
    uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])

    if not os.environ.get("GOOGLE_API_KEY"):
        st.info(
            "No GOOGLE_API_KEY found — running in **offline demo mode** "
            "(local TF-IDF embeddings + extractive answers).\n\n"
            "Set GOOGLE_API_KEY in your environment to enable full Gemini-powered answers.",
            icon="ℹ️",
        )

    if uploaded is not None and st.button("Process PDF", type="primary"):
        with st.spinner("Reading, chunking, and embedding your PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            bot = PDFChatbot(persist_directory="chroma_db_session")
            n_chunks = bot.ingest(tmp_path)
            st.session_state.chatbot = bot
            st.session_state.history = []
            st.success(f"Indexed {n_chunks} chunks from '{uploaded.name}'. Ask away!")

st.header("2. Chat with your document")

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        with st.expander("Sources"):
            for s in turn["sources"]:
                st.markdown(f"**Page {s.page}** (relevance: {s.score})")
                st.caption(s.text[:300] + "...")

question = st.chat_input("Ask a question about the uploaded PDF...")
if question:
    if st.session_state.chatbot is None:
        st.warning("Please upload and process a PDF first.")
    else:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.chatbot.ask(question)
                st.write(result.answer)
                with st.expander("Sources"):
                    for s in result.sources:
                        st.markdown(f"**Page {s.page}** (relevance: {s.score})")
                        st.caption(s.text[:300] + "...")
        st.session_state.history.append(
            {"question": question, "answer": result.answer, "sources": result.sources}
        )