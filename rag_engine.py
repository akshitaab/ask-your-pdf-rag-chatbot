"""
Core RAG engine: PDF -> chunks -> embeddings -> ChromaDB -> retrieval -> answer.
Uses Google Gemini if GOOGLE_API_KEY is set, otherwise falls back to a local
offline TF-IDF embedding + extractive answer mode (no API key needed).
"""
import os
import shutil
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

warnings.filterwarnings("ignore", message="Relevance scores must be between 0 and 1")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

from embeddings import LocalTfidfEmbeddings

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")


@dataclass
class SourceChunk:
    page: int
    text: str
    score: Optional[float] = None


@dataclass
class RagAnswer:
    question: str
    answer: str
    sources: List[SourceChunk] = field(default_factory=list)
    mode: str = "demo"


class PDFChatbot:
    def __init__(self, persist_directory: str = "chroma_db", collection_name: str = "pdf_chatbot"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.vectorstore: Optional[Chroma] = None
        self.use_gemini = GOOGLE_API_KEY is not None

        if self.use_gemini:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
        else:
            self.embeddings = LocalTfidfEmbeddings()
            self.llm = None

    def ingest(self, pdf_path: str, chunk_size: int = 800, chunk_overlap: int = 120) -> int:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks: List[Document] = splitter.split_documents(pages)

        if not self.use_gemini:
            self.embeddings.embed_documents([c.page_content for c in chunks])

        # Use a fresh, unique directory each time instead of deleting the old
        # one — avoids Windows file-lock errors when a previous Chroma client
        # hasn't released its handle yet.
        import uuid
        unique_dir = f"{self.persist_directory}_{uuid.uuid4().hex[:8]}"

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=unique_dir,
        )
        return len(chunks)

    def retrieve(self, question: str, k: int = 4) -> List[SourceChunk]:
        if self.vectorstore is None:
            raise RuntimeError("No PDF ingested yet. Call ingest() first.")
        results = self.vectorstore.similarity_search_with_relevance_scores(question, k=k)
        return [
            SourceChunk(page=doc.metadata.get("page", 0) + 1, text=doc.page_content, score=round(score, 3))
            for doc, score in results
        ]

    def ask(self, question: str, k: int = 4) -> RagAnswer:
        sources = self.retrieve(question, k=k)
        context = "\n\n".join(f"[Page {s.page}] {s.text}" for s in sources)

        if self.use_gemini:
            prompt = (
                "You are a helpful assistant answering questions using ONLY the "
                "context below, extracted from a PDF document. If the answer is "
                "not in the context, say you don't know.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            )
            response = self.llm.invoke(prompt)
            return RagAnswer(question=question, answer=response.content, sources=sources, mode="gemini")

        answer = self._extractive_answer(question, sources)
        return RagAnswer(question=question, answer=answer, sources=sources, mode="demo")

    @staticmethod
    def _extractive_answer(question: str, sources: List[SourceChunk]) -> str:
        if not sources:
            return "I couldn't find anything relevant to that question in the document."
        best = sources[0]
        return (
            f"(demo mode — no GOOGLE_API_KEY set, showing best-matching excerpt)\n\n"
            f"From page {best.page}: {best.text.strip()}"
        )