"""Runs the full RAG pipeline end-to-end and prints results."""
from rag_engine import PDFChatbot

PDF_PATH = "sample_data/employee_handbook.pdf"

QUESTIONS = [
    "How many paid leaves does an employee get per year?",
    "Can I work from home every day?",
    "What is the maximum reimbursable amount per month for travel expenses?",
]

def main():
    print("=" * 70)
    print("ASK YOUR PDF — RAG CHATBOT DEMO")
    print("=" * 70)

    bot = PDFChatbot()
    print(f"\nIngesting: {PDF_PATH}")
    n_chunks = bot.ingest(PDF_PATH)
    print(f"Created and embedded {n_chunks} chunks into ChromaDB.")
    print(f"Mode: {'GEMINI (live LLM)' if bot.use_gemini else 'DEMO (offline, no API key set)'}")

    for q in QUESTIONS:
        print("\n" + "-" * 70)
        print(f"Q: {q}")
        result = bot.ask(q)
        print(f"A: {result.answer}")

if __name__ == "__main__":
    main()