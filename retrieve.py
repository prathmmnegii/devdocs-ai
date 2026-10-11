
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Project paths
BASE_DIR = Path(__file__).resolve().parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_vectorstore():
    """Load the saved FAISS index."""

    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            "FAISS index not found. Run ingest.py first."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    print("FAISS index loaded successfully.")
    return vectorstore


def search_documents(vectorstore, question, k=3):
    """Retrieve the most relevant document chunks."""

    results = vectorstore.similarity_search_with_score(
        question,
        k=k,
    )

    print(f"\nQuestion: {question}")
    print(f"Retrieved {len(results)} chunks.\n")

    for i, (doc, score) in enumerate(results, start=1):
        source = Path(doc.metadata["source"]).name

        print(f"--- Result {i} ---")
        print(f"Source: {source}")
        print(f"Distance: {score:.4f}")
        print(f"Content:\n{doc.page_content}\n")

    return results


if __name__ == "__main__":
    db = load_vectorstore()

    while True:
        question = input(
            "\nAsk a FastAPI question (or type 'exit'): "
        ).strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        search_documents(db, question)
