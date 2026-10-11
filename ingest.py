
from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Project paths
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "data" / "fastapi_docs"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# Free, local embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents():
    """Load Markdown documentation."""

    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"Folder not found: {DOCS_DIR}")

    loader = DirectoryLoader(
        str(DOCS_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )

    documents = loader.load()

    if not documents:
        raise ValueError("No Markdown documents found.")

    print(f"Loaded {len(documents)} documents.")

    for doc in documents:
        print(
            f"{Path(doc.metadata['source']).name}: "
            f"{len(doc.page_content)} characters"
        )

    return documents


def split_documents(documents):
    """Split documents into overlapping chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No chunks created. Check your documents.")

    print(f"Created {len(chunks)} chunks.")

    return chunks


def create_vectorstore(chunks):
    """Embed chunks and save the FAISS index locally."""

    print("\nLoading the embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Generating embeddings and building FAISS index...")

    vectorstore = FAISS.from_documents(chunks, embeddings)

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))

    print(f"FAISS index saved to: {VECTORSTORE_DIR.resolve()}")

    return vectorstore


if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)
    vectorstore = create_vectorstore(chunks)

    print("\nIngestion completed successfully!")
    print(f"Documents: {len(documents)}")
    print(f"Chunks indexed: {vectorstore.index.ntotal}")
