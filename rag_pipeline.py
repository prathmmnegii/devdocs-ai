
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


# Project configuration
BASE_DIR = Path(__file__).resolve().parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"


def load_vectorstore():
    """Load the existing FAISS index."""

    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            "FAISS index not found. Run python ingest.py first."
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

    print("FAISS vectorstore loaded.")
    return vectorstore


def build_rag_chain():
    """Create the retrieval and answer-generation pipeline."""

    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are DevDocs AI, a technical documentation assistant.

Answer the question using ONLY the context provided below.
If the context does not contain enough information, clearly
say that you could not find the answer in the documentation.
Do not invent facts.

Include a concise explanation and a practical example when
the context supports one.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    return retriever, llm, prompt


def ask_question(question, retriever, llm, prompt):
    """Retrieve context and generate a grounded answer."""

    documents = retriever.invoke(question)

    if not documents:
        return {
            "answer": (
                "I couldn't find relevant information in the "
                "available documentation."
            ),
            "sources": [],
        }

    context = "\n\n".join(
        f"Source: {Path(doc.metadata.get('source', 'Unknown')).name}\n"
        f"{doc.page_content}"
        for doc in documents
    )

    messages = prompt.format_messages(
        context=context,
        question=question,
    )

    response = llm.invoke(messages)

    sources = []
    seen = set()

    for doc in documents:
        source = Path(
            doc.metadata.get("source", "Unknown")
        ).name

        if source not in seen:
            seen.add(source)
            sources.append({
                "name": source,
                "content": doc.page_content,
            })

    return {
        "answer": response.content,
        "sources": sources,
    }


if __name__ == "__main__":
    retriever, llm, prompt = build_rag_chain()

    print("\nDevDocs AI is ready. Type 'exit' to quit.")

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            result = ask_question(
                question, retriever, llm, prompt
            )

            print("\nAnswer:\n")
            print(result["answer"])

            print("\nSources:")
            for source in result["sources"]:
                print(f"- {source['name']}")

        except Exception as error:
            print(f"\nAn error occurred: {error}")
