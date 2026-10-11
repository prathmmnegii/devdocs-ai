
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


# Initial threshold based on your observed retrieval scores.
# Lower squared L2 distance generally means a closer match.
MAX_DISTANCE = 1.55
RETRIEVAL_K = 5


def build_rag_chain():
    """Initialize local embeddings, the LLM, and the prompt."""

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    llm = ChatOllama(
        model="llama3.2",
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_template(
        """
        You are DevDocs AI, an assistant that answers questions
        using an uploaded PDF.

        Follow these rules:
        1. Use only the supplied PDF passages.
        2. If the passages do not contain enough evidence,
           say that the PDF does not provide enough information.
        3. Never invent facts or use outside knowledge.
        4. Give a concise, direct answer.
        5. Mention PDF page numbers when useful.

        PDF PASSAGES:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """
    )

    return embeddings, llm, prompt


def index_uploaded_pdf(pdf_path, embeddings):
    """Extract PDF text, create chunks, and build a FAISS index."""

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    documents = [
        doc for doc in documents
        if doc.page_content.strip()
    ]

    if not documents:
        raise ValueError(
            "No selectable text was found in this PDF."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=70,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError(
            "No searchable text chunks could be created."
        )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings,
    )

    return vectorstore, len(documents), len(chunks)


def ask_question(question, vectorstore, llm, prompt):
    """Retrieve relevant passages and answer using PDF evidence."""

    retrieved_docs = vectorstore.similarity_search_with_score(
        question,
        k=RETRIEVAL_K,
    )

    if not retrieved_docs:
        return {
            "answer": (
                "I couldn't retrieve any passages from the PDF."
            ),
            "sources": [],
        }

    print("\n--- RETRIEVAL DEBUG ---")
    print("Question:", question)

    for rank, (document, score) in enumerate(
        retrieved_docs,
        start=1,
    ):
        print(
            f"Result {rank} | "
            f"Distance: {float(score):.4f} | "
            f"Text: {document.page_content[:180]}"
        )

    print("--- END RETRIEVAL DEBUG ---\n")

    # Keep only passages within the initial distance threshold.
    relevant_docs = [
        (document, float(score))
        for document, score in retrieved_docs
        if float(score) <= MAX_DISTANCE
    ]

    if not relevant_docs:
        return {
            "answer": (
                "I couldn't find sufficiently relevant information "
                "in the uploaded PDF to answer that question. "
                "Please try a question related to the document."
            ),
            "sources": [],
        }

    context_parts = []
    sources = []

    for document, score in relevant_docs:
        page = document.metadata.get("page")
        page_number = page + 1 if page is not None else None

        page_label = (
            f"Page {page_number}"
            if page_number is not None
            else "Unknown page"
        )

        context_parts.append(
            f"[{page_label}]\n{document.page_content}"
        )

        sources.append({
            "page": page_number,
            "content": document.page_content,
            "score": score,
        })

    context = "\n\n".join(context_parts)

    response = llm.invoke(
        prompt.format_messages(
            context=context,
            question=question,
        )
    )

    return {
        "answer": response.content,
        "sources": sources,
    }
