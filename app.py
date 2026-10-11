
import tempfile
from pathlib import Path

import streamlit as st

from rag_pipeline import (
    build_rag_chain,
    index_uploaded_pdf,
    ask_question,
)


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="DevDocs AI",
    page_icon="📚",
    layout="wide",
)

st.title("📚 DevDocs AI")
st.caption(
    "A local PDF question-answering assistant powered by "
    "LangChain, FAISS, Hugging Face embeddings, and Ollama."
)


# --------------------------------------------------
# Initialize models
# --------------------------------------------------

@st.cache_resource
def get_models():
    return build_rag_chain()


try:
    embeddings, llm, prompt = get_models()
except Exception as error:
    st.error(
        "Could not initialize the local AI models. "
        "Check that Ollama is running and the llama3.2 model "
        "is installed."
    )
    st.exception(error)
    st.stop()


# --------------------------------------------------
# Initialize session state
# --------------------------------------------------

if "pdf_signature" not in st.session_state:
    st.session_state.pdf_signature = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.header("Document settings")

    if st.button(
        "🗑️ Clear current session",
        use_container_width=True,
    ):
        st.session_state.vectorstore = None
        st.session_state.pdf_signature = None
        st.session_state.messages = []
        st.session_state.pdf_name = None
        st.rerun()

    st.divider()

    st.markdown("### How to use")
    st.markdown(
        """
        1. Upload a PDF.
        2. Wait for indexing to finish.
        3. Ask questions about the document.
        4. Expand the source section to inspect the evidence.
        """
    )

    st.divider()

    st.caption("LLM: Llama 3.2 via Ollama")
    st.caption("Embeddings: all-MiniLM-L6-v2")
    st.caption("Vector store: FAISS")


# --------------------------------------------------
# PDF uploader
# --------------------------------------------------

uploaded_pdf = st.file_uploader(
    "Upload your documentation PDF",
    type=["pdf"],
    help="Upload a PDF containing selectable text.",
)


# --------------------------------------------------
# Process uploaded PDF
# --------------------------------------------------

if uploaded_pdf is None:
    st.info("Upload a PDF to start asking questions.")

else:
    pdf_bytes = uploaded_pdf.getvalue()

    # Detect document changes and avoid rebuilding the index
    # on every Streamlit rerun.
    signature = (
        uploaded_pdf.name,
        len(pdf_bytes),
        hash(pdf_bytes),
    )

    if signature != st.session_state.pdf_signature:
        st.session_state.vectorstore = None
        st.session_state.pdf_signature = None
        st.session_state.pdf_name = None
        st.session_state.messages = []

        try:
            with st.spinner(
                "Extracting PDF text and building the search index..."
            ):
                with tempfile.NamedTemporaryFile(
                    suffix=".pdf",
                    delete=False,
                ) as temp_file:
                    temp_file.write(pdf_bytes)
                    temp_path = Path(temp_file.name)

                try:
                    (
                        vectorstore,
                        page_count,
                        chunk_count,
                    ) = index_uploaded_pdf(
                        temp_path,
                        embeddings,
                    )
                finally:
                    temp_path.unlink(missing_ok=True)

                st.session_state.vectorstore = vectorstore
                st.session_state.pdf_signature = signature
                st.session_state.pdf_name = uploaded_pdf.name

            st.success(
                f"PDF indexed successfully: {page_count} pages "
                f"with text and {chunk_count} searchable chunks."
            )

        except Exception as error:
            st.session_state.vectorstore = None
            st.session_state.pdf_signature = None
            st.session_state.pdf_name = None

            st.error(f"Could not process this PDF: {error}")

    # --------------------------------------------------
    # Chat interface
    # --------------------------------------------------

    if st.session_state.vectorstore is not None:
        st.markdown(
            f"**Current document:** `{st.session_state.pdf_name}`"
        )

        st.divider()

        # Display conversation history.
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                if message.get("sources"):
                    with st.expander("📄 View PDF sources"):
                        for source in message["sources"]:
                            page = source.get("page")

                            if page is not None:
                                st.markdown(f"**Page {page}**")
                            else:
                                st.markdown("**Page unavailable**")

                            st.text(source["content"])

        # Accept a new question.
        question = st.chat_input(
            "Ask a question about your PDF..."
        )

        if question:
            st.session_state.messages.append({
                "role": "user",
                "content": question,
            })

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner(
                    "Searching the PDF and generating an answer..."
                ):
                    try:
                        result = ask_question(
                            question,
                            st.session_state.vectorstore,
                            llm,
                            prompt,
                        )

                        answer = result["answer"]
                        sources = result["sources"]

                        st.markdown(answer)

                        if sources:
                            with st.expander("📄 View PDF sources"):
                                for source in sources:
                                    page = source.get("page")

                                    if page is not None:
                                        st.markdown(f"**Page {page}**")
                                    else:
                                        st.markdown(
                                            "**Page unavailable**"
                                        )

                                    st.text(source["content"])

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        })

                    except Exception as error:
                        error_message = (
                            "An error occurred while generating the answer."
                        )
                        st.error(error_message)
                        st.exception(error)

