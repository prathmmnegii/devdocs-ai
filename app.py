
import hashlib
import os
import tempfile

import streamlit as st

from rag_pipeline import (
    build_rag_chain,
    index_uploaded_pdfs,
    ask_question,
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="DevDocs AI",
    page_icon="📚",
    layout="wide",
)

st.title("📚 DevDocs AI")
st.caption(
    "Ask questions about your PDF documents using a local RAG pipeline."
)


# --------------------------------------------------
# SESSION STATE INITIALIZATION
# --------------------------------------------------

DEFAULTS = {
    "pdf_signature": None,
    "vectorstore": None,
    "messages": [],
    "pdf_names": [],
    "index_error": None,
    "uploader_key": 0,
    "index_info": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value.copy() if isinstance(value, list) else value


# --------------------------------------------------
# CLEAR SESSION CALLBACK
# --------------------------------------------------

def clear_current_session():
    st.session_state.messages = []
    st.session_state.pdf_signature = None
    st.session_state.vectorstore = None
    st.session_state.pdf_names = []
    st.session_state.index_error = None
    st.session_state.index_info = None

    # Use a new widget key to reset the file uploader.
    # Do not assign to the old uploader's session-state value.
    st.session_state.uploader_key += 1


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def get_models():
    return build_rag_chain()


with st.spinner("Loading local embedding and language models..."):
    embeddings, llm, prompt = get_models()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.header("Document Settings")

    st.button(
        "🗑️ Clear Session",
        on_click=clear_current_session,
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("### How to use")
    st.markdown(
        """
        1. Upload one or more PDFs.
        2. Wait for indexing to finish.
        3. Ask questions about their contents.
        4. Check the retrieved source passages.
        """
    )

    st.markdown("---")
    st.caption("Local-first RAG • Python • LangChain • FAISS • Ollama")


# --------------------------------------------------
# PDF UPLOADER
# --------------------------------------------------

uploaded_pdfs = st.file_uploader(
    "Upload your PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
    key=f"pdf_uploader_{st.session_state.uploader_key}",
    help="Upload PDFs containing selectable text.",
)


# --------------------------------------------------
# INDEX UPLOADED PDFS
# --------------------------------------------------

if uploaded_pdfs:
    # Create a signature from file names and contents.
    # This prevents unnecessary re-indexing on each rerun.
    signature_hash = hashlib.sha256()

    for pdf in uploaded_pdfs:
        signature_hash.update(pdf.name.encode("utf-8"))
        signature_hash.update(pdf.getvalue())

    current_signature = signature_hash.hexdigest()

    if current_signature != st.session_state.pdf_signature:
        temp_paths = []

        try:
            st.session_state.index_error = None

            with st.spinner(
                "Reading PDFs and building the searchable index..."
            ):
                pdf_inputs = []

                for pdf in uploaded_pdfs:
                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf",
                        delete=False,
                    ) as temp_file:
                        temp_file.write(pdf.getvalue())
                        temp_path = temp_file.name

                    temp_paths.append(temp_path)
                    pdf_inputs.append((temp_path, pdf.name))

                vectorstore, page_count, chunk_count = index_uploaded_pdfs(
                    pdf_inputs,
                    embeddings,
                )

                if chunk_count == 0:
                    raise ValueError(
                        "No readable text was found in the uploaded PDFs. "
                        "Check whether they contain selectable text."
                    )

                st.session_state.vectorstore = vectorstore
                st.session_state.pdf_signature = current_signature
                st.session_state.pdf_names = [
                    pdf.name for pdf in uploaded_pdfs
                ]
                st.session_state.index_info = {
                    "pages": page_count,
                    "chunks": chunk_count,
                }

                # A new document collection starts a fresh conversation.
                st.session_state.messages = []

        except Exception as exc:
            st.session_state.vectorstore = None
            st.session_state.pdf_signature = None
            st.session_state.pdf_names = []
            st.session_state.index_info = None
            st.session_state.index_error = str(exc)

        finally:
            for path in temp_paths:
                try:
                    os.remove(path)
                except OSError:
                    pass


# --------------------------------------------------
# INDEX STATUS
# --------------------------------------------------

if st.session_state.index_error:
    st.error(f"Could not index the PDFs: {st.session_state.index_error}")

elif st.session_state.vectorstore is not None:
    st.success(
        f"Indexed {len(st.session_state.pdf_names)} PDF(s) successfully."
    )

    if st.session_state.index_info:
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Pages processed",
                st.session_state.index_info["pages"],
            )

        with col2:
            st.metric(
                "Text chunks created",
                st.session_state.index_info["chunks"],
            )

    with st.expander("View uploaded documents"):
        for name in st.session_state.pdf_names:
            st.write(f"📄 {name}")


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("View retrieved sources"):
                for i, source in enumerate(message["sources"], start=1):
                    st.markdown(
                        f"**Source {i}: {source['source']} "
                        f"(Page {source['page']})**"
                    )
                    st.caption(
                        f"Reranker score: {source.get('score', 0):.4f}"
                    )
                    st.write(source["content"])
                    st.markdown("---")


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about your uploaded PDFs...",
    disabled=st.session_state.vectorstore is None,
)

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating an answer..."):
            try:
                result = ask_question(
                    question,
                    st.session_state.vectorstore,
                    llm,
                    prompt,
                )

                answer = result["answer"]
                sources = result.get("sources", [])

                st.markdown(answer)

                if sources:
                    with st.expander("View retrieved sources"):
                        for i, source in enumerate(sources, start=1):
                            st.markdown(
                                f"**Source {i}: {source['source']} "
                                f"(Page {source['page']})**"
                            )
                            st.caption(
                                "Reranker score: "
                                f"{source.get('score', 0):.4f}"
                            )
                            st.write(source["content"])
                            st.markdown("---")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception as exc:
                error_message = (
                    f"An error occurred while answering your question: {exc}"
                )
                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    }
                )
