
import streamlit as st

from rag_pipeline import build_rag_chain, ask_question


st.set_page_config(
    page_title="DevDocs AI",
    page_icon="📚",
    layout="centered",
)

st.title("📚 DevDocs AI")
st.caption("Ask questions about technical documentation")
st.markdown(
    "A local-first RAG assistant powered by LangChain, "
    "FAISS, Hugging Face embeddings, and Ollama."
)


@st.cache_resource
def initialize_rag():
    """Initialize the RAG components once per app session."""
    return build_rag_chain()


try:
    retriever, llm, prompt = initialize_rag()
except Exception as error:
    st.error(
        "Could not initialize DevDocs AI. "
        "Check that Ollama is running and the FAISS index exists."
    )
    st.exception(error)
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []


# Display existing conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            for source in message.get("sources", []):
                with st.expander(f"Source: {source['name']}"):
                    st.write(source["content"])


# Accept a new question
question = st.chat_input("Ask a question about FastAPI...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documentation and generating answer..."):
            try:
                result = ask_question(
                    question,
                    retriever,
                    llm,
                    prompt,
                )

                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                if sources:
                    st.markdown("**Sources**")

                    for source in sources:
                        with st.expander(
                            f"📄 {source['name']}"
                        ):
                            st.write(source["content"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as error:
                st.error(f"Something went wrong: {error}")
