# DevDocs AI — RAG-Based Technical Documentation Assistant

**DevDocs AI** is a local-first Retrieval-Augmented Generation (RAG) project designed to help developers find answers from technical documentation using semantic search and locally hosted language models.

Built with Python, LangChain, Hugging Face embeddings, FAISS, Ollama, and Streamlit, the project aims to provide context-grounded answers with source references without depending on paid LLM APIs.

> **Project status:** In development

## Features

### Implemented
- **Document Ingestion:** Load technical documentation from Markdown files.
- **Text Chunking:** Split documents into smaller chunks using LangChain's `RecursiveCharacterTextSplitter`.
- **Local Embeddings:** Generate vector embeddings using the `all-MiniLM-L6-v2` sentence-transformer model.
- **Vector Storage:** Store document embeddings in a FAISS vector index.
- **Semantic Search:** Retrieve relevant documentation chunks based on a user's question.
- **Persistent Indexing:** Save and reload the FAISS index locally.

### Planned
- **LLM Integration:** Generate answers using a locally hosted Llama model through Ollama.
- **Source Citations:** Display source documents alongside generated answers.
- **Hallucination Mitigation:** Instruct the model to acknowledge when the retrieved context is insufficient.
- **Interactive Chat Interface:** Build a simple Streamlit-based interface.
- **Retrieval Evaluation:** Test retrieval quality using a curated set of questions.
- **Automated Testing:** Add tests for document ingestion, retrieval, and error handling.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| LangChain | RAG pipeline components |
| Hugging Face Sentence Transformers | Local text embeddings |
| FAISS | Vector storage and similarity search |
| Ollama | Local LLM inference |
| Streamlit | Web interface (planned) |
| Git & GitHub | Version control |

## How RAG Works

1. **Document Loading:** Read technical documentation from Markdown files.
2. **Text Splitting:** Divide documents into smaller, overlapping chunks.
3. **Embedding Generation:** Convert each chunk into a numerical vector.
4. **Vector Indexing:** Store the vectors and their corresponding text in FAISS.
5. **Semantic Retrieval:** Find chunks relevant to the user's question.
6. **Answer Generation:** Pass the retrieved context to a local LLM to generate a grounded response. *(Planned)*

## Project Structure

```text
devdocs-ai/
├── data/
│   └── fastapi_docs/
│       ├── first_steps.md
│       ├── path_params.md
│       ├── request_body.md
│       └── dependies.md
├── vectorstore/
├── ingest.py
├── retrieve.py
├── requirements.txt
├── README.md
└── .gitignore
```

The `vectorstore/` directory contains the locally generated FAISS index. Its files are created by the ingestion script.

## Getting Started

### Prerequisites

- Python 3.11 or 3.12
- pip
- An internet connection for the initial package and embedding-model downloads

Ollama will be required when local LLM integration is implemented.

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/devdocs-ai.git
cd devdocs-ai
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

Install the packages used by the current implementation:

```bash
pip install langchain-community langchain-text-splitters langchain-huggingface sentence-transformers faiss-cpu
```

### 4. Prepare the documentation

Place Markdown files inside:

```text
data/fastapi_docs/
```

Add your own technical documentation or use the sample documents included in the repository.

### 5. Build the vector index

```bash
python ingest.py
```

This loads the documents, splits them into chunks, generates embeddings, and saves the FAISS index in `vectorstore/`.

### 6. Search the documentation

```bash
python retrieve.py
```

Enter a question about the indexed documentation to retrieve relevant chunks. Type `exit` to quit.

## Example Queries

- How does FastAPI validate path parameters?
- How do I validate JSON request bodies?
- How does dependency injection work in FastAPI?

The current implementation retrieves relevant documentation chunks. LLM-generated answers and the interactive chat interface are planned features.

## Cost and Privacy

- No paid LLM API is required for the current ingestion and retrieval pipeline.
- Embeddings are generated locally after the model has been downloaded.
- FAISS stores the vector index on your machine.
- Documentation and queries can remain local.
- The planned Ollama integration will support local answer generation without paid inference APIs.

Initial downloads require internet access. Hardware resources and disk space are needed to run local models.

## Future Improvements

- Add support for PDF and other document formats.
- Allow users to upload their own documentation.
- Improve retrieval quality through chunk-size and top-k experiments.
- Add source-grounded answers and unsupported-question handling.
- Build a Streamlit interface with conversation history.
- Evaluate retrieval performance on a curated test set.
- Explore deployment options for a public demo.

## Learning Objectives

This project explores practical concepts in:

- Retrieval-Augmented Generation (RAG)
- Semantic search and vector embeddings
- Text chunking and document preprocessing
- Vector databases and similarity search
- Local LLM inference
- LLM grounding and retrieval evaluation

## License

This project is intended for educational and portfolio use. A license can be added to clarify reuse and distribution permissions.

---

**DevDocs AI** — Making technical documentation easier to search and understand.
