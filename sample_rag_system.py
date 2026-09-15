"""
sample_rag_system.py
A Jupyter-friendly RAG system (retriever + generator) used to demonstrate the
eval framework. Originally adapted from a Streamlit app; all UI code (st.*)
has been removed, and several Windows/version compatibility fixes are baked
in below (see comments). Replace build_vector_store's file loading and/or
build_rag_chain's model with your own real RAG system when ready.
"""

import sys
import types

# ---------------------------------------------------------------------------
# Compatibility patch: langchain_text_splitters unconditionally tries to
# import sentence_transformers (for a class we never use), which drags in
# `transformers` -> `torch`. On some Windows setups torch's DLL fails to load
# and crashes this entire import chain even though we never call
# sentence_transformers ourselves. Stubbing it out here prevents that crash.
# We use TF-IDF embeddings below instead of HuggingFace/torch embeddings, so
# torch is never actually needed for this file to work.
# ---------------------------------------------------------------------------
if "sentence_transformers" not in sys.modules:
    _fake_st_module = types.ModuleType("sentence_transformers")
    _fake_st_module.SentenceTransformer = type("SentenceTransformer", (), {})
    sys.modules["sentence_transformers"] = _fake_st_module

from langchain_community.document_loaders import PyPDFLoader, TextLoader

# RecursiveCharacterTextSplitter moved from `langchain.text_splitter` to the
# standalone `langchain_text_splitters` package in newer langchain releases.
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer

# create_retrieval_chain / create_stuff_documents_chain moved from
# `langchain.chains` to `langchain_classic.chains` in newer langchain
# releases. Try both so this file keeps working regardless of version.
try:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


class SimpleTfidfEmbeddings(Embeddings):
    """
    Lightweight embeddings using TF-IDF (pure scikit-learn/numpy - no torch,
    no ONNX, no model downloads). Avoids the Windows torch DLL crash entirely.
    Good enough for keyword-based retrieval over a handful of documents.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self._fitted = False

    def embed_documents(self, texts):
        vectors = self.vectorizer.fit_transform(texts)
        self._fitted = True
        return vectors.toarray().tolist()

    def embed_query(self, text):
        if not self._fitted:
            raise RuntimeError("Vectorizer not fitted yet - build the vector store first.")
        vector = self.vectorizer.transform([text])
        return vector.toarray().tolist()[0]


def build_vector_store(file_paths):
    """
    file_paths: list of local file paths, mixing .pdf and .txt freely,
    e.g. ["docs/policy.pdf", "docs/notes.txt"]
    """
    docs = []
    for path in file_paths:
        ext = path.split(".")[-1].lower()
        if ext == "pdf":
            loader = PyPDFLoader(path)
        elif ext == "txt":
            loader = TextLoader(path)
        else:
            print(f"Unsupported file type: {path}, skipping.")
            continue
        docs.extend(loader.load())

    if not docs:
        raise ValueError("No valid documents loaded.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    embeddings = SimpleTfidfEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore


def build_rag_chain(vectorstore, groq_api_key: str, model: str = "openai/gpt-oss-120b", k: int = 3):
    """
    model default is a currently-supported Groq model (llama3-70b-8192 was
    decommissioned by Groq - check https://console.groq.com/docs/models for
    the latest list if this one is retired in the future).
    k defaults to 3 - keep this close to your actual number of chunks for
    small demo datasets, or Precision will look artificially low.
    """
    llm = ChatGroq(groq_api_key=groq_api_key, model=model)

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(
        vectorstore.as_retriever(search_kwargs={"k": k}), question_answer_chain
    )
    return rag_chain
