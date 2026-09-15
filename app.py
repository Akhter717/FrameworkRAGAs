"""
app.py
Streamlit front-end for the RAG Testing Framework.

This does NOT change the core logic in rag_eval_framework.py or
sample_rag_system.py - both are imported and used exactly as they are.
This file only adds a web UI (file upload, run button, dashboard,
CSV/Excel download) around them, and fixes the two things that only
work inside a Jupyter kernel:
  1. getpass()        -> st.text_input(type="password")
  2. top-level await   -> asyncio.run(...)
"""

import asyncio
import io
import os
import tempfile

import pandas as pd
import streamlit as st

from rag_eval_framework import RAGTestFramework, load_dataset, run_pipeline
from sample_rag_system import build_vector_store, build_rag_chain

st.set_page_config(page_title="RAG Testing Framework", layout="wide")
st.title("RAG Testing Framework")
st.caption("Streamlit UI around rag_eval_framework.py + sample_rag_system.py")

# ---------------------------------------------------------------------------
# Sidebar - config
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")

    # Prefer a key stored in Streamlit secrets (.streamlit/secrets.toml
    # locally, or the "Secrets" panel on Streamlit Cloud) so nobody has to
    # paste the key by hand every run. Falls back to a manual box if no
    # secret is set.
    try:
        secret_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        secret_key = ""

    if secret_key:
        groq_api_key = secret_key
        st.success("Groq API key loaded from secrets.")
    else:
        groq_api_key = st.text_input(
            "Groq API key",
            type="password",
            help=(
                "Get one free at https://console.groq.com. "
                "Tip: add GROQ_API_KEY to .streamlit/secrets.toml (local) "
                "or the app's Secrets panel (Streamlit Cloud) to skip this box."
            ),
        )
    model_name = st.text_input("Groq model", value="openai/gpt-oss-120b")
    k_chunks = st.number_input(
        "k (chunks retrieved per query)", min_value=1, max_value=20, value=3
    )

    st.divider()
    st.subheader("Pass / fail thresholds")
    st.caption("Applies to every 0-1 metric except hallucination rate (lower is better there).")
    pass_threshold = st.slider("Minimum score to pass", 0.0, 1.0, 0.7, 0.05)
    hallucination_threshold = st.slider(
        "Maximum hallucination rate to pass", 0.0, 1.0, 0.3, 0.05
    )

# ---------------------------------------------------------------------------
# Step 1: documents
# ---------------------------------------------------------------------------
st.subheader("1. Documents")

use_sample_docs = st.checkbox(
    "Use the bundled sample documents (sample_documents/)", value=True
)

uploaded_docs = None
if not use_sample_docs:
    uploaded_docs = st.file_uploader(
        "Upload .pdf / .txt documents", type=["pdf", "txt"], accept_multiple_files=True
    )

# ---------------------------------------------------------------------------
# Step 2: test dataset
# ---------------------------------------------------------------------------
st.subheader("2. Test dataset")

dataset_choice = st.radio(
    "Dataset source",
    ["Bundled: test_dataset_basic.json", "Bundled: test_dataset_challenging.json", "Upload my own JSON"],
    horizontal=False,
)

uploaded_dataset = None
if dataset_choice == "Upload my own JSON":
    uploaded_dataset = st.file_uploader("Upload test dataset (JSON)", type=["json"])

# ---------------------------------------------------------------------------
# Step 3: run
# ---------------------------------------------------------------------------
st.subheader("3. Run evaluation")

run_clicked = st.button("Run evaluation", type="primary")

if run_clicked:
    if not groq_api_key:
        st.error("Enter your Groq API key in the sidebar first.")
        st.stop()

    with st.spinner("Preparing documents..."):
        if use_sample_docs:
            file_paths = [
                os.path.join("sample_documents", f)
                for f in os.listdir("sample_documents")
            ]
        else:
            if not uploaded_docs:
                st.error("Upload at least one document, or tick 'use bundled sample documents'.")
                st.stop()
            tmp_dir = tempfile.mkdtemp()
            file_paths = []
            for f in uploaded_docs:
                path = os.path.join(tmp_dir, f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                file_paths.append(path)

        vectorstore = build_vector_store(file_paths)
        rag_chain = build_rag_chain(
            vectorstore, groq_api_key=groq_api_key, model=model_name, k=int(k_chunks)
        )

    with st.spinner("Loading test dataset..."):
        if dataset_choice == "Bundled: test_dataset_basic.json":
            cases = load_dataset("test_dataset_basic.json")
        elif dataset_choice == "Bundled: test_dataset_challenging.json":
            cases = load_dataset("test_dataset_challenging.json")
        else:
            if not uploaded_dataset:
                st.error("Upload a test dataset JSON, or pick a bundled one.")
                st.stop()
            tmp_dir = tempfile.mkdtemp()
            path = os.path.join(tmp_dir, uploaded_dataset.name)
            with open(path, "wb") as out:
                out.write(uploaded_dataset.getbuffer())
            cases = load_dataset(path)

    from rag_eval_framework import RAGPipelineOutput

    def my_rag_pipeline(question: str) -> RAGPipelineOutput:
        response = rag_chain.invoke({"input": question})
        retrieved_docs = response["context"]

        def make_id(d):
            source = d.metadata.get("source", "unknown")
            if "page" in d.metadata:
                return f"{source}_p{d.metadata['page']}"
            return source

        return RAGPipelineOutput(
            retrieved_context_ids=[make_id(d) for d in retrieved_docs],
            retrieved_contexts=[d.page_content for d in retrieved_docs],
            generated_answer=response["answer"],
        )

    with st.spinner(f"Running the RAG pipeline on {len(cases)} test case(s)..."):
        run_pipeline(cases, my_rag_pipeline)

    with st.spinner("Scoring with RAGAS + LLM judge (this can take a while)..."):
        from langchain_groq import ChatGroq
        from ragas.llms import LangchainLLMWrapper

        llm = ChatGroq(api_key=groq_api_key, model=model_name, temperature=0.5, max_tokens=1024)
        ragas_llm = LangchainLLMWrapper(llm)
        framework = RAGTestFramework(judge_llm=llm, ragas_evaluator_llm=ragas_llm)

        report_df = asyncio.run(framework.evaluate_dataset(cases))

    st.session_state["report_df"] = report_df
    st.success("Evaluation complete.")

# ---------------------------------------------------------------------------
# Step 4: dashboard
# ---------------------------------------------------------------------------
if "report_df" in st.session_state:
    report_df = st.session_state["report_df"]

    st.subheader("4. Results")

    summary = RAGTestFramework.summarize(report_df)
    metric_cols = st.columns(4)
    for i, (metric, value) in enumerate(summary.items()):
        with metric_cols[i % 4]:
            if pd.isna(value):
                st.metric(metric, "N/A")
                continue
            if metric.endswith("hallucination_rate"):
                passed = value <= hallucination_threshold
            else:
                passed = value >= pass_threshold
            st.metric(metric, f"{value:.2f}", delta="PASS" if passed else "FAIL")

    st.divider()
    st.dataframe(report_df, use_container_width=True)

    # downloads
    csv_bytes = report_df.to_csv(index=False).encode("utf-8")
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        report_df.to_excel(writer, index=False, sheet_name="results")
    excel_bytes = excel_buffer.getvalue()

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "Download CSV", data=csv_bytes, file_name="rag_eval_report.csv", mime="text/csv"
        )
    with dl_col2:
        st.download_button(
            "Download Excel",
            data=excel_bytes,
            file_name="rag_eval_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
