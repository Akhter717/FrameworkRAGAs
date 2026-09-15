# %% [markdown]
# # RAG Testing Framework - Full Run
# Copy each `# %%` block below into its own Jupyter cell (VS Code does this
# automatically if you open this .py file with the Jupyter extension).
# Run cells top to bottom. Every known compatibility issue is already fixed
# in this version - see README.md for what was fixed and why.

# %% [markdown]
# ## 1. Groq API key

# %%
import getpass
import os

os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")
api_key = os.environ["GROQ_API_KEY"]
print("Key loaded:", bool(api_key))

# %% [markdown]
# ## 2. Set up the Groq LLM + ragas wrapper

# %%
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper

llm = ChatGroq(
    api_key=api_key,
    model="openai/gpt-oss-120b",   # check https://console.groq.com/docs/models if this is retired later
    temperature=0.5,
    max_tokens=1024,
)

ragas_llm = LangchainLLMWrapper(llm)

# %% [markdown]
# ## 3. Build the eval framework

# %%
from rag_eval_framework import RAGTestFramework

framework = RAGTestFramework(judge_llm=llm, ragas_evaluator_llm=ragas_llm)

# %% [markdown]
# ## 4. Build the sample RAG system (retriever + generator) from your documents
# Swap the paths in `file_paths` for your own real documents whenever ready.
# k=3 matches the number of chunks in this small demo set - raise it once you
# have a larger, real document collection.

# %%
from sample_rag_system import build_vector_store, build_rag_chain

file_paths = [
    "sample_documents/leave_and_wfh_policy.txt",
    "sample_documents/expense_policy.pdf",
    "sample_documents/it_and_conduct_policy.pdf",
]

vectorstore = build_vector_store(file_paths)
rag_chain = build_rag_chain(vectorstore, groq_api_key=api_key, k=3)

# %% [markdown]
# ## 5. Wrap the RAG system in the format the eval framework expects

# %%
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

# %% [markdown]
# ## 6. Quick manual sanity check (optional but recommended)

# %%
test_output = my_rag_pipeline("How many casual leaves per year?")
print("Answer:", test_output.generated_answer)
print("Retrieved from:", test_output.retrieved_context_ids)

# %% [markdown]
# ## 7. Load the test dataset and run the pipeline on all cases

# %%
from rag_eval_framework import load_dataset, run_pipeline

cases = load_dataset("test_dataset_basic.json")   # or "test_dataset_challenging.json"
run_pipeline(cases, my_rag_pipeline)

for c in cases:
    print(c.id, "-> retrieved from:", c.retrieved_context_ids)

# %% [markdown]
# ## 8. Run the full evaluation
# NOTE: this makes several LLM calls per test case. If you hit a Groq rate
# limit error, wait a few minutes and/or test with `cases[:2]` first.

# %%
report_df = await framework.evaluate_dataset(cases)
report_df

# %% [markdown]
# ## 9. Summary + save report

# %%
summary = framework.summarize(report_df)
print(summary)

report_df.to_csv("rag_eval_report.csv", index=False)
print("Saved report to rag_eval_report.csv")
