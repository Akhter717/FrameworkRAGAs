# RAG Testing Framework

A reusable framework to evaluate any RAG (Retrieval-Augmented Generation) system
on **retrieval quality** and **generation quality**, against a defined test dataset.

## Folder contents

```
rag-testing-framework/
├── README.md                    (this file)
├── requirements.txt
├── rag_eval_framework.py        (core, reusable - import this)
├── sample_rag_system.py         (example RAG system - swap in your own later)
├── run_evaluation.py            (full run-through, paste into a Jupyter notebook)
├── test_dataset_basic.json      (6 straightforward Q&A test cases)
├── test_dataset_challenging.json (6 tricky/edge-case test cases)
└── sample_documents/
    ├── leave_and_wfh_policy.txt
    ├── expense_policy.pdf
    └── it_and_conduct_policy.pdf
```

## Setup (fresh machine / fresh VM)

1. Create and activate a virtual environment (Python 3.11 recommended):
   ```
   python -m venv myenv
   myenv\Scripts\activate.bat        (Windows)
   source myenv/bin/activate         (Mac/Linux)
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Open this folder in VS Code, create a new Jupyter Notebook (or open
   `run_evaluation.py` directly with the Jupyter extension - it recognizes
   the `# %%` cell markers), and select your venv as the kernel.

4. Run the cells top to bottom (see `run_evaluation.py`). You'll be prompted
   for your Groq API key (get one free at https://console.groq.com).

## Using your own real RAG system

Two things to change, nothing else:

1. **Documents**: point `file_paths` (in `run_evaluation.py`, step 4) at your
   real documents instead of the sample ones.
2. **Pipeline function**: `my_rag_pipeline()` (step 5) currently calls the
   sample FAISS+Groq system. Replace its body with calls to your own
   retriever + generator, keeping the same `RAGPipelineOutput` return shape:
   ```python
   def my_rag_pipeline(question: str) -> RAGPipelineOutput:
       docs = my_retriever.retrieve(question)
       answer = my_generator.generate(question, docs)
       return RAGPipelineOutput(
           retrieved_context_ids=[d.id for d in docs],
           retrieved_contexts=[d.text for d in docs],
           generated_answer=answer,
       )
   ```
3. Build a matching test dataset (same JSON shape as `test_dataset_basic.json`)
   using questions/answers from your real documents.

Everything else (the metrics, the evaluation loop, the report) is reused as-is.

## Metrics covered

**Retrieval**
| Metric | How it's computed |
|---|---|
| Precision | rule-based, from retrieved vs. ground-truth doc IDs |
| Recall | rule-based, from retrieved vs. ground-truth doc IDs |
| Hit Rate | rule-based - did at least one relevant doc get retrieved |
| Context Precision | LLM-judged via `ragas` |
| Context Recall | LLM-judged via `ragas` |

**Generation**
| Metric | How it's computed |
|---|---|
| Correctness, Relevance, Completeness, Coherence, Conciseness, Citation Accuracy | custom LLM-rubric judge (all 6 scored in ONE combined LLM call, to save tokens) |
| Faithfulness | via `ragas` |
| Hallucination Rate | derived as `1 - Faithfulness` |

## Compatibility fixes already applied in this version

These issues came up while building this on Windows and are already fixed
in the code here - listed for reference in case you hit them again on a
different machine:

1. **`langchain.chains` ModuleNotFoundError** - `create_retrieval_chain` /
   `create_stuff_documents_chain` moved to `langchain_classic.chains` in
   newer langchain releases. `sample_rag_system.py` tries both import paths.
2. **`langchain.text_splitter` ModuleNotFoundError** - `RecursiveCharacterTextSplitter`
   moved to the standalone `langchain_text_splitters` package in newer
   releases. Also handled with a try/except.
3. **`OSError: ... c10.dll` (torch DLL crash on Windows)** - triggered by
   `HuggingFaceEmbeddings`, which pulls in `sentence-transformers` ->
   `transformers` -> `torch`. Fixed by (a) stubbing out `sentence_transformers`
   before it's imported by `langchain_text_splitters`, and (b) using a
   custom TF-IDF-based embeddings class (`SimpleTfidfEmbeddings`, pure
   scikit-learn) instead of HuggingFace/torch embeddings entirely.
4. **`model_decommissioned` error from Groq** - `llama3-70b-8192` was retired.
   Default model changed to `openai/gpt-oss-120b`. Check
   https://console.groq.com/docs/models if this one is retired too.
5. **`RateLimitError` (429) from Groq free tier**  - each test case used to
   trigger ~9 separate LLM calls. Reduced to ~4 by combining the 6 custom
   rubric metrics into a single LLM call that returns one JSON object with
   all 6 scores. If you still hit rate limits: wait a few minutes, test with
   a smaller slice of the dataset (`cases[:2]`), or upgrade your Groq tier.
6. **`retrieval.precision` stuck at a low fixed value** - happens when
   `k` (chunks retrieved per query) is larger than your total number of
   chunks, so every query retrieves everything. `build_rag_chain` now takes
   a `k` parameter - keep it close to your actual chunk count for small
   test sets.

## Notes for future extension

- Run cases in parallel (`asyncio.gather`) once your dataset grows beyond a
  handful of cases - right now it evaluates one case at a time for simplicity.
- The custom rubric judge does a regex/JSON parse of the LLM's response. If a
  model ever returns malformed JSON, that metric will show up as `None` for
  that row rather than crashing the whole run.
- Consider adding retry-with-backoff around LLM calls for more resilience
  against transient rate limits.
