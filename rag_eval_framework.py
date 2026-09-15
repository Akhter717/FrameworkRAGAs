"""
rag_eval_framework.py
======================
A reusable framework to evaluate RAG (Retrieval-Augmented Generation) systems
on Retrieval quality and Generation quality.

See README.md in this project for full setup + usage instructions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# 1. Data model
# ---------------------------------------------------------------------------

@dataclass
class RAGTestCase:
    """One row of the test dataset."""
    id: str
    question: str
    ground_truth_answer: str
    ground_truth_context_ids: List[str] = field(default_factory=list)
    ground_truth_contexts: List[str] = field(default_factory=list)

    # filled in later by run_pipeline()
    retrieved_context_ids: List[str] = field(default_factory=list)
    retrieved_contexts: List[str] = field(default_factory=list)
    generated_answer: Optional[str] = None


def load_dataset(path: str) -> List[RAGTestCase]:
    """Load a JSON test dataset. See test_dataset_basic.json for the format."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [
        RAGTestCase(
            id=row["id"],
            question=row["question"],
            ground_truth_answer=row["ground_truth_answer"],
            ground_truth_context_ids=row.get("ground_truth_context_ids", []),
            ground_truth_contexts=row.get("ground_truth_contexts", []),
        )
        for row in raw
    ]


# ---------------------------------------------------------------------------
# 2. Plugging in YOUR RAG pipeline
# ---------------------------------------------------------------------------

@dataclass
class RAGPipelineOutput:
    retrieved_context_ids: List[str]
    retrieved_contexts: List[str]
    generated_answer: str


RAGPipelineFn = Callable[[str], RAGPipelineOutput]


def run_pipeline(cases: List[RAGTestCase], pipeline_fn: RAGPipelineFn) -> None:
    """Runs your real RAG system on every test case, in place."""
    for case in cases:
        out = pipeline_fn(case.question)
        case.retrieved_context_ids = out.retrieved_context_ids
        case.retrieved_contexts = out.retrieved_contexts
        case.generated_answer = out.generated_answer


# ---------------------------------------------------------------------------
# 3. Retrieval metrics
# ---------------------------------------------------------------------------

def _set_based_metrics(retrieved_ids: List[str], relevant_ids: List[str]) -> Dict[str, float]:
    """Classic IR metrics computed directly from doc IDs. No LLM needed."""
    retrieved_set, relevant_set = set(retrieved_ids), set(relevant_ids)
    true_positives = len(retrieved_set & relevant_set)
    precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
    recall = true_positives / len(relevant_set) if relevant_set else 0.0
    hit_rate = 1.0 if true_positives > 0 else 0.0
    return {"precision": precision, "recall": recall, "hit_rate": hit_rate}


class RetrievalEvaluator:
    """
    precision / recall / hit_rate       -> rule-based, computed from doc IDs
    context_precision / context_recall  -> LLM-judged via ragas (needs evaluator_llm)
    """

    def __init__(self, evaluator_llm=None):
        self.evaluator_llm = evaluator_llm

    async def evaluate_case(self, case: RAGTestCase) -> Dict[str, Any]:
        results: Dict[str, Any] = _set_based_metrics(
            case.retrieved_context_ids, case.ground_truth_context_ids
        )

        if self.evaluator_llm is not None and case.retrieved_contexts:
            cp, cr = await self._ragas_context_metrics(case)
            results["context_precision"] = cp
            results["context_recall"] = cr
        else:
            results["context_precision"] = None
            results["context_recall"] = None
        return results

    async def _ragas_context_metrics(self, case: RAGTestCase):
        from ragas import SingleTurnSample
        from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall

        sample = SingleTurnSample(
            user_input=case.question,
            response=case.generated_answer or "",
            reference=case.ground_truth_answer,
            retrieved_contexts=case.retrieved_contexts,
        )
        precision_metric = LLMContextPrecisionWithReference(llm=self.evaluator_llm)
        recall_metric = LLMContextRecall(llm=self.evaluator_llm)

        precision_score = await precision_metric.single_turn_ascore(sample)
        recall_score = await recall_metric.single_turn_ascore(sample)
        return precision_score, recall_score


# ---------------------------------------------------------------------------
# 4. Generation metrics
# ---------------------------------------------------------------------------

# All 6 rubric metrics below are judged in ONE combined LLM call (see
# _judge_all_rubrics) instead of 6 separate calls, to minimize token usage
# on rate-limited API plans.
_GENERATION_RUBRICS: Dict[str, str] = {
    "correctness": (
        "Does the ANSWER contain factually correct information when compared "
        "against the REFERENCE ANSWER? Score 0.0 (completely wrong) to 1.0 (fully correct)."
    ),
    "relevance": (
        "Does the ANSWER directly address the QUESTION asked, without going off-topic? "
        "Score 0.0 (irrelevant) to 1.0 (fully relevant)."
    ),
    "completeness": (
        "Does the ANSWER cover all the key points present in the REFERENCE ANSWER? "
        "Score 0.0 (missing everything important) to 1.0 (fully complete)."
    ),
    "coherence": (
        "Is the ANSWER logically structured, clear, and easy to follow? "
        "Score 0.0 (incoherent/confusing) to 1.0 (perfectly coherent)."
    ),
    "conciseness": (
        "Is the ANSWER free of unnecessary repetition or filler, while still being complete? "
        "Score 0.0 (very bloated/redundant) to 1.0 (perfectly concise)."
    ),
    "citation_accuracy": (
        "If the ANSWER cites or references the CONTEXT, are those citations accurate "
        "(i.e. the cited context actually says what the answer claims)? "
        "If the answer makes no citations, score 1.0 only if it doesn't need any. "
        "Score 0.0 (citations wrong/fabricated) to 1.0 (fully accurate)."
    ),
}


class GenerationEvaluator:
    """
    faithfulness / hallucination_rate                          -> via ragas
    correctness, relevance, completeness, coherence,
    conciseness, citation_accuracy                              -> custom LLM-rubric judge
                                                                    (1 combined call, not 6)
    """

    def __init__(self, judge_llm, ragas_evaluator_llm=None):
        """
        judge_llm: any LangChain-style chat model with .invoke(prompt) -> message
        ragas_evaluator_llm: ragas-wrapped LLM for Faithfulness (LangchainLLMWrapper).
                   Defaults to wrapping judge_llm if not given.
        """
        self.judge_llm = judge_llm
        self._ragas_evaluator_llm = ragas_evaluator_llm

    async def evaluate_case(self, case: RAGTestCase) -> Dict[str, Any]:
        context_block = (
            "\n---\n".join(case.retrieved_contexts)
            if case.retrieved_contexts
            else "(no context retrieved)"
        )

        results: Dict[str, Any] = self._judge_all_rubrics(case, context_block)

        faithfulness, hallucination_rate = await self._faithfulness_and_hallucination(case)
        results["faithfulness"] = faithfulness
        results["hallucination_rate"] = hallucination_rate
        return results

    def _build_combined_prompt(self, case: RAGTestCase, context_block: str) -> str:
        rubric_lines = "\n".join(f"- {name}: {instr}" for name, instr in _GENERATION_RUBRICS.items())
        metric_keys = ", ".join(_GENERATION_RUBRICS.keys())
        return f"""You are a strict evaluator for a RAG (Retrieval-Augmented Generation) system.

QUESTION:
{case.question}

CONTEXT (retrieved by the system):
{context_block}

REFERENCE ANSWER (ground truth):
{case.ground_truth_answer}

ANSWER (produced by the system, to be evaluated):
{case.generated_answer}

Score the ANSWER on EACH of the following metrics, each from 0.0 to 1.0:
{rubric_lines}

Respond with ONLY a single-line JSON object, nothing else, no markdown fences,
with exactly these keys: {metric_keys}
Example format: {{"correctness": 0.8, "relevance": 1.0, "completeness": 0.7, "coherence": 1.0, "conciseness": 0.9, "citation_accuracy": 0.5}}
"""

    def _judge_all_rubrics(self, case: RAGTestCase, context_block: str) -> Dict[str, Optional[float]]:
        prompt = self._build_combined_prompt(case, context_block)
        raw = self.judge_llm.invoke(prompt)
        text = raw.content if hasattr(raw, "content") else str(raw)

        match = re.search(r"\{.*\}", text, re.DOTALL)
        scores: Dict[str, Optional[float]] = {name: None for name in _GENERATION_RUBRICS}
        if match:
            try:
                parsed = json.loads(match.group(0))
                for name in _GENERATION_RUBRICS:
                    value = parsed.get(name)
                    scores[name] = float(value) if value is not None else None
            except (json.JSONDecodeError, TypeError, ValueError):
                pass  # leave as None if parsing fails - keeps the pipeline running
        return scores

    async def _faithfulness_and_hallucination(self, case: RAGTestCase):
        if not case.retrieved_contexts or not case.generated_answer:
            return None, None
        try:
            from ragas import SingleTurnSample
            from ragas.llms import LangchainLLMWrapper
            from ragas.metrics import Faithfulness

            evaluator_llm = self._ragas_evaluator_llm or LangchainLLMWrapper(self.judge_llm)
            sample = SingleTurnSample(
                user_input=case.question,
                response=case.generated_answer,
                retrieved_contexts=case.retrieved_contexts,
            )
            metric = Faithfulness(llm=evaluator_llm)
            score = await metric.single_turn_ascore(sample)
            hallucination_rate = None if score is None else round(1 - score, 4)
            return score, hallucination_rate
        except Exception:
            return None, None


# ---------------------------------------------------------------------------
# 5. Orchestrator
# ---------------------------------------------------------------------------

class RAGTestFramework:
    """The single entry point: feed it test cases (already run through your
    RAG pipeline), get back a full metrics report as a DataFrame."""

    def __init__(self, judge_llm, ragas_evaluator_llm=None):
        self.retrieval_evaluator = RetrievalEvaluator(evaluator_llm=ragas_evaluator_llm)
        self.generation_evaluator = GenerationEvaluator(
            judge_llm=judge_llm, ragas_evaluator_llm=ragas_evaluator_llm
        )

    async def evaluate_dataset(self, cases: List[RAGTestCase]) -> pd.DataFrame:
        rows = []
        for case in cases:
            retrieval_scores = await self.retrieval_evaluator.evaluate_case(case)
            generation_scores = await self.generation_evaluator.evaluate_case(case)
            rows.append({
                "id": case.id,
                "question": case.question,
                "generated_answer": case.generated_answer,
                "ground_truth_answer": case.ground_truth_answer,
                "retrieved_from": ", ".join(case.retrieved_context_ids) if case.retrieved_context_ids else "(nothing retrieved)",
                **{f"retrieval.{k}": v for k, v in retrieval_scores.items()},
                **{f"generation.{k}": v for k, v in generation_scores.items()},
            })
        return pd.DataFrame(rows)

    @staticmethod
    def summarize(report_df: pd.DataFrame) -> pd.Series:
        """Average every numeric metric column across the whole dataset."""
        numeric_cols = report_df.select_dtypes(include="number").columns
        return report_df[numeric_cols].mean(numeric_only=True)
