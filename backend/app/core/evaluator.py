"""
RAG evaluation using RAGAS metrics with Groq as the backing LLM.
"""
import os
from typing import List

from ragas import evaluate as ragas_evaluate
from ragas import EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def get_ragas_llm() -> LangchainLLMWrapper:
    """Return a RAGAS-compatible LLM wrapper backed by Groq."""
    llm = ChatOpenAI(
        model=GROQ_MODEL,
        openai_api_key=GROQ_API_KEY,
        openai_api_base="https://api.groq.com/openai/v1",
        temperature=0,
    )
    return LangchainLLMWrapper(llm)


def get_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    """Return a RAGAS-compatible embeddings wrapper (local sentence-transformers)."""
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return LangchainEmbeddingsWrapper(hf_embeddings)


def evaluate_rag(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str] | None = None,
) -> dict:
    """
    Evaluate a batch of RAG outputs.

    Args:
        questions:     List of user questions
        answers:       List of generated answers
        contexts:      List of context chunks per question (list of lists)
        ground_truths: Optional reference answers (needed for context_recall)

    Returns:
        dict of metric_name → score
    """
    ragas_llm = get_ragas_llm()
    ragas_emb = get_ragas_embeddings()

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
        ContextPrecision(llm=ragas_llm),
    ]
    if ground_truths:
        metrics.append(ContextRecall(llm=ragas_llm))

    samples = []
    for i, (q, a, ctx) in enumerate(zip(questions, answers, contexts)):
        sample = SingleTurnSample(
            user_input=q,
            response=a,
            retrieved_contexts=ctx,
            reference=ground_truths[i] if ground_truths else None,
        )
        samples.append(sample)

    eval_dataset = EvaluationDataset(samples=samples)
    result = ragas_evaluate(eval_dataset, metrics=metrics)

    return result.to_pandas().mean(numeric_only=True).to_dict()


def evaluate_single(question: str, answer: str, contexts: List[str]) -> dict:
    """Convenience wrapper for evaluating a single Q&A pair."""
    return evaluate_rag(
        questions=[question],
        answers=[answer],
        contexts=[contexts],
    )
