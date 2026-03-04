"""
LLM wrapper connecting to Groq's OpenAI-compatible endpoint.
"""
import os
from functools import lru_cache

from openai import OpenAI

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_TOKENS   = int(os.getenv("MAX_TOKENS", "512"))


@lru_cache(maxsize=1)
def get_groq_client() -> OpenAI:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
    print(f"[LLM] Using Groq model: {GROQ_MODEL}")
    return OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )


def generate(prompt: str) -> str:
    """Send a prompt to Groq and return the assistant reply text."""
    client = get_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=0.0,   # deterministic — best for RAG grounding
        stream=False,
    )
    return response.choices[0].message.content.strip()


# ── Prompt builders (unchanged) ───────────────────────────────────────────────

def build_rag_prompt(question: str, context_chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source: {c['title']}]\n{c['text']}" for c in context_chunks
    )
    return f"""You are a helpful research assistant. Answer the question based ONLY on the provided context.
If the context does not contain enough information, say "I don't have enough information to answer this."
Always cite which source(s) you used.

Context:
{context}

Question: {question}

Answer:"""


def build_query_rewrite_prompt(original_query: str, previous_answer: str) -> str:
    return f"""The following question was asked but the answer was unsatisfactory or incomplete:

Original question: {original_query}
Unsatisfactory answer: {previous_answer}

Rewrite the question to be more specific and targeted to retrieve better information.
Output ONLY the rewritten question, nothing else.

Rewritten question:"""


def build_hallucination_check_prompt(question: str, answer: str, context: str) -> str:
    return f"""You are a fact-checking assistant. Evaluate if the answer is fully supported by the context.

Context: {context}

Question: {question}
Answer: {answer}

Is the answer fully supported by the context? Reply with ONLY one of: SUPPORTED, PARTIALLY_SUPPORTED, NOT_SUPPORTED

Result:"""
