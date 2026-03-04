"""
LLM wrapper connecting to Groq's OpenAI-compatible endpoint.
Includes exponential backoff retry on rate-limit (429) and server errors (5xx).
"""
import os
import asyncio
import logging
from functools import lru_cache

from openai import AsyncOpenAI, RateLimitError, APIStatusError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger("rag.llm")

GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "llama-3.1-8b-instant")
MAX_TOKENS    = int(os.getenv("MAX_TOKENS", "512"))


@lru_cache(maxsize=1)
def get_async_groq_client() -> AsyncOpenAI:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
    logger.info(f"Groq client initialised | model={GROQ_MODEL}")
    return AsyncOpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )


def _is_retryable(exc: BaseException) -> bool:
    """Retry on rate-limit (429) or server errors (5xx)."""
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code >= 500
    return False


@retry(
    retry=retry_if_exception_type((RateLimitError, APIStatusError)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def _call_groq(client: AsyncOpenAI, messages: list[dict], model: str, stream: bool, token_queue=None):
    """Inner call — retried by tenacity. Returns (text, streamed_bool)."""
    if stream and token_queue:
        response = await client.chat.completions.create(
            model=model, messages=messages, max_tokens=MAX_TOKENS,
            temperature=0.0, stream=True,
        )
        full_text = ""
        async for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                full_text += content
                await token_queue.put({"type": "token", "content": content})
        return full_text.strip()
    else:
        response = await client.chat.completions.create(
            model=model, messages=messages, max_tokens=MAX_TOKENS,
            temperature=0.0, stream=False,
        )
        return response.choices[0].message.content.strip()


async def agenerate(
    prompt: str,
    history: list[dict] = None,
    token_queue: asyncio.Queue = None,
) -> str:
    """Send a prompt to Groq, with automatic retry + exponential backoff."""
    client = get_async_groq_client()
    messages = list(history) if history else []
    messages.append({"role": "user", "content": prompt})

    try:
        return await _call_groq(client, messages, GROQ_MODEL, bool(token_queue), token_queue)
    except RateLimitError:
        # Fallback to secondary model on exhausted retries
        if FALLBACK_MODEL != GROQ_MODEL:
            logger.warning(f"Primary model rate-limited → falling back to {FALLBACK_MODEL}")
            return await _call_groq(client, messages, FALLBACK_MODEL, False)
        raise


# ── Prompt builders ────────────────────────────────────────────────────────────

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
