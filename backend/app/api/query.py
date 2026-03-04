# app/api/query.py
import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.agents.rag_agent import run_agent
from app.core.llm import get_groq_client

router = APIRouter()

class QueryRequest(BaseModel):
    question: str
    history: list[dict] = []

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[dict]
    hallucination_status: str
    rewritten_query: str | None
    trace: list[str]

@router.post("/", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    print("\n" + "="*60)
    print(f"[QUERY] Received: '{req.question}'")

    # Warm up / load LLM (cached after first call)
    print("[LLM]   Waking up API (getting client)...")
    t0 = time.time()
    get_groq_client()          # triggers API key check
    print(f"[LLM]   Client ready in {time.time()-t0:.1f}s")

    print("[AGENT] Running agentic pipeline (retrieve → evaluate → generate → hallucination check)...")
    t1 = time.time()
    result = run_agent(req.question, req.history)
    print(f"[AGENT] Done in {time.time()-t1:.1f}s  |  status: {result['hallucination_status']}")
    print("="*60 + "\n")

    return result
