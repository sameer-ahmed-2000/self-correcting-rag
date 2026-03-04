import time
import json
import asyncio
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.agents.rag_agent import run_agent_async
from app.core.llm import get_async_groq_client

router = APIRouter()

class QueryRequest(BaseModel):
    question: str
    history: list[dict] = []

@router.post("/")
async def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    print("\n" + "="*60)
    print(f"[QUERY] Received: '{req.question}'")

    print("[LLM]   Waking up API (getting client)...")
    get_async_groq_client()          # triggers API key check

    print("[AGENT] Running agentic pipeline in streaming mode...")
    
    async def event_generator():
        queue = asyncio.Queue()
        task = asyncio.create_task(run_agent_async(req.question, req.history, queue))
        
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=60.0)
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'type': 'error', 'content': 'Server timeout during generation'})}\n\n"
                break
                
            if event["type"] == "end":
                # Ensure we also yield the final result structure so frontend can finalize its state
                yield f"data: {json.dumps({'type': 'end', 'result': event['result']})}\n\n"
                break
            else:
                yield f"data: {json.dumps(event)}\n\n"
                
    return StreamingResponse(event_generator(), media_type="text/event-stream")
