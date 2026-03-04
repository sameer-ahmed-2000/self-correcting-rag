import time
import json
import asyncio
import os
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator
from app.agents.rag_agent import run_agent_async
from app.core.llm import get_async_groq_client
from app.core.database import (
    create_session, add_message, touch_session,
    update_session_title, get_session
)

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    history: list[dict] = []
    session_id: int | None = None  # omit to create a new session automatically

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty.")
        if len(v) > 1000:
            raise ValueError("Question must be 1000 characters or fewer.")
        return v

    @field_validator("history")
    @classmethod
    def history_max_depth(cls, v: list) -> list:
        # Keep only the last 20 turns (40 messages) to avoid context bloat
        return v[-40:] if len(v) > 40 else v


@router.post("/")
async def query(req: QueryRequest, request: Request):
    print("\n" + "="*60)
    print(f"[QUERY] Received: '{req.question}'")
    get_async_groq_client()  # warm-up / validate API key

    # Resolve or create session
    session_id = req.session_id
    is_new_session = False
    if session_id is None:
        session = create_session(req.question[:60])  # first question as title
        session_id = session.id
        is_new_session = True
    else:
        if not get_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found")

    # Persist user message
    add_message(session_id, "user", req.question)

    print(f"[AGENT] Session {session_id} | streaming pipeline...")

    async def event_generator():
        # First event tells frontend the session_id (important for new sessions)
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id, 'is_new': is_new_session})}\n\n"

        queue = asyncio.Queue()
        task = asyncio.create_task(
            run_agent_async(req.question, req.history, queue)
        )

        full_answer = ""
        final_result = None

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=90.0)
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'type': 'error', 'content': 'Request timed out'})}\n\n"
                task.cancel()
                break

            if event["type"] == "token":
                full_answer += event["content"]

            if event["type"] == "end":
                final_result = event["result"]
                # Persist assistant message with full metadata
                add_message(
                    session_id,
                    "assistant",
                    final_result["answer"],
                    metadata={
                        "sources": final_result.get("sources", []),
                        "hallucination_status": final_result.get("hallucination_status"),
                        "rewritten_query": final_result.get("rewritten_query"),
                        "trace": final_result.get("trace", []),
                    }
                )
                touch_session(session_id)
                yield f"data: {json.dumps({'type': 'end', 'result': final_result})}\n\n"
                break

            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
